"""Client for the official Robinhood Crypto Trading API.

Authentication is an Ed25519 signature over

    api_key + timestamp + path + method + body

sent as the x-api-key, x-timestamp and x-signature headers. `path` is the
request path including any query string; `body` is the empty string when
there is no payload.

Scope note: this covers CRYPTO only. Robinhood publishes no supported API
for stocks, ETFs or options -- see README.md.
"""

import base64
import json
import time
import uuid
from typing import Any, Dict, Optional

import requests
from nacl.signing import SigningKey

BASE_URL = "https://trading.robinhood.com"
API_PREFIX = "/api/v1/crypto"

TIMEOUT_SECONDS = 10


class RobinhoodError(RuntimeError):
    """An API request failed."""


class DryRunBlocked(RuntimeError):
    """An order was attempted while dry run was still enabled."""


class OrderTooLarge(ValueError):
    """An order exceeded the configured per-order notional cap."""


def load_signing_key(private_key_b64: str) -> SigningKey:
    """Build a SigningKey from a base64 private key.

    Robinhood hands back a 32-byte seed, but some exports concatenate the
    32-byte public key onto it. Accept either and keep only the seed.
    """
    raw = base64.b64decode(private_key_b64)
    if len(raw) == 64:
        raw = raw[:32]
    if len(raw) != 32:
        raise ValueError(
            f"Expected a 32- or 64-byte Ed25519 private key, got {len(raw)} bytes. "
            "Check that RH_PRIVATE_KEY was copied whole and is not the public key."
        )
    return SigningKey(raw)


def build_message(api_key: str, timestamp: int, path: str, method: str, body: str) -> str:
    """Assemble the exact string Robinhood expects to be signed.

    Isolated and pure so it can be unit-tested without credentials or network.
    """
    return f"{api_key}{timestamp}{path}{method.upper()}{body}"


class RobinhoodCrypto:
    def __init__(
        self,
        api_key: str,
        private_key_b64: str,
        dry_run: bool = True,
        max_order_usd: float = 10.0,
        base_url: str = BASE_URL,
    ):
        self.api_key = api_key
        self._signing_key = load_signing_key(private_key_b64)
        self.dry_run = dry_run
        self.max_order_usd = max_order_usd
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # --- signing -------------------------------------------------------

    def _headers(self, path: str, method: str, body: str) -> Dict[str, str]:
        timestamp = int(time.time())
        message = build_message(self.api_key, timestamp, path, method, body)
        signature = self._signing_key.sign(message.encode("utf-8")).signature
        return {
            "x-api-key": self.api_key,
            "x-timestamp": str(timestamp),
            "x-signature": base64.b64encode(signature).decode("utf-8"),
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, payload: Optional[Dict] = None) -> Any:
        # The signed body must be byte-identical to the body actually sent,
        # so serialize once and reuse the string for both.
        body = json.dumps(payload) if payload is not None else ""
        headers = self._headers(path, method, body)

        try:
            response = self._session.request(
                method=method.upper(),
                url=f"{self.base_url}{path}",
                headers=headers,
                data=body if body else None,
                timeout=TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise RobinhoodError(f"{method.upper()} {path} failed: {exc}") from exc

        if response.status_code == 401:
            raise RobinhoodError(
                f"401 Unauthorized on {method.upper()} {path}. Usual causes: the public key "
                "is not enrolled, the API key does not match the keypair, an IP allowlist "
                "is blocking you, or your clock has drifted."
            )
        if not response.ok:
            raise RobinhoodError(
                f"{response.status_code} on {method.upper()} {path}: {response.text[:500]}"
            )

        return response.json() if response.content else None

    # --- read-only -----------------------------------------------------

    def get_account(self) -> Any:
        return self._request("GET", f"{API_PREFIX}/trading/accounts/")

    def get_holdings(self) -> Any:
        return self._request("GET", f"{API_PREFIX}/trading/holdings/")

    def get_trading_pairs(self) -> Any:
        return self._request("GET", f"{API_PREFIX}/trading/trading_pairs/")

    def get_best_bid_ask(self, symbol: str) -> Any:
        return self._request("GET", f"{API_PREFIX}/marketdata/best_bid_ask/?symbol={symbol}")

    def get_orders(self) -> Any:
        return self._request("GET", f"{API_PREFIX}/trading/orders/")

    def mid_price(self, symbol: str) -> float:
        """Midpoint of the current spread, used to value an order before sending."""
        quote = self.get_best_bid_ask(symbol)
        results = quote.get("results") or []
        if not results:
            raise RobinhoodError(f"No quote returned for {symbol}.")
        top = results[0]
        return (float(top["bid_inclusive_of_sell_spread"]) + float(top["ask_inclusive_of_buy_spread"])) / 2

    # --- orders --------------------------------------------------------

    def place_market_order(self, symbol: str, side: str, quantity: float) -> Any:
        """Place a market order, subject to the dry-run and notional guards.

        Both guards are checked here rather than at the call site so that no
        code path can place an order without passing them.
        """
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if quantity <= 0:
            raise ValueError("quantity must be greater than zero")

        notional = quantity * self.mid_price(symbol)
        if notional > self.max_order_usd:
            raise OrderTooLarge(
                f"Order is ~${notional:.2f}, over the ${self.max_order_usd:.2f} cap. "
                "Raise RH_MAX_ORDER_USD deliberately if you mean it."
            )

        payload = {
            "client_order_id": str(uuid.uuid4()),
            "side": side,
            "symbol": symbol,
            "type": "market",
            "market_order_config": {"asset_quantity": str(quantity)},
        }

        if self.dry_run:
            raise DryRunBlocked(
                f"DRY RUN -- would have sent {side} {quantity} {symbol} (~${notional:.2f}). "
                "Set RH_DRY_RUN=false to trade for real."
            )

        return self._request("POST", f"{API_PREFIX}/trading/orders/", payload)

    def cancel_order(self, order_id: str) -> Any:
        return self._request("POST", f"{API_PREFIX}/trading/orders/{order_id}/cancel/")
