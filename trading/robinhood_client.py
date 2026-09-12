"""Minimal signed client for the Robinhood Crypto API.

Every request is signed with the Ed25519 private key you generated with
generate_keys.py. The signed message is:

    api_key + timestamp + path_with_query + method + body

where method is upper-case and body is "" for GET requests.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import requests
from nacl.signing import SigningKey

BASE_URL = "https://trading.robinhood.com"
DEFAULT_TIMEOUT = 10


class RobinhoodError(RuntimeError):
    """Base class for every failure this client raises."""


class RobinhoodAPIError(RobinhoodError):
    """The API answered with a non-success status."""

    def __init__(self, status_code: int, body: str, path: str) -> None:
        super().__init__(f"{status_code} on {path}: {body}")
        self.status_code = status_code
        self.body = body
        self.path = path


class RobinhoodConnectionError(RobinhoodError):
    """The request never got an answer — DNS, TLS, proxy, timeout.

    `outcome_known` is False when the request may have reached Robinhood
    even though we never saw the reply. For a POST that means an order
    might exist: reconcile with get_orders() before retrying, and retry
    with the SAME client_order_id so a duplicate cannot fill.
    """

    def __init__(self, path: str, cause: Exception, outcome_known: bool) -> None:
        detail = "never sent" if outcome_known else "OUTCOME UNKNOWN"
        super().__init__(f"could not reach {path} ({detail}): {cause}")
        self.path = path
        self.cause = cause
        self.outcome_known = outcome_known


class RobinhoodCryptoClient:
    def __init__(self, api_key: str, private_key_b64: str) -> None:
        if not api_key:
            raise ValueError("RH_API_KEY is empty")
        if not private_key_b64:
            raise ValueError("RH_PRIVATE_KEY is empty")

        seed = base64.b64decode(private_key_b64)
        if len(seed) == 64:
            # Some tools emit the 64-byte expanded key; the first 32 are the seed.
            seed = seed[:32]
        if len(seed) != 32:
            raise ValueError(
                f"RH_PRIVATE_KEY decodes to {len(seed)} bytes, expected 32. "
                "Re-run generate_keys.py and copy the whole value."
            )

        self.api_key = api_key
        self._signing_key = SigningKey(seed)
        self._session = requests.Session()

    # -- signing ------------------------------------------------------

    def _headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        timestamp = str(int(time.time()))
        message = f"{self.api_key}{timestamp}{path}{method.upper()}{body}"
        signature = self._signing_key.sign(message.encode()).signature
        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signature).decode(),
            "x-timestamp": timestamp,
            "Content-Type": "application/json; charset=utf-8",
        }

    # -- transport ----------------------------------------------------

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = json.dumps(body) if body is not None else ""
        try:
            response = self._session.request(
                method.upper(),
                BASE_URL + path,
                headers=self._headers(method, path, payload),
                data=payload if payload else None,
                timeout=DEFAULT_TIMEOUT,
            )
        except requests.RequestException as exc:
            # A GET that never landed changed nothing, so the outcome is
            # known. A write that never answered might still have been
            # applied — say so rather than guessing.
            raise RobinhoodConnectionError(
                path, exc, outcome_known=method.upper() == "GET"
            ) from exc
        if response.status_code >= 400:
            raise RobinhoodAPIError(response.status_code, response.text[:500], path)
        return response.json() if response.content else {}

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, body)

    # -- read-only endpoints -----------------------------------------

    def get_account(self) -> dict[str, Any]:
        return self.get("/api/v1/crypto/trading/accounts/")

    @staticmethod
    def with_query(base: str, key: str, values: tuple[str, ...]) -> str:
        """Append a query string, or nothing at all when there are no values.

        A bare trailing "?" is NOT harmless here. The signature covers the
        path, and Robinhood drops an empty query before verifying, so
        ".../holdings/?" signs a different message than the one checked and
        the request fails with "Signature is invalid".
        """
        if not values:
            return base
        return base + "?" + "&".join(f"{key}={v}" for v in values)

    def get_holdings(self, *symbols: str) -> dict[str, Any]:
        return self.get(
            self.with_query(
                "/api/v1/crypto/trading/holdings/", "asset_code", symbols
            )
        )

    def get_trading_pairs(self, *symbols: str) -> dict[str, Any]:
        return self.get(
            self.with_query(
                "/api/v1/crypto/trading/trading_pairs/", "symbol", symbols
            )
        )

    def get_best_bid_ask(self, *symbols: str) -> dict[str, Any]:
        return self.get(
            self.with_query(
                "/api/v1/crypto/marketdata/best_bid_ask/", "symbol", symbols
            )
        )

    def get_orders(self) -> dict[str, Any]:
        return self.get("/api/v1/crypto/trading/orders/")

    def get_order(self, order_id: str) -> dict[str, Any]:
        return self.get(f"/api/v1/crypto/trading/orders/{order_id}/")

    # -- write endpoints ---------------------------------------------
    #
    # These are raw transport: they apply NO safety rails. Application
    # code should go through trader.Trader, which is the single choke
    # point where the rails in safety.py are enforced.

    def place_order(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST an already-built, already-validated order body."""
        return self.post("/api/v1/crypto/trading/orders/", body)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        return self.post(f"/api/v1/crypto/trading/orders/{order_id}/cancel/", {})
