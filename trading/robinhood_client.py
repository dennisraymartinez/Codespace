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


class RobinhoodAPIError(RuntimeError):
    """Raised when the API returns a non-success status."""

    def __init__(self, status_code: int, body: str, path: str) -> None:
        super().__init__(f"{status_code} on {path}: {body}")
        self.status_code = status_code
        self.body = body
        self.path = path


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
        response = self._session.request(
            method.upper(),
            BASE_URL + path,
            headers=self._headers(method, path, payload),
            data=payload if payload else None,
            timeout=DEFAULT_TIMEOUT,
        )
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

    def get_holdings(self, *symbols: str) -> dict[str, Any]:
        query = "".join(f"&asset_code={s}" for s in symbols)
        return self.get(f"/api/v1/crypto/trading/holdings/?{query.lstrip('&')}")

    def get_trading_pairs(self, *symbols: str) -> dict[str, Any]:
        query = "".join(f"&symbol={s}" for s in symbols)
        return self.get(f"/api/v1/crypto/trading/trading_pairs/?{query.lstrip('&')}")

    def get_best_bid_ask(self, *symbols: str) -> dict[str, Any]:
        query = "".join(f"&symbol={s}" for s in symbols)
        return self.get(
            f"/api/v1/crypto/marketdata/best_bid_ask/?{query.lstrip('&')}"
        )

    def get_orders(self) -> dict[str, Any]:
        return self.get("/api/v1/crypto/trading/orders/")
