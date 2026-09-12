"""Tests for request signing and the order guards.

No network and no real credentials -- this is the part of the stack that is
both easy to get wrong and cheap to verify.
"""

import base64
import unittest

from nacl.signing import SigningKey

from rh_crypto import (
    DryRunBlocked,
    OrderTooLarge,
    RobinhoodCrypto,
    build_message,
    load_signing_key,
)


class TestBuildMessage(unittest.TestCase):
    def test_concatenation_order(self):
        self.assertEqual(
            build_message("key123", 1700000000, "/api/v1/crypto/trading/accounts/", "GET", ""),
            "key1231700000000/api/v1/crypto/trading/accounts/GET",
        )

    def test_body_is_appended_last(self):
        self.assertEqual(
            build_message("k", 1, "/p", "POST", '{"a":1}'),
            'k1/pPOST{"a":1}',
        )

    def test_method_is_uppercased(self):
        self.assertEqual(build_message("k", 1, "/p", "get", ""), "k1/pGET")


class TestLoadSigningKey(unittest.TestCase):
    def setUp(self):
        self.key = SigningKey.generate()
        self.seed = self.key.encode()

    def test_accepts_32_byte_seed(self):
        loaded = load_signing_key(base64.b64encode(self.seed).decode())
        self.assertEqual(loaded.encode(), self.seed)

    def test_accepts_64_byte_export(self):
        combined = self.seed + self.key.verify_key.encode()
        loaded = load_signing_key(base64.b64encode(combined).decode())
        self.assertEqual(loaded.encode(), self.seed)

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            load_signing_key(base64.b64encode(b"too short").decode())


class TestSignatureVerifies(unittest.TestCase):
    def test_headers_carry_a_valid_signature(self):
        key = SigningKey.generate()
        client = RobinhoodCrypto(
            api_key="test-key",
            private_key_b64=base64.b64encode(key.encode()).decode(),
        )

        headers = client._headers("/api/v1/crypto/trading/accounts/", "GET", "")
        message = build_message(
            "test-key", int(headers["x-timestamp"]), "/api/v1/crypto/trading/accounts/", "GET", ""
        )

        # Raises BadSignatureError if the signature does not match.
        key.verify_key.verify(message.encode(), base64.b64decode(headers["x-signature"]))
        self.assertEqual(headers["x-api-key"], "test-key")


class TestOrderGuards(unittest.TestCase):
    def _client(self, **kwargs):
        key = SigningKey.generate()
        return RobinhoodCrypto(
            api_key="test-key",
            private_key_b64=base64.b64encode(key.encode()).decode(),
            **kwargs,
        )

    def test_dry_run_blocks_orders(self):
        client = self._client(dry_run=True, max_order_usd=1_000_000)
        client.mid_price = lambda symbol: 50_000.0
        with self.assertRaises(DryRunBlocked):
            client.place_market_order("BTC-USD", "buy", 0.0001)

    def test_cap_blocks_before_dry_run_is_consulted(self):
        # A live client must still be stopped by the notional cap.
        client = self._client(dry_run=False, max_order_usd=10)
        client.mid_price = lambda symbol: 50_000.0
        with self.assertRaises(OrderTooLarge):
            client.place_market_order("BTC-USD", "buy", 1.0)

    def test_rejects_bad_side_and_quantity(self):
        client = self._client()
        with self.assertRaises(ValueError):
            client.place_market_order("BTC-USD", "hodl", 0.1)
        with self.assertRaises(ValueError):
            client.place_market_order("BTC-USD", "buy", 0)


if __name__ == "__main__":
    unittest.main()
