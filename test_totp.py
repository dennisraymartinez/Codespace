"""Tests for totp.py, built on the test vectors published in the RFCs.

Run with:  python -m unittest test_totp   (or: python test_totp.py)
"""

import base64
import unittest
import urllib.parse

import totp as m


def b32(raw: bytes) -> str:
    return base64.b32encode(raw).decode("ascii")


# RFC 4226 appendix D / RFC 6238 appendix B seeds.
SEED_SHA1 = b"12345678901234567890"
SEED_SHA256 = b"12345678901234567890123456789012"
SEED_SHA512 = b"1234567890" * 6 + b"1234"


class TestHOTP(unittest.TestCase):
    # RFC 4226 appendix D, counters 0-9 with the 20-byte ASCII seed.
    VECTORS = [
        "755224", "287082", "359152", "969429", "338314",
        "254676", "287922", "162583", "399871", "520489",
    ]

    def test_rfc4226_vectors(self):
        for counter, expected in enumerate(self.VECTORS):
            with self.subTest(counter=counter):
                self.assertEqual(m.hotp(SEED_SHA1, counter), expected)

    def test_base32_secret_matches_raw_bytes(self):
        self.assertEqual(m.hotp(b32(SEED_SHA1), 0), m.hotp(SEED_SHA1, 0))

    def test_negative_counter_rejected(self):
        with self.assertRaises(ValueError):
            m.hotp(SEED_SHA1, -1)

    def test_digit_bounds_enforced(self):
        for bad in (5, 11):
            with self.subTest(digits=bad), self.assertRaises(ValueError):
                m.hotp(SEED_SHA1, 0, digits=bad)


class TestTOTP(unittest.TestCase):
    # RFC 6238 appendix B: timestamp -> (SHA1, SHA256, SHA512), 8 digits.
    VECTORS = [
        (59, "94287082", "46119246", "90693936"),
        (1111111109, "07081804", "68084774", "25091201"),
        (1111111111, "14050471", "67062674", "99943326"),
        (1234567890, "89005924", "91819424", "93441116"),
        (2000000000, "69279037", "90698825", "38618901"),
        (20000000000, "65353130", "77737706", "47863826"),
    ]

    def test_rfc6238_vectors(self):
        seeds = {
            "SHA1": SEED_SHA1,
            "SHA256": SEED_SHA256,
            "SHA512": SEED_SHA512,
        }
        for at, sha1, sha256, sha512 in self.VECTORS:
            for algorithm, expected in (
                ("SHA1", sha1),
                ("SHA256", sha256),
                ("SHA512", sha512),
            ):
                with self.subTest(at=at, algorithm=algorithm):
                    self.assertEqual(
                        m.totp(seeds[algorithm], at=at, digits=8, algorithm=algorithm),
                        expected,
                    )

    def test_code_is_stable_within_a_step_and_changes_across_it(self):
        secret = b32(SEED_SHA1)
        self.assertEqual(m.totp(secret, at=30), m.totp(secret, at=59.999))
        self.assertNotEqual(m.totp(secret, at=59.999), m.totp(secret, at=60))

    def test_unsupported_algorithm_rejected(self):
        with self.assertRaises(ValueError):
            m.totp(SEED_SHA1, at=0, algorithm="MD5")

    def test_nonpositive_period_rejected(self):
        with self.assertRaises(ValueError):
            m.totp(SEED_SHA1, at=0, period=0)


class TestVerify(unittest.TestCase):
    SECRET = b32(SEED_SHA1)

    def test_accepts_current_code(self):
        self.assertTrue(m.verify(self.SECRET, m.totp(self.SECRET, at=1111111111), at=1111111111))

    def test_accepts_drift_inside_the_window(self):
        code = m.totp(self.SECRET, at=1111111111)
        for skew in (-30, 30):
            with self.subTest(skew=skew):
                self.assertTrue(m.verify(self.SECRET, code, at=1111111111 + skew))

    def test_rejects_drift_outside_the_window(self):
        code = m.totp(self.SECRET, at=1111111111)
        for skew in (-60, 60):
            with self.subTest(skew=skew):
                self.assertFalse(m.verify(self.SECRET, code, at=1111111111 + skew))

    def test_window_zero_allows_no_drift(self):
        code = m.totp(self.SECRET, at=1111111111)
        self.assertTrue(m.verify(self.SECRET, code, at=1111111111, window=0))
        self.assertFalse(m.verify(self.SECRET, code, at=1111111111 + 30, window=0))

    def test_rejects_malformed_input_without_raising(self):
        for bad in ("", "abcdef", "12345", "1234567", "12 34 56"):
            with self.subTest(code=bad):
                self.assertFalse(m.verify(self.SECRET, bad, at=1111111111))

    def test_strips_spaces_from_a_pasted_code(self):
        code = m.totp(self.SECRET, at=1111111111)
        spaced = f"{code[:3]} {code[3:]}"
        self.assertTrue(m.verify(self.SECRET, spaced, at=1111111111))

    def test_negative_window_rejected(self):
        with self.assertRaises(ValueError):
            m.verify(self.SECRET, "000000", window=-1)


class TestVerifyHOTP(unittest.TestCase):
    def test_returns_the_matching_counter(self):
        self.assertEqual(m.verify_hotp(SEED_SHA1, "359152", 0), 2)

    def test_respects_the_look_ahead_limit(self):
        self.assertIsNone(m.verify_hotp(SEED_SHA1, "520489", 0, look_ahead=3))
        self.assertEqual(m.verify_hotp(SEED_SHA1, "520489", 0, look_ahead=9), 9)

    def test_rejects_a_replayed_code(self):
        self.assertEqual(m.verify_hotp(SEED_SHA1, "755224", 0), 0)
        self.assertIsNone(m.verify_hotp(SEED_SHA1, "755224", 1))


class TestSecretHandling(unittest.TestCase):
    def test_accepts_lowercase_spaced_and_unpadded_base32(self):
        canonical = b32(SEED_SHA1)
        messy = canonical.rstrip("=").lower()
        messy = " ".join(messy[i : i + 4] for i in range(0, len(messy), 4))
        self.assertEqual(m.normalize_secret(messy), SEED_SHA1)

    def test_rejects_empty_and_non_base32(self):
        for bad in ("", "   ", "not-base32-18!"):
            with self.subTest(secret=bad), self.assertRaises(ValueError):
                m.normalize_secret(bad)

    def test_random_secret_is_usable_and_unique(self):
        a, b = m.random_secret(), m.random_secret()
        self.assertNotEqual(a, b)
        self.assertNotIn("=", a)
        self.assertEqual(len(m.normalize_secret(a)), 20)
        self.assertEqual(len(m.totp(a)), 6)

    def test_random_secret_length_floor(self):
        with self.assertRaises(ValueError):
            m.random_secret(10)


class TestProvisioningURI(unittest.TestCase):
    def test_uri_carries_every_parameter(self):
        secret = b32(SEED_SHA1).rstrip("=")
        uri = m.provisioning_uri(secret, "dennis@example.com", issuer="Codespace")
        parsed = urllib.parse.urlparse(uri)
        params = urllib.parse.parse_qs(parsed.query)

        self.assertEqual(parsed.scheme, "otpauth")
        self.assertEqual(parsed.netloc, "totp")
        self.assertEqual(
            urllib.parse.unquote(parsed.path.lstrip("/")), "Codespace:dennis@example.com"
        )
        self.assertEqual(params["secret"], [secret])
        self.assertEqual(params["issuer"], ["Codespace"])
        self.assertEqual(params["algorithm"], ["SHA1"])
        self.assertEqual(params["digits"], ["6"])
        self.assertEqual(params["period"], ["30"])

    def test_label_omits_the_colon_without_an_issuer(self):
        uri = m.provisioning_uri(b32(SEED_SHA1), "dennis")
        self.assertIn("/dennis?", uri)
        self.assertNotIn("issuer=", uri)

    def test_empty_account_rejected(self):
        with self.assertRaises(ValueError):
            m.provisioning_uri(b32(SEED_SHA1), "")


class TestRemainingSeconds(unittest.TestCase):
    def test_counts_down_within_the_step(self):
        self.assertAlmostEqual(m.remaining_seconds(at=1111111110), 30.0)
        self.assertAlmostEqual(m.remaining_seconds(at=1111111111), 29.0)


class TestCLI(unittest.TestCase):
    def test_now_prints_a_code(self):
        self.assertEqual(m.main(["now", b32(SEED_SHA1), "--quiet"]), 0)

    def test_verify_exit_codes(self):
        secret = b32(SEED_SHA1)
        self.assertEqual(m.main(["verify", m.totp(secret), secret]), 0)
        self.assertEqual(m.main(["verify", "000000", secret, "--window", "0"]), 1)

    def test_bad_secret_exits_two(self):
        self.assertEqual(m.main(["now", "not base32 !"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
