"""Time-based one-time passwords (TOTP) and HMAC-based one-time passwords (HOTP).

Pure standard library implementation of:
  * RFC 4226 - HOTP: An HMAC-Based One-Time Password Algorithm
  * RFC 6238 - TOTP: Time-Based One-Time Password Algorithm
  * Key URI format - otpauth:// provisioning URIs (Google Authenticator)

Typical use:

    >>> secret = random_secret()                  # base32, share with the app
    >>> uri = provisioning_uri(secret, "dennis@example.com", issuer="Codespace")
    >>> code = totp(secret)                       # current 6-digit code
    >>> verify(secret, code)                      # True, accepts +/- 1 step drift
    True

Run ``python totp.py --help`` for the command line interface.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import os
import secrets
import struct
import sys
import time
import urllib.parse

__all__ = [
    "hotp",
    "totp",
    "verify",
    "verify_hotp",
    "random_secret",
    "provisioning_uri",
    "normalize_secret",
    "remaining_seconds",
]

# Digest algorithms an authenticator app is expected to understand.
ALGORITHMS = {
    "SHA1": hashlib.sha1,
    "SHA256": hashlib.sha256,
    "SHA512": hashlib.sha512,
}

DEFAULT_DIGITS = 6
DEFAULT_PERIOD = 30
DEFAULT_ALGORITHM = "SHA1"


def normalize_secret(secret: str | bytes) -> bytes:
    """Return the raw key bytes for a shared secret.

    ``secret`` may be raw ``bytes`` or a base32 string. Base32 input is
    accepted the way users actually paste it: lowercase, with spaces or
    hyphens, and with the trailing ``=`` padding left off.
    """
    if isinstance(secret, (bytes, bytearray)):
        return bytes(secret)

    cleaned = secret.strip().replace(" ", "").replace("-", "").upper()
    if not cleaned:
        raise ValueError("secret is empty")
    # base64.b32decode insists on padding to a multiple of 8 characters.
    cleaned += "=" * (-len(cleaned) % 8)
    try:
        return base64.b32decode(cleaned, casefold=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"secret is not valid base32: {exc}") from exc


def _resolve_algorithm(algorithm: str):
    try:
        return ALGORITHMS[algorithm.upper()]
    except KeyError:
        raise ValueError(
            f"unsupported algorithm {algorithm!r}; choose one of "
            f"{', '.join(sorted(ALGORITHMS))}"
        ) from None


def hotp(
    secret: str | bytes,
    counter: int,
    *,
    digits: int = DEFAULT_DIGITS,
    algorithm: str = DEFAULT_ALGORITHM,
) -> str:
    """Return the RFC 4226 HOTP value for ``counter`` as a zero-padded string."""
    if counter < 0:
        raise ValueError("counter must not be negative")
    if not 6 <= digits <= 10:
        raise ValueError("digits must be between 6 and 10")

    key = normalize_secret(secret)
    digest = hmac.new(key, struct.pack(">Q", counter), _resolve_algorithm(algorithm)).digest()

    # Dynamic truncation (RFC 4226 section 5.3): the low nibble of the last
    # byte picks the 4-byte window; the top bit of that window is masked off
    # so the result is unsigned regardless of platform.
    offset = digest[-1] & 0x0F
    code = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(code % (10**digits)).zfill(digits)


def totp(
    secret: str | bytes,
    *,
    at: float | None = None,
    digits: int = DEFAULT_DIGITS,
    period: int = DEFAULT_PERIOD,
    algorithm: str = DEFAULT_ALGORITHM,
    t0: int = 0,
) -> str:
    """Return the RFC 6238 TOTP value for the time step containing ``at``.

    ``at`` is a Unix timestamp and defaults to now.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    now = time.time() if at is None else at
    counter = int((now - t0) // period)
    return hotp(secret, counter, digits=digits, algorithm=algorithm)


def verify(
    secret: str | bytes,
    code: str,
    *,
    at: float | None = None,
    digits: int = DEFAULT_DIGITS,
    period: int = DEFAULT_PERIOD,
    algorithm: str = DEFAULT_ALGORITHM,
    t0: int = 0,
    window: int = 1,
) -> bool:
    """Check ``code`` against the current time step, +/- ``window`` steps.

    A window of 1 (the default) tolerates roughly 30 seconds of clock skew in
    either direction, which is what most services allow. Comparison is
    constant-time so a caller cannot time its way to a valid code.
    """
    if window < 0:
        raise ValueError("window must not be negative")

    candidate = str(code).strip().replace(" ", "")
    if not candidate.isdigit() or len(candidate) != digits:
        return False

    now = time.time() if at is None else at
    matched = False
    for drift in range(-window, window + 1):
        expected = totp(
            secret,
            at=now + drift * period,
            digits=digits,
            period=period,
            algorithm=algorithm,
            t0=t0,
        )
        # No early exit: every candidate is compared so the runtime does not
        # leak which step matched.
        matched |= hmac.compare_digest(expected, candidate)
    return matched


def verify_hotp(
    secret: str | bytes,
    code: str,
    counter: int,
    *,
    digits: int = DEFAULT_DIGITS,
    algorithm: str = DEFAULT_ALGORITHM,
    look_ahead: int = 3,
) -> int | None:
    """Check an HOTP ``code`` from ``counter`` forward.

    Returns the counter value that matched, or ``None``. Persist
    ``matched + 1`` as the next expected counter so a code is never reusable.
    """
    if look_ahead < 0:
        raise ValueError("look_ahead must not be negative")

    candidate = str(code).strip().replace(" ", "")
    if not candidate.isdigit() or len(candidate) != digits:
        return None

    found = None
    for offset in range(look_ahead + 1):
        expected = hotp(secret, counter + offset, digits=digits, algorithm=algorithm)
        if hmac.compare_digest(expected, candidate) and found is None:
            found = counter + offset
    return found


def random_secret(length: int = 20) -> str:
    """Return a new base32 shared secret.

    The default of 20 bytes (160 bits) is the size RFC 4226 recommends and
    what authenticator apps expect for SHA1. Padding is stripped because
    most apps reject the ``=`` characters on paste.
    """
    if length < 16:
        raise ValueError("secret must be at least 16 bytes (128 bits)")
    return base64.b32encode(secrets.token_bytes(length)).decode("ascii").rstrip("=")


def remaining_seconds(*, at: float | None = None, period: int = DEFAULT_PERIOD) -> float:
    """Seconds left before the current code rolls over."""
    if period <= 0:
        raise ValueError("period must be positive")
    now = time.time() if at is None else at
    return period - (now % period)


def provisioning_uri(
    secret: str,
    account: str,
    *,
    issuer: str | None = None,
    digits: int = DEFAULT_DIGITS,
    period: int = DEFAULT_PERIOD,
    algorithm: str = DEFAULT_ALGORITHM,
) -> str:
    """Build an ``otpauth://totp/`` URI for QR codes / authenticator apps."""
    if not account:
        raise ValueError("account must not be empty")

    label = f"{issuer}:{account}" if issuer else account
    params = {
        "secret": secret.replace(" ", "").replace("-", "").upper().rstrip("="),
        "algorithm": algorithm.upper(),
        "digits": str(digits),
        "period": str(period),
    }
    if issuer:
        params["issuer"] = issuer
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    return f"otpauth://totp/{urllib.parse.quote(label)}?{query}"


# --------------------------------------------------------------------------
# Command line interface
# --------------------------------------------------------------------------


def _read_secret(args) -> str:
    if args.secret:
        return args.secret
    env = os.environ.get("TOTP_SECRET")
    if env:
        return env
    raise SystemExit(
        "no secret given: pass it positionally or set the TOTP_SECRET environment variable"
    )


def _shared_otp_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-d", "--digits", type=int, default=DEFAULT_DIGITS)
    parser.add_argument("-p", "--period", type=int, default=DEFAULT_PERIOD)
    parser.add_argument(
        "-a", "--algorithm", default=DEFAULT_ALGORITHM, choices=sorted(ALGORITHMS)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="totp", description="Generate and verify time-based one-time passwords."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="generate a new base32 shared secret")
    p_new.add_argument("-l", "--length", type=int, default=20, help="key size in bytes")
    p_new.add_argument("--account", help="also print an otpauth:// URI for this account")
    p_new.add_argument("--issuer", help="issuer shown in the authenticator app")
    _shared_otp_flags(p_new)

    p_now = sub.add_parser("now", help="print the current code")
    p_now.add_argument("secret", nargs="?", help="base32 secret (or set TOTP_SECRET)")
    p_now.add_argument(
        "-q", "--quiet", action="store_true", help="print only the code"
    )
    _shared_otp_flags(p_now)

    p_check = sub.add_parser("verify", help="verify a code")
    p_check.add_argument("code")
    p_check.add_argument("secret", nargs="?", help="base32 secret (or set TOTP_SECRET)")
    p_check.add_argument(
        "-w", "--window", type=int, default=1, help="time steps of drift to allow"
    )
    _shared_otp_flags(p_check)

    p_uri = sub.add_parser("uri", help="build an otpauth:// provisioning URI")
    p_uri.add_argument("account")
    p_uri.add_argument("secret", nargs="?", help="base32 secret (or set TOTP_SECRET)")
    p_uri.add_argument("-i", "--issuer")
    _shared_otp_flags(p_uri)

    args = parser.parse_args(argv)

    try:
        if args.command == "new":
            secret = random_secret(args.length)
            print(secret)
            if args.account:
                print(
                    provisioning_uri(
                        secret,
                        args.account,
                        issuer=args.issuer,
                        digits=args.digits,
                        period=args.period,
                        algorithm=args.algorithm,
                    )
                )
            return 0

        if args.command == "now":
            secret = _read_secret(args)
            code = totp(
                secret,
                digits=args.digits,
                period=args.period,
                algorithm=args.algorithm,
            )
            if args.quiet:
                print(code)
            else:
                left = remaining_seconds(period=args.period)
                print(f"{code}  (valid for {left:.0f}s)")
            return 0

        if args.command == "verify":
            secret = _read_secret(args)
            ok = verify(
                secret,
                args.code,
                digits=args.digits,
                period=args.period,
                algorithm=args.algorithm,
                window=args.window,
            )
            print("valid" if ok else "invalid")
            return 0 if ok else 1

        if args.command == "uri":
            secret = _read_secret(args)
            print(
                provisioning_uri(
                    secret,
                    args.account,
                    issuer=args.issuer,
                    digits=args.digits,
                    period=args.period,
                    algorithm=args.algorithm,
                )
            )
            return 0
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
