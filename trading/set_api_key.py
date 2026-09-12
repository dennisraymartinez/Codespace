"""Put the Robinhood API key into .env, correctly.

The API key is the one value you have to copy by hand, from Robinhood's
website. This writes it to the right variable, strips anything pasted
around it, and checks it does not look like a private key.

  python set_api_key.py                  # prompts for the key
  python set_api_key.py rh-api-1f3c...   # or pass it as an argument
"""

from __future__ import annotations

import sys
from pathlib import Path

from envfile import clean, read_value, set_value

ENV_PATH = Path(__file__).with_name(".env")


def looks_like_a_private_key(value: str) -> bool:
    import base64
    import binascii

    if not value or "-" in value:
        return False
    try:
        return len(base64.b64decode(value, validate=True)) in (32, 64)
    except (binascii.Error, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if argv:
        raw = argv[0]
    else:
        print()
        print("Paste the API key Robinhood showed you when it accepted your")
        print("public key. It looks like rh-api-xxxxxxxx-xxxx-... with hyphens.")
        print("This is NOT the private key — that is already in .env.")
        print()
        try:
            raw = input("RH_API_KEY: ")
        except (EOFError, KeyboardInterrupt):
            print("\naborted — nothing changed")
            return 1

    value = clean(raw)

    if not value:
        print("no value given — nothing changed", file=sys.stderr)
        return 1

    if looks_like_a_private_key(value):
        print(file=sys.stderr)
        print("REFUSED: that looks like a PRIVATE KEY, not an API key.", file=sys.stderr)
        print(file=sys.stderr)
        print("  A private key is ~44 characters of base64 ending in '='.", file=sys.stderr)
        print("  An API key is a hyphenated identifier from Robinhood's site.", file=sys.stderr)
        print(file=sys.stderr)
        print("If that private key has been exposed anywhere, delete the", file=sys.stderr)
        print("credential at Robinhood and run: python generate_keys.py --force", file=sys.stderr)
        return 1

    private = read_value(ENV_PATH.read_text() if ENV_PATH.exists() else "", "RH_PRIVATE_KEY")
    if private and value == private:
        print("REFUSED: that is the value already in RH_PRIVATE_KEY.", file=sys.stderr)
        return 1

    set_value(ENV_PATH, "RH_API_KEY", value)
    masked = value[:10] + "..." if len(value) > 13 else value
    print(f"RH_API_KEY set to {masked} in {ENV_PATH.name}")
    print("Next: python check_setup.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
