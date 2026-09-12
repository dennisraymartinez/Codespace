"""Generate an Ed25519 keypair for the Robinhood Crypto API.

By default the PRIVATE key is written straight into .env and never
displayed. Only the PUBLIC key is printed, because that is the only half
you need to handle — you paste it into Robinhood's web form.

This is deliberate: a private key you never see is a private key you
cannot paste into a chat window, a ticket, or a screenshot.

  python generate_keys.py                 # write private key to .env
  python generate_keys.py --force         # overwrite an existing one
  python generate_keys.py --print-private # print it instead (last resort)
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

from envfile import read_value, set_value
from nacl.signing import SigningKey

ENV_PATH = Path(__file__).with_name(".env")
EXAMPLE_PATH = Path(__file__).with_name(".env.example")

ENROLL_STEPS = """Next steps:
  1. Sign in at robinhood.com on WEB CLASSIC (a desktop browser — this
     cannot be done in the mobile app).
  2. Go to your crypto account settings and select "Add key".
  3. Paste the PUBLIC key above and name the credential.
  4. Select the API actions to enable. Read-only is enough for
     check_setup.py; placing orders needs the trading action.
  5. Robinhood shows you an API key. Put it in .env as RH_API_KEY.
  6. Run: python check_setup.py"""


def existing_private_key(text: str) -> str:
    return read_value(text, "RH_PRIVATE_KEY")


def write_private_key(env_path: Path, private_b64: str, force: bool = False) -> None:
    """Set RH_PRIVATE_KEY in .env without disturbing anything else.

    Refuses to clobber a key that is already there unless force is set —
    overwriting a credential you are actively using would be silent
    breakage, and the old key cannot be recovered.
    """
    if env_path.exists():
        text = env_path.read_text()
    elif EXAMPLE_PATH.exists():
        text = EXAMPLE_PATH.read_text()
    else:
        text = "RH_API_KEY=\nRH_PRIVATE_KEY=\n"

    if existing_private_key(text) and not force:
        raise SystemExit(
            f"{env_path.name} already holds an RH_PRIVATE_KEY.\n"
            "Re-run with --force to replace it. The current key cannot be\n"
            "recovered afterwards, and any credential enrolled against it\n"
            "stops working — delete that credential at Robinhood too."
        )

    if not env_path.exists():
        env_path.write_text(text)
    set_value(env_path, "RH_PRIVATE_KEY", private_b64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-private",
        action="store_true",
        help="print the private key instead of writing it to .env",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an RH_PRIVATE_KEY already present in .env",
    )
    args = parser.parse_args(argv)

    signing_key = SigningKey.generate()
    private_b64 = base64.b64encode(bytes(signing_key)).decode()
    public_b64 = base64.b64encode(bytes(signing_key.verify_key)).decode()

    print()
    print("=" * 68)
    print("  Robinhood Crypto API keypair")
    print("=" * 68)
    print()
    print("PUBLIC KEY  (enroll this on Robinhood web classic)")
    print(f"  {public_b64}")
    print()

    if args.print_private:
        print("PRIVATE KEY (paste into .env as RH_PRIVATE_KEY — keep secret)")
        print(f"  {private_b64}")
        print()
        print("*** This is on your screen now. Do not paste it into a chat,")
        print("*** an email, a ticket, or a screenshot. If it leaks before")
        print("*** you enroll the public key, just generate a new pair. If it")
        print("*** leaks afterwards, delete the credential at Robinhood first.")
    else:
        write_private_key(ENV_PATH, private_b64, force=args.force)
        print(f"PRIVATE KEY written to {ENV_PATH.name} — not displayed.")
        print("  You never need to see or copy it. Only the public key above")
        print("  leaves this machine.")

    print()
    print("-" * 68)
    print(ENROLL_STEPS)
    print("-" * 68)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
