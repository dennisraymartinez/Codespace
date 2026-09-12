"""Generate an Ed25519 keypair for the Robinhood Crypto API.

Run this once. It prints:
  * a PUBLIC key  -> paste into Robinhood web (classic) when creating the API
                     credential, under Account > Crypto > API keys
  * a PRIVATE key -> paste into your local .env as RH_PRIVATE_KEY

Nothing is written to disk. The private key exists only in this terminal
output, so copy it into .env before closing the window. If you lose it,
re-run this script and enroll the new public key.
"""

import base64

from nacl.signing import SigningKey


def main() -> None:
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
    print("PRIVATE KEY (paste into .env as RH_PRIVATE_KEY — keep secret)")
    print(f"  {private_b64}")
    print()
    print("-" * 68)
    print("Next steps:")
    print("  1. Go to robinhood.com (classic web) > Account > Crypto > API keys")
    print("  2. Create a new key, paste the PUBLIC key above, and enable the")
    print("     permissions you want. Read-only is enough for check_setup.py;")
    print("     trading permission is only needed to place orders.")
    print("  3. Robinhood shows you an API key (a UUID). Put it in .env as")
    print("     RH_API_KEY, and put the PRIVATE key above in RH_PRIVATE_KEY.")
    print("  4. Run: python check_setup.py")
    print("-" * 68)
    print()
    print("WARNING: the private key above is a bearer credential. Anyone who")
    print("has it plus your API key can act on your account within the")
    print("permissions you granted. Never commit it or paste it in chat.")
    print()


if __name__ == "__main__":
    main()
