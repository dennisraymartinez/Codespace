"""Generate an Ed25519 keypair for Robinhood Crypto API enrollment.

Enroll the PUBLIC key with Robinhood. Put the PRIVATE key in .env and
nowhere else.
"""

import base64

from nacl.signing import SigningKey


def main() -> None:
    signing_key = SigningKey.generate()

    private_b64 = base64.b64encode(signing_key.encode()).decode("utf-8")
    public_b64 = base64.b64encode(signing_key.verify_key.encode()).decode("utf-8")

    print("\nPUBLIC KEY  -- paste this into Robinhood's API settings:")
    print(public_b64)
    print("\nPRIVATE KEY -- put this in .env as RH_PRIVATE_KEY, never anywhere else:")
    print(private_b64)
    print(
        "\nThis private key is not stored and cannot be recovered. If you lose it, "
        "revoke the credential in Robinhood and generate a new pair.\n"
    )


if __name__ == "__main__":
    main()
