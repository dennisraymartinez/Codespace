"""Read-only smoke test. Places no orders.

Run this before anything else. If it passes, your keypair, enrollment and
request signing are all correct.
"""

import sys

from config import Config, ConfigError
from rh_crypto import RobinhoodCrypto, RobinhoodError


def main() -> int:
    try:
        config = Config.load()
    except ConfigError as exc:
        print(f"Config error: {exc}")
        return 1

    client = RobinhoodCrypto(
        api_key=config.api_key,
        private_key_b64=config.private_key,
        dry_run=config.dry_run,
        max_order_usd=config.max_order_usd,
    )

    print(f"Dry run: {config.dry_run}   Per-order cap: ${config.max_order_usd:.2f}")
    if not config.dry_run:
        print("WARNING: dry run is OFF. Orders placed by this code would be real.")
    print()

    try:
        account = client.get_account()
        print("Account:", account)

        holdings = client.get_holdings()
        print("Holdings:", holdings)

        print("BTC-USD quote:", client.get_best_bid_ask("BTC-USD"))
    except RobinhoodError as exc:
        print(f"\nAPI call failed: {exc}")
        return 1

    print("\nRead-only checks passed. Credentials and signing are working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
