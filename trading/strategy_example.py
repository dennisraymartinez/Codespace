"""A deliberately trivial example of the read -> decide -> order loop.

The 'strategy' here is a placeholder that does nothing useful. The point is
the shape: pull a quote, make a decision, route it through the guarded
order path. Replace `decide()` with your own logic and backtest it somewhere
other than a live account.
"""

import sys

from config import Config, ConfigError
from rh_crypto import DryRunBlocked, OrderTooLarge, RobinhoodCrypto, RobinhoodError

SYMBOL = "BTC-USD"
QUANTITY = 0.00001


def decide(price: float) -> str:
    """Return 'buy', 'sell' or 'hold'.

    Placeholder only. A real signal needs a thesis, out-of-sample testing,
    and an accounting of fees and spread -- none of which is here.
    """
    return "hold"


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

    try:
        price = client.mid_price(SYMBOL)
    except RobinhoodError as exc:
        print(f"Could not fetch price: {exc}")
        return 1

    action = decide(price)
    print(f"{SYMBOL} mid ${price:,.2f} -> {action}")

    if action == "hold":
        return 0

    try:
        result = client.place_market_order(SYMBOL, action, QUANTITY)
        print("Order accepted:", result)
    except DryRunBlocked as exc:
        print(exc)
    except OrderTooLarge as exc:
        print(f"Blocked by cap: {exc}")
    except RobinhoodError as exc:
        print(f"Order rejected: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
