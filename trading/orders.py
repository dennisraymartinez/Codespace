"""Show recent orders and how they filled.

  python orders.py              # recent orders, newest first
  python orders.py <order_id>   # one order in full
  python orders.py --watch <id> # poll until it reaches a final state

Read-only: issues GET requests only, never places or cancels anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from safety import plain
from robinhood_client import (
    RobinhoodConnectionError,
    RobinhoodCryptoClient,
    RobinhoodError,
)

ENV_PATH = Path(__file__).with_name(".env")

# States Robinhood will not move an order out of.
FINAL_STATES = {"filled", "canceled", "cancelled", "rejected", "failed"}


def summarise(order: dict[str, Any]) -> str:
    state = order.get("state", "?")
    filled = Decimal(str(order.get("filled_asset_quantity") or 0))
    config = order.get("market_order_config") or order.get("limit_order_config") or {}
    asked = Decimal(str(config.get("asset_quantity") or 0))
    price = order.get("average_price")

    line = (
        f"{str(order.get('id', '?'))[:8]}  "
        f"{order.get('created_at', '')[:19]}  "
        f"{str(order.get('side', '?')).upper():4} "
        f"{order.get('symbol', '?'):9} "
        f"{state:10}"
    )
    if filled > 0 and price:
        as_decimal = Decimal(str(price))
        spent = filled * as_decimal
        # Trim the trailing zeros Robinhood pads prices with, and show a
        # price to the cent — the extra 12 decimal places are noise.
        shown = plain(as_decimal.quantize(Decimal("0.01")))
        line += f" filled {plain(filled)} @ {shown} = ${spent.quantize(Decimal('0.01'))}"
    elif asked > 0:
        line += f" asked {plain(asked)}, filled {plain(filled)}"
    return line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("order_id", nargs="?", help="show one order in full")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="with an order id, poll until it reaches a final state",
    )
    parser.add_argument("--timeout", type=int, default=60, help="seconds to watch")
    args = parser.parse_args(argv)

    load_dotenv(ENV_PATH)
    try:
        client = RobinhoodCryptoClient(
            os.getenv("RH_API_KEY", "").strip(),
            os.getenv("RH_PRIVATE_KEY", "").strip(),
        )
    except ValueError as exc:
        print(f"credentials: {exc}", file=sys.stderr)
        return 1

    try:
        if args.order_id and args.watch:
            deadline = time.time() + args.timeout
            while True:
                order = client.get_order(args.order_id)
                print(summarise(order))
                if order.get("state") in FINAL_STATES or time.time() > deadline:
                    print()
                    print(json.dumps(order, indent=2))
                    return 0
                time.sleep(2)

        if args.order_id:
            print(json.dumps(client.get_order(args.order_id), indent=2))
            return 0

        results = client.get_orders().get("results", [])
        if not results:
            print("no orders")
            return 0
        print(f"{len(results)} order(s), newest first:")
        print()
        for order in results:
            print("  " + summarise(order))
        return 0

    except RobinhoodConnectionError as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        return 3
    except RobinhoodError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
