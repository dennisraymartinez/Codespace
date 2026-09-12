"""Place a single crypto order from the command line, through the rails.

Dry run is the default. You have to ask twice for a live order: pass
--execute AND have TRADING_ENABLED=true in .env. Either one alone sends
nothing.

  # see exactly what would be sent, touch nothing
  python place_order.py --symbol BTC-USD --side buy --quantity 0.0001

  # a real market buy (needs TRADING_ENABLED=true)
  python place_order.py --symbol BTC-USD --side buy --quantity 0.0001 --execute

  # a limit sell
  python place_order.py --symbol ETH-USD --side sell --quantity 0.01 \
      --type limit --limit-price 3200 --execute
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

from dotenv import load_dotenv

from robinhood_client import (
    RobinhoodAPIError,
    RobinhoodConnectionError,
    RobinhoodCryptoClient,
)
from safety import OrderIntent, RailViolation, Rails
from trader import Trader

ENV_PATH = Path(__file__).with_name(".env")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Place one crypto order through the safety rails.",
        epilog="Without --execute this is a dry run and sends nothing.",
    )
    parser.add_argument("--symbol", required=True, help="e.g. BTC-USD")
    parser.add_argument("--side", required=True, choices=["buy", "sell"])
    sizing = parser.add_mutually_exclusive_group(required=True)
    sizing.add_argument("--quantity", help="asset quantity, e.g. 0.0001")
    sizing.add_argument(
        "--usd",
        help="dollar amount to spend, e.g. 50 — treated as a ceiling, "
        "converted to a quantity at the live quote",
    )
    parser.add_argument("--type", default="market", choices=["market", "limit"])
    parser.add_argument("--limit-price", default=None, help="required for --type limit")
    parser.add_argument(
        "--time-in-force", default="gtc", choices=["gtc", "ioc", "fok"]
    )
    parser.add_argument(
        "--client-order-id",
        default=None,
        help="reuse an id to retry a timed-out order without double-filling",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually send the order (also needs TRADING_ENABLED=true)",
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the interactive confirmation"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(ENV_PATH)

    try:
        client = RobinhoodCryptoClient(
            os.getenv("RH_API_KEY", "").strip(),
            os.getenv("RH_PRIVATE_KEY", "").strip(),
        )
    except ValueError as exc:
        print(f"credentials: {exc}", file=sys.stderr)
        print("run check_setup.py first", file=sys.stderr)
        return 1

    rails = Rails.from_env()
    trader = Trader(client, rails, dry_run=not args.execute)

    # --usd needs a live quote to size, so it resolves before the intent.
    quantity = args.quantity
    if args.usd is not None:
        try:
            quantity = trader.quantity_for_usd(
                args.symbol.strip().upper(), args.side, Decimal(str(args.usd))
            )
        except (InvalidOperation, ValueError):
            print(f"bad --usd value: {args.usd!r}", file=sys.stderr)
            return 1
        except RailViolation as exc:
            print("REFUSED — could not size the order:")
            for reason in exc.reasons:
                print(f"  - {reason}")
            return 1
        print(f"sized ${args.usd} -> {quantity} {args.symbol.split('-')[0]}")

    try:
        intent = OrderIntent.build(
            symbol=args.symbol,
            side=args.side,
            quantity=quantity,
            order_type=args.type,
            limit_price=args.limit_price,
            time_in_force=args.time_in_force,
            client_order_id=args.client_order_id,
        )
    except ValueError as exc:
        print(f"bad input: {exc}", file=sys.stderr)
        return 1

    print()
    print("Rails in effect:")
    for line in rails.describe():
        print(f"  {line}")
    print()

    # Rails first, so a doomed order is rejected before we ask anything.
    try:
        approval = trader.preview(intent)
    except RailViolation as exc:
        print("REFUSED — order was not sent:")
        for reason in exc.reasons:
            print(f"  - {reason}")
        print()
        return 1

    mode = "LIVE ORDER" if args.execute else "DRY RUN (nothing will be sent)"
    print(f"{mode}")
    print(f"  {intent.side.upper()} {intent.quantity} {intent.symbol} ({intent.order_type})")
    if intent.limit_price is not None:
        print(f"  limit price      {intent.limit_price} ({intent.time_in_force})")
    print(f"  reference price  {approval.reference_price}")
    print(f"  est. notional    ${approval.estimated_notional}")
    print(f"  client_order_id  {intent.client_order_id}")
    for note in approval.notes:
        print(f"  note: {note}")
    print()

    if args.execute and not args.yes:
        answer = input("Send this live order? type 'yes' to confirm: ").strip()
        if answer != "yes":
            print("aborted — nothing sent")
            return 1

    try:
        result = trader.submit(intent)
    except RailViolation as exc:
        # Re-checked at submit time; state may have moved since the preview.
        print("REFUSED at submit — order was not sent:")
        for reason in exc.reasons:
            print(f"  - {reason}")
        return 1
    except RobinhoodAPIError as exc:
        print(f"Robinhood rejected the order: {exc}", file=sys.stderr)
        print("Nothing was placed.", file=sys.stderr)
        return 2
    except RobinhoodConnectionError as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        if not exc.outcome_known:
            print(file=sys.stderr)
            print("*** THE ORDER MAY HAVE BEEN PLACED ***", file=sys.stderr)
            print(
                "Check your open orders for client_order_id "
                f"{intent.client_order_id} before retrying.",
                file=sys.stderr,
            )
            print(
                "To retry safely, re-run with "
                f"--client-order-id {intent.client_order_id} — Robinhood "
                "dedupes on it, so a duplicate cannot fill.",
                file=sys.stderr,
            )
        return 3

    print(json.dumps(result, indent=2))
    print()
    if not args.execute:
        print("Dry run complete. Re-run with --execute to send it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
