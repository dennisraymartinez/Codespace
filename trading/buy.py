"""Guided buying. One command, asks you everything, explains as it goes.

  python buy.py

Walks through: pick a symbol, pick an amount, see the real numbers,
confirm, place, watch it fill. Arms the kill switch only for the order
itself and disarms immediately afterwards, so the armed window is as
short as it can be.

Every safety rail is the same one place_order.py uses — this is a
friendlier front door, not a shortcut past anything.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path

from dotenv import load_dotenv

import orders as orders_mod
from envfile import read_value, set_value
from robinhood_client import RobinhoodError, RobinhoodCryptoClient
from safety import OrderIntent, RailViolation, Rails, plain
from trader import Trader

ENV_PATH = Path(__file__).with_name(".env")
RULE = "-" * 66


def ask(question: str) -> str:
    try:
        return input(question).strip()
    except (EOFError, KeyboardInterrupt):
        print("\ncancelled — nothing was bought")
        raise SystemExit(1)


def quotes_for(client: RobinhoodCryptoClient, symbols: list[str]) -> dict:
    try:
        payload = client.get_best_bid_ask(*symbols)
    except RobinhoodError:
        return {}
    out = {}
    for q in payload.get("results", []):
        try:
            bid = Decimal(str(q["bid_inclusive_of_sell_spread"]))
            ask_p = Decimal(str(q["ask_inclusive_of_buy_spread"]))
            out[q["symbol"]] = (bid, ask_p)
        except (KeyError, TypeError, InvalidOperation):
            continue
    return out


def spread_pct(bid: Decimal, ask_p: Decimal) -> Decimal:
    mid = (bid + ask_p) / 2
    return (ask_p - bid) / mid * 100 if mid else Decimal(0)


def choose_symbol(symbols: list[str], quotes: dict) -> str:
    print()
    print("What do you want to buy?")
    print()
    for i, symbol in enumerate(symbols, 1):
        line = f"  {i}. {symbol:10}"
        if symbol in quotes:
            bid, ask_p = quotes[symbol]
            line += f" price {plain(ask_p):>14}   round-trip cost {spread_pct(bid, ask_p):.2f}%"
        else:
            line += "  (no quote available)"
        print(line)
    print()
    print("  'round-trip cost' is what you lose buying then selling straight")
    print("  back. The price has to move more than that before you profit.")
    print()
    while True:
        choice = ask(f"Pick 1-{len(symbols)} (or q to quit): ")
        if choice.lower() in ("q", "quit", ""):
            print("cancelled — nothing was bought")
            raise SystemExit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(symbols):
            return symbols[int(choice) - 1]
        print(f"  please type a number from 1 to {len(symbols)}")


def choose_amount(rails: Rails, spent_today: Decimal) -> Decimal:
    remaining = rails.max_daily_usd - spent_today
    ceiling = min(rails.max_order_usd, remaining)
    print()
    print(f"How many dollars? Your limits allow up to ${ceiling} right now")
    print(f"  (per-order cap ${rails.max_order_usd}, "
          f"${remaining} left of today's ${rails.max_daily_usd})")
    print()
    while True:
        raw = ask("Amount in dollars (or q to quit): ").lstrip("$").strip()
        if raw.lower() in ("q", "quit", ""):
            print("cancelled — nothing was bought")
            raise SystemExit(0)
        try:
            amount = Decimal(raw)
        except InvalidOperation:
            print("  that is not a number — try something like 50")
            continue
        if amount <= 0:
            print("  the amount has to be more than zero")
            continue
        if amount > ceiling:
            print(f"  ${amount} is over your limit of ${ceiling}.")
            print("  Raise MAX_ORDER_USD in .env if you meant it.")
            continue
        return amount


def main() -> int:
    print()
    print("=" * 66)
    print("  Guided buy — nothing is ordered until you confirm")
    print("=" * 66)

    load_dotenv(ENV_PATH)
    try:
        client = RobinhoodCryptoClient(
            os.getenv("RH_API_KEY", "").strip(),
            os.getenv("RH_PRIVATE_KEY", "").strip(),
        )
    except ValueError as exc:
        print(f"\ncredentials: {exc}", file=sys.stderr)
        print("run: python check_setup.py", file=sys.stderr)
        return 1

    rails = Rails.from_env()
    symbols = sorted(rails.allowed_symbols)
    if not symbols:
        print("\nALLOWED_SYMBOLS is empty — add one with:", file=sys.stderr)
        print("  python pairs.py --allow BTC-USD", file=sys.stderr)
        return 1

    # Preview against a disarmed trader: it can price and check everything
    # without being able to send.
    preview_trader = Trader(client, rails, dry_run=True)
    day = preview_trader.ledger.load()

    quotes = quotes_for(client, symbols)
    symbol = choose_symbol(symbols, quotes)
    amount = choose_amount(rails, day.notional_usd)

    try:
        quantity = preview_trader.quantity_for_usd(symbol, "buy", amount)
    except RailViolation as exc:
        print("\nCannot size that order:")
        for reason in exc.reasons:
            print(f"  - {reason}")
        return 1

    intent = OrderIntent.build(symbol, "buy", quantity)
    try:
        approval = preview_trader.preview(intent, sending=False)
    except RailViolation as exc:
        print("\nRefused — nothing was sent:")
        for reason in exc.reasons:
            print(f"  - {reason}")
        return 1

    base = symbol.split("-")[0]
    print()
    print(RULE)
    print("  Here is exactly what would happen")
    print(RULE)
    print(f"  buy               {plain(quantity)} {base}")
    print(f"  at about          {plain(approval.reference_price)} each")
    print(f"  costing about     ${approval.estimated_notional}")
    if symbol in quotes:
        bid, ask_p = quotes[symbol]
        cost = amount * spread_pct(bid, ask_p) / 100
        print(f"  round-trip cost   ${cost.quantize(Decimal('0.01'))} "
              f"({spread_pct(bid, ask_p):.2f}%) if you sold straight back")
    print(f"  after this today  {day.order_count + 1} of {rails.max_orders_per_day} "
          f"orders, ${day.notional_usd + approval.estimated_notional} "
          f"of ${rails.max_daily_usd}")
    print(RULE)
    print()
    print("  The exact price is set by the market when the order lands, so")
    print("  the final cost will be close to this but not identical.")
    print()

    if ask("Type 'buy' to place this order (anything else cancels): ") != "buy":
        print("cancelled — nothing was bought")
        return 0

    was_armed = read_value(ENV_PATH.read_text(), "TRADING_ENABLED").lower() == "true"
    if not was_armed:
        print()
        print("Arming the kill switch for this one order...")
        set_value(ENV_PATH, "TRADING_ENABLED", "true")

    try:
        # Armed by construction: the typed 'buy' above is this flow's
        # equivalent of --execute, and the .env switch was just set. Every
        # other rail is untouched and still enforced on submit.
        live_rails = replace(rails, trading_enabled=True)
        trader = Trader(client, live_rails, dry_run=False)
        print("Placing...")
        response = trader.submit(intent)
    except RailViolation as exc:
        print("\nRefused at the last moment — nothing was sent:")
        for reason in exc.reasons:
            print(f"  - {reason}")
        return 1
    except RobinhoodError as exc:
        print(f"\nRobinhood refused the order: {exc}", file=sys.stderr)
        return 2
    finally:
        if not was_armed:
            set_value(ENV_PATH, "TRADING_ENABLED", "false")
            print("Kill switch back off.")

    order_id = response.get("id", "")
    print()
    print(f"Placed. Order {order_id}")
    print("Waiting for it to fill...")
    print()

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            order = client.get_order(order_id)
        except RobinhoodError:
            break
        if order.get("state") in orders_mod.FINAL_STATES:
            print("  " + orders_mod.summarise(order))
            break
        time.sleep(2)
    else:
        print("  still working — check later with:")
        print(f"    python orders.py {order_id}")

    print()
    print("Done. To see everything you hold:  python check_setup.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
