"""List the crypto pairs Robinhood trades, and manage the allow-list.

  python pairs.py                  # every tradable pair
  python pairs.py --quotes         # with live bid/ask and spread
  python pairs.py --allow SOL-USD  # validate, then add to ALLOWED_SYMBOLS
  python pairs.py --deny SOL-USD   # remove from ALLOWED_SYMBOLS

Adding a symbol is checked against Robinhood first: a pair they do not
trade, or one that is not currently tradable, is refused rather than
written into .env to fail later at order time.
"""

from __future__ import annotations

import argparse
import os
import sys
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from envfile import read_value, set_value
from robinhood_client import RobinhoodCryptoClient, RobinhoodError
from safety import plain

ENV_PATH = Path(__file__).with_name(".env")


def allowed_symbols(env_path: Path) -> list[str]:
    text = env_path.read_text() if env_path.exists() else ""
    raw = read_value(text, "ALLOWED_SYMBOLS")
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def set_allowed(env_path: Path, symbols: list[str]) -> None:
    set_value(env_path, "ALLOWED_SYMBOLS", ",".join(sorted(set(symbols))))


def client_from_env() -> RobinhoodCryptoClient:
    load_dotenv(ENV_PATH)
    return RobinhoodCryptoClient(
        os.getenv("RH_API_KEY", "").strip(),
        os.getenv("RH_PRIVATE_KEY", "").strip(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", action="store_true", help="include live prices")
    parser.add_argument("--allow", metavar="SYMBOL", help="add to ALLOWED_SYMBOLS")
    parser.add_argument("--deny", metavar="SYMBOL", help="remove from ALLOWED_SYMBOLS")
    args = parser.parse_args(argv)

    if args.deny:
        symbol = args.deny.strip().upper()
        current = allowed_symbols(ENV_PATH)
        if symbol not in current:
            print(f"{symbol} is not in ALLOWED_SYMBOLS — nothing to do")
            return 0
        set_allowed(ENV_PATH, [s for s in current if s != symbol])
        print(f"removed {symbol}. Now: {', '.join(allowed_symbols(ENV_PATH))}")
        return 0

    try:
        client = client_from_env()
    except ValueError as exc:
        print(f"credentials: {exc}", file=sys.stderr)
        return 1

    try:
        results = client.get_trading_pairs().get("results", [])
    except RobinhoodError as exc:
        print(f"could not read trading pairs: {exc}", file=sys.stderr)
        return 2

    by_symbol = {r.get("symbol"): r for r in results if r.get("symbol")}

    if args.allow:
        symbol = args.allow.strip().upper()
        pair = by_symbol.get(symbol)
        if pair is None:
            print(f"REFUSED: Robinhood does not list {symbol}.", file=sys.stderr)
            close = [s for s in by_symbol if s.startswith(symbol.split("-")[0])]
            if close:
                print(f"  did you mean: {', '.join(sorted(close))}", file=sys.stderr)
            return 1
        status = str(pair.get("status", "")).lower()
        if status and status != "tradable":
            print(f"REFUSED: {symbol} is not tradable (status: {status}).", file=sys.stderr)
            return 1
        current = allowed_symbols(ENV_PATH)
        if symbol in current:
            print(f"{symbol} is already allowed")
            return 0
        set_allowed(ENV_PATH, current + [symbol])
        print(f"added {symbol}. Now: {', '.join(allowed_symbols(ENV_PATH))}")
        print("The caps still apply — this widens what can be traded, not how much.")
        return 0

    allowed = set(allowed_symbols(ENV_PATH))
    quotes: dict[str, tuple[str, str]] = {}
    if args.quotes:
        try:
            payload = client.get_best_bid_ask(*sorted(by_symbol))
            for q in payload.get("results", []):
                quotes[q.get("symbol", "")] = (
                    str(q.get("bid_inclusive_of_sell_spread", "")),
                    str(q.get("ask_inclusive_of_buy_spread", "")),
                )
        except RobinhoodError as exc:
            print(f"(quotes unavailable: {exc})", file=sys.stderr)

    print(f"{len(by_symbol)} pair(s) on Robinhood. * = in your ALLOWED_SYMBOLS")
    print()
    for symbol in sorted(by_symbol):
        pair = by_symbol[symbol]
        mark = "*" if symbol in allowed else " "
        line = f" {mark} {symbol:12} min {pair.get('min_order_size', '?'):>12}"
        if symbol in quotes:
            bid, ask = quotes[symbol]
            try:
                b, a = Decimal(bid), Decimal(ask)
                spread = (a - b) / ((a + b) / 2) * 100
                line += f"  bid {plain(b):>14} ask {plain(a):>14} spread {spread:5.2f}%"
            except Exception:
                pass
        print(line)

    print()
    print("Add one with:  python pairs.py --allow SYMBOL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
