"""Turn the TRADING_ENABLED kill switch on or off.

Arming asks for confirmation. Disarming never does — stopping a bot must
be the easiest thing you can do, not something you have to think about.

  python arm.py          # arm: allows real orders (asks first)
  python arm.py --off    # disarm: blocks all orders, immediately
  python arm.py --status # report the current state, change nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from envfile import read_value, set_value

ENV_PATH = Path(__file__).with_name(".env")


def current_state(env_path: Path) -> bool:
    text = env_path.read_text() if env_path.exists() else ""
    return read_value(text, "TRADING_ENABLED").lower() == "true"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--off", action="store_true", help="disarm immediately")
    group.add_argument("--status", action="store_true", help="report, change nothing")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation")
    args = parser.parse_args(argv)

    if not ENV_PATH.exists():
        print(f"{ENV_PATH.name} not found — run check_setup.py first", file=sys.stderr)
        return 1

    armed = current_state(ENV_PATH)
    state = "ARMED — real orders allowed" if armed else "DISARMED — orders blocked"

    if args.status:
        print(f"TRADING_ENABLED is {str(armed).lower()}  ({state})")
        return 0

    if args.off:
        # No confirmation, no questions. Disarming is always safe.
        set_value(ENV_PATH, "TRADING_ENABLED", "false")
        print("DISARMED. TRADING_ENABLED=false — no order can be sent.")
        return 0

    if armed:
        print("Already armed. TRADING_ENABLED=true.")
        print("To disarm: python arm.py --off")
        return 0

    cap = read_value(ENV_PATH.read_text(), "MAX_ORDER_USD") or "?"
    daily = read_value(ENV_PATH.read_text(), "MAX_DAILY_USD") or "?"
    symbols = read_value(ENV_PATH.read_text(), "ALLOWED_SYMBOLS") or "?"

    print()
    print("This allows place_order.py --execute to send REAL orders.")
    print(f"  per-order cap   ${cap}")
    print(f"  daily cap       ${daily}")
    print(f"  symbols         {symbols}")
    print()
    print("--execute and the typed confirmation still apply to each order.")
    print()

    if not args.yes:
        try:
            answer = input("Type 'arm' to enable trading: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\naborted — still disarmed")
            return 1
        if answer != "arm":
            print("aborted — still disarmed")
            return 1

    set_value(ENV_PATH, "TRADING_ENABLED", "true")
    print()
    print("ARMED. TRADING_ENABLED=true.")
    print("To disarm at any time: python arm.py --off")
    return 0


if __name__ == "__main__":
    sys.exit(main())
