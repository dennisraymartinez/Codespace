"""Read-only preflight check for the Robinhood Crypto API setup.

This script NEVER places, modifies, or cancels an order. It only issues GET
requests, so it is safe to run as often as you like.

It verifies, in order:
  1. .env exists and both credentials are present
  2. the private key decodes to a valid Ed25519 seed
  3. the credential is enrolled and signing works (account endpoint)
  4. market data is reachable
  5. the safety rails in .env are sane
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from robinhood_client import RobinhoodAPIError, RobinhoodCryptoClient

ENV_PATH = Path(__file__).with_name(".env")

PASS = "  [ ok ]"
FAIL = "  [FAIL]"
WARN = "  [warn]"

failures: list[str] = []


def fail(message: str, hint: str = "") -> None:
    print(f"{FAIL} {message}")
    if hint:
        print(f"         -> {hint}")
    failures.append(message)


def ok(message: str) -> None:
    print(f"{PASS} {message}")


def warn(message: str) -> None:
    print(f"{WARN} {message}")


def section(title: str) -> None:
    print()
    print(title)


def main() -> int:
    print()
    print("=" * 68)
    print("  Robinhood Crypto API — setup check (read-only, places nothing)")
    print("=" * 68)

    # 1. credentials present -------------------------------------------
    section("1. Credentials")

    if not ENV_PATH.exists():
        fail(".env not found", "cp .env.example .env, then fill it in")
        return report()

    load_dotenv(ENV_PATH)
    api_key = os.getenv("RH_API_KEY", "").strip()
    private_key = os.getenv("RH_PRIVATE_KEY", "").strip()

    if not api_key:
        fail("RH_API_KEY is empty", "paste the API key Robinhood gave you")
    else:
        ok(f"RH_API_KEY present ({api_key[:12]}...)")

    if not private_key:
        fail("RH_PRIVATE_KEY is empty", "paste the private key from generate_keys.py")
    else:
        ok("RH_PRIVATE_KEY present (value not shown)")

    if failures:
        return report()

    # 2. key is well-formed --------------------------------------------
    section("2. Key format")
    try:
        client = RobinhoodCryptoClient(api_key, private_key)
        ok("private key decodes to a valid Ed25519 seed")
    except ValueError as exc:
        fail(str(exc), "re-run generate_keys.py and re-copy both values")
        return report()

    # 3. signing + enrollment ------------------------------------------
    section("3. Authentication (GET accounts)")
    try:
        account = client.get_account()
        ok(f"authenticated — account {account.get('account_number', '?')}")
        status = account.get("status", "unknown")
        if status == "active":
            ok(f"account status: {status}")
        else:
            warn(f"account status: {status}")
        if "buying_power" in account:
            ok(f"buying power: {account['buying_power']}")
    except RobinhoodAPIError as exc:
        if exc.status_code in (401, 403):
            fail(
                f"rejected with {exc.status_code}",
                "the public key may not be enrolled yet, or the API key does "
                "not match the keypair. Re-check Account > Crypto > API keys.",
            )
        else:
            fail(f"{exc.status_code} from accounts endpoint: {exc.body}")
        return report()
    except Exception as exc:  # network, TLS, DNS
        fail(f"could not reach {type(exc).__name__}: {exc}", "check connectivity")
        return report()

    # 4. market data ---------------------------------------------------
    section("4. Market data")
    symbols = [
        s.strip()
        for s in os.getenv("ALLOWED_SYMBOLS", "BTC-USD").split(",")
        if s.strip()
    ]
    try:
        quotes = client.get_best_bid_ask(*symbols)
        for quote in quotes.get("results", []):
            ok(
                f"{quote.get('symbol')}: bid {quote.get('bid_inclusive_of_sell_spread')} "
                f"/ ask {quote.get('ask_inclusive_of_buy_spread')}"
            )
        if not quotes.get("results"):
            warn("market data returned no results for " + ", ".join(symbols))
    except RobinhoodAPIError as exc:
        fail(f"market data unavailable: {exc}")

    section("5. Holdings")
    try:
        holdings = client.get_holdings()
        results = holdings.get("results", [])
        if results:
            for holding in results:
                ok(f"{holding.get('asset_code')}: {holding.get('total_quantity')}")
        else:
            ok("no crypto holdings (empty account)")
    except RobinhoodAPIError as exc:
        fail(f"holdings unavailable: {exc}")

    # 6. safety rails --------------------------------------------------
    section("6. Safety rails")
    trading_enabled = os.getenv("TRADING_ENABLED", "false").strip().lower()
    if trading_enabled == "true":
        warn("TRADING_ENABLED=true — the bot is cleared to place real orders")
    else:
        ok("TRADING_ENABLED=false — orders are blocked (kill switch on)")

    try:
        max_order = float(os.getenv("MAX_ORDER_USD", "0"))
        if max_order <= 0:
            fail("MAX_ORDER_USD must be greater than 0")
        else:
            ok(f"MAX_ORDER_USD = {max_order:.2f}")
    except ValueError:
        fail("MAX_ORDER_USD is not a number")

    if symbols:
        ok("ALLOWED_SYMBOLS = " + ", ".join(symbols))
    else:
        fail("ALLOWED_SYMBOLS is empty", "the bot would have nothing to trade")

    return report()


def report() -> int:
    print()
    print("=" * 68)
    if failures:
        print(f"  {len(failures)} check(s) failed:")
        for item in failures:
            print(f"    - {item}")
        print("=" * 68)
        print()
        return 1
    print("  All checks passed. Nothing was ordered.")
    print("=" * 68)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
