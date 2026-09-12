"""Read-only preflight check for the Robinhood Crypto API setup.

This script NEVER places, modifies, or cancels an order. It only issues GET
requests, so it is safe to run as often as you like.

Local checks run first, so the safety rails are reported even when the
credentials are missing:

  1. safety rails parse and are internally consistent
  2. today's usage against the daily caps
  3. .env exists and both credentials are present
  4. the private key decodes to a valid Ed25519 seed
  5. the credential is enrolled and signing works
  6. market data is reachable for every allow-listed symbol
  7. holdings are readable
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from robinhood_client import RobinhoodAPIError, RobinhoodCryptoClient
from safety import Ledger, Rails

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


def is_tls_interception(exc: BaseException) -> bool:
    """True when a failure is a cert-verification error, not plain no-network.

    Walks the exception chain: requests wraps the underlying ssl error
    several layers deep, so the useful string is rarely on the outermost
    exception.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = f"{type(exc).__name__}: {exc}"
        if "CERTIFICATE_VERIFY_FAILED" in text or "SSLCertVerificationError" in text:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def check_rails() -> Rails | None:
    """Validate the rails. Local only — no network, no credentials needed."""
    section("1. Safety rails")
    try:
        rails = Rails.from_env()
    except ValueError as exc:
        fail(f"could not parse the rails: {exc}", "check .env for typos")
        return None

    for line in rails.describe():
        print(f"         {line}")
    print()

    if rails.trading_enabled:
        warn("TRADING_ENABLED=true — the bot is cleared to place REAL orders")
    else:
        ok("TRADING_ENABLED=false — orders are blocked (kill switch on)")

    if rails.max_order_usd <= 0:
        fail("MAX_ORDER_USD must be greater than 0", "nothing could ever trade")
    else:
        ok(f"per-order cap ${rails.max_order_usd}")

    if not rails.allowed_symbols:
        fail("ALLOWED_SYMBOLS is empty", "the bot would have nothing to trade")
    else:
        ok(f"{len(rails.allowed_symbols)} symbol(s) allow-listed")

    if rails.max_daily_usd < rails.max_order_usd:
        warn(
            f"MAX_DAILY_USD ${rails.max_daily_usd} is below MAX_ORDER_USD "
            f"${rails.max_order_usd} — the per-order cap can never be reached"
        )
    if rails.max_orders_per_day <= 0:
        fail("MAX_ORDERS_PER_DAY must be greater than 0")
    if rails.slippage_buffer_pct < 0:
        fail("SLIPPAGE_BUFFER_PCT cannot be negative")
    if rails.max_limit_deviation_pct <= 0:
        fail("MAX_LIMIT_DEVIATION_PCT must be greater than 0")

    return rails


def check_usage(rails: Rails) -> None:
    section("2. Today's usage (UTC)")
    day = Ledger().load()
    ok(
        f"{day.order_count} of {rails.max_orders_per_day} orders, "
        f"${day.notional_usd} of ${rails.max_daily_usd} notional used"
    )
    if day.order_count >= rails.max_orders_per_day:
        warn("daily order count is exhausted — further orders will be refused")
    if day.notional_usd >= rails.max_daily_usd:
        warn("daily notional is exhausted — further orders will be refused")


def main() -> int:
    print()
    print("=" * 68)
    print("  Robinhood Crypto API — setup check (read-only, places nothing)")
    print("=" * 68)

    if not ENV_PATH.exists():
        section("0. Environment file")
        fail(".env not found", "cp .env.example .env, then fill it in")
        return report()

    load_dotenv(ENV_PATH)

    rails = check_rails()
    if rails is None:
        return report()
    check_usage(rails)

    # -- credentials ---------------------------------------------------
    section("3. Credentials")
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

    if not api_key or not private_key:
        return report()

    # -- key format ----------------------------------------------------
    section("4. Key format")
    try:
        client = RobinhoodCryptoClient(api_key, private_key)
        ok("private key decodes to a valid Ed25519 seed")
    except ValueError as exc:
        fail(str(exc), "re-run generate_keys.py and re-copy both values")
        return report()

    # -- authentication ------------------------------------------------
    section("5. Authentication (GET accounts)")
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
                "not match the keypair. Re-check robinhood.com web classic > "
                "crypto account settings > Add key.",
            )
        else:
            fail(f"{exc.status_code} from accounts endpoint: {exc.body}")
        return report()
    except Exception as exc:  # network, TLS, DNS
        if is_tls_interception(exc):
            fail(
                "TLS certificate verification failed",
                "something on this machine (antivirus or a corporate proxy) is "
                "re-signing HTTPS with a private root that Python does not "
                "trust. Export the Windows trust store and point requests at "
                "it — see 'TLS certificate errors' in README.md.",
            )
        else:
            fail(f"could not reach {type(exc).__name__}: {exc}", "check connectivity")
        return report()

    # -- market data ---------------------------------------------------
    section("6. Market data")
    symbols = sorted(rails.allowed_symbols)
    quoted: set[str] = set()
    try:
        quotes = client.get_best_bid_ask(*symbols)
        for quote in quotes.get("results", []):
            quoted.add(quote.get("symbol", ""))
            ok(
                f"{quote.get('symbol')}: bid {quote.get('bid_inclusive_of_sell_spread')} "
                f"/ ask {quote.get('ask_inclusive_of_buy_spread')}"
            )
        missing = set(symbols) - quoted
        if missing:
            # An allow-listed symbol with no quote can never be ordered:
            # the rails refuse rather than size an order blind.
            fail(
                "no quote for " + ", ".join(sorted(missing)),
                "remove it from ALLOWED_SYMBOLS or fix the spelling",
            )
    except RobinhoodAPIError as exc:
        fail(f"market data unavailable: {exc}")

    # -- holdings ------------------------------------------------------
    section("7. Holdings")
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
