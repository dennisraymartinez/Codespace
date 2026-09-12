"""Tests for the safety rails and the Trader choke point.

No network: the client is faked, so these never touch Robinhood.

    python test_safety.py          # standalone
    pytest test_safety.py          # if you have pytest
"""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

from robinhood_client import RobinhoodAPIError, RobinhoodConnectionError
from generate_keys import existing_private_key, write_private_key
from safety import Ledger, OrderIntent, RailViolation, Rails
from trader import Trader

D = Decimal


def rails(**overrides) -> Rails:
    defaults = dict(
        trading_enabled=True,
        allowed_symbols=frozenset({"BTC-USD", "ETH-USD"}),
        max_order_usd=D("25"),
        max_limit_deviation_pct=D("5"),
        max_daily_usd=D("100"),
        max_orders_per_day=10,
        slippage_buffer_pct=D("1"),
    )
    defaults.update(overrides)
    return Rails(**defaults)


class FakeClient:
    """Stands in for RobinhoodCryptoClient. Records what it was asked to do."""

    def __init__(
        self,
        bid="100",
        ask="101",
        held="1.0",
        fail_order=None,
        quote_offline=False,
        order_offline=False,
        pair=None,
    ):
        self.pair = (
            pair
            if pair is not None
            else {
                "quantity_increment": "0.00000001",
                "min_order_size": "0.000001",
                "max_order_size": "100",
            }
        )
        self.bid, self.ask, self.held = bid, ask, held
        self.fail_order = fail_order
        self.quote_offline = quote_offline
        self.order_offline = order_offline
        self.placed: list[dict] = []
        self.cancelled: list[str] = []

    def get_best_bid_ask(self, *symbols):
        if self.quote_offline:
            raise RobinhoodConnectionError(
                "/best_bid_ask", OSError("proxy down"), outcome_known=True
            )
        if self.bid is None and self.ask is None:
            return {"results": []}
        return {
            "results": [
                {
                    "symbol": symbols[0],
                    "bid_inclusive_of_sell_spread": self.bid,
                    "ask_inclusive_of_buy_spread": self.ask,
                }
            ]
        }

    def get_trading_pairs(self, *symbols):
        if self.pair is None:
            raise RobinhoodAPIError(500, "boom", "/trading_pairs")
        return {"results": [{"symbol": symbols[0], **self.pair}]}

    def get_holdings(self, *codes):
        if self.held is None:
            raise RobinhoodAPIError(500, "boom", "/holdings")
        return {"results": [{"asset_code": codes[0], "total_quantity": self.held}]}

    def place_order(self, body):
        if self.order_offline:
            raise RobinhoodConnectionError(
                "/orders", OSError("timeout"), outcome_known=False
            )
        if self.fail_order:
            raise RobinhoodAPIError(self.fail_order, "rejected", "/orders")
        self.placed.append(body)
        return {"id": "order-1", "state": "open", **body}

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return {"id": order_id, "state": "canceled"}


def trader_for(client, rail_set=None, dry_run=False, tmp=None) -> Trader:
    tmp = tmp or Path(tempfile.mkdtemp())
    return Trader(
        client,
        rail_set or rails(),
        ledger=Ledger(tmp / "ledger.json"),
        dry_run=dry_run,
        audit_log=tmp / "orders.jsonl",
    )


def buy(quantity="0.1", **kw) -> OrderIntent:
    return OrderIntent.build("BTC-USD", "buy", quantity, **kw)


def refusal(trader, intent) -> list[str]:
    """Assert the order is refused and nothing new was sent; return reasons.

    Counts orders before and after, because some tests place valid orders
    first and only then trip a cumulative rail.
    """
    before = len(trader.client.placed)
    try:
        trader.submit(intent)
    except RailViolation as exc:
        assert (
            len(trader.client.placed) == before
        ), "an order was sent despite a violation"
        return exc.reasons
    raise AssertionError("expected RailViolation, order was allowed through")


# -- individual rails -------------------------------------------------


def test_kill_switch_blocks_everything():
    client = FakeClient()
    reasons = refusal(trader_for(client, rails(trading_enabled=False)), buy())
    assert any("TRADING_ENABLED" in r for r in reasons), reasons


def test_symbol_must_be_allow_listed():
    client = FakeClient()
    intent = OrderIntent.build("DOGE-USD", "buy", "1")
    reasons = refusal(trader_for(client), intent)
    assert any("ALLOWED_SYMBOLS" in r for r in reasons), reasons


def test_empty_allow_list_permits_nothing():
    reasons = refusal(
        trader_for(FakeClient(), rails(allowed_symbols=frozenset())), buy()
    )
    assert any("no symbol is permitted" in r for r in reasons), reasons


def test_notional_cap_blocks_oversized_order():
    # 1.0 BTC at ask 101 = $101, over the $25 cap.
    reasons = refusal(trader_for(FakeClient()), buy("1.0"))
    assert any("MAX_ORDER_USD" in r for r in reasons), reasons


def test_slippage_buffer_is_applied_to_market_orders():
    # 0.2475 * 101 = $24.9975 (under $25), but +1% buffer = $25.25 (over).
    # Without the buffer this order would be allowed.
    reasons = refusal(trader_for(FakeClient()), buy("0.2475"))
    assert any("MAX_ORDER_USD" in r for r in reasons), reasons
    approval = rails(slippage_buffer_pct=D("0")).check(
        buy("0.2475"), bid=D("100"), ask=D("101")
    )
    assert approval.estimated_notional == D("25.00")


def test_fat_finger_limit_price_is_rejected():
    # mid is 100.5; a limit at 10 is a misplaced decimal.
    intent = OrderIntent.build("BTC-USD", "buy", "0.1", "limit", limit_price="10")
    reasons = refusal(trader_for(FakeClient()), intent)
    assert any("MAX_LIMIT_DEVIATION_PCT" in r for r in reasons), reasons


def test_limit_price_near_mid_is_accepted():
    intent = OrderIntent.build("BTC-USD", "buy", "0.1", "limit", limit_price="99")
    approval = trader_for(FakeClient()).preview(intent)
    assert approval.estimated_notional == D("9.90")


def test_sell_cannot_exceed_holdings():
    client = FakeClient(held="0.05")
    intent = OrderIntent.build("BTC-USD", "sell", "0.1")
    reasons = refusal(trader_for(client), intent)
    assert any("exceeds held" in r for r in reasons), reasons


def test_sell_refused_when_holdings_unreadable():
    client = FakeClient(held=None)
    intent = OrderIntent.build("BTC-USD", "sell", "0.01")
    reasons = refusal(trader_for(client), intent)
    assert any("refusing to sell blind" in r for r in reasons), reasons


def test_no_quote_means_no_order():
    client = FakeClient(bid=None, ask=None)
    reasons = refusal(trader_for(client), buy("0.001"))
    assert any("no usable quote" in r for r in reasons), reasons


def test_zero_and_negative_quantity_rejected():
    for quantity in ("0", "-0.5"):
        reasons = refusal(trader_for(FakeClient()), buy(quantity))
        assert any("quantity must be > 0" in r for r in reasons), reasons


def test_limit_order_without_price_rejected():
    intent = OrderIntent.build("BTC-USD", "buy", "0.1", "limit")
    reasons = refusal(trader_for(FakeClient()), intent)
    assert any("requires a limit price" in r for r in reasons), reasons


def test_all_violations_reported_at_once():
    intent = OrderIntent.build("DOGE-USD", "buy", "50")
    reasons = refusal(
        trader_for(FakeClient(), rails(trading_enabled=False)), intent
    )
    joined = " | ".join(reasons)
    assert "TRADING_ENABLED" in joined
    assert "ALLOWED_SYMBOLS" in joined
    assert "MAX_ORDER_USD" in joined
    assert len(reasons) >= 3, reasons


# -- daily caps -------------------------------------------------------


def test_daily_order_count_cap():
    tmp = Path(tempfile.mkdtemp())
    trader = trader_for(FakeClient(), rails(max_orders_per_day=2), tmp=tmp)
    trader.submit(buy("0.01"))
    trader.submit(buy("0.01"))
    reasons = refusal(trader, buy("0.01"))
    assert any("MAX_ORDERS_PER_DAY" in r for r in reasons), reasons
    assert len(trader.client.placed) == 2


def test_daily_notional_cap():
    tmp = Path(tempfile.mkdtemp())
    trader = trader_for(
        FakeClient(), rails(max_daily_usd=D("30"), max_orders_per_day=99), tmp=tmp
    )
    trader.submit(buy("0.2"))  # ~$20.40
    reasons = refusal(trader, buy("0.2"))
    assert any("MAX_DAILY_USD" in r for r in reasons), reasons


def test_ledger_survives_restart():
    tmp = Path(tempfile.mkdtemp())
    rail_set = rails(max_orders_per_day=1)
    trader_for(FakeClient(), rail_set, tmp=tmp).submit(buy("0.01"))
    # A fresh Trader (as after a crash) still sees today's usage.
    reasons = refusal(trader_for(FakeClient(), rail_set, tmp=tmp), buy("0.01"))
    assert any("MAX_ORDERS_PER_DAY" in r for r in reasons), reasons


def test_rejected_order_does_not_consume_daily_budget():
    tmp = Path(tempfile.mkdtemp())
    trader = trader_for(FakeClient(fail_order=400), tmp=tmp)
    try:
        trader.submit(buy("0.01"))
    except RobinhoodAPIError:
        pass
    assert trader.ledger.load().order_count == 0


# -- dry run and submission ------------------------------------------


def test_trader_defaults_to_dry_run():
    assert Trader(FakeClient(), rails()).dry_run is True


def test_dry_run_sends_nothing_but_shows_the_body():
    client = FakeClient()
    trader = trader_for(client, dry_run=True)
    result = trader.submit(buy("0.01"))
    assert client.placed == []
    assert result["dry_run"] is True
    assert result["body_that_would_be_sent"]["symbol"] == "BTC-USD"
    assert trader.ledger.load().order_count == 0


def test_live_submit_sends_and_records():
    tmp = Path(tempfile.mkdtemp())
    client = FakeClient()
    trader = trader_for(client, tmp=tmp)
    trader.submit(buy("0.01"))
    assert len(client.placed) == 1
    body = client.placed[0]
    assert body["side"] == "buy"
    assert body["type"] == "market"
    assert body["market_order_config"] == {"asset_quantity": "0.01"}
    assert trader.ledger.load().order_count == 1
    assert (tmp / "orders.jsonl").read_text().count("\n") == 1


def test_limit_body_shape():
    client = FakeClient(held="1.0")
    intent = OrderIntent.build("ETH-USD", "sell", "0.01", "limit", limit_price="101")
    trader_for(client).submit(intent)
    config = client.placed[0]["limit_order_config"]
    assert config == {
        "asset_quantity": "0.01",
        "limit_price": "101",
        "time_in_force": "gtc",
    }


def test_client_order_id_is_stable_for_retries():
    intent = buy("0.01", client_order_id="fixed-id-123")
    client = FakeClient()
    trader = trader_for(client)
    trader.submit(intent)
    trader.submit(intent)  # a retry re-sends the same id
    ids = {body["client_order_id"] for body in client.placed}
    assert ids == {"fixed-id-123"}


def test_quantity_never_serialized_in_scientific_notation():
    client = FakeClient(bid="30000", ask="30000")
    intent = OrderIntent.build("BTC-USD", "buy", "0.0000008")
    trader_for(client).submit(intent)
    assert client.placed[0]["market_order_config"]["asset_quantity"] == "0.0000008"


def test_cancel_works_while_kill_switch_is_on():
    client = FakeClient()
    trader = trader_for(client, rails(trading_enabled=False))
    trader.cancel("order-9")
    assert client.cancelled == ["order-9"]


# -- env parsing ------------------------------------------------------


def test_rails_from_env(monkeypatch=None):
    import os

    saved = dict(os.environ)
    os.environ.update(
        {
            "TRADING_ENABLED": "TRUE",
            "ALLOWED_SYMBOLS": " btc-usd , eth-usd ",
            "MAX_ORDER_USD": "12.50",
            "MAX_DAILY_USD": "50",
            "MAX_ORDERS_PER_DAY": "3",
        }
    )
    try:
        parsed = Rails.from_env()
        assert parsed.trading_enabled is True
        assert parsed.allowed_symbols == frozenset({"BTC-USD", "ETH-USD"})
        assert parsed.max_order_usd == D("12.50")
        assert parsed.max_orders_per_day == 3
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_trading_enabled_defaults_to_off():
    import os

    saved = os.environ.pop("TRADING_ENABLED", None)
    try:
        assert Rails.from_env().trading_enabled is False
    finally:
        if saved is not None:
            os.environ["TRADING_ENABLED"] = saved


# -- transport failures ----------------------------------------------


def test_unreachable_quote_refuses_instead_of_crashing():
    # Regression: a transport failure used to escape as a traceback.
    reasons = refusal(trader_for(FakeClient(quote_offline=True)), buy("0.001"))
    assert any("no usable quote" in r for r in reasons), reasons


def test_unanswered_post_is_charged_and_flagged():
    tmp = Path(tempfile.mkdtemp())
    trader = trader_for(FakeClient(order_offline=True), tmp=tmp)
    try:
        trader.submit(buy("0.01"))
        raise AssertionError("expected RobinhoodConnectionError")
    except RobinhoodConnectionError as exc:
        assert exc.outcome_known is False
    # Charged against the day, because the order may exist.
    assert trader.ledger.load().order_count == 1
    audit = (tmp / "orders.jsonl").read_text()
    assert "unknown_may_have_been_placed" in audit
    assert "reconcile_hint" in audit


def test_get_transport_failure_is_marked_known():
    exc = RobinhoodConnectionError("/x", OSError("boom"), outcome_known=True)
    assert exc.outcome_known is True
    assert "never sent" in str(exc)


# -- USD-denominated sizing ------------------------------------------


def test_usd_sizing_respects_the_spend_ceiling():
    # $50 of BTC at ask 100000, with a 1% slippage buffer.
    client = FakeClient(bid="99990", ask="100000")
    trader = trader_for(client, rails(max_order_usd=D("60")))
    quantity = trader.quantity_for_usd("BTC-USD", "buy", D("50"))
    # The rails' own estimate must land at or under the $50 asked for,
    # buffer included — otherwise --usd 50 would be refused by the cap.
    approval = trader.preview(OrderIntent.build("BTC-USD", "buy", quantity))
    assert approval.estimated_notional <= D("50"), approval.estimated_notional
    assert approval.estimated_notional > D("49.50"), approval.estimated_notional


def test_usd_sizing_rounds_down_to_the_increment():
    client = FakeClient(bid="100", ask="100", pair={"quantity_increment": "0.01"})
    trader = trader_for(client, rails(max_order_usd=D("100")))
    quantity = trader.quantity_for_usd("BTC-USD", "buy", D("10"))
    # 10 / (100 * 1.01) = 0.09900..., rounds DOWN to 0.09, never up.
    assert quantity == D("0.09")


def test_usd_below_exchange_minimum_is_refused():
    client = FakeClient(bid="100000", ask="100000", pair={"min_order_size": "0.001"})
    trader = trader_for(client, rails())
    try:
        trader.quantity_for_usd("BTC-USD", "buy", D("5"))
        raise AssertionError("expected RailViolation")
    except RailViolation as exc:
        assert any("exchange minimum" in r for r in exc.reasons), exc.reasons


def test_usd_sizing_needs_a_quote():
    trader = trader_for(FakeClient(quote_offline=True))
    try:
        trader.quantity_for_usd("BTC-USD", "buy", D("50"))
        raise AssertionError("expected RailViolation")
    except RailViolation as exc:
        assert any("no usable quote" in r for r in exc.reasons), exc.reasons


def test_usd_must_be_positive():
    trader = trader_for(FakeClient())
    for amount in (D("0"), D("-5")):
        try:
            trader.quantity_for_usd("BTC-USD", "buy", amount)
            raise AssertionError("expected RailViolation")
        except RailViolation as exc:
            assert any("must be > 0" in r for r in exc.reasons), exc.reasons


def test_usd_sizing_still_obeys_the_per_order_cap():
    # Sizing is not a bypass: $50 against a $25 cap is still refused.
    client = FakeClient(bid="100000", ask="100000")
    trader = trader_for(client, rails(max_order_usd=D("25")))
    quantity = trader.quantity_for_usd("BTC-USD", "buy", D("50"))
    reasons = refusal(trader, OrderIntent.build("BTC-USD", "buy", quantity))
    assert any("MAX_ORDER_USD" in r for r in reasons), reasons


def test_usd_sizing_works_when_pair_metadata_unavailable():
    client = FakeClient(bid="100000", ask="100000", pair=None)
    trader = trader_for(client, rails(max_order_usd=D("60")))
    quantity = trader.quantity_for_usd("BTC-USD", "buy", D("50"))
    assert quantity > 0


# -- key handling -----------------------------------------------------


def test_private_key_is_written_without_disturbing_other_lines():
    tmp = Path(tempfile.mkdtemp()) / ".env"
    tmp.write_text("RH_API_KEY=abc\nRH_PRIVATE_KEY=\nTRADING_ENABLED=true\n")
    write_private_key(tmp, "NEWKEY==")
    text = tmp.read_text()
    assert "RH_PRIVATE_KEY=NEWKEY==" in text
    assert "RH_API_KEY=abc" in text          # untouched
    assert "TRADING_ENABLED=true" in text    # untouched


def test_existing_key_is_not_clobbered_without_force():
    tmp = Path(tempfile.mkdtemp()) / ".env"
    tmp.write_text("RH_PRIVATE_KEY=ORIGINAL==\n")
    try:
        write_private_key(tmp, "REPLACEMENT==")
        raise AssertionError("expected a refusal")
    except SystemExit as exc:
        assert "--force" in str(exc)
    assert "ORIGINAL==" in tmp.read_text(), "the old key must survive a refusal"


def test_force_replaces_the_key():
    tmp = Path(tempfile.mkdtemp()) / ".env"
    tmp.write_text("RH_PRIVATE_KEY=ORIGINAL==\n")
    write_private_key(tmp, "REPLACEMENT==", force=True)
    text = tmp.read_text()
    assert "REPLACEMENT==" in text and "ORIGINAL==" not in text


def test_key_line_is_appended_when_absent():
    tmp = Path(tempfile.mkdtemp()) / ".env"
    tmp.write_text("RH_API_KEY=abc\n")
    write_private_key(tmp, "KEY==")
    assert "RH_PRIVATE_KEY=KEY==" in tmp.read_text()


def test_blank_key_counts_as_absent():
    assert existing_private_key("RH_PRIVATE_KEY=\n") == ""
    assert existing_private_key("RH_PRIVATE_KEY=   \n") == ""
    assert existing_private_key("RH_PRIVATE_KEY=abc\n") == "abc"


def test_default_run_never_prints_the_private_key():
    import contextlib, io, re as _re, tempfile as _tf
    import generate_keys

    tmp = Path(_tf.mkdtemp()) / ".env"
    original = generate_keys.ENV_PATH
    generate_keys.ENV_PATH = tmp
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            generate_keys.main([])
        out = buf.getvalue()
        written = existing_private_key(tmp.read_text())
        assert written, "private key should have been written to .env"
        assert written not in out, "the private key leaked into stdout"
        # The public key IS printed, and is a different value.
        keys = _re.findall(r"^  ([A-Za-z0-9+/=]{40,})$", out, _re.M)
        assert len(keys) == 1, f"expected only the public key, got {len(keys)}"
        assert keys[0] != written
    finally:
        generate_keys.ENV_PATH = original


def test_print_private_mode_does_print_it():
    import contextlib, io, tempfile as _tf
    import generate_keys

    original = generate_keys.ENV_PATH
    generate_keys.ENV_PATH = Path(_tf.mkdtemp()) / ".env"
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            generate_keys.main(["--print-private"])
        out = buf.getvalue()
        assert "PRIVATE KEY" in out
        assert not generate_keys.ENV_PATH.exists(), "should not write in print mode"
    finally:
        generate_keys.ENV_PATH = original


# -- .env auditing ----------------------------------------------------


def _audit(text: str) -> list[str]:
    import check_setup

    tmp = Path(tempfile.mkdtemp()) / ".env"
    tmp.write_text(text)
    check_setup.failures.clear()
    try:
        check_setup.audit_env_file(tmp)
        return list(check_setup.failures)
    finally:
        check_setup.failures.clear()


def test_private_key_shape_is_recognised():
    import base64

    from check_setup import looks_like_a_private_key

    seed = base64.b64encode(b"x" * 32).decode()
    assert looks_like_a_private_key(seed)
    assert looks_like_a_private_key(base64.b64encode(b"x" * 64).decode())
    assert not looks_like_a_private_key("rh-api-1f3c9a20-4b7e-4c11-9a3d-2e8f71b0d4aa")
    assert not looks_like_a_private_key("")
    assert not looks_like_a_private_key("hunter2")


def test_private_key_in_the_api_key_field_is_caught():
    import base64

    seed = base64.b64encode(b"y" * 32).decode()
    found = _audit(f"RH_API_KEY={seed}\n")
    assert any("looks like a PRIVATE KEY" in f for f in found), found


def test_misspelled_setting_is_caught():
    found = _audit("RH_API_KEG=abc\n")
    assert any("not a setting this project reads" in f for f in found), found


def test_unreplaced_placeholder_is_caught():
    found = _audit("RH_API_KEY=<paste it here>\n")
    assert any("placeholder" in f for f in found), found


def test_identical_secrets_are_caught():
    import base64

    same = base64.b64encode(b"z" * 32).decode()
    found = _audit(f"RH_API_KEY={same}\nRH_PRIVATE_KEY={same}\n")
    assert any("same value" in f for f in found), found


def test_a_correct_env_file_audits_clean():
    import base64

    found = _audit(
        "RH_API_KEY=rh-api-1f3c9a20-4b7e-4c11-9a3d-2e8f71b0d4aa\n"
        f"RH_PRIVATE_KEY={base64.b64encode(b'q' * 32).decode()}\n"
        "TRADING_ENABLED=false\nALLOWED_SYMBOLS=BTC-USD\nMAX_ORDER_USD=60\n"
        "# a comment\n\n"
    )
    assert found == [], found


# -- setting the API key ----------------------------------------------


def _set_api_key(arg: str, env_text: str = "RH_API_KEY=\nRH_PRIVATE_KEY=\n"):
    """Run set_api_key against a throwaway .env; return (exit code, file text)."""
    import contextlib, io, tempfile as _tf
    import set_api_key

    tmp = Path(_tf.mkdtemp()) / ".env"
    tmp.write_text(env_text)
    original = set_api_key.ENV_PATH
    set_api_key.ENV_PATH = tmp
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = set_api_key.main([arg])
        return code, tmp.read_text(), buf.getvalue()
    finally:
        set_api_key.ENV_PATH = original


def test_api_key_is_written():
    code, text, _ = _set_api_key("rh-api-1f3c9a20-4b7e-4c11-9a3d-2e8f71b0d4aa")
    assert code == 0
    assert "RH_API_KEY=rh-api-1f3c9a20-4b7e-4c11-9a3d-2e8f71b0d4aa" in text


def test_placeholder_brackets_and_quotes_are_stripped():
    from envfile import clean

    assert clean("<rh-api-123>") == "rh-api-123"
    assert clean('"rh-api-123"') == "rh-api-123"
    assert clean("  rh-api-123  ") == "rh-api-123"
    code, text, _ = _set_api_key("<rh-api-123>")
    assert code == 0 and "RH_API_KEY=rh-api-123" in text


def test_private_key_pasted_as_api_key_is_refused():
    import base64

    seed = base64.b64encode(b"k" * 32).decode()
    code, text, out = _set_api_key(seed)
    assert code == 1
    assert "PRIVATE KEY" in out
    assert seed not in text, "a refused value must not be written"


def test_api_key_matching_the_private_key_is_refused():
    import base64

    seed = base64.b64encode(b"m" * 32).decode()
    code, _, out = _set_api_key(seed, f"RH_API_KEY=\nRH_PRIVATE_KEY={seed}\n")
    assert code == 1 and "REFUSED" in out


def test_other_settings_survive():
    code, text, _ = _set_api_key(
        "rh-api-9",
        "RH_API_KEY=\nRH_PRIVATE_KEY=abc\nTRADING_ENABLED=true\nMAX_ORDER_USD=60\n",
    )
    assert code == 0
    assert "RH_PRIVATE_KEY=abc" in text
    assert "TRADING_ENABLED=true" in text
    assert "MAX_ORDER_USD=60" in text


def test_empty_input_changes_nothing():
    code, text, _ = _set_api_key("   ")
    assert code == 1 and "RH_API_KEY=\n" in text


# -- the kill switch command -----------------------------------------


def _arm(args, env_text="TRADING_ENABLED=false\nMAX_ORDER_USD=60\nRH_API_KEY=k\n"):
    import contextlib, io, tempfile as _tf
    import arm as arm_mod

    tmp = Path(_tf.mkdtemp()) / ".env"
    tmp.write_text(env_text)
    original = arm_mod.ENV_PATH
    arm_mod.ENV_PATH = tmp
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = arm_mod.main(args)
        return code, tmp.read_text(), buf.getvalue()
    finally:
        arm_mod.ENV_PATH = original


def test_arm_sets_trading_enabled_true():
    code, text, _ = _arm(["--yes"])
    assert code == 0
    assert "TRADING_ENABLED=true" in text


def test_disarm_sets_false_without_confirmation():
    code, text, out = _arm(["--off"], "TRADING_ENABLED=true\nMAX_ORDER_USD=60\n")
    assert code == 0
    assert "TRADING_ENABLED=false" in text
    assert "DISARMED" in out


def test_arming_preserves_other_settings():
    code, text, _ = _arm(
        ["--yes"],
        "TRADING_ENABLED=false\nRH_API_KEY=rh-api-1\nRH_PRIVATE_KEY=abc\n"
        "MAX_ORDER_USD=60\nALLOWED_SYMBOLS=BTC-USD\n",
    )
    assert code == 0
    assert "RH_API_KEY=rh-api-1" in text
    assert "RH_PRIVATE_KEY=abc" in text
    assert "MAX_ORDER_USD=60" in text
    assert "ALLOWED_SYMBOLS=BTC-USD" in text


def test_status_changes_nothing():
    code, text, out = _arm(["--status"])
    assert code == 0
    assert "TRADING_ENABLED=false" in text
    assert "DISARMED" in out


def test_disarm_is_idempotent():
    code, text, _ = _arm(["--off"], "TRADING_ENABLED=false\n")
    assert code == 0 and "TRADING_ENABLED=false" in text


def test_arming_when_already_armed_is_a_noop():
    code, text, out = _arm(["--yes"], "TRADING_ENABLED=true\n")
    assert code == 0 and "Already armed" in out
    assert "TRADING_ENABLED=true" in text


# -- signed request paths ---------------------------------------------


def test_no_bare_question_mark_when_there_are_no_parameters():
    """Regression: a trailing "?" broke signature verification.

    Robinhood drops an empty query string before checking the signature,
    so ".../holdings/?" signs a message the server never reconstructs and
    the request comes back 401 "Signature is invalid".
    """
    from robinhood_client import RobinhoodCryptoClient as C

    path = C.with_query("/api/v1/crypto/trading/holdings/", "asset_code", ())
    assert path == "/api/v1/crypto/trading/holdings/"
    assert not path.endswith("?")


def test_query_is_built_for_one_and_many_values():
    from robinhood_client import RobinhoodCryptoClient as C

    assert (
        C.with_query("/x/", "asset_code", ("BTC",)) == "/x/?asset_code=BTC"
    )
    assert (
        C.with_query("/x/", "symbol", ("BTC-USD", "ETH-USD"))
        == "/x/?symbol=BTC-USD&symbol=ETH-USD"
    )


def test_every_read_helper_avoids_the_empty_query():
    """No helper may emit a trailing "?" when called with no arguments."""
    import base64

    from robinhood_client import RobinhoodCryptoClient

    client = RobinhoodCryptoClient("k", base64.b64encode(b"s" * 32).decode())
    seen = []
    client.get = lambda path: seen.append(path) or {}
    client.get_holdings()
    client.get_trading_pairs()
    client.get_best_bid_ask()
    assert seen, "no paths captured"
    for path in seen:
        assert not path.endswith("?"), path


def test_signature_covers_the_path_without_a_stray_marker():
    """The signed message and the requested path must be the same string."""
    import base64

    from robinhood_client import RobinhoodCryptoClient

    client = RobinhoodCryptoClient("api-key", base64.b64encode(b"s" * 32).decode())
    path = client.with_query("/api/v1/crypto/trading/holdings/", "asset_code", ())
    headers = client._headers("GET", path)
    expected = f"api-key{headers['x-timestamp']}{path}GET"
    from nacl.signing import SigningKey

    verify = SigningKey(b"s" * 32).verify_key
    verify.verify(expected.encode(), base64.b64decode(headers["x-signature"]))


# -- order display ----------------------------------------------------


def test_order_summary_never_shows_scientific_notation():
    """Regression: an unfilled quantity rendered as 0E-18."""
    import orders

    line = orders.summarise(
        {
            "id": "6aa5d5f2-ce70",
            "created_at": "2026-09-12T18:45:06.980244-04:00",
            "side": "buy",
            "symbol": "BTC-USD",
            "state": "open",
            "filled_asset_quantity": "0.000000000000000000",
            "market_order_config": {"asset_quantity": "0.000025410000000000"},
        }
    )
    assert "E-" not in line and "E+" not in line, line
    assert "0.00002541" in line


def test_order_summary_reports_the_actual_spend():
    import orders

    line = orders.summarise(
        {
            "id": "abc12345",
            "created_at": "2026-09-12T18:45:06",
            "side": "buy",
            "symbol": "BTC-USD",
            "state": "filled",
            "filled_asset_quantity": "0.00002541",
            "average_price": "77935.70",
            "market_order_config": {"asset_quantity": "0.00002541"},
        }
    )
    assert "$1.98" in line, line
    assert "@ 77935.70 " in line, line  # not 77935.700000000000000000


def test_final_states_cover_the_ways_an_order_ends():
    import orders

    for state in ("filled", "canceled", "cancelled", "rejected", "failed"):
        assert state in orders.FINAL_STATES
    assert "open" not in orders.FINAL_STATES


def test_padded_price_is_trimmed_for_display():
    """Robinhood pads prices with a dozen trailing zeros."""
    import orders

    line = orders.summarise(
        {
            "id": "abc",
            "created_at": "2026-09-12T18:45:06",
            "side": "buy",
            "symbol": "BTC-USD",
            "state": "filled",
            "filled_asset_quantity": "0.00002541",
            "average_price": "77920.768969770000000000",
            "market_order_config": {"asset_quantity": "0.00002541"},
        }
    )
    assert "77920.77" in line, line
    assert "0000000000" not in line, line


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, func in tests:
        try:
            func()
            print(f"  ok    {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
