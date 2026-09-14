"""Tests for the positions view and its cost-basis reconstruction.

No network: the client is faked, so these never touch Robinhood.

    python test_positions.py          # standalone
    pytest test_positions.py          # if you have pytest
"""

from __future__ import annotations

import contextlib
import io
import json
from decimal import Decimal

import positions as positions_mod
from positions import Basis, Position, build_basis, collect, reconcile

D = Decimal


def order(symbol, side, quantity, price, created_at, state="filled"):
    return {
        "symbol": symbol,
        "side": side,
        "state": state,
        "filled_asset_quantity": str(quantity),
        "average_price": str(price),
        "created_at": created_at,
    }


class FakeClient:
    """Stands in for RobinhoodCryptoClient. Read-only, records nothing."""

    def __init__(self, holdings=None, orders=None, quotes=None, next_page=None):
        self.holdings = holdings or []
        self.orders = orders or []
        self.quotes = quotes or {}
        self.next_page = next_page

    def get_holdings(self, *symbols):
        return {
            "results": [
                {"asset_code": code, "total_quantity": str(quantity)}
                for code, quantity in self.holdings
            ]
        }

    def get_orders(self):
        payload = {"results": list(self.orders)}
        if self.next_page:
            payload["next"] = self.next_page
        return payload

    def get_best_bid_ask(self, *symbols):
        return {
            "results": [
                {
                    "symbol": symbol,
                    "bid_inclusive_of_sell_spread": str(bid),
                    "ask_inclusive_of_buy_spread": str(ask),
                }
                for symbol, (bid, ask) in self.quotes.items()
                if symbol in symbols
            ]
        }


# -- cost basis -------------------------------------------------------


def test_single_buy_sets_average_to_its_price():
    basis = build_basis([order("LINK-USD", "buy", "3.4164", "11.61", "2026-09-12")])
    assert basis["LINK"].quantity == D("3.4164")
    assert basis["LINK"].average == D("11.61")


def test_two_buys_weight_the_average_by_quantity():
    basis = build_basis(
        [
            order("BTC-USD", "buy", "1", "100", "2021-01-01"),
            order("BTC-USD", "buy", "3", "200", "2021-01-02"),
        ]
    )
    # (1*100 + 3*200) / 4 = 175
    assert basis["BTC"].average == D("175")
    assert basis["BTC"].cost == D("700")


def test_partial_sell_leaves_the_average_unchanged():
    basis = build_basis(
        [
            order("BTC-USD", "buy", "4", "100", "2021-01-01"),
            order("BTC-USD", "sell", "1", "500", "2021-01-02"),
        ]
    )
    assert basis["BTC"].quantity == D("3")
    assert basis["BTC"].average == D("100"), "selling must not change the basis"
    assert basis["BTC"].cost == D("300")
    assert not basis["BTC"].incomplete


def test_selling_everything_resets_the_basis():
    basis = build_basis(
        [
            order("BTC-USD", "buy", "2", "100", "2021-01-01"),
            order("BTC-USD", "sell", "2", "150", "2021-01-02"),
        ]
    )
    assert basis["BTC"].quantity == D("0")
    assert basis["BTC"].cost == D("0")
    assert basis["BTC"].average is None


def test_rebuying_after_a_full_exit_uses_only_the_new_price():
    basis = build_basis(
        [
            order("BTC-USD", "buy", "2", "100", "2021-01-01"),
            order("BTC-USD", "sell", "2", "150", "2021-01-02"),
            order("BTC-USD", "buy", "1", "900", "2021-01-03"),
        ]
    )
    assert basis["BTC"].average == D("900"), "the old lot is gone, not averaged in"


def test_selling_more_than_was_bought_flags_incomplete_history():
    basis = build_basis(
        [
            order("BTC-USD", "buy", "1", "100", "2021-01-02"),
            order("BTC-USD", "sell", "5", "150", "2021-01-03"),
        ]
    )
    assert basis["BTC"].incomplete, "history cannot reach back far enough"


def test_unfilled_orders_are_ignored():
    basis = build_basis(
        [
            order("BTC-USD", "buy", "1", "100", "2021-01-01"),
            order("BTC-USD", "buy", "99", "1", "2021-01-02", state="cancelled"),
            order("BTC-USD", "buy", "99", "1", "2021-01-03", state="rejected"),
        ]
    )
    assert basis["BTC"].quantity == D("1")
    assert basis["BTC"].average == D("100")


def test_orders_are_walked_oldest_first_regardless_of_input_order():
    newest_first = [
        order("BTC-USD", "sell", "1", "500", "2021-01-02"),
        order("BTC-USD", "buy", "2", "100", "2021-01-01"),
    ]
    basis = build_basis(newest_first)
    assert basis["BTC"].quantity == D("1")
    assert not basis["BTC"].incomplete, "a sell must not be applied before its buy"


def test_assets_are_tracked_independently():
    basis = build_basis(
        [
            order("BTC-USD", "buy", "1", "100", "2021-01-01"),
            order("DOGE-USD", "buy", "10", "2", "2021-01-02"),
        ]
    )
    assert basis["BTC"].average == D("100")
    assert basis["DOGE"].average == D("2")


# -- reconciliation ---------------------------------------------------


def test_matching_quantity_produces_no_note():
    position = Position("BTC", "BTC-USD", D("1"), Basis(quantity=D("1"), cost=D("100"), buys=1))
    reconcile(position)
    assert position.notes == []


def test_divergent_quantity_is_flagged():
    position = Position("BTC", "BTC-USD", D("2"), Basis(quantity=D("1"), cost=D("100"), buys=1))
    reconcile(position)
    assert any("may be wrong" in note for note in position.notes)


def test_holding_with_no_history_is_flagged():
    position = Position("BTC", "BTC-USD", D("1"), Basis())
    reconcile(position)
    assert any("cost basis unknown" in note for note in position.notes)


def test_incomplete_history_is_flagged_as_a_lower_bound():
    position = Position(
        "BTC", "BTC-USD", D("1"), Basis(quantity=D("1"), cost=D("100"), buys=1, incomplete=True)
    )
    reconcile(position)
    assert any("lower bound" in note for note in position.notes)


# -- valuation --------------------------------------------------------


def test_value_uses_the_bid_not_the_ask():
    position = Position(
        "BTC", "BTC-USD", D("2"), Basis(quantity=D("2"), cost=D("100"), buys=1),
        bid=D("40"), ask=D("60"),
    )
    assert position.value == D("80"), "a sale fills at the bid"
    assert position.cost == D("100")
    assert position.pnl == D("-20")


def test_money_columns_are_internally_consistent():
    """COST, VALUE and P&L must agree — value - cost has to equal P&L."""
    position = Position(
        "LINK", "LINK-USD", D("3.4164"),
        Basis(quantity=D("3.4164"), cost=D("3.4164") * D("11.61"), buys=1),
        bid=D("11.33232254"), ask=D("11.54954"),
    )
    assert position.value - position.cost == position.pnl


def test_spread_is_measured_against_the_mid():
    position = Position(
        "BTC", "BTC-USD", D("1"), Basis(), bid=D("99"), ask=D("101")
    )
    assert position.spread_pct == D("2")


def test_missing_quote_leaves_the_position_unvalued():
    position = Position("BTC", "BTC-USD", D("1"), Basis(quantity=D("1"), cost=D("100")))
    assert position.value is None
    assert position.pnl is None
    assert position.spread_pct is None


# -- the real account -------------------------------------------------
#
# The actual order history and quotes from a live account, so the whole
# pipeline is checked against numbers verified by hand rather than only
# against synthetic ones. Note the BTC run: repeated buys and sells take
# the position to exactly zero in 2026-08, so the single later buy is the
# entire basis.

REAL_ORDERS = [
    order("BTC-USD", "buy", "0.01022467", "48904.56", "2021-02-23T20:12:01"),
    order("BTC-USD", "buy", "0.00269378", "55682.11", "2021-03-23T14:16:21"),
    order("BTC-USD", "buy", "0.00437541", "57143.02", "2021-04-03T22:13:02"),
    order("BTC-USD", "buy", "0.00172922", "57829.54", "2021-04-06T23:00:37"),
    order("BTC-USD", "buy", "0.00175911", "56849.13", "2021-04-07T10:45:54"),
    order("BTC-USD", "buy", "0.00913092", "54765.29", "2021-04-19T20:34:20"),
    order("BTC-USD", "buy", "0.00096523", "52028.71", "2021-04-22T18:03:21"),
    order("DOGE-USD", "buy", "529", "0.3777761", "2021-05-02T10:38:39"),
    order("DOGE-USD", "buy", "405", "0.492166", "2021-05-09T11:19:41"),
    order("BTC-USD", "buy", "0.00271762", "47873.29", "2021-05-13T13:40:48"),
    order("BTC-USD", "buy", "0.00418743", "47761.20", "2021-05-13T13:42:01"),
    order("BTC-USD", "buy", "0.00281965", "35456.65", "2021-05-19T10:21:29"),
    order("DOGE-USD", "buy", "856.4", "0.3503", "2021-05-19T10:23:18"),
    order("BTC-USD", "sell", "0.03934745", "50828.89", "2021-09-05T16:42:52"),
    order("BTC-USD", "sell", "0.00107603", "46461.01", "2021-09-08T01:16:12"),
    order("BTC-USD", "buy", "0.02138672", "46759.74", "2021-09-08T18:29:40"),
    order("BTC-USD", "sell", "0.02083166", "48005.59", "2021-10-02T16:53:35"),
    order("BTC-USD", "sell", "0.00073443", "59011.08", "2021-11-18T09:35:30"),
    order("DOGE-USD", "sell", "1790.4", "0.2264361", "2021-11-18T13:56:48"),
    order("BTC-USD", "buy", "0.00172099", "58106.43", "2024-09-03T18:59:44"),
    order("BTC-USD", "sell", "0.00172118", "78198.87", "2026-08-26T21:50:15"),
    order("BTC-USD", "buy", "0.00002541", "77920.77", "2026-09-12T18:45:06"),
    order("ONDO-USD", "buy", "140.94", "0.35125529", "2026-09-12T19:08:32"),
    order("LINK-USD", "buy", "3.4164", "11.61", "2026-09-12T21:19:58"),
]

REAL_HOLDINGS = [
    ("LINK", "3.416400000000000000"),
    ("ONDO", "140.940000000000000000"),
    ("BTC", "0.000025410000000000"),
]

REAL_QUOTES = {
    "BTC-USD": ("76579.04389311", "78041.14"),
    "LINK-USD": ("11.33232254", "11.54954"),
    "ONDO-USD": ("0.34758809", "0.35427783"),
}


def real_client():
    return FakeClient(holdings=REAL_HOLDINGS, orders=REAL_ORDERS, quotes=REAL_QUOTES)


def test_real_account_basis_reconciles_against_reported_holdings():
    basis = build_basis(REAL_ORDERS)
    assert basis["BTC"].quantity == D("0.00002541")
    assert basis["LINK"].quantity == D("3.4164")
    assert basis["ONDO"].quantity == D("140.94")
    assert basis["DOGE"].quantity == D("0"), "DOGE was fully sold in 2021"


def test_real_account_btc_basis_is_only_the_final_buy():
    basis = build_basis(REAL_ORDERS)
    assert basis["BTC"].average == D("77920.77"), (
        "the 2026-08 sell zeroed the position, so six years of earlier "
        "lots must not be averaged into the basis"
    )


def test_real_account_positions_have_no_reconciliation_warnings():
    client = real_client()
    found, warnings = collect(client)
    assert warnings == []
    for position in found:
        assert position.notes == [], f"{position.asset}: {position.notes}"


def test_real_account_pnl_matches_hand_calculation():
    client = real_client()
    found, _ = collect(client)
    by_asset = {p.asset: p for p in found}

    assert by_asset["LINK"].cost == D("39.66")
    assert by_asset["LINK"].value == D("38.72")
    assert by_asset["LINK"].pnl == D("-0.94")

    assert by_asset["ONDO"].cost == D("49.51")
    assert by_asset["ONDO"].value == D("48.99")
    assert by_asset["ONDO"].pnl == D("-0.52")

    assert by_asset["BTC"].cost == D("1.98")
    assert by_asset["BTC"].value == D("1.95")
    assert by_asset["BTC"].pnl == D("-0.03")

    total_cost = sum(p.cost for p in found)
    total_value = sum(p.value for p in found)
    assert total_cost == D("91.15")
    assert total_value == D("89.66")
    assert total_value - total_cost == D("-1.49")


def test_real_account_spreads_are_about_one_and_nine_tenths_percent():
    client = real_client()
    found, _ = collect(client)
    for position in found:
        assert D("1.8") < position.spread_pct < D("2.0"), (
            f"{position.asset} spread {position.spread_pct}"
        )


def test_fully_sold_assets_do_not_appear():
    client = real_client()
    found, _ = collect(client)
    assert "DOGE" not in {p.asset for p in found}


# -- output -----------------------------------------------------------


def run_main(client, argv):
    original = positions_mod.RobinhoodCryptoClient
    positions_mod.RobinhoodCryptoClient = lambda *a, **k: client
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = positions_mod.main(argv)
        return code, buf.getvalue()
    finally:
        positions_mod.RobinhoodCryptoClient = original


def test_render_shows_every_holding_and_a_consistent_total():
    code, out = run_main(real_client(), [])
    assert code == 0
    for asset in ("LINK", "ONDO", "BTC"):
        assert asset in out
    assert "$91.15" in out and "$89.66" in out
    assert "-$1.49" in out, "a loss reads -$1.49, not $-1.49"
    assert "TOTAL" in out
    assert "Cost to exit" in out


def test_json_output_is_valid_and_carries_the_numbers():
    code, out = run_main(real_client(), ["--json"])
    assert code == 0
    payload = json.loads(out)
    by_asset = {p["asset"]: p for p in payload["positions"]}
    assert by_asset["LINK"]["cost"] == "39.66"
    assert by_asset["LINK"]["value"] == "38.72"
    assert by_asset["ONDO"]["average_cost"] == "0.35125529"
    assert payload["warnings"] == []


def test_empty_account_says_so_and_succeeds():
    code, out = run_main(FakeClient(), [])
    assert code == 0
    assert "no crypto holdings" in out


def test_truncated_order_history_is_warned_about():
    client = real_client()
    client.next_page = "https://trading.robinhood.com/...?cursor=abc"
    _, warnings = collect(client)
    assert any("first page" in warning for warning in warnings)


def test_holding_without_a_quote_is_reported_not_valued():
    client = FakeClient(
        holdings=[("BTC", "1")],
        orders=[order("BTC-USD", "buy", "1", "100", "2021-01-01")],
        quotes={},
    )
    found, _ = collect(client)
    assert found[0].value is None
    assert any("no quote" in note for note in found[0].notes)
    code, out = run_main(client, [])
    assert code == 0, "an unquotable holding must not crash the view"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, func in tests:
        try:
            func()
            print(f"  ok    {name}")
        except BaseException as exc:  # SystemExit would otherwise end the run
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
