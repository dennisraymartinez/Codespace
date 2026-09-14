"""What you hold, what it cost, and what it is worth now.

  python positions.py          # holdings with cost basis and P&L
  python positions.py --json   # the same numbers, machine-readable

Read-only: issues GET requests only. It never places, cancels, or
modifies an order.

Cost basis is average-cost, walked forward over your filled orders
oldest-first: a buy adds quantity and cost, a sell removes quantity and
the same proportion of the cost. That is the only method reconstructable
from order history alone, and it is what a broker normally reports.

Positions are valued at the BID — the side that would actually fill if
you sold — so the figure shown is what you could realise right now, not
a mid price nobody trades at. Buying at the ask and valuing at the bid
is exactly why a position shows red the moment it fills: that gap is the
spread you have already paid, not the market moving against you.

The basis is only as complete as the order history Robinhood returns. If
the computed quantity disagrees with the holding the exchange reports,
this says so rather than printing a confident wrong number.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from dotenv import load_dotenv

from robinhood_client import (
    RobinhoodConnectionError,
    RobinhoodCryptoClient,
    RobinhoodError,
)
from safety import plain

ENV_PATH = Path(__file__).with_name(".env")

# Quantities are exact decimals on both sides, so a reconstructed basis
# should match the reported holding almost exactly. Anything beyond this
# means the history is missing orders (or the asset was transferred in),
# and the basis cannot be trusted.
RECONCILE_TOLERANCE = Decimal("0.001")  # 0.1%

# Dollar figures are rounded to the cent at the source, not just when
# printed, so the columns agree with each other. Deriving P&L from
# full-precision cost and value instead would show $38.72 - $39.66 =
# -$0.95, and a table whose own arithmetic looks wrong is worse than one
# that is a half-cent less precise.
CENT = Decimal("0.01")

FILLED = "filled"


def to_decimal(value: object, default: Decimal = Decimal(0)) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return default


def money(value: Decimal) -> str:
    """Dollars, always two places, with the sign outside the symbol.

    Not plain() — that would turn 48.90 into 48.9, and money keeps both
    decimals. A loss reads -$0.94, not $-0.94.
    """
    amount = value.quantize(CENT)
    return f"-${abs(amount)}" if amount < 0 else f"${amount}"


def price(value: Decimal) -> str:
    """A price to the cent, or to 8 places when the asset trades below $1.

    Rounding ONDO at 0.34758809 to 0.35 would throw away the price.
    """
    if value >= 1:
        return str(value.quantize(Decimal("0.01")))
    return plain(value.quantize(Decimal("0.00000001")))


@dataclass
class Basis:
    """Running average-cost basis for one asset."""

    quantity: Decimal = Decimal(0)
    cost: Decimal = Decimal(0)
    buys: int = 0
    sells: int = 0
    # True when a sell exceeded the quantity we had seen bought, which
    # means the visible history does not go back far enough.
    incomplete: bool = False

    @property
    def average(self) -> Decimal | None:
        if self.quantity <= 0:
            return None
        return self.cost / self.quantity


@dataclass
class Position:
    asset: str
    symbol: str
    quantity: Decimal
    basis: Basis
    bid: Decimal | None = None
    ask: Decimal | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def value(self) -> Decimal | None:
        """What selling at the current bid would realise, to the cent."""
        if self.bid is None:
            return None
        return (self.quantity * self.bid).quantize(CENT)

    @property
    def cost(self) -> Decimal | None:
        """Cost of the quantity actually held, at the average basis.

        Scaled to the reported holding rather than the reconstructed one,
        so a small divergence still yields a sensible figure.
        """
        average = self.basis.average
        if average is None:
            return None
        return (self.quantity * average).quantize(CENT)

    @property
    def pnl(self) -> Decimal | None:
        value, cost = self.value, self.cost
        if value is None or cost is None:
            return None
        return value - cost

    @property
    def pnl_pct(self) -> Decimal | None:
        pnl, cost = self.pnl, self.cost
        if pnl is None or cost is None or cost <= 0:
            return None
        return pnl / cost * Decimal(100)

    @property
    def spread_pct(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        mid = (self.bid + self.ask) / Decimal(2)
        if mid <= 0:
            return None
        return (self.ask - self.bid) / mid * Decimal(100)


def build_basis(orders: list[dict]) -> dict[str, Basis]:
    """Average-cost basis per asset, from filled orders oldest-first.

    Unfilled, cancelled and rejected orders are ignored: they moved no
    quantity and cost nothing.
    """
    by_asset: dict[str, Basis] = {}

    def when(order: dict) -> str:
        return str(order.get("created_at") or "")

    for order in sorted(orders, key=when):
        if str(order.get("state", "")).lower() != FILLED:
            continue
        symbol = str(order.get("symbol") or "")
        asset = symbol.split("-")[0]
        if not asset:
            continue
        quantity = to_decimal(order.get("filled_asset_quantity"))
        unit_price = to_decimal(order.get("average_price"))
        if quantity <= 0 or unit_price <= 0:
            continue

        basis = by_asset.setdefault(asset, Basis())
        side = str(order.get("side", "")).lower()

        if side == "buy":
            basis.quantity += quantity
            basis.cost += quantity * unit_price
            basis.buys += 1
        elif side == "sell":
            basis.sells += 1
            if basis.quantity <= 0:
                # Sold something we never saw bought: history is truncated.
                basis.incomplete = True
                continue
            sold = min(quantity, basis.quantity)
            if sold < quantity:
                basis.incomplete = True
            # Remove the same proportion of cost as of quantity, which is
            # what keeps the average unchanged across a partial sell.
            unit_cost = basis.cost / basis.quantity
            basis.cost -= sold * unit_cost
            basis.quantity -= sold
            if basis.quantity <= 0:
                basis.quantity = Decimal(0)
                basis.cost = Decimal(0)

    return by_asset


def reconcile(position: Position) -> None:
    """Flag a basis that disagrees with the exchange's own holding."""
    basis = position.basis
    if basis.buys == 0 and basis.sells == 0:
        position.notes.append(
            "no order history for this asset — cost basis unknown "
            "(transferred in, or older than the orders Robinhood returns)"
        )
        return
    if basis.incomplete:
        position.notes.append(
            "order history does not reach back far enough — cost basis is a "
            "lower bound, not the real one"
        )
        return
    if position.quantity <= 0:
        return
    drift = abs(basis.quantity - position.quantity) / position.quantity
    if drift > RECONCILE_TOLERANCE:
        position.notes.append(
            f"history implies {plain(basis.quantity)} held but Robinhood "
            f"reports {plain(position.quantity)} — cost basis may be wrong"
        )


def collect(client: RobinhoodCryptoClient) -> tuple[list[Position], list[str]]:
    warnings: list[str] = []

    holdings = client.get_holdings().get("results", [])
    held = []
    for holding in holdings:
        quantity = to_decimal(holding.get("total_quantity"))
        asset = str(holding.get("asset_code") or "")
        if asset and quantity > 0:
            held.append((asset, quantity))

    if not held:
        return [], warnings

    positions = [
        Position(asset=asset, symbol=f"{asset}-USD", quantity=quantity, basis=Basis())
        for asset, quantity in sorted(held)
    ]

    payload = client.get_orders()
    orders = payload.get("results", [])
    if payload.get("next"):
        warnings.append(
            "Robinhood returned only the first page of orders — a cost basis "
            "that depends on older fills may be incomplete."
        )
    basis_by_asset = build_basis(orders)

    quotes: dict[str, tuple[Decimal | None, Decimal | None]] = {}
    try:
        quoted = client.get_best_bid_ask(*[p.symbol for p in positions])
        for result in quoted.get("results", []):
            symbol = str(result.get("symbol") or "")
            quotes[symbol] = (
                to_decimal(result.get("bid_inclusive_of_sell_spread"), Decimal(0))
                or None,
                to_decimal(result.get("ask_inclusive_of_buy_spread"), Decimal(0))
                or None,
            )
    except RobinhoodError as exc:
        warnings.append(f"quotes unavailable ({exc}) — positions cannot be valued")

    for position in positions:
        position.basis = basis_by_asset.get(position.asset, Basis())
        position.bid, position.ask = quotes.get(position.symbol, (None, None))
        if position.bid is None:
            position.notes.append("no quote — cannot value this holding")
        reconcile(position)

    return positions, warnings


def render(positions: list[Position], warnings: list[str]) -> None:
    print()
    print("=" * 78)
    print("  Positions — valued at the bid, what selling now would realise")
    print("=" * 78)
    print()

    header = (
        f"  {'ASSET':<6} {'QUANTITY':>16} {'AVG COST':>13} {'BID':>13} "
        f"{'COST':>10} {'VALUE':>10} {'P&L':>10} {'P&L%':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_cost = Decimal(0)
    total_value = Decimal(0)
    valued = 0

    for position in positions:
        average = position.basis.average
        cost, value, pnl, pct = (
            position.cost,
            position.value,
            position.pnl,
            position.pnl_pct,
        )
        line = (
            f"  {position.asset:<6} {plain(position.quantity):>16} "
            f"{(price(average) if average else '—'):>13} "
            f"{(price(position.bid) if position.bid else '—'):>13} "
            f"{(money(cost) if cost is not None else '—'):>10} "
            f"{(money(value) if value is not None else '—'):>10} "
            f"{(money(pnl) if pnl is not None else '—'):>10} "
            f"{(f'{pct:.2f}%' if pct is not None else '—'):>8}"
        )
        print(line)
        if cost is not None and value is not None:
            total_cost += cost
            total_value += value
            valued += 1

    print("  " + "-" * (len(header) - 2))

    if valued:
        total_pnl = total_value - total_cost
        total_pct = (
            total_pnl / total_cost * Decimal(100) if total_cost > 0 else Decimal(0)
        )
        print(
            f"  {'TOTAL':<6} {'':>16} {'':>13} {'':>13} "
            f"{money(total_cost):>10} {money(total_value):>10} "
            f"{money(total_pnl):>10} {f'{total_pct:.2f}%':>8}"
        )
        if valued < len(positions):
            print(f"  (totals cover {valued} of {len(positions)} holdings)")

    spreads = [
        f"{p.asset} {p.spread_pct:.2f}%"
        for p in positions
        if p.spread_pct is not None
    ]
    if spreads:
        print()
        print("  Cost to exit (round-trip spread): " + ", ".join(spreads))
        print("  A position has to gain more than that before it is genuinely up.")

    footnotes = [(p.asset, note) for p in positions for note in p.notes]
    if footnotes or warnings:
        print()
        for asset, note in footnotes:
            print(f"  note  {asset}: {note}")
        for warning in warnings:
            print(f"  note  {warning}")

    print()


def as_json(positions: list[Position], warnings: list[str]) -> str:
    def number(value: Decimal | None) -> str | None:
        return plain(value) if value is not None else None

    payload = {
        "positions": [
            {
                "asset": p.asset,
                "symbol": p.symbol,
                "quantity": plain(p.quantity),
                "average_cost": number(p.basis.average),
                "bid": number(p.bid),
                "ask": number(p.ask),
                "cost": number(p.cost),
                "value": number(p.value),
                "pnl": number(p.pnl),
                "pnl_pct": number(p.pnl_pct),
                "spread_pct": number(p.spread_pct),
                "notes": p.notes,
            }
            for p in positions
        ],
        "warnings": warnings,
    }
    return json.dumps(payload, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    load_dotenv(ENV_PATH)
    try:
        client = RobinhoodCryptoClient(
            os.getenv("RH_API_KEY", "").strip(),
            os.getenv("RH_PRIVATE_KEY", "").strip(),
        )
    except ValueError as exc:
        print(f"credentials: {exc}", file=sys.stderr)
        print("run: python check_setup.py", file=sys.stderr)
        return 1

    try:
        positions, warnings = collect(client)
    except RobinhoodConnectionError as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        return 3
    except RobinhoodError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    if not positions:
        print("no crypto holdings")
        return 0

    if args.json:
        print(as_json(positions, warnings))
    else:
        render(positions, warnings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
