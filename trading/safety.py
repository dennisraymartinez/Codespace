"""Safety rails for order placement.

Every rail is checked before an order is built, and all violations are
collected so one run tells you everything that is wrong rather than making
you fix them one at a time.

The rails, and what each one is actually protecting you from:

  TRADING_ENABLED         master kill switch. Off => nothing is ever sent.
  ALLOWED_SYMBOLS         a typo'd or unexpected symbol can't be traded.
  MAX_ORDER_USD           one order can't be larger than you intended.
  MAX_LIMIT_DEVIATION_PCT fat-finger guard: a limit price far from the
                          market (a misplaced decimal) is rejected instead
                          of resting on the book as a gift.
  MAX_DAILY_USD           a bug that loops can't drain the account, because
                          the day's cumulative notional is capped.
  MAX_ORDERS_PER_DAY      same, by order count — catches a fast loop even
                          when each order is individually tiny.
  holdings check          a sell can't exceed the quantity you actually
                          hold, so no accidental short/reject cycle.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

SIDES = ("buy", "sell")
ORDER_TYPES = ("market", "limit")

STATE_DIR = Path(__file__).with_name("state")
LEDGER_PATH = STATE_DIR / "ledger.json"


class RailViolation(Exception):
    """Raised when an order intent fails one or more safety rails.

    Carries every reason, not just the first.
    """

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


def _decimal(value: Any, name: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{name} is not a number: {value!r}") from exc


def plain(value: Decimal) -> str:
    """Format a Decimal for the API — never scientific notation."""
    return format(value.normalize(), "f")


# -- the intent -------------------------------------------------------


@dataclass(frozen=True)
class OrderIntent:
    """What you want to do, before anything has been validated or sent."""

    symbol: str
    side: str
    quantity: Decimal
    order_type: str = "market"
    limit_price: Decimal | None = None
    time_in_force: str = "gtc"
    # Idempotency key. Robinhood dedupes on this, so a retry after a
    # timeout re-sends the SAME id and cannot double-fill.
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def build(
        cls,
        symbol: str,
        side: str,
        quantity: Any,
        order_type: str = "market",
        limit_price: Any = None,
        time_in_force: str = "gtc",
        client_order_id: str | None = None,
    ) -> "OrderIntent":
        return cls(
            symbol=symbol.strip().upper(),
            side=side.strip().lower(),
            quantity=_decimal(quantity, "quantity"),
            order_type=order_type.strip().lower(),
            limit_price=(
                None if limit_price in (None, "") else _decimal(limit_price, "limit_price")
            ),
            time_in_force=time_in_force,
            client_order_id=client_order_id or str(uuid.uuid4()),
        )

    def to_api_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "type": self.order_type,
        }
        if self.order_type == "market":
            body["market_order_config"] = {"asset_quantity": plain(self.quantity)}
        else:
            if self.limit_price is None:
                raise ValueError("limit order requires a limit_price")
            body["limit_order_config"] = {
                "asset_quantity": plain(self.quantity),
                "limit_price": plain(self.limit_price),
                "time_in_force": self.time_in_force,
            }
        return body


@dataclass(frozen=True)
class Approval:
    """Result of a passing rails check."""

    intent: OrderIntent
    reference_price: Decimal
    estimated_notional: Decimal
    notes: list[str]


# -- the rails --------------------------------------------------------


@dataclass(frozen=True)
class Rails:
    trading_enabled: bool
    allowed_symbols: frozenset[str]
    max_order_usd: Decimal
    max_limit_deviation_pct: Decimal
    max_daily_usd: Decimal
    max_orders_per_day: int
    # Market orders can fill worse than the quote. The notional cap is
    # applied to the quote padded by this much, so slippage can't carry a
    # fill past MAX_ORDER_USD.
    slippage_buffer_pct: Decimal

    @classmethod
    def from_env(cls) -> "Rails":
        symbols = frozenset(
            s.strip().upper()
            for s in os.getenv("ALLOWED_SYMBOLS", "").split(",")
            if s.strip()
        )
        return cls(
            trading_enabled=os.getenv("TRADING_ENABLED", "false").strip().lower()
            == "true",
            allowed_symbols=symbols,
            max_order_usd=_decimal(os.getenv("MAX_ORDER_USD", "0"), "MAX_ORDER_USD"),
            max_limit_deviation_pct=_decimal(
                os.getenv("MAX_LIMIT_DEVIATION_PCT", "5"), "MAX_LIMIT_DEVIATION_PCT"
            ),
            max_daily_usd=_decimal(
                os.getenv("MAX_DAILY_USD", "100"), "MAX_DAILY_USD"
            ),
            max_orders_per_day=int(os.getenv("MAX_ORDERS_PER_DAY", "10")),
            slippage_buffer_pct=_decimal(
                os.getenv("SLIPPAGE_BUFFER_PCT", "1"), "SLIPPAGE_BUFFER_PCT"
            ),
        )

    def describe(self) -> list[str]:
        return [
            f"TRADING_ENABLED        = {str(self.trading_enabled).lower()}",
            f"ALLOWED_SYMBOLS        = {', '.join(sorted(self.allowed_symbols)) or '(none)'}",
            f"MAX_ORDER_USD          = {self.max_order_usd}",
            f"MAX_LIMIT_DEVIATION_PCT= {self.max_limit_deviation_pct}",
            f"MAX_DAILY_USD          = {self.max_daily_usd}",
            f"MAX_ORDERS_PER_DAY     = {self.max_orders_per_day}",
            f"SLIPPAGE_BUFFER_PCT    = {self.slippage_buffer_pct}",
        ]

    # -- the check ----------------------------------------------------

    def check(
        self,
        intent: OrderIntent,
        *,
        bid: Decimal | None,
        ask: Decimal | None,
        held_quantity: Decimal | None = None,
        day: "DayLedger | None" = None,
        sending: bool = True,
    ) -> Approval:
        """Validate an intent. Raises RailViolation listing every failure.

        `sending` is False for a preview that cannot place an order. The
        kill switch is then reported as a note rather than a violation:
        it exists to stop orders being sent, and a preview sends nothing.
        Requiring it to be off just to look at an order would push you to
        arm earlier than you need to. Every other rail still applies.
        """
        reasons: list[str] = []
        notes: list[str] = []

        # --- shape of the request -----------------------------------
        if intent.side not in SIDES:
            reasons.append(f"side must be one of {SIDES}, got {intent.side!r}")
        if intent.order_type not in ORDER_TYPES:
            reasons.append(
                f"type must be one of {ORDER_TYPES}, got {intent.order_type!r}"
            )
        if intent.quantity <= 0:
            reasons.append(f"quantity must be > 0, got {intent.quantity}")
        if intent.order_type == "limit":
            if intent.limit_price is None:
                reasons.append("limit order requires a limit price")
            elif intent.limit_price <= 0:
                reasons.append(f"limit price must be > 0, got {intent.limit_price}")

        # --- kill switch --------------------------------------------
        if not self.trading_enabled:
            if sending:
                reasons.append(
                    "TRADING_ENABLED is false — kill switch is on, no order "
                    "will be sent"
                )
            else:
                notes.append(
                    "kill switch is ON (TRADING_ENABLED=false) — this preview "
                    "is informational; run: python arm.py before --execute"
                )

        # --- allow-list ---------------------------------------------
        if not self.allowed_symbols:
            reasons.append("ALLOWED_SYMBOLS is empty — no symbol is permitted")
        elif intent.symbol not in self.allowed_symbols:
            reasons.append(
                f"{intent.symbol} is not in ALLOWED_SYMBOLS "
                f"({', '.join(sorted(self.allowed_symbols))})"
            )

        if self.max_order_usd <= 0:
            reasons.append("MAX_ORDER_USD must be > 0")

        # --- pricing ------------------------------------------------
        # Buys reference the ask, sells the bid: the side that would
        # actually fill, so the notional estimate is never optimistic.
        reference = ask if intent.side == "buy" else bid
        if reference is None or reference <= 0:
            reasons.append(
                f"no usable quote for {intent.symbol} "
                f"(bid={bid}, ask={ask}) — cannot size the order safely"
            )
            raise RailViolation(reasons)

        if intent.order_type == "market":
            buffer = Decimal(1) + self.slippage_buffer_pct / Decimal(100)
            priced_at = reference * buffer
            notes.append(
                f"market order priced off {reference} "
                f"+{self.slippage_buffer_pct}% slippage buffer"
            )
        else:
            priced_at = intent.limit_price or reference

        notional = (intent.quantity * priced_at).quantize(Decimal("0.01"))

        # --- per-order notional cap ---------------------------------
        if notional > self.max_order_usd:
            reasons.append(
                f"order notional ~${notional} exceeds MAX_ORDER_USD "
                f"${self.max_order_usd}"
            )

        # --- fat-finger limit price ---------------------------------
        mid = None
        if bid and ask and bid > 0 and ask > 0:
            mid = (bid + ask) / Decimal(2)
        if intent.order_type == "limit" and intent.limit_price and mid:
            deviation = abs(intent.limit_price - mid) / mid * Decimal(100)
            if deviation > self.max_limit_deviation_pct:
                reasons.append(
                    f"limit price {intent.limit_price} is {deviation:.2f}% from mid "
                    f"{mid} — exceeds MAX_LIMIT_DEVIATION_PCT "
                    f"{self.max_limit_deviation_pct}% (misplaced decimal?)"
                )
            else:
                notes.append(f"limit price is {deviation:.2f}% from mid {mid}")

        # --- can you actually deliver the asset? --------------------
        if intent.side == "sell":
            if held_quantity is None:
                reasons.append(
                    f"could not read holdings for {intent.symbol} — refusing to sell blind"
                )
            elif intent.quantity > held_quantity:
                reasons.append(
                    f"sell quantity {intent.quantity} exceeds held "
                    f"{held_quantity} {intent.symbol.split('-')[0]}"
                )

        # --- daily caps ---------------------------------------------
        if day is not None:
            if day.order_count >= self.max_orders_per_day:
                reasons.append(
                    f"already placed {day.order_count} orders today — "
                    f"MAX_ORDERS_PER_DAY is {self.max_orders_per_day}"
                )
            projected = day.notional_usd + notional
            if projected > self.max_daily_usd:
                reasons.append(
                    f"order would bring today's notional to ~${projected} — "
                    f"MAX_DAILY_USD is ${self.max_daily_usd} "
                    f"(${day.notional_usd} used)"
                )
            else:
                notes.append(
                    f"day usage after this order: ~${projected} of "
                    f"${self.max_daily_usd}, order {day.order_count + 1} of "
                    f"{self.max_orders_per_day}"
                )

        if reasons:
            raise RailViolation(reasons)

        return Approval(
            intent=intent,
            reference_price=reference,
            estimated_notional=notional,
            notes=notes,
        )


# -- daily usage ledger -----------------------------------------------


@dataclass
class DayLedger:
    """How much has been traded on a given UTC day."""

    date: str
    order_count: int = 0
    notional_usd: Decimal = Decimal(0)


class Ledger:
    """Persists daily usage so the daily caps survive a process restart.

    Without this, a crash-loop would reset the counters and the daily caps
    would mean nothing.
    """

    def __init__(self, path: Path | None = None) -> None:
        # Resolved at call time, not import time, so tests (and anything
        # else) can redirect LEDGER_PATH without the default having been
        # frozen into this signature already.
        self.path = path or LEDGER_PATH

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def load(self) -> DayLedger:
        today = self._today()
        try:
            raw = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return DayLedger(date=today)
        if raw.get("date") != today:
            return DayLedger(date=today)  # new UTC day, counters reset
        return DayLedger(
            date=today,
            order_count=int(raw.get("order_count", 0)),
            notional_usd=_decimal(raw.get("notional_usd", 0), "notional_usd"),
        )

    def record(self, notional: Decimal) -> DayLedger:
        day = self.load()
        day.order_count += 1
        day.notional_usd += notional
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "date": day.date,
                    "order_count": day.order_count,
                    "notional_usd": str(day.notional_usd),
                },
                indent=2,
            )
        )
        return day
