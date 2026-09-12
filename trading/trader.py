"""The single choke point for placing orders.

Nothing else in this project should call client.place_order directly.
Trader.submit is the only path that applies the rails in safety.py, and it
applies them in this order:

  1. read a live quote (and holdings, for a sell)
  2. run every rail — abort on any violation, sending nothing
  3. honour dry-run: build the exact body, log it, do not POST
  4. POST, then record the order against the daily ledger
  5. append to the audit log either way

Steps 1-2 happen before the request is built, so a rejected order never
reaches the network.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from robinhood_client import (
    RobinhoodAPIError,
    RobinhoodConnectionError,
    RobinhoodCryptoClient,
    RobinhoodError,
)
from safety import (
    Approval,
    Ledger,
    OrderIntent,
    RailViolation,
    Rails,
    STATE_DIR,
)

AUDIT_LOG = STATE_DIR / "orders.jsonl"


class Trader:
    def __init__(
        self,
        client: RobinhoodCryptoClient,
        rails: Rails,
        *,
        ledger: Ledger | None = None,
        dry_run: bool = True,
        audit_log: Path = AUDIT_LOG,
    ) -> None:
        self.client = client
        self.rails = rails
        self.ledger = ledger or Ledger()
        # Defaults to True on purpose: forgetting to pass dry_run must fail
        # safe, not send a live order.
        self.dry_run = dry_run
        self.audit_log = audit_log

    # -- market data --------------------------------------------------

    def quote(self, symbol: str) -> tuple[Decimal | None, Decimal | None]:
        """Return (bid, ask) for a symbol, or (None, None) if unavailable."""
        try:
            payload = self.client.get_best_bid_ask(symbol)
        except RobinhoodError:
            # Includes transport failures. No quote means the rails refuse
            # the order rather than size it blind.
            return None, None

        for result in payload.get("results", []):
            if result.get("symbol") != symbol:
                continue
            bid = result.get("bid_inclusive_of_sell_spread") or result.get("price")
            ask = result.get("ask_inclusive_of_buy_spread") or result.get("price")
            return (
                Decimal(str(bid)) if bid is not None else None,
                Decimal(str(ask)) if ask is not None else None,
            )
        return None, None

    def held_quantity(self, symbol: str) -> Decimal | None:
        """Quantity of the base asset currently held, or None if unreadable."""
        asset_code = symbol.split("-")[0]
        try:
            payload = self.client.get_holdings(asset_code)
        except RobinhoodError:
            return None

        for result in payload.get("results", []):
            if result.get("asset_code") == asset_code:
                return Decimal(str(result.get("total_quantity", 0)))
        return Decimal(0)  # endpoint answered, asset simply not held

    # -- rails --------------------------------------------------------

    def preview(self, intent: OrderIntent) -> Approval:
        """Run every rail without sending anything. Raises RailViolation."""
        bid, ask = self.quote(intent.symbol)
        held = self.held_quantity(intent.symbol) if intent.side == "sell" else None
        return self.rails.check(
            intent,
            bid=bid,
            ask=ask,
            held_quantity=held,
            day=self.ledger.load(),
        )

    # -- submission ---------------------------------------------------

    def submit(self, intent: OrderIntent) -> dict[str, Any]:
        """Place an order, or explain why it was refused.

        Raises RailViolation (nothing sent) or RobinhoodAPIError (sent,
        rejected by Robinhood).
        """
        try:
            approval = self.preview(intent)
        except RailViolation as exc:
            self._audit("rejected", intent, reasons=exc.reasons)
            raise

        body = approval.intent.to_api_body()

        if self.dry_run:
            self._audit("dry_run", intent, approval=approval, body=body)
            return {
                "dry_run": True,
                "estimated_notional": str(approval.estimated_notional),
                "reference_price": str(approval.reference_price),
                "notes": approval.notes,
                "body_that_would_be_sent": body,
            }

        try:
            response = self.client.place_order(body)
        except RobinhoodAPIError as exc:
            # Robinhood answered and refused. Nothing was placed, so the
            # daily budget is untouched.
            self._audit(
                "api_error",
                intent,
                approval=approval,
                body=body,
                reasons=[f"{exc.status_code}: {exc.body}"],
            )
            raise
        except RobinhoodConnectionError as exc:
            # We never saw a reply. The order may exist. Charge it against
            # the daily budget anyway — over-counting costs you a little
            # headroom, under-counting would let a flapping connection
            # place unlimited orders.
            self.ledger.record(approval.estimated_notional)
            self._audit(
                "unknown_may_have_been_placed",
                intent,
                approval=approval,
                body=body,
                reasons=[str(exc)],
                reconcile_hint=(
                    "check get_orders() for client_order_id "
                    f"{intent.client_order_id} before retrying; retry with "
                    "the same id so a duplicate cannot fill"
                ),
            )
            raise

        # Only a request Robinhood accepted counts against the daily caps.
        day = self.ledger.record(approval.estimated_notional)
        self._audit(
            "placed",
            intent,
            approval=approval,
            body=body,
            response=response,
            day_usage={
                "order_count": day.order_count,
                "notional_usd": str(day.notional_usd),
            },
        )
        return response

    def cancel(self, order_id: str) -> dict[str, Any]:
        """Cancel an open order. Never gated by the rails — stopping an
        order is always allowed, including while the kill switch is on."""
        if self.dry_run:
            return {"dry_run": True, "would_cancel": order_id}
        response = self.client.cancel_order(order_id)
        self._audit("cancelled", None, response=response, order_id=order_id)
        return response

    # -- audit --------------------------------------------------------

    def _audit(
        self, outcome: str, intent: OrderIntent | None, **extra: Any
    ) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "outcome": outcome,
            "dry_run": self.dry_run,
        }
        if intent is not None:
            record["intent"] = {
                "client_order_id": intent.client_order_id,
                "symbol": intent.symbol,
                "side": intent.side,
                "quantity": str(intent.quantity),
                "type": intent.order_type,
                "limit_price": (
                    str(intent.limit_price) if intent.limit_price is not None else None
                ),
            }
        for key, value in extra.items():
            if key == "approval" and value is not None:
                record["approval"] = {
                    "reference_price": str(value.reference_price),
                    "estimated_notional": str(value.estimated_notional),
                    "notes": value.notes,
                }
            else:
                record[key] = value

        try:
            self.audit_log.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_log.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError:
            pass  # never let an audit-log failure block or crash a trade
