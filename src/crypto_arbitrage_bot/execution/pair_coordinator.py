"""Durable, testnet-only two-leg IOC orchestration.

This module never chooses whether to trade. Callers must supply bounded entry
and recovery orders. A persisted attempt is never submitted twice; ambiguous
submissions are queried once and otherwise left for explicit recovery.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError
from crypto_arbitrage_bot.execution.types import (
    ExecutionMarket,
    IocLimitOrder,
    OrderReceipt,
    OrderState,
)
from crypto_arbitrage_bot.models.market import Side
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


class PairExecutionState(StrEnum):
    PREPARED = "PREPARED"
    OPENING_SPOT = "OPENING_SPOT"
    OPENING_PERP = "OPENING_PERP"
    HEDGED = "HEDGED"
    CLOSING_SPOT = "CLOSING_SPOT"
    CLOSING_PERP = "CLOSING_PERP"
    CLOSED = "CLOSED"
    FLATTENED = "FLATTENED"
    FAILED = "FAILED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    DUPLICATE_BLOCKED = "DUPLICATE_BLOCKED"


class PairOrderGateway(Protocol):
    market: ExecutionMarket

    def place_ioc_limit_order(self, order: IocLimitOrder) -> OrderReceipt: ...

    def get_order(self, symbol: str, client_order_id: str) -> OrderReceipt: ...


@dataclass(frozen=True, slots=True)
class PairOrderPlan:
    """Caller-provided Long Spot + Short Perp entry and bounded exits."""

    attempt_id: str
    spot_entry: IocLimitOrder
    perp_entry: IocLimitOrder
    spot_exit: IocLimitOrder
    perp_exit: IocLimitOrder
    scheduled_exit_at: datetime | None = field(default=None, kw_only=True)
    funding_settlement_at: datetime | None = field(default=None, kw_only=True)
    opportunity_id: str | None = field(default=None, kw_only=True)
    projection_json: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        timing = (self.scheduled_exit_at, self.funding_settlement_at, self.opportunity_id)
        if any(value is not None for value in timing):
            if any(value is None for value in timing):
                raise ValueError("funding execution plan requires exit, settlement and opportunity together")
            assert self.scheduled_exit_at is not None and self.funding_settlement_at is not None
            close = utc_timestamp(self.scheduled_exit_at)
            settlement = utc_timestamp(self.funding_settlement_at)
            if close <= settlement or not isinstance(self.opportunity_id, str) or not self.opportunity_id.strip():
                raise ValueError("scheduled exit must follow settlement and identify its opportunity")
            object.__setattr__(self, "scheduled_exit_at", close)
            object.__setattr__(self, "funding_settlement_at", settlement)
        if self.projection_json is not None and self.scheduled_exit_at is None:
            raise ValueError("projection evidence requires a scheduled exit")
        if type(self.attempt_id) is not str or not self.attempt_id.strip() or len(self.attempt_id) > 128:
            raise ValueError("attempt_id must be a non-empty string of at most 128 characters")
        orders = (self.spot_entry, self.perp_entry, self.spot_exit, self.perp_exit)
        if any(not isinstance(order, IocLimitOrder) for order in orders):
            raise TypeError("all pair orders must be IocLimitOrder")
        if self.spot_entry.market is not ExecutionMarket.SPOT or self.perp_entry.market is not ExecutionMarket.USDTM:
            raise ValueError("entry orders must target Spot and USDT-M respectively")
        if self.spot_exit.market is not ExecutionMarket.SPOT or self.perp_exit.market is not ExecutionMarket.USDTM:
            raise ValueError("exit orders must target Spot and USDT-M respectively")
        if self.spot_entry.side is not Side.BUY or self.perp_entry.side is not Side.SELL:
            raise ValueError("entry direction must be Long Spot + Short Perp")
        if self.spot_exit.side is not Side.SELL or self.perp_exit.side is not Side.BUY:
            raise ValueError("recovery/exit direction must close Long Spot + Short Perp")
        symbols = {order.symbol for order in orders}
        quantities = {order.quantity for order in (self.spot_entry, self.perp_entry)}
        if len(symbols) != 1 or len(quantities) != 1:
            raise ValueError("all orders must share one symbol and entries must share one quantity")
        if any(order.quantity != self.spot_entry.quantity for order in (self.spot_exit, self.perp_exit)):
            raise ValueError("exit orders must cover the full planned entry quantity")
        if len({order.client_order_id for order in orders}) != 4:
            raise ValueError("each leg must have a distinct client_order_id")

    @property
    def symbol(self) -> str:
        return self.spot_entry.symbol


@dataclass(frozen=True, slots=True)
class PairExecutionResult:
    attempt_id: str
    state: PairExecutionState
    spot_entry: OrderReceipt | None = None
    perp_entry: OrderReceipt | None = None
    spot_exit: OrderReceipt | None = None
    perp_exit: OrderReceipt | None = None
    failure_code: str | None = None
    failure_reason: str | None = None
    previous_state: PairExecutionState | None = None


class ExecutionJournalError(RuntimeError):
    pass


def _order_data(order: IocLimitOrder) -> dict[str, str | bool]:
    data: dict[str, str | bool] = {
        "market": order.market.value,
        "symbol": order.symbol,
        "side": order.side.value,
        "quantity": str(order.quantity),
        "limit_price": str(order.limit_price),
        "client_order_id": order.client_order_id,
        "reduce_only": order.reduce_only,
    }
    if order.submit_before is not None:
        data["submit_before"] = order.submit_before.isoformat()
    return data


def _plan_json(plan: PairOrderPlan) -> str:
    data: dict[str, object] = {
        "attempt_id": plan.attempt_id,
        "spot_entry": _order_data(plan.spot_entry),
        "perp_entry": _order_data(plan.perp_entry),
        "spot_exit": _order_data(plan.spot_exit),
        "perp_exit": _order_data(plan.perp_exit),
    }
    if plan.scheduled_exit_at is not None:
        assert plan.funding_settlement_at is not None
        data.update(scheduled_exit_at=plan.scheduled_exit_at.isoformat(),
                    funding_settlement_at=plan.funding_settlement_at.isoformat(), opportunity_id=plan.opportunity_id)
        if plan.projection_json is not None:
            data["projection_json"] = plan.projection_json
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _receipt_json(receipt: OrderReceipt | None) -> str | None:
    if receipt is None:
        return None
    data: dict[str, object] = {
        "market": receipt.market.value,
        "symbol": receipt.symbol,
        "order_id": receipt.order_id,
        "client_order_id": receipt.client_order_id,
        "status": receipt.status.value,
        "original_quantity": str(receipt.original_quantity),
        "executed_quantity": str(receipt.executed_quantity),
        "cumulative_quote_quantity": (
            None if receipt.cumulative_quote_quantity is None else str(receipt.cumulative_quote_quantity)
        ),
        "average_price": None if receipt.average_price is None else str(receipt.average_price),
        "base_commission": None if receipt.base_commission is None else str(receipt.base_commission),
    }
    # Preserve unknown legacy evidence without synthesizing fields from a plan.
    for name in ('side', 'order_type', 'time_in_force', 'limit_price', 'reduce_only', 'position_side'):
        value = getattr(receipt, name)
        if value is not None:
            data[name] = str(value) if isinstance(value, (Side, Decimal)) else value
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _receipt_from_json(payload: str | None) -> OrderReceipt | None:
    if payload is None:
        return None
    data = json.loads(payload)
    return OrderReceipt(
        market=ExecutionMarket(data["market"]), symbol=data["symbol"], order_id=data["order_id"],
        client_order_id=data["client_order_id"], status=OrderState(data["status"]),
        original_quantity=Decimal(data["original_quantity"]),
        executed_quantity=Decimal(data["executed_quantity"]),
        cumulative_quote_quantity=(None if data["cumulative_quote_quantity"] is None
                                   else Decimal(data["cumulative_quote_quantity"])),
        average_price=None if data["average_price"] is None else Decimal(data["average_price"]),
        base_commission=None if data.get("base_commission") is None else Decimal(data["base_commission"]),
        side=None if data.get('side') is None else Side(data['side']),
        order_type=data.get('order_type'), time_in_force=data.get('time_in_force'),
        limit_price=None if data.get('limit_price') is None else Decimal(data['limit_price']),
        reduce_only=data.get('reduce_only'), position_side=data.get('position_side'),
    )


class SQLitePairExecutionJournal:
    """Small file-backed journal; plan IDs are unique and writes are durable."""

    def __init__(self, database_path: Path) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path must be Path")
        self.database_path = database_path.resolve()

    @contextmanager
    def _connect(self, *, create: bool = False) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        try:
            if create:
                self.database_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "rwc" if create else "rw"
            connection = sqlite3.connect(self.database_path.as_uri() + f"?mode={mode}", uri=True, timeout=1)
            connection.row_factory = sqlite3.Row
            with connection:
                yield connection
        except (sqlite3.Error, OSError) as error:
            raise ExecutionJournalError("pair execution journal unavailable") from error
        finally:
            if connection is not None:
                connection.close()

    def initialize(self) -> None:
        try:
            with self._connect(create=True) as db:
                db.execute("""CREATE TABLE IF NOT EXISTS testnet_pair_attempts (
                    attempt_id TEXT PRIMARY KEY NOT NULL,
                    plan_json TEXT NOT NULL,
                    plan_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL,
                    spot_entry TEXT, perp_entry TEXT, spot_exit TEXT, perp_exit TEXT,
                    failure_code TEXT, failure_reason TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                )""")
                db.execute("""CREATE TABLE IF NOT EXISTS testnet_order_intents (
                    client_order_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL,
                    order_json TEXT NOT NULL, created_at TEXT NOT NULL
                )""")
        except (sqlite3.Error, OSError) as error:
            raise ExecutionJournalError("pair execution journal initialization failed") from error

    def create(self, plan: PairOrderPlan) -> bool:
        encoded = _plan_json(plan)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                existing = db.execute("SELECT attempt_id, plan_json FROM testnet_pair_attempts WHERE state NOT IN "
                                      "('CLOSED','FLATTENED','FAILED')").fetchall()
                if any(row["attempt_id"] != plan.attempt_id and
                       json.loads(row["plan_json"])["spot_entry"]["symbol"] == plan.symbol for row in existing):
                    raise ValueError("symbol already has an active or unresolved pair")
                cursor = db.execute("""INSERT OR IGNORE INTO testnet_pair_attempts
                    (attempt_id,plan_json,plan_sha256,state,created_at,updated_at)
                    VALUES(?,?,?,'PREPARED',?,?)""", (plan.attempt_id, encoded, digest, now, now))
                row = db.execute("SELECT plan_sha256 FROM testnet_pair_attempts WHERE attempt_id=?",
                                 (plan.attempt_id,)).fetchone()
                if row is None or row["plan_sha256"] != digest:
                    raise ValueError("attempt_id already belongs to a different order plan")
                return cursor.rowcount == 1
        except (sqlite3.Error, OSError) as error:
            raise ExecutionJournalError("pair execution intent could not be persisted") from error

    def claim_close(self, plan: PairOrderPlan) -> bool:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT plan_json FROM testnet_pair_attempts WHERE attempt_id=?",
                             (plan.attempt_id,)).fetchone()
            stored = None if row is None else json.loads(row["plan_json"])
            if stored is not None:
                for name in ("spot_entry", "perp_entry", "spot_exit", "perp_exit"):
                    stored[name].setdefault("reduce_only", False)
            if stored != json.loads(_plan_json(plan)):
                raise ValueError("close plan differs from persisted entry intent")
            return db.execute("UPDATE testnet_pair_attempts SET state='CLOSING_SPOT',updated_at=? "
                              "WHERE attempt_id=? AND state='HEDGED'",
                              (datetime.now(UTC).isoformat(), plan.attempt_id)).rowcount == 1

    def record_order(self, order: IocLimitOrder) -> None:
        # The exact actual order (including partial recovery quantity) precedes POST.
        with self._connect() as db:
            parents = db.execute("SELECT attempt_id,plan_json FROM testnet_pair_attempts").fetchall()
            matches = [row["attempt_id"] for row in parents if any(
                json.loads(row["plan_json"])[name]["client_order_id"] == order.client_order_id
                for name in ("spot_entry", "perp_entry", "spot_exit", "perp_exit"))]
            if len(matches) != 1:
                raise ExecutionJournalError("order intent does not belong to exactly one pair")
            db.execute("INSERT INTO testnet_order_intents VALUES(?,?,?,?)",
                       (order.client_order_id, matches[0], json.dumps(_order_data(order)), datetime.now(UTC).isoformat()))

    def plans(self) -> tuple[tuple[PairOrderPlan, datetime], ...]:
        with self._connect() as db:
            rows = db.execute("SELECT plan_json,created_at FROM testnet_pair_attempts ORDER BY created_at").fetchall()
        def order(data: dict[str, object]) -> IocLimitOrder:
            return IocLimitOrder(ExecutionMarket(str(data["market"])), str(data["symbol"]), Side(str(data["side"])),
                                 Decimal(str(data["quantity"])), Decimal(str(data["limit_price"])),
                                 str(data["client_order_id"]), bool(data.get("reduce_only", False)),
                                 submit_before=None if data.get("submit_before") is None else
                                 datetime.fromisoformat(str(data["submit_before"])))
        return tuple((PairOrderPlan(data["attempt_id"], *(order(data[name]) for name in
                     ("spot_entry", "perp_entry", "spot_exit", "perp_exit")),
                     scheduled_exit_at=None if data.get("scheduled_exit_at") is None else datetime.fromisoformat(data["scheduled_exit_at"]),
                     funding_settlement_at=None if data.get("funding_settlement_at") is None else datetime.fromisoformat(data["funding_settlement_at"]),
                     opportunity_id=data.get("opportunity_id"), projection_json=data.get("projection_json")),
                     datetime.fromisoformat(row["created_at"]))
                     for row in rows for data in (json.loads(row["plan_json"]),))

    def intents(self, attempt_id: str) -> tuple[IocLimitOrder, ...]:
        with self._connect() as db:
            rows = db.execute("SELECT order_json FROM testnet_order_intents WHERE attempt_id=? ORDER BY created_at",
                              (attempt_id,)).fetchall()
        return tuple(IocLimitOrder(ExecutionMarket(d["market"]), d["symbol"], Side(d["side"]),
                                  Decimal(d["quantity"]), Decimal(d["limit_price"]), d["client_order_id"],
                                  d.get("reduce_only", False),
                                  submit_before=None if d.get("submit_before") is None else
                                  datetime.fromisoformat(d["submit_before"]))
                     for row in rows for d in (json.loads(row[0]),))

    def update(self, attempt_id: str, *, state: PairExecutionState,
               spot_entry: OrderReceipt | None = None, perp_entry: OrderReceipt | None = None,
               spot_exit: OrderReceipt | None = None, perp_exit: OrderReceipt | None = None,
               failure_code: str | None = None, failure_reason: str | None = None) -> None:
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as db:
                cursor = db.execute("""UPDATE testnet_pair_attempts SET state=?,spot_entry=COALESCE(?,spot_entry),
                    perp_entry=COALESCE(?,perp_entry),spot_exit=COALESCE(?,spot_exit),
                    perp_exit=COALESCE(?,perp_exit),failure_code=?,failure_reason=?,updated_at=? WHERE attempt_id=?""",
                    (state.value, _receipt_json(spot_entry), _receipt_json(perp_entry),
                     _receipt_json(spot_exit), _receipt_json(perp_exit), failure_code,
                     None if failure_reason is None else failure_reason[:240], now, attempt_id))
                if cursor.rowcount != 1:
                    raise ValueError("execution attempt does not exist")
        except (sqlite3.Error, OSError) as error:
            raise ExecutionJournalError("pair execution state could not be persisted") from error

    def get(self, attempt_id: str) -> PairExecutionResult | None:
        try:
            with self._connect() as db:
                row = db.execute("SELECT * FROM testnet_pair_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None:
                return None
            return PairExecutionResult(
                attempt_id=row["attempt_id"], state=PairExecutionState(row["state"]),
                spot_entry=_receipt_from_json(row["spot_entry"]),
                perp_entry=_receipt_from_json(row["perp_entry"]),
                spot_exit=_receipt_from_json(row["spot_exit"]),
                perp_exit=_receipt_from_json(row["perp_exit"]),
                failure_code=row["failure_code"], failure_reason=row["failure_reason"],
            )
        except (sqlite3.Error, OSError, ValueError, KeyError, TypeError) as error:
            raise ExecutionJournalError("pair execution result could not be loaded") from error


class TestnetPairCoordinator:
    """Submit both IOC legs, flatten known partials, and stop on ambiguity."""

    def __init__(self, spot: PairOrderGateway, perp: PairOrderGateway,
                 journal: SQLitePairExecutionJournal) -> None:
        if spot.market is not ExecutionMarket.SPOT or perp.market is not ExecutionMarket.USDTM:
            raise ValueError("coordinator requires Spot and USDT-M testnet clients")
        self.spot = spot
        self.perp = perp
        self.journal = journal

    def reconcile(self, plan: PairOrderPlan) -> PairExecutionResult:
        """Restart recovery queries persisted IDs only. Never sends a new order."""
        current = self._result(plan.attempt_id)
        if current.state in (PairExecutionState.CLOSED, PairExecutionState.FLATTENED, PairExecutionState.FAILED):
            return current
        if current.state is PairExecutionState.HEDGED:
            return current
        receipts: dict[str, OrderReceipt] = {}
        for order in self.journal.intents(plan.attempt_id):
            try:
                gateway = self.spot if order.market is ExecutionMarket.SPOT else self.perp
                receipt = gateway.get_order(order.symbol, order.client_order_id)
                if not self._matches(receipt, order) or receipt.status in (OrderState.NEW, OrderState.PARTIALLY_FILLED):
                    return self._recovery(plan, "RECONCILIATION_UNRESOLVED", "Order identity or terminal state unresolved")
                name = next(name for name in ("spot_entry", "perp_entry", "spot_exit", "perp_exit")
                            if getattr(plan, name).client_order_id == order.client_order_id)
                previous = getattr(current, name)
                if previous is not None and (previous.order_id != receipt.order_id
                        or self._matches(previous, order) and receipt.executed_quantity < previous.executed_quantity):
                    return self._recovery(plan, 'RECONCILIATION_UNRESOLVED', 'Order identity or confirmed quantity changed')
                receipts[name] = receipt
            except Exception:
                return self._recovery(plan, "RECONCILIATION_UNRESOLVED", "Could not query a persisted order")
        if not receipts:
            return self._finish(plan, PairExecutionState.FAILED, failure_code="UNSUBMITTED_INTENT",
                                failure_reason="Restart found no submitted order intent; no order resent")
        se, pe, sx, px = (receipts.get(name) for name in ("spot_entry", "perp_entry", "spot_exit", "perp_exit"))
        if se and pe and se.base_commission == 0 and all(self._fully_filled(r, plan.spot_entry.quantity) for r in (se, pe)):
            if sx and px and all(self._fully_filled(r, plan.spot_entry.quantity) for r in (sx, px)):
                self.journal.update(plan.attempt_id, state=PairExecutionState.CLOSED,
                                    spot_entry=se, perp_entry=pe, spot_exit=sx, perp_exit=px)
                return self._result(plan.attempt_id)
            if sx is None and px is None:
                self.journal.update(plan.attempt_id, state=PairExecutionState.HEDGED, spot_entry=se, perp_entry=pe)
                return self._result(plan.attempt_id)
        return self._recovery(plan, "RECONCILIATION_UNRESOLVED", "Known exposure needs explicit recovery", **receipts)

    def open_pair(self, plan: PairOrderPlan) -> PairExecutionResult:
        created = self.journal.create(plan)
        if not created:
            previous = self.journal.get(plan.attempt_id)
            assert previous is not None
            return replace(previous, state=PairExecutionState.DUPLICATE_BLOCKED,
                           previous_state=previous.state)

        self.journal.update(plan.attempt_id, state=PairExecutionState.OPENING_SPOT)
        spot_receipt, spot_ambiguous = self._submit(self.spot, plan.spot_entry)
        if spot_ambiguous:
            return self._recovery(plan, "SPOT_SUBMISSION_UNKNOWN", "Spot order status remains unknown",
                                  spot_entry=spot_receipt)
        if spot_receipt is None:
            return self._finish(plan, PairExecutionState.FAILED, failure_code="SPOT_ENTRY_FAILED",
                                failure_reason="Spot entry was rejected before execution")
        self.journal.update(plan.attempt_id, state=PairExecutionState.OPENING_PERP, spot_entry=spot_receipt)
        if not self._fully_filled(spot_receipt, plan.spot_entry.quantity):
            return self._flatten(plan, spot_receipt, None, reason="SPOT_ENTRY_PARTIAL")
        if spot_receipt.base_commission is None:
            return self._recovery(plan, "SPOT_COMMISSION_UNKNOWN", "Cannot hedge unknown net Spot inventory",
                                  spot_entry=spot_receipt)
        if spot_receipt.base_commission != 0:
            return self._flatten(plan, spot_receipt, None, reason="SPOT_BASE_COMMISSION")

        perp_receipt, perp_ambiguous = self._submit(self.perp, plan.perp_entry)
        if perp_ambiguous:
            self.journal.update(plan.attempt_id, state=PairExecutionState.RECOVERY_REQUIRED,
                                spot_entry=spot_receipt, perp_entry=perp_receipt,
                                failure_code="PERP_SUBMISSION_UNKNOWN",
                                failure_reason="Perp order status remains unknown; no blind resubmission")
            return self._result(plan.attempt_id)
        if perp_receipt is None:
            return self._flatten(plan, spot_receipt, None, reason="PERP_ENTRY_FAILED")
        self.journal.update(plan.attempt_id, state=PairExecutionState.HEDGED if self._fully_filled(
            perp_receipt, plan.perp_entry.quantity) else PairExecutionState.OPENING_PERP,
            spot_entry=spot_receipt, perp_entry=perp_receipt)
        if self._fully_filled(perp_receipt, plan.perp_entry.quantity):
            return self._result(plan.attempt_id)
        return self._flatten(plan, spot_receipt, perp_receipt, reason="PERP_ENTRY_PARTIAL")

    def close_pair(self, plan: PairOrderPlan, *, spot_price: Decimal | None = None,
                   perp_price: Decimal | None = None) -> PairExecutionResult:
        current = self.journal.get(plan.attempt_id)
        if current is None:
            raise ValueError("pair attempt has not been journaled")
        if current.state is not PairExecutionState.HEDGED:
            return replace(current, state=PairExecutionState.DUPLICATE_BLOCKED,
                           previous_state=current.state)
        if current.spot_entry is None or current.perp_entry is None:
            return self._recovery(plan, "ENTRY_RECEIPT_MISSING", "hedged state lacks both entry receipts")
        if not self.journal.claim_close(plan):
            return replace(self._result(plan.attempt_id), state=PairExecutionState.DUPLICATE_BLOCKED)
        if current.spot_entry.base_commission is None:
            return self._recovery(plan, "SPOT_COMMISSION_UNKNOWN", "Spot net inventory is unknown")
        spot_close = self._close_order(plan.spot_exit, current.spot_entry.executed_quantity - current.spot_entry.base_commission)
        if spot_price is not None:
            spot_close = replace(spot_close, limit_price=spot_price)
        spot_receipt, ambiguous = self._submit(self.spot, spot_close)
        if ambiguous or spot_receipt is None or not self._fully_filled(spot_receipt, spot_close.quantity):
            return self._recovery(plan, "SPOT_EXIT_INCOMPLETE", "Spot exit is incomplete; manual recovery required",
                                  spot_exit=spot_receipt)
        self.journal.update(plan.attempt_id, state=PairExecutionState.CLOSING_PERP, spot_exit=spot_receipt)
        perp_close = self._close_order(plan.perp_exit, current.perp_entry.executed_quantity)
        if perp_price is not None:
            perp_close = replace(perp_close, limit_price=perp_price)
        perp_receipt, ambiguous = self._submit(self.perp, perp_close)
        if ambiguous or perp_receipt is None or not self._fully_filled(perp_receipt, perp_close.quantity):
            return self._recovery(plan, "PERP_EXIT_INCOMPLETE", "Perp exit is incomplete; manual recovery required",
                                  spot_exit=spot_receipt, perp_exit=perp_receipt)
        self.journal.update(plan.attempt_id, state=PairExecutionState.CLOSED,
                            spot_exit=spot_receipt, perp_exit=perp_receipt)
        return self._result(plan.attempt_id)

    def _submit(self, gateway: PairOrderGateway, order: IocLimitOrder) -> tuple[OrderReceipt | None, bool]:
        self.journal.record_order(order)
        try:
            receipt = gateway.place_ioc_limit_order(order)
        except TestnetExecutionError as error:
            if not error.submission_may_have_reached_exchange:
                return None, False
            return self._query(gateway, order)
        except Exception:
            # Unknown gateway errors are conservatively ambiguous; never POST again.
            return None, True
        if not self._matches(receipt, order):
            return receipt, True
        if receipt.status in (OrderState.NEW, OrderState.PARTIALLY_FILLED):
            return self._query(gateway, order, receipt)
        return receipt, False

    def _query(self, gateway: PairOrderGateway, order: IocLimitOrder, fallback: OrderReceipt | None = None
               ) -> tuple[OrderReceipt | None, bool]:
        try:
            receipt = gateway.get_order(order.symbol, order.client_order_id)
            if fallback is not None and (receipt.order_id != fallback.order_id
                    or receipt.executed_quantity < fallback.executed_quantity):
                return fallback, True
            return receipt, not self._matches(receipt, order) or receipt.status in (
                OrderState.NEW, OrderState.PARTIALLY_FILLED)
        except Exception:
            return fallback, True

    @staticmethod
    def _matches(receipt: OrderReceipt, order: IocLimitOrder) -> bool:
        return receipt.matches_order(order)

    @staticmethod
    def _fully_filled(receipt: OrderReceipt, requested: Decimal) -> bool:
        return receipt.status is OrderState.FILLED and receipt.executed_quantity == requested

    @staticmethod
    def _close_order(order: IocLimitOrder, quantity: Decimal) -> IocLimitOrder:
        if quantity <= 0:
            raise ValueError("cannot close a zero fill")
        return replace(order, quantity=quantity, reduce_only=order.market is ExecutionMarket.USDTM)

    def _flatten(self, plan: PairOrderPlan, spot_entry: OrderReceipt,
                 perp_entry: OrderReceipt | None, *, reason: str) -> PairExecutionResult:
        self.journal.update(plan.attempt_id, state=PairExecutionState.CLOSING_SPOT,
                            spot_entry=spot_entry, perp_entry=perp_entry,
                            failure_code=reason, failure_reason="Known partial exposure; attempting one bounded IOC flatten")
        spot_exit: OrderReceipt | None = None
        perp_exit: OrderReceipt | None = None
        if spot_entry.executed_quantity > 0:
            if spot_entry.base_commission is None:
                return self._recovery(plan, "SPOT_COMMISSION_UNKNOWN", "Spot commission requires reconciliation",
                                      spot_entry=spot_entry, perp_entry=perp_entry)
            quantity = spot_entry.executed_quantity - spot_entry.base_commission
            if quantity <= 0:
                return self._recovery(plan, "SPOT_INVENTORY_INVALID", "No confirmed net Spot inventory")
            spot_order = self._close_order(plan.spot_exit, quantity)
            spot_exit, ambiguous = self._submit(self.spot, spot_order)
            if ambiguous or spot_exit is None or not self._fully_filled(spot_exit, spot_order.quantity):
                return self._recovery(plan, "SPOT_FLATTEN_INCOMPLETE", "Spot exposure could not be fully flattened",
                                      spot_entry=spot_entry, perp_entry=perp_entry, spot_exit=spot_exit)
            self.journal.update(plan.attempt_id, state=PairExecutionState.CLOSING_PERP,
                                spot_entry=spot_entry, perp_entry=perp_entry, spot_exit=spot_exit,
                                failure_code=reason, failure_reason="Spot exposure flattened; closing any Perp fill")
        if perp_entry is not None and perp_entry.executed_quantity > 0:
            perp_order = self._close_order(plan.perp_exit, perp_entry.executed_quantity)
            perp_exit, ambiguous = self._submit(self.perp, perp_order)
            if ambiguous or perp_exit is None or not self._fully_filled(perp_exit, perp_order.quantity):
                return self._recovery(plan, "PERP_FLATTEN_INCOMPLETE", "Perp exposure could not be fully flattened",
                                      spot_entry=spot_entry, perp_entry=perp_entry,
                                      spot_exit=spot_exit, perp_exit=perp_exit)
        self.journal.update(plan.attempt_id, state=PairExecutionState.FLATTENED,
                            spot_entry=spot_entry, perp_entry=perp_entry,
                            spot_exit=spot_exit, perp_exit=perp_exit,
                            failure_code=reason, failure_reason="Entry was not fully hedged; known fills flattened")
        return self._result(plan.attempt_id)

    def _recovery(self, plan: PairOrderPlan, code: str, reason: str, **receipts: OrderReceipt | None
                  ) -> PairExecutionResult:
        self.journal.update(plan.attempt_id, state=PairExecutionState.RECOVERY_REQUIRED,
                            failure_code=code, failure_reason=reason, **receipts)
        return self._result(plan.attempt_id)

    def _finish(self, plan: PairOrderPlan, state: PairExecutionState, *, failure_code: str,
                failure_reason: str) -> PairExecutionResult:
        self.journal.update(plan.attempt_id, state=state, failure_code=failure_code, failure_reason=failure_reason)
        return self._result(plan.attempt_id)

    def _result(self, attempt_id: str) -> PairExecutionResult:
        result = self.journal.get(attempt_id)
        if result is None:
            raise ExecutionJournalError("pair execution journal lost an attempt")
        return result
