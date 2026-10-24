"""Actual abort fill/fee/currency evidence. No PnL formula or order submission."""

import json
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from crypto_arbitrage_bot.execution.abort_balance_proof import (
    ABORT_SCOPE,
    REPAIR_TABLES,
    AbortBalanceProofRepository,
    journal_fingerprint,
    supported_abort,
)
from crypto_arbitrage_bot.execution.account_balance_audit import (
    audit_balance_changes,
    capture_balance_snapshot,
)
from crypto_arbitrage_bot.execution.balance_accounting import (
    BalanceFollowupResult,
    BalanceFollowupStatus,
)
from crypto_arbitrage_bot.execution.balance_audit_repository import BalanceAuditRepository
from crypto_arbitrage_bot.execution.balance_check_repository import BalanceCheckRepository
from crypto_arbitrage_bot.execution.balance_frame_store import BalanceFrameStore
from crypto_arbitrage_bot.execution.pair_coordinator import (
    PairExecutionState,
    PairOrderPlan,
    _receipt_from_json,
)
from crypto_arbitrage_bot.execution.reconciliation_repository import ReconciliationRepository
from crypto_arbitrage_bot.execution.spot_exit_recovery import _signature
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError
from crypto_arbitrage_bot.execution.trade_reconciliation import (
    LegTradeEvidence,
    ReconciliationStatus,
    TradeReconciliation,
)
from crypto_arbitrage_bot.execution.types import (
    AccountTrade,
    ExecutionMarket,
    FundingPayment,
    IocLimitOrder,
    OrderReceipt,
    OrderState,
)
from crypto_arbitrage_bot.models.calculation import TradeLeg
from crypto_arbitrage_bot.models.market import SUPPORTED_SYMBOLS
from crypto_arbitrage_bot.utils.decimal_math import decimal_calculation, utc_timestamp

if TYPE_CHECKING:
    from crypto_arbitrage_bot.execution.runner import TestnetBot


class AbortAccountingCode(StrEnum):
    EVIDENCE_CONFLICT = "ABORT_EVIDENCE_CONFLICT"
    INCOMPLETE_FILLS = "ABORT_INCOMPLETE_FILLS"
    UNRESOLVED_ORDER = "ABORT_ORDER_UNRESOLVED"
    INVENTORY_CONFLICT = "ABORT_INVENTORY_CONFLICT"
    ACCOUNT_CONFLICT = "ABORT_ACCOUNT_CONFLICT"
    FUNDING_PENDING = "FUNDING_EVIDENCE_PENDING"


class AbortAccountingError(ValueError):
    def __init__(self, code: AbortAccountingCode) -> None:
        self.code = code
        super().__init__(code.value)


@dataclass(frozen=True)
class _Source:
    leg: TradeLeg
    order: IocLimitOrder
    cached: OrderReceipt


def _require(condition: bool, code: AbortAccountingCode = AbortAccountingCode.EVIDENCE_CONFLICT) -> None:
    if not condition:
        raise AbortAccountingError(code)


def _sources(bot: "TestnetBot", plan: PairOrderPlan) -> tuple[_Source, ...]:
    pair = bot.journal.get(plan.attempt_id)
    _require(pair is not None and supported_abort(pair))
    assert pair is not None
    expected = {order.client_order_id: (leg, order, cached) for leg, order, cached in (
        (TradeLeg.SPOT_ENTRY, plan.spot_entry, pair.spot_entry),
        (TradeLeg.SPOT_EXIT, plan.spot_exit, pair.spot_exit),
        (TradeLeg.PERP_ENTRY, plan.perp_entry, pair.perp_entry),
        (TradeLeg.PERP_EXIT, plan.perp_exit, pair.perp_exit))}
    intents = bot.journal.intents(plan.attempt_id)
    _require(len({i.client_order_id for i in intents}) == len(intents) and bool(intents))
    sources: list[_Source] = []
    for order in intents:
        _require(order.client_order_id in expected)
        leg, intended, cached = expected[order.client_order_id]
        _require(cached is not None and (order.market, order.symbol, order.side) ==
                 (intended.market, intended.symbol, intended.side))
        assert cached is not None
        if leg in (TradeLeg.SPOT_ENTRY, TradeLeg.PERP_ENTRY):
            _require(order == intended and not order.reduce_only)
        else:
            _require(order.submit_before is None and order.reduce_only is (leg is TradeLeg.PERP_EXIT)
                     and order.quantity <= plan.spot_entry.quantity)
        sources.append(_Source(leg, order, cached))
    submitted = {i.client_order_id for i in intents}
    _require(plan.spot_entry.client_order_id in submitted)
    for _, intended, cached in expected.values():
        _require(cached is None or intended.client_order_id in submitted)
    with bot.journal._connect() as db:
        for table, leg, original, cached_original, suffix in (
            (REPAIR_TABLES[0], TradeLeg.SPOT_EXIT, plan.spot_exit, pair.spot_exit, "sxr"),
            (REPAIR_TABLES[1], TradeLeg.PERP_EXIT, plan.perp_exit, pair.perp_exit, "rx")):
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
                continue
            rows = db.execute(f"SELECT * FROM {table} WHERE attempt_id=? ORDER BY ordinal", (plan.attempt_id,)).fetchall()
            for ordinal, row in enumerate(rows, start=1):
                data = json.loads(row["order_json"])
                cached = _receipt_from_json(row["receipt_json"])
                initial = _receipt_from_json(row["original_exit_json"])
                _require(row["ordinal"] == ordinal <= bot.config.max_exit_recovery_attempts
                         and original.client_order_id in submitted and cached_original is not None
                         and cached is not None and initial is not None)
                assert cached is not None and initial is not None and cached_original is not None
                _require(_signature(initial) == _signature(cached_original)
                         and data["client_order_id"] == row["client_order_id"] == f"{plan.attempt_id[:24]}-{suffix}{ordinal}"
                         and data["client_order_id"] not in submitted and data.get("submit_before") is None
                         and data["symbol"] == plan.symbol and data["market"] == original.market.value
                         and data["side"] == original.side.value and data["reduce_only"] is (leg is TradeLeg.PERP_EXIT)
                         and type(data["quantity"]) is str and type(data["limit_price"]) is str)
                order = IocLimitOrder(original.market, plan.symbol, original.side,
                                      Decimal(data["quantity"]), Decimal(data["limit_price"]),
                                      data["client_order_id"], leg is TradeLeg.PERP_EXIT)
                submitted.add(order.client_order_id)
                sources.append(_Source(leg, order, cached))
    return tuple(sources)


@decimal_calculation
def _fill_report(bot: "TestnetBot", plan: PairOrderPlan) -> TradeReconciliation:
    grouped: dict[TradeLeg, list[AccountTrade]] = {leg: [] for leg in TradeLeg}
    seen: set[tuple[ExecutionMarket, str]] = set()
    checked_at = utc_timestamp(bot.clock())
    terminal = {OrderState.FILLED, OrderState.CANCELED, OrderState.EXPIRED,
                OrderState.EXPIRED_IN_MATCH, OrderState.REJECTED}
    for source in _sources(bot, plan):
        order = source.order
        client = bot.spot if order.market is ExecutionMarket.SPOT else bot.perp
        receipt = client.get_order(plan.symbol, order.client_order_id, include_base_commission=False)
        _require(isinstance(receipt, OrderReceipt) and bot.coordinator._matches(receipt, order)
                 and _signature(receipt) == _signature(source.cached))
        _require(receipt.status in terminal
                 and (receipt.status is not OrderState.FILLED or receipt.executed_quantity == order.quantity)
                 and (receipt.status is not OrderState.REJECTED or receipt.executed_quantity == 0),
                 AbortAccountingCode.UNRESOLVED_ORDER)
        trades = client.get_order_trades(plan.symbol, receipt.order_id) if receipt.executed_quantity else ()
        _require(isinstance(trades, tuple) and all(isinstance(t, AccountTrade) for t in trades))
        _require(sum((t.quantity for t in trades), Decimal(0)) == receipt.executed_quantity,
                 AbortAccountingCode.INCOMPLETE_FILLS)
        quote = sum((t.quote_quantity for t in trades), Decimal(0))
        _require(receipt.cumulative_quote_quantity is not None and quote == receipt.cumulative_quote_quantity,
                 AbortAccountingCode.INCOMPLETE_FILLS)
        _require(source.cached.cumulative_quote_quantity is None
                 or source.cached.cumulative_quote_quantity == receipt.cumulative_quote_quantity)
        base_fee = sum((t.commission for t in trades if t.commission_asset == plan.symbol[:-4]), Decimal(0))
        if order.market is ExecutionMarket.SPOT:
            _require(source.cached.base_commission == base_fee or not trades and
                     source.cached.base_commission in (None, Decimal(0)))
        for trade in trades:
            _require((trade.market, trade.symbol, trade.order_id, trade.side) ==
                     (order.market, plan.symbol, receipt.order_id, order.side)
                     and trade.quote_quantity == trade.price * trade.quantity and trade.traded_at <= checked_at)
            _require(trade.price <= order.limit_price if order.side.value == "BUY" else trade.price >= order.limit_price)
            _require((trade.market, trade.trade_id) not in seen)
            if source.leg is TradeLeg.PERP_ENTRY:
                _require(trade.official_perp_pnl == 0)
            seen.add((trade.market, trade.trade_id))
            grouped[source.leg].append(trade)
    evidence = []
    for leg, trade_list in grouped.items():
        qty = sum((t.quantity for t in trade_list), Decimal(0))
        quote = sum((t.quote_quantity for t in trade_list), Decimal(0))
        commissions: dict[str, Decimal] = {}
        for trade in trade_list:
            commissions[trade.commission_asset] = commissions.get(trade.commission_asset, Decimal(0)) + trade.commission
        evidence.append(LegTradeEvidence(leg, tuple(trade_list), qty, quote, None if qty == 0 else quote / qty,
                                         tuple(sorted(commissions.items()))))
    by_leg = {item.leg: item for item in evidence}
    se, sx, pe, px = (by_leg[leg] for leg in (TradeLeg.SPOT_ENTRY, TradeLeg.SPOT_EXIT,
                                            TradeLeg.PERP_ENTRY, TradeLeg.PERP_EXIT))
    base_fees = sum((t.commission for item in (se, sx) for t in item.trades
                     if t.commission_asset == plan.symbol[:-4]), Decimal(0))
    _require(se.quantity - sx.quantity - base_fees == 0 and pe.quantity == px.quantity,
             AbortAccountingCode.INVENTORY_CONFLICT)
    for entry, exit_leg in ((se, sx), (pe, px)):
        if entry.trades and exit_leg.trades:
            _require(max(t.traded_at for t in entry.trades) <= min(t.traded_at for t in exit_leg.trades))
    pair = bot.journal.get(plan.attempt_id)
    assert pair is not None
    _require((pair.state is PairExecutionState.FAILED) == (se.quantity == pe.quantity == 0))
    payments: tuple[FundingPayment, ...] = ()
    total: Decimal | None = Decimal(0) if pe.quantity == 0 else None
    entered = min((t.traded_at for t in pe.trades), default=None)
    exited = max((t.traded_at for t in px.trades), default=None)
    if entered is not None and exited is not None and entered < exited:
        payments = bot.perp.get_funding_payments(plan.symbol, entered, exited)
        _require(isinstance(payments, tuple) and all(isinstance(p, FundingPayment) for p in payments)
                 and len({p.transaction_id for p in payments}) == len(payments)
                 and all(p.symbol == plan.symbol and entered < p.posted_at <= exited for p in payments))
        with bot.journal._connect() as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='testnet_funding_payments'").fetchone():
                for payment in payments:
                    owner = db.execute("SELECT attempt_id FROM testnet_funding_payments WHERE transaction_id=?",
                                       (payment.transaction_id,)).fetchone()
                    _require(owner is None or owner[0] == plan.attempt_id)
            if payments and db.execute("SELECT 1 FROM sqlite_master WHERE name='testnet_trade_reconciliations'").fetchone():
                ids = {p.transaction_id for p in payments}
                for row in db.execute("SELECT report_json FROM testnet_trade_reconciliations WHERE attempt_id<>?",
                                      (plan.attempt_id,)):
                    archived = json.loads(row[0])
                    _require(not any(p["transaction_id"] in ids for p in archived.get("funding_payments", [])))
        if payments:
            total = sum((p.amount for p in payments), Decimal(0))
        if plan.funding_settlement_at is not None and entered < plan.funding_settlement_at <= exited:
            _require(total is not None, AbortAccountingCode.FUNDING_PENDING)
    return TradeReconciliation(plan.attempt_id, plan.symbol, checked_at, ReconciliationStatus.INCOMPLETE, (),
                               legs=tuple(evidence), funding_payments=payments, funding_total=total,
                               spot_inventory_change=Decimal(0), entry_time=entered, exit_time=exited,
                               accounting_scope=ABORT_SCOPE)


def reconcile_abort_balance_evidence(bot: "TestnetBot", directory: Path,
                                     attempt_id: str) -> BalanceFollowupResult:
    """Read-only proof followed by independent archives; original pair is untouched."""
    try:
        plans = [p for p, _ in bot.journal.plans() if p.attempt_id == attempt_id]
        if len(plans) != 1 or not supported_abort(bot.journal.get(attempt_id)):
            return BalanceFollowupResult(BalanceFollowupStatus.SKIPPED, attempt_id, error_code="ABORT_NOT_SUPPORTED")
        plan = plans[0]
        fingerprint = journal_fingerprint(bot.journal, plan)
        markers = [i for i in BalanceCheckRepository(bot.journal).before_intents() if i.attempt_id == attempt_id]
        before = BalanceFrameStore(directory).get(attempt_id, plan.symbol, plan.spot_entry.quantity)
        _require(len(markers) == 1 and before.success and before.frame is not None)
        assert before.frame is not None
        from crypto_arbitrage_bot.execution.balance_ledger import frame_digest
        _require((markers[0].symbol, markers[0].quantity) == (plan.symbol, plan.spot_entry.quantity))
        _require(frame_digest(markers[0], before.frame.snapshots) == markers[0].frame_sha256)
        report = _fill_report(bot, plan)
        states = [bot.spot.get_account_state(plan.symbol)]
        states.extend(bot.perp.get_account_state(symbol) for symbol in SUPPORTED_SYMBOLS)
        _require(states[0].market is ExecutionMarket.SPOT and states[0].symbol == plan.symbol
                 and states[0].can_trade and states[0].open_order_count == 0, AbortAccountingCode.ACCOUNT_CONFLICT)
        for state, symbol in zip(states[1:], SUPPORTED_SYMBOLS, strict=True):
            _require(state.market is ExecutionMarket.USDTM and state.symbol == symbol
                     and state.can_trade and state.one_way and state.single_asset and state.leverage == 1
                     and state.position_quantity == 0 and state.open_order_count == 0, AbortAccountingCode.ACCOUNT_CONFLICT)
        baseline = next((b.amount for s in before.frame.snapshots if s.market is ExecutionMarket.SPOT
                         for b in s.balances if b.asset == plan.symbol[:-4]), None)
        _require(baseline is not None and states[0].base_available == baseline, AbortAccountingCode.INVENTORY_CONFLICT)
        saved = ReconciliationRepository(bot.journal).save(report)
        if not saved.success:
            return BalanceFollowupResult(BalanceFollowupStatus.FAILED, attempt_id,
                                         error_code=saved.error_code.value if saved.error_code else "ACCOUNTING_SAVE_FAILED")
        after = tuple(capture_balance_snapshot(c, clock=bot.clock) for c in (bot.spot, bot.perp))
        audit = audit_balance_changes(before.frame.snapshots, after,
                                      tuple(t for leg in report.legs for t in leg.trades), report.funding_payments,
                                      funding_evidence_complete=report.funding_total is not None)
        archived = BalanceAuditRepository(bot.journal).save(attempt_id, audit)
        if not archived.success or archived.audit_id is None:
            return BalanceFollowupResult(BalanceFollowupStatus.FAILED, attempt_id,
                                         error_code=archived.error_code.value if archived.error_code else "BALANCE_SAVE_FAILED")
        proof = AbortBalanceProofRepository(bot.journal).save(plan, fingerprint, markers[0].frame_sha256,
                                                            archived.audit_id, tuple(states))
        return BalanceFollowupResult(BalanceFollowupStatus.ARCHIVED if proof.success else BalanceFollowupStatus.FAILED,
                                     attempt_id, archived.audit_id if proof.success else None,
                                     audit.status.value, proof.error_code)
    except Exception as error:
        code = (error.code.value if isinstance(error, AbortAccountingError) else
                error.kind.value if isinstance(error, TestnetExecutionError) else type(error).__name__)
        return BalanceFollowupResult(BalanceFollowupStatus.FAILED, attempt_id, error_code=code,
                                     diagnostics=error.diagnostic_fields() if isinstance(error, TestnetExecutionError) else None)
