"""Explicit testnet balance-evidence workflow, never a trading decision.

Capture before constructing a fresh executable plan. After the existing bot
closes its pair, reuse official trade reconciliation and the currency audit.
This helper submits no orders and does not alter existing profit formulas.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from crypto_arbitrage_bot.execution.account_balance_audit import (
    audit_balance_changes,
    capture_balance_snapshot,
)
from crypto_arbitrage_bot.execution.balance_audit_repository import BalanceAuditRepository
from crypto_arbitrage_bot.execution.balance_frame_store import BalanceFrameStore
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState, PairOrderPlan
from crypto_arbitrage_bot.execution.reconciliation_repository import ReconciliationRepository
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError
from crypto_arbitrage_bot.execution.trade_reconciliation import (
    TradeReconciliation,
    reconcile_closed_attempt,
)

if TYPE_CHECKING:
    from crypto_arbitrage_bot.execution.runner import TestnetBot


class BalanceFollowupStatus(StrEnum):
    ARCHIVED = "ARCHIVED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class BalanceFollowupResult:
    status: BalanceFollowupStatus
    attempt_id: str
    audit_id: str | None = None
    balance_status: str | None = None
    error_code: str | None = None
    diagnostics: dict[str, object] | None = None


def plan_with_balance_evidence(bot: "TestnetBot", directory: Path, symbol: str,
                              quantity: Decimal, attempt_id: str) -> PairOrderPlan:
    """One new intent only; failed pre-evidence blocks this optional workflow.

    All additional wallet GETs occur before the original _plan obtains its
    fresh executable snapshot and submission deadline. No stale quote reuse.
    """
    from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
    if bot.journal.get(attempt_id) is not None:
        raise ExecutionGuardError("BALANCE_ATTEMPT_ALREADY_EXISTS")
    before = BalanceFrameStore(directory).capture_before(attempt_id, symbol, quantity,
                                                        bot.spot, bot.perp, clock=bot.clock)
    if not before.success:
        raise ExecutionGuardError("BALANCE_PREENTRY_EVIDENCE_FAILED", diagnostics={
            "error_code": before.error_code.value if before.error_code else None,
            "error_kind": before.error_kind,
            "source_error": before.diagnostics})
    # Existing model, gates, price bounds and financial logic are unchanged.
    plan = bot._plan(symbol, quantity, attempt_id)
    if (plan.attempt_id, plan.symbol, plan.spot_entry.quantity, plan.perp_entry.quantity) != (
        attempt_id, symbol, quantity, quantity
    ):
        raise ExecutionGuardError("BALANCE_PLAN_IDENTITY_MISMATCH")
    return plan


def reconcile_balance_evidence(bot: "TestnetBot", directory: Path, attempt_id: str, *,
                               report: TradeReconciliation | None = None) -> BalanceFollowupResult:
    """Independent failure semantics; no order/decision/Funding inference.

    ARCHIVED can still contain INCOMPLETE or MISMATCH. It is not a success or
    profitability verdict. Prior pre-entry evidence cannot be recaptured.
    """
    try:
        plans = [p for p, _ in bot.journal.plans() if p.attempt_id == attempt_id]
        pair = bot.journal.get(attempt_id)
        if len(plans) != 1 or pair is None or pair.state is not PairExecutionState.CLOSED:
            return BalanceFollowupResult(BalanceFollowupStatus.SKIPPED, attempt_id,
                                         error_code="ATTEMPT_NOT_CLOSED")
        plan = plans[0]
        before = BalanceFrameStore(directory).get(attempt_id, plan.symbol, plan.spot_entry.quantity)
        if not before.success or before.frame is None:
            return BalanceFollowupResult(BalanceFollowupStatus.FAILED, attempt_id,
                                         error_code=before.error_code.value if before.error_code else "INVALID_FRAME")
        report = report if report is not None else reconcile_closed_attempt(bot, attempt_id)
        if report.attempt_id != attempt_id or report.symbol != plan.symbol:
            return BalanceFollowupResult(BalanceFollowupStatus.FAILED, attempt_id,
                                         error_code="ACCOUNTING_IDENTITY_MISMATCH")
        # A transient missing fill must not be archived as a final wallet
        # observation. Funding-only incompleteness retains its NULL semantics.
        if report.status.value == 'INVALID':
            return BalanceFollowupResult(BalanceFollowupStatus.FAILED, attempt_id,
                                         error_code='INVALID_TRADE_ACCOUNTING')
        if len(report.legs) != 4 or any(not leg.trades for leg in report.legs):
            return BalanceFollowupResult(BalanceFollowupStatus.FAILED, attempt_id,
                                         error_code='INCOMPLETE_TRADE_EVIDENCE')
        if plan.funding_settlement_at is not None and report.funding_total is None:
            return BalanceFollowupResult(BalanceFollowupStatus.FAILED, attempt_id,
                                         error_code='FUNDING_EVIDENCE_PENDING')
        archived = ReconciliationRepository(bot.journal).save(report)
        if not archived.success:
            return BalanceFollowupResult(BalanceFollowupStatus.FAILED, attempt_id,
                                         error_code=archived.error_code.value if archived.error_code else "ACCOUNTING_SAVE_FAILED")
        after = tuple(capture_balance_snapshot(c, clock=bot.clock) for c in (bot.spot, bot.perp))
        audit = audit_balance_changes(before.frame.snapshots, after,
            tuple(t for leg in report.legs for t in leg.trades), report.funding_payments,
            funding_evidence_complete=report.funding_total is not None)
        saved = BalanceAuditRepository(bot.journal).save(attempt_id, audit)
        return BalanceFollowupResult(BalanceFollowupStatus.ARCHIVED if saved.success else BalanceFollowupStatus.FAILED,
            attempt_id, saved.audit_id, audit.status.value,
            saved.error_code.value if saved.error_code else None)
    except Exception as error:
        # Read/account/storage errors do not rewrite the original execution.
        return BalanceFollowupResult(BalanceFollowupStatus.FAILED, attempt_id,
                                     error_code=error.kind.value if isinstance(error, TestnetExecutionError) else type(error).__name__,
                                     diagnostics=error.diagnostic_fields() if isinstance(error, TestnetExecutionError) else None)
