"""One optional closed-pair wallet check per round; all exits take priority."""

import hashlib
from decimal import Decimal
from typing import TYPE_CHECKING

from crypto_arbitrage_bot.execution.abort_balance_proof import (
    AbortBalanceProofRepository,
    supported_abort,
)
from crypto_arbitrage_bot.execution.balance_accounting import reconcile_balance_evidence
from crypto_arbitrage_bot.execution.balance_audit_repository import BalanceAuditRepository
from crypto_arbitrage_bot.execution.balance_check_repository import (
    BalanceCheckRepository,
    BalanceCheckStatus,
    BeforeBalanceIntent,
)
from crypto_arbitrage_bot.execution.balance_frame_store import BalanceFrameCode, BalanceFrameStore
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState
from crypto_arbitrage_bot.storage.opportunity_records import exact_json

if TYPE_CHECKING:
    from crypto_arbitrage_bot.execution.runner import TestnetBot

TERMINAL = frozenset({PairExecutionState.CLOSED, PairExecutionState.FLATTENED, PairExecutionState.FAILED})
TERMINAL_ERRORS = frozenset({BalanceFrameCode.MISSING_FRAME.value, BalanceFrameCode.INVALID_FRAME.value,
    BalanceFrameCode.IDENTITY_MISMATCH.value, 'FRAME_HASH_MISMATCH', 'PLAN_IDENTITY_MISMATCH',
    'INVALID_TRADE_ACCOUNTING', 'ACCOUNTING_IDENTITY_MISMATCH', 'EVIDENCE_CONFLICT',
    'ABORT_EVIDENCE_CONFLICT'})


def frame_digest(intent: BeforeBalanceIntent, before: object) -> str:
    return hashlib.sha256(exact_json({'attempt_id': intent.attempt_id, 'symbol': intent.symbol,
        'quantity': str(intent.quantity), 'snapshots': before, 'schema_version': 1}).encode('utf-8')).hexdigest()


def balance_entry_blocker(bot: "TestnetBot") -> str | None:
    """Local-only execution safety, never changes the research Decision.

    Isolate account windows: no overlapping pair while balance evidence is
    enabled. A known, fully explained Funding-only INCOMPLETE audit is allowed
    to remain research evidence; it is not called reconciled profit.
    """
    plans = bot.journal.plans()
    if any((p := bot.journal.get(plan.attempt_id)) is None or p.state not in TERMINAL for plan, _ in plans):
        return 'BALANCE_WINDOW_REQUIRES_FLAT_JOURNAL'
    checks, audits = BalanceCheckRepository(bot.journal), BalanceAuditRepository(bot.journal)
    marked = {i.attempt_id: i for i in checks.before_intents()}
    for plan, _ in plans:
        if plan.attempt_id not in marked:
            continue  # Legacy attempts have no promised pre-entry evidence.
        pair = bot.journal.get(plan.attempt_id)
        if pair is None or pair.state is not PairExecutionState.CLOSED:
            proof = (AbortBalanceProofRepository(bot.journal).get_verified(plan, marked[plan.attempt_id].frame_sha256)
                     if supported_abort(pair) else None)
            if proof is None:
                return 'BALANCE_NON_CLOSED_ATTEMPT_REQUIRES_REVIEW'
            check = checks.get(plan.attempt_id)
            if check is None or check.audit_id != proof['audit_id']:
                return 'BALANCE_CHECK_UNRESOLVED'
        check = checks.get(plan.attempt_id)
        if check is None or check.status is not BalanceCheckStatus.ARCHIVED or check.audit_id is None:
            return 'BALANCE_CHECK_UNRESOLVED'
        audit = audits.get(check.audit_id)
        if audit is None or audit['status'] not in ('MATCHED', 'INCOMPLETE') or not audit['movements']:
            return 'BALANCE_AUDIT_UNRESOLVED'
        if any(i['code'] != 'FUNDING_EVIDENCE_INCOMPLETE' for i in audit['issues']) or any(
            m['unexplained_delta'] is None or abs(Decimal(m['unexplained_delta'])) > Decimal(m['tolerance'])
            for m in audit['movements']
        ):
            return 'BALANCE_AUDIT_UNEXPLAINED_CHANGE'
    return None


def reconcile_balance_ledger(bot: "TestnetBot") -> None:
    from crypto_arbitrage_bot.execution.abort_balance_accounting import (
        reconcile_abort_balance_evidence,
    )
    if not bot.config.balance_evidence_enabled or bot.stop.is_set():
        return
    plans = bot.journal.plans()
    if any((p := bot.journal.get(plan.attempt_id)) is None or p.state not in TERMINAL for plan, _ in plans):
        return  # Even non-overdue exposure precedes optional wallet GETs.
    checks, audits = BalanceCheckRepository(bot.journal), BalanceAuditRepository(bot.journal)
    intents = {i.attempt_id: i for i in checks.before_intents()}
    store = BalanceFrameStore(bot.balance_directory)
    for plan, _ in plans:
        intent = intents.get(plan.attempt_id)
        pair = bot.journal.get(plan.attempt_id)
        if intent is None or pair is None or (pair.state is not PairExecutionState.CLOSED and not supported_abort(pair)):
            continue
        prior = checks.get(plan.attempt_id)
        if prior is not None and prior.status in (BalanceCheckStatus.ARCHIVED, BalanceCheckStatus.FAILED):
            continue
        # Recover a committed full audit without any account/trade GET. Its
        # before frame must match the immutable pre-entry marker exactly.
        recovered = False
        abort_proof = (AbortBalanceProofRepository(bot.journal).get_verified(plan, intent.frame_sha256)
                       if supported_abort(pair) else None)
        for row in audits.list_for_attempt(plan.attempt_id):
            if supported_abort(pair) and (abort_proof is None or row['audit_id'] != abort_proof['audit_id']):
                continue
            archive = audits.get(row['audit_id'])
            if archive is not None and frame_digest(intent, archive['before']) == intent.frame_sha256:
                recovered = checks.recover_archive(plan.attempt_id, row['audit_id'], bot.clock())
                break
        if recovered:
            bot.logger.info(exact_json({'stage': 'TESTNET_BALANCE_LEDGER', 'attempt_id': plan.attempt_id,
                'status': 'ARCHIVED', 'reason': 'RECOVERED_COMMITTED_AUDIT', 'account_requests': 0}))
            continue
        claim = checks.claim(plan.attempt_id, bot.clock(), max_attempts=bot.config.balance_check_max_attempts,
                             lease_seconds=bot.config.balance_check_lease_seconds)
        if claim is None:
            continue
        before = store.get(plan.attempt_id, intent.symbol, intent.quantity)
        failure = None
        if (intent.symbol, intent.quantity) != (plan.symbol, plan.spot_entry.quantity):
            failure = 'PLAN_IDENTITY_MISMATCH'
        elif not before.success or before.frame is None:
            failure = before.error_code.value if before.error_code else 'INVALID_FRAME'
        elif frame_digest(intent, before.frame.snapshots) != intent.frame_sha256:
            failure = 'FRAME_HASH_MISMATCH'
        workflow = reconcile_abort_balance_evidence if supported_abort(pair) else reconcile_balance_evidence
        result = None if failure else workflow(bot, bot.balance_directory, plan.attempt_id)
        if result is not None:
            failure = result.error_code
        archived = result is not None and result.status.value == 'ARCHIVED'
        finished = checks.finish(claim, bot.clock(), max_attempts=bot.config.balance_check_max_attempts,
            retry_seconds=bot.config.balance_check_interval_seconds,
            audit_id=result.audit_id if archived and result is not None else None,
            balance_status=result.balance_status if result is not None else None,
            failure_code=failure, terminal_failure=failure in TERMINAL_ERRORS)
        bot.logger.info(exact_json({'stage': 'TESTNET_BALANCE_LEDGER', 'attempt_id': plan.attempt_id,
            'status': 'ARCHIVED' if archived and finished else 'CHECK_FAILED', 'attempts': claim.attempts,
            'balance_status': result.balance_status if result else None, 'failure_code': failure,
            'source_error': result.diagnostics if result else None,
            'checkpoint_saved': finished, 'meaning': 'Account currency evidence, not profit or Decision'}))
        return  # At most one bounded account workflow per poll.
