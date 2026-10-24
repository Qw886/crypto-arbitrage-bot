"""Durable leases, retries, ownership and immutable before markers."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from test_balance_runner_integration import bot as bot
from test_balance_runner_integration import closed as closed
from test_trade_accounting import source_rows

from crypto_arbitrage_bot.execution.balance_accounting import reconcile_balance_evidence
from crypto_arbitrage_bot.execution.balance_check_repository import (
    BalanceCheckRepository,
    BalanceCheckStatus,
)
from crypto_arbitrage_bot.execution.balance_frame_store import BalanceFrameStore
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState


def test_schema_initialization_idempotent_and_parent_unchanged(closed):
    instance, _, plan, _, _, _ = closed
    original = source_rows(instance)
    repo = BalanceCheckRepository(instance.journal)
    repo.initialize()
    repo.initialize()
    assert repo.get(plan.attempt_id) is None and source_rows(instance) == original
    marker = repo.before_intents()[0]
    assert marker.quantity == Decimal('.01') and len(marker.frame_sha256) == 64


def test_claim_active_lease_and_restart_recovery(closed):
    instance, _, plan, _, _, _ = closed
    now = instance.clock()
    repo = BalanceCheckRepository(instance.journal)
    first = repo.claim(plan.attempt_id, now, max_attempts=3, lease_seconds=300)
    assert first.attempts == 1
    assert repo.claim(plan.attempt_id, now, max_attempts=3, lease_seconds=300) is None
    restarted = BalanceCheckRepository(instance.journal)
    assert restarted.claim(plan.attempt_id, now + timedelta(seconds=299), max_attempts=3, lease_seconds=300) is None
    second = restarted.claim(plan.attempt_id, now + timedelta(seconds=300), max_attempts=3, lease_seconds=300)
    assert second.attempts == 2 and second.token != first.token
    assert not repo.finish(first, now, max_attempts=3, retry_seconds=60, failure_code='OLD_OWNER')
    assert restarted.get(plan.attempt_id).token == second.token


def test_pending_interval_and_attempt_budget_survive_restart(closed):
    instance, _, plan, _, _, _ = closed
    now = instance.clock()
    repo = BalanceCheckRepository(instance.journal)
    first = repo.claim(plan.attempt_id, now, max_attempts=2, lease_seconds=300)
    assert repo.finish(first, now, max_attempts=2, retry_seconds=60, failure_code='TIMEOUT')
    assert repo.get(plan.attempt_id).status is BalanceCheckStatus.PENDING
    restarted = BalanceCheckRepository(instance.journal)
    assert restarted.claim(plan.attempt_id, now + timedelta(seconds=59), max_attempts=2, lease_seconds=300) is None
    second = restarted.claim(plan.attempt_id, now + timedelta(seconds=60), max_attempts=2, lease_seconds=300)
    assert second.attempts == 2
    assert restarted.finish(second, now, max_attempts=2, retry_seconds=60, failure_code='TIMEOUT')
    assert restarted.get(plan.attempt_id).status is BalanceCheckStatus.FAILED
    assert restarted.claim(plan.attempt_id, now + timedelta(days=1), max_attempts=2, lease_seconds=300) is None


def test_crash_at_last_attempt_does_not_reset_budget(closed):
    instance, _, plan, _, _, _ = closed
    now = instance.clock()
    repo = BalanceCheckRepository(instance.journal)
    assert repo.claim(plan.attempt_id, now, max_attempts=1, lease_seconds=300)
    assert repo.claim(plan.attempt_id, now + timedelta(seconds=300), max_attempts=1, lease_seconds=300) is None
    row = repo.get(plan.attempt_id)
    assert row.status is BalanceCheckStatus.FAILED and row.failure_code == 'CHECK_ATTEMPTS_EXHAUSTED'


def test_archived_evidence_recovery_and_no_reexecution(closed):
    instance, _, plan, _, _, _ = closed
    outcome = reconcile_balance_evidence(instance, instance.balance_directory, plan.attempt_id)
    repo = BalanceCheckRepository(instance.journal)
    assert repo.recover_archive(plan.attempt_id, outcome.audit_id, instance.clock())
    assert repo.recover_archive(plan.attempt_id, outcome.audit_id, instance.clock())
    assert repo.get(plan.attempt_id).status is BalanceCheckStatus.ARCHIVED
    assert repo.claim(plan.attempt_id, instance.clock() + timedelta(days=1), max_attempts=3, lease_seconds=300) is None


def test_recovery_does_not_attach_another_attempts_audit(closed):
    instance, _, plan, _, _, _ = closed
    outcome = reconcile_balance_evidence(instance, instance.balance_directory, plan.attempt_id)
    other = replace(plan, attempt_id='other-attempt')
    assert instance.journal.create(other)
    repo = BalanceCheckRepository(instance.journal)
    assert not repo.recover_archive(other.attempt_id, outcome.audit_id, instance.clock())
    assert repo.get(other.attempt_id) is None


def test_finish_checks_audit_identity_and_balance_status(closed):
    instance, _, plan, _, _, _ = closed
    outcome = reconcile_balance_evidence(instance, instance.balance_directory, plan.attempt_id)
    repo = BalanceCheckRepository(instance.journal)
    claim = repo.claim(plan.attempt_id, instance.clock(), max_attempts=3, lease_seconds=300)
    assert not repo.finish(claim, instance.clock(), max_attempts=3, retry_seconds=60,
                           audit_id='missing', balance_status='MATCHED')
    assert not repo.finish(claim, instance.clock(), max_attempts=3, retry_seconds=60,
                           audit_id=outcome.audit_id, balance_status='MISMATCH')
    assert repo.finish(claim, instance.clock(), max_attempts=3, retry_seconds=60,
                       audit_id=outcome.audit_id, balance_status='MATCHED')


def test_nonclosed_or_missing_parent_not_claimed(closed):
    instance, _, plan, _, _, _ = closed
    repo = BalanceCheckRepository(instance.journal)
    assert repo.claim('missing', instance.clock(), max_attempts=3, lease_seconds=300) is None
    instance.journal.update(plan.attempt_id, state=PairExecutionState.HEDGED)
    assert repo.claim(plan.attempt_id, instance.clock(), max_attempts=3, lease_seconds=300) is None


def test_preentry_marker_idempotent_conflict_immutable(closed):
    instance, _, plan, _, _, _ = closed
    frame = BalanceFrameStore(instance.balance_directory).get(plan.attempt_id, plan.symbol, plan.spot_entry.quantity).frame
    repo = BalanceCheckRepository(instance.journal)
    original = repo.before_intents()
    assert repo.register_before(frame)
    assert not repo.register_before(replace(frame, quantity=Decimal('.02')))
    assert repo.before_intents() == original


def test_marker_cannot_be_invented_after_parent_exists(closed):
    instance, _, plan, _, _, _ = closed
    frame = BalanceFrameStore(instance.balance_directory).get(plan.attempt_id, plan.symbol, plan.spot_entry.quantity).frame
    with instance.journal._connect() as db:
        db.execute('DELETE FROM testnet_balance_frame_intents')
    assert not BalanceCheckRepository(instance.journal).register_before(frame)


@pytest.mark.parametrize('attempts', [0, 11, True])
def test_retry_bounds_rejected(closed, attempts):
    instance, _, plan, _, _, _ = closed
    with pytest.raises(ValueError):
        BalanceCheckRepository(instance.journal).claim(plan.attempt_id, instance.clock(),
                                                      max_attempts=attempts, lease_seconds=300)


@pytest.mark.parametrize('lease', [0, 29, 3601, True])
def test_lease_bounds_rejected(closed, lease):
    instance, _, plan, _, _, _ = closed
    with pytest.raises(ValueError):
        BalanceCheckRepository(instance.journal).claim(plan.attempt_id, instance.clock(), max_attempts=3, lease_seconds=lease)
