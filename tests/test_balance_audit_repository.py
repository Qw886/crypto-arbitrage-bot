"""Workspace SQLite: exact immutable audits linked to existing Demo attempts."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from test_account_balance_audit import run, snapshot
from test_trade_accounting import AT as TRADE_AT
from test_trade_accounting import evidence_bot as evidence_bot
from test_trade_accounting import source_rows

from crypto_arbitrage_bot.execution.account_balance_audit import (
    BalanceAuditStatus as Status,
)
from crypto_arbitrage_bot.execution.balance_audit_repository import (
    BalanceAuditRepository,
)
from crypto_arbitrage_bot.execution.balance_audit_repository import (
    BalancePersistenceCode as Code,
)
from crypto_arbitrage_bot.execution.pair_coordinator import ExecutionJournalError
from crypto_arbitrage_bot.execution.reconciliation_repository import ReconciliationRepository
from crypto_arbitrage_bot.execution.trade_reconciliation import reconcile_closed_attempt
from crypto_arbitrage_bot.execution.types import ExecutionMarket as Market


@pytest.fixture
def example(evidence_bot):
    bot, plan, _, _ = evidence_bot
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert ReconciliationRepository(bot.journal).save(report).success
    before = (snapshot(Market.SPOT, 0, {"USDT": "1000", "ETH": "1", "BNB": "1"}),
              snapshot(Market.USDTM, 0, {"USDT": "1000", "BNB": "1"}))
    after = (snapshot(Market.SPOT, 400, {"USDT": "999.96999", "ETH": "1", "BNB": "1"}),
             snapshot(Market.USDTM, 400, {"USDT": "1000.06994", "BNB": "1"}))
    def moved(s, seconds):
        at = TRADE_AT + timedelta(seconds=seconds)
        return replace(s, requested_at=at, received_at=at)
    return (tuple(moved(s, 0) for s in before), tuple(moved(s, 400) for s in after),
            tuple(t for leg in report.legs for t in leg.trades), report.funding_payments)


def test_initialize_repeat_save_and_exact_roundtrip(evidence_bot, example):
    bot, plan, _, _ = evidence_bot
    before = source_rows(bot)
    repo = BalanceAuditRepository(bot.journal)
    repo.initialize()
    repo.initialize()
    audit = run(example)
    saved = repo.save(plan.attempt_id, audit)
    assert saved.success and repo.save(plan.attempt_id, audit) == saved
    data = repo.get(saved.audit_id)
    assert data["status"] == "MATCHED"
    assert Decimal(data["movements"][-1]["known_event_delta"]) == Decimal(".06994")
    assert data["before"][0]["received_at"].endswith("+00:00")
    assert len(repo.list_for_attempt(plan.attempt_id)) == 1
    assert source_rows(bot) == before


@pytest.mark.parametrize("status", list(Status))
def test_each_audit_status_can_be_archived(evidence_bot, example, status):
    bot, plan, _, _ = evidence_bot
    audit = run(example)
    if status is not Status.MATCHED:
        audit = replace(audit, status=status)
    repo = BalanceAuditRepository(bot.journal)
    saved = repo.save(plan.attempt_id, audit)
    assert saved.success and repo.get(saved.audit_id)["status"] == status.value


def test_multiple_actual_frames_distinguished(evidence_bot, example):
    bot, plan, _, _ = evidence_bot
    repo = BalanceAuditRepository(bot.journal)
    first = run(example)
    later = replace(first, after=tuple(replace(s, requested_at=s.requested_at + timedelta(seconds=1),
                                             received_at=s.received_at + timedelta(seconds=1)) for s in first.after))
    a, b = repo.save(plan.attempt_id, first), repo.save(plan.attempt_id, later)
    assert a.success and b.success and a.audit_id != b.audit_id
    assert len(repo.list_for_attempt(plan.attempt_id)) == 2


def test_same_frame_different_evidence_cannot_overwrite(evidence_bot, example):
    bot, plan, _, _ = evidence_bot
    repo = BalanceAuditRepository(bot.journal)
    first = repo.save(plan.attempt_id, run(example))
    old = repo.get(first.audit_id)
    other = replace(run(example), funding_evidence_complete=False, status=Status.INCOMPLETE)
    rejected = repo.save(plan.attempt_id, other)
    assert rejected.error_code is Code.EVIDENCE_CONFLICT
    assert repo.get(first.audit_id) == old


def test_missing_parent_or_wrong_symbol_rejected(evidence_bot, example):
    bot, plan, _, _ = evidence_bot
    repo = BalanceAuditRepository(bot.journal)
    assert repo.save("missing", run(example)).error_code is Code.MISSING_PARENT
    audit = run(example)
    wrong = replace(audit, trades=(replace(audit.trades[0], symbol="BTCUSDT"), *audit.trades[1:]))
    assert repo.save(plan.attempt_id, wrong).error_code is Code.SYMBOL_MISMATCH
    assert repo.list_for_attempt(plan.attempt_id) == ()


def test_fk_enabled_on_write_connections(evidence_bot, example, monkeypatch):
    bot, plan, _, _ = evidence_bot
    connect = bot.journal._connect
    flags = []
    @contextmanager
    def observed():
        with connect() as db:
            yield db
            flags.append(db.execute("PRAGMA foreign_keys").fetchone()[0])
    monkeypatch.setattr(bot.journal, "_connect", observed)
    assert BalanceAuditRepository(bot.journal).save(plan.attempt_id, run(example)).success
    assert flags == [1, 1]


def test_write_failure_never_changes_original_reports_or_execution(evidence_bot, example, monkeypatch):
    bot, plan, _, _ = evidence_bot
    initial = source_rows(bot)
    report = run(example)
    original = repr(report)
    @contextmanager
    def failed():
        raise ExecutionJournalError("mock locked SQLite")
        yield
    with monkeypatch.context() as scoped:
        scoped.setattr(bot.journal, "_connect", failed)
        result = BalanceAuditRepository(bot.journal).save(plan.attempt_id, report)
    assert result.error_code is Code.PERSISTENCE_FAILED
    assert repr(report) == original and source_rows(bot) == initial


def test_null_observation_kept_null_in_storage(evidence_bot, example):
    bot, plan, _, _ = evidence_bot
    before, after, trades, funding = example
    after = (replace(after[0], balances=tuple(b for b in after[0].balances if b.asset != "ETH")), after[1])
    repo = BalanceAuditRepository(bot.journal)
    saved = repo.save(plan.attempt_id, run((before, after, trades, funding)))
    data = repo.get(saved.audit_id)
    assert next(m for m in data["movements"] if m["asset"] == "ETH")["observed_delta"] is None


def test_fake_matched_with_missing_proof_rejected(evidence_bot, example):
    bot, plan, _, _ = evidence_bot
    report = replace(run(example), funding_evidence_complete=False)
    result = BalanceAuditRepository(bot.journal).save(plan.attempt_id, report)
    assert result.error_code is Code.EVIDENCE_CONFLICT


def test_different_same_symbol_fill_cannot_attach_to_attempt(evidence_bot, example):
    bot, plan, _, _ = evidence_bot
    audit = run(example)
    bad = replace(audit, trades=(replace(audit.trades[0], commission=Decimal("99")), *audit.trades[1:]))
    assert BalanceAuditRepository(bot.journal).save(plan.attempt_id, bad).error_code is Code.EVIDENCE_CONFLICT


def test_nonarchived_trade_rejected(evidence_bot, example):
    bot, plan, _, _ = evidence_bot
    audit = run(example)
    bad = replace(audit, trades=(replace(audit.trades[0], trade_id="999999"), *audit.trades[1:]))
    assert BalanceAuditRepository(bot.journal).save(plan.attempt_id, bad).error_code is Code.MISSING_SOURCE_EVIDENCE


def test_different_funding_amount_cannot_override_source(evidence_bot, example):
    bot, plan, _, _ = evidence_bot
    audit = run(example)
    bad = replace(audit, funding=(replace(audit.funding[0], amount=Decimal("999")),))
    assert BalanceAuditRepository(bot.journal).save(plan.attempt_id, bad).error_code is Code.EVIDENCE_CONFLICT
