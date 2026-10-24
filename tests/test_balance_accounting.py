"""Offline pre-plan capture and closed-pair audit integration."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from test_trade_accounting import AT, source_rows
from test_trade_accounting import evidence_bot as evidence_bot

from crypto_arbitrage_bot.execution import balance_accounting as workflow
from crypto_arbitrage_bot.execution.account_balance_audit import AssetBalance, BalanceSnapshot
from crypto_arbitrage_bot.execution.balance_audit_repository import BalanceAuditRepository
from crypto_arbitrage_bot.execution.balance_frame_store import (
    BalanceFrameCode,
    BalanceFrameStore,
    BeforeBalanceFrame,
    FrameResult,
)
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.trade_reconciliation import reconcile_closed_attempt
from crypto_arbitrage_bot.execution.types import ExecutionMarket as Market


@pytest.fixture
def balance_evidence(evidence_bot, tmp_path):
    bot, plan, _, _ = evidence_bot
    def frame(market, at, amounts):
        return BalanceSnapshot(market, at, at, tuple(AssetBalance(k, Decimal(v)) for k, v in amounts.items()))
    before = (frame(Market.SPOT, AT, {'USDT': '1000', 'ETH': '1'}),
              frame(Market.USDTM, AT, {'USDT': '1000'}))
    after_at = AT + timedelta(seconds=400)
    after = (frame(Market.SPOT, after_at, {'USDT': '999.96999', 'ETH': '1'}),
             frame(Market.USDTM, after_at, {'USDT': '1000.06994'}))
    directory = tmp_path / 'before-frames'
    assert BalanceFrameStore(directory).save(BeforeBalanceFrame(plan.attempt_id, plan.symbol,
                                                               plan.spot_entry.quantity, before)).success
    return directory, before, after


def test_full_workflow_archives_matching_balance_without_changing_source(evidence_bot, balance_evidence, monkeypatch):
    bot, plan, _, _ = evidence_bot
    directory, _, after = balance_evidence
    old = source_rows(bot)
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    snapshot = repr(report)
    monkeypatch.setattr(workflow, 'capture_balance_snapshot', lambda c, **kw: after[0 if c is bot.spot else 1])
    result = workflow.reconcile_balance_evidence(bot, directory, plan.attempt_id, report=report)
    assert result.status is workflow.BalanceFollowupStatus.ARCHIVED and result.balance_status == 'MATCHED'
    archived = BalanceAuditRepository(bot.journal).get(result.audit_id)
    assert archived['before'][0]['balances'][0]['amount'] == '1000'
    assert len(archived['trades']) == 4 and len(archived['funding']) == 1
    assert source_rows(bot) == old and repr(report) == snapshot


def test_restart_reads_same_preentry_frame(evidence_bot, balance_evidence, monkeypatch):
    bot, plan, _, _ = evidence_bot
    directory, _, after = balance_evidence
    monkeypatch.setattr(workflow, 'capture_balance_snapshot', lambda c, **kw: after[0 if c is bot.spot else 1])
    a = workflow.reconcile_balance_evidence(bot, directory, plan.attempt_id)
    b = workflow.reconcile_balance_evidence(bot, directory, plan.attempt_id)
    assert a.status is workflow.BalanceFollowupStatus.ARCHIVED and a.audit_id == b.audit_id
    assert len(BalanceAuditRepository(bot.journal).list_for_attempt(plan.attempt_id)) == 1


def test_missing_before_does_not_capture_new_one_after_close(evidence_bot, tmp_path, monkeypatch):
    bot, plan, _, _ = evidence_bot
    def forbidden(*a, **kw):
        pytest.fail('must not fabricate before frame')
    monkeypatch.setattr(workflow, 'capture_balance_snapshot', forbidden)
    result = workflow.reconcile_balance_evidence(bot, tmp_path, plan.attempt_id)
    assert result.error_code == 'MISSING_FRAME'


def test_open_attempt_not_followed_up(evidence_bot, balance_evidence, monkeypatch):
    bot, plan, _, _ = evidence_bot
    prior = bot.journal.get(plan.attempt_id)
    monkeypatch.setattr(bot.journal, 'get', lambda _: replace(prior, state=PairExecutionState.HEDGED))
    result = workflow.reconcile_balance_evidence(bot, balance_evidence[0], plan.attempt_id)
    assert result.status is workflow.BalanceFollowupStatus.SKIPPED


def test_wrong_accounting_identity_rejected(evidence_bot, balance_evidence):
    bot, plan, _, _ = evidence_bot
    report = replace(reconcile_closed_attempt(bot, plan.attempt_id), attempt_id='wrong')
    result = workflow.reconcile_balance_evidence(bot, balance_evidence[0], plan.attempt_id, report=report)
    assert result.error_code == 'ACCOUNTING_IDENTITY_MISMATCH'


def test_no_funding_proof_remains_incomplete(evidence_bot, balance_evidence, monkeypatch):
    bot, plan, _, _ = evidence_bot
    directory, _, after = balance_evidence
    report = replace(reconcile_closed_attempt(bot, plan.attempt_id), funding_total=None)
    monkeypatch.setattr(workflow, 'capture_balance_snapshot', lambda c, **kw: after[0 if c is bot.spot else 1])
    result = workflow.reconcile_balance_evidence(bot, directory, plan.attempt_id, report=report)
    assert result.status is workflow.BalanceFollowupStatus.ARCHIVED and result.balance_status == 'INCOMPLETE'


def test_after_capture_failure_does_not_modify_execution(evidence_bot, balance_evidence, monkeypatch):
    bot, plan, _, _ = evidence_bot
    old = source_rows(bot)
    def fail(*a, **kw):
        raise TimeoutError()
    monkeypatch.setattr(workflow, 'capture_balance_snapshot', fail)
    result = workflow.reconcile_balance_evidence(bot, balance_evidence[0], plan.attempt_id)
    assert result.status is workflow.BalanceFollowupStatus.FAILED and result.error_code == 'TimeoutError'
    assert source_rows(bot) == old


def test_save_failure_not_execution_reject(evidence_bot, balance_evidence, monkeypatch):
    bot, plan, _, _ = evidence_bot
    directory, _, after = balance_evidence
    old = source_rows(bot)
    monkeypatch.setattr(workflow, 'capture_balance_snapshot', lambda c, **kw: after[0 if c is bot.spot else 1])
    def fail(*a, **kw):
        raise OSError()
    monkeypatch.setattr(BalanceAuditRepository, 'save', fail)
    result = workflow.reconcile_balance_evidence(bot, directory, plan.attempt_id)
    assert result.status is workflow.BalanceFollowupStatus.FAILED and result.error_code == 'OSError'
    assert source_rows(bot) == old


def test_capture_precedes_fresh_plan_and_does_not_submit(evidence_bot, tmp_path, monkeypatch):
    bot, plan, _, _ = evidence_bot
    proposed = replace(plan, attempt_id='new-plan')
    calls = []
    def capture(*a, **kw):
        calls.append('balance')
        return FrameResult(True)
    def make_plan(*a, **kw):
        calls.append('fresh_quote_plan')
        return proposed
    monkeypatch.setattr(BalanceFrameStore, 'capture_before', capture)
    monkeypatch.setattr(bot, '_plan', make_plan)
    assert workflow.plan_with_balance_evidence(bot, tmp_path, plan.symbol, plan.spot_entry.quantity,
                                              proposed.attempt_id) == proposed
    assert calls == ['balance', 'fresh_quote_plan']
    assert bot.journal.get(proposed.attempt_id) is None


def test_pre_capture_failure_blocks_plan(evidence_bot, tmp_path, monkeypatch):
    bot, plan, _, _ = evidence_bot
    monkeypatch.setattr(BalanceFrameStore, 'capture_before', lambda *a, **kw:
                        FrameResult(False, error_code=BalanceFrameCode.CAPTURE_FAILED))
    monkeypatch.setattr(bot, '_plan', lambda *a, **kw: pytest.fail('no new quote/entry on missing evidence'))
    with pytest.raises(ExecutionGuardError, match='BALANCE_PREENTRY_EVIDENCE_FAILED'):
        workflow.plan_with_balance_evidence(bot, tmp_path, plan.symbol, plan.spot_entry.quantity, 'new-plan')


def test_existing_execution_cannot_recapture_before(evidence_bot, tmp_path, monkeypatch):
    bot, plan, _, _ = evidence_bot
    monkeypatch.setattr(BalanceFrameStore, 'capture_before', lambda *a, **kw: pytest.fail('already entered'))
    with pytest.raises(ExecutionGuardError, match='BALANCE_ATTEMPT_ALREADY_EXISTS'):
        workflow.plan_with_balance_evidence(bot, tmp_path, plan.symbol, plan.spot_entry.quantity, plan.attempt_id)
