"""Waiting for an existing test never submits/restarts an execution attempt."""

import json
from unittest.mock import Mock

import pytest
from test_trade_accounting import evidence_bot as evidence_bot

from crypto_arbitrage_bot.execution.acceptance_accounting import AccountingFollowupStatus as Status
from crypto_arbitrage_bot.execution.acceptance_accounting import reconcile_after_acceptance
from crypto_arbitrage_bot.execution.reconciliation_repository import (
    AccountingSaveResult,
    ReconciliationRepository,
)


def write_report(bot, plan, path, status="WAITING", **updates):
    data = {"sample_type": "TESTNET_CONNECTIVITY_ACCEPTANCE", "attempt_id": plan.attempt_id,
            "symbol": plan.symbol, "journal_path": str(bot.config.journal_path.resolve()), "status": status}
    data.update(updates)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_wait_then_account_after_completion_has_no_orders(evidence_bot, tmp_path, monkeypatch):
    bot, plan, _, _ = evidence_bot
    path = tmp_path / "acceptance.json"
    write_report(bot, plan, path)
    getter = bot.spot.get_order
    spy = Mock(wraps=getter)
    monkeypatch.setattr(bot.spot, "get_order", spy)
    def wait(seconds):
        assert seconds == bot.settings.poll_interval_seconds > 0
        assert spy.call_count == 0
        write_report(bot, plan, path, "VERIFIED")
        return False
    monkeypatch.setattr(bot.stop, "wait", wait)
    result = reconcile_after_acceptance(bot, path)
    assert result.status is Status.SAVED and result.accounting_status == "RECONCILED"
    outcome = json.loads(path.with_suffix(".accounting.json").read_text())
    assert outcome["accounting_summary"]["net_profit"] == "0.039930"
    assert outcome["orders_submitted"] is False
    assert bot.spot.place_ioc_limit_order.call_count == bot.perp.place_ioc_limit_order.call_count == 0


@pytest.mark.parametrize("status", ["ABORTED", "NEEDS_ATTENTION", "CANCELLED"])
def test_failed_acceptance_never_resubmits_or_requests_fills(evidence_bot, tmp_path, monkeypatch, status):
    bot, plan, _, _ = evidence_bot
    path = tmp_path / "acceptance.json"
    write_report(bot, plan, path, status)
    getter = Mock(side_effect=AssertionError("should not fetch fills"))
    monkeypatch.setattr(bot.spot, "get_order", getter)
    assert reconcile_after_acceptance(bot, path).status is Status.SKIPPED
    getter.assert_not_called()


def test_unavailable_funding_stays_incomplete_when_archived(evidence_bot, tmp_path, monkeypatch):
    bot, plan, _, _ = evidence_bot
    path = tmp_path / "acceptance.json"
    write_report(bot, plan, path, "UNAVAILABLE")
    monkeypatch.setattr(bot.perp, "get_funding_payments", Mock(return_value=()))
    result = reconcile_after_acceptance(bot, path)
    assert result.status is Status.SAVED and result.accounting_status == "INCOMPLETE"
    outcome = json.loads(path.with_suffix(".accounting.json").read_text())
    assert outcome["accounting_summary"]["net_profit"] is None


def test_wrong_symbol_blocked_before_network(evidence_bot, tmp_path, monkeypatch):
    bot, plan, _, _ = evidence_bot
    path = tmp_path / "acceptance.json"
    write_report(bot, plan, path, "VERIFIED", symbol="BTCUSDT")
    getter = Mock(side_effect=AssertionError("wrong symbol"))
    monkeypatch.setattr(bot.spot, "get_order", getter)
    assert reconcile_after_acceptance(bot, path).status is Status.INVALID_REPORT
    getter.assert_not_called()


def test_malformed_report_safely_rejected(evidence_bot, tmp_path):
    bot, _, _, _ = evidence_bot
    path = tmp_path / "acceptance.json"
    path.write_text("invalid JSON", encoding="utf-8")
    assert reconcile_after_acceptance(bot, path).status is Status.INVALID_REPORT


def test_stop_waiting_does_not_modify_acceptance(evidence_bot, tmp_path, monkeypatch):
    bot, plan, _, _ = evidence_bot
    path = tmp_path / "acceptance.json"
    write_report(bot, plan, path)
    original = path.read_bytes()
    def wait(seconds):
        bot.stop.set()
        return True
    monkeypatch.setattr(bot.stop, "wait", wait)
    assert reconcile_after_acceptance(bot, path).status is Status.STOPPED
    assert path.read_bytes() == original


def test_monotonic_wait_deadline_is_bounded(evidence_bot, tmp_path, monkeypatch):
    bot, _, _, _ = evidence_bot
    times = iter((0, 2_000_000_000))
    monkeypatch.setattr("crypto_arbitrage_bot.execution.acceptance_accounting.monotonic_ns", lambda: next(times))
    assert reconcile_after_acceptance(bot, tmp_path / "absent.json", max_wait_seconds=1).status is Status.TIMED_OUT


def test_persistence_failure_does_not_change_execution_report(evidence_bot, tmp_path, monkeypatch):
    bot, plan, _, _ = evidence_bot
    path = tmp_path / "acceptance.json"
    write_report(bot, plan, path, "VERIFIED")
    original = path.read_bytes()
    monkeypatch.setattr(ReconciliationRepository, "save", Mock(return_value=AccountingSaveResult(False)))
    assert reconcile_after_acceptance(bot, path).status is Status.FAILED
    assert path.read_bytes() == original
