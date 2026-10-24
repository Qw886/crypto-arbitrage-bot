"""Offline checks for the controlled process handoff; no network credentials."""

import importlib.util
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_funding_acceptance import enter
from test_funding_lifecycle import live_clock_bot as live_clock_bot

from crypto_arbitrage_bot.execution.funding_acceptance import FundingAcceptance
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot


@pytest.fixture
def harness():
    spec = importlib.util.spec_from_file_location("restart_acceptance", Path("work/restart_acceptance.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stop_after_durable_hedged_then_fresh_bot_does_not_reenter(harness, live_clock_bot, tmp_path):
    bot, transport, state, acceptance = enter(live_clock_bot, tmp_path)
    evidence_path = tmp_path / "first.json"
    harness.stop_after_hedged(acceptance, evidence_path)
    assert acceptance.tick() == "HOLDING"
    assert bot.stop.is_set()
    evidence = json.loads(evidence_path.read_text())
    original_pair = bot.journal.get(acceptance.state["attempt_id"])
    assert evidence["status"] == "FIRST_PROCESS_EXITING_HEDGED"
    assert tuple(evidence["entry_ids"]) == harness.entry_ids(original_pair)
    new_bot = Bot(bot.settings, bot.spot, bot.perp, clock=lambda: state["now"])
    resumed = FundingAcceptance(new_bot, acceptance.report_path)
    state["now"] += timedelta(seconds=1)
    assert resumed.tick() == "HOLDING"
    assert not new_bot.stop.is_set()
    assert len(transport.orders) == 2
    assert harness.entry_ids(new_bot.journal.get(original_pair.attempt_id)) == harness.entry_ids(original_pair)


def test_stop_before_entry_does_not_request_or_submit(harness, live_clock_bot, tmp_path):
    from test_funding_acceptance import arm
    bot, transport, _, acceptance = arm(live_clock_bot, tmp_path)
    calls = len(transport.requests)
    harness.stop_after_hedged(acceptance, tmp_path / "first.json")
    assert acceptance.tick() == "WAITING"
    assert not bot.stop.is_set()
    assert len(transport.requests) == calls
    assert transport.orders == {}


def test_non_hedged_handoff_fails_closed(harness, live_clock_bot, tmp_path, monkeypatch):
    _, _, _, acceptance = enter(live_clock_bot, tmp_path)
    monkeypatch.setattr(acceptance.bot.journal, "get", lambda attempt: None)
    acceptance.tick = Mock(return_value="HOLDING")
    harness.stop_after_hedged(acceptance, tmp_path / "first.json")
    with pytest.raises(ExecutionGuardError, match="RESTART_REQUIRES_DURABLE_HEDGED_PAIR"):
        acceptance.tick()
    assert not (tmp_path / "first.json").exists()


@pytest.mark.parametrize("first_code", [0, 1, 130])
def test_supervisor_waits_first_child_before_resuming_same_report(harness, tmp_path, monkeypatch, first_code):
    report = tmp_path / "acceptance.json"
    report.write_text("{}")
    report.with_suffix(".restart.json").write_text("{}")
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        assert kwargs == {"check": False}
        return SimpleNamespace(returncode=first_code if len(calls) == 1 else 0)
    monkeypatch.setattr(harness.subprocess, "run", child)
    args = SimpleNamespace(report=report, config=Path("docs/testnet-funding-acceptance.toml"))
    assert harness.supervise(args) == first_code
    assert len(calls) == (2 if first_code == 0 else 1)
    assert calls[0][-1] == "first"
    if first_code == 0:
        assert calls[1][-1] == "resume"
        assert calls[0][:-1] == calls[1][:-1]


def test_supervisor_missing_baseline_never_launches_child(harness, tmp_path, monkeypatch):
    child = Mock()
    monkeypatch.setattr(harness.subprocess, "run", child)
    args = SimpleNamespace(report=tmp_path / "absent.json", config=Path("docs/testnet-funding-acceptance.toml"))
    with pytest.raises(ExecutionGuardError, match="RESTART_REQUIRES_ARMED_REPORT_AND_BASELINE"):
        harness.supervise(args)
    child.assert_not_called()


def test_missing_receipts_cannot_be_successful_handoff(harness):
    with pytest.raises(ExecutionGuardError, match="RESTART_ENTRY_RECEIPTS_MISSING"):
        harness.entry_ids(None)


def test_atomic_evidence_preserves_decimal_and_does_not_keep_temp(harness, tmp_path):
    from decimal import Decimal
    path = tmp_path / "audit.json"
    harness.save(path, {"value": Decimal("0.12345678901234567890123456789")})
    assert json.loads(path.read_text())["value"] == "0.12345678901234567890123456789"
    assert not path.with_suffix(".json.tmp").exists()
