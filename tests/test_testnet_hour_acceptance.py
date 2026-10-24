"""Offline one-hour supervisor tests; no waiting or exchange connections."""

import importlib.util
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_funding_lifecycle import live_clock_bot as live_clock_bot

from crypto_arbitrage_bot.execution.runner import ExecutionGuardError


@pytest.fixture
def hour(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("work").resolve()))
    spec = importlib.util.spec_from_file_location("hour_acceptance", Path("work/testnet_hour_acceptance.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_session(hour, monkeypatch, *, first_status="PASS", complete=True, loop_ms=1800000):
    calls = []
    def run(bot, path, *, duration_seconds, entry_symbol):
        calls.append((bot, duration_seconds, entry_symbol))
        status = first_status if len(calls) == 1 else "PASS"
        hour.save(path, {"status": status, "attempt_id": entry_symbol,
                        "duration_deadline_reached": complete, "loop_elapsed_ms": loop_ms,
                        "elapsed_ms": 1900000})
        return 0 if status == "PASS" else 1
    monkeypatch.setattr(hour, "run_acceptance", run)
    return calls


def test_two_bounded_distinct_sessions(hour, tmp_path, monkeypatch):
    calls = fake_session(hour, monkeypatch)
    factory = Mock(side_effect=[object(), object()])
    path = tmp_path / "hour.json"
    assert hour.run_hour(factory, path) == 0
    assert [(seconds, symbol) for _, seconds, symbol in calls] == [(1800, "BTCUSDT"), (1800, "ETHUSDT")]
    assert calls[0][0] is not calls[1][0]
    result = json.loads(path.read_text())
    assert result["status"] == "PASS" and len(result["sessions"]) == 2
    assert result["new_entry_limit"] == 2 and result["decision_bypassed"]


@pytest.mark.parametrize("status,complete,loop_ms", [
    ("NEEDS_ATTENTION", True, 1800000), ("PASS", False, 1800000), ("PASS", True, 1700000),
])
def test_failed_or_early_session_blocks_next_entry(hour, tmp_path, monkeypatch, status, complete, loop_ms):
    calls = fake_session(hour, monkeypatch, first_status=status, complete=complete, loop_ms=loop_ms)
    path = tmp_path / "hour.json"
    assert hour.run_hour(Mock(return_value=object()), path) == 1
    assert len(calls) == 1
    assert json.loads(path.read_text())["status"] == "NEEDS_ATTENTION"


@pytest.mark.parametrize("existing", ["hour.json", "hour-BTCUSDT.json", "hour-ETHUSDT.json"])
def test_existing_reports_block_all_entries(hour, tmp_path, existing):
    (tmp_path / existing).write_text("{}")
    factory = Mock()
    with pytest.raises(ExecutionGuardError):
        hour.run_hour(factory, tmp_path / "hour.json")
    factory.assert_not_called()


def test_exception_no_retry_and_no_private_message(hour, tmp_path, monkeypatch):
    run = Mock(side_effect=TimeoutError("credentials must not appear"))
    monkeypatch.setattr(hour, "run_acceptance", run)
    path = tmp_path / "hour.json"
    assert hour.run_hour(Mock(return_value=object()), path) == 1
    assert run.call_count == 1
    assert "credentials" not in path.read_text()
    assert json.loads(path.read_text())["error_type"] == "TimeoutError"


def test_main_disabled_gate_no_account_or_orders(hour, tmp_path, monkeypatch):
    factory = Mock()
    monkeypatch.setattr(hour, "create_testnet_bot", factory)
    monkeypatch.setattr(hour, "os", SimpleNamespace(environ={}))
    monkeypatch.setattr("sys.argv", ["tool", "--report", str(tmp_path / "hour.json")])
    assert hour.main() == 1
    factory.assert_not_called()


@pytest.mark.parametrize("symbol,quantity", [("BTCUSDT", ".001"), ("ETHUSDT", ".01")])
def test_controlled_entry_reuses_original_plan(hour, live_clock_bot, tmp_path, symbol, quantity):
    from short_restart_acceptance import first_process
    bot, transport, _ = live_clock_bot
    path = tmp_path / "entry.json"
    assert first_process(bot, path, entry_symbol=symbol) == 0
    plan, _ = bot.journal.plans()[0]
    assert plan.symbol == symbol and plan.spot_entry.quantity == Decimal(quantity)
    assert len(transport.orders) == 2 and json.loads(path.read_text())["decision_bypassed"]
    assert plan.funding_settlement_at is plan.scheduled_exit_at is None


def test_unsupported_symbol_never_orders(hour, live_clock_bot, tmp_path):
    from short_restart_acceptance import first_process
    bot, transport, _ = live_clock_bot
    with pytest.raises(ExecutionGuardError, match="UNSUPPORTED_SYMBOL"):
        first_process(bot, tmp_path / "entry.json", entry_symbol="BNBUSDT")
    assert transport.orders == {}


def test_btc_quantity_guard(hour, live_clock_bot, tmp_path):
    from short_restart_acceptance import first_process
    bot, transport, _ = live_clock_bot
    bot.research.sizing["BTCUSDT"] = bot.research.sizing["BTCUSDT"].model_copy(update={"target_quantity": Decimal(".002")})
    with pytest.raises(ExecutionGuardError, match="EXPLICIT_BTC"):
        first_process(bot, tmp_path / "entry.json", entry_symbol="BTCUSDT")
    assert transport.orders == {}
