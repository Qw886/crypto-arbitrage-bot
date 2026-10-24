"""Bounded offline soak and final read-only/failure acceptance."""

import ast
import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_collection_failures import collect, inject, install
from test_collection_failures import transport as transport
from test_forward_validation import OPENED, shift_fixture
from test_runner import make_runner

from crypto_arbitrage_bot.config.research import ValidationSchedulerSettings
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.funding_history import FundingWindow
from crypto_arbitrage_bot.runner import CycleStage, CycleStatus
from crypto_arbitrage_bot.soak import SoakLimits, main, run_soak
from crypto_arbitrage_bot.storage.forward_validation_repository import (
    SQLiteForwardValidationRepository,
)
from crypto_arbitrage_bot.storage.validation_tasks import SQLiteValidationTaskRepository
from crypto_arbitrage_bot.validation_scheduler import ValidationScheduler


class FakeStop:
    def __init__(self):
        self.seconds = 0
        self.stopped = False
        self.waits = []

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, duration):
        assert duration > 0
        self.waits.append(duration)
        self.seconds += duration
        return self.stopped


def test_three_round_offline_soak_runner_scheduler_sqlite_restart(tmp_path, monkeypatch):
    runner, assemblies, _ = make_runner(tmp_path, monkeypatch)
    stop = FakeStop()
    runner.stop_signal = stop
    runner.clock = lambda: OPENED + timedelta(seconds=stop.seconds)
    runner.collector.collect.side_effect = lambda s: shift_fixture(assemblies[s], timedelta(seconds=stop.seconds))
    funding = Mock()
    funding.fetch_window.side_effect = lambda s, o, c: FundingWindow(s, o, c)
    runner.scheduler = ValidationScheduler(
        ValidationSchedulerSettings(validation_horizons_seconds=(5, 10)), runner.repository,
        SQLiteForwardValidationRepository(runner.repository.config),
        SQLiteValidationTaskRepository(runner.repository.config), runner.collector, funding,
        clock=runner.clock, stopped=stop.is_set, logger=Mock())
    summary = run_soak(runner, SoakLimits(30, 3), clock_ns=lambda: int(stop.seconds * 1_000_000_000))
    assert (summary.rounds, summary.saved_cycles, summary.failed_cycles) == (3, 6, 0)
    assert summary.stopped and summary.unexpected_round_errors == 0
    assert stop.waits == [5, 5] and summary.elapsed_ns == 10_000_000_000
    assert 0 <= summary.traced_end_bytes <= summary.traced_peak_bytes
    with closing(sqlite3.connect(runner.repository.config.database_path)) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
        assert db.execute("SELECT count(*) FROM opportunities").fetchone()[0] == 6
        assert db.execute("SELECT count(*) FROM validation_tasks").fetchone()[0] == 12
        assert db.execute("SELECT count(*) FROM forward_validations").fetchone()[0] == 6
        assert not db.execute("SELECT symbol,snapshot_assembled_at,count(*) FROM opportunities "
                              "GROUP BY symbol,snapshot_assembled_at HAVING count(*)>1").fetchall()
        assert not db.execute("SELECT opportunity_id,horizon_seconds,count(*) FROM validation_tasks "
                              "GROUP BY opportunity_id,horizon_seconds HAVING count(*)>1").fetchall()
    prior = runner.scheduler.validations.list_recent_validations()
    restarted = ValidationScheduler(
        runner.scheduler.settings, runner.repository, runner.scheduler.validations,
        SQLiteValidationTaskRepository(runner.repository.config), runner.collector, funding,
        clock=runner.clock, logger=Mock())
    restarted.process_due()
    assert restarted.validations.list_recent_validations() == prior


@pytest.mark.parametrize("failure", ["connect_timeout", "read_timeout", "429", "503", "invalid_json"])
def test_real_adapters_spot_success_perp_failure_has_no_snapshot(transport, failure):
    connections = install(transport)
    inject(connections[1], failure)
    assembly = collect()
    assert assembly.sources.spot_depth.ok
    assert not assembly.sources.perp_depth.ok
    assert not assembly.complete and assembly.snapshot is None


def test_real_sqlite_lock_is_safe_and_next_round_recovers(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path, monkeypatch)
    runner.repository.initialize()
    with closing(sqlite3.connect(runner.repository.config.database_path, timeout=0)) as lock:
        lock.execute("BEGIN IMMEDIATE")
        results = runner.run_once()
        assert all(c.status is CycleStatus.FAILED and c.failure_stage is CycleStage.PERSISTENCE for c in results)
        assert all(c.opportunity_id is None for c in results)
        assert runner.repository.list_recent_opportunities() == ()
        lock.rollback()
    assert all(c.status is CycleStatus.SAVED for c in runner.run_once())
    assert len(runner.repository.list_recent_opportunities()) == 2


def test_soak_failure_rounds_wait_and_stop_at_duration(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path, monkeypatch)
    stop = FakeStop()
    runner.stop_signal = stop
    runner.run_once = Mock(side_effect=RuntimeError("fault injection"))
    result = run_soak(runner, SoakLimits(7, 100), clock_ns=lambda: int(stop.seconds * 1_000_000_000))
    assert result.rounds == result.unexpected_round_errors == 2
    assert result.stopped and stop.waits == [5, 2]
    assert result.elapsed_ns == 7_000_000_000


def test_soak_interrupt_and_already_stopped(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path, monkeypatch)
    runner.run_once = Mock(side_effect=KeyboardInterrupt)
    assert run_soak(runner, SoakLimits(10, 3)).stopped
    assert run_soak(runner, SoakLimits(10, 3)).rounds == 0
    runner.run_once.assert_called_once()


@pytest.mark.parametrize("duration,cycles", [(0, 1), (1, 0), (-1, 2), (True, 1), (1, 1.5)])
def test_soak_requires_positive_explicit_bounds(duration, cycles):
    with pytest.raises(ValueError):
        SoakLimits(duration, cycles)


def test_soak_cli_help_has_no_network(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--max-cycles" in capsys.readouterr().out


def test_acceptance_configuration_uses_existing_system():
    cfg = load_settings(Path(__file__).resolve().parents[1] / "docs/acceptance-smoke.toml")
    assert cfg.symbols == ("BTCUSDT", "ETHUSDT")
    assert cfg.research.validation_scheduler.validation_horizons_seconds == (5, 10)
    assert all(scenario.funding_events == () for scenario in cfg.research.scenarios.values())


def test_final_source_read_only_audit():
    root = Path(__file__).resolve().parents[1] / "src/crypto_arbitrage_bot"
    execution_root = root / "execution"
    transport_paths = {"exchanges/binance_spot.py", "exchanges/binance_usdtm.py",
                       "exchanges/funding_history.py", "exchanges/exchange_metadata.py"}
    actual_transports = set()
    for path in root.rglob("*.py"):
        if path.is_relative_to(execution_root):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "request":
                assert node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "GET"
                actual_transports.add(path.relative_to(root).as_posix())
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not any(route in node.value for route in ("/order", "/account", "/leverage", "/marginType"))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name not in {"create_order", "place_order", "change_leverage", "set_margin", "sign_request"}
            if isinstance(node, ast.Import):
                assert all(item.name.split(".")[0] != "hmac" for item in node.names)
    assert actual_transports == transport_paths
    assert all(endpoint.value.startswith(("https://data-api.binance.vision/", "https://fapi.binance.com/", "https://api.binance.com/"))
               for endpoint in PublicEndpoint)
    runner_source = (root / "runner.py").read_text(encoding="utf-8-sig")
    assert "crypto_arbitrage_bot.execution" not in runner_source
    execution_sources = "\n".join(path.read_text(encoding="utf-8-sig") for path in execution_root.rglob("*.py"))
    assert "https://testnet.binance.vision" in execution_sources
    assert "https://demo-fapi.binance.com" in execution_sources
    assert "https://fapi.binance.com" not in execution_sources
    assert "https://api.binance.com" not in execution_sources
    assert "/api/v3/order" in execution_sources and "/fapi/v1/order" in execution_sources
    assert "/marginType" not in execution_sources
    from crypto_arbitrage_bot.execution.testnet_client import _ROUTES
    assert {(market.value, path) for (market, _), (method, path) in _ROUTES.items() if method != "GET"} == {
        ("SPOT", "/api/v3/order"), ("USDTM", "/fapi/v1/order"),
        ("USDTM", "/fapi/v1/leverage"),
    }
    assert all(method in ("GET", "POST") for method, _ in _ROUTES.values())
