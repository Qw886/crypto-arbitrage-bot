"""Complete research cycles use offline adapters/fixtures and temporary SQLite."""

import json
from calendar import timegm
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
from test_offline_estimate import OPENED, RATES, assembly_for, books, config
from test_scenario import event
from test_scenario_decision import limits

from crypto_arbitrage_bot import runner as module
from crypto_arbitrage_bot.config.research import ResearchSettings, ScenarioSettings, SizingSettings
from crypto_arbitrage_bot.config.settings import FeeSettings, Settings, load_settings
from crypto_arbitrage_bot.market_data.snapshot import SnapshotCollector, assemble_snapshot
from crypto_arbitrage_bot.models.estimate import EstimateStatus
from crypto_arbitrage_bot.models.spot import SpotErrorKind, SpotFetchError
from crypto_arbitrage_bot.models.usdtm import UsdtmErrorKind, UsdtmFetchError
from crypto_arbitrage_bot.runner import CycleStage, CycleStatus, ResearchRunner
from crypto_arbitrage_bot.storage.sqlite_repository import SQLiteConfig, SQLiteOpportunityRepository

D = Decimal


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("tests must not contact Binance"))
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("http.client.HTTPSConnection", forbidden)


def settings_for(tmp_path, symbols=("BTCUSDT", "ETHUSDT"), with_funding=True):
    base = config()
    research = ResearchSettings(
        sqlite_path=tmp_path / "opportunities.sqlite", timeout_seconds=5,
        formula_version="synthetic-1", config_version="synthetic-1",
        data_quality=base.data_quality, decision=limits(),
        sizing={symbol: SizingSettings(spot_rules=base.spot_rules, perp_rules=base.perp_rules,
                                       target_quantity=D("2"), stress_reserve=D("0"), other_costs=D(".1"))
                for symbol in symbols},
        scenarios={symbol: ScenarioSettings(
            baseline="reverse_execution", holding_seconds=7200, spot_exit_mid_mode="entry_mid",
            convergence_exit_basis_pct=D(".02"), adverse_exit_basis_pct=D(".16"),
            funding_events=(event(".1", symbol=symbol),) if with_funding else (),
        ) for symbol in symbols},
    )
    return Settings(symbols=symbols, poll_interval_seconds=5, log_level="INFO", log_dir=tmp_path / "logs",
                    fees=FeeSettings(spot_entry=RATES.spot_entry, spot_exit=RATES.spot_exit,
                                     perp_entry=RATES.perp_entry, perp_exit=RATES.perp_exit), research=research.model_dump())


def make_runner(tmp_path, monkeypatch, **settings_changes):
    settings = settings_for(tmp_path, **settings_changes)
    assemblies = {symbol: assembly_for((books(symbol)[0], books(symbol)[2])) for symbol in settings.symbols}
    collector = Mock(spec=SnapshotCollector)
    collector.collect.side_effect = lambda symbol: assemblies[symbol]
    repository = SQLiteOpportunityRepository(SQLiteConfig(settings.research.sqlite_path, busy_timeout_ms=0))
    logger = Mock()
    instance = ResearchRunner(settings, collector, repository, clock=lambda: OPENED, logger=logger)
    return instance, assemblies, logger


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_complete_cycle_calls_estimate_once_per_scenario_and_saves(tmp_path, monkeypatch, symbol):
    runner, _, logger = make_runner(tmp_path, monkeypatch, symbols=(symbol,))
    from crypto_arbitrage_bot.arbitrage import scenario as scenario_module
    estimate = Mock(wraps=scenario_module.estimate_once)
    monkeypatch.setattr(scenario_module, "estimate_once", estimate)
    decision = Mock(wraps=module.decide_scenarios)
    monkeypatch.setattr(module, "decide_scenarios", decision)
    builder = Mock(wraps=module.build_and_save_opportunity)
    monkeypatch.setattr(module, "build_and_save_opportunity", builder)
    cycle, = runner.run_once()
    assert cycle.status is CycleStatus.SAVED and cycle.decision == "PASS"
    assert cycle.symbol == symbol and cycle.data_quality.valid
    assert cycle.failure_stage is cycle.failure_reason is None
    assert estimate.call_count == 3 and decision.call_count == builder.call_count == 1
    record = runner.repository.get_opportunity_by_id(cycle.opportunity_id)
    assert record.symbol == symbol and record.net_profit == D("9.0144")
    assert len(record.scenarios) == 3 and record.decision == cycle.decision
    assert record.formula_version == record.config_version == "synthetic-1"
    log = json.loads(logger.info.call_args.args[0])
    assert log["cycle_id"] == cycle.cycle_id and log["opportunity_id"] == record.id
    assert log["started_at"] == log["ended_at"] == OPENED.isoformat(timespec="microseconds")


def test_symbols_run_in_order_and_get_distinct_cycle_ids(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path, monkeypatch)
    cycles = runner.run_once()
    assert [call.args[0] for call in runner.collector.collect.call_args_list] == ["BTCUSDT", "ETHUSDT"]
    assert len({cycle.cycle_id for cycle in cycles}) == 2
    assert len(runner.repository.list_recent_opportunities()) == 2


@pytest.mark.parametrize("source_name", ["spot_depth", "perp_depth", "premium_index", "open_interest"])
def test_typed_api_failure_stops_symbol_but_next_symbol_succeeds(tmp_path, monkeypatch, source_name):
    runner, assemblies, logger = make_runner(tmp_path, monkeypatch)
    original = assemblies["BTCUSDT"]
    source = getattr(original.sources, source_name)
    if source_name == "spot_depth":
        failed = replace(source, order_book=None, error=SpotFetchError(SpotErrorKind.NETWORK_ERROR, "synthetic DNS error"))
    else:
        failed = replace(source, data=None, error=UsdtmFetchError(UsdtmErrorKind.TIMEOUT, "synthetic timeout"))
    assemblies["BTCUSDT"] = assemble_snapshot("BTCUSDT", replace(original.sources, **{source_name: failed}), assembled_at=OPENED)
    cycles = runner.run_once()
    assert cycles[0].status is CycleStatus.FAILED and cycles[0].failure_stage is CycleStage.COLLECTION
    assert cycles[0].opportunity_id is cycles[0].decision is cycles[0].data_quality is None
    assert "synthetic" in cycles[0].failure_reason
    assert cycles[1].status is CycleStatus.SAVED
    assert len(runner.repository.list_recent_opportunities()) == 1 and logger.info.call_count == 2


def test_data_quality_invalid_prevents_estimate_and_storage(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path, monkeypatch, symbols=("BTCUSDT",))
    runner.clock = lambda: OPENED + timedelta(seconds=11)
    estimate = Mock(side_effect=AssertionError("quality must block estimate"))
    monkeypatch.setattr(module, "evaluate_scenario", estimate)
    cycle, = runner.run_once()
    assert cycle.failure_stage is CycleStage.DATA_QUALITY and not cycle.data_quality.valid
    assert cycle.decision is cycle.opportunity_id is None
    assert not runner.repository.config.database_path.exists()
    estimate.assert_not_called()


def test_estimate_failure_prevents_scenarios_and_decision(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path, monkeypatch, symbols=("BTCUSDT",))
    original = module.evaluate_scenario
    def failed(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, estimate=replace(result.estimate, status=EstimateStatus.INSUFFICIENT_DEPTH, calculation=None))
    monkeypatch.setattr(module, "evaluate_scenario", failed)
    following = Mock(side_effect=AssertionError("failed estimate must stop cycle"))
    monkeypatch.setattr(module, "evaluate_scenarios", following)
    cycle, = runner.run_once()
    assert cycle.failure_stage is CycleStage.ESTIMATE and cycle.decision is None
    following.assert_not_called()


def test_persistence_failure_keeps_existing_decision_and_other_symbol_continues(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path, monkeypatch)
    original = runner.repository.save_opportunity
    def save(record):
        if record.symbol == "BTCUSDT":
            raise OSError("synthetic storage failure")
        return original(record)
    monkeypatch.setattr(runner.repository, "save_opportunity", save)
    cycles = runner.run_once()
    assert cycles[0].failure_stage is CycleStage.PERSISTENCE
    assert cycles[0].decision == "PASS" and cycles[0].opportunity_id is None
    assert cycles[1].status is CycleStatus.SAVED


@pytest.mark.parametrize("function,stage", [("evaluate_scenarios", CycleStage.SCENARIOS), ("decide_scenarios", CycleStage.DECISION)])
def test_unexpected_stage_failure_is_isolated(tmp_path, monkeypatch, function, stage):
    runner, _, _ = make_runner(tmp_path, monkeypatch)
    original = getattr(module, function)
    calls = 0
    def fail_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic stage error")
        return original(*args, **kwargs)
    monkeypatch.setattr(module, function, fail_first)
    cycles = runner.run_once()
    assert cycles[0].failure_stage is stage and cycles[0].opportunity_id is None
    assert cycles[1].status is CycleStatus.SAVED


def test_no_events_does_not_convert_positive_indicative_funding_to_income(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path, monkeypatch, symbols=("BTCUSDT",), with_funding=False)
    cycle, = runner.run_once()
    assert cycle.status is CycleStatus.SAVED and cycle.decision == "REJECT"
    record = runner.repository.get_opportunity_by_id(cycle.opportunity_id)
    assert record.funding_rate > 0 and record.estimated_funding == 0


def test_failed_cycle_never_reuses_previous_success_snapshot(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path, monkeypatch, symbols=("BTCUSDT",))
    assert runner.run_once()[0].status is CycleStatus.SAVED
    runner.collector.collect.side_effect = TimeoutError("synthetic second fetch timeout")
    cycle, = runner.run_once()
    assert cycle.failure_stage is CycleStage.COLLECTION and cycle.opportunity_id is None
    assert len(runner.repository.list_recent_opportunities()) == 1


def test_run_forever_waits_after_failed_rounds_without_fast_retry(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path, monkeypatch)
    runner.collector.collect.side_effect = OSError("network down")
    stop = Mock()
    stop.is_set.return_value = False
    stop.wait.side_effect = [False, False, True]
    runner.stop_signal = stop
    runner.run_forever()
    assert runner.collector.collect.call_count == 6
    assert [call.args[0] for call in stop.wait.call_args_list] == [5, 5, 5]


def test_keyboard_interrupt_in_cycle_or_wait_exits_cleanly(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path, monkeypatch)
    runner.collector.collect.side_effect = KeyboardInterrupt
    runner.run_forever()
    assert runner.stop_signal.is_set() and runner.collector.collect.call_count == 1
    runner, _, _ = make_runner(tmp_path, monkeypatch)
    stop = Mock()
    stop.is_set.return_value = False
    stop.wait.side_effect = KeyboardInterrupt
    runner.stop_signal = stop
    runner.run_forever()
    stop.set.assert_called_once()


def test_explicit_stop_prevents_new_cycles_and_logging_failure_is_isolated(tmp_path, monkeypatch):
    runner, _, logger = make_runner(tmp_path, monkeypatch)
    logger.info.side_effect = OSError("log unavailable")
    cycles = runner.run_once()
    assert len(cycles) == 2 and all(cycle.failure_stage is CycleStage.LOGGING for cycle in cycles)
    assert all(cycle.opportunity_id is not None for cycle in cycles)
    runner.stop()
    assert runner.run_once() == ()


def test_settings_reuse_environment_override_and_validate_decimal_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("CAB_RESEARCH__SQLITE_PATH", str(tmp_path / "overridden.sqlite"))
    settings = settings_for(tmp_path)
    assert settings.research.sqlite_path == tmp_path / "overridden.sqlite"
    data = settings.research.model_dump()
    data["sizing"]["BTCUSDT"]["target_quantity"] = 0.1
    with pytest.raises(ValueError, match="never floats"):
        ResearchSettings.model_validate(data)


def test_missing_research_settings_do_not_enable_network():
    settings = load_settings()
    with pytest.raises(ValueError, match="research"):
        module.create_runner(settings)


def test_cli_routes_only_explicit_runner_modes(tmp_path, monkeypatch, capsys):
    from crypto_arbitrage_bot.main import main
    runner, _, _ = make_runner(tmp_path, monkeypatch)
    factory = Mock(return_value=runner)
    monkeypatch.setattr(module, "create_runner", factory)
    assert main([]) == 0
    factory.assert_not_called()
    assert main(["--once"]) == 0
    factory.assert_called_once()
    assert "opportunity_id" in capsys.readouterr().out


def test_real_collector_is_reused_with_mock_public_adapters(tmp_path, monkeypatch):
    runner, assemblies, _ = make_runner(tmp_path, monkeypatch, symbols=("BTCUSDT",))
    sources = assemblies["BTCUSDT"].sources
    spot, perp = Mock(), Mock()
    spot.fetch_order_book.return_value = sources.spot_depth
    perp.fetch_order_book.return_value = sources.perp_depth
    perp.fetch_premium_index.return_value = sources.premium_index
    perp.fetch_open_interest.return_value = sources.open_interest
    monkeypatch.setattr("crypto_arbitrage_bot.market_data.snapshot.utc_now", lambda: OPENED)
    runner.collector = SnapshotCollector(spot, perp)
    assert runner.run_once()[0].status is CycleStatus.SAVED
    spot.fetch_order_book.assert_called_once_with("BTCUSDT", limit=100)
    perp.fetch_order_book.assert_called_once_with("BTCUSDT", limit=100)
    perp.fetch_premium_index.assert_called_once_with("BTCUSDT")
    perp.fetch_open_interest.assert_called_once_with("BTCUSDT")


def test_full_runner_public_transport_boundary_without_network(tmp_path, monkeypatch):
    settings = settings_for(tmp_path, with_funding=False)
    calls = []
    stamp = timegm(OPENED.utctimetuple()) * 1000
    allowed = {
        ("data-api.binance.vision", "/api/v3/depth"),
        ("fapi.binance.com", "/fapi/v1/depth"),
        ("fapi.binance.com", "/fapi/v1/premiumIndex"),
        ("fapi.binance.com", "/fapi/v1/openInterest"),
    }
    def connect(host, *, timeout):
        assert timeout == settings.research.timeout_seconds
        connection, response = Mock(), Mock()
        response.status = 200
        response.getheaders.return_value = []
        connection.getresponse.return_value = response
        def request(method, target, *, headers):
            parts = urlsplit(target)
            params = parse_qs(parts.query)
            assert method == "GET" and (host, parts.path) in allowed
            assert set(headers) == {"Accept", "User-Agent"}
            assert set(params) <= {"symbol", "limit"}
            symbol = params["symbol"][0]
            assert symbol in settings.symbols
            calls.append((symbol, host, parts.path))
            if parts.path == "/api/v3/depth":
                body = {"lastUpdateId": 1, "bids": [["99", "5"]], "asks": [["101", "5"]]}
            elif parts.path == "/fapi/v1/depth":
                body = {"lastUpdateId": 2, "E": stamp, "T": stamp,
                        "bids": [["109", "5"]], "asks": [["111", "5"]]}
            elif parts.path == "/fapi/v1/premiumIndex":
                body = {"symbol": symbol, "markPrice": "110", "indexPrice": "100",
                        "lastFundingRate": "0.01", "nextFundingTime": stamp + 3600000, "time": stamp}
            else:
                body = {"symbol": symbol, "openInterest": "123.456", "time": stamp}
            response.read.return_value = json.dumps(body).encode()
        connection.request.side_effect = request
        return connection
    for adapter in ("binance_spot", "binance_usdtm"):
        monkeypatch.setattr(f"crypto_arbitrage_bot.exchanges.{adapter}.HTTPSConnection", connect)
        monkeypatch.setattr(f"crypto_arbitrage_bot.exchanges.{adapter}.utc_now", lambda: OPENED)
    monkeypatch.setattr("crypto_arbitrage_bot.market_data.snapshot.utc_now", lambda: OPENED)
    monkeypatch.setattr(module, "configure_logging", Mock(return_value=Mock()))
    runner = module.create_runner(settings)
    assert calls == []  # Constructing the runner cannot send a request.
    runner.clock = lambda: OPENED
    cycles = runner.run_once()
    assert len(calls) == 8 and all(cycle.status is CycleStatus.SAVED for cycle in cycles)
    assert [item[0] for item in calls] == ["BTCUSDT"] * 4 + ["ETHUSDT"] * 4
    assert all("key" not in field.lower() and "secret" not in field.lower()
               for model in (Settings, ResearchSettings) for field in model.model_fields)
    with pytest.raises(ValueError):
        ResearchSettings.model_validate({**settings.research.model_dump(), "api_key": "not-supported"})


def test_runner_toml_load_and_nested_environment_override(tmp_path, monkeypatch):
    source = settings_for(tmp_path, with_funding=False)
    data = source.model_dump(mode="json")
    lines = []
    def table(mapping, path=()):
        if path:
            lines.append("[" + ".".join(path) + "]")
        for key, value in mapping.items():
            if value is not None and not isinstance(value, dict):
                lines.append(f"{key} = {json.dumps(value)}")
        for key, value in mapping.items():
            if isinstance(value, dict):
                table(value, (*path, key))
    table(data)
    path = tmp_path / "runner.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setenv("CAB_RESEARCH__SQLITE_PATH", str(tmp_path / "overridden.sqlite"))
    loaded = load_settings(path)
    assert loaded.research.sqlite_path == tmp_path / "overridden.sqlite"
    assert loaded.research.sizing["BTCUSDT"].target_quantity == D("2")
    assert loaded.research.decision == source.research.decision
    assert loaded.research.data_quality == source.research.data_quality
    assert loaded.research.scenarios == source.research.scenarios
