"""Explicit research identities reuse original cash flow; no trade authorization."""

import ast
import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_candidate_catalog import NOW
from test_candidate_market import catalog as catalog
from test_candidate_market import collector, install
from test_candidate_market import criteria as criteria
from test_candidate_market import offline as offline

from crypto_arbitrage_bot import candidate_cost_research as cli
from crypto_arbitrage_bot.arbitrage.candidate_estimate import compare_candidate_windows
from crypto_arbitrage_bot.arbitrage.estimate import estimate_once
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.arbitrage.profit import (
    decompose_fills,
    decomposed_net_profit,
    round_trip_from_fills,
)
from crypto_arbitrage_bot.arbitrage.scenario import reverse_execution_scenario
from crypto_arbitrage_bot.config.research import (
    CandidateEconomicsSettings,
    FundingProjectionSettings,
)
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import prepare_get
from crypto_arbitrage_bot.execution.types import ExecutionMarket, IocLimitOrder
from crypto_arbitrage_bot.models.estimate import EstimateConfig, EstimateStatus
from crypto_arbitrage_bot.models.market import SUPPORTED_SYMBOLS, BidAsk, Side
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import (
    RepositoryError,
    RepositoryOperation,
    SQLiteConfig,
    SQLiteOpportunityRepository,
)

D = Decimal


@pytest.fixture
def settings():
    return load_settings(Path("docs/research-candidate-costs.toml"))


@pytest.fixture
def observation(catalog, criteria, monkeypatch):
    install(monkeypatch)
    return collector(catalog, criteria).collect("SOLUSDT")


def compare(catalog, criteria, observation, settings, **kwargs):
    research = settings.research
    return compare_candidate_windows(catalog, criteria, observation,
        kwargs.get("economics", research.candidate_economics.model_copy(update={"spot_budget": D("101")})),
        settings.fees.as_rates(), kwargs.get("quality", research.data_quality), research.decision,
        evaluated_at=kwargs.get("at", NOW))


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
def test_original_math_and_exact_metadata_lots_apply_to_all_candidates(catalog, criteria, settings, monkeypatch, symbol):
    calls = install(monkeypatch)
    observation = collector(catalog, criteria).collect(symbol)
    before = exact_json((catalog, observation, settings.research.candidate_economics.model_dump()))
    results = compare(catalog, criteria, observation, settings)
    assert [item.profile_name for item in results] == ["1h", "8h", "24h"]
    assert [item.assumed_holding_duration for item in results] == [timedelta(hours=h) for h in (1, 8, 24)]
    for item in results:
        for scenario in item.scenarios:
            estimate = scenario.estimate
            assert estimate.complete and estimate.matched_quantity == 1
            assert estimate.config.spot_rules == observation.candidate.spot_rules
            assert estimate.config.perp_rules == observation.candidate.perp_rules
            assert estimate.fees.total == D(".3") or scenario.scenario_type.value == "ADVERSE"
            identity = observation.assembly.research_instrument
            calculated = round_trip_from_fills(*estimate.fills, estimate.fees,
                estimate.estimated_funding, estimate.config.other_costs, research_instrument=identity)
            assert calculated == estimate.calculation
            assert decomposed_net_profit(estimate.decomposition, estimate.estimated_funding,
                estimate.fees, estimate.config.other_costs) == estimate.net_profit
            assert estimate.scenario.perp_exit_book.research_instrument == identity
        baseline = item.scenarios[0].estimate
        assert baseline.price_pnl == -4 and baseline.net_profit == D("-4.3")
        assert baseline.estimated_funding == 0 and item.decision.status.value == "REJECT"
    assert before == exact_json((catalog, observation, settings.research.candidate_economics.model_dump()))
    assert len(calls) == 4 and all(c[0] == "GET" for c in calls)


@pytest.mark.parametrize("rate,expected", [(".01", "-3.3"), ("-.01", "-5.3"), ("0", "-4.3")])
def test_explicit_funding_hand_calculation_and_no_double_costs(catalog, criteria, observation, settings, rate, expected):
    economics = settings.research.candidate_economics.model_copy(update={"spot_budget": D("101")})
    event = FundingEvent("explicit-assumption", NOW + timedelta(hours=1), D("100"), D(rate), "SOLUSDT")
    profile = economics.windows["1h"].model_copy(update={"funding_events": (event,)})
    economics = economics.model_copy(update={"windows": {"1h": profile}})
    estimate = compare(catalog, criteria, observation, settings, economics=economics)[0].scenarios[0].estimate
    assert estimate.price_pnl == -4
    assert estimate.fees.spot_entry == D(".101") and estimate.fees.spot_exit == D(".099")
    assert estimate.fees.perp_entry == D(".0495") and estimate.fees.perp_exit == D(".0505")
    assert estimate.estimated_funding == D(rate) * D("100")
    assert estimate.net_profit == D(expected) and estimate.capital_occupied == D("200.3")
    with localcontext() as context:
        context.prec = 50
        assert estimate.net_return == D(expected) / D("200.3")


def test_recurring_vwap_keeps_original_notional_cash_flow(observation, settings):
    candidate = observation.candidate
    config = EstimateConfig(candidate.spot_rules, candidate.perp_rules, settings.fees.as_rates(), target_quantity=D(3))
    snapshot = observation.assembly.snapshot
    definition = reverse_execution_scenario(name="no-convergence", spot_entry_book=snapshot.spot_book,
        perp_entry_book=snapshot.perp_book, opened_at=NOW, closed_at=NOW + timedelta(hours=24),
        funding_events=(), description="No future funding or convergence assumed")
    with localcontext() as context:
        context.prec = 3
        estimate = estimate_once(observation.assembly, config, definition.exit_scenario)
    assert estimate.complete and estimate.net_profit == D("-20.9")
    assert estimate.fills[0].quote_notional == 305 and estimate.fills[2].quote_notional == 295
    with localcontext() as context:
        context.prec = 50
        assert estimate.spot_entry_vwap == D(305) / D(3)
    identity = observation.assembly.research_instrument
    with pytest.raises(ValueError):
        round_trip_from_fills(*estimate.fills, estimate.fees, D(0))
    quote = BidAsk(D(99), D(101))
    decomposition = decompose_fills(*estimate.fills, spot_entry_quote=quote, spot_exit_quote=quote,
        perp_entry_quote=quote, perp_exit_quote=quote, research_instrument=identity)
    assert decomposition == estimate.decomposition


@pytest.mark.parametrize("case", ["edited_candidate", "edited_catalog", "stale_metadata", "stale_quotes", "identity", "incomplete"])
def test_invalid_or_edited_research_inputs_never_produce_profit(catalog, criteria, observation, settings, case):
    at = NOW
    if case == "edited_candidate":
        observation = replace(observation, candidate=replace(observation.candidate, base_asset="ETH"))
    elif case == "edited_catalog":
        catalog = replace(catalog, candidates=catalog.candidates[:-1])
    elif case == "stale_metadata":
        at += timedelta(seconds=criteria.max_metadata_age_seconds + 1)
    elif case == "stale_quotes":
        at += timedelta(seconds=11)
    elif case == "identity":
        observation = replace(observation, assembly=replace(observation.assembly, research_instrument=None))
    else:
        observation = replace(observation, assembly=replace(observation.assembly, snapshot=None))
    with pytest.raises(ValueError):
        compare(catalog, criteria, observation, settings, at=at)


@pytest.mark.parametrize("delta,symbol", [(0, "SOLUSDT"), (3601, "SOLUSDT"), (1, "ETHUSDT")])
def test_explicit_wrong_funding_window_or_symbol_is_not_silently_ignored(catalog, criteria, observation, settings, delta, symbol):
    economics = settings.research.candidate_economics
    event = FundingEvent("wrong-window", NOW + timedelta(seconds=delta), D(100), D(".01"), symbol)
    profile = economics.windows["1h"].model_copy(update={"funding_events": (event,)})
    with pytest.raises(ValueError):
        compare(catalog, criteria, observation, settings,
                economics=economics.model_copy(update={"windows": {"1h": profile}}))


def test_insufficient_candidate_depth_cannot_expose_profit(catalog, criteria, observation, settings):
    economics = settings.research.candidate_economics.model_copy(update={"spot_budget": D("1000"), "capital_limit": None})
    for item in compare(catalog, criteria, observation, settings, economics=economics):
        assert item.decision.status.value == "REJECT"
        assert all(s.estimate.status is EstimateStatus.INSUFFICIENT_DEPTH for s in item.scenarios)
        assert all(s.net_profit is None and s.net_return is None for s in item.scenarios)


@pytest.mark.parametrize("field,value", [("spot_budget", "0"), ("spot_budget", "NaN"),
    ("spot_budget", 100.0), ("capital_limit", "-1"), ("stress_reserve", "-1"),
    ("other_costs", "Infinity"), ("windows", {}), ("windows", {"invalid name": {}})])
def test_candidate_economics_config_fails_closed(settings, field, value):
    values = settings.research.candidate_economics.model_dump()
    values[field] = value
    with pytest.raises(ValueError):
        CandidateEconomicsSettings.model_validate(values)


def test_projected_window_is_explicit_and_never_hourly_prorated(catalog, criteria, observation, settings):
    economics = settings.research.candidate_economics
    projection = FundingProjectionSettings(mode="next_settlement", positive_rate_multiplier=D(".5"),
        max_wait_seconds=86400, exit_delay_seconds=60, min_entry_lead_seconds=60, entry_window_seconds=3600)
    profile = economics.windows["24h"].model_copy(update={"funding_projection": projection,
        "adverse_exit_basis_mode": "entry_offset"})
    result = compare(catalog, criteria, observation, settings,
        economics=economics.model_copy(update={"windows": {"next_settlement": profile}}))[0]
    assert result.configured_holding_seconds == 86400
    assert result.assumed_holding_duration == timedelta(seconds=3660)
    baseline, _, stress = result.scenarios
    assert len(baseline.estimate.scenario.funding_events) == 1
    assert "FORECAST_ONLY" in baseline.assumptions
    assert stress.funding_total == 0  # removes assumed positive income
    assert baseline.estimate.scenario.funding_events[0].event_id.startswith("FORECAST_ONLY:")


def configured_cli(monkeypatch, catalog, settings, tmp_path):
    install(monkeypatch)
    monkeypatch.setattr(cli, "collect_catalog", Mock(return_value=catalog))
    monkeypatch.setattr(cli, "utc_now", lambda: NOW)
    research = settings.research.model_copy(update={"sqlite_path": tmp_path / "research.sqlite"})
    return settings.model_copy(update={"research": research})


def test_public_cycle_saves_three_windows_without_scheduling_or_trading(monkeypatch, catalog, settings, tmp_path):
    settings = configured_cli(monkeypatch, catalog, settings, tmp_path)
    result = cli.research_once(settings, ("BTCUSDT", "ETHUSDT", "SOLUSDT"), tmp_path / "report.json")
    assert all(item["status"] == "SAVED" for item in result["captures"])
    repository = SQLiteOpportunityRepository(SQLiteConfig(settings.research.sqlite_path))
    records = repository.list_recent_opportunities()
    assert len(records) == 9 and {r.symbol for r in records} == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert all(len(r.scenarios) == 3 for r in records)
    assert all(r.created_at == NOW and r.decision == "REJECT" for r in records)
    assert all(isinstance(r.net_profit, D) and len(r.source_timings) == 4 for r in records)
    body = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert body["schema_version"] == "candidate-cost-research-v1" and len(body["catalog"]["candidates"]) == 3
    with repository._connection(RepositoryOperation.LIST) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not any("testnet" in name or "validation_task" in name for name in tables)


def test_one_collection_failure_does_not_block_other_candidate(monkeypatch, catalog, settings, tmp_path):
    settings = configured_cli(monkeypatch, catalog, settings, tmp_path)
    original = cli.CandidateMarketCollector.collect
    def fail(self, symbol):
        if symbol == "BTCUSDT":
            raise TimeoutError("offline public timeout")
        return original(self, symbol)
    monkeypatch.setattr(cli.CandidateMarketCollector, "collect", fail)
    result = cli.research_once(settings, ("BTCUSDT", "ETHUSDT"), tmp_path / "report.json")
    assert [item["status"] for item in result["captures"]] == ["FAILED", "SAVED"]
    assert len(SQLiteOpportunityRepository(SQLiteConfig(settings.research.sqlite_path)).list_recent_opportunities()) == 3


def test_sqlite_failure_does_not_turn_research_into_trading_reject(monkeypatch, catalog, settings, tmp_path):
    settings = configured_cli(monkeypatch, catalog, settings, tmp_path)
    monkeypatch.setattr(cli.SQLiteOpportunityRepository, "save_opportunity",
        Mock(side_effect=RepositoryError(RepositoryOperation.SAVE, OSError("offline locked"))))
    result = cli.research_once(settings, ("SOLUSDT",), tmp_path / "report.json")
    assert result["captures"][0]["status"] == "PERSISTENCE_FAILED"
    assert all(item["opportunity_id"] is None and item["net_profit"] is not None
               for item in result["captures"][0]["windows"])


@pytest.mark.parametrize("symbols", [(), ("SOLUSDT", "SOLUSDT"), tuple(f"X{i}USDT" for i in range(11))])
def test_invalid_batch_rejected_before_io(settings, tmp_path, symbols, monkeypatch):
    catalog = Mock(side_effect=AssertionError("no collection"))
    monkeypatch.setattr(cli, "collect_catalog", catalog)
    with pytest.raises(ValueError):
        cli.research_once(settings, symbols, tmp_path / "report.json")
    catalog.assert_not_called()


def test_existing_evidence_is_never_overwritten(settings, tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("retained", encoding="utf-8")
    collect = Mock(side_effect=AssertionError("no collection"))
    monkeypatch.setattr(cli, "collect_catalog", collect)
    with pytest.raises(FileExistsError):
        cli.research_once(settings, ("SOLUSDT",), output)
    assert output.read_text(encoding="utf-8") == "retained" and not collect.called


def test_requested_symbol_cannot_be_replaced_by_another_valid_observation(monkeypatch, catalog, settings, tmp_path):
    settings = configured_cli(monkeypatch, catalog, settings, tmp_path)
    observation = collector(catalog, settings.research.candidate_catalog).collect("ETHUSDT")
    monkeypatch.setattr(cli.CandidateMarketCollector, "collect", Mock(return_value=observation))
    result = cli.research_once(settings, ("BTCUSDT",), tmp_path / "report.json")
    assert result["captures"][0]["status"] == "FAILED"
    assert not SQLiteOpportunityRepository(SQLiteConfig(settings.research.sqlite_path)).list_recent_opportunities()


@pytest.mark.parametrize("leg", ["spot_exit_book", "perp_exit_book"])
def test_legacy_exit_book_cannot_strip_research_identity(monkeypatch, catalog, criteria, settings, leg):
    install(monkeypatch)
    observed = collector(catalog, criteria).collect("BTCUSDT")
    estimate = compare(catalog, criteria, observed, settings)[0].scenarios[0].estimate
    changed = replace(estimate.scenario, **{leg: replace(getattr(estimate.scenario, leg), research_instrument=None)})
    with pytest.raises(ValueError, match="research identity"):
        estimate_once(observed.assembly, estimate.config, changed)


def test_duplicate_explicit_settlements_cannot_double_funding(catalog, criteria, observation, settings):
    economics = settings.research.candidate_economics
    event = FundingEvent("duplicate", NOW + timedelta(hours=1), D(100), D(".01"), "SOLUSDT")
    profile = economics.windows["1h"].model_copy(update={"funding_events": (event, event)})
    with pytest.raises(ValueError, match="duplicate funding"):
        compare(catalog, criteria, observation, settings,
                economics=economics.model_copy(update={"windows": {"1h": profile}}))


def test_research_identity_does_not_authorize_public_default_or_testnet_order(observation):
    assert set(SUPPORTED_SYMBOLS) == {"BTCUSDT", "ETHUSDT"}
    with pytest.raises(ValueError):
        prepare_get(PublicEndpoint.SPOT_DEPTH, {"symbol": "SOLUSDT"})
    with pytest.raises(ValueError):
        IocLimitOrder(ExecutionMarket.SPOT, "SOLUSDT", Side.BUY, D(1), D(100), "candidate")
    identity = ResearchInstrument("ETHUSDT", "ETH")
    with pytest.raises(ValueError):
        identity.require_symbol(observation.candidate.symbol)


def test_new_research_modules_have_no_network_transport_or_execution_call():
    for module in (cli, __import__("crypto_arbitrage_bot.arbitrage.candidate_estimate", fromlist=["x"])):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert not calls & {"request", "urlopen", "place_ioc_limit_order", "open_pair", "close_pair"}
