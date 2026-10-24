"""Offline qualified preselection consumes public evidence, never trade authority."""

import ast
import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_candidate_catalog import MS, NOW, instrument, source
from test_candidate_market import install, payload, response

from crypto_arbitrage_bot import candidate_cost_research
from crypto_arbitrage_bot import funding_screen_research as cli
from crypto_arbitrage_bot.config.research import FundingScreenSettings
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.exchanges import binance_spot, binance_usdtm, proxy_transport
from crypto_arbitrage_bot.exchanges.binance_usdtm import BinanceUsdtmClient
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import (
    PremiumCatalogGetRequest,
    PublicGetRequest,
    RequestPhase,
    prepare_get,
    prepare_premium_catalog_get,
)
from crypto_arbitrage_bot.market_data import candidate_market, snapshot
from crypto_arbitrage_bot.market_data.candidate_catalog import build_candidate_catalog
from crypto_arbitrage_bot.market_data.funding_screen import ScreenReason as R
from crypto_arbitrage_bot.market_data.funding_screen import ScreenStatus as S
from crypto_arbitrage_bot.market_data.funding_screen import screen_candidates
from crypto_arbitrage_bot.models.market import SUPPORTED_SYMBOLS
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.models.usdtm import UsdtmErrorKind as E
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import SQLiteConfig, SQLiteOpportunityRepository

D = Decimal


@pytest.fixture
def settings():
    return load_settings(Path("docs/research-funding-screen.toml"))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for module in (binance_spot, binance_usdtm, candidate_market, snapshot, candidate_cost_research, cli):
        monkeypatch.setattr(module, "utc_now", lambda: NOW)
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("offline only")))


@pytest.fixture
def catalog(settings):
    assets = ("BTC", "ETH", "SOL", "ADA", "DOGE", "BNB")
    return build_candidate_catalog(source([instrument(a) for a in assets]),
        source([instrument(a, True) for a in assets], True), settings.research.candidate_catalog, now=NOW)


def rows():
    result = []
    for symbol, rate in (("BTCUSDT", "-0.01"), ("ETHUSDT", "0"), ("SOLUSDT", "0.02"),
                         ("ADAUSDT", "0.03"), ("DOGEUSDT", "0.01"), ("BNBUSDT", "0.01")):
        result.append(payload("premiumIndex", symbol, rate))
    return result


def fetch(monkeypatch, data=None, status=200, fault=None):
    connection = response(rows() if data is None else data, status)
    if fault is not None:
        connection.request.side_effect = fault
    monkeypatch.setattr(binance_usdtm, "open_public_connection", Mock(return_value=connection))
    return BinanceUsdtmClient(3).fetch_premium_catalog(), connection


def screen(settings, catalog, result, *, at=NOW, budget=5):
    config = settings.research.funding_screen.model_copy(update={"max_candidates": budget})
    return screen_candidates(catalog, settings.research.candidate_catalog, result, config,
                             settings.research.data_quality, evaluated_at=at)


def test_bulk_get_raw_times_and_prices_are_preserved(monkeypatch):
    result, connection = fetch(monkeypatch)
    assert result.ok and isinstance(result.request, PremiumCatalogGetRequest)
    assert result.data[0]["lastFundingRate"] == "-0.01"
    assert json.loads(result.raw_response) == rows()
    assert result.started_at == result.received_at == result.finished_at == NOW
    method, route = connection.request.call_args.args
    assert method == "GET" and route == "/fapi/v1/premiumIndex?"
    assert "X-MBX-APIKEY" not in connection.request.call_args.kwargs["headers"]
    connection.close.assert_called_once()


@pytest.mark.parametrize("status,kind", [(400, E.HTTP_ERROR), (429, E.RATE_LIMITED),
    (418, E.RATE_LIMITED), (500, E.HTTP_ERROR)])
def test_bulk_http_failures_keep_existing_structure(monkeypatch, status, kind):
    result, _ = fetch(monkeypatch, status=status)
    assert not result.ok and result.data is None and result.error.kind is kind
    assert result.raw_response is not None and result.error.phase is RequestPhase.RESPONSE_BODY


@pytest.mark.parametrize("failure,kind", [(TimeoutError(), E.TIMEOUT),
    (ConnectionRefusedError(), E.NETWORK_ERROR), (OSError(), E.NETWORK_ERROR)])
def test_bulk_network_failures_never_produce_catalog_data(monkeypatch, failure, kind):
    result, connection = fetch(monkeypatch, fault=failure)
    assert result.data is None and result.error.kind is kind
    assert result.error.phase is RequestPhase.REQUEST
    connection.close.assert_called_once()


@pytest.mark.parametrize("bad", [b"", b"not json", [], {}, [None], [{}],
    [{"symbol": "SOLUSDT"}, {"symbol": "SOLUSDT"}],
    b'[{"symbol":"BTCUSDT","symbol":"ETHUSDT"}]', b'[{"symbol":"BTCUSDT","x":NaN}]'])
def test_malformed_bulk_payload_is_not_a_partial_success(monkeypatch, bad):
    result, _ = fetch(monkeypatch, data=bad)
    assert result.data is None and result.error.kind is E.INVALID_RESPONSE


def test_bulk_api_error_retains_api_code(monkeypatch):
    result, _ = fetch(monkeypatch, {"code": -1003, "msg": "too many requests"})
    assert result.data is None and result.error.kind is E.API_ERROR and result.error.api_code == -1003


def test_explicit_bulk_boundary_does_not_widen_default_symbol_or_endpoint_permissions():
    assert prepare_premium_catalog_get().method == "GET"
    with pytest.raises(ValueError):
        prepare_get(PublicEndpoint.PERP_PREMIUM_INDEX)
    with pytest.raises(ValueError):
        prepare_get(PublicEndpoint.PERP_PREMIUM_INDEX, {"symbol": "ADAUSDT"})
    with pytest.raises(ValueError):
        PremiumCatalogGetRequest(PublicEndpoint.SPOT_DEPTH)
    with pytest.raises(ValueError):
        PremiumCatalogGetRequest(PublicEndpoint.PERP_PREMIUM_INDEX, (("symbol", "BTCUSDT"),))
    with pytest.raises(ValueError):
        PremiumCatalogGetRequest(PublicEndpoint.PERP_PREMIUM_INDEX, research_instrument=ResearchInstrument("SOLUSDT", "SOL"))
    with pytest.raises(ValueError):
        BinanceUsdtmClient(research_instrument=ResearchInstrument("SOLUSDT", "SOL")).fetch_premium_catalog()
    assert set(SUPPORTED_SYMBOLS) == {"BTCUSDT", "ETHUSDT"}


def test_bulk_get_uses_existing_proxy_transport(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://research-proxy.example:9123")
    monkeypatch.setattr(proxy_transport, "proxy_bypass", lambda host: False)
    connection = response(rows())
    factory = Mock(return_value=connection)
    monkeypatch.setattr(binance_usdtm, "HTTPSConnection", factory)
    monkeypatch.setattr(binance_usdtm, "open_public_connection", proxy_transport.open_public_connection)
    result = BinanceUsdtmClient(3).fetch_premium_catalog()
    assert result.ok
    factory.assert_called_once_with("research-proxy.example", port=9123, timeout=3)
    connection.set_tunnel.assert_called_once_with("fapi.binance.com", port=443)


def test_ranking_keeps_negative_zero_benchmarks_then_positive_decimal_priority(settings, catalog, monkeypatch):
    source, _ = fetch(monkeypatch)
    before = exact_json((catalog, source))
    result = screen(settings, catalog, source)
    assert result.status is S.COMPLETE
    assert [r.candidate.symbol for r in result.selected] == ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "BNBUSDT"]
    assert result.selected[0].premium.last_funding_rate == D("-.01")
    assert result.selected[1].premium.last_funding_rate == 0
    assert all(isinstance(r.premium.mark_price, Decimal) for r in result.selected)
    assert result.exclusions[0].symbol == "DOGEUSDT" and result.exclusions[0].reason_codes == (R.OUTSIDE_COLLECTION_BUDGET,)
    assert before == exact_json((catalog, source))
    assert not any(hasattr(item, "decision") or hasattr(item, "net_profit") for item in result.selected)


def test_ranking_preserves_sub_float_and_sub_context_rate_differences(settings, catalog, monkeypatch):
    data = rows()
    data[2]["lastFundingRate"] = "0.010000000000000000000000000000000000001"
    data[3]["lastFundingRate"] = "0.010000000000000000000000000000000000000"
    data[4]["lastFundingRate"] = "0"
    data[5]["lastFundingRate"] = "0"
    source, _ = fetch(monkeypatch, data)
    with localcontext() as context:
        context.prec = 3
        result = screen(settings, catalog, source)
    assert [r.candidate.symbol for r in result.selected] == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]


@pytest.mark.parametrize("field,value,reason", [("markPrice", "0", R.INVALID_PREMIUM),
    ("indexPrice", "-1", R.INVALID_PREMIUM), ("lastFundingRate", None, R.INVALID_PREMIUM),
    ("lastFundingRate", 0.03, R.INVALID_PREMIUM), ("lastFundingRate", "-0.03", R.FUNDING_BELOW_SCREEN_THRESHOLD),
    ("lastFundingRate", "0", R.FUNDING_BELOW_SCREEN_THRESHOLD),
    ("time", MS-11000, R.STALE_PREMIUM), ("time", MS+1, R.FUTURE_PREMIUM),
    ("nextFundingTime", MS, R.NEXT_SETTLEMENT_OUTSIDE_WINDOW),
    ("nextFundingTime", MS+86400001, R.NEXT_SETTLEMENT_OUTSIDE_WINDOW)])
def test_invalid_candidate_row_is_excluded_without_faking_fields(settings, catalog, monkeypatch, field, value, reason):
    data = rows()
    data[3][field] = value
    source, _ = fetch(monkeypatch, data)
    result = screen(settings, catalog, source)
    assert result.status is S.COMPLETE
    assert "ADAUSDT" not in [r.candidate.symbol for r in result.selected]
    assert next(r for r in result.exclusions if r.symbol == "ADAUSDT").reason_codes == (reason,)


@pytest.mark.parametrize("case", ["metadata", "source", "time", "identity", "duplicate", "missing_benchmark"])
def test_invalid_global_source_cannot_select_candidates(settings, catalog, monkeypatch, case):
    data = rows()
    if case == "missing_benchmark":
        data = data[1:]
    result, _ = fetch(monkeypatch, data)
    if case == "metadata":
        catalog = replace(catalog, candidates=catalog.candidates[:-1])
    elif case == "source":
        result = replace(result, data=None)
    elif case == "time":
        result = replace(result, started_at=NOW-timedelta(seconds=11))
    elif case == "identity":
        result = replace(result, request=PublicGetRequest(PublicEndpoint.PERP_PREMIUM_INDEX, (("symbol", "BTCUSDT"),)))
    elif case == "duplicate":
        result = replace(result, data=(data[0], data[0]))
    checked = screen(settings, catalog, result)
    assert checked.status is S.INVALID and checked.selected == () and checked.reason_codes


def test_unqualified_or_scaled_assets_cannot_enter_ranked_pool(settings, catalog, monkeypatch):
    data = rows() + [payload("premiumIndex", "1000PEPEUSDT", "100"), payload("premiumIndex", "UNKNOWNUSDT", "100")]
    result, _ = fetch(monkeypatch, data)
    checked = screen(settings, catalog, result, budget=8)
    assert all(item.candidate.symbol not in {"1000PEPEUSDT", "UNKNOWNUSDT"} for item in checked.selected)


@pytest.mark.parametrize("field,value", [("max_candidates", 11), ("max_candidates", 2),
    ("benchmark_symbols", []), ("benchmark_symbols", ["BTCUSDT", "BTCUSDT"]),
    ("min_observed_funding_rate", "-.01"), ("min_observed_funding_rate", "NaN"),
    ("min_observed_funding_rate", 0.1), ("max_next_settlement_wait_seconds", 0)])
def test_config_rejects_unbounded_or_invalid_screen_settings(settings, field, value):
    values = settings.research.funding_screen.model_dump()
    values[field] = value
    with pytest.raises(ValueError):
        FundingScreenSettings.model_validate(values)


def test_new_research_config_keeps_original_fees_and_risk_decisions(settings):
    original = load_settings(Path("docs/research-candidate-costs.toml"))
    assert settings.fees == original.fees
    assert settings.research.decision == original.research.decision
    assert settings.research.data_quality.max_data_age == original.research.data_quality.max_data_age
    assert settings.research.data_quality.max_sampling_skew == original.research.data_quality.max_sampling_skew
    assert settings.research.data_quality.open_interest_time_semantics.value == "last_transaction"
    assert original.research.data_quality.open_interest_time_semantics.value == "exchange_event"
    assert settings.research.validation_scheduler == original.research.validation_scheduler
    assert settings.research.sqlite_path != original.research.sqlite_path
    profile = settings.research.candidate_economics.windows["next_settlement"]
    assert profile.funding_projection.positive_rate_multiplier == D(".5")
    assert profile.funding_events == () and profile.adverse_exit_basis_mode == "entry_offset"


def setup_cli(monkeypatch, settings, catalog, tmp_path, *, fault=None):
    installed = install(monkeypatch)
    for module in (candidate_market, snapshot, candidate_cost_research, cli):
        monkeypatch.setattr(module, "utc_now", lambda: NOW)
    monkeypatch.setattr(cli, "collect_catalog", lambda _: catalog)
    result, _ = fetch(monkeypatch, fault=fault)
    # Restore existing per-symbol HTTP mocks after capturing the bulk fixture.
    install(monkeypatch)
    monkeypatch.setattr(cli.BinanceUsdtmClient, "fetch_premium_catalog", lambda _: result)
    research = settings.research.model_copy(update={"sqlite_path": tmp_path / "research.sqlite"})
    return settings.model_copy(update={"research": research}), installed


def test_full_offline_screen_reuses_original_projection_costs_decision_and_sqlite(settings, catalog, monkeypatch, tmp_path):
    settings, _ = setup_cli(monkeypatch, settings, catalog, tmp_path)
    before = exact_json(settings.research.decision)
    result = cli.research_screen_once(settings, tmp_path / "screen.json")
    assert result["screen_status"] is S.COMPLETE and len(result["outcomes"]) == 6
    records = SQLiteOpportunityRepository(SQLiteConfig(settings.research.sqlite_path)).list_recent_opportunities()
    assert len(records) == 6 and all(len(r.scenarios) == 3 for r in records)
    assert all(r.decision == "REJECT" for r in records)
    assert all("FORECAST_ONLY" in r.details_json and r.net_profit < 0 for r in records)
    assert all(r.funding_rate == D("0.000123456789123456789") for r in records)
    assert all("reason_codes" in item for item in result["outcomes"])
    assert before == exact_json(settings.research.decision)
    assert (tmp_path / "screen.costs.json").is_file()
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.ImportFrom) and ".execution" in node.module for node in ast.walk(tree))


def test_bulk_failure_saves_diagnostics_but_no_opportunity(settings, catalog, monkeypatch, tmp_path):
    settings, _ = setup_cli(monkeypatch, settings, catalog, tmp_path, fault=TimeoutError())
    result = cli.research_screen_once(settings, tmp_path / "fail.json")
    assert result["screen_status"] is S.INVALID and result["outcomes"] == []
    assert not settings.research.sqlite_path.exists()
    report = json.loads((tmp_path / "fail.json").read_text(encoding="utf-8"))
    assert report["premium_source"]["error"]["kind"] == "TIMEOUT"


def test_screen_does_not_overwrite_any_existing_evidence(settings, monkeypatch, tmp_path):
    output = tmp_path / "retained.json"
    output.with_suffix(".costs.json").write_text("original", encoding="utf-8")
    collect = Mock(side_effect=AssertionError("no requests"))
    monkeypatch.setattr(cli, "collect_catalog", collect)
    with pytest.raises(FileExistsError):
        cli.research_screen_once(settings, output)
    assert not collect.called and output.with_suffix(".costs.json").read_text(encoding="utf-8") == "original"


def test_refinement_failure_is_partial_and_does_not_block_other_candidates(settings, catalog, monkeypatch, tmp_path):
    settings, _ = setup_cli(monkeypatch, settings, catalog, tmp_path)
    original = candidate_market.CandidateMarketCollector.collect
    def collect(self, symbol):
        if symbol == "ADAUSDT":
            raise TimeoutError("offline one-candidate collection failure")
        return original(self, symbol)
    monkeypatch.setattr(candidate_market.CandidateMarketCollector, "collect", collect)
    result = cli.research_screen_once(settings, tmp_path / "partial.json")
    assert result["screen_status"] is S.COMPLETE and result["research_status"] is S.PARTIAL
    assert len(result["outcomes"]) == 5
    assert all(item["symbol"] != "ADAUSDT" for item in result["outcomes"])
    report = json.loads((tmp_path / "partial.json").read_text(encoding="utf-8"))
    assert report["research_status"] == "PARTIAL"
    assert result["collection_failures"][0]["symbol"] == "ADAUSDT"


def test_invalid_collected_snapshot_preserved_without_fake_profit(settings, catalog, monkeypatch, tmp_path):
    settings, _ = setup_cli(monkeypatch, settings, catalog, tmp_path)
    original = candidate_market.CandidateMarketCollector.collect
    def collect(self, symbol):
        observation = original(self, symbol)
        if symbol == "ADAUSDT":
            return replace(observation, assembly=replace(observation.assembly, snapshot=None))
        return observation
    monkeypatch.setattr(candidate_market.CandidateMarketCollector, "collect", collect)
    result = cli.research_screen_once(settings, tmp_path / "invalid.json")
    assert result["research_status"] is S.PARTIAL and len(result["outcomes"]) == 5
    report = json.loads((tmp_path / "invalid.costs.json").read_text(encoding="utf-8"))
    failure = next(item for item in report["captures"] if item["symbol"] == "ADAUSDT")
    assert "INCOMPLETE_SNAPSHOT" in failure["quality_reason_codes"]
    assert failure["observation"]["assembly"]["sources"]["spot_depth"]["raw_response"]
    assert "net_profit" not in failure and failure["status"] == "FAILED"


def test_boundary_revalidates_tampered_bulk_request_before_network(monkeypatch):
    request = prepare_premium_catalog_get()
    object.__setattr__(request, "endpoint", PublicEndpoint.SPOT_TIME)
    connection = Mock(side_effect=AssertionError("must fail before I/O"))
    monkeypatch.setattr(binance_usdtm, "open_public_connection", connection)
    with pytest.raises(ValueError):
        BinanceUsdtmClient()._fetch(request, lambda value, at: value)
    connection.assert_not_called()
