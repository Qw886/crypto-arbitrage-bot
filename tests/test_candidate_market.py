"""Qualified research assets retain source provenance without execution authority."""

import copy
import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
from test_candidate_catalog import MS, NOW, instrument, source

from crypto_arbitrage_bot import candidate_market_research as cli
from crypto_arbitrage_bot.arbitrage.execution import simulate_fill
from crypto_arbitrage_bot.arbitrage.funding import accrue_short_funding
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.exchanges import binance_spot, binance_usdtm, funding_history
from crypto_arbitrage_bot.exchanges.binance_spot import BinanceSpotClient
from crypto_arbitrage_bot.exchanges.binance_usdtm import BinanceUsdtmClient
from crypto_arbitrage_bot.exchanges.endpoints import SUPPORTED_SYMBOLS, PublicEndpoint
from crypto_arbitrage_bot.exchanges.funding_history import FundingFailure
from crypto_arbitrage_bot.exchanges.http_client import prepare_get, prepare_research_get
from crypto_arbitrage_bot.execution.types import ExecutionMarket, IocLimitOrder
from crypto_arbitrage_bot.market_data import candidate_market, snapshot
from crypto_arbitrage_bot.market_data.candidate_catalog import (
    CatalogStatus,
    build_candidate_catalog,
)
from crypto_arbitrage_bot.market_data.candidate_market import CandidateMarketCollector
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.market_data.snapshot import SnapshotCollector, assemble_snapshot
from crypto_arbitrage_bot.models.calculation import FillStatus, TradeLeg
from crypto_arbitrage_bot.models.data_quality import DataQualityConfig, DataQualityReason
from crypto_arbitrage_bot.models.market import Market, OrderBook, Side
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.storage.opportunity_records import exact_json

IDENTITY = ResearchInstrument("SOLUSDT", "SOL")
ROUTES = (PublicEndpoint.SPOT_DEPTH, PublicEndpoint.PERP_DEPTH,
          PublicEndpoint.PERP_PREMIUM_INDEX, PublicEndpoint.PERP_OPEN_INTEREST,
          PublicEndpoint.PERP_FUNDING_RATE)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for module in (binance_spot, binance_usdtm, funding_history, candidate_market, snapshot, cli):
        monkeypatch.setattr(module, "utc_now", lambda: NOW)
    for module in (binance_spot, binance_usdtm, funding_history):
        monkeypatch.setattr(module, "open_public_connection", Mock(side_effect=AssertionError("network forbidden")))


@pytest.fixture
def criteria():
    return load_settings(Path("docs/research-candidate-catalog.toml")).research.candidate_catalog


@pytest.fixture
def catalog(criteria):
    assets = ["BTC", "ETH", "SOL"]
    return build_candidate_catalog(source([instrument(a) for a in assets]),
                                   source([instrument(a, True) for a in assets], True), criteria, now=NOW)


def collector(catalog, criteria, **kwargs):
    return CandidateMarketCollector(catalog, criteria, DataQualityConfig(), timeout_seconds=3,
                                    funding_page_limit=kwargs.get("page_limit", 1000),
                                    funding_max_pages=kwargs.get("max_pages", 10))


def payload(kind, symbol="SOLUSDT", rate="0.000123456789123456789"):
    if kind == "depth":
        return {"symbol": symbol, "lastUpdateId": 42, "E": MS, "T": MS,
                "bids": [["99", "1"], ["98", "2"]], "asks": [["101", "1"], ["102", "2"]]}
    if kind == "premiumIndex":
        return {"symbol": symbol, "markPrice": "100.123456789123456789",
                "indexPrice": "100.000000000000000001", "lastFundingRate": rate,
                "nextFundingTime": MS + 3600000, "time": MS}
    if kind == "openInterest":
        return {"symbol": symbol, "openInterest": "12345.123456789123456789", "time": MS}
    return [{"symbol": symbol, "fundingTime": MS - 1,
             "fundingRate": rate, "markPrice": "100.123456789123456789"}]


def response(body, status=200):
    item = Mock()
    item.status = status
    item.getheaders.return_value = ()
    item.read.return_value = body if isinstance(body, bytes) else json.dumps(body).encode()
    connection = Mock()
    connection.getresponse.return_value = item
    return connection


def install(monkeypatch, *, rate="0.000123456789123456789", fault=None):
    calls = []

    def open_connection(url, **kwargs):
        if fault and fault[0] in url:
            raise fault[1]
        connection = Mock()

        def request(method, path, **request_kwargs):
            query = parse_qs(urlsplit(path).query)
            symbol = query["symbol"][0]
            calls.append((method, url, query, kwargs, request_kwargs))
            connection.getresponse.return_value = response(payload(url.rsplit("/", 1)[1], symbol, rate)).getresponse()

        connection.request.side_effect = request
        return connection

    for module in (binance_spot, binance_usdtm, funding_history):
        monkeypatch.setattr(module, "open_public_connection", open_connection)
    return calls


@pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL", "1INCH"])
def test_research_identity_is_explicit_immutable_and_not_trade_authorization(asset):
    identity = ResearchInstrument(asset + "USDT", asset)
    assert identity.symbol == asset + "USDT"
    with pytest.raises(AttributeError):
        identity.base_asset = "BTC"
    if asset not in ("BTC", "ETH"):
        with pytest.raises(ValueError):
            prepare_get(PublicEndpoint.SPOT_DEPTH, {"symbol": identity.symbol})
        with pytest.raises(ValueError):
            IocLimitOrder(ExecutionMarket.SPOT, identity.symbol, Side.BUY, Decimal("1"), Decimal("100"), "research")
    assert set(SUPPORTED_SYMBOLS) == {"BTCUSDT", "ETHUSDT"}


@pytest.mark.parametrize("symbol,asset", [("SOLUSDT", "ETH"), ("solUSDT", "sol"), ("SOLUSDC", "SOL"),
    ("SOLUSDT&x=y", "SOL"), ("1000PEPEUSDT", "1000PEPE"), ("1MBABYDOGEUSDT", "1MBABYDOGE"),
    ("USDT", ""), (None, "SOL"), ("SOLUSDT", 1)])
def test_invalid_and_scaled_identities_rejected(symbol, asset):
    with pytest.raises(ValueError):
        ResearchInstrument(symbol, asset)


@pytest.mark.parametrize("endpoint", ROUTES)
def test_only_five_existing_public_get_routes_accept_explicit_identity(endpoint):
    request = prepare_research_get(endpoint, IDENTITY, {"symbol": "SOLUSDT"})
    assert request.research_instrument == IDENTITY
    assert request.method == "GET"
    assert request.url == endpoint.value
    with pytest.raises(ValueError):
        prepare_research_get(endpoint, IDENTITY, {"symbol": "ETHUSDT"})
    with pytest.raises(ValueError):
        prepare_get(endpoint, {"symbol": "SOLUSDT"})


@pytest.mark.parametrize("endpoint", [e for e in PublicEndpoint if e not in ROUTES])
def test_research_context_does_not_expand_endpoint_allowlist(endpoint):
    with pytest.raises(ValueError):
        prepare_research_get(endpoint, IDENTITY, {"symbol": "SOLUSDT"})


@pytest.mark.parametrize("params", [{"symbol": "SOLUSDT", "signature": "secret"},
    {"symbol": "SOLUSDT", "limit": True}, {"symbol": "SOLUSDT", "limit": 0},
    {"symbol": "SOLUSDT", "startTime": -1}, {"symbol": "SOLUSDT", "limit": 1001},
    {"symbol": "SOLUSDT", "startTime": 2, "endTime": 1}])
def test_research_requests_retain_parameter_boundary(params):
    with pytest.raises((ValueError, TypeError)):
        prepare_research_get(PublicEndpoint.PERP_FUNDING_RATE, IDENTITY, params)


@pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
@pytest.mark.parametrize("rate", ["0.000123456789123456789", "0", "-0.000123456789123456789"])
def test_public_observation_preserves_decimal_symbol_raw_and_official_times(monkeypatch, catalog, criteria, asset, rate):
    calls = install(monkeypatch, rate=rate)
    observed = collector(catalog, criteria).collect(asset + "USDT")
    assert observed.assembly.complete and observed.data_quality.valid
    book = observed.assembly.snapshot
    assert book.symbol == observed.candidate.symbol == asset + "USDT"
    assert book.spot_book.base_asset == book.perp_book.base_asset == asset
    assert book.funding_rate == Decimal(rate)
    assert book.mark_price == Decimal("100.123456789123456789")
    assert book.index_price == Decimal("100.000000000000000001")
    assert book.open_interest == Decimal("12345.123456789123456789")
    assert book.spot_book.event_time is None and book.perp_book.event_time == NOW
    assert book.next_funding_time == NOW + timedelta(hours=1)
    assert book.spot_book.best_ask == Decimal("101") and book.perp_book.best_bid == Decimal("99")
    assert len(book.spot_book.asks) == len(book.perp_book.bids) == 2
    assert observed.assembly.spot_perp_sample_delta == timedelta(0)
    raw = observed.assembly.sources.spot_depth.raw_response
    assert json.loads(raw)["asks"][0] == ["101", "1"]
    assert len(calls) == 4 and all(c[0] == "GET" for c in calls)
    assert all(c[2]["symbol"] == [asset + "USDT"] for c in calls)
    assert all(c[3]["timeout"] == 3 for c in calls)
    assert all("Authorization" not in c[4]["headers"] and "X-MBX-APIKEY" not in c[4]["headers"] for c in calls)
    assert all(value.started_at == value.received_at == value.finished_at == NOW for value in (
        observed.assembly.sources.spot_depth, observed.assembly.sources.perp_depth,
        observed.assembly.sources.premium_index, observed.assembly.sources.open_interest))


def test_existing_vwap_handles_actual_research_identity_recurring_average_and_insufficient_depth(monkeypatch, catalog, criteria):
    install(monkeypatch)
    observed = collector(catalog, criteria).collect("SOLUSDT")
    book = observed.assembly.snapshot.spot_book
    filled = simulate_fill(book, TradeLeg.SPOT_ENTRY, Decimal("3"))
    with localcontext() as context:
        context.prec = 50
        assert filled.vwap == Decimal("305") / Decimal("3")
    assert filled.symbol == "SOLUSDT" and filled.quote_notional == Decimal("305")
    insufficient = simulate_fill(book, TradeLeg.SPOT_ENTRY, Decimal("4"))
    assert insufficient.status is FillStatus.INSUFFICIENT_DEPTH
    assert not insufficient.complete and insufficient.vwap is None
    assert insufficient.filled_quantity == Decimal("3")


@pytest.mark.parametrize("fragment", ["api/v3/depth", "fapi/v1/depth", "premiumIndex", "openInterest"])
@pytest.mark.parametrize("error", [TimeoutError("timed out"), OSError("network down")])
def test_failed_source_never_becomes_complete_market_data(monkeypatch, catalog, criteria, fragment, error):
    calls = install(monkeypatch, fault=(fragment, error))
    observed = collector(catalog, criteria).collect("SOLUSDT")
    assert observed.assembly.snapshot is None and not observed.assembly.complete
    assert not observed.data_quality.valid
    assert DataQualityReason.INCOMPLETE_SNAPSHOT in observed.data_quality.reason_codes
    assert len(calls) == 3


@pytest.mark.parametrize("module,call", [(binance_spot, "spot"), (binance_usdtm, "perp")])
@pytest.mark.parametrize("body,status,kind", [(b"{", 200, "INVALID_RESPONSE"), (b"", 200, "INVALID_RESPONSE"),
    (b"{}", 200, "INVALID_RESPONSE"), (b"{}", 429, "RATE_LIMITED"), (b"{}", 500, "HTTP_ERROR")])
def test_existing_failure_semantics_apply_to_research_assets(monkeypatch, module, call, body, status, kind):
    monkeypatch.setattr(module, "open_public_connection", Mock(return_value=response(body, status)))
    result = (BinanceSpotClient(research_instrument=IDENTITY) if call == "spot" else
              BinanceUsdtmClient(research_instrument=IDENTITY)).fetch_order_book("SOLUSDT")
    assert not result.ok and result.error.kind.value == kind
    assert getattr(result, "order_book", getattr(result, "data", None)) is None


def test_unadmitted_asset_and_invalid_catalog_rejected_before_io(catalog, criteria):
    client = collector(catalog, criteria)
    with pytest.raises(ValueError):
        client.collect("XRPUSDT")
    with pytest.raises(ValueError):
        collector(replace(catalog, status=CatalogStatus.INVALID), criteria)
    fake = replace(catalog, candidates=tuple(replace(c, base_asset="BTC") if c.symbol == "SOLUSDT" else c for c in catalog.candidates))
    with pytest.raises(ValueError):
        collector(fake, criteria)


def test_expired_metadata_and_future_history_are_rejected_before_io(monkeypatch, catalog, criteria):
    client = collector(catalog, criteria)
    with pytest.raises(ValueError):
        client.fetch_funding_window("SOLUSDT", NOW - timedelta(hours=1), NOW + timedelta(seconds=1))
    monkeypatch.setattr(candidate_market, "utc_now", lambda: NOW + timedelta(seconds=criteria.max_metadata_age_seconds + 1))
    with pytest.raises(ValueError):
        client.collect("SOLUSDT")
    with pytest.raises(ValueError):
        client.fetch_funding_window("SOLUSDT", NOW - timedelta(hours=1), NOW)


def test_wrong_research_context_in_source_fails_assembly_and_quality(monkeypatch, catalog, criteria):
    install(monkeypatch)
    observed = collector(catalog, criteria).collect("SOLUSDT")
    original = observed.assembly
    btc = ResearchInstrument("BTCUSDT", "BTC")
    forged_request = copy.deepcopy(original.sources.spot_depth.request)
    object.__setattr__(forged_request, "research_instrument", btc)
    bad_spot = replace(original.sources.spot_depth, request=forged_request)
    bad_sources = replace(original.sources, spot_depth=bad_spot)
    attempt = assemble_snapshot("SOLUSDT", bad_sources, assembled_at=NOW, research_instrument=IDENTITY)
    assert not attempt.complete and attempt.snapshot is None
    bypass = replace(original, sources=bad_sources)
    assert DataQualityReason.SOURCE_MISMATCH in validate_market_data(bypass, evaluated_at=NOW).reason_codes
    without_context = replace(original, research_instrument=None)
    assert DataQualityReason.SYMBOL_MISMATCH in validate_market_data(without_context, evaluated_at=NOW).reason_codes


def test_native_adapter_cannot_accept_research_request_through_private_helper():
    request = prepare_research_get(PublicEndpoint.PERP_OPEN_INTEREST, IDENTITY, {"symbol": "SOLUSDT"})
    with pytest.raises(ValueError):
        BinanceUsdtmClient()._fetch(request, lambda payload, at: payload)
    with pytest.raises(ValueError):
        SnapshotCollector(BinanceSpotClient(), BinanceUsdtmClient(), research_instrument=IDENTITY)
    with pytest.raises(ValueError):
        OrderBook("SOLUSDT", Market.SPOT, (), (), NOW)


@pytest.mark.parametrize("rate", ["0.0001", "0", "-0.0001"])
def test_observed_history_uses_actual_settlement_mark_and_existing_short_cashflow(monkeypatch, catalog, criteria, rate):
    install(monkeypatch, rate=rate)
    start = NOW - timedelta(hours=8)
    history = collector(catalog, criteria).fetch_funding_window("SOLUSDT", start, NOW)
    assert history.failure is None and len(history.events) == 1
    event = history.events[0]
    assert event.symbol == "SOLUSDT" and event.funding_rate == Decimal(rate)
    assert start < event.settled_at <= NOW
    value = accrue_short_funding(Decimal("2"), start, NOW, history.events, "SOLUSDT")
    assert value == Decimal("2") * Decimal("100.123456789123456789") * Decimal(rate)
    assert history.pages[0].raw_response


def test_empty_successful_history_has_no_manufactured_funding(monkeypatch, catalog, criteria):
    monkeypatch.setattr(funding_history, "open_public_connection", Mock(return_value=response([])))
    start = NOW - timedelta(hours=8)
    history = collector(catalog, criteria).fetch_funding_window("SOLUSDT", start, NOW)
    assert history.failure is None and history.events == ()
    assert accrue_short_funding(Decimal("2"), start, NOW, history.events, "SOLUSDT") == 0


@pytest.mark.parametrize("case", ["entry_boundary", "after_exit", "wrong_symbol", "duplicate_time", "missing_mark"])
def test_history_rejects_unverified_or_outside_window_events_without_partial_accrual(monkeypatch, catalog, criteria, case):
    start = NOW - timedelta(hours=8)
    rows = payload("fundingRate")
    if case == "entry_boundary":
        rows[0]["fundingTime"] = MS - 8 * 3600000
    elif case == "after_exit":
        rows[0]["fundingTime"] = MS + 1
    elif case == "wrong_symbol":
        rows[0]["symbol"] = "ETHUSDT"
    elif case == "duplicate_time":
        rows.append(dict(rows[0]))
    else:
        rows[0].pop("markPrice")
    monkeypatch.setattr(funding_history, "open_public_connection", Mock(return_value=response(rows)))
    history = collector(catalog, criteria).fetch_funding_window("SOLUSDT", start, NOW)
    assert history.failure is FundingFailure.INVALID_RESPONSE and history.events == ()


def test_history_pagination_inclusive_exit_boundary_advances_by_one_millisecond(monkeypatch, catalog, criteria):
    rows = payload("fundingRate")
    rows[0]["fundingTime"] = MS - 3600000
    second = copy.deepcopy(rows)
    second[0]["fundingTime"] = MS
    connections = [response(rows), response(second)]
    factory = Mock(side_effect=connections)
    monkeypatch.setattr(funding_history, "open_public_connection", factory)
    history = collector(catalog, criteria, page_limit=1, max_pages=2).fetch_funding_window("SOLUSDT", NOW - timedelta(hours=8), NOW)
    assert history.failure is None and len(history.events) == 2
    first_query = parse_qs(urlsplit(connections[0].request.call_args.args[1]).query)
    next_query = parse_qs(urlsplit(connections[1].request.call_args.args[1]).query)
    assert int(first_query["startTime"][0]) == MS - 8 * 3600000 + 1
    assert int(next_query["startTime"][0]) == MS - 3600000 + 1
    assert int(next_query["endTime"][0]) == MS
    assert all(c.request.call_args.args[0] == "GET" for c in connections)


def test_finite_history_pages_never_expose_partial_events(monkeypatch, catalog, criteria):
    monkeypatch.setattr(funding_history, "open_public_connection", Mock(return_value=response(payload("fundingRate"))))
    result = collector(catalog, criteria, page_limit=1, max_pages=1).fetch_funding_window("SOLUSDT", NOW - timedelta(hours=8), NOW)
    assert result.failure is FundingFailure.PAGE_LIMIT and result.events == () and len(result.pages) == 1


def test_one_shot_cli_preserves_anonymous_evidence_and_never_overwrites(monkeypatch, catalog, tmp_path):
    install(monkeypatch)
    monkeypatch.setattr(cli, "collect_catalog", lambda settings: catalog)
    output = tmp_path / "capture.json"
    args = ["--config", "docs/research-candidate-catalog.toml", "--symbols", "BTCUSDT", "ETHUSDT", "SOLUSDT",
            "--history-seconds", "86400", "--output", str(output)]
    assert cli.main(args) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["schema_version"] == "public-candidate-market-v1"
    assert len(data["captures"]) == 3
    sol = data["captures"][2]["observation"]["assembly"]["snapshot"]
    assert sol["mark_price"] == "100.123456789123456789"
    assert sol["symbol"] == "SOLUSDT" and sol["research_instrument"]["base_asset"] == "SOL"
    before = output.read_bytes()
    assert cli.main(args) == 2 and output.read_bytes() == before


def test_serialization_preserves_full_public_provenance(monkeypatch, catalog, criteria):
    install(monkeypatch)
    result = collector(catalog, criteria).collect("SOLUSDT")
    body = json.loads(exact_json(result))
    assert body["assembly"]["sources"]["spot_depth"]["raw_response"]["base64"]
    assert body["assembly"]["sources"]["spot_depth"]["order_book"]["event_time"] is None
    assert body["assembly"]["sources"]["perp_depth"]["data"]["transaction_time"].endswith("+00:00")


@pytest.mark.parametrize("module", [binance_spot, binance_usdtm, funding_history])
def test_adapter_rejects_untyped_identity_before_io(module):
    with pytest.raises(TypeError):
        if module is funding_history:
            module.BinanceFundingHistoryClient(timeout_seconds=3, page_limit=1000,
                                               max_pages=2, research_instrument="SOLUSDT")
        else:
            cls = module.BinanceSpotClient if module is binance_spot else module.BinanceUsdtmClient
            cls(research_instrument="SOLUSDT")


def test_cli_one_symbol_failure_does_not_discard_other_anonymous_captures(monkeypatch, catalog, tmp_path):
    install(monkeypatch)
    monkeypatch.setattr(cli, "collect_catalog", lambda settings: catalog)
    output = tmp_path / "mixed.json"
    assert cli.main(["--config", "docs/research-candidate-catalog.toml", "--symbols", "XRPUSDT", "SOLUSDT",
                     "--history-seconds", "3600", "--output", str(output)]) == 1
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["captures"][0]["error_type"] == "ValueError"
    assert data["captures"][1]["observation"]["assembly"]["symbol"] == "SOLUSDT"


def test_contract_lifetime_is_checked_again_without_stale_metadata(monkeypatch, criteria):
    perp = instrument("SOL", True)
    perp["deliveryDate"] = MS + (criteria.min_remaining_lifetime_seconds + 10) * 1000
    catalog = build_candidate_catalog(source([instrument("SOL")]), source([perp], True), criteria, now=NOW)
    client = collector(catalog, criteria)
    monkeypatch.setattr(candidate_market, "utc_now", lambda: NOW + timedelta(seconds=11))
    with pytest.raises(ValueError, match="lifetime"):
        client.collect("SOLUSDT")
