"""A forward exit needs fresh current rules for its symbol, not every Spot pair."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
from test_candidate_catalog import NOW, instrument, source
from test_exchange_metadata import Connection, Response

from crypto_arbitrage_bot import candidate_forward_research as forward
from crypto_arbitrage_bot import candidate_research
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.exchanges import exchange_metadata as module
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint as E
from crypto_arbitrage_bot.exchanges.exchange_metadata import MetadataError
from crypto_arbitrage_bot.exchanges.exchange_metadata import MetadataErrorKind as Error
from crypto_arbitrage_bot.exchanges.http_client import RequestPhase, prepare_get
from crypto_arbitrage_bot.market_data.candidate_catalog import (
    CatalogStatus,
    build_candidate_catalog,
)
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument

BTC = ResearchInstrument("BTCUSDT", "BTC")
SOL = ResearchInstrument("SOLUSDT", "SOL")


@pytest.fixture
def settings():
    return load_settings(Path("docs/research-funding-screen.toml"))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("offline only")))
    monkeypatch.setattr(module, "utc_now", lambda: NOW)
    monkeypatch.setattr(candidate_research, "utc_now", lambda: NOW)
    monkeypatch.setattr(forward, "utc_now", lambda: NOW)


def transport(monkeypatch, rows, status=200, fault=None):
    body = json.dumps({"symbols": rows}).encode()
    connection = Connection(Response(body, status))
    if fault is not None:
        connection.response.body = fault
    factory = Mock(return_value=connection)
    monkeypatch.setattr(module, "open_public_connection", factory)
    return connection, factory


@pytest.mark.parametrize("identity", [BTC, SOL, ResearchInstrument("1INCHUSDT", "1INCH")])
def test_scoped_get_only_uses_fixed_public_parameters_and_preserves_raw(monkeypatch, identity):
    connection, factory = transport(monkeypatch, [instrument(identity.base_asset)])
    result = module.BinanceMetadataClient(timeout_seconds=3).fetch(E.SPOT_EXCHANGE_INFO,
                                                             research_instrument=identity)
    assert result.error is None and result.research_instrument == identity
    assert result.raw_response == connection.response.body
    assert result.requested_at <= result.received_at <= result.finished_at
    args, kwargs = connection.requests[0]
    route = urlsplit(args[1])
    assert args[0] == "GET" and route.path == "/api/v3/exchangeInfo"
    assert parse_qs(route.query) == {"symbol": [identity.symbol], "showPermissionSets": ["false"]}
    assert set(kwargs["headers"]) == {"Accept", "User-Agent"}
    assert factory.call_args.args == (E.SPOT_EXCHANGE_INFO.value,)
    assert connection.closed


@pytest.mark.parametrize("identity,endpoint", [(BTC,E.PERP_EXCHANGE_INFO), ("BTCUSDT",E.SPOT_EXCHANGE_INFO),
    ({"symbol":"BTCUSDT"},E.SPOT_EXCHANGE_INFO), (BTC,"https://api.binance.com/api/v3/account"),
    (BTC,E.SPOT_DEPTH)])
def test_scope_cannot_widen_routes_or_accept_untyped_identity(monkeypatch, identity, endpoint):
    connection, factory = transport(monkeypatch, [])
    with pytest.raises(ValueError):
        module.BinanceMetadataClient(timeout_seconds=3).fetch(endpoint, research_instrument=identity)
    factory.assert_not_called()
    assert not connection.requests


@pytest.mark.parametrize("rows", [[], [instrument("ETH")], [instrument("BTC"),instrument("ETH")],
                                 [None], [{"baseAsset":"BTC"}], [instrument("BTC"),instrument("BTC")]])
def test_missing_extra_or_wrong_symbol_payload_is_not_scope_success(monkeypatch, rows):
    connection, _ = transport(monkeypatch, rows)
    result = module.BinanceMetadataClient(timeout_seconds=3).fetch(E.SPOT_EXCHANGE_INFO, research_instrument=BTC)
    assert result.error.kind is Error.INVALID_RESPONSE
    assert result.raw_response == connection.response.body


@pytest.mark.parametrize("status,kind", [(400,Error.HTTP_ERROR),(429,Error.RATE_LIMIT),(503,Error.HTTP_ERROR)])
def test_scoped_http_failure_has_existing_failure_semantics(monkeypatch, status, kind):
    _, factory = transport(monkeypatch, [instrument("BTC")], status)
    result = module.BinanceMetadataClient(timeout_seconds=3).fetch(E.SPOT_EXCHANGE_INFO, research_instrument=BTC)
    assert result.error.kind is kind and factory.call_count == 1


@pytest.mark.parametrize("fault,kind", [(TimeoutError(),Error.TIMEOUT),(ConnectionResetError(),Error.NETWORK_ERROR)])
def test_scoped_network_failure_never_falls_back(monkeypatch, fault, kind):
    _, factory = transport(monkeypatch, [instrument("BTC")], fault=fault)
    result = module.BinanceMetadataClient(timeout_seconds=3).fetch(E.SPOT_EXCHANGE_INFO, research_instrument=BTC)
    assert result.error.kind is kind and result.raw_response is None and factory.call_count == 1


def test_legacy_descriptor_and_default_request_remain_unchanged(monkeypatch):
    connection, _ = transport(monkeypatch, [])
    result = module.BinanceMetadataClient(timeout_seconds=3).fetch(E.SPOT_EXCHANGE_INFO)
    assert result.research_instrument is None and connection.requests[0][0][1] == "/api/v3/exchangeInfo"
    assert prepare_get(E.SPOT_EXCHANGE_INFO,{"symbol":"BTCUSDT"}).params == (("symbol","BTCUSDT"),)
    with pytest.raises(ValueError):
        prepare_get(E.SPOT_EXCHANGE_INFO, {"symbol":"SOLUSDT"})


def catalog(settings, identity):
    perp = source([instrument("BTC",True),instrument("ETH",True),instrument("SOL",True)],True)
    spot = replace(source([instrument(identity.base_asset)]), research_instrument=identity)
    return build_candidate_catalog(spot,perp,settings.research.candidate_catalog,now=NOW)


def test_scoped_qualification_is_identical_for_requested_asset(settings):
    whole = build_candidate_catalog(source([instrument("BTC"),instrument("ETH"),instrument("SOL")]),
        source([instrument("BTC",True),instrument("ETH",True),instrument("SOL",True)],True),
        settings.research.candidate_catalog,now=NOW)
    scoped = catalog(settings,SOL)
    assert scoped.status is CatalogStatus.COMPLETE and len(scoped.candidates) == 1
    assert scoped.candidates[0] == next(c for c in whole.candidates if c.symbol == SOL.symbol)
    wrong = replace(scoped.sources[0],raw_response=source([instrument("BTC")]).raw_response)
    assert build_candidate_catalog(wrong,scoped.sources[1],settings.research.candidate_catalog,now=NOW).status is CatalogStatus.INVALID


@pytest.mark.parametrize("field,value", [("status","HALT"),("isSpotTradingAllowed",False),("filters",[]),
                                         ("quoteAsset","USDC"),("baseAsset","OTHER")])
def test_scoped_metadata_does_not_bypass_original_eligibility(settings, field, value):
    current = catalog(settings,SOL)
    row = instrument("SOL")
    row[field] = value
    spot = replace(current.sources[0],raw_response=source([row]).raw_response)
    result = build_candidate_catalog(spot,current.sources[1],settings.research.candidate_catalog,now=NOW)
    assert not result.candidates and result.exclusions


def test_collect_catalog_scopes_only_spot_and_keeps_original_perp_rules(settings):
    transport = Mock()
    identity = SOL
    scoped = catalog(settings,identity)
    transport.fetch.side_effect = lambda endpoint,**kw: scoped.sources[0 if endpoint is E.SPOT_EXCHANGE_INFO else 1]
    result = candidate_research.collect_catalog(settings,client=transport,spot_instrument=identity)
    assert result == scoped
    transport.fetch.assert_any_call(E.SPOT_EXCHANGE_INFO,research_instrument=identity)
    transport.fetch.assert_any_call(E.PERP_EXCHANGE_INFO)


def test_forward_switches_scope_without_reusing_other_symbol_catalog(settings,monkeypatch):
    collect = Mock(side_effect=lambda _,spot_instrument: catalog(settings,spot_instrument))
    def factory(current,*args,**kwargs):
        symbol = current.candidates[0].symbol
        instance = Mock()
        instance.collect.return_value = Mock(candidate=Mock(symbol=symbol),assembly=Mock(symbol=symbol))
        return instance
    monkeypatch.setattr(forward,"collect_catalog",collect)
    monkeypatch.setattr(forward,"CandidateMarketCollector",factory)
    bridge = forward.CandidateValidationSources(settings)
    bridge.collect("BTCUSDT")
    bridge.fetch_window("BTCUSDT",NOW-timedelta(hours=1),NOW)
    assert collect.call_count == 1
    bridge.collect("SOLUSDT")
    assert collect.call_count == 2 and bridge.spot_instrument == SOL
    assert collect.call_args.kwargs == {"spot_instrument":SOL}
    collect.side_effect = TimeoutError("new scope refresh failed")
    with pytest.raises(TimeoutError):
        bridge.collect("BTCUSDT")
    assert bridge.collector is None and bridge.catalog is None


def test_fresh_scope_still_checks_original_request_age(settings,monkeypatch):
    bridge = forward.CandidateValidationSources(settings)
    bridge._select_instrument("SOLUSDT")
    bridge.catalog = catalog(settings,SOL)
    bridge.collector = Mock()
    monkeypatch.setattr(forward,"utc_now",lambda:NOW+timedelta(seconds=181))
    collect = Mock(side_effect=TimeoutError("original age requires refresh"))
    monkeypatch.setattr(forward,"collect_catalog",collect)
    with pytest.raises(TimeoutError):
        bridge._current()
    assert collect.call_args.kwargs == {"spot_instrument":SOL}
    assert bridge.catalog is None and bridge.collector is None


def test_failed_refresh_retains_raw_diagnostics_without_caching_invalid_rules(settings,monkeypatch):
    good = catalog(settings,SOL)
    bad = replace(good.sources[0],error=MetadataError(Error.TIMEOUT,RequestPhase.RESPONSE_BODY))
    failed = build_candidate_catalog(bad,good.sources[1],settings.research.candidate_catalog,now=NOW)
    assert failed.status is CatalogStatus.INVALID
    monkeypatch.setattr(forward,"collect_catalog",Mock(return_value=failed))
    bridge = forward.CandidateValidationSources(settings)
    with pytest.raises(ValueError):
        bridge.collect("SOLUSDT")
    assert bridge.catalog is None and bridge.collector is None
    assert bridge.catalogs == [failed] and bridge.catalogs[0].sources[0].raw_response == bad.raw_response
    assert bridge.catalogs[0].sources[0].error.kind is Error.TIMEOUT
