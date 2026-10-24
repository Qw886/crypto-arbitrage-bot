"""Offline composition tests using real adapters and mocked HTTP responses."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest

from crypto_arbitrage_bot.exchanges import binance_spot, binance_usdtm
from crypto_arbitrage_bot.exchanges.binance_spot import BinanceSpotClient
from crypto_arbitrage_bot.exchanges.binance_usdtm import BinanceUsdtmClient
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import prepare_get
from crypto_arbitrage_bot.market_data import snapshot as snapshot_module
from crypto_arbitrage_bot.market_data.snapshot import SnapshotCollector, assemble_snapshot
from crypto_arbitrage_bot.models.market import Market
from crypto_arbitrage_bot.models.snapshot import (
    SnapshotIssueKind,
    SnapshotSource,
    SnapshotSources,
    SnapshotStatus,
)
from crypto_arbitrage_bot.models.spot import SpotErrorKind
from crypto_arbitrage_bot.models.usdtm import UsdtmErrorKind

FIXTURES = Path(__file__).parent / "fixtures"
AT = datetime(2026, 9, 20, 12, tzinfo=UTC)
SOURCE_NAMES = ("spot_depth", "perp_depth", "premium_index", "open_interest")
SOURCE_KINDS = tuple(SnapshotSource)
SOURCES = tuple(zip(SOURCE_NAMES, SOURCE_KINDS, strict=True))


@pytest.fixture(autouse=True)
def transports(monkeypatch):
    factories = tuple(
        Mock(side_effect=AssertionError("Real network is forbidden in these tests"))
        for _ in range(2)
    )
    monkeypatch.setattr(binance_spot, "HTTPSConnection", factories[0])
    monkeypatch.setattr(binance_usdtm, "HTTPSConnection", factories[1])
    return factories


def fixture_bodies(symbol):
    asset = symbol[:3].lower()
    return (
        (FIXTURES / f"spot_depth_{asset}.json").read_bytes(),
        (FIXTURES / f"usdtm_depth_{asset}.json").read_bytes(),
        (FIXTURES / f"usdtm_premium_{asset}.json").read_bytes(),
        (FIXTURES / f"usdtm_open_interest_{asset}.json").read_bytes(),
    )


def response_connection(body):
    response = Mock()
    response.status = 200
    response.getheaders.return_value = (
        ("Content-Type", "application/json"), ("Content-Length", str(len(body)))
    )
    response.read.return_value = body
    connection = Mock()
    connection.getresponse.return_value = response
    return connection


def install_attempt(transports, bodies):
    connections = tuple(response_connection(body) for body in bodies)
    transports[0].side_effect = [connections[0]]
    transports[1].side_effect = list(connections[1:])
    return connections


@pytest.fixture
def sources(transports):
    install_attempt(transports, fixture_bodies("BTCUSDT"))
    spot = BinanceSpotClient()
    perp = BinanceUsdtmClient()
    return SnapshotSources(
        spot_depth=spot.fetch_order_book("BTCUSDT"),
        perp_depth=perp.fetch_order_book("BTCUSDT"),
        premium_index=perp.fetch_premium_index("BTCUSDT"),
        open_interest=perp.fetch_open_interest("BTCUSDT"),
    )


def assert_incomplete(result, source, kind):
    assert result.complete is False
    assert result.status is SnapshotStatus.INCOMPLETE
    assert result.snapshot is None
    assert any(issue.source is source and issue.kind is kind for issue in result.issues)


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_collector_builds_complete_decimal_snapshot_from_four_public_gets(
    transports, monkeypatch, symbol
):
    bodies = fixture_bodies(symbol)
    connections = install_attempt(transports, bodies)
    monkeypatch.setattr(snapshot_module, "utc_now", lambda: AT)

    result = SnapshotCollector(BinanceSpotClient(), BinanceUsdtmClient()).collect(
        symbol, spot_limit=50, perp_limit=20
    )

    assert result.complete is True
    assert result.status is SnapshotStatus.COMPLETE
    assert result.symbol == result.snapshot.symbol == symbol
    assert result.issues == ()
    assert result.assembled_at == result.snapshot.observed_at == AT
    snapshot = result.snapshot
    spot, depth, premium, interest = (getattr(result.sources, name) for name in SOURCE_NAMES)
    assert snapshot.spot_book is spot.order_book
    assert snapshot.perp_book is depth.data.order_book
    assert snapshot.mark_price == premium.data.mark_price
    assert snapshot.index_price == premium.data.index_price
    assert snapshot.funding_rate == premium.data.last_funding_rate
    assert snapshot.next_funding_time == premium.data.next_funding_time
    assert snapshot.open_interest == interest.data.open_interest
    for value in (
        snapshot.mark_price, snapshot.index_price, snapshot.funding_rate,
        snapshot.open_interest, snapshot.spot_book.best_bid, snapshot.spot_book.best_ask,
        snapshot.perp_book.best_bid, snapshot.perp_book.best_ask,
    ):
        assert type(value) is Decimal
    for book, body in ((snapshot.spot_book, bodies[0]), (snapshot.perp_book, bodies[1])):
        payload = json.loads(body)
        for side in ("bids", "asks"):
            levels = getattr(book, side)
            assert [(level.price, level.quantity) for level in levels] == [
                (Decimal(price), Decimal(quantity)) for price, quantity in payload[side]
            ]
            assert all(type(value) is Decimal for level in levels
                       for value in (level.price, level.quantity))
    if symbol == "BTCUSDT":
        assert snapshot.mark_price == Decimal("65000.123456789123456789")
        assert snapshot.index_price == Decimal("64999.987654321987654321")
        assert snapshot.funding_rate == Decimal("0.000123456789123456789")
    assert snapshot.spot_book.event_time is None
    assert not hasattr(snapshot, "event_time")
    assert not hasattr(result, "event_time")
    assert result.spot_perp_sample_delta == depth.received_at - spot.received_at
    assert result.spot_perp_sample_gap == abs(depth.received_at - spot.received_at)
    assert transports[0].call_count == 1
    assert transports[1].call_count == 3
    paths = ("/api/v3/depth", "/fapi/v1/depth", "/fapi/v1/premiumIndex",
             "/fapi/v1/openInterest")
    for source, body, connection, path in zip(
        (spot, depth, premium, interest), bodies, connections, paths, strict=True
    ):
        assert source.raw_response == body
        assert source.started_at <= source.received_at <= source.finished_at
        assert source.request.method == "GET"
        connection.request.assert_called_once()
        args, kwargs = connection.request.call_args
        assert args[0] == "GET"
        parsed = urlsplit(args[1])
        assert parsed.path == path
        expected = {"symbol": [symbol]}
        if source is spot:
            expected["limit"] = ["50"]
        elif source is depth:
            expected["limit"] = ["20"]
        assert parse_qs(parsed.query) == expected
        assert kwargs.get("body") is None
        connection.close.assert_called_once()


@pytest.mark.parametrize("name,source", SOURCES)
def test_missing_source_is_incomplete_without_zero_fill(sources, name, source):
    supplied = replace(sources, **{name: None})
    result = assemble_snapshot("BTCUSDT", supplied, assembled_at=AT)
    assert_incomplete(result, source, SnapshotIssueKind.MISSING)
    assert result.sources is supplied
    assert getattr(result.sources, name) is None
    for other in set(SOURCE_NAMES) - {name}:
        assert getattr(result.sources, other) is getattr(sources, other)
    if name in ("spot_depth", "perp_depth"):
        assert result.spot_perp_sample_delta is None
        assert result.spot_perp_sample_gap is None
    else:
        assert result.spot_perp_sample_delta == (
            sources.perp_depth.received_at - sources.spot_depth.received_at
        )


def test_all_missing_sources_are_reported_without_network(transports):
    result = assemble_snapshot("BTCUSDT", SnapshotSources(), assembled_at=AT)
    assert result.status is SnapshotStatus.INCOMPLETE
    assert result.snapshot is None
    assert {issue.source for issue in result.issues} == set(SOURCE_KINDS)
    assert all(issue.kind is SnapshotIssueKind.MISSING for issue in result.issues)
    assert result.spot_perp_sample_delta is None
    for transport in transports:
        transport.assert_not_called()


@pytest.mark.parametrize("rate", ["0.0001", "-0.0002", "0"])
def test_signed_funding_and_actual_zero_open_interest_remain_complete(transports, rate):
    bodies = list(fixture_bodies("BTCUSDT"))
    premium = json.loads(bodies[2])
    interest = json.loads(bodies[3])
    premium["lastFundingRate"] = rate
    interest["openInterest"] = "0"
    bodies[2] = json.dumps(premium).encode()
    bodies[3] = json.dumps(interest).encode()
    install_attempt(transports, bodies)
    result = SnapshotCollector(BinanceSpotClient(), BinanceUsdtmClient()).collect("BTCUSDT")
    assert result.complete
    assert result.snapshot.funding_rate == Decimal(rate)
    assert result.snapshot.open_interest == Decimal("0")


@pytest.mark.parametrize(
    "index,field,name,source",
    [(2, "lastFundingRate", "premium_index", SnapshotSource.PREMIUM_INDEX),
     (3, "openInterest", "open_interest", SnapshotSource.OPEN_INTEREST)],
)
def test_missing_payload_field_preserves_adapter_error_and_raw_response(
    transports, index, field, name, source
):
    bodies = list(fixture_bodies("BTCUSDT"))
    payload = json.loads(bodies[index])
    del payload[field]
    bodies[index] = json.dumps(payload).encode()
    install_attempt(transports, bodies)
    result = SnapshotCollector(BinanceSpotClient(), BinanceUsdtmClient()).collect("BTCUSDT")
    assert_incomplete(result, source, SnapshotIssueKind.SOURCE_ERROR)
    failed = getattr(result.sources, name)
    assert failed.error.kind is UsdtmErrorKind.INVALID_RESPONSE
    assert failed.data is None
    assert failed.raw_response == bodies[index]
    assert failed.received_at is not None
    assert result.spot_perp_sample_delta is not None
    assert transports[1].call_count == 3


@pytest.mark.parametrize("name,source", SOURCES)
def test_parsed_symbol_mismatch_is_never_combined(sources, name, source):
    original = getattr(sources, name)
    if name == "spot_depth":
        mismatched = replace(original, order_book=replace(original.order_book, symbol="ETHUSDT"))
    elif name == "perp_depth":
        depth = replace(original.data, order_book=replace(original.data.order_book, symbol="ETHUSDT"))
        mismatched = replace(original, data=depth)
    else:
        mismatched = replace(original, data=replace(original.data, symbol="ETHUSDT"))
    supplied = replace(sources, **{name: mismatched})
    result = assemble_snapshot("BTCUSDT", supplied, assembled_at=AT)
    assert_incomplete(result, source, SnapshotIssueKind.SYMBOL_MISMATCH)
    assert getattr(result.sources, name) is mismatched
    if name in ("spot_depth", "perp_depth"):
        assert result.spot_perp_sample_delta is None


@pytest.mark.parametrize("name,source", SOURCES)
def test_request_symbol_mismatch_is_never_combined(sources, name, source):
    original = getattr(sources, name)
    params = dict(original.request.params)
    params["symbol"] = "ETHUSDT"
    mismatch = replace(original, request=prepare_get(original.request.endpoint, params))
    result = assemble_snapshot("BTCUSDT", replace(sources, **{name: mismatch}), assembled_at=AT)
    assert_incomplete(result, source, SnapshotIssueKind.SYMBOL_MISMATCH)


@pytest.mark.parametrize("name,source", SOURCES)
def test_wrong_endpoint_cannot_claim_a_required_source(sources, name, source):
    original = getattr(sources, name)
    mismatch = replace(original, request=prepare_get(
        PublicEndpoint.SPOT_BOOK_TICKER, {"symbol": "BTCUSDT"}
    ))
    result = assemble_snapshot("BTCUSDT", replace(sources, **{name: mismatch}), assembled_at=AT)
    assert_incomplete(result, source, SnapshotIssueKind.ENDPOINT_MISMATCH)


@pytest.mark.parametrize("name,source", SOURCES)
def test_missing_receive_time_is_not_invented(sources, name, source):
    missing = replace(getattr(sources, name), received_at=None)
    result = assemble_snapshot("BTCUSDT", replace(sources, **{name: missing}), assembled_at=AT)
    assert_incomplete(result, source, SnapshotIssueKind.MISSING_RECEIVE_TIME)
    assert getattr(result.sources, name).received_at is None


@pytest.mark.parametrize("delta", [timedelta(microseconds=1_234_567),
                                  timedelta(microseconds=-1_234_567), timedelta(0)])
def test_sampling_delta_uses_exact_signed_depth_receive_times(sources, delta):
    supplied = replace(
        sources,
        spot_depth=replace(sources.spot_depth, received_at=AT),
        perp_depth=replace(sources.perp_depth, received_at=AT + delta),
    )
    result = assemble_snapshot("BTCUSDT", supplied, assembled_at=AT + timedelta(minutes=1))
    assert result.complete
    assert result.spot_perp_sample_delta == delta
    assert result.spot_perp_sample_gap == abs(delta)
    assert type(result.spot_perp_sample_delta) is timedelta


def test_all_independent_times_and_raw_source_records_survive_assembly(sources):
    records = {}
    for index, name in enumerate(SOURCE_NAMES):
        received_at = AT + timedelta(milliseconds=index * 111 + 10)
        record = getattr(sources, name)
        records[name] = replace(record, started_at=received_at - timedelta(milliseconds=7),
                                received_at=received_at,
                                finished_at=received_at + timedelta(milliseconds=3))
    supplied = replace(sources, **records)
    assembled_at = AT + timedelta(seconds=20)
    result = assemble_snapshot("BTCUSDT", supplied, assembled_at=assembled_at)
    assert result.complete
    assert result.sources is supplied
    for name, record in records.items():
        assert getattr(result.sources, name) is record
        assert record.raw_response == getattr(sources, name).raw_response
        assert record.started_at != record.received_at != record.finished_at
    depth_payload = json.loads(records["perp_depth"].raw_response)
    premium_payload = json.loads(records["premium_index"].raw_response)
    oi_payload = json.loads(records["open_interest"].raw_response)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    assert result.snapshot.spot_book.event_time is None
    assert result.snapshot.perp_book.event_time == epoch + timedelta(milliseconds=depth_payload["E"])
    assert result.sources.perp_depth.data.transaction_time == epoch + timedelta(
        milliseconds=depth_payload["T"]
    )
    assert result.sources.premium_index.data.event_time == epoch + timedelta(
        milliseconds=premium_payload["time"]
    )
    assert result.sources.open_interest.data.event_time == epoch + timedelta(
        milliseconds=oi_payload["time"]
    )
    assert result.snapshot.next_funding_time == epoch + timedelta(
        milliseconds=premium_payload["nextFundingTime"]
    )
    assert result.snapshot.observed_at == assembled_at


@pytest.mark.parametrize("name,source", [
    ("spot_depth", SnapshotSource.SPOT_DEPTH), ("perp_depth", SnapshotSource.PERP_DEPTH)
])
def test_empty_depth_is_not_a_complete_snapshot(sources, name, source):
    original = getattr(sources, name)
    if name == "spot_depth":
        empty = replace(original, order_book=replace(original.order_book, bids=()))
    else:
        empty = replace(original, data=replace(
            original.data, order_book=replace(original.data.order_book, asks=())
        ))
    result = assemble_snapshot("BTCUSDT", replace(sources, **{name: empty}), assembled_at=AT)
    assert_incomplete(result, source, SnapshotIssueKind.INVALID_DATA)
    assert result.spot_perp_sample_delta is None


def test_spot_slot_requires_spot_market(sources):
    wrong = replace(sources.spot_depth, order_book=replace(
        sources.spot_depth.order_book, market=Market.USDTM
    ))
    result = assemble_snapshot("BTCUSDT", replace(sources, spot_depth=wrong), assembled_at=AT)
    assert_incomplete(result, SnapshotSource.SPOT_DEPTH, SnapshotIssueKind.INVALID_DATA)


def test_perp_depth_slot_requires_depth_data(sources):
    wrong = replace(sources.perp_depth, data=sources.premium_index.data)
    result = assemble_snapshot("BTCUSDT", replace(sources, perp_depth=wrong), assembled_at=AT)
    assert_incomplete(result, SnapshotSource.PERP_DEPTH, SnapshotIssueKind.INVALID_DATA)


def test_collector_construction_does_no_io(transports):
    SnapshotCollector(BinanceSpotClient(), BinanceUsdtmClient())
    for transport in transports:
        transport.assert_not_called()


@pytest.mark.parametrize("symbol,options", [
    ("BNBUSDT", {}), ("BTCUSDT", {"spot_limit": 0}),
    ("BTCUSDT", {"perp_limit": 1}), ("BTCUSDT", {"perp_limit": True}),
])
def test_collector_validates_all_arguments_before_first_request(transports, symbol, options):
    with pytest.raises((ValueError, TypeError)):
        SnapshotCollector(BinanceSpotClient(), BinanceUsdtmClient()).collect(symbol, **options)
    for transport in transports:
        transport.assert_not_called()


def test_failed_attempt_collects_remaining_sources_without_reusing_cached_data(transports):
    collector = SnapshotCollector(BinanceSpotClient(), BinanceUsdtmClient())
    install_attempt(transports, fixture_bodies("BTCUSDT"))
    first = collector.collect("BTCUSDT")
    assert first.complete

    connections = install_attempt(transports, fixture_bodies("BTCUSDT"))
    connections[0].getresponse.side_effect = TimeoutError("offline simulated timeout")
    failed = collector.collect("BTCUSDT")
    assert_incomplete(failed, SnapshotSource.SPOT_DEPTH, SnapshotIssueKind.SOURCE_ERROR)
    assert failed.sources.spot_depth.error.kind is SpotErrorKind.TIMEOUT
    assert failed.sources.spot_depth.order_book is None
    assert failed.sources.spot_depth is not first.sources.spot_depth
    assert failed.spot_perp_sample_delta is None
    for source in (failed.sources.perp_depth, failed.sources.premium_index,
                   failed.sources.open_interest):
        assert source.ok
    for connection in connections:
        connection.request.assert_called_once()

    install_attempt(transports, fixture_bodies("ETHUSDT"))
    recovered = collector.collect("ETHUSDT")
    assert recovered.complete
    assert recovered.snapshot.symbol == "ETHUSDT"
    assert recovered.snapshot.spot_book.symbol == recovered.snapshot.perp_book.symbol == "ETHUSDT"
    assert transports[0].call_count == 3
    assert transports[1].call_count == 9
