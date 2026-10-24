"""Offline transport -> assembly -> data quality -> estimate failure contracts."""

import json
from datetime import UTC, datetime, timedelta
from http.client import IncompleteRead
from socket import gaierror
from unittest.mock import Mock

import pytest
from test_market_snapshot import fixture_bodies, install_attempt
from test_offline_estimate import OPENED, books, config, scenario_for

from crypto_arbitrage_bot.arbitrage import estimate as engine
from crypto_arbitrage_bot.exchanges import binance_spot, binance_usdtm
from crypto_arbitrage_bot.exchanges.binance_spot import BinanceSpotClient
from crypto_arbitrage_bot.exchanges.binance_usdtm import BinanceUsdtmClient
from crypto_arbitrage_bot.exchanges.http_client import RequestPhase
from crypto_arbitrage_bot.market_data import snapshot as assembly_module
from crypto_arbitrage_bot.market_data.snapshot import SnapshotCollector
from crypto_arbitrage_bot.models.estimate import EstimateStatus, ExitFundingScenario
from crypto_arbitrage_bot.models.snapshot import SnapshotStatus

SOURCES = ("spot_depth", "perp_depth", "premium_index", "open_interest")
FAILURES = (
    ("connect_timeout", "TIMEOUT", RequestPhase.REQUEST),
    ("headers_timeout", "TIMEOUT", RequestPhase.RESPONSE_HEADERS),
    ("read_timeout", "TIMEOUT", RequestPhase.RESPONSE_BODY),
    ("dns", "NETWORK_ERROR", RequestPhase.REQUEST),
    ("connection", "NETWORK_ERROR", RequestPhase.REQUEST),
    ("reset", "NETWORK_ERROR", RequestPhase.RESPONSE_BODY),
    ("partial", "NETWORK_ERROR", RequestPhase.RESPONSE_BODY),
    ("400", "HTTP_ERROR", RequestPhase.RESPONSE_BODY),
    ("429", "RATE_LIMITED", RequestPhase.RESPONSE_BODY),
    ("503", "HTTP_ERROR", RequestPhase.RESPONSE_BODY),
    ("invalid_json", "INVALID_RESPONSE", RequestPhase.PARSE),
    ("empty", "INVALID_RESPONSE", RequestPhase.PARSE),
    ("missing_payload", "INVALID_RESPONSE", RequestPhase.PARSE),
    ("api_error", "API_ERROR", RequestPhase.PARSE),
    ("close", "NETWORK_ERROR", RequestPhase.CLOSE),
)


@pytest.fixture
def transport(monkeypatch):
    factories = (Mock(), Mock())
    for module, factory in zip((binance_spot, binance_usdtm), factories, strict=True):
        factory.side_effect = AssertionError("Unexpected network call")
        monkeypatch.setattr(module, "HTTPSConnection", factory)
        monkeypatch.setattr(module, "utc_now", lambda: OPENED)
    monkeypatch.setattr(assembly_module, "utc_now", lambda: OPENED)
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("No network")))
    return factories


def install(transport, symbol="BTCUSDT"):
    # Existing fixtures, with explicit synthetic exchange timestamps for replay.
    millis = (OPENED - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(milliseconds=1)
    payloads = [json.loads(body) for body in fixture_bodies(symbol)]
    payloads[1].update(E=millis, T=millis)
    payloads[2].update(time=millis, nextFundingTime=millis + 28_800_000)
    payloads[3].update(time=millis)
    return install_attempt(transport, [json.dumps(p).encode() for p in payloads])


def inject(connection, failure):
    response = connection.getresponse.return_value
    if failure == "connect_timeout":
        connection.request.side_effect = TimeoutError("connect")
    elif failure == "headers_timeout":
        connection.getresponse.side_effect = TimeoutError("headers")
    elif failure == "read_timeout":
        response.read.side_effect = TimeoutError("body")
    elif failure == "dns":
        connection.request.side_effect = gaierror("DNS unavailable")
    elif failure == "connection":
        connection.request.side_effect = ConnectionRefusedError("refused")
    elif failure == "reset":
        response.read.side_effect = ConnectionResetError("reset")
    elif failure == "partial":
        response.read.side_effect = IncompleteRead(b'{"partial":', 20)
    elif failure == "close":
        connection.close.side_effect = OSError("close failed")
    else:
        if failure.isdigit():
            response.status = int(failure)
            body = b'{"code":-1000,"msg":"simulated HTTP failure"}'
        else:
            body = {
                "invalid_json": b"{not json", "empty": b"", "missing_payload": b"{}",
                "api_error": b'{"code":-1121,"msg":"simulated API failure"}',
            }[failure]
        response.read.return_value = body
        response.getheaders.return_value = (("Content-Length", str(len(body))), ("Retry-After", "7"))


def collect():
    return SnapshotCollector(BinanceSpotClient(), BinanceUsdtmClient()).collect("BTCUSDT")


@pytest.mark.parametrize("source_index", range(4), ids=SOURCES)
@pytest.mark.parametrize("failure,kind,phase", FAILURES)
def test_every_source_failure_blocks_profit(transport, monkeypatch, source_index, failure, kind, phase):
    connections = install(transport)
    inject(connections[source_index], failure)
    assembly = collect()
    assert assembly.status is SnapshotStatus.INCOMPLETE and assembly.snapshot is None
    failed = getattr(assembly.sources, SOURCES[source_index])
    assert not failed.ok and failed.error.kind.value == kind
    assert failed.error.phase is phase
    assert (failed.order_book if source_index == 0 else failed.data) is None
    if source_index == 0:
        assert failed.last_update_id is None
    if failure == "close":
        assert failed.cleanup_error is not None
    if failure == "429":
        assert ("Retry-After", "7") in failed.response_headers
    forbidden = Mock(side_effect=AssertionError("Failed collection reached math"))
    for name in ("size_position", "simulate_fill", "accrue_short_funding", "round_trip_from_fills"):
        monkeypatch.setattr(engine, name, forbidden)
    result = engine.estimate_once(assembly, config(), scenario_for(books()))
    assert result.status is EstimateStatus.INCOMPLETE_SNAPSHOT
    assert not result.data_quality.valid
    assert result.net_profit is result.net_return is result.calculation is None
    assert result.fills == ()
    forbidden.assert_not_called()
    assert transport[0].call_count == 1 and transport[1].call_count == 3
    for connection in connections:
        connection.request.assert_called_once()
        assert connection.request.call_args.args[0] == "GET"
        connection.close.assert_called_once()


@pytest.mark.parametrize("source_index", range(4), ids=SOURCES)
@pytest.mark.parametrize("body_failure", ["read_timeout", "partial", "oversized", "bad_length"])
def test_rate_limit_survives_broken_body(transport, source_index, body_failure):
    connections = install(transport)
    connection = connections[source_index]
    response = connection.getresponse.return_value
    inject(connection, "429")
    if body_failure in ("read_timeout", "partial"):
        inject(connection, body_failure)
    elif body_failure == "oversized":
        response.read.return_value = b" " * (binance_spot.MAX_BODY_BYTES + 1)
    else:
        response.getheaders.return_value = (("Content-Length", "bad"), ("Retry-After", "7"))
    assembly = collect()
    failed = getattr(assembly.sources, SOURCES[source_index])
    assert failed.error.kind.value == "RATE_LIMITED" and failed.status_code == 429
    assert not assembly.complete
    assert ("Retry-After", "7") in failed.response_headers
    result = engine.estimate_once(assembly, config(), scenario_for(books()))
    assert result.net_profit is result.net_return is None


@pytest.mark.parametrize("source_index", range(4), ids=SOURCES)
def test_primary_failure_and_cleanup_failure_both_preserved(transport, source_index):
    connections = install(transport)
    inject(connections[source_index], "connect_timeout")
    inject(connections[source_index], "close")
    assembly = collect()
    failed = getattr(assembly.sources, SOURCES[source_index])
    assert failed.error.kind.value == "TIMEOUT"
    assert failed.error.phase is RequestPhase.REQUEST
    assert failed.cleanup_error.kind.value == "NETWORK_ERROR"
    assert failed.cleanup_error.phase is RequestPhase.CLOSE


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_success_failure_success_never_reuses_old_snapshot(transport, symbol):
    from decimal import Decimal

    collector = SnapshotCollector(BinanceSpotClient(), BinanceUsdtmClient())
    install(transport, symbol)
    first = collector.collect(symbol)
    assert first.complete
    settings = config(target_quantity=Decimal("0.1"))
    scenario = ExitFundingScenario(
        first.snapshot.spot_book, first.snapshot.perp_book,
        OPENED, OPENED + timedelta(hours=1), (), "Explicit static exit; no funding events",
    )
    original = engine.estimate_once(first, settings, scenario)
    assert original.complete and original.data_quality.valid
    connections = install(transport, symbol)
    inject(connections[2], "read_timeout")
    failed = collector.collect(symbol)
    assert failed.snapshot is None and failed.sources.premium_index.data is None
    assert first.complete  # The previous immutable object is not mutated or substituted.
    rejected = engine.estimate_once(failed, settings, scenario)
    assert rejected.net_profit is rejected.net_return is None
    install(transport, symbol)
    recovered = collector.collect(symbol)
    assert recovered.complete and recovered is not first
    result = engine.estimate_once(recovered, settings, scenario)
    assert result.calculation == original.calculation
    assert result.net_return == original.net_return
    assert transport[0].call_count == 3 and transport[1].call_count == 9
