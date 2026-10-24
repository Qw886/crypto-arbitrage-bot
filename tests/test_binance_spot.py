"""Offline Spot adapter regression tests; every transport is replaced by a mock."""

import json
from datetime import timedelta
from decimal import Decimal
from http.client import HTTPException, IncompleteRead
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest

from crypto_arbitrage_bot.exchanges import binance_spot
from crypto_arbitrage_bot.exchanges.binance_spot import BinanceSpotClient
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.models.market import Asset, Market, OrderBook
from crypto_arbitrage_bot.models.spot import SpotErrorKind

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def transport(monkeypatch):
    """Deny actual network access even if a test forgets to configure its response."""
    factory = Mock(side_effect=AssertionError("Real network is forbidden in these tests"))
    monkeypatch.setattr(binance_spot, "HTTPSConnection", factory)
    return factory


def install_response(transport, body, status=200, headers=()):
    response = Mock()
    response.status = status
    response.getheaders.return_value = list(headers)
    response.read.return_value = body
    connection = Mock()
    connection.getresponse.return_value = response
    transport.side_effect = None
    transport.return_value = connection
    return connection, response


def payload_bytes(payload):
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def valid_payload():
    return {"lastUpdateId": 123, "bids": [["100.01", "2"]], "asks": [["100.02", "3"]]}


def assert_failed(result, kind):
    assert result.ok is False
    assert result.order_book is None
    assert result.last_update_id is None
    assert result.error is not None
    assert result.error.kind is kind
    assert result.error.message
    assert result.started_at.utcoffset() == timedelta(0)
    assert result.finished_at.utcoffset() == timedelta(0)
    assert type(result.elapsed_ms) is int
    assert result.elapsed_ms >= 0


@pytest.mark.parametrize(
    ("symbol", "filename", "base", "bid", "ask", "count"),
    [
        (
            "BTCUSDT", "spot_depth_btc.json", Asset.BTC,
            "65000.123456789123456789", "65001.123456789123456789", 3,
        ),
        ("ETHUSDT", "spot_depth_eth.json", Asset.ETH, "2500.01", "2500.02", 2),
    ],
)
def test_supported_spot_books_preserve_same_snapshot_and_exact_decimal(
    transport, symbol, filename, base, bid, ask, count
):
    raw = (FIXTURES / filename).read_bytes()
    headers = (("Content-Type", "application/json"), ("X-MBX-USED-WEIGHT-1M", "5"))
    connection, response = install_response(transport, raw, headers=headers)

    result = BinanceSpotClient(timeout_seconds=7).fetch_order_book(symbol, limit=100)

    assert result.ok is True
    assert result.error is None
    assert result.status_code == 200
    assert result.response_headers == headers
    assert result.raw_response == raw
    assert result.body_truncated is False
    assert result.last_update_id == json.loads(raw)["lastUpdateId"]
    assert type(result.last_update_id) is int
    book = result.order_book
    assert isinstance(book, OrderBook)
    assert book.market is Market.SPOT
    assert book.symbol == symbol
    assert book.base_asset is base
    assert book.quote_asset is Asset.USDT
    assert len(book.bids) == len(book.asks) == count
    assert book.best_bid == book.bids[0].price == Decimal(bid)
    assert book.best_ask == book.asks[0].price == Decimal(ask)
    for level in (*book.bids, *book.asks):
        assert type(level.price) is Decimal
        assert type(level.quantity) is Decimal
    if symbol == "BTCUSDT":
        assert book.bids[0].quantity == Decimal("0.100000000000000001")
        assert book.asks[0].quantity == Decimal("0.200000000000000002")
    assert book.event_time is None  # REST depth does not supply exchange event time.
    assert book.observed_at == result.received_at
    assert result.started_at <= result.received_at <= result.finished_at
    for timestamp in (result.started_at, result.received_at, result.finished_at):
        assert timestamp.utcoffset() == timedelta(0)
    assert type(result.elapsed_ms) is int
    assert result.elapsed_ms >= 0
    assert result.request.endpoint is PublicEndpoint.SPOT_DEPTH
    assert result.request.method == "GET"
    assert dict(result.request.params) == {"symbol": symbol, "limit": 100}
    transport.assert_called_once()
    constructor = transport.call_args
    assert constructor.args[0] == "data-api.binance.vision"
    assert constructor.kwargs["timeout"] == 7
    connection.request.assert_called_once()
    args, kwargs = connection.request.call_args
    method = args[0] if args else kwargs["method"]
    path = args[1] if len(args) > 1 else kwargs["url"]
    assert method == "GET"
    parsed = urlsplit(path)
    assert parsed.path == "/api/v3/depth"
    assert parsed.scheme == parsed.netloc == ""
    assert parse_qs(parsed.query) == {"symbol": [symbol], "limit": ["100"]}
    body = args[2] if len(args) > 2 else kwargs.get("body")
    assert body is None
    sent_headers = args[3] if len(args) > 3 else kwargs.get("headers", {})
    assert {key.lower(): value for key, value in sent_headers.items()} == {
        "accept": "application/json",
        "user-agent": "crypto-arbitrage-bot/0.1",
    }
    connection.getresponse.assert_called_once()
    response.read.assert_called_once_with(binance_spot.MAX_BODY_BYTES + 1)
    connection.close.assert_called_once()


def test_construction_does_not_connect(transport):
    BinanceSpotClient()
    transport.assert_not_called()


@pytest.mark.parametrize("symbol", ["BNBUSDT", "BTCUSDC", "btcusdt", "BTCUSDT&limit=5000"])
def test_unsupported_symbol_rejected_before_io(transport, symbol):
    with pytest.raises(ValueError):
        BinanceSpotClient().fetch_order_book(symbol)
    transport.assert_not_called()


@pytest.mark.parametrize("limit", [0, 5001, True, "100"])
def test_invalid_depth_limit_rejected_before_io(transport, limit):
    with pytest.raises((ValueError, TypeError)):
        BinanceSpotClient().fetch_order_book("BTCUSDT", limit=limit)
    transport.assert_not_called()


@pytest.mark.parametrize("option", ["url", "method", "headers", "body", "api_key", "secret"])
def test_adapter_has_no_custom_endpoint_or_mutation_options(transport, option):
    with pytest.raises(TypeError):
        BinanceSpotClient().fetch_order_book("BTCUSDT", **{option: "unsupported"})
    transport.assert_not_called()


@pytest.mark.parametrize(
    ("exception", "kind"),
    [
        (TimeoutError("timed out"), SpotErrorKind.TIMEOUT),
        (OSError("connection failed"), SpotErrorKind.NETWORK_ERROR),
        (HTTPException("malformed HTTP"), SpotErrorKind.NETWORK_ERROR),
    ],
)
def test_transport_errors_return_typed_failure_and_close(transport, exception, kind):
    connection, _ = install_response(transport, b"")
    connection.getresponse.side_effect = exception
    result = BinanceSpotClient().fetch_order_book("BTCUSDT")
    assert_failed(result, kind)
    assert result.status_code is None
    assert result.raw_response is None
    assert result.received_at is None
    connection.close.assert_called_once()


def test_incomplete_body_never_returns_an_order_book(transport):
    connection, response = install_response(transport, b"")
    response.read.side_effect = IncompleteRead(b'{"lastUpdateId":123,', 100)
    result = BinanceSpotClient().fetch_order_book("BTCUSDT")
    assert_failed(result, SpotErrorKind.NETWORK_ERROR)
    connection.close.assert_called_once()


@pytest.mark.parametrize("status", [301, 302, 307, 400, 403, 451, 500, 503])
def test_http_error_and_redirect_never_follow_another_url(transport, status):
    raw = payload_bytes(valid_payload())
    connection, _ = install_response(
        transport, raw, status=status, headers=(("Location", "https://evil.example/order"),)
    )
    result = BinanceSpotClient().fetch_order_book("BTCUSDT")
    assert_failed(result, SpotErrorKind.HTTP_ERROR)
    assert result.status_code == status
    assert result.raw_response == raw
    assert result.response_headers == (("Location", "https://evil.example/order"),)
    transport.assert_called_once()
    connection.request.assert_called_once()
    connection.close.assert_called_once()


@pytest.mark.parametrize("status", [418, 429])
def test_rate_limit_is_identified_and_preserves_retry_metadata(transport, status):
    raw = b'{"code":-1003,"msg":"Too many requests"}'
    connection, _ = install_response(transport, raw, status=status, headers=(("Retry-After", "60"),))
    result = BinanceSpotClient().fetch_order_book("ETHUSDT")
    assert_failed(result, SpotErrorKind.RATE_LIMITED)
    assert result.status_code == status
    assert result.raw_response == raw
    assert result.response_headers == (("Retry-After", "60"),)
    connection.request.assert_called_once()
    connection.close.assert_called_once()


def test_api_error_in_success_http_status_is_not_an_order_book(transport):
    raw = b'{"code":-1121,"msg":"Invalid symbol."}'
    connection, _ = install_response(transport, raw)
    result = BinanceSpotClient().fetch_order_book("BTCUSDT")
    assert_failed(result, SpotErrorKind.API_ERROR)
    assert result.error.api_code == -1121
    assert result.status_code == 200
    assert result.raw_response == raw
    connection.close.assert_called_once()


@pytest.mark.parametrize("raw", [b"", b"<html>unavailable</html>", b"{", b"[]", b"null", b"\xff"])
def test_invalid_response_bytes_return_failure_without_escaping(transport, raw):
    connection, _ = install_response(transport, raw)
    result = BinanceSpotClient().fetch_order_book("BTCUSDT")
    assert_failed(result, SpotErrorKind.INVALID_RESPONSE)
    assert result.raw_response == raw
    assert result.status_code == 200
    connection.close.assert_called_once()


@pytest.mark.parametrize("field", ["lastUpdateId", "bids", "asks"])
def test_missing_required_depth_fields(transport, field):
    payload = valid_payload()
    del payload[field]
    raw = payload_bytes(payload)
    install_response(transport, raw)
    result = BinanceSpotClient().fetch_order_book("BTCUSDT")
    assert_failed(result, SpotErrorKind.INVALID_RESPONSE)
    assert result.raw_response == raw


@pytest.mark.parametrize("update_id", [True, -1, 1.25, "123", None])
def test_update_id_requires_nonnegative_integer(transport, update_id):
    payload = valid_payload()
    payload["lastUpdateId"] = update_id
    install_response(transport, payload_bytes(payload))
    assert_failed(BinanceSpotClient().fetch_order_book("BTCUSDT"), SpotErrorKind.INVALID_RESPONSE)


@pytest.mark.parametrize(
    "level",
    [["100.01"], ["100.01", "2", "unexpected"], [100.01, "2"], ["100.01", 2],
     [True, "2"], ["NaN", "2"], ["Infinity", "2"], ["100.01", "NaN"],
     ["0", "2"], ["100.01", "-1"], ["100.01", "0"], ["bad", "2"]],
)
def test_invalid_price_quantity_and_level_shape_are_rejected(transport, level):
    payload = valid_payload()
    payload["bids"] = [level]
    install_response(transport, payload_bytes(payload))
    assert_failed(BinanceSpotClient().fetch_order_book("BTCUSDT"), SpotErrorKind.INVALID_RESPONSE)


@pytest.mark.parametrize(
    ("field", "levels"),
    [("bids", []), ("asks", []), ("bids", None), ("asks", {}),
     ("bids", [["99", "1"], ["100", "1"]]),
     ("asks", [["102", "1"], ["101", "1"]]),
     ("bids", [["100.01", "1"], ["100.01", "1"]]),
     ("asks", [["100.02", "1"], ["100.02", "1"]]),
     ("bids", [["101", "1"]])],
)
def test_empty_unsorted_duplicate_or_crossed_book_is_not_normalized(transport, field, levels):
    payload = valid_payload()
    payload[field] = levels
    install_response(transport, payload_bytes(payload))
    assert_failed(BinanceSpotClient().fetch_order_book("BTCUSDT"), SpotErrorKind.INVALID_RESPONSE)


def test_optional_mismatched_symbol_is_rejected(transport):
    payload = valid_payload()
    payload["symbol"] = "ETHUSDT"
    install_response(transport, payload_bytes(payload))
    assert_failed(BinanceSpotClient().fetch_order_book("BTCUSDT"), SpotErrorKind.INVALID_RESPONSE)


def test_oversized_body_is_bounded_and_cannot_be_a_success(transport):
    raw = b" " * (binance_spot.MAX_BODY_BYTES + 1)
    connection, response = install_response(transport, raw)
    result = BinanceSpotClient().fetch_order_book("BTCUSDT")
    assert_failed(result, SpotErrorKind.INVALID_RESPONSE)
    assert result.body_truncated is True
    assert result.raw_response is not None
    assert len(result.raw_response) <= binance_spot.MAX_BODY_BYTES
    response.read.assert_called_once_with(binance_spot.MAX_BODY_BYTES + 1)
    connection.close.assert_called_once()


def test_failed_btc_request_does_not_prevent_next_eth_request(transport):
    first = Mock()
    first.getresponse.side_effect = TimeoutError("BTC timeout")
    second, _ = install_response(transport, (FIXTURES / "spot_depth_eth.json").read_bytes())
    transport.side_effect = [first, second]
    client = BinanceSpotClient()
    assert_failed(client.fetch_order_book("BTCUSDT"), SpotErrorKind.TIMEOUT)
    recovered = client.fetch_order_book("ETHUSDT")
    assert recovered.ok is True
    assert recovered.order_book.symbol == "ETHUSDT"
    assert transport.call_count == 2
    first.close.assert_called_once()
    second.close.assert_called_once()
