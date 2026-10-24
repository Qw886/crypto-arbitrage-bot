"""Offline USDT-M adapter tests: no Binance connection is permitted."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from http.client import HTTPException, IncompleteRead
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest

from crypto_arbitrage_bot.arbitrage.execution import simulate_fill
from crypto_arbitrage_bot.exchanges import binance_usdtm
from crypto_arbitrage_bot.exchanges.binance_usdtm import BinanceUsdtmClient
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import prepare_get
from crypto_arbitrage_bot.models.calculation import TradeLeg
from crypto_arbitrage_bot.models.market import Market, OrderBook
from crypto_arbitrage_bot.models.usdtm import OpenInterest, PerpDepth, PremiumIndex, UsdtmErrorKind

FIXTURES = Path(__file__).parent / "fixtures"
METHODS = ("fetch_order_book", "fetch_premium_index", "fetch_open_interest")
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def transport(monkeypatch):
    factory = Mock(side_effect=AssertionError("Real network is forbidden in these tests"))
    monkeypatch.setattr(binance_usdtm, "HTTPSConnection", factory)
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


def fixture_payload(kind, symbol="BTCUSDT"):
    return json.loads((FIXTURES / f"usdtm_{kind}_{symbol[:3].lower()}.json").read_bytes())


def payload_bytes(payload):
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def assert_failed(result, kind):
    assert result.ok is False
    assert result.data is None
    assert result.error is not None
    assert result.error.kind is kind
    assert result.error.message
    assert result.started_at.utcoffset() == result.finished_at.utcoffset() == timedelta(0)
    assert type(result.elapsed_ms) is int
    assert result.elapsed_ms >= 0


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
@pytest.mark.parametrize(
    ("method", "kind", "endpoint", "path", "data_type"),
    [
        ("fetch_order_book", "depth", PublicEndpoint.PERP_DEPTH, "/fapi/v1/depth", PerpDepth),
        ("fetch_premium_index", "premium", PublicEndpoint.PERP_PREMIUM_INDEX,
         "/fapi/v1/premiumIndex", PremiumIndex),
        ("fetch_open_interest", "open_interest", PublicEndpoint.PERP_OPEN_INTEREST,
         "/fapi/v1/openInterest", OpenInterest),
    ],
)
def test_public_get_preserves_raw_response_and_timing(
    transport, symbol, method, kind, endpoint, path, data_type
):
    raw = (FIXTURES / f"usdtm_{kind}_{symbol[:3].lower()}.json").read_bytes()
    payload = json.loads(raw)
    headers = (("Content-Type", "application/json"), ("Content-Length", str(len(raw))))
    connection, response = install_response(transport, raw, headers=headers)

    result = getattr(BinanceUsdtmClient(timeout_seconds=7), method)(symbol)

    assert result.ok is True
    assert result.error is None
    assert isinstance(result.data, data_type)
    assert result.raw_response == raw
    assert result.status_code == 200
    assert result.response_headers == headers
    assert result.body_truncated is False
    assert result.started_at <= result.received_at <= result.finished_at
    assert all(at.utcoffset() == timedelta(0) for at in (
        result.started_at, result.received_at, result.finished_at
    ))
    assert type(result.elapsed_ms) is int and result.elapsed_ms >= 0
    expected_params = {"symbol": symbol}
    if kind == "depth":
        expected_params["limit"] = 100
    assert result.request.endpoint is endpoint
    assert result.request.method == "GET"
    assert dict(result.request.params) == expected_params
    transport.assert_called_once_with("fapi.binance.com", timeout=7)
    connection.request.assert_called_once()
    args, kwargs = connection.request.call_args
    assert args[0] == "GET"
    parsed = urlsplit(args[1])
    assert parsed.path == path
    assert parsed.scheme == parsed.netloc == ""
    assert parse_qs(parsed.query) == {key: [str(value)] for key, value in expected_params.items()}
    assert kwargs.get("body") is None
    assert kwargs["headers"] == {
        "Accept": "application/json", "User-Agent": "crypto-arbitrage-bot/0.1"
    }
    response.read.assert_called_once_with(binance_usdtm.MAX_BODY_BYTES + 1)
    connection.close.assert_called_once()
    if kind == "depth":
        book = result.data.order_book
        assert isinstance(book, OrderBook)
        assert book.symbol == symbol
        assert book.market is Market.USDTM
        assert len(book.bids) == len(book.asks) == 3
        assert book.best_bid == book.bids[0].price == Decimal(payload["bids"][0][0])
        assert book.best_ask == book.asks[0].price == Decimal(payload["asks"][0][0])
        for actual, rows in ((book.bids, payload["bids"]), (book.asks, payload["asks"])):
            assert [(level.price, level.quantity) for level in actual] == [
                (Decimal(price), Decimal(quantity)) for price, quantity in rows
            ]
            assert all(type(v) is Decimal for level in actual for v in (level.price, level.quantity))
        assert book.observed_at == result.received_at
        assert book.event_time == EPOCH + timedelta(milliseconds=payload["E"])
        assert result.data.transaction_time == EPOCH + timedelta(milliseconds=payload["T"])
        assert result.data.last_update_id == payload["lastUpdateId"]
    else:
        assert result.data.symbol == symbol
        assert result.data.observed_at == result.received_at
        assert result.data.event_time == EPOCH + timedelta(milliseconds=payload["time"])
        if kind == "premium":
            for attribute, key in (("mark_price", "markPrice"), ("index_price", "indexPrice"),
                                   ("last_funding_rate", "lastFundingRate")):
                actual = getattr(result.data, attribute)
                assert type(actual) is Decimal
                assert actual == Decimal(payload[key])
            assert result.data.next_funding_time == EPOCH + timedelta(
                milliseconds=payload["nextFundingTime"]
            )
        else:
            assert type(result.data.open_interest) is Decimal
            assert result.data.open_interest == Decimal(payload["openInterest"])


@pytest.mark.parametrize("rate", ["0.0001", "-0.0002", "0"])
def test_funding_rate_sign_is_preserved_without_settlement_or_scaling(transport, rate):
    payload = fixture_payload("premium")
    payload["lastFundingRate"] = rate
    install_response(transport, payload_bytes(payload))
    result = BinanceUsdtmClient().fetch_premium_index("BTCUSDT")
    assert result.ok
    assert result.data.last_funding_rate == Decimal(rate)


def test_zero_open_interest_is_valid(transport):
    payload = fixture_payload("open_interest")
    payload["openInterest"] = "0"
    install_response(transport, payload_bytes(payload))
    result = BinanceUsdtmClient().fetch_open_interest("BTCUSDT")
    assert result.ok
    assert result.data.open_interest == Decimal("0")


@pytest.mark.parametrize(
    ("leg", "expected_vwap"),
    [(TradeLeg.PERP_ENTRY, "2500.05"), (TradeLeg.PERP_EXIT, "2500.25")],
)
def test_depth_can_feed_existing_multilevel_vwap(transport, leg, expected_vwap):
    install_response(transport, payload_bytes(fixture_payload("depth", "ETHUSDT")))
    result = BinanceUsdtmClient().fetch_order_book("ETHUSDT")
    fill = simulate_fill(result.data.order_book, leg, Decimal("2"))
    assert fill.complete
    assert len(fill.fills) == 2
    assert fill.vwap == Decimal(expected_vwap)


def test_construction_is_offline(transport):
    BinanceUsdtmClient()
    transport.assert_not_called()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("symbol", ["BNBUSDT", "BTCUSDC", "btcusdt", "BTCUSDT&limit=500"])
def test_unsupported_symbols_fail_before_io(transport, method, symbol):
    with pytest.raises(ValueError):
        getattr(BinanceUsdtmClient(), method)(symbol)
    transport.assert_not_called()


@pytest.mark.parametrize("limit", [0, 1, 1001, True, "100"])
def test_invalid_depth_limits_fail_before_io(transport, limit):
    with pytest.raises((TypeError, ValueError)):
        BinanceUsdtmClient().fetch_order_book("BTCUSDT", limit=limit)
    transport.assert_not_called()


@pytest.mark.parametrize("option", ["url", "method", "headers", "body", "api_key", "secret"])
def test_no_custom_endpoints_credentials_or_mutations(transport, option):
    with pytest.raises(TypeError):
        BinanceUsdtmClient().fetch_order_book("BTCUSDT", **{option: "unsupported"})
    transport.assert_not_called()


def test_even_internal_fetch_rejects_other_whitelisted_routes(transport):
    request = prepare_get(PublicEndpoint.SPOT_DEPTH, {"symbol": "BTCUSDT"})
    with pytest.raises(ValueError):
        BinanceUsdtmClient()._fetch(request, lambda payload, at: payload)
    transport.assert_not_called()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("failure", "kind"),
    [(TimeoutError("timeout"), UsdtmErrorKind.TIMEOUT),
     (OSError("network"), UsdtmErrorKind.NETWORK_ERROR),
     (HTTPException("malformed HTTP"), UsdtmErrorKind.NETWORK_ERROR)],
)
def test_network_failures_are_typed_and_closed(transport, method, failure, kind):
    connection, _ = install_response(transport, b"")
    connection.getresponse.side_effect = failure
    result = getattr(BinanceUsdtmClient(), method)("BTCUSDT")
    assert_failed(result, kind)
    assert result.received_at is None
    assert result.status_code is None
    assert result.raw_response is None
    connection.close.assert_called_once()


@pytest.mark.parametrize("status", [301, 307, 400, 451, 500])
def test_http_errors_never_retry_or_redirect(transport, status):
    raw = b'{"code":-1000,"msg":"Unavailable"}'
    connection, _ = install_response(
        transport, raw, status=status, headers=(("Location", "https://other.example/order"),)
    )
    result = BinanceUsdtmClient().fetch_open_interest("BTCUSDT")
    assert_failed(result, UsdtmErrorKind.HTTP_ERROR)
    assert result.status_code == status
    assert result.raw_response == raw
    transport.assert_called_once()
    connection.request.assert_called_once()
    connection.close.assert_called_once()


@pytest.mark.parametrize("status", [418, 429])
def test_rate_limits_preserve_retry_metadata_without_retrying(transport, status):
    raw = b'{"code":-1003,"msg":"Too many requests"}'
    connection, _ = install_response(transport, raw, status, (("Retry-After", "60"),))
    result = BinanceUsdtmClient().fetch_premium_index("ETHUSDT")
    assert_failed(result, UsdtmErrorKind.RATE_LIMITED)
    assert result.status_code == status
    assert result.raw_response == raw
    assert result.response_headers == (("Retry-After", "60"),)
    connection.request.assert_called_once()


def test_binance_error_in_http_200_is_a_failure(transport):
    install_response(transport, b'{"code":-1121,"msg":"Invalid symbol"}')
    result = BinanceUsdtmClient().fetch_order_book("BTCUSDT")
    assert_failed(result, UsdtmErrorKind.API_ERROR)
    assert result.error.api_code == -1121


@pytest.mark.parametrize("raw", [b"", b"<html>bad</html>", b"{", b"[]", b"null", b"\xff",
                                  b'{"symbol":"BTCUSDT","symbol":"ETHUSDT"}',
                                  b'{"openInterest":NaN}'])
def test_malformed_json_is_a_typed_failure(transport, raw):
    install_response(transport, raw)
    result = BinanceUsdtmClient().fetch_open_interest("BTCUSDT")
    assert_failed(result, UsdtmErrorKind.INVALID_RESPONSE)
    assert result.raw_response == raw


@pytest.mark.parametrize(
    ("method", "kind", "field"),
    [("fetch_order_book", "depth", field) for field in ("lastUpdateId", "E", "T", "bids", "asks")]
    + [("fetch_premium_index", "premium", field) for field in
       ("symbol", "markPrice", "indexPrice", "lastFundingRate", "nextFundingTime", "time")]
    + [("fetch_open_interest", "open_interest", field) for field in ("symbol", "openInterest", "time")],
)
def test_missing_required_fields(transport, method, kind, field):
    payload = fixture_payload(kind)
    del payload[field]
    install_response(transport, payload_bytes(payload))
    assert_failed(getattr(BinanceUsdtmClient(), method)("BTCUSDT"), UsdtmErrorKind.INVALID_RESPONSE)


@pytest.mark.parametrize(
    ("method", "kind", "field", "value"),
    [("fetch_premium_index", "premium", "markPrice", "0"),
     ("fetch_premium_index", "premium", "markPrice", 65000),
     ("fetch_premium_index", "premium", "indexPrice", "-1"),
     ("fetch_premium_index", "premium", "indexPrice", "Infinity"),
     ("fetch_premium_index", "premium", "lastFundingRate", "NaN"),
     ("fetch_premium_index", "premium", "lastFundingRate", 0.0001),
     ("fetch_open_interest", "open_interest", "openInterest", "-1"),
     ("fetch_open_interest", "open_interest", "openInterest", 12.3),
     ("fetch_open_interest", "open_interest", "openInterest", "NaN"),
     ("fetch_order_book", "depth", "lastUpdateId", True)],
)
def test_invalid_numeric_fields_never_produce_data(transport, method, kind, field, value):
    payload = fixture_payload(kind)
    payload[field] = value
    install_response(transport, payload_bytes(payload))
    assert_failed(getattr(BinanceUsdtmClient(), method)("BTCUSDT"), UsdtmErrorKind.INVALID_RESPONSE)


@pytest.mark.parametrize(
    ("method", "kind", "field", "value"),
    [("fetch_order_book", "depth", "E", -1),
     ("fetch_order_book", "depth", "T", True),
     ("fetch_premium_index", "premium", "time", "1789776000123"),
     ("fetch_premium_index", "premium", "nextFundingTime", 253402300800000),
     ("fetch_open_interest", "open_interest", "time", 1789776000123.5),
     ("fetch_open_interest", "open_interest", "time", None)],
)
def test_invalid_timestamps_are_rejected(transport, method, kind, field, value):
    payload = fixture_payload(kind)
    payload[field] = value
    install_response(transport, payload_bytes(payload))
    assert_failed(getattr(BinanceUsdtmClient(), method)("BTCUSDT"), UsdtmErrorKind.INVALID_RESPONSE)


@pytest.mark.parametrize(
    ("method", "kind"),
    [("fetch_order_book", "depth"), ("fetch_premium_index", "premium"),
     ("fetch_open_interest", "open_interest")],
)
def test_response_must_match_requested_symbol(transport, method, kind):
    payload = fixture_payload(kind)
    payload["symbol"] = "ETHUSDT"
    install_response(transport, payload_bytes(payload))
    assert_failed(getattr(BinanceUsdtmClient(), method)("BTCUSDT"), UsdtmErrorKind.INVALID_RESPONSE)


@pytest.mark.parametrize(
    ("field", "levels"),
    [("bids", []), ("asks", None), ("bids", [["65000"]]),
     ("bids", [[65000, "1"]]), ("bids", [["65000", "0"]]),
     ("asks", [["65001", "-1"]]), ("asks", [["NaN", "1"]]),
     ("bids", [["65000", "1"], ["65000", "2"]]),
     ("bids", [["64999", "1"], ["65000", "1"]]),
     ("asks", [["65002", "1"], ["65001", "1"]]),
     ("bids", [["66000", "1"]])],
)
def test_bad_order_book_is_not_repaired_or_used(transport, field, levels):
    payload = fixture_payload("depth")
    payload[field] = levels
    install_response(transport, payload_bytes(payload))
    assert_failed(BinanceUsdtmClient().fetch_order_book("BTCUSDT"), UsdtmErrorKind.INVALID_RESPONSE)


def test_more_levels_than_requested_are_rejected(transport):
    payload = fixture_payload("depth")
    payload["bids"] = [[str(65000 - i), "1"] for i in range(6)]
    install_response(transport, payload_bytes(payload))
    assert_failed(BinanceUsdtmClient().fetch_order_book("BTCUSDT", 5), UsdtmErrorKind.INVALID_RESPONSE)


def test_incomplete_http_body_preserves_partial_bytes(transport):
    connection, response = install_response(transport, b"")
    response.read.side_effect = IncompleteRead(b'{"symbol":', 100)
    result = BinanceUsdtmClient().fetch_premium_index("BTCUSDT")
    assert_failed(result, UsdtmErrorKind.NETWORK_ERROR)
    assert result.raw_response == b'{"symbol":'
    assert result.received_at is None
    connection.close.assert_called_once()


@pytest.mark.parametrize(
    ("content_length", "kind"),
    [("10000", UsdtmErrorKind.NETWORK_ERROR), ("bad", UsdtmErrorKind.INVALID_RESPONSE)],
)
def test_incorrect_content_length_prevents_success(transport, content_length, kind):
    raw = payload_bytes(fixture_payload("open_interest"))
    install_response(transport, raw, headers=(("Content-Length", content_length),))
    result = BinanceUsdtmClient().fetch_open_interest("BTCUSDT")
    assert_failed(result, kind)
    assert result.raw_response == raw
    assert result.received_at is None


def test_oversized_body_is_bounded(transport):
    install_response(transport, b" " * (binance_usdtm.MAX_BODY_BYTES + 1))
    result = BinanceUsdtmClient().fetch_order_book("BTCUSDT")
    assert_failed(result, UsdtmErrorKind.INVALID_RESPONSE)
    assert result.body_truncated is True
    assert len(result.raw_response) == binance_usdtm.MAX_BODY_BYTES
    assert result.received_at is None


def test_client_recovers_after_a_failed_request(transport):
    first = Mock()
    first.getresponse.side_effect = TimeoutError("temporary outage")
    second, _ = install_response(transport, payload_bytes(fixture_payload("open_interest", "ETHUSDT")))
    transport.side_effect = [first, second]
    client = BinanceUsdtmClient()
    assert_failed(client.fetch_order_book("BTCUSDT"), UsdtmErrorKind.TIMEOUT)
    recovered = client.fetch_open_interest("ETHUSDT")
    assert recovered.ok
    assert recovered.data.symbol == "ETHUSDT"
    first.close.assert_called_once()
    second.close.assert_called_once()
