"""Bounded testnet compression preserves payloads, filters and failure semantics."""

import gzip
import json
from decimal import Decimal
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

import pytest
from test_testnet_execution import Opener, Response, client, exchange_info

from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionClient as Client
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as GatewayError
from crypto_arbitrage_bot.execution.types import ExecutionMarket


class EncodedResponse(Response):
    def __init__(self, payload, *, encoding="gzip", body=None):
        super().__init__(payload)
        self._body = gzip.compress(self._body) if body is None else body
        self.headers = {"Content-Encoding": encoding}


@pytest.mark.parametrize("market", list(ExecutionMarket))
def test_gzip_info_has_identical_rules_to_identity(market):
    payload = exchange_info()
    compressed = Opener([EncodedResponse(payload)])
    plain = Opener([Response(payload)])
    zipped = client(market, compressed).get_symbol_rules("BTCUSDT")
    assert zipped == client(market, plain).get_symbol_rules("BTCUSDT")
    assert compressed.requests[0][0].get_header("Accept-encoding") == "gzip"
    assert isinstance(zipped.step_size, Decimal)


def test_gzip_account_numbers_use_decimal_and_signed_request_is_unchanged():
    payload = {"canTrade": True}
    # Exact decimal JSON comes from the exchange, not Python's float encoder.
    body = gzip.compress(b'{"availableBalance":12.1234567890123456789,"canTrade":true}')
    opener = Opener([Response({"serverTime": 10000}), EncodedResponse(payload, body=body)])
    result = client(ExecutionMarket.USDTM, opener)._request("account", signed=True)
    assert result["availableBalance"] == Decimal("12.1234567890123456789")
    request = opener.requests[-1][0]
    assert request.get_header("Accept-encoding") == "gzip"
    assert request.get_header("X-mbx-apikey") == "test-api-key"
    assert "signature=" in request.full_url and request.get_method() == "GET"


@pytest.mark.parametrize("action", ["depth", "order"])
def test_public_books_and_order_requests_do_not_negotiate_compression(action):
    opener = Opener([Response({})])
    client(ExecutionMarket.USDTM, opener)._request(action)
    assert opener.requests[0][0].get_header("Accept-encoding") is None


@pytest.mark.parametrize("body", [b"not-gzip", gzip.compress(b'{}')[:-3]])
def test_bad_gzip_is_structured_decode_failure_not_network_error(body):
    opener = Opener([EncodedResponse({}, body=body)])
    with pytest.raises(GatewayError) as caught:
        client(ExecutionMarket.USDTM, opener)._request("exchange_info")
    assert caught.value.kind is ExecutionErrorKind.INVALID_RESPONSE
    assert caught.value.phase == "DECODE"
    assert len(opener.requests) == 1


def test_expansion_bomb_cannot_bypass_account_response_limit():
    body = gzip.compress(b"x" * 1_000_001)
    assert len(body) < 2000
    opener = Opener([EncodedResponse({}, body=body)])
    with pytest.raises(GatewayError) as caught:
        client(ExecutionMarket.USDTM, opener)._request("account")
    assert caught.value.kind is ExecutionErrorKind.INVALID_RESPONSE and caught.value.phase == "DECODE"


@pytest.mark.parametrize("encoding", ["br", "gzip, deflate", "invalid"])
def test_unnegotiated_encoding_fails_safely(encoding):
    opener = Opener([EncodedResponse({}, encoding=encoding)])
    with pytest.raises(GatewayError) as caught:
        client(ExecutionMarket.USDTM, opener)._request("exchange_info")
    assert caught.value.kind is ExecutionErrorKind.INVALID_RESPONSE


@pytest.mark.parametrize("status,code,kind,ambiguous", [
    (429, -1003, ExecutionErrorKind.RATE_LIMITED, False),
    (400, -2010, ExecutionErrorKind.HTTP_REJECTED, False),
    (500, -1007, ExecutionErrorKind.SUBMISSION_UNKNOWN, True),
])
def test_compressed_api_errors_keep_codes_and_post_ambiguity(status, code, kind, ambiguous):
    headers = Message()
    headers["Content-Encoding"] = "gzip"
    raw = gzip.compress(json.dumps({"code": code, "msg": "test-api-key test-secret-do-not-print"}).encode())
    error = HTTPError("https://demo-fapi.binance.com/fapi/v1/order?signature=private", status, "failed", headers, BytesIO(raw))
    opener = Opener([Response({"serverTime": 10000}), error])
    with pytest.raises(GatewayError) as caught:
        client(ExecutionMarket.USDTM, opener)._request("order", signed=True)
    assert caught.value.kind is kind and caught.value.api_code == code
    assert caught.value.submission_may_have_reached_exchange is ambiguous
    assert "test-secret" not in str(caught.value) and "test-api-key" not in str(caught.value)
    assert len(opener.requests) == 2  # No blind POST retry.


def test_corrupt_rate_limit_payload_still_preserves_rate_limit():
    headers = Message()
    headers["Content-Encoding"] = "gzip"
    error = HTTPError("https://demo-fapi.binance.com/fapi/v1/exchangeInfo", 429, "failed", headers, BytesIO(b"bad"))
    with pytest.raises(GatewayError) as caught:
        client(ExecutionMarket.USDTM, Opener([error]))._request("exchange_info")
    assert caught.value.kind is ExecutionErrorKind.RATE_LIMITED and caught.value.http_status == 429


@pytest.mark.parametrize("encoding", ["identity", "", " GZIP "])
def test_decoder_exact_size_boundary(encoding):
    wire = gzip.compress(b"a" * 100) if encoding.strip() else b"a" * 100
    if encoding == "identity":
        wire = b"a" * 100
    assert Client._decode_body(wire, encoding, 100) == b"a" * 100
    with pytest.raises(ValueError):
        Client._decode_body(wire, encoding, 99)
