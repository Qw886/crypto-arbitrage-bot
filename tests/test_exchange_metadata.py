"""Mock-only GET boundary and structured transport failures for metadata reads."""

import json
from datetime import UTC, datetime, timedelta
from http.client import HTTPException

import pytest

from crypto_arbitrage_bot.exchanges import exchange_metadata as module
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.exchange_metadata import (
    BinanceMetadataClient,
    MetadataErrorKind,
)


class Response:
    def __init__(self, body=b'{"symbols":[]}', status=200):
        self.body, self.status = body, status
        self.read_limits = []

    def read(self, limit):
        self.read_limits.append(limit)
        if isinstance(self.body, Exception):
            raise self.body
        return self.body[:limit]


class Connection:
    def __init__(self, response):
        self.response, self.closed, self.requests = response, False, []

    def request(self, *args, **kwargs):
        self.requests.append((args, kwargs))

    def getresponse(self):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def close(self):
        self.closed = True


@pytest.fixture
def transport(monkeypatch):
    calls = []
    connection = Connection(Response())
    def open_connection(*args, **kwargs):
        calls.append((args, kwargs))
        return connection
    monkeypatch.setattr(module, "open_public_connection", open_connection)
    return connection, calls


@pytest.mark.parametrize("endpoint", list(module.METADATA_ENDPOINTS))
def test_metadata_reads_use_existing_proxy_boundary_and_only_get(transport, endpoint):
    connection, calls = transport
    result = BinanceMetadataClient(timeout_seconds=3).fetch(endpoint)
    assert result.error is None and result.raw_response == b'{"symbols":[]}'
    assert result.official_time is None and result.received_at is not None
    assert result.requested_at <= result.received_at <= result.finished_at
    assert calls[0][0] == (endpoint.value,) and calls[0][1]["timeout"] == 3
    args, kwargs = connection.requests[0]
    assert args[0] == "GET" and "?" not in args[1]
    assert set(kwargs["headers"]) == {"Accept", "User-Agent"}
    assert connection.closed and connection.response.read_limits == [module.MAX_METADATA_BYTES + 1]


def test_official_milliseconds_preserved_without_float(transport):
    connection, _ = transport
    ms = 1791158400123
    connection.response.body = json.dumps({"symbols": [], "serverTime": ms}).encode()
    result = BinanceMetadataClient(timeout_seconds=3).fetch(PublicEndpoint.PERP_EXCHANGE_INFO)
    assert result.official_time == datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=ms)


@pytest.mark.parametrize("status,kind", [(400, MetadataErrorKind.HTTP_ERROR), (403, MetadataErrorKind.HTTP_ERROR),
    (429, MetadataErrorKind.RATE_LIMIT), (500, MetadataErrorKind.HTTP_ERROR), (503, MetadataErrorKind.HTTP_ERROR),
    (301, MetadataErrorKind.HTTP_ERROR)])
def test_http_failures_and_redirects_are_not_success_or_retried(transport, status, kind):
    connection, calls = transport
    connection.response.status = status
    result = BinanceMetadataClient(timeout_seconds=3).fetch(PublicEndpoint.PERP_EXCHANGE_INFO)
    assert result.error.kind is kind and result.status_code == status
    assert connection.closed and len(calls) == 1 and len(connection.requests) == 1


@pytest.mark.parametrize("phase", ["connect", "headers", "read"])
@pytest.mark.parametrize("error,kind", [(TimeoutError(), MetadataErrorKind.TIMEOUT),
    (ConnectionRefusedError(), MetadataErrorKind.NETWORK_ERROR), (HTTPException(), MetadataErrorKind.NETWORK_ERROR)])
def test_structured_network_failures(transport, monkeypatch, phase, error, kind):
    connection, _ = transport
    if phase == "connect":
        def failed(*a, **kw):
            raise error
        monkeypatch.setattr(module, "open_public_connection", failed)
    elif phase == "headers":
        connection.response = error
    else:
        connection.response.body = error
    result = BinanceMetadataClient(timeout_seconds=3).fetch(PublicEndpoint.SPOT_EXCHANGE_INFO)
    assert result.error.kind is kind
    assert result.raw_response is None and result.received_at is None
    if phase != "connect":
        assert connection.closed


@pytest.mark.parametrize("body", [b"", b"{", b"[]", b'{}', b'{"symbols":null}',
    b'{"symbols":[],"symbols":[]}', b'{"symbols":[],"bad":NaN}', b'{"symbols":[],"serverTime":true}'])
def test_malformed_metadata_never_returns_success(transport, body):
    connection, _ = transport
    connection.response.body = body
    result = BinanceMetadataClient(timeout_seconds=3).fetch(PublicEndpoint.SPOT_EXCHANGE_INFO)
    assert result.error.kind is MetadataErrorKind.INVALID_RESPONSE and result.raw_response == body


def test_api_error_is_structured_and_does_not_echo_payload(transport):
    connection, _ = transport
    connection.response.body = b'{"code":-1003,"msg":"large or sensitive context"}'
    result = BinanceMetadataClient(timeout_seconds=3).fetch(PublicEndpoint.PERP_EXCHANGE_INFO)
    assert result.error.kind is MetadataErrorKind.API_ERROR and result.error.api_code == -1003
    assert "sensitive" not in str(result.error)


def test_size_limit_rejects_incomplete_body(transport, monkeypatch):
    monkeypatch.setattr(module, "MAX_METADATA_BYTES", 16)
    connection, _ = transport
    connection.response.body = b"x" * 17
    result = BinanceMetadataClient(timeout_seconds=3).fetch(PublicEndpoint.SPOT_EXCHANGE_INFO)
    assert result.body_truncated and len(result.raw_response) == 16
    assert result.error.kind is MetadataErrorKind.INVALID_RESPONSE


@pytest.mark.parametrize("endpoint", [PublicEndpoint.PERP_DEPTH, PublicEndpoint.SPOT_TIME,
    "https://api.binance.com/api/v3/account", "https://fapi.binance.com/fapi/v1/order",
    PublicEndpoint.SPOT_EXCHANGE_INFO.value])
def test_metadata_cannot_send_arbitrary_or_private_requests(transport, endpoint):
    connection, calls = transport
    with pytest.raises(ValueError):
        BinanceMetadataClient(timeout_seconds=3).fetch(endpoint)
    assert not calls and not connection.requests


@pytest.mark.parametrize("timeout", [0, 61, True, .5])
def test_metadata_timeout_is_finite_and_configurable(timeout):
    with pytest.raises(ValueError):
        BinanceMetadataClient(timeout_seconds=timeout)
