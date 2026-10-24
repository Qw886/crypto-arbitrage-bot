"""Only mocked official public GETs; pagination never invents missing settlements."""

import json
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
from test_forward_validation import AT, OPENED, event

from crypto_arbitrage_bot.exchanges import funding_history as module
from crypto_arbitrage_bot.exchanges.funding_history import BinanceFundingHistoryClient
from crypto_arbitrage_bot.exchanges.funding_history import FundingFailure as F


def row(rate="0.01", at=None, symbol="BTCUSDT"):
    return {"symbol": symbol, "fundingTime": module._epoch_ms(event().settled_at if at is None else at),
            "fundingRate": rate, "markPrice": "110.123456789012345678901234567890"}


def transport(monkeypatch, bodies, statuses=None):
    connections = []
    statuses = iter([200] * len(bodies) if statuses is None else statuses)
    bodies = iter(bodies)
    def create(host, *, timeout):
        assert host == "fapi.binance.com" and timeout == 5
        conn, response = Mock(), Mock()
        response.status = next(statuses)
        value = next(bodies)
        response.read.return_value = value if isinstance(value, bytes) else json.dumps(value).encode()
        conn.getresponse.return_value = response
        connections.append(conn)
        return conn
    monkeypatch.setattr(module, "HTTPSConnection", create)
    return connections


def client(**kwargs):
    return BinanceFundingHistoryClient(timeout_seconds=5, page_limit=kwargs.get("page_limit", 1000),
                                       max_pages=kwargs.get("max_pages", 3))


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
@pytest.mark.parametrize("rate", ["0.01", "-0.01", "0"])
def test_exact_decimal_public_get_boundary(monkeypatch, symbol, rate):
    calls = transport(monkeypatch, [[row(rate, symbol=symbol)]])
    window = client().fetch_window(symbol, OPENED, AT)
    assert window.failure is None and len(window.events) == 1
    assert window.events[0].funding_rate == Decimal(rate)
    assert window.events[0].settlement_mark_price == Decimal(row()["markPrice"])
    assert window.events[0].settled_at == event().settled_at
    method, target = calls[0].request.call_args.args
    assert method == "GET" and urlsplit(target).path == "/fapi/v1/fundingRate"
    params = parse_qs(urlsplit(target).query)
    assert set(params) == {"symbol", "limit", "startTime", "endTime"}
    assert params["startTime"] == [str(module._epoch_ms(OPENED) + 1)]
    assert params["endTime"] == [str(module._epoch_ms(AT))]
    assert set(calls[0].request.call_args.kwargs["headers"]) == {"Accept", "User-Agent"}
    assert window.pages[0].received_at >= window.pages[0].requested_at
    assert window.pages[0].raw_response
    calls[0].close.assert_called_once()


def test_pagination_ascending_exclusive_entry_inclusive_exit(monkeypatch):
    calls = transport(monkeypatch, [[row()], [row("-0.01", at=AT)]])
    window = client(page_limit=1).fetch_window("BTCUSDT", OPENED, AT)
    assert window.failure is None and len(window.events) == 2
    params = parse_qs(urlsplit(calls[1].request.call_args.args[1]).query)
    assert params["startTime"] == [str(module._epoch_ms(event().settled_at) + 1)]


def test_empty_array_complete_history_has_no_manufactured_events(monkeypatch):
    transport(monkeypatch, [[]])
    window = client().fetch_window("BTCUSDT", OPENED, AT)
    assert window.failure is None and window.events == ()


def test_pagination_budget_discards_partial_events(monkeypatch):
    transport(monkeypatch, [[row()]])
    window = client(page_limit=1, max_pages=1).fetch_window("BTCUSDT", OPENED, AT)
    assert window.failure is F.PAGE_LIMIT and not window.events


@pytest.mark.parametrize("status,expected", [(429, F.RATE_LIMIT), (400, F.HTTP_ERROR), (403, F.HTTP_ERROR),
                                           (500, F.HTTP_ERROR), (503, F.HTTP_ERROR), (302, F.HTTP_ERROR)])
def test_http_errors_never_retry_or_follow_redirect(monkeypatch, status, expected):
    calls = transport(monkeypatch, [b"failure"], [status])
    result = client().fetch_window("BTCUSDT", OPENED, AT)
    assert result.failure is expected and not result.events and len(calls) == 1


@pytest.mark.parametrize("phase", ["request", "getresponse", "read"])
@pytest.mark.parametrize("error,expected", [(TimeoutError("timeout"), F.TIMEOUT), (OSError("DNS/connection"), F.NETWORK_ERROR)])
def test_connect_read_network_failures(monkeypatch, phase, error, expected):
    conn = Mock()
    setattr(conn.getresponse.return_value if phase == "read" else conn, phase, Mock(side_effect=error))
    monkeypatch.setattr(module, "HTTPSConnection", Mock(return_value=conn))
    result = client().fetch_window("BTCUSDT", OPENED, AT)
    assert result.failure is expected and result.events == ()
    conn.close.assert_called_once()


@pytest.mark.parametrize("payload", [b"", b"not json", {}, {"code": -1}, [None],
                                    [row(at=OPENED)], [row(at=AT + timedelta(milliseconds=1))],
                                    [row(symbol="ETHUSDT")], [row(), row()],
                                    [dict(row(), markPrice=None)], [dict(row(), markPrice="0")],
                                    [dict(row(), fundingRate=0.1)], [dict(row(), fundingRate="NaN")],
                                    [dict(row(), fundingTime=1.5)]])
def test_malformed_missing_out_of_window_or_duplicate_payload(monkeypatch, payload):
    transport(monkeypatch, [payload])
    result = client().fetch_window("BTCUSDT", OPENED, AT)
    assert result.failure is F.INVALID_RESPONSE and result.events == ()


def test_second_page_error_does_not_expose_first_page_events(monkeypatch):
    transport(monkeypatch, [[row()], b"failure"], [200, 500])
    result = client(page_limit=1).fetch_window("BTCUSDT", OPENED, AT)
    assert result.failure is F.HTTP_ERROR and result.events == ()


def test_unsupported_symbol_rejected_before_network(monkeypatch):
    connect = Mock(side_effect=AssertionError("no IO"))
    monkeypatch.setattr(module, "HTTPSConnection", connect)
    with pytest.raises(ValueError):
        client().fetch_window("BNBUSDT", OPENED, AT)
    connect.assert_not_called()


@pytest.mark.parametrize("horizon", [3600, 28800, 86400])
def test_submillisecond_entry_and_exit_boundaries_for_research_horizons(monkeypatch, horizon):
    opened = OPENED + timedelta(microseconds=500)
    closed = opened + timedelta(seconds=horizon)
    first = OPENED + timedelta(milliseconds=1)
    last = closed - timedelta(microseconds=500)
    calls = transport(monkeypatch, [[row(at=first), row(at=last)]])
    result = client().fetch_window("BTCUSDT", opened, closed)
    assert result.failure is None and [e.settled_at for e in result.events] == [first, last]
    params = parse_qs(urlsplit(calls[0].request.call_args.args[1]).query)
    assert int(params["startTime"][0]) == module._epoch_ms(first)
    assert int(params["endTime"][0]) == module._epoch_ms(last)


def test_page_cursor_preserves_nonzero_settlement_milliseconds(monkeypatch):
    first = event().settled_at + timedelta(milliseconds=1)
    calls = transport(monkeypatch, [[row(at=first)], [row(at=AT)]])
    result = client(page_limit=1).fetch_window("BTCUSDT", OPENED, AT)
    assert result.failure is None
    assert [e.settled_at for e in result.events] == [first, AT]
    params = parse_qs(urlsplit(calls[1].request.call_args.args[1]).query)
    assert int(params["startTime"][0]) == module._epoch_ms(first) + 1


def test_no_representable_settlement_in_submillisecond_window_skips_io(monkeypatch):
    connect = Mock(side_effect=AssertionError("no representable millisecond in window"))
    monkeypatch.setattr(module, "HTTPSConnection", connect)
    result = client().fetch_window("BTCUSDT", OPENED, OPENED+timedelta(microseconds=999))
    assert result.failure is None and result.events == ()
    connect.assert_not_called()
