"""Fixed Demo GET ledger parsing and inclusive API boundaries, no network."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from urllib.parse import parse_qs

import pytest
from test_testnet_execution import Opener, Response, client

from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as Error
from crypto_arbitrage_bot.execution.types import ExecutionMarket

OPENED = datetime(2026, 10, 3, tzinfo=UTC)
CLOSED = OPENED + timedelta(hours=8)


def payment(amount=".1", **changes):
    row = {"symbol": "ETHUSDT", "incomeType": "FUNDING_FEE", "income": amount, "asset": "USDT",
           "time": (CLOSED - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(milliseconds=1), "tranId": 123}
    row.update(changes)
    return row


@pytest.mark.parametrize("amount", [".12345678901234567890123456789", "0", "-.1"])
def test_signed_get_actual_amount_decimal_and_window(amount):
    opener = Opener([Response({"serverTime": 10000}), Response([payment(amount)])])
    result = client(ExecutionMarket.USDTM, opener).get_funding_payments("ETHUSDT", OPENED, CLOSED)
    assert result[0].amount == Decimal(amount) and result[0].posted_at == CLOSED
    request = opener.requests[-1][0]
    assert request.get_method() == "GET" and request.full_url.startswith("https://demo-fapi.binance.com/fapi/v1/income?")
    query = parse_qs(request.full_url.split("?", 1)[1])
    assert query["incomeType"] == ["FUNDING_FEE"] and query["symbol"] == ["ETHUSDT"]
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    assert int(query["startTime"][0]) == (OPENED - epoch) // timedelta(milliseconds=1) + 1
    assert int(query["endTime"][0]) == (CLOSED - epoch) // timedelta(milliseconds=1)
    assert "signature" in query and request.data is None


@pytest.mark.parametrize("changes", [{"symbol": "BTCUSDT"}, {"asset": "BNB"}, {"incomeType": "COMMISSION"},
    {"income": "NaN"}, {"income": 0.1}, {"time": True}, {"tranId": True}, {"tranId": ""},
    {"time": 1}, {"time": 999999999999999999999}, {"income": None}])
def test_invalid_payload_does_not_become_funding(changes):
    opener = Opener([Response({"serverTime": 10000}), Response([payment(**changes)])])
    with pytest.raises(Error) as error:
        client(ExecutionMarket.USDTM, opener).get_funding_payments("ETHUSDT", OPENED, CLOSED)
    assert error.value.kind is ExecutionErrorKind.INVALID_RESPONSE


def test_duplicate_transaction_is_rejected():
    opener = Opener([Response({"serverTime": 10000}), Response([payment(), payment()])])
    with pytest.raises(Error):
        client(ExecutionMarket.USDTM, opener).get_funding_payments("ETHUSDT", OPENED, CLOSED)


def test_empty_ledger_returns_no_payment_not_forecast_income():
    opener = Opener([Response({"serverTime": 10000}), Response([])])
    assert client(ExecutionMarket.USDTM, opener).get_funding_payments("ETHUSDT", OPENED, CLOSED) == ()


def test_wrong_market_refused_without_network():
    opener = Opener([])
    with pytest.raises(ValueError):
        client(ExecutionMarket.SPOT, opener).get_funding_payments("ETHUSDT", OPENED, CLOSED)
    assert opener.requests == []


@pytest.mark.parametrize("close", [OPENED, OPENED - timedelta(seconds=1), OPENED + timedelta(days=8)])
def test_invalid_window_refused_without_network(close):
    opener = Opener([])
    with pytest.raises(ValueError):
        client(ExecutionMarket.USDTM, opener).get_funding_payments("ETHUSDT", OPENED, close)
    assert opener.requests == []


def test_full_page_is_followed_and_duplicates_across_pages_rejected():
    full = [payment(tranId=i) for i in range(1000)]
    opener = Opener([Response({"serverTime": 10000}), Response(full), Response([payment(tranId=2000)])])
    result = client(ExecutionMarket.USDTM, opener).get_funding_payments("ETHUSDT", OPENED, CLOSED)
    assert len(result) == 1001 and "page=2" in opener.requests[-1][0].full_url


def test_api_timeout_is_structured_no_retry():
    opener = Opener([Response({"serverTime": 10000}), TimeoutError("mock")])
    with pytest.raises(Error) as error:
        client(ExecutionMarket.USDTM, opener).get_funding_payments("ETHUSDT", OPENED, CLOSED)
    assert error.value.kind is ExecutionErrorKind.NETWORK_ERROR and len(opener.requests) == 2


def test_pagination_has_finite_cap_without_partial_amount():
    pages = [[payment(tranId=p * 1000 + i) for i in range(1000)] for p in range(3)]
    opener = Opener([Response({"serverTime": 10000}), *(Response(page) for page in pages)])
    with pytest.raises(Error, match="pagination limit"):
        client(ExecutionMarket.USDTM, opener).get_funding_payments("ETHUSDT", OPENED, CLOSED)
    assert len(opener.requests) == 4  # One clock GET, at most three ledger pages.


def test_duplicate_across_pages_does_not_double_count():
    first = [payment(tranId=i) for i in range(1000)]
    opener = Opener([Response({"serverTime": 10000}), Response(first), Response([payment(tranId=0)])])
    with pytest.raises(Error, match="invalid funding ledger payload"):
        client(ExecutionMarket.USDTM, opener).get_funding_payments("ETHUSDT", OPENED, CLOSED)


@pytest.mark.parametrize("at", [OPENED, CLOSED + timedelta(milliseconds=1)])
def test_payment_at_entry_or_after_exit_is_not_included(at):
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    opener = Opener([Response({"serverTime": 10000}),
                     Response([payment(time=(at - epoch) // timedelta(milliseconds=1))])])
    with pytest.raises(Error):
        client(ExecutionMarket.USDTM, opener).get_funding_payments("ETHUSDT", OPENED, CLOSED)
