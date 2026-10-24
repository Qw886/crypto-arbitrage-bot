"""Official account fills are parsed offline; fixed Demo GETs only."""

from datetime import UTC, datetime
from decimal import Decimal
from urllib.parse import parse_qs, urlsplit

import pytest
from test_testnet_execution import Opener, Response, client

from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as Error
from crypto_arbitrage_bot.execution.types import ExecutionMarket


def row(market, **changes):
    data = {"symbol": "ETHUSDT", "id": 10, "orderId": 123, "price": "2000.123456789",
            "qty": "0.01", "quoteQty": "20.00123456789", "commission": "0.02000123456789",
            "commissionAsset": "USDT", "time": 1790985600000}
    if market is ExecutionMarket.SPOT:
        data.update(isBuyer=True)
    else:
        data.update(buyer=False, side="SELL", positionSide="BOTH", realizedPnl="0")
    data.update(changes)
    return data


@pytest.mark.parametrize("market", list(ExecutionMarket))
def test_decimal_timestamp_commission_and_fixed_signed_get(market):
    opener = Opener([Response({"serverTime": 10000}), Response([row(market)])])
    trade, = client(market, opener).get_order_trades("ETHUSDT", "123")
    assert trade.price == Decimal("2000.123456789")
    assert trade.quote_quantity == Decimal("20.00123456789")
    assert trade.commission == Decimal("0.02000123456789")
    assert trade.traded_at == datetime(2026, 10, 3, tzinfo=UTC)
    assert trade.raw_json and "commissionAsset" in trade.raw_json
    for request, _ in opener.requests:
        assert request.get_method() == "GET"
        assert urlsplit(request.full_url).hostname in ("testnet.binance.vision", "demo-fapi.binance.com")
    request = opener.requests[-1][0]
    query = parse_qs(urlsplit(request.full_url).query)
    assert query["orderId"] == ["123"] and query["fromId"] == ["0"]
    assert query["limit"] == ["1000"] and "signature" in query
    assert "test-secret" not in trade.raw_json


@pytest.mark.parametrize("market", list(ExecutionMarket))
@pytest.mark.parametrize("changes", [
    {"symbol": "BTCUSDT"}, {"orderId": 124}, {"orderId": True}, {"id": True}, {"id": -1},
    {"time": True}, {"time": 0}, {"time": 999999999999999999999}, {"price": "NaN"},
    {"price": 0.1}, {"price": "0"}, {"qty": "-1"}, {"quoteQty": None},
    {"commission": "Infinity"}, {"commissionAsset": ""},
])
def test_malformed_payload_never_becomes_evidence(market, changes):
    opener = Opener([Response({"serverTime": 10000}), Response([row(market, **changes)])])
    with pytest.raises(Error) as error:
        client(market, opener).get_order_trades("ETHUSDT", "123")
    assert error.value.kind is ExecutionErrorKind.INVALID_RESPONSE


@pytest.mark.parametrize("changes", [{"buyer": 1}, {"side": "BUY"}, {"positionSide": "SHORT"}, {"realizedPnl": None}])
def test_perp_requires_one_way_and_official_signed_pnl(changes):
    opener = Opener([Response({"serverTime": 10000}), Response([row(ExecutionMarket.USDTM, **changes)])])
    with pytest.raises(Error):
        client(ExecutionMarket.USDTM, opener).get_order_trades("ETHUSDT", "123")


def test_full_page_uses_bounded_from_id_pagination():
    market = ExecutionMarket.USDTM
    first = [row(market, id=index) for index in range(1000)]
    opener = Opener([Response({"serverTime": 10000}), Response(first), Response([row(market, id=1000)])])
    trades = client(market, opener).get_order_trades("ETHUSDT", "123")
    assert len(trades) == 1001
    assert parse_qs(urlsplit(opener.requests[-1][0].full_url).query)["fromId"] == ["1000"]


def test_page_limit_never_returns_partial_evidence():
    market = ExecutionMarket.SPOT
    pages = [[row(market, id=page * 1000 + index) for index in range(1000)] for page in range(3)]
    opener = Opener([Response({"serverTime": 10000}), *(Response(page) for page in pages)])
    with pytest.raises(Error, match="pagination limit"):
        client(market, opener).get_order_trades("ETHUSDT", "123")
    assert len(opener.requests) == 4


def test_duplicate_fill_rejected_no_double_count():
    market = ExecutionMarket.SPOT
    opener = Opener([Response({"serverTime": 10000}), Response([row(market), row(market)])])
    with pytest.raises(Error):
        client(market, opener).get_order_trades("ETHUSDT", "123")


def test_missing_required_field_is_structured():
    data = row(ExecutionMarket.SPOT)
    del data["commission"]
    opener = Opener([Response({"serverTime": 10000}), Response([data])])
    with pytest.raises(Error) as error:
        client(ExecutionMarket.SPOT, opener).get_order_trades("ETHUSDT", "123")
    assert error.value.kind is ExecutionErrorKind.INVALID_RESPONSE


def test_network_failure_is_not_retried():
    opener = Opener([Response({"serverTime": 10000}), TimeoutError()])
    with pytest.raises(Error) as error:
        client(ExecutionMarket.USDTM, opener).get_order_trades("ETHUSDT", "123")
    assert error.value.kind is ExecutionErrorKind.NETWORK_ERROR and len(opener.requests) == 2


@pytest.mark.parametrize("symbol,order_id", [("BNBUSDT", "123"), ("ETHUSDT", "abc"), ("ETHUSDT", "123&symbol=BTCUSDT")])
def test_bad_input_does_not_send_a_request(symbol, order_id):
    opener = Opener([])
    with pytest.raises(ValueError):
        client(ExecutionMarket.SPOT, opener).get_order_trades(symbol, order_id)
    assert opener.requests == []
