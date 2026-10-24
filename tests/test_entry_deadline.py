"""Submission deadlines survive restart and guard AFTER slow preparation."""
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from test_testnet_execution import Opener, Response, client, exchange_info, order_reply

from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as Error
from crypto_arbitrage_bot.execution.types import ExecutionMarket, IocLimitOrder
from crypto_arbitrage_bot.models.market import Side


@pytest.mark.parametrize("market", [ExecutionMarket.SPOT, ExecutionMarket.USDTM])
def test_slow_time_sync_must_not_submit_expired_entry(market):
    clock = {"ms": 10000}
    class SlowOpener(Opener):
        def open(self, request, timeout):
            result = super().open(request, timeout)
            if "/time" in request.full_url:
                clock["ms"] = 30000
            return result
    opener = SlowOpener([Response(exchange_info()), Response({"serverTime": 10000})])
    gateway = client(market, opener, clock_ms=lambda: clock["ms"])
    # Future at planning; expires while time-sync is in flight.
    order = IocLimitOrder(market, "BTCUSDT", Side.BUY if market is ExecutionMarket.SPOT else Side.SELL,
                          Decimal(".001"), Decimal("50000"), "deadline-entry",
                          submit_before=datetime.fromtimestamp(20, UTC))
    with pytest.raises(Error, match="ENTRY_SUBMISSION_DEADLINE_PASSED") as error:
        gateway.place_ioc_limit_order(order)
    assert not error.value.submission_may_have_reached_exchange
    assert all(request.get_method() == "GET" for request, _ in opener.requests)


def test_fresh_deadline_is_local_only_not_sent_as_exchange_parameter():
    opener = Opener([Response(exchange_info()), Response({"serverTime": 10000}),
                     Response(dict(order_reply(cid="deadline-entry"), side='SELL', price='50000',
                                   reduceOnly=False, positionSide='BOTH'))])
    order = IocLimitOrder(ExecutionMarket.USDTM, "BTCUSDT", Side.SELL, Decimal(".001"),
                          Decimal("50000"), "deadline-entry", submit_before=datetime.fromtimestamp(20, UTC))
    receipt = client(ExecutionMarket.USDTM, opener).place_ioc_limit_order(order)
    assert receipt.executed_quantity == Decimal(".001")
    assert b"submit_before" not in opener.requests[-1][0].data


def test_deadline_cannot_block_reduce_only_exit():
    with pytest.raises(ValueError, match="reduce-only"):
        IocLimitOrder(ExecutionMarket.USDTM, "BTCUSDT", Side.BUY, Decimal(".001"),
                      Decimal("50000"), "exit", True, submit_before=datetime.now(UTC))
