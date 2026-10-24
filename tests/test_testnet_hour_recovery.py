"""BTC/ETH fault injection: simulated expired exit and lost recovery ACK."""

from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs

import pytest

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.demo import DemoResponse, demo_bot
from crypto_arbitrage_bot.execution.exit_recovery import recover_perp_exit
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState as State
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot


@pytest.mark.parametrize("symbol,quantity", [("BTCUSDT", ".001"), ("ETHUSDT", ".01")])
@pytest.mark.parametrize("lost_ack", [False, True])
def test_each_symbol_exit_recovery_durable_without_duplicate_orders(tmp_path, monkeypatch, symbol, quantity, lost_ack):
    bot, transport = demo_bot(load_settings(Path("docs/testnet-funding-acceptance.toml")), tmp_path)
    plan = bot._plan(symbol, Decimal(quantity), "hour-recovery")
    assert bot.coordinator.open_pair(plan).state is State.HEDGED
    original = transport.open
    def expired(request, **kwargs):
        params = parse_qs((request.data or b"").decode())
        if params.get("newClientOrderId") == [plan.perp_exit.client_order_id]:
            receipt = {"symbol": symbol, "orderId": 4, "clientOrderId": plan.perp_exit.client_order_id,
                       "status": "EXPIRED", "origQty": quantity, "executedQty": "0",
                       "cumQuote": "0", "avgPrice": "0"}
            receipt.update(side='BUY', type='LIMIT', timeInForce='IOC',
                           price=params['price'][0], reduceOnly=True, positionSide='BOTH')
            transport.orders[plan.perp_exit.client_order_id] = receipt
            return DemoResponse(receipt)
        return original(request, **kwargs)
    monkeypatch.setattr(transport, "open", expired)
    assert bot._close(plan).state is State.RECOVERY_REQUIRED
    monkeypatch.setattr(transport, "open", original)
    place = bot.perp.place_ioc_limit_order
    submitted = []
    def recovery(order):
        assert order.reduce_only and order.quantity == Decimal(quantity)
        submitted.append(order.client_order_id)
        result = place(order)
        if lost_ack:
            raise TimeoutError("simulated acknowledgement loss after execution")
        return result
    monkeypatch.setattr(bot.perp, "place_ioc_limit_order", recovery)
    result = recover_perp_exit(bot, plan)
    if lost_ack:
        assert result.state is State.RECOVERY_REQUIRED
        restarted = Bot(bot.settings, bot.spot, bot.perp)
        result = recover_perp_exit(restarted, plan)
    assert result.state is State.CLOSED
    assert transport.base[symbol] == transport.positions[symbol] == 0
    assert len(transport.orders) == 5 and len(submitted) == 1
    recover_perp_exit(bot, plan)
    assert len(submitted) == 1
