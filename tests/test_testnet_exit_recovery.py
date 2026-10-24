"""Offline recovery tests: no blind POST retries, no new Spot exposure."""

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs

import pytest

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.demo import DemoResponse, demo_bot
from crypto_arbitrage_bot.execution.exit_recovery import recover_perp_exit
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState as State
from crypto_arbitrage_bot.execution.types import OrderState
from crypto_arbitrage_bot.models.market import Market

D = Decimal


@pytest.fixture
def failed_exit(tmp_path, monkeypatch):
    bot, transport = demo_bot(load_settings(Path("docs/testnet-bot.toml")), tmp_path)
    plan = bot._plan("ETHUSDT", D(".01"), "recovery-test")
    assert bot.coordinator.open_pair(plan).state is State.HEDGED
    original = transport.open
    def expired(request, **kwargs):
        params = parse_qs((request.data or b"").decode())
        if params.get("newClientOrderId") == [plan.perp_exit.client_order_id]:
            payload = {"symbol": plan.symbol, "orderId": 4, "clientOrderId": plan.perp_exit.client_order_id,
                       "status": "EXPIRED", "origQty": "0.01", "executedQty": "0",
                       "cumQuote": "0", "avgPrice": "0"}
            payload.update(side='BUY', type='LIMIT', timeInForce='IOC',
                           price=params['price'][0], reduceOnly=True, positionSide='BOTH')
            transport.orders[plan.perp_exit.client_order_id] = payload
            return DemoResponse(payload)
        return original(request, **kwargs)
    monkeypatch.setattr(transport, "open", expired)
    assert bot._close(plan).state is State.RECOVERY_REQUIRED
    monkeypatch.setattr(transport, "open", original)
    assert transport.base[plan.symbol] == 0 and transport.positions[plan.symbol] == D("-.01")
    return bot, transport, plan


def rows(bot):
    with bot.journal._connect() as db:
        return db.execute("SELECT * FROM testnet_exit_recoveries ORDER BY ordinal").fetchall()


def test_recovery_is_reduce_only_durable_and_idempotent(failed_exit, monkeypatch):
    bot, transport, plan = failed_exit
    old_receipt = bot.journal.get(plan.attempt_id).perp_exit
    original = bot.perp.place_ioc_limit_order
    seen = []
    def observed(order):
        persisted = rows(bot)
        assert len(persisted) == 1 and persisted[0]["receipt_json"] is None
        assert order.client_order_id != plan.perp_exit.client_order_id
        assert order.reduce_only and order.quantity == D(".01") and order.side.value == "BUY"
        assert order.limit_price == D("2012.1")  # book ask 2010 + explicit .001, tick ceiling
        seen.append(order)
        return original(order)
    monkeypatch.setattr(bot.perp, "place_ioc_limit_order", observed)
    result = recover_perp_exit(bot, plan)
    assert result.state is State.CLOSED and transport.positions[plan.symbol] == 0
    assert transport.base[plan.symbol] == 0 and len(transport.orders) == 5
    assert json.loads(rows(bot)[0]["original_exit_json"])["order_id"] == old_receipt.order_id
    assert recover_perp_exit(bot, plan) == result and len(seen) == 1


def test_query_can_enrich_price_without_blocking_confirmed_zero_fill(failed_exit, monkeypatch):
    bot, _, plan = failed_exit
    query = bot.perp.get_order
    monkeypatch.setattr(bot.perp, "get_order", lambda symbol, cid: replace(query(symbol, cid),
                        average_price=None, cumulative_quote_quantity=None))
    assert recover_perp_exit(bot, plan).state is State.CLOSED


def test_lost_ack_is_queried_after_restart_never_resubmitted(failed_exit, monkeypatch):
    bot, transport, plan = failed_exit
    original = bot.perp.place_ioc_limit_order
    calls = []
    def lost_ack(order):
        calls.append(order)
        original(order)
        raise TimeoutError("ack lost after mock execution")
    monkeypatch.setattr(bot.perp, "place_ioc_limit_order", lost_ack)
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert transport.positions[plan.symbol] == 0 and rows(bot)[0]["receipt_json"] is None
    from crypto_arbitrage_bot.execution.runner import TestnetBot
    restarted = TestnetBot(bot.settings, bot.spot, bot.perp)
    assert recover_perp_exit(restarted, plan).state is State.CLOSED
    assert len(calls) == 1 and len(transport.orders) == 5


def test_unknown_submission_blocks_new_orders(failed_exit, monkeypatch):
    bot, transport, plan = failed_exit
    def failed(order):
        raise TimeoutError("request outcome unknown")
    monkeypatch.setattr(bot.perp, "place_ioc_limit_order", failed)
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    with pytest.raises(KeyError):
        recover_perp_exit(bot, plan)
    assert len(rows(bot)) == 1 and len(transport.orders) == 4


def test_terminal_zero_fills_have_bounded_attempts_across_restart(failed_exit, monkeypatch):
    bot, transport, plan = failed_exit
    def unfilled(order):
        receipt = replace(bot.journal.get(plan.attempt_id).perp_exit,
                          client_order_id=order.client_order_id, order_id=str(5 + len(rows(bot))),
                          limit_price=order.limit_price)
        return receipt
    monkeypatch.setattr(bot.perp, "place_ioc_limit_order", unfilled)
    for _ in range(4):
        assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(rows(bot)) == 2 and transport.positions[plan.symbol] == D("-.01")
    from crypto_arbitrage_bot.execution.runner import TestnetBot
    assert recover_perp_exit(TestnetBot(bot.settings, bot.spot, bot.perp), plan).state is State.RECOVERY_REQUIRED
    assert len(rows(bot)) == 2


@pytest.mark.parametrize("field,value", [("position_quantity", D("-.02")), ("open_order_count", 1),
    ("can_trade", False), ("one_way", False), ("single_asset", False), ("leverage", D("2"))])
def test_account_conflicts_block_recovery(failed_exit, monkeypatch, field, value):
    bot, transport, plan = failed_exit
    state = bot.perp.get_account_state(plan.symbol)
    monkeypatch.setattr(bot.perp, "get_account_state", lambda _: replace(state, **{field: value}))
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert rows(bot) == [] and len(transport.orders) == 4


@pytest.mark.parametrize("field,value", [("status", OrderState.NEW),
                                      ("executed_quantity", D(".001"))])
def test_original_exit_must_be_confirmed_terminal_zero_fill(failed_exit, monkeypatch, field, value):
    bot, transport, plan = failed_exit
    query = bot.perp.get_order
    monkeypatch.setattr(bot.perp, "get_order", lambda symbol, cid: replace(query(symbol, cid), **{field: value}))
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 4


def test_stale_market_blocks_recovery(failed_exit, monkeypatch):
    bot, transport, plan = failed_exit
    snapshot = bot.collector.collect_exit_depth(plan.symbol, markets=(Market.USDTM,))
    monkeypatch.setattr(bot.collector, "collect_exit_depth", lambda *a, **kw: snapshot)
    now = bot.clock()
    bot.clock = lambda: now + timedelta(minutes=1)
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 4 and rows(bot) == []


def test_disabled_recovery_does_not_submit(failed_exit):
    bot, transport, plan = failed_exit
    bot.config = bot.config.model_copy(update={"max_exit_recovery_attempts": 0})
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 4


def test_partial_recovery_only_closes_remaining_quantity(failed_exit, monkeypatch):
    bot, transport, plan = failed_exit
    original = bot.perp.place_ioc_limit_order
    orders = []
    def partial_first(order):
        orders.append(order)
        receipt = original(order)
        if len(orders) == 1:
            transport.positions[plan.symbol] = D("-.006")
            return replace(receipt, status=OrderState.EXPIRED, executed_quantity=D(".004"),
                           cumulative_quote_quantity=D(".004") * order.limit_price)
        return receipt
    monkeypatch.setattr(bot.perp, "place_ioc_limit_order", partial_first)
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert recover_perp_exit(bot, plan).state is State.CLOSED
    assert [order.quantity for order in orders] == [D(".01"), D(".006")]
    assert all(order.reduce_only for order in orders) and len(rows(bot)) == 2
    assert transport.positions[plan.symbol] == 0 and transport.base[plan.symbol] == 0


def test_durable_claim_failure_prevents_submission(failed_exit, monkeypatch):
    from contextlib import contextmanager

    from crypto_arbitrage_bot.execution.pair_coordinator import ExecutionJournalError
    bot, transport, plan = failed_exit
    connect = bot.journal._connect
    calls = 0
    @contextmanager
    def failed_claim(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:  # parent read, recovery rows, then durable claim before POST
            raise ExecutionJournalError("mock write failure")
        with connect(**kwargs) as db:
            yield db
    monkeypatch.setattr(bot.journal, "_connect", failed_claim)
    with pytest.raises(ExecutionJournalError):
        recover_perp_exit(bot, plan)
    assert len(transport.orders) == 4


def test_receipt_write_failure_is_reconciled_without_duplicate_post(failed_exit, monkeypatch):
    from contextlib import contextmanager

    from crypto_arbitrage_bot.execution.pair_coordinator import ExecutionJournalError
    bot, transport, plan = failed_exit
    connect = bot.journal._connect
    calls = 0
    @contextmanager
    def failed_receipt(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:  # receipt write after submission
            raise ExecutionJournalError("mock receipt write failure")
        with connect(**kwargs) as db:
            yield db
    monkeypatch.setattr(bot.journal, "_connect", failed_receipt)
    with pytest.raises(ExecutionJournalError):
        recover_perp_exit(bot, plan)
    assert len(transport.orders) == 5 and transport.positions[plan.symbol] == 0
    monkeypatch.setattr(bot.journal, "_connect", connect)
    assert rows(bot)[0]["receipt_json"] is None
    assert recover_perp_exit(bot, plan).state is State.CLOSED
    assert len(transport.orders) == 5


def test_managed_exits_recovers_without_repeating_old_exit(failed_exit):
    bot, transport, plan = failed_exit
    bot._managed_exits()
    assert bot.journal.get(plan.attempt_id).state is State.CLOSED
    bot._managed_exits()
    assert len(transport.orders) == 5 and transport.positions[plan.symbol] == 0


@pytest.mark.parametrize("field,value", [("max_exit_recovery_attempts", 4),
    ("max_exit_recovery_attempts", -1), ("max_recovery_price_deviation", D(".02")),
    ("max_recovery_price_deviation", D("NaN"))])
def test_recovery_configuration_is_bounded(field, value):
    settings = load_settings(Path("docs/testnet-bot.toml"))
    values = settings.testnet.model_dump()
    values[field] = value
    with pytest.raises(ValueError):
        type(settings.testnet).model_validate(values)
