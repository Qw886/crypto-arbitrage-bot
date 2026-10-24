"""Confirmed terminal partial exits are recovered only for proven residuals."""

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs

import pytest
from test_trade_accounting import AT

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.account_balance_audit import (
    AssetBalance,
    BalanceSnapshot,
    audit_balance_changes,
)
from crypto_arbitrage_bot.execution.demo import DemoResponse, demo_bot
from crypto_arbitrage_bot.execution.exit_recovery import recover_perp_exit
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState as State
from crypto_arbitrage_bot.execution.pair_coordinator import _receipt_json
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot
from crypto_arbitrage_bot.execution.testnet_client import (
    ExecutionErrorKind,
)
from crypto_arbitrage_bot.execution.testnet_client import (
    TestnetExecutionError as GatewayError,
)
from crypto_arbitrage_bot.execution.trade_reconciliation import reconcile_closed_attempt
from crypto_arbitrage_bot.execution.types import (
    AccountTrade,
    ExecutionMarket,
    FundingPayment,
    OrderReceipt,
    OrderState,
)
from crypto_arbitrage_bot.models.calculation import TradeLeg

D = Decimal


@pytest.fixture
def partial(tmp_path, monkeypatch, request):
    symbol, quantity, filled = getattr(request, 'param', ('ETHUSDT', '.01', '.004'))
    q, done = D(quantity), D(filled)
    bot, transport = demo_bot(load_settings(Path('docs/testnet-bot.toml')), tmp_path)
    plan = bot._plan(symbol, q, 'partial-' + symbol)
    assert bot.coordinator.open_pair(plan).state is State.HEDGED
    original = transport.open
    def terminal_partial(request, **kw):
        params = parse_qs((request.data or b'').decode())
        if params.get('newClientOrderId') == [plan.perp_exit.client_order_id]:
            payload = {'symbol': symbol, 'orderId': 4, 'clientOrderId': plan.perp_exit.client_order_id,
                       'status': 'EXPIRED', 'origQty': str(q), 'executedQty': str(done),
                       'cumQuote': str(done * D('2010')), 'avgPrice': '2010'}
            payload.update(side='BUY', type='LIMIT', timeInForce='IOC',
                           price=params['price'][0], reduceOnly=True, positionSide='BOTH')
            if done == 0:
                payload['avgPrice'] = '0'
            transport.orders[plan.perp_exit.client_order_id] = payload
            transport.positions[symbol] += done
            return DemoResponse(payload)
        return original(request, **kw)
    monkeypatch.setattr(transport, 'open', terminal_partial)
    assert bot._close(plan).state is State.RECOVERY_REQUIRED
    monkeypatch.setattr(transport, 'open', original)
    assert transport.base[symbol] == 0 and transport.positions[symbol] == -(q - done)
    return bot, transport, plan, done


def rows(bot):
    with bot.journal._connect() as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='testnet_exit_recoveries'").fetchone():
            return []
        return db.execute('SELECT * FROM testnet_exit_recoveries ORDER BY ordinal').fetchall()


@pytest.mark.parametrize('partial', [('ETHUSDT', '.01', '.004'), ('BTCUSDT', '.002', '.001')], indirect=True)
def test_known_partial_original_closes_only_remaining_and_preserves_evidence(partial, monkeypatch):
    bot, transport, plan, original_fill = partial
    submit, seen = bot.perp.place_ioc_limit_order, []
    original_payload = dict(transport.orders[plan.perp_exit.client_order_id])
    def observe(order):
        seen.append(order)
        assert order.quantity == plan.perp_entry.quantity - original_fill
        assert order.reduce_only and order.side.value == 'BUY'
        assert len(rows(bot)) == 1 and rows(bot)[0]['receipt_json'] is None
        return submit(order)
    monkeypatch.setattr(bot.perp, 'place_ioc_limit_order', observe)
    assert recover_perp_exit(bot, plan).state is State.CLOSED
    assert transport.positions[plan.symbol] == 0 and transport.base[plan.symbol] == 0
    assert len(seen) == 1 and len(transport.orders) == 5
    assert transport.orders[plan.perp_exit.client_order_id] == original_payload
    assert D(json.loads(rows(bot)[0]['original_exit_json'])['executed_quantity']) == original_fill
    recover_perp_exit(bot, plan)
    assert len(seen) == 1


def test_unknown_recovery_receipt_is_queried_after_restart_without_resubmission(partial, monkeypatch):
    bot, transport, plan, _ = partial
    original = bot.perp.place_ioc_limit_order
    seen = []
    def lost_ack(order):
        seen.append(order)
        original(order)
        raise TimeoutError('ack lost after exchange execution')
    monkeypatch.setattr(bot.perp, 'place_ioc_limit_order', lost_ack)
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert transport.positions[plan.symbol] == 0 and rows(bot)[0]['receipt_json'] is None
    restarted = Bot(bot.settings, bot.spot, bot.perp)
    assert recover_perp_exit(restarted, plan).state is State.CLOSED
    assert len(seen) == 1 and len(transport.orders) == 5


@pytest.mark.parametrize('status', [OrderState.NEW, OrderState.PARTIALLY_FILLED])
def test_still_active_original_never_creates_a_new_order(partial, monkeypatch, status):
    bot, transport, plan, _ = partial
    receipt = replace(bot.journal.get(plan.attempt_id).perp_exit, status=status)
    bot.journal.update(plan.attempt_id, state=State.RECOVERY_REQUIRED, perp_exit=receipt,
                       failure_code='PERP_EXIT_INCOMPLETE')
    monkeypatch.setattr(bot.perp, 'get_order', lambda *a: receipt)
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 4 and not rows(bot)


def test_changed_original_quantity_is_not_silently_assumed_stable(partial, monkeypatch):
    bot, transport, plan, _ = partial
    receipt = bot.journal.get(plan.attempt_id).perp_exit
    monkeypatch.setattr(bot.perp, 'get_order', lambda *a: replace(receipt, executed_quantity=D('.005')))
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 4 and not rows(bot)


def test_account_residual_conflict_blocks_recovery(partial):
    bot, transport, plan, _ = partial
    transport.positions[plan.symbol] = D('-.007')
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 4 and not rows(bot)


@pytest.mark.parametrize('partial', [('BTCUSDT', '.001', '.0004')], indirect=True)
def test_dust_residual_does_not_bypass_exchange_filters(partial):
    bot, transport, plan, _ = partial
    with pytest.raises(GatewayError) as caught:
        recover_perp_exit(bot, plan)
    assert caught.value.kind is ExecutionErrorKind.ORDER_GUARD_REJECTED
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED
    assert transport.positions[plan.symbol] == D('-.0006')
    assert len(transport.orders) == 4 and not rows(bot)


@pytest.mark.parametrize('status', [OrderState.CANCELED, OrderState.EXPIRED_IN_MATCH])
def test_terminal_status_partial_fills_are_respected(partial, status):
    bot, transport, plan, _ = partial
    receipt = replace(bot.journal.get(plan.attempt_id).perp_exit, status=status)
    bot.journal.update(plan.attempt_id, state=State.RECOVERY_REQUIRED, perp_exit=receipt,
                       failure_code='PERP_EXIT_INCOMPLETE')
    transport.orders[plan.perp_exit.client_order_id]['status'] = status.value
    assert recover_perp_exit(bot, plan).state is State.CLOSED
    assert transport.positions[plan.symbol] == 0


@pytest.mark.parametrize('status', [OrderState.FILLED, OrderState.REJECTED])
def test_contradictory_terminal_status_is_not_permission_to_repair(partial, status):
    bot, transport, plan, _ = partial
    receipt = replace(bot.journal.get(plan.attempt_id).perp_exit, status=status)
    bot.journal.update(plan.attempt_id, state=State.RECOVERY_REQUIRED, perp_exit=receipt,
                       failure_code='PERP_EXIT_INCOMPLETE')
    transport.orders[plan.perp_exit.client_order_id]['status'] = status.value
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 4


def test_second_partial_repair_counts_original_and_each_recovery_once(partial, monkeypatch):
    bot, transport, plan, _ = partial
    original = bot.perp.place_ioc_limit_order
    seen = []
    def partial_first(order):
        seen.append(order)
        result = original(order)
        if len(seen) == 1:
            transport.positions[plan.symbol] = D('-.004')
            transport.orders[order.client_order_id].update(status='EXPIRED', executedQty='.002',
                cumQuote=str(D('.002') * order.limit_price))
            return replace(result, status=OrderState.EXPIRED, executed_quantity=D('.002'),
                           cumulative_quote_quantity=D('.002') * order.limit_price)
        return result
    monkeypatch.setattr(bot.perp, 'place_ioc_limit_order', partial_first)
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert recover_perp_exit(bot, plan).state is State.CLOSED
    assert [o.quantity for o in seen] == [D('.006'), D('.004')]
    assert len(rows(bot)) == 2 and transport.positions[plan.symbol] == 0


@pytest.mark.parametrize('field,value', [('symbol', 'BTCUSDT'), ('market', 'SPOT'), ('side', 'SELL'),
    ('reduce_only', False), ('client_order_id', 'different-id'), ('quantity', '.005'),
    ('quantity', 0.006), ('limit_price', 2010.0)])
def test_corrupt_persisted_repair_is_not_reinterpreted_as_valid(partial, monkeypatch, field, value, caplog):
    bot, transport, plan, _ = partial
    submit = bot.perp.place_ioc_limit_order
    def lose_ack(order):
        submit(order)
        raise TimeoutError('lost receipt')
    monkeypatch.setattr(bot.perp, 'place_ioc_limit_order', lose_ack)
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    data = json.loads(rows(bot)[0]['order_json'])
    data[field] = value
    with bot.journal._connect() as db:
        db.execute('UPDATE testnet_exit_recoveries SET order_json=?', (json.dumps(data),))
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert 'RECOVERY_EVIDENCE_INVALID' in caplog.text and len(transport.orders) == 5


def test_original_receipt_in_archive_must_match_confirmed_partial(partial, monkeypatch):
    bot, transport, plan, _ = partial
    submit = bot.perp.place_ioc_limit_order
    def lose_ack(order):
        submit(order)
        raise TimeoutError('lost receipt')
    monkeypatch.setattr(bot.perp, 'place_ioc_limit_order', lose_ack)
    recover_perp_exit(bot, plan)
    data = json.loads(rows(bot)[0]['original_exit_json'])
    data['executed_quantity'] = '.003'
    with bot.journal._connect() as db:
        db.execute('UPDATE testnet_exit_recoveries SET original_exit_json=?', (json.dumps(data),))
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 5


def test_managed_exits_uses_same_original_and_remaining_only(partial):
    bot, transport, plan, _ = partial
    bot._managed_exits()
    assert bot.journal.get(plan.attempt_id).state is State.CLOSED
    bot._managed_exits()
    assert len(transport.orders) == 5 and transport.positions[plan.symbol] == 0


def test_partial_original_plus_repair_cashflow_and_wallet_events_counted_once(partial, monkeypatch):
    bot, _, plan, _ = partial
    assert recover_perp_exit(bot, plan).state is State.CLOSED
    orders = (plan.spot_entry, plan.spot_exit, plan.perp_entry, plan.perp_exit)
    receipts, trades = {}, {}
    for i, (order, price) in enumerate(zip(orders, map(D, ('2000','2001','2010','2002')), strict=True)):
        quantity = D('.004') if i == 3 else D('.01')
        oid = str(i + 1)
        receipts[order.client_order_id] = OrderReceipt(order.market, plan.symbol, oid, order.client_order_id,
            OrderState.EXPIRED if i == 3 else OrderState.FILLED, D('.01'), quantity, quantity * price, price,
            side=order.side, order_type='LIMIT', time_in_force='IOC', limit_price=order.limit_price,
            reduce_only=order.reduce_only if order.market is ExecutionMarket.USDTM else None,
            position_side='BOTH' if order.market is ExecutionMarket.USDTM else None)
        fee = quantity * price * (D('.001') if order.market is ExecutionMarket.SPOT else D('.0005'))
        pnl = None if order.market is ExecutionMarket.SPOT else (D('0') if i == 2 else D('.032'))
        trades[oid] = (AccountTrade(order.market, plan.symbol, oid, oid, order.side, price, quantity,
            price * quantity, fee, 'USDT', AT + timedelta(seconds=1 if i in (0,2) else 300), pnl,
            '{"fixture":"official trade"}'),)
    recovery = rows(bot)[0]
    cid = recovery['client_order_id']
    receipts[cid] = OrderReceipt(ExecutionMarket.USDTM, plan.symbol, '5', cid, OrderState.FILLED,
                                D('.006'), D('.006'), D('12.024'), D('2004'),
                                side=plan.perp_exit.side, order_type='LIMIT', time_in_force='IOC',
                                limit_price=D(json.loads(recovery['order_json'])['limit_price']),
                                reduce_only=True, position_side='BOTH')
    trades['5'] = (AccountTrade(ExecutionMarket.USDTM, plan.symbol, '5', '5', plan.perp_exit.side,
        D('2004'), D('.006'), D('12.024'), D('.006012'), 'USDT', AT + timedelta(seconds=301),
        D('.036'), '{"fixture":"official repair trade"}'),)
    for client in (bot.spot, bot.perp):
        monkeypatch.setattr(client, 'get_order', lambda symbol, client_id, **kw: receipts[client_id])
        monkeypatch.setattr(client, 'get_order_trades', lambda symbol, oid: trades[oid])
    payment = FundingPayment('actual-posted-amount', plan.symbol, AT + timedelta(seconds=200), D('.01'))
    monkeypatch.setattr(bot.perp, 'get_funding_payments', lambda *a: (payment,))
    bot.clock = lambda: AT + timedelta(seconds=400)
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert report.status.value == 'RECONCILED' and report.calculation.net_profit == D('.027924')
    assert report.price_pnl == D('.078') and report.fees.total == D('.060076')
    assert report.funding_total == D('.01') and report.net_return is None
    exit_leg = next(leg for leg in report.legs if leg.leg is TradeLeg.PERP_EXIT)
    assert exit_leg.quantity == D('.01') and len(exit_leg.trades) == 2
    before = tuple(BalanceSnapshot(market, AT, AT,
        (AssetBalance('USDT', D('1000')), AssetBalance('ETH', D('0')))
        if market is ExecutionMarket.SPOT else (AssetBalance('USDT', D('1000')),)) for market in ExecutionMarket)
    after = tuple(BalanceSnapshot(market, bot.clock(), bot.clock(),
        (AssetBalance('USDT', D('999.96999')), AssetBalance('ETH', D('0')))
        if market is ExecutionMarket.SPOT else (AssetBalance('USDT', D('1000.057934')),)) for market in ExecutionMarket)
    audit = audit_balance_changes(before, after, tuple(t for leg in report.legs for t in leg.trades),
                                  report.funding_payments, funding_evidence_complete=True)
    assert audit.status.value == 'MATCHED'
    assert all(m.unexplained_delta == 0 for m in audit.movements)


def test_filled_status_with_zero_quantity_is_contradictory_evidence(partial):
    bot, transport, plan, _ = partial
    current = bot.journal.get(plan.attempt_id)
    receipt = replace(current.perp_exit, status=OrderState.FILLED, executed_quantity=D('0'))
    bot.journal.update(plan.attempt_id, state=State.RECOVERY_REQUIRED, perp_exit=receipt,
                       failure_code='PERP_EXIT_INCOMPLETE')
    transport.orders[plan.perp_exit.client_order_id].update(status='FILLED', executedQty='0')
    transport.positions[plan.symbol] = -plan.perp_entry.quantity
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 4 and not rows(bot)


def test_original_query_failure_cannot_allocate_another_order_id(partial, monkeypatch):
    bot, transport, plan, _ = partial
    monkeypatch.setattr(bot.perp, 'get_order', lambda *a: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(TimeoutError):
        recover_perp_exit(bot, plan)
    assert len(transport.orders) == 4 and not rows(bot)


@pytest.mark.parametrize('mutation', ['invalid_json', 'missing_field', 'ordinal_gap'])
def test_malformed_checkpoint_preserves_recovery_required(partial, monkeypatch, mutation):
    bot, transport, plan, _ = partial
    original = bot.perp.place_ioc_limit_order
    def lose_ack(order):
        original(order)
        raise TimeoutError()
    monkeypatch.setattr(bot.perp, 'place_ioc_limit_order', lose_ack)
    recover_perp_exit(bot, plan)
    with bot.journal._connect() as db:
        if mutation == 'invalid_json':
            db.execute("UPDATE testnet_exit_recoveries SET order_json='{' ")
        elif mutation == 'ordinal_gap':
            db.execute('UPDATE testnet_exit_recoveries SET ordinal=2')
        else:
            data = json.loads(rows(bot)[0]['order_json'])
            del data['quantity']
            db.execute('UPDATE testnet_exit_recoveries SET order_json=?', (json.dumps(data),))
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 5


@pytest.mark.parametrize('status,qty', [(OrderState.FILLED, D('.003')), (OrderState.REJECTED, D('.006'))])
@pytest.mark.parametrize('cached', [True, False])
def test_contradictory_recovery_terminal_status_cannot_close_pair(partial, monkeypatch, status, qty, cached):
    bot, transport, plan, _ = partial
    submit = bot.perp.place_ioc_limit_order
    def lose_ack(order):
        submit(order)
        raise TimeoutError('offline lost ACK')
    monkeypatch.setattr(bot.perp, 'place_ioc_limit_order', lose_ack)
    recover_perp_exit(bot, plan)
    row = rows(bot)[0]
    receipt = replace(bot.perp.get_order(plan.symbol, row['client_order_id']), status=status, executed_quantity=qty)
    if cached:
        with bot.journal._connect() as db:
            db.execute('UPDATE testnet_exit_recoveries SET receipt_json=?', (_receipt_json(receipt),))
    else:
        get = bot.perp.get_order
        monkeypatch.setattr(bot.perp, 'get_order',
            lambda symbol, cid, **kw: receipt if cid == row['client_order_id'] else get(symbol, cid, **kw))
    assert recover_perp_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 5
