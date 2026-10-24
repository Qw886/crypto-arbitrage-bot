"""Native pre-entry proof, partial Spot exit and durable no-resubmit recovery."""

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs

import pytest
from test_trade_accounting import AT

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution import balance_frame_store
from crypto_arbitrage_bot.execution.account_balance_audit import AssetBalance, BalanceSnapshot
from crypto_arbitrage_bot.execution.demo import DemoResponse, demo_bot
from crypto_arbitrage_bot.execution.pair_coordinator import ExecutionJournalError
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState as State
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot
from crypto_arbitrage_bot.execution.spot_exit_recovery import (
    read_spot_exit_proof,
    recover_spot_exit,
)
from crypto_arbitrage_bot.execution.types import (
    AccountTrade,
    ExecutionMarket,
    FundingPayment,
    OrderReceipt,
    OrderState,
)
from crypto_arbitrage_bot.models.market import Market

D = Decimal


@pytest.fixture
def spot_partial(tmp_path, monkeypatch, request):
    symbol, qty, filled = getattr(request, 'param', ('ETHUSDT', '.01', '.004'))
    quantity, done = D(qty), D(filled)
    settings = load_settings(Path('docs/testnet-bot.toml'))
    settings = settings.model_copy(update={'testnet': settings.testnet.model_copy(update={
        'balance_evidence_enabled': True, 'balance_evidence_directory': tmp_path / 'frames'})})
    bot, transport = demo_bot(settings, tmp_path)
    transport.base[symbol] = D('1')  # Must never be sold by this attempt.
    before = (BalanceSnapshot(ExecutionMarket.SPOT, AT, AT,
        (AssetBalance('USDT', D('10000')), AssetBalance(symbol[:-4], D('1')))),
              BalanceSnapshot(ExecutionMarket.USDTM, AT, AT, (AssetBalance('USDT', D('10000')),)))
    monkeypatch.setattr(balance_frame_store, 'capture_balance_snapshot',
                        lambda client, **kw: before[0 if client is bot.spot else 1])
    plan = bot._plan(symbol, quantity, 'spot-partial-' + symbol)
    assert bot.coordinator.open_pair(plan).state is State.HEDGED
    original = transport.open
    def partial_exit(request, **kw):
        params = parse_qs((request.data or b'').decode())
        if params.get('newClientOrderId') == [plan.spot_exit.client_order_id]:
            transport.base[symbol] -= done
            payload = {'symbol': symbol, 'orderId': 3, 'clientOrderId': plan.spot_exit.client_order_id,
                'status': 'EXPIRED', 'origQty': str(quantity), 'executedQty': str(done),
                'cummulativeQuoteQty': str(done * D('2000')), 'avgPrice': '2000',
                'fills': [{'qty': str(done), 'price': '2000', 'commission': '0', 'commissionAsset': 'USDT'}]}
            fill_price = '2000' if symbol == 'ETHUSDT' else '50000'
            payload.update(side='SELL', type='LIMIT', timeInForce='IOC', price=params['price'][0],
                           avgPrice=fill_price, cummulativeQuoteQty=str(done * D(fill_price)))
            payload['fills'][0]['price'] = fill_price
            if done == 0:
                payload.update(fills=[], avgPrice='0')
            transport.orders[plan.spot_exit.client_order_id] = payload
            return DemoResponse(payload)
        return original(request, **kw)
    monkeypatch.setattr(transport, 'open', partial_exit)
    assert bot._close(plan).state is State.RECOVERY_REQUIRED
    monkeypatch.setattr(transport, 'open', original)
    assert transport.base[symbol] == 1 + quantity - done and len(transport.orders) == 3
    return bot, transport, plan, done


def rows(bot):
    with bot.journal._connect() as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='testnet_spot_exit_recoveries'").fetchone():
            return []
        return db.execute('SELECT * FROM testnet_spot_exit_recoveries ORDER BY ordinal').fetchall()


def finish(bot, plan):
    for _ in range(4):
        bot._managed_exits()
        if bot.journal.get(plan.attempt_id).state is State.CLOSED:
            break


@pytest.mark.parametrize('spot_partial', [('ETHUSDT', '.01', '.004'), ('BTCUSDT', '.002', '.001')], indirect=True)
def test_native_runner_recovers_spot_without_selling_original_inventory(spot_partial):
    bot, transport, plan, done = spot_partial
    original = dict(transport.orders[plan.spot_exit.client_order_id])
    finish(bot, plan)
    result = bot.journal.get(plan.attempt_id)
    assert result.state is State.CLOSED and result.spot_exit.status is OrderState.EXPIRED
    assert result.spot_exit.executed_quantity == done  # Never synthesize a FILLED original.
    assert transport.orders[plan.spot_exit.client_order_id] == original
    assert transport.base[plan.symbol] == 1 and transport.positions[plan.symbol] == 0
    assert len(transport.orders) == 5 and len(rows(bot)) == 1
    order = json.loads(rows(bot)[0]['order_json'])
    assert D(order['quantity']) == plan.spot_entry.quantity - done and order['side'] == 'SELL'
    assert not order['reduce_only'] and order['client_order_id'] != plan.spot_exit.client_order_id
    finish(bot, plan)
    assert len(transport.orders) == 5


def test_unknown_spot_recovery_receipt_queries_same_id_after_restart(spot_partial, monkeypatch):
    bot, transport, plan, _ = spot_partial
    submit, seen = bot.spot.place_ioc_limit_order, []
    def lose_ack(order):
        seen.append(order.client_order_id)
        submit(order)
        raise TimeoutError('lost Spot acknowledgement')
    monkeypatch.setattr(bot.spot, 'place_ioc_limit_order', lose_ack)
    recover_spot_exit(bot, plan)
    assert len(rows(bot)) == 1 and rows(bot)[0]['receipt_json'] is None
    assert transport.base[plan.symbol] == 1 and len(transport.orders) == 4
    restarted = Bot(bot.settings, bot.spot, bot.perp)
    finish(restarted, plan)
    assert restarted.journal.get(plan.attempt_id).state is State.CLOSED
    assert len(seen) == 1 and len(transport.orders) == 5


def test_original_perp_receipt_loss_does_not_resubmit_after_spot_repair(spot_partial, monkeypatch):
    bot, transport, plan, _ = spot_partial
    submit, seen = bot.perp.place_ioc_limit_order, []
    def lose_ack(order):
        seen.append(order.client_order_id)
        submit(order)
        raise TimeoutError('lost original Perp exit ACK')
    monkeypatch.setattr(bot.perp, 'place_ioc_limit_order', lose_ack)
    recover_spot_exit(bot, plan)
    recover_spot_exit(bot, plan)
    assert bot.journal.get(plan.attempt_id).perp_exit is None
    assert transport.positions[plan.symbol] == 0
    restarted = Bot(bot.settings, bot.spot, bot.perp)
    finish(restarted, plan)
    assert restarted.journal.get(plan.attempt_id).state is State.CLOSED
    assert seen == [plan.perp_exit.client_order_id] and len(transport.orders) == 5


@pytest.mark.parametrize('mutation', ['missing', 'invalid_json', 'changed_balance'])
def test_invalid_preentry_frame_does_not_capture_replacement_or_trade(spot_partial, mutation):
    bot, transport, plan, _ = spot_partial
    path = bot.balance_directory / (plan.attempt_id + '.before.json')
    if mutation == 'missing':
        path.unlink()
    elif mutation == 'invalid_json':
        path.write_text('{', encoding='utf-8')
    else:
        data = json.loads(path.read_text(encoding='utf-8'))
        data['snapshots'][0]['balances'][1]['amount'] = '0'
        path.write_text(json.dumps(data), encoding='utf-8')
    requests = len(transport.requests)
    assert recover_spot_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.requests) == requests and len(transport.orders) == 3


def test_current_inventory_conflict_never_sells_historical_base(spot_partial):
    bot, transport, plan, _ = spot_partial
    transport.base[plan.symbol] += D('.01')
    assert recover_spot_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert not rows(bot) and len(transport.orders) == 3


@pytest.mark.parametrize('field,value', [('position_quantity', D('-.02')), ('open_order_count', 1),
    ('one_way', False), ('single_asset', False), ('can_trade', False), ('leverage', D('2'))])
def test_conflicting_perp_account_does_not_authorize_spot_sale(spot_partial, monkeypatch, field, value):
    bot, transport, plan, _ = spot_partial
    account = bot.perp.get_account_state(plan.symbol)
    monkeypatch.setattr(bot.perp, 'get_account_state', lambda *a: replace(account, **{field: value}))
    assert recover_spot_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 3 and not rows(bot)


@pytest.mark.parametrize('status', [OrderState.NEW, OrderState.PARTIALLY_FILLED])
def test_active_original_cannot_trigger_replacement(spot_partial, status):
    bot, transport, plan, _ = spot_partial
    receipt = replace(bot.journal.get(plan.attempt_id).spot_exit, status=status)
    bot.journal.update(plan.attempt_id, state=State.RECOVERY_REQUIRED, spot_exit=receipt,
                       failure_code='SPOT_EXIT_INCOMPLETE')
    transport.orders[plan.spot_exit.client_order_id]['status'] = status.value
    assert recover_spot_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 3 and not rows(bot)


@pytest.mark.parametrize('code', [None, 'MANUAL_REVIEW_REQUIRED'])
def test_manual_recovery_is_not_implicit_trade_permission(spot_partial, code):
    bot, transport, plan, _ = spot_partial
    bot.journal.update(plan.attempt_id, state=State.RECOVERY_REQUIRED, failure_code=code)
    count = len(transport.requests)
    assert recover_spot_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 3 and len(transport.requests) == count


def test_disabled_balance_evidence_cannot_invent_native_inventory_proof(spot_partial):
    bot, transport, plan, _ = spot_partial
    bot.config = bot.config.model_copy(update={'balance_evidence_enabled': False})
    count = len(transport.requests)
    assert recover_spot_exit(bot, plan).state is State.RECOVERY_REQUIRED
    assert len(transport.requests) == count and len(transport.orders) == 3


def test_stale_exit_quote_is_not_submitted(spot_partial, monkeypatch):
    from datetime import timedelta
    bot, transport, plan, _ = spot_partial
    snapshot = bot.collector.collect_exit_depth(plan.symbol, markets=(Market.SPOT,))
    now = bot.clock()
    monkeypatch.setattr(bot.collector, 'collect_exit_depth', lambda *a, **kw: snapshot)
    bot.clock = lambda: now + timedelta(minutes=1)
    with pytest.raises(ExecutionGuardError):
        recover_spot_exit(bot, plan)
    assert len(transport.orders) == 3 and not rows(bot)


def test_multiple_partial_spot_repairs_keep_quantity_and_budget_after_restart(spot_partial, monkeypatch):
    bot, transport, plan, _ = spot_partial
    submit, seen = bot.spot.place_ioc_limit_order, []
    def partial_first(order):
        seen.append(order)
        receipt = submit(order)
        if len(seen) == 1:
            done = D('.002')
            transport.base[plan.symbol] += order.quantity - done
            transport.orders[order.client_order_id].update(status='EXPIRED', executedQty=str(done),
                cummulativeQuoteQty=str(done * order.limit_price),
                fills=[{'qty': str(done), 'price': str(order.limit_price), 'commission': '0', 'commissionAsset': 'USDT'}])
            return replace(receipt, status=OrderState.EXPIRED, executed_quantity=done,
                           cumulative_quote_quantity=done * order.limit_price)
        return receipt
    monkeypatch.setattr(bot.spot, 'place_ioc_limit_order', partial_first)
    recover_spot_exit(bot, plan)
    restarted = Bot(bot.settings, bot.spot, bot.perp)
    finish(restarted, plan)
    assert restarted.journal.get(plan.attempt_id).state is State.CLOSED
    assert [o.quantity for o in seen] == [D('.006'), D('.004')]
    assert len(rows(bot)) == 2 and len(transport.orders) == 6
    assert transport.base[plan.symbol] == 1 and transport.positions[plan.symbol] == 0


def test_zero_fill_repairs_have_durable_finite_budget(spot_partial, monkeypatch):
    bot, transport, plan, _ = spot_partial
    submit, seen = bot.spot.place_ioc_limit_order, []
    def zero_fill(order):
        seen.append(order)
        receipt = submit(order)
        transport.base[plan.symbol] += order.quantity
        transport.orders[order.client_order_id].update(status='EXPIRED', executedQty='0',
            cummulativeQuoteQty='0', avgPrice='0', fills=[])
        return replace(receipt, status=OrderState.EXPIRED, executed_quantity=D('0'),
                       cumulative_quote_quantity=D('0'), average_price=D('0'), base_commission=D('0'))
    monkeypatch.setattr(bot.spot, 'place_ioc_limit_order', zero_fill)
    for _ in range(8):
        restarted = Bot(bot.settings, bot.spot, bot.perp)
        finish(restarted, plan)
    assert len(seen) == bot.config.max_exit_recovery_attempts == 2
    assert len(rows(bot)) == 2 and len(transport.orders) == 5
    assert transport.base[plan.symbol] == D('1.006') and transport.positions[plan.symbol] == D('-.01')
    assert restarted.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


@pytest.mark.parametrize('phase', ['claim', 'receipt'])
def test_sqlite_failure_never_resubmits_unknown_spot_order(spot_partial, phase):
    bot, transport, plan, _ = spot_partial
    assert read_spot_exit_proof(bot, plan) is not None
    event = 'INSERT' if phase == 'claim' else 'UPDATE'
    with bot.journal._connect() as db:
        db.execute(f"""CREATE TRIGGER fail_spot BEFORE {event} ON testnet_spot_exit_recoveries
                       BEGIN SELECT RAISE(ABORT, 'offline injected write failure'); END""")
    with pytest.raises(ExecutionJournalError):
        recover_spot_exit(bot, plan)
    assert len(transport.orders) == (3 if phase == 'claim' else 4)
    with bot.journal._connect() as db:
        db.execute('DROP TRIGGER fail_spot')
    restarted = Bot(bot.settings, bot.spot, bot.perp)
    finish(restarted, plan)
    assert restarted.journal.get(plan.attempt_id).state is State.CLOSED
    assert len(transport.orders) == 5 and len(rows(bot)) == 1


@pytest.mark.parametrize('field,value', [('symbol', 'BTCUSDT'), ('market', 'USDTM'),
    ('side', 'BUY'), ('reduce_only', True), ('client_order_id', 'other-id'),
    ('quantity', '.005'), ('quantity', 0.006), ('limit_price', 2000.0)])
def test_corrupt_spot_repair_identity_blocks_further_orders(spot_partial, field, value, caplog):
    bot, transport, plan, _ = spot_partial
    recover_spot_exit(bot, plan)
    data = json.loads(rows(bot)[0]['order_json'])
    data[field] = value
    with bot.journal._connect() as db:
        db.execute('UPDATE testnet_spot_exit_recoveries SET order_json=?', (json.dumps(data),))
    finish(bot, plan)
    assert 'SPOT_RECOVERY_EVIDENCE_INVALID' in caplog.text
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 4 and transport.positions[plan.symbol] == D('-.01')


@pytest.mark.parametrize('mutation', ['invalid_json', 'ordinal_gap', 'original_quantity', 'original_fee'])
def test_spot_repair_checkpoint_damage_cannot_fake_completion(spot_partial, mutation):
    bot, transport, plan, _ = spot_partial
    recover_spot_exit(bot, plan)
    with bot.journal._connect() as db:
        if mutation == 'invalid_json':
            db.execute("UPDATE testnet_spot_exit_recoveries SET order_json='{'")
        elif mutation == 'ordinal_gap':
            db.execute('UPDATE testnet_spot_exit_recoveries SET ordinal=2')
        else:
            data = json.loads(rows(bot)[0]['original_exit_json'])
            data['executed_quantity' if mutation == 'original_quantity' else 'base_commission'] = '.003'
            db.execute('UPDATE testnet_spot_exit_recoveries SET original_exit_json=?', (json.dumps(data),))
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED and len(transport.orders) == 4


def test_unknown_original_order_blocks_repair(spot_partial, monkeypatch):
    bot, transport, plan, _ = spot_partial
    monkeypatch.setattr(bot.spot, 'get_order', lambda *a, **kw: (_ for _ in ()).throw(TimeoutError()))
    finish(bot, plan)  # Runner records a bounded error; remaining exposure stays visible.
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED and len(transport.orders) == 3


def test_changed_original_execution_requires_reconciliation(spot_partial):
    bot, transport, plan, _ = spot_partial
    transport.orders[plan.spot_exit.client_order_id]['executedQty'] = '.005'
    finish(bot, plan)
    assert len(transport.orders) == 3 and bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


@pytest.mark.parametrize('spot_partial', [('ETHUSDT', '.01', '0')], indirect=True)
@pytest.mark.parametrize('status', ['EXPIRED', 'CANCELED', 'EXPIRED_IN_MATCH', 'REJECTED'])
def test_terminal_zero_original_spot_exit_recovers_only_known_inventory(spot_partial, status):
    bot, transport, plan, _ = spot_partial
    current = bot.journal.get(plan.attempt_id)
    transport.orders[plan.spot_exit.client_order_id]['status'] = status
    bot.journal.update(plan.attempt_id, state=State.RECOVERY_REQUIRED,
        spot_exit=replace(current.spot_exit, status=OrderState(status)), failure_code='SPOT_EXIT_INCOMPLETE')
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.CLOSED and transport.base[plan.symbol] == 1
    assert D(json.loads(rows(bot)[0]['order_json'])['quantity']) == D('.01') and len(transport.orders) == 5


def make_partial_perp(bot, transport, plan, monkeypatch, done):
    submit = bot.perp.place_ioc_limit_order
    def partial_original(order):
        receipt = submit(order)
        if order.client_order_id == plan.perp_exit.client_order_id:
            transport.positions[plan.symbol] -= order.quantity - done
            transport.orders[order.client_order_id].update(status='EXPIRED', executedQty=str(done),
                cummulativeQuoteQty=str(done * order.limit_price),
                avgPrice=str(order.limit_price if done else D('0')))
            return replace(receipt, status=OrderState.EXPIRED, executed_quantity=done,
                           cumulative_quote_quantity=done * order.limit_price,
                           average_price=order.limit_price if done else D('0'))
        return receipt
    monkeypatch.setattr(bot.perp, 'place_ioc_limit_order', partial_original)


@pytest.mark.parametrize('done', [D('0'), D('.004')])
def test_spot_repair_then_partial_perp_uses_separate_unique_reduce_only_repair(spot_partial, monkeypatch, done):
    bot, transport, plan, _ = spot_partial
    make_partial_perp(bot, transport, plan, monkeypatch, done)
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.CLOSED
    assert transport.base[plan.symbol] == 1 and transport.positions[plan.symbol] == 0
    assert len(transport.orders) == 6
    with bot.journal._connect() as db:
        repairs = db.execute('SELECT order_json FROM testnet_exit_recoveries').fetchall()
    assert len(repairs) == 1
    order = json.loads(repairs[0][0])
    assert order['reduce_only'] is True and D(order['quantity']) == D('.01') - done


def test_inventory_conflict_after_spot_repair_also_blocks_perp_repair(spot_partial, monkeypatch, caplog):
    bot, transport, plan, _ = spot_partial
    make_partial_perp(bot, transport, plan, monkeypatch, D('.004'))
    recover_spot_exit(bot, plan)
    recover_spot_exit(bot, plan)  # Original Perp exit partially filled, no repair yet.
    transport.base[plan.symbol] += D('.001')
    finish(bot, plan)
    assert 'SPOT_RECOVERY_INVENTORY_CONFLICT' in caplog.text
    assert len(transport.orders) == 5 and transport.positions[plan.symbol] == D('-.006')
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


@pytest.mark.parametrize('spot_partial', [('ETHUSDT', '.01', '.0095')], indirect=True)
def test_remaining_spot_below_filter_never_rounds_up_into_original_inventory(spot_partial):
    bot, transport, plan, _ = spot_partial
    finish(bot, plan)
    assert len(transport.orders) == 3 and not rows(bot)
    assert transport.base[plan.symbol] == D('1.0005') and transport.positions[plan.symbol] == D('-.01')
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED


def test_confirmed_base_exit_commission_reduces_sale_without_consuming_old_inventory(spot_partial, monkeypatch):
    bot, transport, plan, _ = spot_partial
    commission = D('.00001')
    original = bot.spot.get_order
    monkeypatch.setattr(bot.spot, 'get_order', lambda symbol, cid, **kw:
        replace(original(symbol, cid, **kw), base_commission=commission)
        if cid == plan.spot_exit.client_order_id else original(symbol, cid, **kw))
    transport.base[plan.symbol] -= commission
    rules = bot.spot.get_symbol_rules(plan.symbol)
    monkeypatch.setattr(bot.spot, 'get_symbol_rules', lambda *a:
        replace(rules, step_size=D('.000001'), min_quantity=D('.000001')))
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.CLOSED
    assert D(json.loads(rows(bot)[0]['order_json'])['quantity']) == D('.00599')
    assert transport.base[plan.symbol] == 1 and transport.positions[plan.symbol] == 0


@pytest.mark.parametrize('mutation', ['unknown_fee', 'filled_partial', 'rejected_positive'])
def test_ambiguous_spot_fill_cannot_authorize_perp_exit(spot_partial, mutation):
    bot, transport, plan, _ = spot_partial
    recover_spot_exit(bot, plan)
    receipt = json.loads(rows(bot)[0]['receipt_json'])
    if mutation == 'unknown_fee':
        receipt['base_commission'] = None
    elif mutation == 'filled_partial':
        receipt['executed_quantity'] = '.003'
    else:
        receipt['status'] = 'REJECTED'
    with bot.journal._connect() as db:
        db.execute('UPDATE testnet_spot_exit_recoveries SET receipt_json=?', (json.dumps(receipt),))
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 4 and transport.positions[plan.symbol] == D('-.01')


@pytest.mark.parametrize('funding', [D('.01'), D('0'), D('-.01')])
def test_six_actual_fills_have_one_cashflow_fee_and_funding_accounting(spot_partial, monkeypatch, funding):
    from crypto_arbitrage_bot.execution.account_balance_audit import audit_balance_changes
    from crypto_arbitrage_bot.execution.trade_reconciliation import reconcile_closed_attempt
    from crypto_arbitrage_bot.models.calculation import TradeLeg
    bot, transport, plan, _ = spot_partial
    make_partial_perp(bot, transport, plan, monkeypatch, D('.004'))
    finish(bot, plan)
    assert bot.journal.get(plan.attempt_id).state is State.CLOSED
    with bot.journal._connect() as db:
        perp_data = json.loads(db.execute('SELECT order_json FROM testnet_exit_recoveries').fetchone()[0])
        before_rows = tuple(tuple(r) for r in db.execute('SELECT * FROM testnet_pair_attempts'))
    spot_data = json.loads(rows(bot)[0]['order_json'])
    intents = {order.client_order_id: order for order in bot.journal.intents(plan.attempt_id)}
    limits = {cid: order.limit_price for cid, order in intents.items()}
    limits.update({data['client_order_id']: D(data['limit_price']) for data in (spot_data, perp_data)})
    orders = ((plan.spot_entry.client_order_id, ExecutionMarket.SPOT, plan.spot_entry.side, D('.01'), D('2000'), D('0')),
        (plan.spot_exit.client_order_id, ExecutionMarket.SPOT, plan.spot_exit.side, D('.004'), D('2001'), D('0')),
        (plan.perp_entry.client_order_id, ExecutionMarket.USDTM, plan.perp_entry.side, D('.01'), D('2010'), D('0')),
        (plan.perp_exit.client_order_id, ExecutionMarket.USDTM, plan.perp_exit.side, D('.004'), D('2002'), D('.032')),
        (spot_data['client_order_id'], ExecutionMarket.SPOT, plan.spot_exit.side, D('.006'), D('2003'), D('0')),
        (perp_data['client_order_id'], ExecutionMarket.USDTM, plan.perp_exit.side, D('.006'), D('2004'), D('.036')))
    receipts, trades = {}, {}
    for i, (cid, market, side, qty, price, pnl) in enumerate(orders, start=1):
        original_qty = D('.01') if i <= 4 else D('.006')
        receipts[cid] = OrderReceipt(market, plan.symbol, str(i), cid,
            OrderState.EXPIRED if i in (2, 4) else OrderState.FILLED, original_qty, qty, qty * price, price,
            side=side, order_type='LIMIT', time_in_force='IOC', limit_price=limits[cid],
            reduce_only=(side.value == 'BUY') if market is ExecutionMarket.USDTM else None,
            position_side='BOTH' if market is ExecutionMarket.USDTM else None)
        fee = qty * price * (D('.001') if market is ExecutionMarket.SPOT else D('.0005'))
        trades[str(i)] = (AccountTrade(market, plan.symbol, str(i), str(i), side, price, qty,
            price * qty, fee, 'USDT', AT + timedelta(seconds=1 if i in (1, 3) else 300 + i),
            None if market is ExecutionMarket.SPOT else pnl, '{"fixture":"actual fill"}'),)
    for client in (bot.spot, bot.perp):
        monkeypatch.setattr(client, 'get_order', lambda symbol, cid, **kw: receipts[cid])
        monkeypatch.setattr(client, 'get_order_trades', lambda symbol, oid: trades[oid])
        monkeypatch.setattr(client, 'place_ioc_limit_order',
            lambda *a: (_ for _ in ()).throw(AssertionError('accounting must never order')))
    payment = FundingPayment('one-explicit-official-event', plan.symbol, AT + timedelta(seconds=200), funding)
    monkeypatch.setattr(bot.perp, 'get_funding_payments', lambda *a: (payment,))
    bot.clock = lambda: AT + timedelta(seconds=400)
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert report.status.value == 'RECONCILED'
    assert report.price_pnl == D('.09') and report.fees.total == D('.060088')
    assert report.funding_total == funding and report.calculation.net_profit == D('.029912') + funding
    assert report.net_return is None and report.funding_payments == (payment,)
    for leg in (TradeLeg.SPOT_EXIT, TradeLeg.PERP_EXIT):
        evidence = next(e for e in report.legs if e.leg is leg)
        assert evidence.quantity == D('.01') and len(evidence.trades) == 2
    before = (BalanceSnapshot(ExecutionMarket.SPOT, AT, AT,
        (AssetBalance('USDT', D('1000')), AssetBalance('ETH', D('1')))),
              BalanceSnapshot(ExecutionMarket.USDTM, AT, AT, (AssetBalance('USDT', D('1000')),)))
    after = (BalanceSnapshot(ExecutionMarket.SPOT, bot.clock(), bot.clock(),
        (AssetBalance('USDT', D('999.981978')), AssetBalance('ETH', D('1')))),
             BalanceSnapshot(ExecutionMarket.USDTM, bot.clock(), bot.clock(),
                (AssetBalance('USDT', D('1000.047934') + funding),)))
    audit = audit_balance_changes(before, after, tuple(t for e in report.legs for t in e.trades),
        report.funding_payments, funding_evidence_complete=True)
    assert audit.status.value == 'MATCHED' and all(m.unexplained_delta == 0 for m in audit.movements)
    with bot.journal._connect() as db:
        assert tuple(tuple(r) for r in db.execute('SELECT * FROM testnet_pair_attempts')) == before_rows
