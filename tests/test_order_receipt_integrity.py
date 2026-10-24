"""Order replies are exchange evidence, never inferred from the submitted intent."""

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
from test_testnet_execution import Opener, Response, client, exchange_info
from test_testnet_pair_coordinator import make_plan
from test_trade_accounting import evidence_bot as evidence_bot

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.demo import DemoResponse, demo_bot
from crypto_arbitrage_bot.execution.pair_coordinator import (
    PairExecutionState,
    SQLitePairExecutionJournal,
)
from crypto_arbitrage_bot.execution.testnet_client import (
    ExecutionErrorKind,
)
from crypto_arbitrage_bot.execution.testnet_client import (
    TestnetExecutionError as ExecutionError,
)
from crypto_arbitrage_bot.execution.types import ExecutionMarket, IocLimitOrder
from crypto_arbitrage_bot.models.market import Side


def intended(market=ExecutionMarket.USDTM, *, side=Side.SELL, reduce_only=False):
    return IocLimitOrder(market, 'BTCUSDT', side, Decimal('0.001'),
                         Decimal('50000'), 'receipt-check', reduce_only)


def reply(order):
    payload = {
        'symbol': 'BTCUSDT', 'orderId': 123, 'clientOrderId': 'receipt-check',
        'status': 'FILLED', 'origQty': '0.001', 'executedQty': '0.001',
        'price': '50000', 'side': order.side.value, 'type': 'LIMIT',
        'timeInForce': 'IOC', 'cumQuote': '50', 'avgPrice': '50000',
    }
    if order.market is ExecutionMarket.USDTM:
        payload.update(reduceOnly=order.reduce_only, positionSide='BOTH')
    else:
        payload['fills'] = [{'qty': '0.001', 'price': '50000',
                             'commission': '0', 'commissionAsset': 'USDT'}]
    return payload


def test_contradictory_direction_is_unknown_after_one_submission():
    order = intended()
    payload = reply(order)
    payload['side'] = 'BUY'
    transport = Opener([Response(exchange_info()), Response({'serverTime': 10_000}),
                        Response(payload)])
    api = client(order.market, transport)
    with pytest.raises(ExecutionError) as error:
        api.place_ioc_limit_order(order)
    assert error.value.kind is ExecutionErrorKind.SUBMISSION_UNKNOWN
    assert error.value.submission_may_have_reached_exchange
    assert [r.get_method() for r, _ in transport.requests] == ['GET', 'GET', 'POST']


@pytest.mark.parametrize('field,value', [
    ('type', 'MARKET'), ('timeInForce', 'GTC'), ('price', '50001'),
    ('reduceOnly', True), ('positionSide', 'SHORT'),
])
def test_conflicting_order_semantics_do_not_confirm_a_fill(field, value):
    order = intended()
    payload = reply(order)
    payload[field] = value
    transport = Opener([Response(exchange_info()), Response({'serverTime': 10_000}),
                        Response(payload)])
    with pytest.raises(ExecutionError) as error:
        client(order.market, transport).place_ioc_limit_order(order)
    assert error.value.kind is ExecutionErrorKind.SUBMISSION_UNKNOWN
    assert error.value.submission_may_have_reached_exchange
    assert sum(r.get_method() == 'POST' for r, _ in transport.requests) == 1


def test_restart_retains_returned_semantics_in_durable_receipt(tmp_path):
    plan = make_plan('receipt-check')
    order = plan.spot_entry
    payload = reply(order)
    payload['clientOrderId'] = order.client_order_id
    transport = Opener([Response(exchange_info()), Response({'serverTime': 10_000}),
                        Response(payload)])
    receipt = client(order.market, transport).place_ioc_limit_order(order)
    journal = SQLitePairExecutionJournal(tmp_path / 'execution.sqlite')
    journal.initialize()
    assert journal.create(plan)
    journal.update(plan.attempt_id, state=PairExecutionState.OPENING_PERP, spot_entry=receipt)
    reopened = SQLitePairExecutionJournal(journal.database_path)
    saved = reopened.get(plan.attempt_id).spot_entry
    assert saved == receipt
    assert saved.matches_order(order)
    assert not saved.matches_order(replace(order, side=Side.SELL))


@pytest.mark.parametrize('side,field,value', [
    (Side.BUY, 'avgPrice', '50001'), (Side.BUY, 'cumQuote', '50.001'),
    (Side.SELL, 'avgPrice', '49999'), (Side.SELL, 'cumQuote', '49.999'),
])
def test_fill_outside_submitted_limit_is_unknown(side, field, value):
    order = intended(side=side)
    payload = reply(order)
    payload[field] = value
    transport = Opener([Response(exchange_info()), Response({'serverTime': 10_000}),
                        Response(payload)])
    with pytest.raises(ExecutionError) as error:
        client(order.market, transport).place_ioc_limit_order(order)
    assert error.value.kind is ExecutionErrorKind.SUBMISSION_UNKNOWN
    assert sum(r.get_method() == 'POST' for r, _ in transport.requests) == 1


@pytest.mark.parametrize('status,quantity,quote', [('FILLED', '0.0005', '25'),
                                                ('REJECTED', '0.001', '50')])
def test_terminal_status_must_agree_with_executed_quantity(status, quantity, quote):
    order = intended()
    payload = reply(order)
    payload.update(status=status, executedQty=quantity, cumQuote=quote)
    transport = Opener([Response(exchange_info()), Response({'serverTime': 10_000}),
                        Response(payload)])
    with pytest.raises(ExecutionError) as error:
        client(order.market, transport).place_ioc_limit_order(order)
    assert error.value.kind is ExecutionErrorKind.SUBMISSION_UNKNOWN


def test_spot_individual_fill_cannot_exceed_limit_despite_valid_average():
    order = intended(ExecutionMarket.SPOT, side=Side.BUY)
    payload = reply(order)
    payload['fills'] = [
        {'qty': '0.0005', 'price': '49000', 'commission': '0', 'commissionAsset': 'USDT'},
        {'qty': '0.0005', 'price': '51000', 'commission': '0', 'commissionAsset': 'USDT'},
    ]
    transport = Opener([Response(exchange_info()), Response({'serverTime': 10_000}),
                        Response(payload)])
    with pytest.raises(ExecutionError) as error:
        client(order.market, transport).place_ioc_limit_order(order)
    assert error.value.kind is ExecutionErrorKind.SUBMISSION_UNKNOWN


@pytest.fixture
def receipt_bot(tmp_path):
    return demo_bot(load_settings(Path('docs/testnet-bot.toml')), tmp_path)


def corrupt_wire(transport, monkeypatch, client_id, changes, *, post_only=False):
    original = transport.open

    def open_reply(request, **kwargs):
        response = original(request, **kwargs)
        payload = json.loads(response.body)
        if (isinstance(payload, dict) and payload.get('clientOrderId') == client_id
                and (not post_only or request.get_method() == 'POST')):
            payload.update(changes)
            return DemoResponse(payload)
        return response

    monkeypatch.setattr(transport, 'open', open_reply)


@pytest.mark.parametrize('leg', ['spot_entry', 'perp_entry', 'spot_exit', 'perp_exit'])
def test_conflicting_reply_stops_later_orders_and_survives_restart(receipt_bot, monkeypatch, leg):
    bot, transport = receipt_bot
    plan = bot._plan('ETHUSDT', Decimal('.01'), 'receipt-conflict')
    if leg.endswith('exit'):
        assert bot.coordinator.open_pair(plan).state is PairExecutionState.HEDGED
    order = getattr(plan, leg)
    corrupt_wire(transport, monkeypatch, order.client_order_id,
                 {'side': 'SELL' if order.side is Side.BUY else 'BUY'})
    result = (bot.coordinator.close_pair(plan) if leg.endswith('exit')
              else bot.coordinator.open_pair(plan))
    assert result.state is PairExecutionState.RECOVERY_REQUIRED
    counts = {'spot_entry': 1, 'perp_entry': 2, 'spot_exit': 3, 'perp_exit': 4}
    assert len(transport.orders) == counts[leg]
    from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot

    fresh = Bot(bot.settings, bot.spot, bot.perp)
    assert fresh.coordinator.reconcile(plan).state is PairExecutionState.RECOVERY_REQUIRED
    assert len(transport.orders) == counts[leg]


def test_valid_original_query_resolves_conflicting_ack_without_resubmission(receipt_bot, monkeypatch):
    bot, transport = receipt_bot
    plan = bot._plan('ETHUSDT', Decimal('.01'), 'query-resolves')
    corrupt_wire(transport, monkeypatch, plan.perp_entry.client_order_id, {'side': 'BUY'}, post_only=True)
    assert bot.coordinator.open_pair(plan).state is PairExecutionState.HEDGED
    assert len(transport.orders) == 2


@pytest.mark.parametrize('order_id', [True, '', 'not-a-number', -1])
def test_invalid_exchange_order_identity_cannot_confirm_submission(order_id):
    order = intended()
    payload = reply(order)
    payload['orderId'] = order_id
    transport = Opener([Response(exchange_info()), Response({'serverTime': 10_000}),
                        Response(payload)])
    with pytest.raises(ExecutionError) as error:
        client(order.market, transport).place_ioc_limit_order(order)
    assert error.value.kind is ExecutionErrorKind.SUBMISSION_UNKNOWN


def test_query_cannot_replace_an_acknowledged_exchange_order_identity(receipt_bot, monkeypatch):
    bot, transport = receipt_bot
    plan = bot._plan('ETHUSDT', Decimal('.01'), 'identity-conflict')
    original = transport.open

    def changed_identity(request, **kwargs):
        response = original(request, **kwargs)
        payload = json.loads(response.body)
        if isinstance(payload, dict) and payload.get('clientOrderId') == plan.perp_entry.client_order_id:
            if request.get_method() == 'POST':
                payload.update(status='NEW', executedQty='0', cummulativeQuoteQty='0', avgPrice='0')
            else:
                payload['orderId'] = 999
            return DemoResponse(payload)
        return response

    monkeypatch.setattr(transport, 'open', changed_identity)
    result = bot.coordinator.open_pair(plan)
    assert result.state is PairExecutionState.RECOVERY_REQUIRED
    assert result.perp_entry.order_id == str(transport.orders[plan.perp_entry.client_order_id]['orderId'])
    assert bot.coordinator.reconcile(plan).state is PairExecutionState.RECOVERY_REQUIRED
    assert len(transport.orders) == 2


def test_restart_query_must_preserve_previously_saved_order_identity(receipt_bot, monkeypatch):
    bot, transport = receipt_bot
    plan = bot._plan('ETHUSDT', Decimal('.01'), 'restart-identity')
    assert bot.coordinator.open_pair(plan).state is PairExecutionState.HEDGED
    bot.journal.update(plan.attempt_id, state=PairExecutionState.RECOVERY_REQUIRED)
    corrupt_wire(transport, monkeypatch, plan.perp_entry.client_order_id, {'orderId': 999})
    assert bot.coordinator.reconcile(plan).state is PairExecutionState.RECOVERY_REQUIRED
    assert len(transport.orders) == 2


def test_cashflow_report_refuses_contradictory_receipt_semantics(evidence_bot, monkeypatch):
    from crypto_arbitrage_bot.execution.trade_reconciliation import (
        ReconciliationCode,
        ReconciliationStatus,
        reconcile_closed_attempt,
    )
    bot, plan, receipts, _ = evidence_bot
    original = receipts[plan.perp_entry.client_order_id]
    receipts[plan.perp_entry.client_order_id] = replace(original, order_type='MARKET')
    report = reconcile_closed_attempt(bot, plan.attempt_id)
    assert report.status is not ReconciliationStatus.RECONCILED
    assert report.calculation is None
    assert any(issue.code is ReconciliationCode.IDENTITY_MISMATCH for issue in report.issues)


@pytest.mark.parametrize('field', ['avgPrice', 'cumQuote'])
def test_zero_fill_cannot_claim_nonzero_execution_amount(field):
    order = intended()
    payload = reply(order)
    payload.update(status='EXPIRED', executedQty='0', cumQuote='0', avgPrice='0')
    payload[field] = '1'
    transport = Opener([Response(exchange_info()), Response({'serverTime': 10_000}),
                        Response(payload)])
    with pytest.raises(ExecutionError) as error:
        client(order.market, transport).place_ioc_limit_order(order)
    assert error.value.kind is ExecutionErrorKind.SUBMISSION_UNKNOWN


@pytest.mark.parametrize('market,field', [
    (market, field) for market in ExecutionMarket for field in ('side', 'type', 'timeInForce', 'price')
] + [(ExecutionMarket.USDTM, 'reduceOnly'), (ExecutionMarket.USDTM, 'positionSide')])
def test_missing_returned_semantics_are_never_inferred_from_request(market, field):
    order = intended(market)
    payload = reply(order)
    del payload[field]
    transport = Opener([Response(exchange_info()), Response({'serverTime': 10_000}),
                        Response(payload)])
    with pytest.raises(ExecutionError) as error:
        client(order.market, transport).place_ioc_limit_order(order)
    assert error.value.kind is ExecutionErrorKind.SUBMISSION_UNKNOWN


@pytest.mark.parametrize('changes', [{'symbol': 'ETHUSDT'}, {'clientOrderId': 'wrong-id'},
                                    {'side': None}, {'reduceOnly': 'false'}, {'price': 50000}])
def test_bad_query_is_a_read_failure_with_no_submission_flag(changes):
    order = intended()
    payload = reply(order)
    payload.update(changes)
    transport = Opener([Response({'serverTime': 10_000}), Response(payload)])
    with pytest.raises(ExecutionError) as error:
        client(order.market, transport).get_order(order.symbol, order.client_order_id)
    assert error.value.kind is ExecutionErrorKind.INVALID_RESPONSE
    assert error.value.market is order.market
    assert not error.value.submission_may_have_reached_exchange
    assert all(r.get_method() == 'GET' for r, _ in transport.requests)


@pytest.mark.parametrize('market', list(ExecutionMarket))
@pytest.mark.parametrize('side', list(Side))
def test_complete_original_query_preserves_semantics_without_orders(market, side):
    order = intended(market, side=side, reduce_only=market is ExecutionMarket.USDTM and side is Side.BUY)
    payload = reply(order)
    if market is ExecutionMarket.SPOT:
        del payload['avgPrice']  # Spot can establish a bound using cumulative quote alone.
    transport = Opener([Response({'serverTime': 10_000}), Response(payload)])
    receipt = client(market, transport).get_order(order.symbol, order.client_order_id,
                                               include_base_commission=False)
    assert receipt.matches_order(order)
    assert receipt.limit_price == Decimal('50000')
    assert receipt.side is side
    assert all(r.get_method() == 'GET' for r, _ in transport.requests)


def test_legacy_receipt_remains_readable_but_cannot_establish_semantics(tmp_path):
    from crypto_arbitrage_bot.execution.types import OrderReceipt, OrderState

    plan = make_plan('legacy-record')
    old = OrderReceipt(ExecutionMarket.SPOT, 'BTCUSDT', '123', plan.spot_entry.client_order_id,
                       OrderState.FILLED, Decimal('.001'), Decimal('.001'), Decimal('50'), None)
    journal = SQLitePairExecutionJournal(tmp_path / 'legacy.sqlite')
    journal.initialize()
    assert journal.create(plan)
    journal.update(plan.attempt_id, state=PairExecutionState.OPENING_PERP, spot_entry=old)
    reopened = SQLitePairExecutionJournal(journal.database_path)
    saved = reopened.get(plan.attempt_id).spot_entry
    assert saved == old
    assert saved.side is saved.limit_price is saved.reduce_only is None
    assert not saved.matches_order(plan.spot_entry)


@pytest.mark.parametrize('changes', [{'side': Side.SELL}, {'limit_price': Decimal('50001')},
                                    {'reduce_only': True}, {'time_in_force': 'GTC'}])
def test_typed_receipt_match_requires_exact_submitted_order(changes):
    order = intended()
    transport = Opener([Response({'serverTime': 10_000}), Response(reply(order))])
    receipt = client(order.market, transport).get_order(order.symbol, order.client_order_id)
    if 'side' in changes:
        changes = {'side': Side.BUY}
    assert not replace(receipt, **changes).matches_order(order)


def test_quote_limit_comparison_is_independent_of_low_precision_caller():
    from decimal import localcontext

    order = intended(side=Side.BUY)
    transport = Opener([Response({'serverTime': 10_000}), Response(reply(order))])
    receipt = client(order.market, transport).get_order(order.symbol, order.client_order_id)
    with localcontext() as context:
        context.prec = 2
        assert receipt.matches_order(order)
        assert not replace(receipt, cumulative_quote_quantity=Decimal('50.000000001')).matches_order(order)


@pytest.mark.parametrize('order_id', [None, True, 'opaque-order'])
def test_typed_gateway_receipt_requires_a_numeric_exchange_identity(order_id):
    order = intended()
    transport = Opener([Response({'serverTime': 10_000}), Response(reply(order))])
    receipt = client(order.market, transport).get_order(order.symbol, order.client_order_id)
    assert not replace(receipt, order_id=order_id).matches_order(order)


def test_query_cannot_reduce_a_previously_confirmed_partial_fill(receipt_bot, monkeypatch):
    bot, transport = receipt_bot
    plan = bot._plan('ETHUSDT', Decimal('.01'), 'partial-rollback')
    original = transport.open

    def rollback(request, **kwargs):
        response = original(request, **kwargs)
        payload = json.loads(response.body)
        if isinstance(payload, dict) and payload.get('clientOrderId') == plan.perp_entry.client_order_id:
            quantity = Decimal('.004') if request.get_method() == 'POST' else Decimal('.002')
            payload.update(status='PARTIALLY_FILLED' if request.get_method() == 'POST' else 'EXPIRED',
                           executedQty=str(quantity), cummulativeQuoteQty=str(quantity * plan.perp_entry.limit_price))
            return DemoResponse(payload)
        return response

    monkeypatch.setattr(transport, 'open', rollback)
    result = bot.coordinator.open_pair(plan)
    assert result.state is PairExecutionState.RECOVERY_REQUIRED
    assert result.perp_entry.executed_quantity == Decimal('.004')
    assert bot.coordinator.reconcile(plan).state is PairExecutionState.RECOVERY_REQUIRED
    assert len(transport.orders) == 2
