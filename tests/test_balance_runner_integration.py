"""Native Runner lifecycle using synthetic official fills and wallet evidence."""

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_trade_accounting import AT, source_rows

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution import balance_accounting, balance_frame_store
from crypto_arbitrage_bot.execution.account_balance_audit import AssetBalance, BalanceSnapshot
from crypto_arbitrage_bot.execution.balance_audit_repository import BalanceAuditRepository
from crypto_arbitrage_bot.execution.balance_check_repository import (
    BalanceCheckRepository,
    BalanceCheckStatus,
)
from crypto_arbitrage_bot.execution.balance_ledger import (
    balance_entry_blocker,
    reconcile_balance_ledger,
)
from crypto_arbitrage_bot.execution.demo import demo_bot
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.testnet_client import (
    ExecutionErrorKind,
)
from crypto_arbitrage_bot.execution.testnet_client import (
    TestnetExecutionError as GatewayError,
)
from crypto_arbitrage_bot.execution.types import (
    AccountTrade,
    ExecutionMarket,
    FundingPayment,
    OrderReceipt,
    OrderState,
)
from crypto_arbitrage_bot.runner import CycleStatus

D = Decimal


def gateway_failure():
    error = GatewayError(ExecutionErrorKind.NETWORK_ERROR, 'unsafe-payload-credential',
                         market=ExecutionMarket.USDTM)
    error.endpoint, error.method, error.phase = '/fapi/v2/account', 'GET', 'READ'
    error.transport_error_type, error.timeout_seconds = 'TimeoutError', 5
    return error


def test_preentry_error_preserves_safe_source_metadata(bot, monkeypatch):
    instance, transport, _ = bot
    monkeypatch.setattr(balance_frame_store, 'capture_balance_snapshot', Mock(side_effect=gateway_failure()))
    with pytest.raises(ExecutionGuardError) as captured:
        instance._plan('ETHUSDT', D('.01'), 'diagnostic-before')
    source = captured.value.diagnostics['source_error']
    assert source['kind'] == 'NETWORK_ERROR' and source['endpoint'] == '/fapi/v2/account'
    assert source['phase'] == 'READ' and source['timeout_seconds'] == '5'
    assert 'unsafe-payload-credential' not in str(captured.value.diagnostics)
    assert not transport.orders and not transport.requests


def test_postentry_error_retains_safe_diagnostics_and_pending_state(closed, monkeypatch, caplog):
    bot, _, plan, _, _, _ = closed
    monkeypatch.setattr(balance_accounting, 'capture_balance_snapshot', Mock(side_effect=gateway_failure()))
    with caplog.at_level('INFO'):
        reconcile_balance_ledger(bot)
    check = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    assert check.status is BalanceCheckStatus.PENDING and check.failure_code == 'NETWORK_ERROR'
    event = next(json.loads(r.message) for r in caplog.records if 'TESTNET_BALANCE_LEDGER' in r.message)
    assert event['source_error']['endpoint'] == '/fapi/v2/account'
    assert event['source_error']['transport_error_type'] == 'TimeoutError'
    assert 'unsafe-payload-credential' not in caplog.text
    assert bot.journal.get(plan.attempt_id).state is PairExecutionState.CLOSED
    bot.spot.place_ioc_limit_order.assert_not_called()
    bot.perp.place_ioc_limit_order.assert_not_called()


def snapshot(market, seconds, amounts):
    at = AT + timedelta(seconds=seconds)
    return BalanceSnapshot(market, at, at, tuple(AssetBalance(k, D(v)) for k, v in amounts.items()))


@pytest.fixture
def bot(tmp_path, monkeypatch):
    settings = load_settings(Path('docs/testnet-funding.toml'))
    settings = settings.model_copy(update={'testnet': settings.testnet.model_copy(update={
        'balance_evidence_enabled': True, 'balance_evidence_directory': tmp_path / 'frames'})})
    instance, transport = demo_bot(settings, tmp_path)
    before = (snapshot(ExecutionMarket.SPOT, 0, {'USDT': '1000', 'BTC': '1', 'ETH': '1'}),
              snapshot(ExecutionMarket.USDTM, 0, {'USDT': '1000'}))
    monkeypatch.setattr(balance_frame_store, 'capture_balance_snapshot',
                        lambda c, **kw: before[0 if c is instance.spot else 1])
    return instance, transport, before


@pytest.fixture
def closed(bot, monkeypatch, request):
    instance, transport, before = bot
    parameter = getattr(request, 'param', 'ETHUSDT')
    symbol = parameter[0] if isinstance(parameter, tuple) else parameter
    quantity = D('.001') if symbol == 'BTCUSDT' else D('.01')
    plan = instance._plan(symbol, quantity, 'native-' + symbol)
    if isinstance(parameter, tuple):
        plan = replace(plan, funding_settlement_at=AT + timedelta(seconds=200),
                       scheduled_exit_at=AT + timedelta(seconds=300), opportunity_id='synthetic-opportunity')
    assert instance.coordinator.open_pair(plan).state is PairExecutionState.HEDGED
    assert instance.coordinator.close_pair(plan).state is PairExecutionState.CLOSED
    orders = (plan.spot_entry, plan.spot_exit, plan.perp_entry, plan.perp_exit)
    prices = tuple(map(D, ('50000','50002','50010','50005') if symbol == 'BTCUSDT'
                       else ('2000','2001','2010','2002')))
    receipts, trades = {}, {}
    for i, (order, price) in enumerate(zip(orders, prices, strict=True)):
        oid = str(i + 1)
        receipts[order.client_order_id] = OrderReceipt(order.market, symbol, oid, order.client_order_id,
            OrderState.FILLED, quantity, quantity, quantity * price, price,
            side=order.side, order_type='LIMIT', time_in_force='IOC', limit_price=order.limit_price,
            reduce_only=order.reduce_only if order.market is ExecutionMarket.USDTM else None,
            position_side='BOTH' if order.market is ExecutionMarket.USDTM else None)
        fee = price * quantity * (D('.001') if order.market is ExecutionMarket.SPOT else D('.0005'))
        pnl = None if order.market is ExecutionMarket.SPOT else (D('0') if i == 2 else
                                                                D('.005') if symbol == 'BTCUSDT' else D('.08'))
        trades[oid] = (AccountTrade(order.market, symbol, oid, oid, order.side, price, quantity,
            price * quantity, fee, 'USDT', AT + timedelta(seconds=1 if i in (0,2) else 300), pnl,
            '{"fixture":"official fill"}'),)
    for client in (instance.spot, instance.perp):
        monkeypatch.setattr(client, 'get_order', lambda s, cid, **kw: receipts[cid])
        monkeypatch.setattr(client, 'get_order_trades', Mock(side_effect=lambda s, oid: trades[oid]))
        monkeypatch.setattr(client, 'place_ioc_limit_order', Mock(side_effect=AssertionError('balance checks never order')))
    monkeypatch.setattr(instance.perp, 'get_funding_payments', Mock(return_value=(
        FundingPayment('posted-' + symbol, symbol, AT + timedelta(seconds=200), D('.01')),)))
    after = (snapshot(ExecutionMarket.SPOT, 400, {'USDT': '999.901998' if symbol == 'BTCUSDT' else '999.96999',
                                               'BTC': '1', 'ETH': '1'}),
             snapshot(ExecutionMarket.USDTM, 400, {'USDT': '999.9649925' if symbol == 'BTCUSDT' else '1000.06994'}))
    monkeypatch.setattr(balance_accounting, 'capture_balance_snapshot',
                        Mock(side_effect=lambda c, **kw: after[0 if c is instance.spot else 1]))
    instance.clock = lambda: AT + timedelta(seconds=400)
    monkeypatch.setattr(instance.research_runner, 'run_once', lambda: tuple(SimpleNamespace(
        symbol=s, status=CycleStatus.SAVED, opportunity_id=None, decision='REJECT', failure_reason=None)
        for s in instance.settings.symbols))
    return instance, transport, plan, before, after, trades


@pytest.mark.parametrize('closed', ['BTCUSDT', 'ETHUSDT'], indirect=True)
def test_native_run_once_archives_both_symbols_without_orders(closed):
    bot, _, plan, _, _, _ = closed
    before = source_rows(bot)
    cycles = bot.run_once()
    check = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    assert check.status is BalanceCheckStatus.ARCHIVED and check.balance_status == 'MATCHED'
    assert all(c.status == 'NO_ENTRY' and c.reason == 'REJECT' for c in cycles)
    audit = BalanceAuditRepository(bot.journal).get(check.audit_id)
    assert len(audit['trades']) == 4 and len(audit['funding']) == 1
    assert all(D(m['unexplained_delta']) == 0 for m in audit['movements'])
    bot.spot.place_ioc_limit_order.assert_not_called()
    bot.perp.place_ioc_limit_order.assert_not_called()
    assert source_rows(bot) == before


def test_marker_written_before_first_order_and_quote(bot):
    instance, transport, _ = bot
    plan = instance._plan('ETHUSDT', D('.01'), 'pre-plan')
    marker = BalanceCheckRepository(instance.journal).before_intents()[0]
    assert marker.attempt_id == plan.attempt_id and marker.quantity == D('.01')
    assert instance.journal.get(plan.attempt_id) is None and not transport.orders
    assert all(method == 'GET' for method, _ in transport.requests)
    assert (instance.balance_directory / 'pre-plan.before.json').is_file()


def test_capture_failure_blocks_before_quote_or_orders(bot, monkeypatch):
    instance, transport, _ = bot
    def fail(*a, **kw):
        raise TimeoutError()
    monkeypatch.setattr(balance_frame_store, 'capture_balance_snapshot', fail)
    with pytest.raises(ExecutionGuardError, match='BALANCE_PREENTRY_EVIDENCE_FAILED'):
        instance._plan('ETHUSDT', D('.01'), 'capture-failed')
    assert not transport.requests and not transport.orders
    assert not BalanceCheckRepository(instance.journal).before_intents()


def test_marker_commit_failure_blocks_entry(bot, monkeypatch):
    instance, transport, _ = bot
    monkeypatch.setattr(BalanceCheckRepository, 'register_before', lambda *a: False)
    with pytest.raises(ExecutionGuardError, match='MARKER_CONFLICT'):
        instance._plan('ETHUSDT', D('.01'), 'marker-failed')
    assert not transport.requests and not transport.orders


def test_archived_is_not_requeried_even_after_restart(closed):
    bot, _, plan, _, _, _ = closed
    reconcile_balance_ledger(bot)
    count = bot.spot.get_order_trades.call_count + bot.perp.get_order_trades.call_count
    for _ in range(3):
        reconcile_balance_ledger(bot)
    assert bot.spot.get_order_trades.call_count + bot.perp.get_order_trades.call_count == count
    assert len(BalanceAuditRepository(bot.journal).list_for_attempt(plan.attempt_id)) == 1
    assert balance_entry_blocker(bot) is None


def test_pending_has_interval_and_bounded_durable_retry(closed, monkeypatch):
    bot, _, plan, _, _, _ = closed
    snapshots = Mock(side_effect=TimeoutError())
    monkeypatch.setattr(balance_accounting, 'capture_balance_snapshot', snapshots)
    repo = BalanceCheckRepository(bot.journal)
    now = bot.clock()
    reconcile_balance_ledger(bot)
    assert repo.get(plan.attempt_id).status is BalanceCheckStatus.PENDING
    reconcile_balance_ledger(bot)
    assert snapshots.call_count == 1
    for i in (1,2):
        bot.clock = lambda i=i: now + timedelta(seconds=i * 60)
        reconcile_balance_ledger(bot)
    assert repo.get(plan.attempt_id).status is BalanceCheckStatus.FAILED
    assert repo.get(plan.attempt_id).attempts == 3
    assert snapshots.call_count == 3
    bot.clock = lambda: now + timedelta(days=1)
    reconcile_balance_ledger(bot)
    assert snapshots.call_count == 3 and balance_entry_blocker(bot) == 'BALANCE_CHECK_UNRESOLVED'


def test_deleted_preentry_file_not_mistaken_for_legacy(closed):
    bot, _, plan, _, _, _ = closed
    (bot.balance_directory / (plan.attempt_id + '.before.json')).unlink()
    reconcile_balance_ledger(bot)
    row = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    assert row.status is BalanceCheckStatus.FAILED and row.failure_code == 'MISSING_FRAME'
    assert bot.spot.get_order_trades.call_count == bot.perp.get_order_trades.call_count == 0


def test_tampered_frame_hash_rejected(closed):
    bot, _, plan, _, _, _ = closed
    path = bot.balance_directory / (plan.attempt_id + '.before.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    data['snapshots'][0]['balances'][0]['amount'] = '999'
    path.write_text(json.dumps(data), encoding='utf-8')
    reconcile_balance_ledger(bot)
    check = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    assert check.status is BalanceCheckStatus.FAILED and check.failure_code == 'FRAME_HASH_MISMATCH'
    assert bot.spot.get_order_trades.call_count == 0


def test_missing_fill_retry_does_not_commit_partial_audit(closed):
    bot, _, plan, _, _, trades = closed
    original = trades['4']
    trades['4'] = ()
    reconcile_balance_ledger(bot)
    check = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    assert check.failure_code == 'INCOMPLETE_TRADE_EVIDENCE' and check.status is BalanceCheckStatus.PENDING
    assert not BalanceAuditRepository(bot.journal).list_for_attempt(plan.attempt_id)
    trades['4'] = original
    bot.clock = lambda: AT + timedelta(seconds=460)
    reconcile_balance_ledger(bot)
    assert BalanceCheckRepository(bot.journal).get(plan.attempt_id).status is BalanceCheckStatus.ARCHIVED


def test_invalid_trade_accounting_terminal_and_no_profit(closed):
    bot, _, plan, _, _, trades = closed
    trades['4'] = (replace(trades['4'][0], quote_quantity=trades['4'][0].quote_quantity + 1),)
    reconcile_balance_ledger(bot)
    check = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    assert check.status is BalanceCheckStatus.FAILED and check.failure_code == 'INVALID_TRADE_ACCOUNTING'
    assert not BalanceAuditRepository(bot.journal).list_for_attempt(plan.attempt_id)


def test_committed_audit_recovers_checkpoint_after_crash_no_get(closed, monkeypatch):
    bot, _, plan, _, _, _ = closed
    original = BalanceCheckRepository.finish
    def crash(*a, **kw):
        raise OSError('simulated checkpoint commit crash')
    monkeypatch.setattr(BalanceCheckRepository, 'finish', crash)
    bot.run_once()
    assert len(BalanceAuditRepository(bot.journal).list_for_attempt(plan.attempt_id)) == 1
    count = bot.spot.get_order_trades.call_count + bot.perp.get_order_trades.call_count
    monkeypatch.setattr(BalanceCheckRepository, 'finish', original)
    (bot.balance_directory / (plan.attempt_id + '.before.json')).unlink()
    reconcile_balance_ledger(bot)
    assert BalanceCheckRepository(bot.journal).get(plan.attempt_id).status is BalanceCheckStatus.ARCHIVED
    assert bot.spot.get_order_trades.call_count + bot.perp.get_order_trades.call_count == count


def test_balance_mismatch_freezes_new_entry_but_does_not_reject_decision(closed, monkeypatch):
    bot, transport, plan, _, after, _ = closed
    changed = replace(after[1], balances=(AssetBalance('USDT', D('999')),))
    monkeypatch.setattr(balance_accounting, 'capture_balance_snapshot',
                        lambda c, **kw: after[0] if c is bot.spot else changed)
    cycles = bot.run_once()
    assert BalanceCheckRepository(bot.journal).get(plan.attempt_id).balance_status == 'MISMATCH'
    assert all(c.reason == 'REJECT' for c in cycles)
    with pytest.raises(ExecutionGuardError, match='BALANCE_AUDIT_UNRESOLVED'):
        bot._plan('BTCUSDT', D('.001'), 'another-entry')
    assert len(transport.orders) == 4


def test_no_funding_proof_remains_null_known_balance_delta_can_match(closed, monkeypatch):
    bot, _, plan, _, after, _ = closed
    monkeypatch.setattr(bot.perp, 'get_funding_payments', Mock(return_value=()))
    changed = replace(after[1], balances=(AssetBalance('USDT', D('1000.05994')),))
    monkeypatch.setattr(balance_accounting, 'capture_balance_snapshot',
                        lambda c, **kw: after[0] if c is bot.spot else changed)
    reconcile_balance_ledger(bot)
    check = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    assert check.status is BalanceCheckStatus.ARCHIVED and check.balance_status == 'INCOMPLETE'
    assert balance_entry_blocker(bot) is None
    audit = BalanceAuditRepository(bot.journal).get(check.audit_id)
    assert not audit['funding_evidence_complete']
    assert audit['issues'] == [{'code': 'FUNDING_EVIDENCE_INCOMPLETE', 'market': None, 'asset': None}]


def test_active_position_precedes_all_accounting_and_blocks_overlap(bot):
    instance, _, _ = bot
    plan = instance._plan('ETHUSDT', D('.01'), 'open-one')
    assert instance.coordinator.open_pair(plan).state is PairExecutionState.HEDGED
    reconcile_balance_ledger(instance)
    assert BalanceCheckRepository(instance.journal).get(plan.attempt_id) is None
    with pytest.raises(ExecutionGuardError):
        instance._plan('BTCUSDT', D('.001'), 'overlap')


def test_graceful_stop_performs_no_optional_accounting(closed):
    bot, _, plan, _, _, _ = closed
    bot.stop.set()
    reconcile_balance_ledger(bot)
    assert BalanceCheckRepository(bot.journal).get(plan.attempt_id) is None
    assert bot.spot.get_order_trades.call_count == 0


def test_legacy_closed_without_marker_not_recreated(closed):
    bot, _, plan, _, _, _ = closed
    with bot.journal._connect() as db:
        db.execute('DELETE FROM testnet_balance_frame_intents')
    reconcile_balance_ledger(bot)
    assert BalanceCheckRepository(bot.journal).get(plan.attempt_id) is None
    assert bot.spot.get_order_trades.call_count == 0


@pytest.mark.parametrize('closed', [('ETHUSDT', 'funding')], indirect=True)
def test_scheduled_settlement_requires_actual_funding_evidence(closed, monkeypatch):
    bot, _, plan, _, _, _ = closed
    monkeypatch.setattr(bot.perp, 'get_funding_payments', Mock(return_value=()))
    reconcile_balance_ledger(bot)
    check = BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    assert check.status is BalanceCheckStatus.PENDING and check.failure_code == 'FUNDING_EVIDENCE_PENDING'
    assert not BalanceAuditRepository(bot.journal).list_for_attempt(plan.attempt_id)
    monkeypatch.setattr(bot.perp, 'get_funding_payments', Mock(return_value=(
        FundingPayment('actually-posted', plan.symbol, AT + timedelta(seconds=200), D('.01')),)))
    bot.clock = lambda: AT + timedelta(seconds=460)
    reconcile_balance_ledger(bot)
    assert BalanceCheckRepository(bot.journal).get(plan.attempt_id).balance_status == 'MATCHED'


def test_pending_exit_skips_balance_stage_and_research(bot, monkeypatch):
    instance, transport, _ = bot
    plan = instance._plan('ETHUSDT', D('.01'), 'due-before-ledger')
    assert instance.coordinator.open_pair(plan).state is PairExecutionState.HEDGED
    opened = instance.journal.plans()[0][1]
    instance.clock = lambda: opened + timedelta(seconds=65)
    monkeypatch.setattr(instance, '_close', Mock(side_effect=TimeoutError()))
    ledger, research = Mock(), Mock()
    monkeypatch.setattr('crypto_arbitrage_bot.execution.runner.reconcile_balance_ledger', ledger)
    monkeypatch.setattr(instance.research_runner, 'run_once', research)
    cycles = instance.run_once()
    assert all(c.status == 'EXIT_PRIORITY' for c in cycles)
    ledger.assert_not_called()
    research.assert_not_called()
    assert len(transport.orders) == 2


def test_optional_ledger_exception_does_not_change_closed_state_or_stop_research(closed, monkeypatch):
    bot, _, plan, _, _, _ = closed
    original = source_rows(bot)
    monkeypatch.setattr('crypto_arbitrage_bot.execution.runner.reconcile_balance_ledger', Mock(side_effect=OSError()))
    cycles = bot.run_once()
    assert all(c.status == 'NO_ENTRY' for c in cycles)
    assert source_rows(bot) == original and bot.journal.get(plan.attempt_id).state is PairExecutionState.CLOSED
