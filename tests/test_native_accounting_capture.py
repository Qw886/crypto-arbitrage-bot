"""Offline wallet archives must be captured before the observed run finishes."""
import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from test_witness_acceptance import local_clock_fixture as local_clock_fixture
from test_witness_acceptance import offline as offline
from test_witness_acceptance import registration, run_fixture

from crypto_arbitrage_bot.execution import main
from crypto_arbitrage_bot.execution.acceptance_proof import (
    ProofExportFailure,
    export_acceptance_proof,
)
from crypto_arbitrage_bot.execution.balance_audit_repository import BalanceAuditRepository
from crypto_arbitrage_bot.execution.balance_check_repository import (
    BalanceCheckRepository,
    BalanceCheckStatus,
)
from crypto_arbitrage_bot.execution.demo import DemoResponse
from crypto_arbitrage_bot.execution.funding_acceptance import FundingAcceptance
from crypto_arbitrage_bot.execution.risk_witness import read_witness_events
from crypto_arbitrage_bot.execution.types import AccountTrade, ExecutionMarket


def enable_balance_fixture(bot,transport,state,monkeypatch):
    bot.config = bot.config.model_copy(update={'balance_evidence_enabled':True})
    bot.settings = bot.settings.model_copy(update={'testnet':bot.config})
    original = transport.open
    posted_at = {}

    def wallet_response(request,**kwargs):
        if request.get_method()=='POST' and urlsplit(request.full_url).path.endswith('/order'):
            state['now'] += timedelta(milliseconds=1)
        response = original(request,**kwargs)
        if request.get_method()=='POST' and urlsplit(request.full_url).path.endswith('/order'):
            posted_at[str(json.loads(response.body)['orderId'])]=state['now']
        if '/account?' in request.full_url:
            payload = json.loads(response.body)
            if 'balances' in payload:
                for row in payload['balances']:
                    row['locked']='0'
            else:
                payload['assets']=[{'asset':'USDT','walletBalance':'10000'}]
            return DemoResponse(payload)
        return response

    monkeypatch.setattr(transport,'open',wallet_response)

    def trades(client,symbol,order_id):
        row = next(row for row in transport.orders.values() if str(row['orderId'])==order_id)
        plan, = [plan for plan,_ in bot.journal.plans()]
        intent = next(order for order in bot.journal.intents(plan.attempt_id)
                      if order.client_order_id==row['clientOrderId'])
        price,quantity = Decimal(row['avgPrice']),Decimal(row['executedQty'])
        pnl = None
        if client.market is ExecutionMarket.USDTM:
            entry = transport.orders[plan.perp_entry.client_order_id]
            pnl = Decimal('0') if intent==plan.perp_entry else (Decimal(entry['avgPrice'])-price)*quantity
        return (AccountTrade(client.market,symbol,str(int(row['orderId'])*10),order_id,intent.side,
            price,quantity,Decimal(row['cummulativeQuoteQty']),Decimal('0'),'USDT',posted_at[order_id],pnl,'{}'),)

    for client in (bot.spot,bot.perp):
        monkeypatch.setattr(client,'get_order_trades',lambda symbol,order_id,current=client:trades(current,symbol,order_id))


def test_captured_accounting_archives_wallets_before_session_end(live_clock_bot,tmp_path,monkeypatch):
    bot,transport,state,acceptance = run_fixture(live_clock_bot,tmp_path,monkeypatch)
    enable_balance_fixture(bot,transport,state,monkeypatch)
    status = acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True)
    assert status=='VERIFIED', (acceptance.state.get('reason'),acceptance.state.get('failure_stage'))
    outcome = acceptance.state['captured_accounting']
    assert outcome['status']=='ARCHIVED', outcome.get('error_code')
    checkpoint = BalanceCheckRepository(bot.journal).get(str(acceptance.state['attempt_id']))
    assert checkpoint.status is BalanceCheckStatus.ARCHIVED
    archive = BalanceAuditRepository(bot.journal).get(outcome['audit_id'])
    assert archive is not None
    events = read_witness_events(registration(acceptance))
    ended = datetime.fromisoformat(events[-1]['at'])
    assert all(datetime.fromisoformat(wallet['received_at'])<=ended for wallet in archive['after'])
    assert len(transport.orders)==4
    assert acceptance.state['witness_origin']=='OFFLINE_FIXTURE'
    exported = export_acceptance_proof(bot.settings,acceptance.report_path,tmp_path/'proof.json',clock=bot.clock)
    assert not exported.success and exported.failure is ProofExportFailure.NON_NATIVE_CAPTURE
    assert not (tmp_path/'proof.json').exists()


@pytest.mark.parametrize('capture',[True,False])
def test_terminal_report_does_not_recapture_or_resubmit(live_clock_bot,tmp_path,monkeypatch,capture):
    bot,transport,state,acceptance = run_fixture(live_clock_bot,tmp_path,monkeypatch)
    enable_balance_fixture(bot,transport,state,monkeypatch)
    assert acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=capture)=='VERIFIED'
    old_report = acceptance.report_path.read_bytes()
    calls = len(transport.requests)
    assert acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True)=='VERIFIED'
    assert acceptance.report_path.read_bytes()==old_report and len(transport.requests)==calls


@pytest.mark.parametrize('failure',['disabled_balance','missing_witness','not_bool'])
def test_invalid_capture_policy_rejects_before_any_gateway_call(live_clock_bot,tmp_path,failure):
    bot,transport,state = live_clock_bot
    acceptance = FundingAcceptance(bot,tmp_path/'report.json')
    directory = tmp_path/'witness'
    flag = True
    if failure!='disabled_balance':
        bot.config = bot.config.model_copy(update={'balance_evidence_enabled':True})
    if failure=='missing_witness':
        directory = None
    if failure=='not_bool':
        flag = 1
    calls = len(transport.requests)
    with pytest.raises(ValueError):
        acceptance.run(witness_directory=directory,capture_accounting=flag)
    assert len(transport.requests)==calls and not acceptance.report_path.exists()


@pytest.mark.parametrize('arguments',[
    ['--check','--capture-accounting'],
    ['--funding-acceptance','--capture-accounting'],
    ['--funding-acceptance','--witness-directory','fixture','--capture-accounting','--account-after-acceptance'],
])
def test_cli_capture_mode_requires_explicit_compatible_options(monkeypatch,arguments):
    monkeypatch.setattr('sys.argv',['bot',*arguments])
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code==2


def test_cli_captured_wallet_mismatch_keeps_funding_and_refuses_success(live_clock_bot,tmp_path,monkeypatch,capsys):
    bot,transport,state,acceptance = run_fixture(live_clock_bot,tmp_path,monkeypatch)
    enable_balance_fixture(bot,transport,state,monkeypatch)
    config = tmp_path/'explicit.toml'
    config.write_text(Path('docs/testnet-funding-acceptance.toml').read_text()+'\nbalance_evidence_enabled = true\n')
    def wait(interval):
        status = json.loads(acceptance.report_path.read_text())['status']
        state['now'] = state['settlement']+timedelta(seconds=-180 if status=='WAITING' else 61)
        return False
    monkeypatch.setattr(bot.stop,'wait',wait)
    monkeypatch.setattr('sys.argv',['bot','--config',str(config),'--funding-acceptance',
        '--acceptance-report',str(acceptance.report_path),'--witness-directory',str(tmp_path/'witness'),
        '--capture-accounting'])
    monkeypatch.setenv('CAB_TESTNET_ORDERS_ENABLED','true')
    monkeypatch.setattr(main,'execution_environment',lambda *a:{})
    monkeypatch.setattr(main,'create_testnet_bot',lambda *a,**kw:bot)
    monkeypatch.setattr(main,'configure_logging',lambda *a:None)
    assert main.main()==1
    report = json.loads(acceptance.report_path.read_text())
    assert report['status']=='VERIFIED' and report['captured_accounting']['balance_status']=='MISMATCH'
    assert 'CAPTURED_ACCOUNTING' in capsys.readouterr().out
    assert report['witness_origin']=='OFFLINE_FIXTURE' and len(transport.orders)==4


def test_wallet_failure_preserves_funding_and_does_not_retry_terminal_capture(live_clock_bot,tmp_path,monkeypatch):
    bot,transport,state,acceptance = run_fixture(live_clock_bot,tmp_path,monkeypatch)
    enable_balance_fixture(bot,transport,state,monkeypatch)
    original = transport.open
    def unavailable_wallet(request,**kwargs):
        response = original(request,**kwargs)
        if '/account?' in request.full_url and len(transport.orders)==4:
            payload = json.loads(response.body)
            if 'balances' in payload:
                for row in payload['balances']:
                    row.pop('locked',None)
            return DemoResponse(payload)
        return response
    monkeypatch.setattr(transport,'open',unavailable_wallet)
    assert acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True)=='VERIFIED'
    assert acceptance.state['captured_accounting']['status']=='FAILED'
    assert acceptance.state['actual_funding'] is not None and len(transport.orders)==4
    report = acceptance.report_path.read_bytes()
    count = len(transport.requests)
    assert acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True)=='VERIFIED'
    assert acceptance.report_path.read_bytes()==report and len(transport.requests)==count


@pytest.mark.parametrize('boundary',['stop','deadline'])
def test_stopped_or_expired_capture_skips_wallet_requests_after_funding(live_clock_bot,tmp_path,monkeypatch,boundary):
    bot,transport,state,acceptance = run_fixture(live_clock_bot,tmp_path,monkeypatch)
    enable_balance_fixture(bot,transport,state,monkeypatch)
    timer = [0]
    monkeypatch.setattr('crypto_arbitrage_bot.execution.funding_acceptance.monotonic_ns',lambda:timer[0])
    original = bot.perp.get_account_state
    def completed_account(symbol):
        result = original(symbol)
        if len(transport.orders)==4:
            if boundary=='stop':
                bot.stop.set()
            else:
                timer[0]=10**18
        return result
    monkeypatch.setattr(bot.perp,'get_account_state',completed_account)
    assert acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True)=='VERIFIED'
    assert acceptance.state['captured_accounting']['status']=='SKIPPED'
    assert 'audit_id' not in acceptance.state['captured_accounting']
    assert len(transport.orders)==4


def test_exhausted_original_attempt_budget_blocks_new_capture(live_clock_bot,tmp_path,monkeypatch):
    bot,transport,state,acceptance = run_fixture(live_clock_bot,tmp_path,monkeypatch)
    enable_balance_fixture(bot,transport,state,monkeypatch)
    bot.config = bot.config.model_copy(update={'max_total_attempts':1})
    bot.settings = bot.settings.model_copy(update={'testnet':bot.config})
    assert acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True)=='VERIFIED'
    second = FundingAcceptance(bot,tmp_path/'second.json')
    assert second.run(witness_directory=tmp_path/'second-witness',capture_accounting=True)=='ABORTED'
    assert len(transport.orders)==4 and not (tmp_path/'second-witness').exists()


def test_disabled_balance_cli_rejects_before_reading_credentials(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr('sys.argv',['bot','--config','docs/testnet-funding-acceptance.toml',
        '--funding-acceptance','--witness-directory',str(tmp_path/'witness'),'--capture-accounting'])
    def forbidden(*args):
        pytest.fail('must reject before credential storage')
    monkeypatch.setattr(main,'execution_environment',forbidden)
    assert main.main()==1
    assert 'CAPTURE_ACCOUNTING_REQUIRES_PREENTRY_BALANCE_EVIDENCE' in capsys.readouterr().out
