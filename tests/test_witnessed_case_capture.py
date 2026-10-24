"""Preregister actual observed branches; offline fixtures cannot gain qualification."""

import json
from datetime import datetime
from decimal import Decimal
from urllib.parse import parse_qs, urlsplit

import pytest
from test_native_accounting_capture import enable_balance_fixture
from test_witness_acceptance import local_clock_fixture as local_clock_fixture
from test_witness_acceptance import offline as offline
from test_witness_acceptance import registration, run_fixture

from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
from crypto_arbitrage_bot.execution import main
from crypto_arbitrage_bot.execution.balance_audit_repository import BalanceAuditRepository
from crypto_arbitrage_bot.execution.balance_check_repository import BalanceCheckRepository
from crypto_arbitrage_bot.execution.captured_case_proof import export_captured_case_proof
from crypto_arbitrage_bot.execution.demo import DemoResponse
from crypto_arbitrage_bot.execution.reconciliation_repository import ReconciliationRepository
from crypto_arbitrage_bot.execution.risk_witness import read_witness_events
from crypto_arbitrage_bot.execution.witness_acceptance import observe_acceptance
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase as Case


@pytest.mark.parametrize('case',list(Case),ids=lambda case:case.name.lower())
def test_case_is_fixed_before_entry_without_claiming_observed_branch(live_clock_bot,tmp_path,monkeypatch,case):
    bot,transport,state,acceptance = run_fixture(live_clock_bot,tmp_path,monkeypatch)
    enable_balance_fixture(bot,transport,state,monkeypatch)
    assert acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True,witness_case=case)=='VERIFIED'
    assert registration(acceptance).case is case
    assert acceptance.state['capture_case']==case.value
    assert acceptance.state['witness_origin']=='OFFLINE_FIXTURE' and len(transport.orders)==4
    ref = StudyRegistrationSettings(**acceptance.state['witness_trace'])
    result = export_captured_case_proof(bot.settings,registration(acceptance).reference,ref,
                                      tmp_path/'proof.json',clock=bot.clock)
    assert not result.success and not (tmp_path/'proof.json').exists()
    before,calls = acceptance.report_path.read_bytes(),len(transport.requests)
    assert acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True)=='VERIFIED'
    assert acceptance.report_path.read_bytes()==before and len(transport.requests)==calls


def test_actual_partial_abort_captures_wallet_before_witness_ends(live_clock_bot,tmp_path,monkeypatch):
    bot,transport,state,acceptance = run_fixture(live_clock_bot,tmp_path,monkeypatch)
    enable_balance_fixture(bot,transport,state,monkeypatch)
    opener = transport.open

    def partial(request,**kwargs):
        response = opener(request,**kwargs)
        params = parse_qs((request.data or b'').decode())
        spot = urlsplit(request.full_url).hostname=='testnet.binance.vision'
        if spot and request.get_method()=='POST' and params.get('side')==['BUY']:
            payload = json.loads(response.body)
            original,done = Decimal(payload['origQty']),Decimal('.004')
            price = Decimal(payload['avgPrice'])
            payload.update(status='EXPIRED',executedQty=str(done),cummulativeQuoteQty=str(done*price))
            payload['fills'][0]['qty']=str(done)
            transport.orders[payload['clientOrderId']]=payload
            transport.base['ETHUSDT']-=original-done
            return DemoResponse(payload)
        return response

    monkeypatch.setattr(transport,'open',partial)
    assert acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True)=='ABORTED'
    outcome = acceptance.state['captured_accounting']
    assert outcome['status']=='ARCHIVED'
    archived = BalanceAuditRepository(bot.journal).get(outcome['audit_id'])
    checkpoint = BalanceCheckRepository(bot.journal).get(str(acceptance.state['attempt_id']))
    assert checkpoint.audit_id==outcome['audit_id']
    ended = datetime.fromisoformat(read_witness_events(registration(acceptance))[-1]['at'])
    assert all(datetime.fromisoformat(frame['received_at'])<=ended for frame in archived['after'])
    report = ReconciliationRepository(bot.journal).get_report(str(acceptance.state['attempt_id']))
    assert report['calculation'] is report['net_return'] is None
    assert len(transport.orders)==2 and bot.perp.get_funding_payments.call_count==0
    old_bytes,calls = acceptance.report_path.read_bytes(),len(transport.requests)
    assert acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True)=='ABORTED'
    assert acceptance.report_path.read_bytes()==old_bytes and len(transport.requests)==calls


@pytest.mark.parametrize('flags',[
    ['--check','--capture-case','NORMAL_ENTRY_EXIT'],
    ['--funding-acceptance','--witness-directory','fixture','--capture-case','PARTIAL_ENTRY_RECOVERY'],
])
def test_cli_case_requires_full_accounting_capture_before_credentials(monkeypatch,flags):
    monkeypatch.setattr('sys.argv',['bot',*flags])
    with pytest.raises(SystemExit) as result:
        main.main()
    assert result.value.code==2


def test_terminal_registration_cannot_be_relabelled_or_recaptured(live_clock_bot,tmp_path,monkeypatch):
    bot,transport,state,acceptance = run_fixture(live_clock_bot,tmp_path,monkeypatch)
    enable_balance_fixture(bot,transport,state,monkeypatch)
    assert acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True,
                          witness_case=Case.NORMAL_ENTRY_EXIT)=='VERIFIED'
    before,calls = acceptance.report_path.read_bytes(),len(transport.requests)
    with pytest.raises(ValueError,match='case cannot change'):
        acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True,
                       witness_case=Case.PARTIAL_ENTRY_RECOVERY)
    assert len(transport.requests)==calls and acceptance.report_path.read_bytes()==before
    assert registration(acceptance).case is Case.NORMAL_ENTRY_EXIT


def test_changed_case_before_claim_blocks_entry_and_preserves_registration(live_clock_bot,tmp_path,monkeypatch):
    bot,transport,state,acceptance = run_fixture(live_clock_bot,tmp_path,monkeypatch)
    enable_balance_fixture(bot,transport,state,monkeypatch)
    with observe_acceptance(acceptance,tmp_path/'witness',case=Case.NORMAL_ENTRY_EXIT):
        original = registration(acceptance)
    before,calls = original.reference.path.read_bytes(),len(transport.requests)
    assert acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True,
                          witness_case=Case.PARTIAL_ENTRY_RECOVERY)=='ABORTED'
    assert acceptance.state['witness_status']=='FAILED'
    assert not transport.orders and original.reference.path.read_bytes()==before
    assert len(transport.requests)==calls


def test_case_conflict_during_hold_does_not_block_original_reducing_exit(live_clock_bot,tmp_path,monkeypatch):
    bot,transport,state,acceptance = run_fixture(live_clock_bot,tmp_path,monkeypatch,stop_after_entry=True)
    enable_balance_fixture(bot,transport,state,monkeypatch)
    assert acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True,
                          witness_case=Case.NORMAL_ENTRY_EXIT)=='HOLDING'
    original = registration(acceptance)
    before = original.reference.path.read_bytes()
    assert len(transport.orders)==2
    bot.stop.clear()
    from datetime import timedelta
    state['now']=state['settlement']+timedelta(seconds=61)
    assert acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True,
                          witness_case=Case.PARTIAL_ENTRY_RECOVERY)=='VERIFIED'
    assert bot.journal.get(str(acceptance.state['attempt_id'])).state.value=='CLOSED'
    assert len(transport.orders)==4 and acceptance.state['witness_status']=='FAILED'
    assert original.reference.path.read_bytes()==before
    assert 'witness_trace' not in acceptance.state


@pytest.mark.parametrize('case,capture',[(Case.NORMAL_ENTRY_EXIT,False),('NORMAL_ENTRY_EXIT',True),(True,True)],
                         ids=['requires-wallet','no-string-alias','no-bool'])
def test_invalid_case_rejects_before_any_gateway_call(live_clock_bot,tmp_path,monkeypatch,case,capture):
    bot,transport,state,acceptance = run_fixture(live_clock_bot,tmp_path,monkeypatch)
    enable_balance_fixture(bot,transport,state,monkeypatch)
    calls = len(transport.requests)
    with pytest.raises(ValueError):
        acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=capture,witness_case=case)
    assert len(transport.requests)==calls and not acceptance.report_path.exists()


def test_unknown_entry_never_captures_wallet_or_claims_flatness(live_clock_bot,tmp_path,monkeypatch):
    bot,transport,state,acceptance = run_fixture(live_clock_bot,tmp_path,monkeypatch)
    enable_balance_fixture(bot,transport,state,monkeypatch)
    original = transport.open

    def lose_response(request,**kwargs):
        params = parse_qs((request.data or b'').decode())
        spot = urlsplit(request.full_url).hostname=='testnet.binance.vision'
        if spot and urlsplit(request.full_url).path.endswith('/order'):
            if request.get_method()=='POST' and params.get('side')==['BUY']:
                original(request,**kwargs)  # External offline exchange accepted it.
                raise TimeoutError('explicit offline loss of receipt')
            if request.get_method()=='GET':
                raise TimeoutError('explicit offline unresolved fixed-ID query')
        return original(request,**kwargs)

    monkeypatch.setattr(transport,'open',lose_response)
    assert acceptance.run(witness_directory=tmp_path/'witness',capture_accounting=True,
                          witness_case=Case.SPOT_ACK_RECOVERY)=='NEEDS_ATTENTION'
    attempt = str(acceptance.state['attempt_id'])
    assert bot.journal.get(attempt).state.value=='RECOVERY_REQUIRED'
    assert 'captured_accounting' not in acceptance.state
    assert BalanceCheckRepository(bot.journal).get(attempt) is None
    assert len(transport.orders)==1 and transport.base['ETHUSDT']>0


def test_cli_passes_preregistered_case_to_real_offline_capture(live_clock_bot,tmp_path,monkeypatch,capsys):
    from pathlib import Path

    bot,transport,state,acceptance = run_fixture(live_clock_bot,tmp_path,monkeypatch)
    enable_balance_fixture(bot,transport,state,monkeypatch)
    config = tmp_path/'explicit.toml'
    config.write_text(Path('docs/testnet-funding-acceptance.toml').read_text(encoding='utf-8')+
                      '\nbalance_evidence_enabled = true\n',encoding='utf-8')
    monkeypatch.setattr('sys.argv',['bot','--config',str(config),'--funding-acceptance',
        '--acceptance-report',str(acceptance.report_path),'--witness-directory',str(tmp_path/'witness'),
        '--capture-accounting','--capture-case','NORMAL_ENTRY_EXIT'])
    monkeypatch.setenv('CAB_TESTNET_ORDERS_ENABLED','true')
    monkeypatch.setattr(main,'execution_environment',lambda *args:{})
    monkeypatch.setattr(main,'create_testnet_bot',lambda *args,**kwargs:bot)
    monkeypatch.setattr(main,'configure_logging',lambda *args:None)
    def wait(interval):
        from datetime import timedelta
        status = json.loads(acceptance.report_path.read_bytes())['status']
        state['now']=state['settlement']+timedelta(seconds=-180 if status=='WAITING' else 61)
        return False
    monkeypatch.setattr(bot.stop,'wait',wait)
    assert main.main()==1  # This offline wallet fixture has MISMATCH, never qualification.
    report = json.loads(acceptance.report_path.read_bytes())
    assert report['capture_case']=='NORMAL_ENTRY_EXIT' and report['witness_origin']=='OFFLINE_FIXTURE'
    assert report['witness_status']=='EXPORTED' and len(transport.orders)==4
    assert 'CAPTURED_ACCOUNTING' in capsys.readouterr().out
