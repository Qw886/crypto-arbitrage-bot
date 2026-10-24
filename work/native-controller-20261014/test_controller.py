"""Controller boundaries use explicit offline gateways and disposable journals."""
import hashlib
from datetime import timedelta
from pathlib import Path

import native_case_control as control
import pytest

from crypto_arbitrage_bot.config.research import StudyRegistrationSettings as Reference
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.demo import demo_bot
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot
from crypto_arbitrage_bot.utils.clock import utc_now


def offline_approval(ref,**updates):
    body = {'schema':'native-case-current-authorization-v1','controller_sha256':ref.sha256,
            'scope':'TESTNET_NATIVE_CASES_ONLY','native_case_slots':14,'normal_trading_allowed':False,
            'current_user_confirmed':True,'authorization_record':'OFFLINE_FIXTURE','approved_at':utc_now()}
    body.update(updates)
    path = ref.path.parent/'offline-approval.json'
    control.protocol.exclusive(path,body)
    return Reference(path=path,sha256=control.protocol.digest(path))


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    monkeypatch.setattr('socket.create_connection',lambda *args:pytest.fail('offline controller'))
    settings = load_settings(Path('docs/testnet-funding-acceptance.toml'))
    settings = settings.model_copy(update={'testnet':settings.testnet.model_copy(update={'max_total_attempts':1})})
    bot,transport = demo_bot(settings,tmp_path/'original')
    assert bot.round_trip('ETHUSDT',settings.research.sizing['ETHUSDT'].target_quantity).state.value=='CLOSED'
    ref = control.prepare(bot.settings,tmp_path/'controller',workspace=tmp_path)
    return bot,transport,ref


def test_missing_current_approval_refuses_before_claim_or_credential_factory(prepared):
    bot,transport,ref = prepared
    before = hashlib.sha256(bot.config.journal_path.read_bytes()).hexdigest()
    calls = len(transport.requests)
    def forbidden(*args):
        pytest.fail('credential factory must stay behind approval')
    with pytest.raises(control.ControlError,match='CURRENT_EXPLICIT_AUTHORIZATION_REQUIRED'):
        control.run_slot(ref,0,approval=None,bot_factory=forbidden)
    assert control.status(ref)['claims_consumed']==0
    assert hashlib.sha256(bot.config.journal_path.read_bytes()).hexdigest()==before
    assert len(transport.requests)==calls


@pytest.mark.parametrize('changed',[
    {'controller_sha256':'0'*64},{'scope':'NORMAL_TRADING'},{'native_case_slots':15},
    {'normal_trading_allowed':True},{'current_user_confirmed':False},
    {'approved_at':utc_now()+timedelta(days=8)},
],ids=['different-plan','wrong-scope','extra-slot','normal-permission','no-current-user','outside-window'])
def test_wrong_approval_never_calls_factory_or_consumes_slot(prepared,changed):
    _,_,ref = prepared
    approved = offline_approval(ref,**changed)
    def forbidden(*args):
        pytest.fail('must remain before credentials')
    with pytest.raises(control.ControlError):
        control.run_slot(ref,0,approval=approved,bot_factory=forbidden)
    assert control.status(ref)['claims_consumed']==0


def test_stop_before_start_leaves_slot_available_and_no_factory(prepared):
    _,_,ref = prepared
    approved = offline_approval(ref)
    assert control.stop(ref)['workers_killed'] is False
    with pytest.raises(control.ControlError,match='STOP_REQUESTED'):
        control.run_offline_slot(ref,0,approval=approved,bot_factory=lambda *args:pytest.fail('no credentials'),
                                 clock=utc_now,wait=lambda seconds:None)
    assert control.status(ref)['claims_consumed']==0


def test_half_written_claim_is_not_refunded_or_recreated(prepared):
    _,_,ref = prepared
    approved = offline_approval(ref)
    claim = ref.path.parent/'claims/00.json'
    claim.write_bytes(b'{half')
    with pytest.raises(control.ControlError,match='PARTIAL_OR_INVALID'):
        control.run_offline_slot(ref,0,approval=approved,bot_factory=lambda *args:pytest.fail('no credentials'),
                                 clock=utc_now,wait=lambda seconds:None)
    assert claim.read_bytes()==b'{half' and control.status(ref)['claims_consumed']==1


def test_stop_during_wait_preserves_claim_and_finished_resume_has_no_io(prepared):
    old,transport,ref = prepared
    approved = offline_approval(ref)
    old_orders = len(transport.orders)
    def factory(settings):
        return Bot(settings,old.spot,old.perp)
    result = control.run_offline_slot(ref,0,approval=approved,bot_factory=factory,clock=utc_now,
                                      wait=lambda seconds:control.stop(ref))
    assert result['status']=='CANCELLED' and result['origin']=='OFFLINE_FIXTURE'
    assert len(transport.orders)==old_orders and control.status(ref)['claims_consumed']==1
    calls = len(transport.requests)
    repeated = control.run_offline_slot(ref,0,approval=approved,
        bot_factory=lambda *args:pytest.fail('finished state must be read-only'),clock=utc_now,wait=lambda seconds:None)
    assert repeated['status']=='CANCELLED' and len(transport.requests)==calls


def test_full_offline_capture_reuses_original_trades_and_rejects_native_proof(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path.cwd()/'tests'))
    import json
    from decimal import Decimal

    from test_funding_lifecycle import live_clock_bot

    from crypto_arbitrage_bot.execution.demo import DemoResponse
    from crypto_arbitrage_bot.execution.types import FundingPayment

    bot,transport,state = live_clock_bot.__wrapped__(tmp_path,monkeypatch)
    bot.config=bot.config.model_copy(update={'max_total_attempts':1})
    bot.settings=bot.settings.model_copy(update={'testnet':bot.config})
    assert bot.round_trip('ETHUSDT',Decimal('.01')).state.value=='CLOSED'
    ref = control.prepare(bot.settings,tmp_path/'controller',workspace=tmp_path)
    state['now']=utc_now()
    state['settlement']=state['now']+timedelta(seconds=300)
    approved = offline_approval(ref,approved_at=state['now'])
    original = transport.open
    def account_rows(request,**kwargs):
        response = original(request,**kwargs)
        if '/account?' in request.full_url:
            payload = json.loads(response.body)
            if 'balances' in payload:
                for row in payload['balances']:
                    row['locked']='0'
            else:
                payload['assets']=[{'asset':'USDT','walletBalance':'10000'}]
            return DemoResponse(payload)
        return response
    monkeypatch.setattr(transport,'open',account_rows)
    monkeypatch.setattr(bot.perp,'get_funding_payments',lambda *args:(
        FundingPayment('fixture-event','ETHUSDT',state['settlement'],Decimal('.01')),))
    created = []
    def factory(settings):
        result = Bot(settings,bot.spot,bot.perp,clock=lambda:state['now'])
        result.collector.clock=result.research_runner.clock=lambda:state['now']
        created.append(result)
        return result
    def wait(seconds):
        if len(transport.orders)==4:
            state['now']=state['settlement']-timedelta(seconds=180)
        else:
            state['now']=state['settlement']+timedelta(seconds=61)
    result = control.run_offline_slot(ref,7,approval=approved,bot_factory=factory,
        clock=lambda:state['now'],wait=wait)
    assert len(transport.orders)==8 and result['status']=='NOT_PROVEN'
    assert created[0].journal.get(result['attempt_id']).state.value=='CLOSED'
    assert result['proof'] is None and not (ref.path.parent/'cases/07/proof.json').exists()
    assert control.status(ref)['claims_consumed']==1


def test_current_prototype_cannot_start_native_even_with_an_approval_file(prepared):
    _,_,ref = prepared
    approved = offline_approval(ref)
    with pytest.raises(control.ControlError,match='NATIVE_EXECUTION_NOT_REVIEWED'):
        control.run_slot(ref,0,approval=approved,bot_factory=lambda *args:pytest.fail('no live credentials'))
    assert control.status(ref)['claims_consumed']==0


def test_same_process_cannot_claim_independent_restart(prepared):
    from crypto_arbitrage_bot.utils.native_identity import current_native_identity
    _,_,ref = prepared
    folder = ref.path.parent/'cases/05'
    folder.mkdir()
    control.write_state(folder/'state.json',{'controller_sha256':ref.sha256,'slot':5,
        'status':'RESTART_PENDING_NEW_PROCESS','owner':current_native_identity().payload()})
    with pytest.raises(control.ControlError,match='PREVIOUS_OWNER_NOT_CONFIRMED_GONE'):
        control.replay_slot(ref,5)
