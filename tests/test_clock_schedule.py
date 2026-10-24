"""Finite independent diagnostic slots, fake public collector and OS service."""
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot import clock_health, monitor_archive
from crypto_arbitrage_bot import clock_schedule as module
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.utils.file_lock import LockBusyError, file_lock

AT=datetime(2026,10,12,tzinfo=UTC)
SOURCE='a'*64


@pytest.fixture
def context(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'work').mkdir()
    monitor=tmp_path/'work/monitor'
    monitor.mkdir()
    original={'ends_at':(AT+timedelta(hours=1)).isoformat()}
    monkeypatch.setattr(monitor_archive,'load_plan',lambda *a:(original,'b'*64))
    monkeypatch.setattr(monitor_archive,'load_binding',lambda *a:{})
    monkeypatch.setattr(module,'implementation_fingerprint',lambda:SOURCE)
    path=tmp_path/'work/clock'
    plan=module.prepare(path,monitor,'b'*64,clock=lambda:AT)
    return path,plan,monitor,original


def health(at=AT):
    values=tuple(clock_health.Sample(endpoint,at,at,0,at,200) for endpoint in
                 (PublicEndpoint.SPOT_TIME,PublicEndpoint.PERP_TIME))
    return clock_health.assess(values,evaluated_at=at)


def run(context,*,at=AT,collector=None,clock=None,source=SOURCE):
    path,_,_,_=context
    return module.run_slot(path,module._sha(path/'plan.json'),source,clock=clock or (lambda:at),
        collect=collector or (lambda timeout:health(at)),free_bytes=lambda p:module.MIN_FREE_BYTES)


def test_prepare_uses_original_deadline_and_fixed_diagnostic_budget_without_research_writes(context):
    path,plan,monitor,original=context
    assert plan['max_slots']==6 and plan['ends_at']==monitor_archive.at(original['ends_at'])
    assert plan['interval_seconds']==600 and plan['clock_modification_authorized'] is False
    assert list(monitor.iterdir())==[]
    assert module.load_plan(path)==json.loads((path/'plan.json').read_text(encoding='utf-8'))
    with pytest.raises(FileExistsError):
        module.prepare(path,monitor,'b'*64,clock=lambda:AT)


def test_claim_is_durable_before_public_io_and_same_slot_is_never_replayed(context):
    path,*_=context
    def collect(timeout):
        assert timeout==5 and monitor_archive.read(path/'claims/00000.json')['index']==0
        return health()
    collector=Mock(side_effect=collect)
    assert run(context,collector=collector)==0
    assert run(context,at=AT+timedelta(seconds=60),collector=collector)==0
    assert collector.call_count==1 and len(list((path/'claims').iterdir()))==1
    receipt=monitor_archive.read(path/'receipts/00000.json')
    assert receipt['report']['status']=='WITHIN_DIAGNOSTIC_BOUND'
    assert receipt['report']['market_quality_verified'] is False


@pytest.mark.parametrize('shift',[-1,120,599,3592,3600,3601])
def test_before_start_late_or_insufficient_remaining_window_does_no_io(context,shift):
    collect=Mock(side_effect=AssertionError('no public call'))
    assert run(context,at=AT+timedelta(seconds=shift),collector=collect)==0
    collect.assert_not_called()
    assert not list((context[0]/'claims').iterdir())


@pytest.mark.parametrize('where',['clock','monitor'])
def test_stop_marker_prevents_claim_and_public_io(context,where):
    path,_,monitor,_=context
    ((path if where=='clock' else monitor)/'stop').touch()
    collect=Mock(side_effect=AssertionError('no public call'))
    assert run(context,collector=collect)==0
    collect.assert_not_called()


@pytest.mark.parametrize('race',['stop','boundary','source'])
def test_postclaim_changes_abandon_consumed_slot_without_replay(context,monkeypatch,race):
    path,*_=context
    calls=[0]
    def now():
        calls[0]+=1
        if calls[0]==2:
            if race=='stop':
                (path/'stop').touch()
            elif race=='source':
                monkeypatch.setattr(module,'implementation_fingerprint',lambda:'c'*64)
            else:
                return AT+timedelta(seconds=121)
        return AT
    collect=Mock(side_effect=AssertionError('no public call'))
    assert run(context,clock=now,collector=collect)==0
    collect.assert_not_called()
    assert (path/'claims/00000.json').exists() and not list((path/'receipts').iterdir())
    monkeypatch.setattr(module,'implementation_fingerprint',lambda:SOURCE)
    (path/'stop').unlink(missing_ok=True)
    assert run(context,collector=collect)==0
    collect.assert_not_called()


def test_forward_gap_and_backward_clock_never_refund_or_fill_missed_slots(context):
    path,*_=context
    collect=Mock(return_value=health(AT+timedelta(seconds=1800)))
    assert run(context,at=AT+timedelta(seconds=1800),collector=collect)==0
    assert run(context,at=AT+timedelta(seconds=600),collector=collect)==0
    assert collect.call_count==1 and sorted(p.name for p in (path/'claims').iterdir())==['00003.json']
    value=module.status(path,clock=lambda:AT+timedelta(seconds=1801))
    assert value['slots_consumed']==1 and value['unobserved_elapsed_slots']==3


def test_collector_failure_is_unknown_with_sanitized_reason_and_consumed_claim(context):
    path,*_=context
    collect=Mock(side_effect=RuntimeError('do not expose secret message'))
    assert run(context,collector=collect)==0
    report=monitor_archive.read(path/'receipts/00000.json')['report']
    assert report['status']=='UNVERIFIED' and report['reason_codes']==['CLOCK_DIAGNOSTIC_FAILED']
    assert 'secret' not in json.dumps(report)
    assert run(context,collector=collect)==0 and collect.call_count==1


def test_keyboard_interrupt_after_claim_is_not_replayed(context):
    path,*_=context
    with pytest.raises(KeyboardInterrupt):
        run(context,collector=Mock(side_effect=KeyboardInterrupt()))
    assert (path/'claims/00000.json').exists()
    collect=Mock(side_effect=AssertionError('no replay'))
    assert run(context,collector=collect)==0
    collect.assert_not_called()


def test_missing_receipt_after_public_check_failure_is_not_a_success_or_repeat(context,monkeypatch):
    path,*_=context
    write=monitor_archive.write
    def failing(target,value,**kwargs):
        if target.parent.name=='receipts':
            raise PermissionError('not exposed')
        write(target,value,**kwargs)
    monkeypatch.setattr(monitor_archive,'write',failing)
    collect=Mock(return_value=health())
    with pytest.raises(PermissionError):
        run(context,collector=collect)
    assert run(context,collector=collect)==0 and collect.call_count==1
    value=module.status(path,clock=lambda:AT)
    assert value['diagnostic_status']=='UNVERIFIED' and value['slots_consumed']==1


def test_locked_slot_and_low_disk_cannot_launch_public_request(context):
    path,*_=context
    collect=Mock(side_effect=AssertionError('no io'))
    with file_lock(path/'slot.lock'),pytest.raises(LockBusyError):
        run(context,collector=collect)
    with pytest.raises(ValueError):
        module.run_slot(path,module._sha(path/'plan.json'),SOURCE,clock=lambda:AT,collect=collect,free_bytes=lambda p:0)
    collect.assert_not_called()


@pytest.mark.parametrize('field,value',[
    ('max_slots',7),('max_slots',True),('interval_seconds',300),('lateness_seconds',121),
    ('timeout_seconds',10),('diagnostic_bound_us',500000),('orders_submitted',True),
    ('clock_modification_authorized',True),('ends_at',(AT+timedelta(hours=2)).isoformat())])
def test_changed_plan_policy_or_end_refused_before_public_io(context,field,value):
    path,*_=context
    plan=monitor_archive.read(path/'plan.json')
    plan[field]=value
    monitor_archive.write(path/'plan.json',plan)
    with pytest.raises(ValueError):
        run(context)


@pytest.mark.parametrize('age,source',[(-1,SOURCE),(16,SOURCE),(609,SOURCE),(1,'c'*64),(3601,SOURCE)])
def test_stale_future_expired_or_old_source_receipt_not_current_diagnostic(context,monkeypatch,age,source):
    path,*_=context
    run(context)
    monkeypatch.setattr(module,'implementation_fingerprint',lambda:source)
    value=module.status(path,clock=lambda:AT+timedelta(seconds=age))
    assert value['diagnostic_status']=='UNVERIFIED' and value['latest_sample_fresh'] is False


def test_meaningful_events_do_not_repeat_unchanged_unknown_or_good_status(context):
    path,*_=context
    run(context)
    run(context,at=AT+timedelta(seconds=600))
    assert len(list(path.glob('event-*.json')))==1
    run(context,at=AT+timedelta(seconds=1200),collector=Mock(side_effect=OSError('ignore message')))
    assert len(list(path.glob('event-*.json')))==2


def test_new_backend_preserves_original_calendar_and_claims_for_source_update(context,monkeypatch):
    path,plan,*_=context
    run(context)
    before=(path/'plan.json').read_bytes()
    claim=(path/'claims/00000.json').read_bytes()
    monkeypatch.setattr(module,'utc_now',lambda:AT+timedelta(seconds=1))
    first=path.parent/'backend-a'
    module.prepare_backend(path,first)
    monkeypatch.setattr(module,'implementation_fingerprint',lambda:'c'*64)
    second=path.parent/'backend-b'
    module.prepare_backend(path,second)
    assert (path/'plan.json').read_bytes()==before and (path/'claims/00000.json').read_bytes()==claim
    runner=monitor_archive.read(second/'runner.json')
    assert runner['source_sha256']=='c'*64 and runner['clock_directory']==str(path)
    install=(second/'install.ps1').read_text(encoding='utf-8')
    assert "ExecutionTimeLimit='PT30S'" in install and "LogonType=3" in install
    assert "pythonw.exe" in install and 'powershell.exe' not in install
    assert 'CREATE_NO_WINDOW' in (second/'launch.py').read_text(encoding='utf-8')
    assert module.status(path,clock=lambda:AT+timedelta(seconds=1))['slots_consumed']==1


def test_unknown_claim_is_not_inferred_as_zero_budget(context):
    path,*_=context
    (path/'claims/not-a-claim.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError):
        run(context)


def test_clock_report_with_raw_unknown_fields_or_qualification_is_refused(context):
    path,*_=context
    report=health()
    report['raw_secret']='not output'
    report['qualification_issued']=True
    run(context,collector=lambda timeout:report)
    receipt=monitor_archive.read(path/'receipts/00000.json')
    assert receipt['report']['status']=='UNVERIFIED' and 'raw_secret' not in json.dumps(receipt)


def test_current_freshness_reuses_original_sample_start_not_only_receipt_evaluation(context):
    path,*_=context
    samples=tuple(clock_health.Sample(endpoint,AT,AT+timedelta(seconds=2),2_000_000_000,
        AT+timedelta(seconds=1),200) for endpoint in (PublicEndpoint.SPOT_TIME,PublicEndpoint.PERP_TIME))
    report=clock_health.assess(samples,evaluated_at=AT+timedelta(seconds=2))
    run(context,at=AT+timedelta(seconds=2),collector=lambda timeout:report)
    value=module.status(path,clock=lambda:AT+timedelta(seconds=16))
    assert value['latest_sample_fresh'] is False and value['diagnostic_status']=='UNVERIFIED'
    assert value['latest_report_age_us']==14_000_000


def test_real_cli_status_serializes_integer_age_and_preserves_claim(context,monkeypatch,capsys):
    path,*_=context
    run(context)
    reader=module.status
    monkeypatch.setattr(module,'status',lambda directory:reader(directory,clock=lambda:AT+timedelta(seconds=2)))
    assert module.main(['status','--directory',str(path)])==0
    value=json.loads(capsys.readouterr().out)
    assert value['latest_report_age_us']==2_000_000 and value['slots_consumed']==1
    assert value['latest_report']['qualification_issued'] is False
