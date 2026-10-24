from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from crypto_arbitrage_bot import desktop_alerts as alerts
from crypto_arbitrage_bot import monitor_archive as archive

NOW = datetime(2026,10,13,0,0,tzinfo=UTC)
SHA = 'a'*64
SOURCE = 'b'*64


def observation(health=(), failed=8, collection=0, interrupted=1):
    return {'health':list(health),'counters':{'failed_validations':failed,
        'collection_failures':collection,'interrupted_attempts':interrupted}}


def native_result():
    return {'api_attempted':True,'api_accepted':True,'shell_reported_shown':False,'user_clicked':False,
            'shell_reported_hidden':False,'shell_reported_timeout':False,'user_read_verified':False}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monitor=tmp_path/'work/monitor'
    monitor.mkdir(parents=True)
    (monitor/'monitor-binding.json').write_bytes(b'binding')
    original={'started_at':(NOW-timedelta(hours=1)).isoformat(),'ends_at':(NOW+timedelta(minutes=10)).isoformat()}
    monkeypatch.setattr(archive,'load_plan',lambda directory, sha: (original,SHA))
    monkeypatch.setattr(archive,'load_binding',lambda directory, sha: {})
    monkeypatch.setattr(alerts,'implementation_fingerprint',lambda: SOURCE)
    directory=tmp_path/'work/alerts'
    result=alerts.prepare(directory,monitor,SHA,clock=lambda:NOW)
    return directory,result['plan_sha256'],monitor


def tick(prepared, seconds, current, sender=None):
    directory,sha,_=prepared
    return alerts.run_slot(directory,sha,SOURCE,clock=lambda:NOW+timedelta(seconds=seconds),
        observer=lambda plan,now:current,sender=(lambda title,body:native_result()) if sender is None else sender,
        availability=lambda:True)


def test_original_end_and_finite_separate_budget(prepared):
    directory,sha,_=prepared
    plan=alerts.load_plan(directory,sha)
    assert plan['ends_at']==(NOW+timedelta(minutes=10)).isoformat()
    assert plan['max_slots']==10 and plan['max_notices']==64


def test_existing_directory_is_not_a_budget_reset(prepared):
    directory,_,monitor=prepared
    with pytest.raises(FileExistsError):
        alerts.prepare(directory,monitor,SHA,clock=lambda:NOW)


def test_healthy_baseline_and_old_failures_are_quiet(prepared):
    assert tick(prepared,0,observation())['status']=='QUIET'
    assert tick(prepared,60,observation())['status']=='QUIET'
    assert alerts.status(prepared[0])['notices_consumed']==0


def test_fault_deduplicates_then_recovers_once(prepared):
    assert tick(prepared,0,observation(['MONITOR_UNCONFIRMED']))['status']=='ATTEMPTED'
    assert tick(prepared,60,observation(['MONITOR_UNCONFIRMED']))['status']=='QUIET'
    assert tick(prepared,120,observation())['status']=='ATTEMPTED'
    assert tick(prepared,180,observation())['status']=='QUIET'
    assert alerts.status(prepared[0])['notices_consumed']==2


def test_new_failed_validation_is_one_event_not_false_recovery(prepared):
    tick(prepared,0,observation())
    assert tick(prepared,60,observation(failed=12))['status']=='ATTEMPTED'
    assert tick(prepared,120,observation(failed=12))['status']=='QUIET'


@pytest.mark.parametrize('field', ['collection','interrupted'])
def test_new_collection_failure_or_interruption(prepared,field):
    tick(prepared,0,observation())
    values={'collection':0,'interrupted':1}
    values[field]+=1
    assert tick(prepared,60,observation(**values))['status']=='ATTEMPTED'


def test_unknown_counts_are_not_zero_or_recovered():
    assert alerts.transition(observation(failed=None),observation()) is None


@pytest.mark.parametrize('code', ['MONITOR_UNCONFIRMED','REPORT_UNAVAILABLE','RESEARCH_ATTENTION','SAVED_DATA_INVALID'])
def test_only_fixed_public_operational_text(code):
    title,body=alerts.transition(observation([code]),None)
    assert '套利机器人' in title and '未自动交易' in body


def test_arbitrary_failure_text_is_rejected_before_notification():
    with pytest.raises(ValueError):
        alerts.transition(observation(['private-key-value']),None)


@pytest.mark.parametrize('count', [-1,True,'secret',2**64])
def test_invalid_counters_are_rejected(count):
    with pytest.raises(ValueError):
        alerts.transition(observation(failed=count),None)


@pytest.mark.parametrize('seconds', [-1,580,600])
def test_before_start_or_end_margin_has_no_side_effect(prepared,seconds):
    sent=[]
    result=tick(prepared,seconds,observation(['REPORT_UNAVAILABLE']),lambda *args:sent.append(args))
    assert result['status']=='SKIPPED' and not sent
    assert alerts.status(prepared[0])['notices_consumed']==0


@pytest.mark.parametrize('which', ['own','monitor'])
def test_stop_prevents_notification(prepared,which):
    (prepared[0] if which=='own' else prepared[2]).joinpath('stop').touch()
    assert tick(prepared,0,observation(['REPORT_UNAVAILABLE']))['status']=='SKIPPED'


def test_source_mismatch_prevents_probe_and_notification(prepared,monkeypatch):
    monkeypatch.setattr(alerts,'implementation_fingerprint',lambda:'c'*64)
    assert tick(prepared,0,observation(['REPORT_UNAVAILABLE']))['status']=='SKIPPED'


def test_slot_is_consumed_and_clock_reverse_does_not_replay(prepared):
    assert tick(prepared,120,observation(['REPORT_UNAVAILABLE']))['status']=='ATTEMPTED'
    assert tick(prepared,120,observation())['status']=='SKIPPED'
    assert tick(prepared,60,observation())['status']=='SKIPPED'


def test_native_failure_is_sanitized_and_not_retried(prepared):
    def fail(*args):
        raise OSError('credential-like private exception body')
    result=tick(prepared,0,observation(['REPORT_UNAVAILABLE']),fail)
    assert result['status']=='NOTIFICATION_FAILED' and result['error_type']=='OSError'
    assert 'private' not in str(result)
    assert tick(prepared,60,observation(['REPORT_UNAVAILABLE']))['status']=='QUIET'
    assert alerts.status(prepared[0])['notices_consumed']==1


def test_unknown_native_result_never_enters_receipt(prepared):
    result=tick(prepared,0,observation(['REPORT_UNAVAILABLE']),lambda *args:{'secret':'value'})
    assert result['status']=='NOTIFICATION_FAILED' and 'secret' not in str(result)


def test_notify_claim_exists_before_native_io(prepared):
    def send(*args):
        assert (prepared[0]/'claims/0000.json').exists()
        return native_result()
    assert tick(prepared,0,observation(['REPORT_UNAVAILABLE']),send)['status']=='ATTEMPTED'


def test_stop_after_claim_consumes_attempt_without_native_call(prepared,monkeypatch):
    original=archive.write
    def write(path,value,**kwargs):
        original(path,value,**kwargs)
        if path.parent.name=='claims':
            (prepared[0]/'stop').touch()
    monkeypatch.setattr(archive,'write',write)
    sent=[]
    result=tick(prepared,0,observation(['REPORT_UNAVAILABLE']),lambda *args:sent.append(args))
    assert result['status']=='NOT_ATTEMPTED' and not sent
    assert alerts.status(prepared[0])['notices_consumed']==1


def test_receipt_write_failure_does_not_reissue_notice(prepared,monkeypatch):
    original=archive.write
    sent=[]
    def write(path,value,**kwargs):
        if path.parent.name=='receipts':
            raise OSError('disk failure')
        return original(path,value,**kwargs)
    monkeypatch.setattr(archive,'write',write)
    with pytest.raises(OSError):
        tick(prepared,0,observation(['REPORT_UNAVAILABLE']),lambda *args:sent.append(args) or native_result())
    monkeypatch.setattr(archive,'write',original)
    assert tick(prepared,60,observation(['REPORT_UNAVAILABLE']))['status']=='QUIET'
    assert len(sent)==1 and alerts.status(prepared[0])['notices_consumed']==1


def test_unknown_claim_file_does_not_refund_budget(prepared):
    (prepared[0]/'claims/unknown').write_text('bad')
    with pytest.raises(ValueError):
        tick(prepared,0,observation())


def test_cursor_rollback_is_refused(prepared):
    tick(prepared,120,observation(['REPORT_UNAVAILABLE']))
    archive.write(prepared[0]/'cursor.json',{'slot':-1})
    with pytest.raises(ValueError,match='ROLLBACK'):
        tick(prepared,180,observation())


def test_missing_baseline_with_existing_claim_is_refused(prepared):
    tick(prepared,0,observation(['REPORT_UNAVAILABLE']))
    (prepared[0]/'observation.json').unlink()
    with pytest.raises(ValueError,match='MISSING'):
        tick(prepared,60,observation(['REPORT_UNAVAILABLE']))


def test_budget_exhaustion_cannot_reset(tmp_path,prepared,monkeypatch):
    directory,sha,_=prepared
    for i in range(64):
        archive.write(directory/'claims'/f'{i:04d}.json',{'index':i,'slot':i,'plan_sha256':sha,
            'source_sha256':SOURCE,'claimed_at':NOW},exclusive=True)
    archive.write(directory/'observation.json',observation())
    # Extend only the injected fixture end, never a real plan.
    plan=alerts.load_plan(directory,sha)
    plan['ends_at']=(NOW+timedelta(hours=2)).isoformat()
    plan['max_slots']=120
    monkeypatch.setattr(alerts,'load_plan',lambda *args:plan)
    archive.write(directory/'cursor.json',{'slot':63})
    sent=[]
    result=tick(prepared,64*60,observation(['REPORT_UNAVAILABLE']),lambda *args:sent.append(args))
    assert result['status']=='NOTICE_BUDGET_EXHAUSTED' and not sent


def test_backend_pins_source_and_original_calendar(prepared,monkeypatch):
    monkeypatch.setattr(alerts,'utc_now',lambda:NOW)
    output=Path.cwd()/'work/backend'
    alerts.prepare_backend(prepared[0],output)
    runner=archive.read(output/'runner.json')
    assert runner['source_sha256']==SOURCE and runner['ends_at']==(NOW+timedelta(minutes=10)).isoformat()
    launch=(output/'launch.py').read_text(encoding='utf-8')
    assert 'CREATE_NO_WINDOW' in launch and 'desktop_alerts' in launch
    install=(output/'install.ps1').read_text(encoding='utf-8')
    assert "LogonType=3" in install and "'PT30S'" in install and "'PT1M'" in install


def test_busy_defers_without_consuming_notice_or_losing_transition(prepared):
    directory,sha,_ = prepared
    result=alerts.run_slot(directory,sha,SOURCE,clock=lambda:NOW,
        observer=lambda *args:observation(['REPORT_UNAVAILABLE']),availability=lambda:False)
    assert result['status']=='DEFERRED_OS_BUSY_OR_UNKNOWN'
    assert alerts.status(directory)['notices_consumed']==0
    assert tick(prepared,60,observation(['REPORT_UNAVAILABLE']))['status']=='ATTEMPTED'


def test_resolved_unsent_fault_is_not_displayed_later(prepared):
    directory,sha,_ = prepared
    alerts.run_slot(directory,sha,SOURCE,clock=lambda:NOW,
        observer=lambda *args:observation(['REPORT_UNAVAILABLE']),availability=lambda:False)
    assert tick(prepared,60,observation())['status']=='QUIET'


def test_availability_race_has_no_ui_api_and_keeps_claim_consumed(prepared):
    result={k:False for k in native_result()}
    assert tick(prepared,0,observation(['REPORT_UNAVAILABLE']),lambda *args:result)['status']=='OS_BLOCKED_NO_API'
    assert tick(prepared,60,observation(['REPORT_UNAVAILABLE']))['status']=='ATTEMPTED'
    assert alerts.status(prepared[0])['notices_consumed']==2


def test_missing_receipt_is_delivery_unknown_not_empty_success(prepared,monkeypatch):
    original=archive.write
    def write(path,value,**kwargs):
        if path.parent.name=='receipts':
            raise OSError('disk failure')
        return original(path,value,**kwargs)
    monkeypatch.setattr(archive,'write',write)
    with pytest.raises(OSError):
        tick(prepared,0,observation(['REPORT_UNAVAILABLE']))
    monkeypatch.setattr(archive,'write',original)
    result=alerts.status(prepared[0])
    assert result['status']=='NOTIFICATION_EVIDENCE_INCOMPLETE' and result['missing_receipts']==1


def test_counter_rollback_is_an_alert_and_does_not_lower_watermark(prepared):
    tick(prepared,0,observation())
    assert tick(prepared,60,observation(failed=0))['status']=='ATTEMPTED'
    assert tick(prepared,120,observation(failed=0))['status']=='QUIET'
    baseline=archive.read(prepared[0]/'observation.json')
    assert baseline['counters']['failed_validations']==8
    assert baseline['health']==['REPORT_UNAVAILABLE']


def test_malformed_latest_cannot_claim_notification_success(prepared):
    tick(prepared,0,observation(['REPORT_UNAVAILABLE']))
    value=archive.read(prepared[0]/'latest.json')
    value['notices_consumed']=2
    archive.write(prepared[0]/'latest.json',value)
    with pytest.raises(ValueError,match='DIFFERS'):
        alerts.status(prepared[0])
