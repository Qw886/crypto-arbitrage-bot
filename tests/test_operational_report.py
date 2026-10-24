"""Read-only quality evidence and real finite report publishing; no exchange."""
import copy
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import timedelta
from unittest.mock import Mock

import pytest
from test_offline_estimate import OPENED

from crypto_arbitrage_bot import operational_report as module
from crypto_arbitrage_bot import research_control, runtime_status
from crypto_arbitrage_bot.storage.opportunity_records import exact_json


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr('socket.create_connection',Mock(side_effect=AssertionError('no network')))


@pytest.fixture
def database(tmp_path):
    path=tmp_path/'research.sqlite'
    with closing(sqlite3.connect(path)) as db,db:
        db.execute('CREATE TABLE opportunities(symbol,created_at,snapshot_status,data_quality_status,data_quality_reason_codes,decision)')
        db.execute('INSERT INTO opportunities VALUES(?,?,?,?,?,?)',
            ('BTCUSDT',OPENED.isoformat(),'COMPLETE','VALID','[]','REJECT'))
    return path


def insert(path,**change):
    row={'symbol':'ETHUSDT','created_at':(OPENED+timedelta(seconds=1)).isoformat(),
         'snapshot_status':'COMPLETE','data_quality_status':'INVALID',
         'data_quality_reason_codes':'["FUTURE_TIMESTAMP"]','decision':'REJECT'}
    row.update(change)
    with closing(sqlite3.connect(path)) as db,db:
        db.execute('INSERT INTO opportunities VALUES(?,?,?,?,?,?)',tuple(row.values()))


def test_recent_metadata_is_read_only_in_original_snapshot_and_null_is_preserved(database):
    insert(database,data_quality_status=None,data_quality_reason_codes=None,snapshot_status='INCOMPLETE')
    before=hashlib.sha256(database.read_bytes()).hexdigest()
    report=research_control.database_status(database)
    value=report['recent_saved_quality']
    assert report['status']=='READ' and report['decisions']=={'REJECT':2}
    assert value['scope']=='LAST_SAVED_ROWS_NOT_ALL_ATTEMPTS'
    assert value['rows'][0]['quality_status'] is None and value['rows'][0]['quality_reasons'] is None
    assert module.quality_signal(value)=={'status':'READ','valid':1,'invalid':0,'unknown':1,'reasons':[]}
    assert hashlib.sha256(database.read_bytes()).hexdigest()==before


def test_last_saved_eight_are_insertion_order_not_timestamp_or_complete_attempts(database):
    for number in range(12):
        insert(database,created_at=(OPENED-timedelta(seconds=number)).isoformat())
    value=research_control.database_status(database)['recent_saved_quality']
    assert len(value['rows'])==8 and value['limit']==8
    assert value['rows'][0]['saved_at']==(OPENED-timedelta(seconds=11)).isoformat()
    assert module.quality_signal(value)['invalid']==8
    assert module.quality_signal(value)['reasons']==['FUTURE_TIMESTAMP']


@pytest.mark.parametrize('change',[
    {'symbol':'private-secret-or-script'}, {'created_at':'2026-01-01T00:00:00'},
    {'created_at':'bad'}, {'snapshot_status':'BROKEN'}, {'data_quality_status':'GOOD'},
    {'data_quality_reason_codes':'"secret text"'}, {'data_quality_reason_codes':'["UNKNOWN"]'},
    {'data_quality_reason_codes':'[["FUTURE_TIMESTAMP"]]'},
    {'data_quality_reason_codes':'["FUTURE_TIMESTAMP","FUTURE_TIMESTAMP"]'},
    {'decision':'QUALIFIED'}, {'data_quality_status':'VALID'},
    {'data_quality_status':'VALID','data_quality_reason_codes':None},
    {'data_quality_status':'VALID','data_quality_reason_codes':'[]','snapshot_status':'INCOMPLETE'},
    {'data_quality_reason_codes':'x'*4097},
],ids=['symbol','naive','date','snapshot','quality','string','unknown-code','nested-code',
       'duplicate-code','decision','contradiction','missing-valid-reasons','incomplete-valid','oversize'])
def test_bad_or_large_metadata_remains_unavailable_without_exposing_raw_fields(database,change):
    insert(database,**change)
    report=research_control.database_status(database)
    value=report['recent_saved_quality']
    assert report['status']=='READ' and report['decisions']['REJECT']>=1
    assert value['status']=='UNAVAILABLE' and value['rows'] is None
    assert value['reason']=='METADATA_INVALID_OR_UNREADABLE'
    assert 'secret' not in exact_json(value) and 'xxxx' not in exact_json(value)


def test_empty_and_missing_columns_are_not_zero_success_samples(database,tmp_path):
    with closing(sqlite3.connect(database)) as db,db:
        db.execute('DELETE FROM opportunities')
    empty=research_control.database_status(database)['recent_saved_quality']
    assert empty['status']=='EMPTY' and empty['rows']==[]
    assert module.quality_signal(empty)['valid'] is None
    other=tmp_path/'old.sqlite'
    with closing(sqlite3.connect(other)) as db,db:
        db.execute('CREATE TABLE opportunities(decision)')
    missing=research_control.database_status(other)['recent_saved_quality']
    assert missing['status']=='NOT_AVAILABLE' and missing['rows'] is None


@pytest.fixture
def snapshot(database):
    return {'evaluated_at':OPENED,'orders_submitted':False,'credentials_read':False,
        'research':{'status':'RUNNING_CONFIRMED','worker_pid':4242,'budget_consumed':3,'budget_limit':10,
            'original_deadline':OPENED+timedelta(seconds=100),
            'database':research_control.database_status(database)},
        'testnet':{'status':'READ','budget_consumed':20,'budget_limit':20}}


def test_markdown_shows_separate_evidence_limits_and_exact_source_digest(snapshot):
    report=module.render_report(snapshot)
    digest=hashlib.sha256(exact_json(snapshot).encode()).hexdigest()
    assert digest in report and '3 / 10' in report and '20 / 20' in report
    assert '| FAILED | NULL |' in report and '不是全部尝试' in report
    assert '工程风险资格、独立净优势：本报告未核查' in report
    assert '| BTCUSDT |' in report and 'VALID' in report and 'REJECT' in report


def test_renderer_never_exposes_unknown_text_or_large_errors_and_keeps_null(snapshot):
    snapshot['research'].update(status='secret key text',worker_pid=True,error='private-secret'*100000)
    snapshot['testnet']['budget_consumed']=False
    snapshot['research']['database']['recent_saved_quality']={'status':'READ','rows':[
        {'symbol':'<script>secret</script>','quality_status':'private-secret','saved_at':'bad','decision':'private-secret'}]}
    report=module.render_report(snapshot)
    assert 'secret' not in report and '<script>' not in report and len(report)<10000
    assert 'PID：NULL' in report and 'NULL / 20' in report


def test_complete_grouped_task_counts_distinguish_zero_from_unknown(snapshot):
    database=snapshot['research']['database']
    database['tasks']={'FAILED':8,'PENDING':4}
    report=module.render_report(snapshot)
    assert '| COMPLETED | 0 |' in report and '| RUNNING | 0 |' in report
    database['tasks']={}
    assert '| COMPLETED | 0 |' in module.render_report(snapshot)
    database['tasks']=None
    assert '| COMPLETED | NULL |' in module.render_report(snapshot)


@pytest.mark.parametrize('tasks',[{'BROKEN':1},{'FAILED':True},{'FAILED':-1},{'FAILED':'8'}])
def test_unverified_grouped_counts_never_infer_missing_category_zero(snapshot,tasks):
    snapshot['research']['database']['tasks']=tasks
    assert '| COMPLETED | NULL |' in module.render_report(snapshot)


def test_quality_signal_ignores_row_identity_time_but_exposes_real_quality_changes(snapshot):
    report=copy.deepcopy(snapshot)
    before=runtime_status._meaningful(report)
    row=report['research']['database']['recent_saved_quality']['rows'][0]
    row['saved_at']=(OPENED+timedelta(seconds=10)).isoformat()
    row['symbol']='ETHUSDT'
    assert runtime_status._meaningful(report)==before
    row.update(quality_status='INVALID',quality_reasons=['FUTURE_TIMESTAMP'])
    after=runtime_status._meaningful(report)
    assert after!=before and 'FUTURE_TIMESTAMP' in after


def test_publish_error_changes_event_fingerprint_without_changing_native_status(snapshot):
    healthy=runtime_status._meaningful(snapshot)
    snapshot['operational_report']={'status':'UNAVAILABLE'}
    fingerprint=runtime_status._meaningful(snapshot)
    assert 'REPORT_PUBLICATION_FAILED' in fingerprint and healthy!=fingerprint
    assert snapshot['research']['status']=='RUNNING_CONFIRMED'


def test_collection_failure_changes_event_while_owner_and_saved_quality_stay_normal(snapshot):
    snapshot['research']['collection_counters']={'finished':3,'failed':1,'interrupted':0,'last_failure_present':False}
    previous=runtime_status._meaningful(snapshot)
    snapshot['research']['collection_counters'].update(finished=4,failed=2,last_failure_present=True)
    current=runtime_status._meaningful(snapshot)
    assert previous!=current and 'collection_failure_signal' in current
    assert snapshot['research']['status']=='RUNNING_CONFIRMED'
    assert '失败尝试：2' in module.render_report(snapshot)


def test_same_fsynced_report_is_atomic_and_failed_replace_keeps_old_file(tmp_path,snapshot,monkeypatch):
    module.publish_report(tmp_path,snapshot)
    path=tmp_path/'report.md'
    old=path.read_bytes()
    snapshot['evaluated_at']=OPENED+timedelta(seconds=1)
    monkeypatch.setattr(module,'publish_state',Mock(side_effect=PermissionError('not exposed')))
    with pytest.raises(PermissionError):
        module.publish_report(tmp_path,snapshot)
    assert path.read_bytes()==old and (tmp_path/'report.md.tmp').read_bytes()!=old


def test_finite_monitor_publishes_matching_report_and_quality_change_event(tmp_path,snapshot):
    elapsed=[0]
    reports=[copy.deepcopy(snapshot),copy.deepcopy(snapshot)]
    reports[1]['research']['database']['recent_saved_quality']['rows'][0].update(
        quality_status='INVALID',quality_reasons=['FUTURE_TIMESTAMP'])
    def check():
        report=reports.pop(0)
        report['evaluated_at']=OPENED+timedelta(seconds=elapsed[0])
        return report
    def wait(seconds):
        elapsed[0]+=seconds
        return False
    directory=tmp_path/'monitor'
    result=runtime_status.monitor_status(check,directory,duration_seconds=10,interval_seconds=1,max_checks=2,
        clock=lambda:OPENED+timedelta(seconds=elapsed[0]),steady_ns=lambda:int(elapsed[0]*1e9),wait=wait)
    assert result['checks']==2 and result['changes']==2 and result['orders_submitted'] is False
    latest=(directory/'latest.json').read_bytes()
    assert hashlib.sha256(latest).hexdigest() in (directory/'report.md').read_text(encoding='utf-8')
    assert 'FUTURE_TIMESTAMP' in (directory/'event-00001.json').read_text(encoding='utf-8')


def test_report_failure_is_retained_without_replaying_claims_or_stopping_monitor(tmp_path,snapshot,monkeypatch):
    monkeypatch.setattr(runtime_status,'publish_report',Mock(side_effect=PermissionError('sensitive error text')))
    result=runtime_status.monitor_status(lambda:snapshot,tmp_path/'monitor',duration_seconds=10,interval_seconds=1,
        max_checks=1,clock=lambda:OPENED,steady_ns=lambda:0,wait=lambda seconds:False)
    latest=json.loads((tmp_path/'monitor/latest.json').read_text(encoding='utf-8'))
    assert result['checks']==1 and len(list((tmp_path/'monitor/check-claims').glob('*.json')))==1
    assert latest['operational_report']['status']=='UNAVAILABLE'
    assert 'sensitive' not in exact_json(latest)
    assert 'REPORT_PUBLICATION_FAILED' in (tmp_path/'monitor/event-00000.json').read_text(encoding='utf-8')


@pytest.mark.parametrize('age',[0,91,-1])
def test_cli_marks_saved_snapshot_age_not_current_native_or_market_proof(tmp_path,snapshot,monkeypatch,capsys,age):
    from crypto_arbitrage_bot import monitor_archive
    monitor_archive.write(tmp_path/'latest.json',snapshot)
    monkeypatch.setattr(monitor_archive,'load_plan',lambda *args:({},'a'*64))
    monkeypatch.setattr(module,'utc_now',lambda:OPENED+timedelta(seconds=age))
    assert module.main(['--monitor-dir',str(tmp_path),'--original-plan-sha256','a'*64])==0
    value=capsys.readouterr().out
    assert ('以下是历史快照' in value) is (age!=0)
    assert '工程风险资格、独立净优势：本报告未核查' in value
