"""Read-only failed-public-task diagnosis; synthetic evidence, no qualification."""

import gzip
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock
from uuid import uuid4

import pytest
from test_candidate_catalog import NOW
from test_candidate_market import offline as offline
from test_candidate_study import study as study

from crypto_arbitrage_bot import candidate_diagnostics as module
from crypto_arbitrage_bot import candidate_study_control as cli
from crypto_arbitrage_bot.candidate_study import write_public_archive
from crypto_arbitrage_bot.exchanges.funding_history import FundingFailure, FundingWindow
from crypto_arbitrage_bot.storage.opportunity_records import exact_json


@pytest.fixture
def failed(study):
    settings, catalog, runner, _, directory = study
    runner.run_once()
    with closing(sqlite3.connect(settings.research.sqlite_path)) as db:
        db.row_factory = sqlite3.Row
        db.execute("UPDATE validation_tasks SET status='FAILED',failure_code='FUNDING_UNAVAILABLE',"
                   "attempts=1,failure_reason='sensitive arbitrary message' WHERE symbol='THEUSDT'")
        db.commit()
        task = dict(db.execute("SELECT * FROM validation_tasks WHERE symbol='THEUSDT'").fetchone())
    value = {"task_id":task['id'], "opportunity_id":task['opportunity_id'], "validation_id":task['validation_id'],
        "symbol":task['symbol'], "registration_sha256":runner.studies['THEUSDT'].sha256,
        "phase":"ACTUAL_FUNDING", "recorded_at":NOW+timedelta(seconds=3630), "catalogs":[],
        "captures":[{"phase":"ACTUAL_FUNDING", "symbol":"THEUSDT", "window":FundingWindow(
            'THEUSDT',NOW,NOW+timedelta(seconds=3630),failure=FundingFailure.TIMEOUT,
            message='sensitive transport text')} ]}
    path = directory/'registry/exit-evidence'/f'{uuid4().hex}.json'
    path.parent.mkdir()
    return settings, catalog, runner, task, json.loads(exact_json(value)), path


def diagnose(failed, **kwargs):
    return module.diagnose_candidate_exits(failed[0],evaluated_at=NOW+timedelta(hours=2),**kwargs)


def write(failed, value=None):
    path = failed[-1]
    write_public_archive(path,failed[-2] if value is None else value)
    return path.with_suffix('.json.gz')


def test_stable_source_cause_report_preserves_failed_db_and_policy(failed):
    settings, _, _, _, _, _ = failed
    write(failed)
    database = settings.research.sqlite_path
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    result = diagnose(failed)
    assert result['status']=='READ'
    assert result['task_states']=={'FAILED':1,'PENDING':3}
    assert result['failure_codes']=={'FUNDING_UNAVAILABLE':1}
    assert result['failed_tasks'][0]['recorded_cause_codes']==['FUNDING:TIMEOUT']
    text=exact_json(result)
    assert 'sensitive' not in text and 'raw_response' not in text
    assert not result['trade_allowed'] and not result['qualification_issued']
    assert not result['financial_values_recomputed'] and not result['external_authentication_proved']
    assert before==hashlib.sha256(database.read_bytes()).hexdigest()


def test_metadata_and_book_source_failures_are_retained_without_echoing_headers(failed):
    _, catalog, runner, _, value, _ = failed
    observation = runner.observation('THEUSDT')[1]
    value.update(phase='EXIT_MARKET',catalogs=json.loads(exact_json([catalog])),
        captures=[{'phase':'EXIT_MARKET','symbol':'THEUSDT','observation':json.loads(exact_json(observation))}])
    value['catalogs'][0]['reason_codes']=['SOURCE_FAILED']
    source=value['catalogs'][0]['sources'][0]
    source['error']={'kind':'TIMEOUT','message':'sensitive metadata message'}
    book=value['captures'][0]['observation']['assembly']['sources']['spot_depth']
    book['error']={'kind':'NETWORK_ERROR','message':'sensitive socket message'}
    book['response_headers']={'example':'sensitive header'}
    value['captures'][0]['observation']['data_quality']['issues']=[{'code':'SAMPLING_SKEW_TOO_LARGE'}]
    write(failed,value)
    report=diagnose(failed)
    codes=report['failed_tasks'][0]['recorded_cause_codes']
    assert 'CATALOG:SOURCE_FAILED' in codes and 'SPOT_DEPTH:NETWORK_ERROR' in codes
    assert 'SPOT_EXCHANGE_INFO:TIMEOUT' in codes and 'QUALITY:SAMPLING_SKEW_TOO_LARGE' in codes
    assert 'sensitive' not in exact_json(report)


@pytest.mark.parametrize('field', ['symbol','task_id','opportunity_id','validation_id','registration_sha256','phase','recorded_at'])
def test_wrong_task_registration_phase_or_future_archive_cannot_be_bound(failed,field):
    value=failed[-2]
    value[field]=(NOW+timedelta(days=10)).isoformat() if field=='recorded_at' else 'WRONG'
    write(failed)
    report=diagnose(failed)
    assert report['status']=='INCOMPLETE'
    assert report['failed_tasks'][0]['evidence_status']=='MISSING'
    assert report['failed_tasks'][0]['recorded_cause_codes']==[]
    assert report['archive_issues'][0]['code']=='CORRUPT_OR_UNBOUND_ARCHIVE'


@pytest.mark.parametrize('damage',['truncated_gzip','duplicate_json','noncanonical','unrecognized_name'])
def test_corrupt_or_unrecognized_archive_never_becomes_complete(failed,damage):
    p=write(failed)
    if damage=='truncated_gzip':
        p.write_bytes(p.read_bytes()[:20])
    elif damage=='duplicate_json':
        p.write_bytes(gzip.compress(b'{"task_id":"a","task_id":"b"}'))
    elif damage=='noncanonical':
        p.write_bytes(gzip.compress(json.dumps(failed[-2]).encode()))
    else:
        p.rename(p.with_name('unknown.json.gz'))
    report=diagnose(failed)
    assert report['status']=='INCOMPLETE' and report['archive_issues']


def test_ambiguous_attempt_archives_are_not_selected_by_latest_time(failed):
    path=write(failed)
    path.with_name(f'{uuid4().hex}.json.gz').write_bytes(path.read_bytes())
    report=diagnose(failed)
    assert report['status']=='INCOMPLETE'
    item=report['failed_tasks'][0]
    assert item['evidence_status']=='AMBIGUOUS' and item['recorded_cause_codes']==[]
    assert len(item['archives'])==2


def test_absent_raw_failure_archive_is_explicitly_missing(failed):
    report=diagnose(failed)
    assert report['status']=='INCOMPLETE'
    assert report['failed_tasks'][0]['evidence_status']=='MISSING'


def test_archive_bytes_and_files_are_bounded(failed):
    path=write(failed)
    report=diagnose(failed,limits=module.DiagnosticLimits(max_total_bytes=10))
    assert report['status']=='INCOMPLETE' and report['scan']['decompressed_bytes']==11
    path.with_name(f'{uuid4().hex}.json.gz').write_bytes(path.read_bytes())
    report=diagnose(failed,limits=module.DiagnosticLimits(max_files=1))
    assert report['status']=='INCOMPLETE'
    assert any(v['code']=='DIAGNOSTIC_FILE_LIMIT' for v in report['archive_issues'])


def test_task_limit_rejects_incomplete_database_snapshot(failed):
    with pytest.raises(ValueError,match='DIAGNOSTIC_TASK_LIMIT'):
        diagnose(failed,limits=module.DiagnosticLimits(max_tasks=1))


def test_expired_audit_deadline_does_not_emit_partial_read_success(failed,monkeypatch):
    monkeypatch.setattr(module,'monotonic',Mock(side_effect=[0,100]))
    with pytest.raises((ValueError,sqlite3.OperationalError)):
        diagnose(failed,limits=module.DiagnosticLimits(max_seconds=1))


@pytest.mark.parametrize('value',[True,0,-1,121])
def test_invalid_resource_limits_rejected(value):
    with pytest.raises(ValueError,match='BOUNDED_DIAGNOSTIC_LIMITS_REQUIRED'):
        module.DiagnosticLimits(max_seconds=value)


def test_cli_writes_new_report_and_refuses_existing_output_before_diagnosis(failed,monkeypatch,capsys):
    settings=failed[0]
    output=failed[-1].parent.parent.parent/'diagnostics.json'
    monkeypatch.setattr(cli,'load_settings',lambda _:settings)
    monkeypatch.setattr(cli,'utc_now',lambda:NOW+timedelta(hours=2))
    write(failed)
    argv=['diagnostics','--config','synthetic.toml','--output',str(output)]
    assert cli.main(argv)==0
    result=json.loads(output.read_text())
    assert result['status']=='READ' and not result['orders_submitted']
    sentinel=output.read_bytes()
    monkeypatch.setattr(cli,'diagnose_candidate_exits',Mock(side_effect=AssertionError('should not scan')))
    assert cli.main(argv)==1 and output.read_bytes()==sentinel
    assert 'sensitive' not in capsys.readouterr().out


def test_cli_incomplete_report_is_saved_but_has_nonzero_status(failed,monkeypatch):
    settings=failed[0]
    output=failed[-1].parent.parent.parent/'incomplete.json'
    monkeypatch.setattr(cli,'load_settings',lambda _:settings)
    monkeypatch.setattr(cli,'utc_now',lambda:NOW+timedelta(hours=2))
    assert cli.main(['diagnostics','--config','synthetic.toml','--output',str(output)])==1
    assert json.loads(output.read_text())['status']=='INCOMPLETE'


@pytest.mark.parametrize('field',['request_symbol','source_endpoint','unknown_error_code'])
def test_unbound_or_unknown_source_claims_are_rejected_and_never_echoed(failed,field):
    _, _, runner, _, value, _ = failed
    observation=json.loads(exact_json(runner.observation('THEUSDT')[1]))
    value.update(phase='EXIT_MARKET',captures=[{'phase':'EXIT_MARKET','symbol':'THEUSDT','observation':observation}])
    source=observation['assembly']['sources']['spot_depth']
    if field=='request_symbol':
        params=source['request']['params']
        if isinstance(params,list):
            source['request']['params']=[[key,'BTCUSDT' if key=='symbol' else value] for key,value in params]
        else:
            params['symbol']='BTCUSDT'
    elif field=='source_endpoint':
        source['request']['endpoint']='https://fapi.binance.com/fapi/v1/depth'
    else:
        source['error']={'kind':'SENSITIVE_UNKNOWN_TEXT','message':'sensitive'}
    write(failed)
    report=diagnose(failed)
    assert report['status']=='INCOMPLETE' and report['failed_tasks'][0]['evidence_status']=='MISSING'
    assert 'SENSITIVE_UNKNOWN_TEXT' not in exact_json(report)


def test_inflight_archive_is_counted_separately_from_terminal_failed_tasks(failed):
    write(failed)
    with closing(sqlite3.connect(failed[0].research.sqlite_path)) as db:
        db.execute("UPDATE validation_tasks SET status='RUNNING',failure_code=NULL WHERE symbol='THEUSDT'")
        db.commit()
    report=diagnose(failed)
    assert report['status']=='READ' and report['failed_tasks']==[]
    assert report['task_states']=={'RUNNING':1,'PENDING':3}


def test_unavailable_db_does_not_create_a_replacement_or_issue_success(failed):
    database=failed[0].research.sqlite_path
    database.rename(database.with_suffix('.preserved'))
    with pytest.raises(sqlite3.OperationalError):
        diagnose(failed)
    assert not database.exists()


def test_one_archive_decompression_limit_applies_before_json_parsing(failed,monkeypatch):
    write(failed)
    monkeypatch.setattr(module,'MAX_RAW_ARCHIVE_BYTES',20)
    report=diagnose(failed)
    assert report['status']=='INCOMPLETE'
    assert report['scan']['decompressed_bytes']==21
    assert report['archive_issues']==[{'code':'DIAGNOSTIC_ARCHIVE_BYTES_LIMIT'}]


def test_changed_policy_is_rejected_before_reading_archive_files(failed):
    settings=failed[0]
    changed=settings.model_copy(update={'fees':settings.fees.model_copy(update={'spot_entry':Decimal('.2')})})
    with pytest.raises(ValueError):
        module.diagnose_candidate_exits(changed,evaluated_at=NOW+timedelta(hours=2))


def test_archived_canonical_hash_is_independently_reproducible(failed):
    p=write(failed)
    report=diagnose(failed)
    archived=report['failed_tasks'][0]['archives'][0]
    assert archived['canonical_sha256']==hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()
