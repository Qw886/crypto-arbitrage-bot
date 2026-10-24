"""Finite PUBLIC clock slots; no time changes, research replay or credentials."""

import argparse
import hashlib
import math
import re
import shutil
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from crypto_arbitrage_bot import clock_health, monitor_archive
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.execution.risk_review import implementation_fingerprint
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.file_lock import LockBusyError, file_lock

INTERVAL_SECONDS = 600
LATENESS_SECONDS = 120
TIMEOUT_SECONDS = 5
MIN_REMAINING_SECONDS = 8
MAX_REPORT_BYTES = 16384
MIN_FREE_BYTES = 512*1024*1024


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _within(path: Path, root: Path) -> Path:
    value=path.resolve()
    if path.is_symlink() or not value.is_relative_to(root/'work'):
        raise ValueError('CLOCK_WORK_SCOPE_REQUIRED')
    return value


def prepare(directory: Path, monitor: Path, original_sha: str, *, clock: Callable[[],datetime]=utc_now) -> dict[str, Any]:
    root=Path.cwd().resolve()
    directory=_within(directory,root)
    monitor=monitor.resolve()
    if not monitor.is_relative_to(root) or monitor.is_symlink():
        raise ValueError('ORIGINAL_MONITOR_SCOPE_REQUIRED')
    original,sha=monitor_archive.load_plan(monitor,original_sha)
    monitor_archive.load_binding(monitor,sha)
    start=utc_timestamp(clock())
    end=monitor_archive.at(original['ends_at'])
    if not start+timedelta(seconds=MIN_REMAINING_SECONDS)<end<=start+timedelta(days=8):
        raise ValueError('ORIGINAL_CLOCK_WINDOW_REQUIRED')
    value={'schema_version':1,'root':str(root),'monitor_directory':str(monitor),'monitor_sha256':sha,
        'started_at':start,'ends_at':end,'interval_seconds':INTERVAL_SECONDS,'lateness_seconds':LATENESS_SECONDS,
        'timeout_seconds':TIMEOUT_SECONDS,'diagnostic_bound_us':clock_health.OFFSET_BOUND_US,
        'max_slots':math.ceil((end-start).total_seconds()/INTERVAL_SECONDS),
        'orders_submitted':False,'clock_modification_authorized':False}
    directory.mkdir(parents=True,exist_ok=False)
    monitor_archive.write(directory/'plan.json',value,exclusive=True)
    (directory/'claims').mkdir()
    (directory/'receipts').mkdir()
    return value


def load_plan(directory: Path, expected_sha: str | None=None) -> dict[str, Any]:
    directory=directory.resolve()
    if expected_sha is not None and _sha(directory/'plan.json')!=expected_sha:
        raise ValueError('CLOCK_PLAN_CHANGED')
    value=monitor_archive.read(directory/'plan.json')
    if set(value)!={'schema_version','root','monitor_directory','monitor_sha256','started_at','ends_at',
                   'interval_seconds','lateness_seconds','timeout_seconds','diagnostic_bound_us','max_slots',
                   'orders_submitted','clock_modification_authorized'}:
        raise ValueError('EXACT_CLOCK_PLAN_REQUIRED')
    root=Path(value['root'])
    if root!=root.resolve() or root!=Path.cwd().resolve() or _within(directory,root)!=directory:
        raise ValueError('CLOCK_ORIGINAL_ROOT_REQUIRED')
    fixed={'schema_version':1,'interval_seconds':600,'lateness_seconds':120,'timeout_seconds':5,'diagnostic_bound_us':250000}
    if any(type(value[k]) is not int or value[k]!=v for k,v in fixed.items()) or clock_health.OFFSET_BOUND_US!=250000:
        raise ValueError('CLOCK_DIAGNOSTIC_POLICY_CHANGED')
    start,end=monitor_archive.at(value['started_at']),monitor_archive.at(value['ends_at'])
    if not start<end<=start+timedelta(days=8) or type(value['max_slots']) is not int or (
        value['max_slots']!=math.ceil((end-start).total_seconds()/600)) or (
        value['orders_submitted'] is not False or value['clock_modification_authorized'] is not False):
        raise ValueError('CLOCK_ORIGINAL_BUDGET_REQUIRED')
    monitor=Path(value['monitor_directory'])
    if not monitor.is_relative_to(root) or monitor!=monitor.resolve() or monitor.is_symlink():
        raise ValueError('CLOCK_ORIGINAL_MONITOR_REQUIRED')
    original,sha=monitor_archive.load_plan(monitor,value['monitor_sha256'])
    monitor_archive.load_binding(monitor,sha)
    if monitor_archive.at(original['ends_at'])!=end:
        raise ValueError('CLOCK_ORIGINAL_DEADLINE_CHANGED')
    return value


def _claims(directory: Path, plan: dict[str, Any]) -> dict[int,dict[str,Any]]:
    values={}
    paths=tuple((directory/'claims').iterdir())
    if len(paths)>plan['max_slots']:
        raise ValueError('CLOCK_CLAIM_LIMIT_EXCEEDED')
    sha=_sha(directory/'plan.json')
    for path in paths:
        if re.fullmatch(r'[0-9]{5}\.json',path.name) is None or path.is_symlink():
            raise ValueError('CLOCK_CLAIM_NAME_INVALID')
        item=monitor_archive.read(path)
        index=int(path.stem)
        if set(item)!={'index','plan_sha256','claimed_at','source_sha256'} or type(item['index']) is not int or (
                item['index']!=index or not 0<=index<plan['max_slots'] or item['plan_sha256']!=sha or
                re.fullmatch(r'[0-9a-f]{64}',item['source_sha256']) is None):
            raise ValueError('CLOCK_CLAIM_INVALID')
        when=monitor_archive.at(item['claimed_at'])
        due=monitor_archive.at(plan['started_at'])+timedelta(seconds=index*600)
        if not due<=when<min(due+timedelta(seconds=120),monitor_archive.at(plan['ends_at'])):
            raise ValueError('CLOCK_CLAIM_TIME_INVALID')
        values[index]=item
    return values


def run_slot(directory: Path, plan_sha: str, source_sha: str, *,
             clock: Callable[[],datetime]=utc_now,
             collect: Callable[[int],dict[str,Any]]=clock_health.check,
             free_bytes: Callable[[Path],int]=lambda p:shutil.disk_usage(p).free) -> int:
    if source_sha!=implementation_fingerprint():
        raise ValueError('CLOCK_RUNNER_SOURCE_CHANGED')
    plan=load_plan(directory,plan_sha)
    start,end=monitor_archive.at(plan['started_at']),monitor_archive.at(plan['ends_at'])
    with file_lock(directory/'slot.lock'):
        now=utc_timestamp(clock())
        if (directory/'stop').exists() or (Path(plan['monitor_directory'])/'stop').exists() or (
                now<start or now+timedelta(seconds=8)>=end):
            return 0
        index=int((now-start).total_seconds()//600)
        claims=_claims(directory,plan)
        # A backward clock or corrupt gaps never replays old/missing slots.
        if claims and index<=max(claims) or not 0<=index<plan['max_slots']:
            return 0
        due=start+timedelta(seconds=index*600)
        if now>=due+timedelta(seconds=120):
            return 0
        if free_bytes(directory)<MIN_FREE_BYTES:
            raise ValueError('CLOCK_DISK_RESERVE_REQUIRED')
        claim={'index':index,'plan_sha256':plan_sha,'claimed_at':now,'source_sha256':source_sha}
        monitor_archive.write(directory/'claims'/f'{index:05d}.json',claim,exclusive=True)
        # Claim before public I/O; stop/time/source races consume but don't replay.
        now=utc_timestamp(clock())
        if (directory/'stop').exists() or (Path(plan['monitor_directory'])/'stop').exists() or (
            not due<=now<due+timedelta(seconds=120) or now+timedelta(seconds=8)>=end or
            implementation_fingerprint()!=source_sha):
            return 0
        try:
            report=collect(TIMEOUT_SECONDS)
            expected={'schema_version','evaluated_at','status','attention_required','reason_codes','samples',
                'server_minus_local_intervals','diagnostic_bound_us','max_round_trip_ns','max_wall_step_ns',
                'clock_modified','credentials_read','orders_submitted','market_quality_verified','qualification_issued'}
            if set(report)!=expected or len(exact_json(report).encode())>MAX_REPORT_BYTES or report.get('diagnostic_bound_us')!=250000 or (
                any(report.get(key) is not False for key in ('clock_modified','credentials_read','orders_submitted',
                    'market_quality_verified','qualification_issued')) or report.get('status') not in set(clock_health.Health)):
                raise ValueError('CLOCK_REPORT_SCOPE_INVALID')
        except Exception:
            report={'evaluated_at':utc_timestamp(clock()),'status':'UNVERIFIED','reason_codes':['CLOCK_DIAGNOSTIC_FAILED'],
                'clock_modified':False,'credentials_read':False,'orders_submitted':False,
                'market_quality_verified':False,'qualification_issued':False,'diagnostic_bound_us':250000}
        receipt={'index':index,'plan_sha256':plan_sha,'source_sha256':source_sha,'report':report}
        monitor_archive.write(directory/'receipts'/f'{index:05d}.json',receipt,exclusive=True)
        previous=monitor_archive.read(directory/'latest.json') if (directory/'latest.json').exists() else None
        signal={'status':report['status'],'reason_codes':report.get('reason_codes',[])}
        if previous is None or signal!=previous['signal']:
            monitor_archive.write(directory/f'event-{index:05d}.json',
                {'index':index,'observed_at':utc_timestamp(clock()),'signal':signal},exclusive=True)
        monitor_archive.write(directory/'latest.json',{'index':index,'signal':signal,'receipt_sha256':_sha(directory/'receipts'/f'{index:05d}.json')})
        return 0


def status(directory: Path, *, clock: Callable[[],datetime]=utc_now) -> dict[str, Any]:
    plan=load_plan(directory)
    claims=_claims(directory,plan)
    now=utc_timestamp(clock())
    elapsed=max(0,min(plan['max_slots'],int((now-monitor_archive.at(plan['started_at'])).total_seconds()//600)))
    result={'status':'CLOCK_SCHEDULE_READ','evaluated_at':now,'ends_at':plan['ends_at'],
        'slots_consumed':len(claims),'max_slots':plan['max_slots'],'highest_claimed_slot':max(claims) if claims else None,
        'unobserved_elapsed_slots':len(set(range(elapsed))-set(claims)),
        'stop_requested':(directory/'stop').exists(),'diagnostic_status':'UNVERIFIED','latest_slot':None,
        'clock_modified':False,'credentials_read':False,'orders_submitted':False,'qualification_issued':False}
    if (directory/'latest.json').exists():
        latest=monitor_archive.read(directory/'latest.json')
        index=latest['index']
        path=directory/'receipts'/f'{index:05d}.json'
        receipt=monitor_archive.read(path)
        if type(index) is not int or index not in claims or _sha(path)!=latest['receipt_sha256'] or (
            receipt['index']!=index or receipt['plan_sha256']!=_sha(directory/'plan.json') or
            receipt['source_sha256']!=claims[index]['source_sha256']):
            raise ValueError('CLOCK_LATEST_RECEIPT_INVALID')
        report=receipt['report']
        if not isinstance(report,dict) or report.get('status') not in set(clock_health.Health) or any(
                report.get(key) is not False for key in ('clock_modified','credentials_read','orders_submitted',
                    'market_quality_verified','qualification_issued')):
            raise ValueError('CLOCK_RECEIPT_SCOPE_INVALID')
        age=(now-monitor_archive.at(report['evaluated_at'])).total_seconds()
        source_current=receipt['source_sha256']==implementation_fingerprint()
        fresh=0<=age<=clock_health.MAX_SAMPLE_AGE.total_seconds() and now<monitor_archive.at(plan['ends_at']) and (
            source_current)
        current_status='UNVERIFIED'
        if 'samples' in report:
            raw=report['samples']
            if not isinstance(raw,list) or len(raw)!=2:
                raise ValueError('CLOCK_RECEIPT_SAMPLES_INVALID')
            samples=[]
            for item in raw:
                if not isinstance(item,dict) or set(item)!={'endpoint','started_at','finished_at','elapsed_ns',
                        'server_time','http_status','failure'}:
                    raise ValueError('CLOCK_RECEIPT_SAMPLE_INVALID')
                samples.append(clock_health.Sample(PublicEndpoint(item['endpoint']),
                    monitor_archive.at(item['started_at']),monitor_archive.at(item['finished_at']),item['elapsed_ns'],
                    None if item['server_time'] is None else monitor_archive.at(item['server_time']),item['http_status'],
                    None if item['failure'] is None else clock_health.Failure(item['failure'])))
            pair=(samples[0],samples[1])
            historical=clock_health.assess(pair,evaluated_at=monitor_archive.at(report['evaluated_at']))
            if exact_json(historical)!=exact_json(report):
                raise ValueError('CLOCK_RECEIPT_ASSESSMENT_MISMATCH')
            reassessed=clock_health.assess(pair,evaluated_at=now)
            if any(now-sample.started_at>clock_health.MAX_SAMPLE_AGE or sample.finished_at>now for sample in samples):
                fresh=False
            if fresh:
                current_status=reassessed['status']
        elif report.get('reason_codes')!=['CLOCK_DIAGNOSTIC_FAILED'] or report['status']!='UNVERIFIED':
            raise ValueError('CLOCK_RECEIPT_FAILURE_INVALID')
        result.update(latest_slot=index,diagnostic_status=current_status,
            latest_sample_fresh=fresh,latest_observation_status=report['status'],
            latest_report_age_us=clock_health._microseconds(now-monitor_archive.at(report['evaluated_at'])),
            latest_report=report,diagnostic_owner_live_not_proved=True)
    return result


def prepare_backend(directory: Path, output: Path) -> None:
    plan=load_plan(directory)
    output=_within(output,Path(plan['root']))
    if utc_now()+timedelta(seconds=8)>=monitor_archive.at(plan['ends_at']):
        raise ValueError('CLOCK_BACKEND_WINDOW_ENDED')
    runner={'root':plan['root'],'clock_directory':str(directory.resolve()),'plan_sha256':_sha(directory/'plan.json'),
        'source_sha256':implementation_fingerprint(),'ends_at':plan['ends_at'],
        'task_name':'CryptoArbitrageBot_Clock_'+uuid4().hex}
    output.mkdir(parents=True,exist_ok=False)
    monitor_archive.write(output/'runner.json',runner,exclusive=True)
    runner_sha=_sha(output/'runner.json')
    launch='''import hashlib,json,os,subprocess,sys
from pathlib import Path
try:
    directory=Path(__file__).resolve().parent
    body=(directory/"runner.json").read_bytes()
    if len(body)>16384 or hashlib.sha256(body).hexdigest()!="RUNNER_SHA":
        raise SystemExit(2)
    value=json.loads(body)
    root=Path(value["root"])
    result=subprocess.run([str(root/".venv/Scripts/python.exe"),"-m","crypto_arbitrage_bot.clock_schedule",
        "run-slot","--directory",value["clock_directory"],"--expected-plan-sha256",value["plan_sha256"],
        "--expected-source-sha256",value["source_sha256"]],cwd=root,
        env={**os.environ,"PYTHONPATH":str(root/"src")},stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    raise SystemExit(result.returncode)
except Exception:
    raise SystemExit(2)
'''
    header=r'''$ErrorActionPreference='Stop'
$clockDir=Split-Path -Parent $PSCommandPath
if((Get-FileHash -LiteralPath (Join-Path $clockDir 'runner.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'RUNNER_SHA'){throw 'CLOCK_RUNNER_CHANGED'}
$clockData=Get-Content -LiteralPath (Join-Path $clockDir 'runner.json') -Raw|ConvertFrom-Json
$clockService=New-Object -ComObject 'Schedule.Service';$clockService.Connect()
'''
    install=header+r'''$clockDefinition=$clockService.NewTask(0)
$clockDefinition.Principal.UserId=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$clockDefinition.Principal.LogonType=3;$clockDefinition.Principal.RunLevel=0
$clockDefinition.Settings.Hidden=$true;$clockDefinition.Settings.MultipleInstances=2
$clockDefinition.Settings.ExecutionTimeLimit='PT30S';$clockDefinition.Settings.StartWhenAvailable=$false
$clockDefinition.Settings.DisallowStartIfOnBatteries=$false;$clockDefinition.Settings.StopIfGoingOnBatteries=$false
$clockTrigger=$clockDefinition.Triggers.Create(1)
$clockTrigger.StartBoundary=[DateTime]::UtcNow.AddSeconds(3).ToString('o')
if($clockData.ends_at -is [DateTime]){$clockEnd=([DateTime]$clockData.ends_at).ToUniversalTime().ToString('o')}else{$clockEnd=[DateTimeOffset]::Parse([string]$clockData.ends_at).UtcDateTime.ToString('o')}
$clockTrigger.EndBoundary=$clockEnd;$clockTrigger.Repetition.Interval='PT1M'
$clockTrigger.Repetition.StopAtDurationEnd=$true
$clockTrigger.Repetition.Duration='PT'+[math]::Ceiling(([DateTimeOffset]::Parse($clockEnd).UtcDateTime-[DateTime]::UtcNow).TotalSeconds)+'S'
$clockAction=$clockDefinition.Actions.Create(0)
$clockAction.Path=Join-Path $clockData.root '.venv\Scripts\pythonw.exe'
$clockAction.Arguments='"'+(Join-Path $clockDir 'launch.py')+'"';$clockAction.WorkingDirectory=$clockData.root
$clockRegistered=$clockService.GetFolder('\').RegisterTaskDefinition($clockData.task_name,$clockDefinition,2,$null,$null,3)
[IO.File]::WriteAllText((Join-Path $clockDir 'registered-task.xml'),$clockRegistered.Xml,[Text.UTF8Encoding]::new($false))
[ordered]@{status='CLOCK_REGISTERED';task_name=$clockData.task_name;orders_submitted=$false}|ConvertTo-Json -Compress
'''
    check=header+r'''$clockTask=$clockService.GetFolder('\').GetTask($clockData.task_name)
if($clockTask.Xml -cne [IO.File]::ReadAllText((Join-Path $clockDir 'registered-task.xml'))){throw 'CLOCK_DEFINITION_CHANGED'}
'''
    state=check+r'''[ordered]@{status='CLOCK_DEFINITION_VERIFIED';task_name=$clockData.task_name;state=$clockTask.State;last_result=$clockTask.LastTaskResult;orders_submitted=$false}|ConvertTo-Json -Compress
'''
    remove=check+r'''if($clockTask.State -eq 4){throw 'CLOCK_CURRENT_TASK_RUNNING'}
$clockService.GetFolder('\').DeleteTask($clockData.task_name,0)
[ordered]@{status='CLOCK_DEFINITION_REMOVED';task_name=$clockData.task_name;orders_submitted=$false}|ConvertTo-Json -Compress
'''
    for name,body in [('launch.py',launch),('install.ps1',install),('status.ps1',state),('remove.ps1',remove)]:
        (output/name).write_text(body.replace('RUNNER_SHA',runner_sha),encoding='utf-8')


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description='Finite original-scope PUBLIC clock diagnostics; no time changes')
    parser.add_argument('command',choices=('prepare','prepare-backend','run-slot','status','stop'))
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--monitor-dir',type=Path)
    parser.add_argument('--original-monitor-sha256')
    parser.add_argument('--backend-dir',type=Path)
    parser.add_argument('--expected-plan-sha256')
    parser.add_argument('--expected-source-sha256')
    args=parser.parse_args(argv)
    try:
        directory=args.directory.resolve()
        if args.command=='prepare':
            if args.monitor_dir is None or args.original_monitor_sha256 is None:
                raise ValueError('ORIGINAL_CLOCK_MONITOR_REQUIRED')
            prepare(directory,args.monitor_dir,args.original_monitor_sha256)
            result={'status':'CLOCK_PREPARED_NOT_REGISTERED'}
        elif args.command=='prepare-backend':
            if args.backend_dir is None:
                raise ValueError('NEW_CLOCK_BACKEND_REQUIRED')
            prepare_backend(directory,args.backend_dir)
            result={'status':'CLOCK_BACKEND_PREPARED_NOT_REGISTERED'}
        elif args.command=='run-slot':
            if args.expected_plan_sha256 is None or args.expected_source_sha256 is None:
                raise ValueError('PINNED_CLOCK_RUNNER_REQUIRED')
            return run_slot(directory,args.expected_plan_sha256,args.expected_source_sha256)
        elif args.command=='stop':
            load_plan(directory)
            (directory/'stop').touch(exist_ok=True)
            result={'status':'CLOCK_STOP_REQUESTED'}
        else:
            result=status(directory)
        print(exact_json(result))
        return 0
    except (OSError,ValueError,TypeError,KeyError,LockBusyError):
        print(exact_json({'status':'CLOCK_SCHEDULE_UNAVAILABLE','clock_modified':False,'orders_submitted':False}))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
