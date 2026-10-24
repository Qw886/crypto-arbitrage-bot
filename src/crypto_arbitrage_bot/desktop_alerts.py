"""Finite desktop-only operational alerts; no network, orders or qualification."""

import argparse
import hashlib
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from crypto_arbitrage_bot import monitor_archive as archive
from crypto_arbitrage_bot.execution.risk_review import implementation_fingerprint
from crypto_arbitrage_bot.operational_report import quality_signal
from crypto_arbitrage_bot.runtime_status import _owner
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.file_lock import file_lock, lock_is_held
from crypto_arbitrage_bot.utils.process_probe import observe_process
from crypto_arbitrage_bot.utils.windows_notifications import notification_available, notify

INTERVAL = 60
MAX_NOTICES = 64
_CODES = {'MONITOR_UNCONFIRMED', 'REPORT_UNAVAILABLE', 'RESEARCH_ATTENTION', 'SAVED_DATA_INVALID',
          'EXECUTION_ATTENTION'}
_COUNTERS = ('failed_validations', 'collection_failures', 'interrupted_attempts')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _work(path: Path, root: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(root / 'work') or path == root / 'work':
        raise ValueError('NEW_WORK_ALERT_DIRECTORY_REQUIRED')
    return path


def prepare(directory: Path, monitor: Path, original_sha: str, *,
            clock: Callable[[], datetime] = utc_now) -> dict[str, Any]:
    root = Path.cwd().resolve()
    directory = _work(directory, root)
    monitor = _work(monitor, root)
    if directory == monitor or directory in monitor.parents or monitor in directory.parents:
        raise ValueError('SEPARATE_ALERT_DIRECTORY_REQUIRED')
    original, _ = archive.load_plan(monitor, original_sha)
    archive.load_binding(monitor, original_sha)
    now = utc_timestamp(clock())
    end = archive.at(original['ends_at'])
    if not archive.at(original['started_at']) <= now < end - timedelta(seconds=20):
        raise ValueError('ORIGINAL_ALERT_WINDOW_ENDED')
    remaining_us = (end-now) // timedelta(microseconds=1)
    plan = {'root': str(root), 'monitor_directory': str(monitor), 'original_monitor_sha256': original_sha,
            'binding_sha256': _sha(monitor/'monitor-binding.json'), 'started_at': now.isoformat(),
            'ends_at': original['ends_at'], 'interval_seconds': INTERVAL,
            'max_slots': (remaining_us + INTERVAL*1_000_000-1)//(INTERVAL*1_000_000),
            'max_notices': MAX_NOTICES, 'orders_submitted': False, 'credentials_read': False}
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'claims').mkdir()
    (directory/'receipts').mkdir()
    archive.write(directory/'plan.json', plan, exclusive=True)
    archive.write(directory/'cursor.json', {'slot': -1}, exclusive=True)
    return {'status': 'DESKTOP_ALERT_PLAN_PREPARED', 'plan_sha256': _sha(directory/'plan.json'),
            'original_deadline': plan['ends_at'], 'max_notices': MAX_NOTICES, 'orders_submitted': False}


def load_plan(directory: Path, expected_sha: str | None = None) -> dict[str, Any]:
    plan = archive.read(directory/'plan.json')
    fields = {'root', 'monitor_directory', 'original_monitor_sha256', 'binding_sha256', 'started_at',
              'ends_at', 'interval_seconds', 'max_slots', 'max_notices', 'orders_submitted', 'credentials_read'}
    if set(plan) != fields or expected_sha is not None and _sha(directory/'plan.json') != expected_sha:
        raise ValueError('ALERT_PLAN_DIFFERS')
    root = Path(plan['root']).resolve()
    if root != Path.cwd().resolve() or _work(directory, root) != directory.resolve():
        raise ValueError('ALERT_ROOT_DIFFERS')
    monitor = _work(Path(plan['monitor_directory']), root)
    original, _ = archive.load_plan(monitor, plan['original_monitor_sha256'])
    archive.load_binding(Path(plan['monitor_directory']), plan['original_monitor_sha256'])
    remaining_us = (archive.at(plan['ends_at'])-archive.at(plan['started_at'])) // timedelta(microseconds=1)
    if (not 0 < remaining_us <= 8*86400*1_000_000 or plan['ends_at'] != original['ends_at']
            or archive.at(plan['started_at']) < archive.at(original['started_at'])
            or plan['interval_seconds'] != INTERVAL or plan['max_notices'] != MAX_NOTICES
            or type(plan['max_slots']) is not int
            or plan['max_slots'] != (remaining_us+INTERVAL*1_000_000-1)//(INTERVAL*1_000_000)
            or plan['orders_submitted'] is not False or plan['credentials_read'] is not False
            or _sha(Path(plan['monitor_directory'])/'monitor-binding.json') != plan['binding_sha256']):
        raise ValueError('ALERT_ORIGINAL_SCOPE_DIFFERS')
    return plan


def observe(plan: dict[str, Any], now: datetime) -> dict[str, Any]:
    """Actual native owner plus a fresh bounded original monitor report."""
    monitor = Path(plan['monitor_directory'])
    original, sha = archive.load_plan(monitor, plan['original_monitor_sha256'])
    binding = archive.load_binding(monitor, sha)
    progress = archive.read(monitor/'progress.json')
    if (progress.get('plan_sha256') != sha or progress.get('mode') not in ('LEGACY', 'RESUMED')
            or type(progress.get('worker_pid')) is not int or type(progress.get('checks')) is not int
            or not 0 <= progress['checks'] <= original['max_checks']
            or progress.get('state') not in ('RUNNING', 'FINISHED', 'STOPPED', 'INTERRUPTED')):
        raise ValueError('ALERT_MONITOR_PROGRESS_INVALID')
    created = progress['worker_created_at']
    if created is None:
        if progress['mode']!='LEGACY' or progress['worker_pid']!=binding['legacy_pid']:
            raise ValueError('ALERT_MONITOR_BIRTH_UNKNOWN')
        created = binding['legacy_created_at']
    birth = archive.at(created)
    matches = (lambda a: archive.matches_resumed(a, monitor, sha)) if progress['mode']=='RESUMED' else (
        lambda a: archive.matches_legacy(a, monitor, binding['spec'], original))
    def probe(pid: int, **kwargs: Any) -> Any:
        return observe_process(pid, matches=matches, not_before=birth, not_after=birth)
    owner = _owner(monitor/'monitor.lock', progress['worker_pid'], birth, archive.at(progress['updated_at']),
        now, timedelta(seconds=binding['spec']['max_heartbeat_age_seconds']), matches,
        probe, lock_is_held, progress['state'])
    health = [] if owner['status']=='RUNNING_CONFIRMED' else ['MONITOR_UNCONFIRMED']
    report = archive.read(monitor/'latest.json')
    if not timedelta(0) <= now-archive.at(report['evaluated_at']) <= timedelta(
            seconds=binding['spec']['max_heartbeat_age_seconds']):
        health.append('REPORT_UNAVAILABLE')
    research = report.get('research')
    if not isinstance(research, dict):
        raise ValueError('ALERT_RESEARCH_REPORT_REQUIRED')
    if research.get('status') not in ('RUNNING_CONFIRMED', 'FINISHED_NO_OWNER'):
        health.append('RESEARCH_ATTENTION')
    db = research.get('database', {})
    quality = quality_signal(db.get('recent_saved_quality'))
    if quality['status']=='UNAVAILABLE' or quality['invalid']:
        health.append('SAVED_DATA_INVALID')
    tasks = db.get('tasks')
    failed = None
    if (db.get('status')=='READ' and isinstance(tasks, dict)
            and not set(tasks)-{'COMPLETED','FAILED','PENDING','RUNNING'}
            and all(type(v) is int and 0 <= v <= 2**63-1 for v in tasks.values())):
        failed = tasks.get('FAILED', 0)
    counts = research.get('collection_counters', {})
    counters = {'failed_validations': failed, 'collection_failures': counts.get('failed'),
                'interrupted_attempts': counts.get('interrupted')}
    if failed is None:
        health.append('REPORT_UNAVAILABLE')
    execution = report.get('testnet')
    if execution is not None and (not isinstance(execution, dict) or execution.get('status')!='READ'
            or not isinstance(execution.get('active_attempts'), list) or execution['active_attempts']):
        health.append('EXECUTION_ATTENTION')
    if any(v is not None and (type(v) is not int or not 0 <= v <= 2**63-1) for v in counters.values()):
        raise ValueError('ALERT_COUNTER_INVALID')
    return {'health': sorted(set(health)), 'counters': counters}


def transition(current: dict[str, Any], previous: dict[str, Any] | None) -> tuple[str, str] | None:
    """Healthy initial baseline is quiet; historical failures are not new events."""
    health = current['health']
    counters = current['counters']
    if (not isinstance(health, list) or any(code not in _CODES for code in health)
            or set(counters) != set(_COUNTERS)
            or any(v is not None and (type(v) is not int or not 0 <= v <= 2**63-1) for v in counters.values())):
        raise ValueError('STRUCTURED_ALERT_OBSERVATION_REQUIRED')
    if previous is not None:
        transition(previous, None)  # Validate previous structured observation too.
    if health and (previous is None or health != previous['health']):
        labels = {'MONITOR_UNCONFIRMED': '监督进程或心跳异常', 'REPORT_UNAVAILABLE': '状态报告过期或不可读',
                  'RESEARCH_ATTENTION': '研究任务需要检查', 'SAVED_DATA_INVALID': '已保存样本存在质量问题',
                  'EXECUTION_ATTENTION': '本地执行账本存在待处理或未知状态'}
        return '套利机器人需要检查', '；'.join(labels[c] for c in health) + '。请查看项目状态；未自动交易。'
    if previous is not None and not health and previous['health']:
        return '套利机器人运行状态恢复', '本次检查的故障状态已恢复；这不表示策略盈利或交易验收通过。'
    if previous is not None and any(counters[k] is not None and previous['counters'][k] is not None
            and counters[k] > previous['counters'][k] for k in _COUNTERS):
        return '套利机器人出现新失败', '新检查记录了采集失败、中断或错过验证窗口。失败已保留，请查看本地验收报告。'
    return None


def _claims(directory: Path) -> int:
    names = sorted(p.name for p in (directory/'claims').iterdir())
    if len(names)>MAX_NOTICES or names != [f'{i:04d}.json' for i in range(len(names))]:
        raise ValueError('ALERT_CLAIM_LEDGER_UNKNOWN')
    previous_slot = -1
    for index, name in enumerate(names):
        value = archive.read(directory/'claims'/name)
        if (set(value)!={'index','slot','plan_sha256','source_sha256','claimed_at'}
                or type(value['index']) is not int or value['index']!=index or type(value['slot']) is not int
                or value['slot']<=previous_slot or value['plan_sha256']!=_sha(directory/'plan.json')
                or not isinstance(value['source_sha256'],str)
                or re.fullmatch('[0-9a-f]{64}',value['source_sha256']) is None):
            raise ValueError('ALERT_CLAIM_LEDGER_INVALID')
        archive.at(value['claimed_at'])
        previous_slot = value['slot']
    return len(names)


def run_slot(directory: Path, plan_sha: str, source_sha: str, *,
             clock: Callable[[], datetime] = utc_now,
             observer: Callable[[dict[str, Any], datetime], dict[str, Any]] = observe,
             sender: Callable[[str, str], dict[str, bool]] = notify,
             availability: Callable[[], bool] = notification_available) -> dict[str, Any]:
    directory = directory.resolve()
    plan = load_plan(directory, plan_sha)
    with file_lock(directory/'alert.lock'):
        now = utc_timestamp(clock())
        monitor = Path(plan['monitor_directory'])
        def allowed() -> bool:
            at = utc_timestamp(clock())
            load_plan(directory, plan_sha)  # Policy/binding/UTC are rechecked before native I/O.
            return (archive.at(plan['started_at']) <= at < archive.at(plan['ends_at'])-timedelta(seconds=20)
                    and not (directory/'stop').exists() and not (monitor/'stop').exists()
                    and source_sha == implementation_fingerprint())
        if not allowed():
            return {'status':'SKIPPED','reason':'ALERT_ORIGINAL_GATE_CLOSED','orders_submitted':False}
        slot = (now-archive.at(plan['started_at'])) // timedelta(seconds=INTERVAL)
        cursor = archive.read(directory/'cursor.json')
        if set(cursor)!={'slot'} or type(cursor['slot']) is not int or not -1<=cursor['slot']<plan['max_slots']:
            raise ValueError('ALERT_CURSOR_UNKNOWN')
        if slot<=cursor['slot'] or slot>=plan['max_slots']:
            return {'status':'SKIPPED','reason':'ALERT_SLOT_CONSUMED_OR_CLOCK_REVERSED','orders_submitted':False}
        used = _claims(directory)
        if used:
            highest = archive.read(directory/'claims'/f'{used-1:04d}.json')['slot']
            if not highest < plan['max_slots'] or cursor['slot'] < highest:
                raise ValueError('ALERT_CURSOR_ROLLBACK_OR_CLAIM_OUT_OF_RANGE')
        previous_path = directory/'observation.json'
        previous = archive.read(previous_path) if previous_path.exists() else None
        if previous is None and used:
            raise ValueError('ALERT_OBSERVATION_MISSING_AFTER_CLAIM')
        archive.write(directory/'cursor.json', {'slot':slot})  # Consume before native probe/notification.
        try:
            current = observer(plan, now)
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            current = {'health':['REPORT_UNAVAILABLE'], 'counters':dict.fromkeys(_COUNTERS)}
        if previous is not None and any(current['counters'].get(k) is not None
                and previous['counters'].get(k) is not None
                and current['counters'][k] < previous['counters'][k] for k in _COUNTERS):
            current = {'health':sorted(set(current['health']) | {'REPORT_UNAVAILABLE'}),
                       'counters':previous['counters']}
        message = transition(current, previous)
        if message is None or used>=MAX_NOTICES or not allowed():
            archive.write(previous_path, current)
            return {'status':'QUIET' if used<MAX_NOTICES else 'NOTICE_BUDGET_EXHAUSTED',
                    'slot':slot,'notices_consumed':used,'orders_submitted':False}
        if not availability():
            archive.write(directory/'deferred.json', {'slot':slot,'evaluated_at':utc_timestamp(clock()),
                'reason':'OS_BUSY_OR_UNKNOWN','native_api_attempted':False})
            return {'status':'DEFERRED_OS_BUSY_OR_UNKNOWN','slot':slot,'notices_consumed':used,
                    'orders_submitted':False}
        claim = {'index':used,'slot':slot,'plan_sha256':plan_sha,'source_sha256':source_sha,'claimed_at':utc_timestamp(clock())}
        archive.write(directory/'claims'/f'{used:04d}.json', claim, exclusive=True)
        # Persist baseline before I/O: uncertain delivery never creates an automatic replay.
        archive.write(previous_path, current)
        result: dict[str, Any] = {'index':used,'slot':slot,'status':'NOT_ATTEMPTED',
                                  'orders_submitted':False,'qualification_issued':False}
        if allowed():
            try:
                native = sender(*message)
                fields = {'api_attempted','api_accepted','shell_reported_shown','user_clicked','shell_reported_hidden',
                          'shell_reported_timeout','user_read_verified'}
                if (set(native)!=fields or any(type(v) is not bool for v in native.values())
                        or native['user_read_verified'] is not False
                        or not native['api_attempted'] and any(v for k,v in native.items() if k!='api_attempted')):
                    raise ValueError('STRUCTURED_NATIVE_NOTIFICATION_RESULT_REQUIRED')
                result.update({'status':'ATTEMPTED','native_result':native})
                if not native['api_attempted']:
                    result['status'] = 'OS_BLOCKED_NO_API'
                    # No UI API was called: preserve the unsent transition for a fresh later check.
                    # The already claimed attempt remains consumed, never refunded.
                    archive.write(previous_path, previous if previous is not None else {
                        'health':[], 'counters':dict.fromkeys(_COUNTERS)})
            except Exception as error:
                kind = type(error).__name__
                result.update({'status':'NOTIFICATION_FAILED','error_type':kind if kind in
                    ('OSError','ValueError','TypeError','RuntimeError','TimeoutError') else 'OTHER_ERROR'})
        archive.write(directory/'receipts'/f'{used:04d}.json', result, exclusive=True)
        archive.write(directory/'latest.json', {**result,'evaluated_at':utc_timestamp(clock()),'notices_consumed':used+1})
        return result


def status(directory: Path) -> dict[str, Any]:
    plan = load_plan(directory)
    used = _claims(directory)
    latest = archive.read(directory/'latest.json') if (directory/'latest.json').exists() else None
    receipts = {}
    for path in (directory/'receipts').iterdir():
        if re.fullmatch(r'\d{4}\.json', path.name) is None or int(path.stem)>=used:
            raise ValueError('UNKNOWN_ALERT_RECEIPT')
        receipt = archive.read(path)
        claim = archive.read(directory/'claims'/path.name)
        fields = {'index','slot','status','orders_submitted','qualification_issued'}
        if receipt.get('status') in ('ATTEMPTED','OS_BLOCKED_NO_API'):
            fields.add('native_result')
            native = receipt.get('native_result')
            native_fields = {'api_attempted','api_accepted','shell_reported_shown','user_clicked',
                             'shell_reported_hidden','shell_reported_timeout','user_read_verified'}
            if (not isinstance(native,dict) or set(native)!=native_fields
                    or any(type(v) is not bool for v in native.values()) or native['user_read_verified'] is not False):
                raise ValueError('INVALID_NATIVE_ALERT_RECEIPT')
        elif receipt.get('status')=='NOTIFICATION_FAILED':
            fields.add('error_type')
            if receipt.get('error_type') not in ('OSError','ValueError','TypeError','RuntimeError','TimeoutError','OTHER_ERROR'):
                raise ValueError('INVALID_ALERT_FAILURE_TYPE')
        if (set(receipt)!=fields or type(receipt.get('index')) is not int or receipt['index']!=int(path.stem)
                or type(receipt.get('slot')) is not int or receipt['slot']!=claim['slot'] or receipt.get('status') not in
                ('ATTEMPTED','NOT_ATTEMPTED','NOTIFICATION_FAILED','OS_BLOCKED_NO_API')
                or receipt.get('orders_submitted') is not False or receipt.get('qualification_issued') is not False):
            raise ValueError('INVALID_ALERT_RECEIPT')
        receipts[int(path.stem)] = receipt
    if latest is not None:
        index = latest.get('index')
        if (type(index) is not int or index not in receipts or latest.get('notices_consumed')!=index+1
                or {k:v for k,v in latest.items() if k not in ('evaluated_at','notices_consumed')}!=receipts[index]):
            raise ValueError('ALERT_LATEST_RECEIPT_DIFFERS')
        archive.at(latest['evaluated_at'])
    missing = used-len(receipts)
    return {'status':'NOTIFICATION_EVIDENCE_INCOMPLETE' if missing else 'DESKTOP_ALERT_ARCHIVE_READ',
            'notices_consumed':used,'missing_receipts':missing,'max_notices':MAX_NOTICES,
            'original_deadline':plan['ends_at'],'latest_attempt':latest,
            'last_os_deferral':archive.read(directory/'deferred.json') if (directory/'deferred.json').exists() else None,
            'user_read_verified':False,
            'orders_submitted':False,'qualification_issued':False}


def prepare_backend(directory: Path, output: Path) -> dict[str, Any]:
    plan = load_plan(directory)
    output = _work(output, Path(plan['root']))
    if utc_now() + timedelta(seconds=20) >= archive.at(plan['ends_at']):
        raise ValueError('ALERT_BACKEND_WINDOW_ENDED')
    runner = {'root':plan['root'],'alert_directory':str(directory.resolve()),
              'plan_sha256':_sha(directory/'plan.json'),'source_sha256':implementation_fingerprint(),
              'ends_at':plan['ends_at'],'task_name':'CryptoArbitrageBot_Alerts_'+uuid4().hex}
    output.mkdir(parents=True, exist_ok=False)
    archive.write(output/'runner.json', runner, exclusive=True)
    runner_sha = _sha(output/'runner.json')
    launch = '''import hashlib,json,os,subprocess
from pathlib import Path
try:
    d=Path(__file__).resolve().parent
    body=(d/'runner.json').read_bytes()
    if len(body)>16384 or hashlib.sha256(body).hexdigest()!='RUNNER_SHA':
        raise SystemExit(2)
    v=json.loads(body);root=Path(v['root'])
    p=subprocess.run([str(root/'.venv/Scripts/python.exe'),'-m','crypto_arbitrage_bot.desktop_alerts',
        'run-slot','--directory',v['alert_directory'],'--expected-plan-sha256',v['plan_sha256'],
        '--expected-source-sha256',v['source_sha256']],cwd=root,
        env={**os.environ,'PYTHONPATH':str(root/'src')},stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    raise SystemExit(p.returncode)
except Exception:
    raise SystemExit(2)
'''
    header = r'''$ErrorActionPreference='Stop'
$alertDir=Split-Path -Parent $PSCommandPath
if((Get-FileHash -LiteralPath (Join-Path $alertDir 'runner.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'RUNNER_SHA'){throw 'ALERT_RUNNER_CHANGED'}
$alertData=Get-Content -LiteralPath (Join-Path $alertDir 'runner.json') -Raw|ConvertFrom-Json
$alertService=New-Object -ComObject 'Schedule.Service';$alertService.Connect()
'''
    install = header + r'''$alertDef=$alertService.NewTask(0)
$alertDef.Principal.UserId=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$alertDef.Principal.LogonType=3;$alertDef.Principal.RunLevel=0
$alertDef.Settings.Hidden=$true;$alertDef.Settings.MultipleInstances=2
$alertDef.Settings.ExecutionTimeLimit='PT30S';$alertDef.Settings.StartWhenAvailable=$false
$alertDef.Settings.DisallowStartIfOnBatteries=$false;$alertDef.Settings.StopIfGoingOnBatteries=$false
$alertTrigger=$alertDef.Triggers.Create(1)
$alertTrigger.StartBoundary=[DateTime]::UtcNow.AddSeconds(3).ToString('o')
if($alertData.ends_at -is [DateTime]){$alertEnd=([DateTime]$alertData.ends_at).ToUniversalTime().ToString('o')}else{$alertEnd=[DateTimeOffset]::Parse([string]$alertData.ends_at).UtcDateTime.ToString('o')}
$alertTrigger.EndBoundary=$alertEnd;$alertTrigger.Repetition.Interval='PT1M'
$alertTrigger.Repetition.Duration='PT'+[math]::Ceiling(([DateTimeOffset]::Parse($alertEnd).UtcDateTime-[DateTime]::UtcNow).TotalSeconds)+'S'
$alertTrigger.Repetition.StopAtDurationEnd=$true
$alertAction=$alertDef.Actions.Create(0)
$alertAction.Path=Join-Path $alertData.root '.venv\Scripts\pythonw.exe'
$alertAction.Arguments='"'+(Join-Path $alertDir 'launch.py')+'"';$alertAction.WorkingDirectory=$alertData.root
$alertRegistered=$alertService.GetFolder('\').RegisterTaskDefinition($alertData.task_name,$alertDef,2,$null,$null,3)
[IO.File]::WriteAllText((Join-Path $alertDir 'registered-task.xml'),$alertRegistered.Xml,[Text.UTF8Encoding]::new($false))
[ordered]@{status='DESKTOP_ALERT_REGISTERED';task_name=$alertData.task_name;orders_submitted=$false}|ConvertTo-Json -Compress
'''
    check = header + r'''$alertTask=$alertService.GetFolder('\').GetTask($alertData.task_name)
if($alertTask.Xml -cne [IO.File]::ReadAllText((Join-Path $alertDir 'registered-task.xml'))){throw 'ALERT_DEFINITION_CHANGED'}
'''
    show = check + r'''[ordered]@{status='DESKTOP_ALERT_DEFINITION_VERIFIED';state=$alertTask.State;last_result=$alertTask.LastTaskResult;orders_submitted=$false}|ConvertTo-Json -Compress
'''
    remove = check + r'''if($alertTask.State -eq 4){throw 'ALERT_CURRENT_TASK_RUNNING'}
$alertService.GetFolder('\').DeleteTask($alertData.task_name,0)
[ordered]@{status='DESKTOP_ALERT_DEFINITION_REMOVED';orders_submitted=$false}|ConvertTo-Json -Compress
'''
    for name, body in [('launch.py',launch),('install.ps1',install),('status.ps1',show),('remove.ps1',remove)]:
        (output/name).write_text(body.replace('RUNNER_SHA',runner_sha), encoding='utf-8')
    return {'status':'DESKTOP_ALERT_BACKEND_PREPARED','task_name':runner['task_name'],'orders_submitted':False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Original-deadline desktop alerts, no network or trading')
    parser.add_argument('command', choices=('prepare','prepare-backend','run-slot','status','stop'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--monitor-dir', type=Path)
    parser.add_argument('--original-monitor-sha256')
    parser.add_argument('--expected-plan-sha256')
    parser.add_argument('--expected-source-sha256')
    parser.add_argument('--backend-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command=='prepare':
            if args.monitor_dir is None or args.original_monitor_sha256 is None:
                raise ValueError('ORIGINAL_MONITOR_REQUIRED')
            result = prepare(args.directory,args.monitor_dir,args.original_monitor_sha256)
        elif args.command=='prepare-backend':
            if args.backend_dir is None:
                raise ValueError('NEW_ALERT_BACKEND_DIRECTORY_REQUIRED')
            result = prepare_backend(args.directory,args.backend_dir)
        elif args.command=='run-slot':
            if args.expected_plan_sha256 is None or args.expected_source_sha256 is None:
                raise ValueError('PINNED_ALERT_RUNNER_REQUIRED')
            result = run_slot(args.directory,args.expected_plan_sha256,args.expected_source_sha256)
        elif args.command=='stop':
            load_plan(args.directory)
            (args.directory/'stop').touch(exist_ok=True)
            result = {'status':'DESKTOP_ALERT_STOP_REQUESTED','orders_submitted':False}
        else:
            result = status(args.directory)
        print(exact_json(result))
        return 0
    except (OSError,ValueError,TypeError,KeyError,OverflowError) as error:
        print(exact_json({'status':'UNAVAILABLE','error_type':type(error).__name__,'orders_submitted':False}))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
