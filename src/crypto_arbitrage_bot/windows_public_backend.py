"""Finite user-session Task Scheduler backend for original PUBLIC jobs only.

The gate is read-only. Original resume commands keep all locks, counts, deadlines
and policy checks. No credential loader, TESTNET execution or live-owner stopping.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from crypto_arbitrage_bot import monitor_archive
from crypto_arbitrage_bot.execution.risk_review import implementation_fingerprint
from crypto_arbitrage_bot.legacy_research_control import load_legacy_job
from crypto_arbitrage_bot.research_control import load_job
from crypto_arbitrage_bot.runtime_status import research_status
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or path.is_symlink():
        raise ValueError("BACKEND_SCOPE_OUTSIDE_PROJECT")
    return resolved


def scope(root: Path, kind: str, target: Path, sha: str | None) -> dict[str, Any]:
    root = root.resolve()
    target = _within(target, root)
    if kind == "research":
        if (target / "recovery-binding.json").exists() and not (target / "plan.json").exists():
            job = load_legacy_job(target)
            if job.settings.testnet is not None:
                raise ValueError("PUBLIC_ONLY")
            ends = job.deadline
            paths = [target / "recovery-binding.json", target / "product-control.json", Path(job.binding["config_path"])]
        else:
            plan, settings = load_job(target)
            if settings.testnet is not None:
                raise ValueError("PUBLIC_ONLY")
            ends = datetime.fromisoformat(plan["ends_at"])
            paths = [target / "plan.json", Path(plan["config_path"])]
        arguments = ["-m", "crypto_arbitrage_bot.research_control", "resume", "--job-dir", str(target)]
    elif kind == "monitor":
        if sha is None:
            raise ValueError("ORIGINAL_MONITOR_SHA_REQUIRED")
        plan, original = monitor_archive.load_plan(target, sha)
        binding = monitor_archive.load_binding(target, original)
        ends = monitor_archive.at(plan["ends_at"])
        paths = [target / "plan.json", target / "monitor-binding.json"]
        for key in ("research_config", "testnet_config"):
            if binding["spec"][key] is not None:
                paths.append(Path(binding["spec"][key]))
        arguments = ["-m", "crypto_arbitrage_bot.runtime_status", "resume-monitor", "--output-dir", str(target), "--original-plan-sha256", original]
    else:
        raise ValueError("PUBLIC_BACKEND_KIND_REQUIRED")
    return {"kind": kind, "target": str(target), "original_monitor_sha256": sha,
            "ends_at": utc_timestamp(ends).isoformat(), "arguments": arguments,
            "frozen_files": {str(_within(path, root)): _sha(path) for path in paths}}


def prepare(root: Path, kind: str, target: Path, output: Path, *, sha: str | None = None,
            now: datetime | None = None) -> dict[str, Any]:
    root = root.resolve()
    output = _within(output, root)
    value = scope(root, kind, target, sha)
    at = utc_timestamp(utc_now() if now is None else now)
    ends = utc_timestamp(datetime.fromisoformat(value["ends_at"]))
    if not at < ends <= at + timedelta(days=8):
        raise ValueError("ORIGINAL_WINDOW_ENDED_OR_INVALID")
    value.update(schema_version=1, root=str(root), created_at=at.isoformat(),
        source_sha256=implementation_fingerprint(), interval_seconds=60,
        execution_limit_seconds=math.ceil((ends-at).total_seconds()),
        task_name="CryptoArbitrageBot_Public_" + uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "plan.json").open("xb") as stream:
        stream.write(exact_json(value).encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return dict(value)


def load_plan(directory: Path) -> dict[str, Any]:
    with (directory / "plan.json").open("rb") as stream:
        body = stream.read(256001)
    if len(body) > 256000:
        raise ValueError("BOUNDED_BACKEND_PLAN_REQUIRED")
    value = json.loads(body)
    if exact_json(value).encode() != body or type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("CANONICAL_BACKEND_PLAN_REQUIRED")
    root = Path(value["root"])
    _within(directory, root)
    if value["source_sha256"] != implementation_fingerprint():
        raise ValueError("BACKEND_SOURCE_CHANGED")
    current = scope(root, value["kind"], Path(value["target"]), value["original_monitor_sha256"])
    if any(value[key] != item for key, item in current.items()) or value["interval_seconds"] != 60:
        raise ValueError("ORIGINAL_BACKEND_SCOPE_CHANGED")
    if type(value["execution_limit_seconds"]) is not int or value["execution_limit_seconds"] != math.ceil(
        (datetime.fromisoformat(value["ends_at"])-datetime.fromisoformat(value["created_at"])).total_seconds()
    ) or not 1 <= value["execution_limit_seconds"] <= 8*86400:
        raise ValueError("FINITE_BACKEND_LIMIT_REQUIRED")
    name = value["task_name"]
    if not isinstance(name, str) or not name.startswith("CryptoArbitrageBot_Public_") or len(name) != 58 or any(c not in "0123456789abcdef" for c in name[26:]):
        raise ValueError("OWNED_BACKEND_TASK_NAME_REQUIRED")
    return dict(value)


def gate_result(report: dict[str, Any], *, stop_requested: bool, now: datetime,
                ends_at: datetime) -> bool:
    used, maximum = report.get("budget_consumed"), report.get("budget_limit")
    return (not stop_requested and utc_timestamp(now) < utc_timestamp(ends_at)
        and report.get("status") == "STOPPED_CONFIRMED"
        and report.get("process_observation") == "MISSING" and report.get("worker_lock_held") is False
        and type(used) is int and type(maximum) is int and 0 <= used < maximum)


def gate(directory: Path) -> bool:
    value = load_plan(directory)
    now = utc_now()
    target = Path(value["target"])
    if (directory / "stop").exists() or (target / "stop").exists():
        return False
    if value["kind"] == "research":
        if (target / "recovery-binding.json").exists() and not (target / "plan.json").exists():
            settings = load_legacy_job(target).settings
        else:
            _, settings = load_job(target)
        report = research_status(settings, target, at=now, max_heartbeat_age=timedelta(seconds=90))
    else:
        args = [sys.executable, "-m", "crypto_arbitrage_bot.runtime_status", "status-monitor",
                "--output-dir", str(target), "--original-plan-sha256", value["original_monitor_sha256"]]
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", timeout=15,
            env={**os.environ, "PYTHONPATH": str(Path(value["root"])/"src")}, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        report = json.loads(result.stdout)
    return gate_result(report, stop_requested=bool(report.get("stop_requested")), now=now,
                       ends_at=datetime.fromisoformat(value["ends_at"]))


def run_backend(directory: Path, expected_sha: str) -> int:
    """Task entry hosted by pythonw; gate and wait without a console or shell."""
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha) or (
        _sha(directory / "plan.json") != expected_sha
    ):
        raise ValueError("BACKEND_PLAN_CHANGED")
    if not gate(directory):
        return 0
    value = load_plan(directory)
    target = Path(value["target"])
    if _sha(directory / "plan.json") != expected_sha:
        raise ValueError("BACKEND_PLAN_CHANGED")
    if (directory / "stop").exists() or (target / "stop").exists() or utc_now() >= datetime.fromisoformat(value["ends_at"]):
        return 0
    root = Path(value["root"])
    # The original owner identity pins python.exe. CREATE_NO_WINDOW keeps
    # that interpreter silent without changing the policy to accept pythonw.
    result = subprocess.run([str(root / ".venv/Scripts/python.exe"), *value["arguments"]],
        cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")},
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, check=False)
    return result.returncode


def write_scripts(directory: Path) -> None:
    value = load_plan(directory)
    expected = _sha(directory / "plan.json")
    # Standalone bootstrap: Task Scheduler does not inherit CLI PYTHONPATH.
    launch = '''import hashlib
import json
import os
import sys
from pathlib import Path

try:
    directory = Path(__file__).resolve().parent
    with (directory / "plan.json").open("rb") as stream:
        body = stream.read(256001)
    if len(body) > 256000 or hashlib.sha256(body).hexdigest() != "EXPECTED_PLAN_SHA":
        raise SystemExit(2)
    root = Path(json.loads(body)["root"])
    sys.path.insert(0, str(root / "src"))
    os.environ["PYTHONPATH"] = str(root / "src")
    from crypto_arbitrage_bot.windows_public_backend import run_backend
    raise SystemExit(run_backend(directory, "EXPECTED_PLAN_SHA"))
except Exception:
    raise SystemExit(2)
'''
    # All variable paths/arguments are JSON data, never interpolated shell code.
    action = r'''$ErrorActionPreference='Stop'
$backendDir=Split-Path -Parent $PSCommandPath
if((Get-FileHash -LiteralPath (Join-Path $backendDir 'plan.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'EXPECTED_PLAN_SHA'){exit 2}
$backendData=Get-Content -LiteralPath (Join-Path $backendDir 'plan.json') -Raw|ConvertFrom-Json
$env:PYTHONPATH=Join-Path $backendData.root 'src'
$backendPython=Join-Path $backendData.root '.venv\Scripts\python.exe'
$backendGateArgs=@('-m','crypto_arbitrage_bot.windows_public_backend','gate','--directory',$backendDir)
$backendGate=Start-Process -FilePath $backendPython -ArgumentList (($backendGateArgs|ForEach-Object {'"'+($_ -replace '"','\"')+'"'}) -join ' ') -WindowStyle Hidden -Wait -PassThru
if($backendGate.ExitCode -ne 0){exit 0}
$backendArgs=$backendData.arguments|ForEach-Object {'"'+($_ -replace '"','\"')+'"'}
$backendWorker=Start-Process -FilePath $backendPython -ArgumentList ($backendArgs -join ' ') -WorkingDirectory $backendData.root -WindowStyle Hidden -Wait -PassThru
exit $backendWorker.ExitCode
'''
    install = r'''$ErrorActionPreference='Stop'
$backendDir=Split-Path -Parent $PSCommandPath
if((Get-FileHash -LiteralPath (Join-Path $backendDir 'plan.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'EXPECTED_PLAN_SHA'){throw 'BACKEND_PLAN_CHANGED'}
$backendData=Get-Content -LiteralPath (Join-Path $backendDir 'plan.json') -Raw|ConvertFrom-Json
$backendService=New-Object -ComObject 'Schedule.Service';$backendService.Connect()
$backendDefinition=$backendService.NewTask(0)
$backendDefinition.Principal.UserId=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$backendDefinition.Principal.LogonType=3;$backendDefinition.Principal.RunLevel=0
$backendDefinition.Settings.Hidden=$true;$backendDefinition.Settings.MultipleInstances=2
$backendDefinition.Settings.ExecutionTimeLimit='PT'+$backendData.execution_limit_seconds+'S'
$backendDefinition.Settings.StartWhenAvailable=$false
$backendDefinition.Settings.DisallowStartIfOnBatteries=$false;$backendDefinition.Settings.StopIfGoingOnBatteries=$false
$backendTrigger=$backendDefinition.Triggers.Create(1)
$backendTrigger.StartBoundary=[DateTime]::UtcNow.AddSeconds(3).ToString('o')
if($backendData.ends_at -is [DateTime]){$backendEnd=([DateTime]$backendData.ends_at).ToUniversalTime().ToString('o')}else{$backendEnd=[DateTimeOffset]::Parse([string]$backendData.ends_at).UtcDateTime.ToString('o')}
$backendTrigger.EndBoundary=$backendEnd
$backendTrigger.Repetition.Interval='PT1M'
$backendTrigger.Repetition.Duration='PT'+$backendData.execution_limit_seconds+'S'
$backendTrigger.Repetition.StopAtDurationEnd=$true
$backendAction=$backendDefinition.Actions.Create(0)
$backendAction.Path=Join-Path $backendData.root '.venv\Scripts\pythonw.exe'
$backendAction.Arguments='"'+(Join-Path $backendDir 'launch.py')+'"'
$backendAction.WorkingDirectory=$backendData.root
$backendRegistered=$backendService.GetFolder('\').RegisterTaskDefinition($backendData.task_name,$backendDefinition,2,$null,$null,3)
[IO.File]::WriteAllText((Join-Path $backendDir 'registered-task.xml'),$backendRegistered.Xml,[Text.UTF8Encoding]::new($false))
[ordered]@{status='REGISTERED_NOT_YET_EXECUTED';task_name=$backendData.task_name;ends_at=$backendData.ends_at;credentials_read=$false;orders_submitted=$false}|ConvertTo-Json -Compress
'''
    status = r'''$ErrorActionPreference='Stop'
$backendDir=Split-Path -Parent $PSCommandPath
if((Get-FileHash -LiteralPath (Join-Path $backendDir 'plan.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'EXPECTED_PLAN_SHA'){throw 'BACKEND_PLAN_CHANGED'}
$backendData=Get-Content -LiteralPath (Join-Path $backendDir 'plan.json') -Raw|ConvertFrom-Json
$backendService=New-Object -ComObject 'Schedule.Service';$backendService.Connect()
$backendTask=$backendService.GetFolder('\').GetTask($backendData.task_name)
$backendSaved=[IO.File]::ReadAllText((Join-Path $backendDir 'registered-task.xml'))
if($backendTask.Xml -cne $backendSaved){throw 'BACKEND_DEFINITION_CHANGED'}
[ordered]@{status='DEFINITION_VERIFIED';task_name=$backendData.task_name;state=$backendTask.State;last_result=$backendTask.LastTaskResult;next_run=$backendTask.NextRunTime.ToUniversalTime().ToString('o');ends_at=$backendData.ends_at;credentials_read=$false;orders_submitted=$false}|ConvertTo-Json -Compress
'''
    remove = status[:status.index("[ordered]@")]+r'''[IO.File]::WriteAllText((Join-Path $backendDir 'stop'),'WITHDRAW_AUTOMATIC_RECOVERY')
if($backendTask.State -eq 4){[ordered]@{status='WITHDRAWAL_MARKED_CURRENT_TASK_RUNNING';orders_submitted=$false}|ConvertTo-Json -Compress;exit 1}
$backendService.GetFolder('\').DeleteTask($backendData.task_name,0)
[ordered]@{status='BACKEND_REGISTRATION_REMOVED';task_name=$backendData.task_name;orders_submitted=$false}|ConvertTo-Json -Compress
'''
    for name, text in (("launch.py", launch), ("action.ps1", action), ("install.ps1", install), ("status.ps1", status), ("remove.ps1", remove)):
        with (directory / name).open("x", encoding="utf-8") as stream:
            stream.write(text.replace("EXPECTED_PLAN_SHA", expected))
    # Source/policy validation is repeated by the gate at every actual trigger.
    assert value["arguments"][1] in ("crypto_arbitrage_bot.research_control", "crypto_arbitrage_bot.runtime_status")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bound user-session PUBLIC recovery backend; no credentials or orders")
    parser.add_argument("command", choices=("prepare", "gate", "run"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--kind", choices=("research", "monitor"))
    parser.add_argument("--target", type=Path)
    parser.add_argument("--original-monitor-sha256")
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            if args.kind is None or args.target is None:
                parser.error("prepare requires original kind and target")
            prepare(Path.cwd(), args.kind, args.target, args.directory, sha=args.original_monitor_sha256)
            write_scripts(args.directory.resolve())
            print(exact_json({"status": "PREPARED_NOT_REGISTERED", "orders_submitted": False}))
            return 0
        if args.command == "run":
            if args.expected_plan_sha256 is None:
                raise ValueError("EXPECTED_BACKEND_PLAN_SHA_REQUIRED")
            return run_backend(args.directory.resolve(), args.expected_plan_sha256)
        return 0 if gate(args.directory.resolve()) else 1
    except Exception:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
