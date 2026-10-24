"""Unified local research/TESTNET supervision, no network, orders or recovery.

Reports actual owner evidence separately from journal/heartbeat descriptions.
Bounded monitoring records meaningful changes locally; it never restarts jobs,
extends a study, reads credentials or treats local CLOSED as exchange flatness.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from time import monotonic_ns
from typing import Any

from crypto_arbitrage_bot.config.settings import Settings, load_settings
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState
from crypto_arbitrage_bot.legacy_research_control import matches_legacy_worker
from crypto_arbitrage_bot.operational_report import publish_report, quality_signal
from crypto_arbitrage_bot.research_control import FileStopSignal, database_status, load_job
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.file_lock import file_lock, lock_is_held
from crypto_arbitrage_bot.utils.process_probe import (
    ProcessObservation,
    ProcessState,
    observe_process,
)


class RuntimeState(StrEnum):
    RUNNING = "RUNNING_CONFIRMED"
    STALE = "RUNNING_STALE_HEARTBEAT"
    STOPPED = "STOPPED_CONFIRMED"
    FINISHED = "FINISHED_NO_OWNER"
    UNKNOWN = "OWNER_UNVERIFIED"
    UNAVAILABLE = "UNAVAILABLE"


class RuntimeReason(StrEnum):
    WORKER_MISSING = "WORKER_MISSING"
    STALE_HEARTBEAT = "STALE_HEARTBEAT"
    IDENTITY_MISMATCH = "WORKER_IDENTITY_MISMATCH"
    OWNER_UNVERIFIED = "OWNER_UNVERIFIED"
    INVALID_LOCAL_STATE = "INVALID_LOCAL_STATE"
    FAILED_VALIDATIONS = "FAILED_VALIDATIONS_PRESENT"
    PENDING_WITHOUT_OWNER = "PENDING_VALIDATIONS_WITHOUT_OWNER"
    ACTIVE_WITHOUT_OWNER = "LOCAL_ACTIVE_POSITION_WITHOUT_CONFIRMED_OWNER"
    RECOVERY_REQUIRED = "LOCAL_RECOVERY_REQUIRED"
    UNKNOWN_ATTEMPT_STATE = "UNKNOWN_ATTEMPT_STATE"
    ORDER_BUDGET_EXHAUSTED = "ORDER_BUDGET_EXHAUSTED"
    JOURNAL_UNAVAILABLE = "JOURNAL_UNAVAILABLE"
    RESEARCH_DATABASE_UNAVAILABLE = "RESEARCH_DATABASE_UNAVAILABLE"
    CONTROL_BUDGET_ENDED = "CONTROL_BUDGET_ENDED"
    OVERDUE_VALIDATIONS = "OVERDUE_VALIDATIONS_PRESENT"
    VALIDATIONS_BEYOND_DEADLINE = "UNFINISHED_VALIDATIONS_BEYOND_DEADLINE"
    NO_VALIDATION_SCHEDULER = "UNFINISHED_VALIDATIONS_WITHOUT_SCHEDULER"


Probe = Callable[..., ProcessObservation]
LockProbe = Callable[[Path], bool]


def _at(value: str) -> datetime:
    return utc_timestamp(datetime.fromisoformat(value))


def _read(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        body = stream.read(256001)

    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = dict(items)
        if len(result) != len(items):
            raise ValueError("duplicate state fields")
        return result

    value = json.loads(body, object_pairs_hook=unique)
    if len(body) > 256000 or not isinstance(value, dict):
        raise ValueError("bounded local object required")
    return dict(value)


def _option(arguments: tuple[str, ...], name: str) -> str:
    if arguments.count(name) != 1:
        raise ValueError("exact single worker option required")
    return arguments[arguments.index(name) + 1]


def _owner(
    lock: Path,
    pid: int | None,
    started: datetime,
    updated: datetime,
    at: datetime,
    age: timedelta,
    matches: Callable[[tuple[str, ...]], bool],
    probe: Probe,
    lock_probe: LockProbe,
    reported: str,
) -> dict[str, Any]:
    held = lock_probe(lock) if lock.is_file() else False
    observation = (
        probe(pid, matches=matches, not_before=started, not_after=updated)
        if pid is not None
        else ProcessObservation(ProcessState.MISSING, None)
    )
    reasons = []
    if updated > at or updated < started:
        raise ValueError("invalid heartbeat time")
    if held and observation.state is ProcessState.MATCHED:
        status = RuntimeState.STALE if at - updated > age else RuntimeState.RUNNING
        if status is RuntimeState.STALE:
            reasons.append(RuntimeReason.STALE_HEARTBEAT)
    elif not held and observation.state is ProcessState.MISSING:
        status = (
            RuntimeState.FINISHED
            if reported in ("FINISHED", "STOPPED", "CANCELLED")
            else RuntimeState.STOPPED
        )
        if reported == "RUNNING":
            reasons.append(RuntimeReason.WORKER_MISSING)
    else:
        status = RuntimeState.UNKNOWN
        reasons.append(
            RuntimeReason.IDENTITY_MISMATCH
            if observation.state is ProcessState.DIFFERENT_IDENTITY
            else RuntimeReason.OWNER_UNVERIFIED
        )
    return {
        "status": status,
        "reported_status": reported,
        "worker_lock_held": held,
        "process_observation": observation.state,
        "worker_pid": observation.pid,
        "process_created_at": observation.created_at,
        "updated_at": updated,
        "reason_codes": tuple(reasons),
    }


def research_status(
    settings: Settings,
    directory: Path,
    *,
    at: datetime,
    max_heartbeat_age: timedelta,
    probe: Probe = observe_process,
    lock_probe: LockProbe = lock_is_held,
) -> dict[str, Any]:
    if settings.research is None or settings.testnet is not None:
        raise ValueError("public research settings required")
    database = settings.research.sqlite_path.resolve()
    data = database_status(database)
    try:
        state = _read(directory / "state.json")
        if state["state"] not in ("RUNNING", "FINISHED", "STOPPED", "INTERRUPTED", "PREPARED"):
            raise ValueError("unsupported local lifecycle")
        if (directory / "plan.json").is_file():
            plan, actual = load_job(directory)
            if actual != settings or plan["job_id"] != state["job_id"]:
                raise ValueError("managed job settings differ")
            started, deadline = _at(plan["created_at"]), _at(plan["ends_at"])
            used, limit = state["cycles_started"], plan["max_cycles"]

            def matches(args: tuple[str, ...]) -> bool:
                return (
                    Path(args[0]).resolve() == Path(sys.executable).resolve()
                    and len(args) == 6
                    and args[1:3] == ("-m", "crypto_arbitrage_bot.research_control")
                    and args[3] in ("_worker", "resume") and args[4] == "--job-dir"
                    and Path(_option(args, "--job-dir")).resolve() == directory.resolve()
                )

            kind = "MANAGED_PUBLIC_RESEARCH"
        else:
            binding = _read(directory / "recovery-binding.json")
            config = Path(binding["config_path"])
            if (
                binding["schema_version"] != 1
                or Path(binding["sqlite_path"]) != database
                or hashlib.sha256(config.read_bytes()).hexdigest() != binding["config_sha256"]
                or load_settings(config) != settings
                or Path(state["sqlite_path"]).resolve() != database
            ):
                raise ValueError("legacy binding/config differs")
            started, deadline = (
                _at(binding["original_started_at"]),
                _at(binding["original_deadline"]),
            )
            if (
                started != _at(state["started_at"])
                or deadline != _at(state["original_deadline"])
                or deadline - started != timedelta(seconds=binding["original_duration_seconds"])
                or state["duration_seconds"] != binding["original_duration_seconds"]
                or state["poll_seconds"] != binding["poll_seconds"]
            ):
                raise ValueError("legacy budget differs")
            used, limit = state["completed_batches"], binding["original_max_batches"]

            def matches(args: tuple[str, ...]) -> bool:
                return matches_legacy_worker(args, directory, binding)

            kind = "LEGACY_PUBLIC_VALIDATION"
        if type(used) is not int or type(limit) is not int or not 0 <= used <= limit <= 30000:
            raise ValueError("invalid consumed budget")
        owner = _owner(
            directory / "worker.lock",
            state["worker_pid"],
            started,
            _at(state["updated_at"]),
            at,
            max_heartbeat_age,
            matches,
            probe,
            lock_probe,
            str(state["state"]),
        )
        reasons = list(owner["reason_codes"])
        if data.get("status") in ("UNAVAILABLE", "NOT_CREATED"):
            reasons.append(RuntimeReason.RESEARCH_DATABASE_UNAVAILABLE)
        if owner["status"] in (RuntimeState.RUNNING, RuntimeState.STALE) and (at >= deadline or used >= limit):
            # The final claimed cycle may still be processing. Report its
            # exhausted continuation budget without declaring process death.
            reasons.append(RuntimeReason.CONTROL_BUDGET_ENDED)
        tasks = data.get("tasks", {})
        scheduler = settings.research.validation_scheduler
        earliest, latest = data.get("earliest_unfinished_due_at"), data.get("latest_unfinished_due_at")
        if earliest is not None and latest is not None:
            if scheduler is None:
                reasons.append(RuntimeReason.NO_VALIDATION_SCHEDULER)
            elif scheduler.max_lateness_seconds is not None and at > _at(earliest) + timedelta(
                seconds=scheduler.max_lateness_seconds
            ):
                reasons.append(RuntimeReason.OVERDUE_VALIDATIONS)
            # The original worker deadline is exclusive; it cannot start a
            # task at that boundary. Never extend the plan to hide this gap.
            if _at(latest) >= deadline:
                reasons.append(RuntimeReason.VALIDATIONS_BEYOND_DEADLINE)
        if tasks.get("FAILED", 0):
            reasons.append(RuntimeReason.FAILED_VALIDATIONS)
        if tasks.get("PENDING", 0) + tasks.get("RUNNING", 0) and owner["status"] not in (
            RuntimeState.RUNNING,
            RuntimeState.STALE,
        ):
            reasons.append(RuntimeReason.PENDING_WITHOUT_OWNER)
        counter_keys=('cycles_completed','failed_cycles','interrupted_cycles') if kind=='MANAGED_PUBLIC_RESEARCH' else (
            None,'failed_batches','interrupted_batches')
        counters={name:(state.get(key) if key is not None else None)
                  for name,key in zip(('finished','failed','interrupted'),counter_keys,strict=True)}
        counters={name:value if type(value) is int and 0<=value<=used else None for name,value in counters.items()}
        error_key='last_error_type' if kind=='MANAGED_PUBLIC_RESEARCH' else 'last_error'
        counters['last_failure_present']=None if error_key not in state else state[error_key] is not None
        return {
            "kind": kind,
            "collection_counters": counters,
            **owner,
            "reason_codes": tuple(dict.fromkeys(reasons)),
            "original_deadline": deadline,
            "budget_consumed": used,
            "budget_limit": limit,
            "validation_lateness_limit_seconds": None if scheduler is None else scheduler.max_lateness_seconds,
            "stop_requested": (directory / "stop").exists(),
            "database": data,
        }
    except Exception:
        return {
            "status": RuntimeState.UNAVAILABLE,
            "reason_codes": (RuntimeReason.INVALID_LOCAL_STATE,),
            "database": data,
        }


def testnet_status(settings: Settings, *, lock_probe: LockProbe = lock_is_held) -> dict[str, Any]:
    if settings.testnet is None:
        raise ValueError("explicit TESTNET configuration required")
    path = settings.testnet.journal_path.resolve()
    base = {
        "scope": "LOCAL_ARCHIVE_NOT_EXCHANGE_ACCOUNT",
        "exchange_account_flat_verified": False,
        "normal_entry_configured": settings.testnet.admission is not None,
        "budget_limit": settings.testnet.max_total_attempts,
    }
    if not path.is_file():
        return {
            **base,
            "status": "NOT_CREATED",
            "reason_codes": (RuntimeReason.JOURNAL_UNAVAILABLE,),
        }
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("PRAGMA busy_timeout=1000")
            db.execute("BEGIN")
            counts = dict(
                db.execute("SELECT state,COUNT(*) FROM testnet_pair_attempts GROUP BY state")
            )
            known = {state.value for state in PairExecutionState}
            active = [
                {
                    "attempt_id": row["attempt_id"],
                    "symbol": json.loads(row["plan_json"])["spot_entry"]["symbol"],
                    "state": row["state"],
                    "failure_code": row["failure_code"],
                }
                for row in db.execute(
                    "SELECT attempt_id,plan_json,state,failure_code FROM testnet_pair_attempts WHERE state NOT IN ('CLOSED','FAILED','FLATTENED') ORDER BY created_at,attempt_id"
                )
            ]
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            archives = {}
            for name in (
                "testnet_funding_checks",
                "testnet_balance_audits",
                "testnet_trade_reconciliations",
            ):
                if name in tables:
                    archives[name] = dict(
                        db.execute(f"SELECT status,COUNT(*) FROM {name} GROUP BY status")
                    )
        lock = path.with_suffix(".lock")
        held = lock_probe(lock) if lock.is_file() else False
        reasons = []
        if active:
            reasons.append(RuntimeReason.ACTIVE_WITHOUT_OWNER)
        if counts.get("RECOVERY_REQUIRED", 0):
            reasons.append(RuntimeReason.RECOVERY_REQUIRED)
        if set(counts) - known:
            reasons.append(RuntimeReason.UNKNOWN_ATTEMPT_STATE)
        used = sum(counts.values())
        if used >= settings.testnet.max_total_attempts:
            reasons.append(RuntimeReason.ORDER_BUDGET_EXHAUSTED)
        return {
            **base,
            "status": "READ",
            "states": counts,
            "active_attempts": active,
            "budget_consumed": used,
            "worker_lock_held": held,
            "process_observation": "LOCK_HELD_OWNER_UNVERIFIED"
            if held
            else "NO_JOURNAL_LOCK_OWNER",
            "reason_codes": tuple(reasons),
            "archives": archives,
        }
    except Exception:
        return {
            **base,
            "status": RuntimeState.UNAVAILABLE,
            "reason_codes": (RuntimeReason.JOURNAL_UNAVAILABLE,),
        }


def runtime_status(
    *,
    research_settings: Settings | None = None,
    research_directory: Path | None = None,
    testnet_settings: Settings | None = None,
    max_heartbeat_age: timedelta,
    clock: Callable[[], datetime] = utc_now,
    probe: Probe = observe_process,
    lock_probe: LockProbe = lock_is_held,
) -> dict[str, Any]:
    if not isinstance(max_heartbeat_age, timedelta) or not timedelta(
        0
    ) < max_heartbeat_age <= timedelta(days=1):
        raise ValueError("explicit heartbeat age in (0,1 day] required")
    if (research_settings is None) != (
        research_directory is None
    ) or research_settings is testnet_settings is None:
        raise ValueError("research settings/directory pair or TESTNET settings required")
    at = utc_timestamp(clock())
    result: dict[str, Any] = {
        "evaluated_at": at,
        "orders_submitted": False,
        "credentials_read": False,
    }
    if research_settings is not None and research_directory is not None:
        result["research"] = research_status(
            research_settings,
            research_directory.resolve(),
            at=at,
            max_heartbeat_age=max_heartbeat_age,
            probe=probe,
            lock_probe=lock_probe,
        )
    if testnet_settings is not None:
        result["testnet"] = testnet_status(testnet_settings, lock_probe=lock_probe)
        controller = _testnet_controller(
            testnet_settings,
            result["testnet"],
            at=at,
            max_heartbeat_age=max_heartbeat_age,
            probe=probe,
            lock_probe=lock_probe,
        )
        if controller is not None:
            result["testnet"]["controller"] = controller
    result["attention_required"] = any(
        item.get("status")
        in (
            RuntimeState.STALE,
            RuntimeState.STOPPED,
            RuntimeState.UNKNOWN,
            RuntimeState.UNAVAILABLE,
            "NOT_CREATED",
        )
        or RuntimeReason.PENDING_WITHOUT_OWNER in item.get("reason_codes", ())
        or RuntimeReason.RESEARCH_DATABASE_UNAVAILABLE in item.get("reason_codes", ())
        or any(code in item.get("reason_codes", ()) for code in (
            RuntimeReason.CONTROL_BUDGET_ENDED, RuntimeReason.OVERDUE_VALIDATIONS,
            RuntimeReason.VALIDATIONS_BEYOND_DEADLINE, RuntimeReason.NO_VALIDATION_SCHEDULER,
        ))
        or bool(item.get("active_attempts"))
        or item.get("controller", {}).get("status")
        in (
            RuntimeState.STALE,
            RuntimeState.STOPPED,
            RuntimeState.UNKNOWN,
            RuntimeState.UNAVAILABLE,
        )
        or RuntimeReason.CONTROL_BUDGET_ENDED in item.get("controller", {}).get("reason_codes", ())
        for name in ("research", "testnet")
        if (item := result.get(name)) is not None
    )
    return result


def _testnet_controller(
    settings: Settings,
    archive: dict[str, Any],
    *,
    at: datetime,
    max_heartbeat_age: timedelta,
    probe: Probe,
    lock_probe: LockProbe,
) -> dict[str, Any] | None:
    """Read a bound owner; neither resume it nor construct exchange clients."""
    assert settings.testnet is not None
    journal = settings.testnet.journal_path.resolve()
    marker = journal.with_suffix(journal.suffix + ".testnet-control.json")
    if not marker.exists():
        return None
    try:
        from crypto_arbitrage_bot.execution.control import load_job as load_testnet_job
        from crypto_arbitrage_bot.execution.control import matches_worker

        directory = Path(_read(marker)["directory"])
        plan, state, _ = load_testnet_job(directory)
        if (
            plan["journal"] != str(journal)
            or plan["max_total_attempts"] != archive["budget_limit"]
            or archive["status"] != "READ"
        ):
            raise ValueError("controller/archive identity unavailable")
        owner = _owner(
            directory / "worker.lock",
            state["worker_pid"],
            _at(plan["created_at"]),
            _at(state["updated_at"]),
            at,
            max_heartbeat_age,
            lambda args: matches_worker(args, directory),
            probe,
            lock_probe,
            "RUNNING" if state["status"] in ("RUNNING", "DRAINING") else state["status"],
        )
        if owner["status"] in (RuntimeState.RUNNING, RuntimeState.STALE):
            if not archive["worker_lock_held"]:
                owner.update(
                    status=RuntimeState.UNKNOWN, reason_codes=(RuntimeReason.OWNER_UNVERIFIED,)
                )
            else:
                archive["reason_codes"] = tuple(
                    code
                    for code in archive["reason_codes"]
                    if code != RuntimeReason.ACTIVE_WITHOUT_OWNER
                )
        owner.update(
            job_id=plan["job_id"],
            mode=plan["mode"],
            control_status=state["status"],
            original_deadline=_at(plan["ends_at"]),
            budget_consumed=state["cycles_started"],
            budget_limit=plan["max_cycles"],
            stop_requested=(directory / "stop").exists(),
        )
        if state["status"] in ("RUNNING", "DRAINING") and (
            at >= _at(plan["ends_at"]) or state["cycles_started"] >= plan["max_cycles"]
        ):
            owner["reason_codes"] = (*owner["reason_codes"], RuntimeReason.CONTROL_BUDGET_ENDED)
        return owner
    except Exception:
        return {
            "status": RuntimeState.UNAVAILABLE,
            "reason_codes": (RuntimeReason.INVALID_LOCAL_STATE,),
        }


def _meaningful(report: dict[str, Any]) -> str:
    result: dict[str, Any] = {}
    publication=report.get('operational_report')
    if isinstance(publication,dict) and publication.get('status')=='UNAVAILABLE':
        result['local_report']='REPORT_PUBLICATION_FAILED'
    for name in ("research", "testnet"):
        if name in report:
            item = report[name]
            result[name] = {
                k: item.get(k)
                for k in (
                    "status",
                    "reason_codes",
                    "stop_requested",
                    "active_attempts",
                    "process_observation",
                    "worker_pid",
                    "process_created_at",
                )
            }
            if name == "research":
                database = item.get("database", {})
                result[name]["validation_counts"] = database.get("validations")
                result[name]["task_counts"] = database.get("tasks")
                result[name]["saved_quality_signal"] = quality_signal(database.get("recent_saved_quality"))
                counters=item.get('collection_counters')
                if isinstance(counters,dict):
                    result[name]['collection_failure_signal']={k:counters.get(k) for k in
                        ('failed','interrupted','last_failure_present')}
            elif "controller" in item:
                controller = item["controller"]
                result[name]["controller"] = {
                    k: controller.get(k)
                    for k in (
                        "status",
                        "reason_codes",
                        "job_id",
                        "mode",
                        "control_status",
                        "stop_requested",
                        "worker_pid",
                        "process_created_at",
                    )
                }
    return exact_json(result)


def monitor_status(
    check: Callable[[], dict[str, Any]],
    directory: Path,
    *,
    duration_seconds: int | None,
    interval_seconds: int | None,
    max_checks: int | None,
    clock: Callable[[], datetime] = utc_now,
    steady_ns: Callable[[], int] = monotonic_ns,
    wait: Callable[[float], bool] | None = None,
    resume: bool = False,
    original_plan_sha256: str | None = None,
    probe: Probe = observe_process,
    worker_created_at: datetime | None = None,
    monitor_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from crypto_arbitrage_bot import monitor_archive as archive

    if not resume and (
        type(duration_seconds) is not int
        or not 1 <= duration_seconds <= 8 * 86400
        or type(interval_seconds) is not int
        or not 1 <= interval_seconds <= 3600
        or type(max_checks) is not int
        or not 1 <= max_checks <= 30000
    ):
        raise ValueError("bounded monitor time, interval and count required")
    if resume and (original_plan_sha256 is None or worker_created_at is None or any(
        value is not None for value in (duration_seconds, interval_seconds, max_checks)
    )):
        raise ValueError("resume requires the original plan hash and native owner creation time")
    directory.mkdir(parents=True, exist_ok=True)
    with file_lock(directory / "monitor.lock"):
        started = utc_timestamp(clock())
        if resume:
            assert original_plan_sha256 is not None
            plan, sha = archive.load_plan(directory, original_plan_sha256)
            progress = archive.resume_progress(directory, sha, now=started, probe=probe)
            if started >= archive.at(plan["ends_at"]) or progress["checks"] >= plan["max_checks"]:
                raise ValueError("original monitor budget ended")
            (directory / "stop").unlink(missing_ok=True)
        else:
            if (directory / "plan.json").exists():
                raise ValueError("new monitor output directory required; never reset a prior budget")
            assert duration_seconds is not None and interval_seconds is not None and max_checks is not None
            plan = {"started_at": started, "ends_at": started + timedelta(seconds=duration_seconds),
                    "interval_seconds": interval_seconds, "max_checks": max_checks, "orders_submitted": False}
            archive.write(directory / "plan.json", plan, exclusive=True)
            plan, sha = archive.load_plan(directory)
            if monitor_spec is not None:
                archive.bind_live(directory, sha, os.getpid(), monitor_spec, clock=clock, probe=probe)
            progress = {"schema_version": 1, "plan_sha256": sha, "initial_checks": 0,
                        "checks": 0, "changes": 0, "previous": None, "active_check": None,
                        "next_check_at": started.isoformat(), "interrupted_checks": 0}
        ends_at = archive.at(plan["ends_at"])
        interval_seconds, max_checks = plan["interval_seconds"], plan["max_checks"]
        progress.update(state="RUNNING", worker_pid=os.getpid(), worker_created_at=worker_created_at,
                        mode="RESUMED" if resume else "LEGACY", updated_at=started)
        archive.write(directory / "progress.json", progress)
        claims = directory / "check-claims"
        claims.mkdir(exist_ok=True)
        signal = FileStopSignal(directory / "stop")
        wait = signal.wait if wait is None else wait
        remaining_delta = ends_at - started
        remaining_ns = ((remaining_delta.days*86400+remaining_delta.seconds)*1_000_000
                        + remaining_delta.microseconds)*1000
        deadline = steady_ns() + remaining_ns
        count, changes, previous = progress["checks"], progress["changes"], progress["previous"]
        state = "FINISHED"
        try:
            while (
                not signal.is_set()
                and count < max_checks
                and steady_ns() < deadline
                and utc_timestamp(clock()) < ends_at
            ):
                now = utc_timestamp(clock())
                next_at = archive.at(progress["next_check_at"])
                if now < next_at:
                    remaining = min((deadline-steady_ns())/1_000_000_000, (ends_at-now).total_seconds())
                    if remaining <= 0:
                        break
                    if wait(min((next_at-now).total_seconds(), remaining)):
                        signal.set()
                    continue
                # A check slot is consumed before I/O. Interrupted claims are
                # retained on resume and never replayed or inferred from time.
                archive.write(claims / f"{count:05d}.json",
                              {"index": count, "plan_sha256": sha, "claimed_at": now}, exclusive=True)
                count += 1
                progress.update(checks=count, active_check=count-1, updated_at=now)
                archive.write(directory / "progress.json", progress)
                try:
                    report = check()
                except Exception as error:
                    report = {
                        "evaluated_at": utc_timestamp(clock()),
                        "research": {
                            "status": "UNAVAILABLE",
                            "reason_codes": ("CHECK_FAILED",),
                            "error_type": type(error).__name__,
                        },
                    }
                report['operational_report'] = {'status':'PUBLISHED','file':'report.md','scope':'LOCAL_ONLY'}
                try:
                    publish_report(directory, report)
                except (OSError,ValueError,TypeError,RecursionError):
                    report['operational_report'] = {'status':'UNAVAILABLE','reason':'REPORT_PUBLICATION_FAILED'}
                archive.write(directory / "latest.json", report)
                fingerprint = _meaningful(report)
                if fingerprint != previous:
                    event = {
                        "observed_at": utc_timestamp(clock()),
                        "change_index": changes,
                        "status": json.loads(fingerprint),
                    }
                    archive.write(directory / f"event-{changes:05d}.json", event, exclusive=True)
                    print(exact_json(event), flush=True)
                    changes += 1
                    previous = fingerprint
                progress.update(changes=changes, previous=previous, active_check=None,
                                updated_at=utc_timestamp(clock()),
                                next_check_at=(utc_timestamp(clock())+timedelta(seconds=interval_seconds)).isoformat())
                archive.write(directory / "progress.json", progress)
                if count >= max_checks:
                    break
                remaining = min(
                    (deadline - steady_ns()) / 1_000_000_000,
                    (ends_at - utc_timestamp(clock())).total_seconds(),
                )
                if remaining <= 0:
                    break
                if wait(min(interval_seconds, remaining)):
                    signal.set()
                    break
            state = "STOPPED" if signal.is_set() else "FINISHED"
        except KeyboardInterrupt:
            state = "INTERRUPTED"
        result = {
            "state": state,
            "checks": count,
            "changes": changes,
            "ended_at": utc_timestamp(clock()),
            "orders_submitted": False,
        }
        progress.update(state=state, updated_at=result["ended_at"])
        archive.write(directory / "progress.json", progress)
        archive.write(directory / "state.json", result)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local runtime supervision; no credentials, orders or auto restart"
    )
    parser.add_argument("command", choices=("status", "monitor", "bind-monitor", "stop-monitor", "resume-monitor", "status-monitor"))
    parser.add_argument("--research-config", type=Path)
    parser.add_argument("--research-dir", type=Path)
    parser.add_argument("--testnet-config", type=Path)
    parser.add_argument("--max-heartbeat-age-seconds", type=int)
    parser.add_argument("--original-plan-sha256")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--duration-seconds", type=int)
    parser.add_argument("--interval-seconds", type=int)
    parser.add_argument("--max-checks", type=int)
    args = parser.parse_args(argv)
    try:
        from crypto_arbitrage_bot import monitor_archive as archive

        if args.command in ("bind-monitor", "stop-monitor", "resume-monitor", "status-monitor"):
            if args.output_dir is None or args.original_plan_sha256 is None or any(v is not None for v in (
                args.duration_seconds, args.interval_seconds, args.max_checks
            )):
                parser.error("bound monitor commands require output directory and original plan hash; budgets are frozen")
            directory = args.output_dir.resolve()
            plan, sha = archive.load_plan(directory, args.original_plan_sha256)
            if args.command == "bind-monitor":
                if args.pid is None or args.max_heartbeat_age_seconds is None:
                    parser.error("bind-monitor requires observed PID and original scopes/heartbeat age")
                spec = {"research_config": None if args.research_config is None else str(args.research_config.resolve()),
                        "research_directory": None if args.research_dir is None else str(args.research_dir.resolve()),
                        "testnet_config": None if args.testnet_config is None else str(args.testnet_config.resolve()),
                        "max_heartbeat_age_seconds": args.max_heartbeat_age_seconds,
                        "working_directory": str(Path.cwd().resolve())}
                bound = archive.bind_live(directory, sha, args.pid, spec)
                print(exact_json({"status": "MONITOR_BOUND", "original_plan_sha256": sha,
                                  "original_deadline": plan["ends_at"], "owner_pid": bound["legacy_pid"],
                                  "orders_submitted": False}))
                return 0
            if any(v is not None for v in (args.pid, args.research_config, args.research_dir,
                                           args.testnet_config, args.max_heartbeat_age_seconds)):
                parser.error("bound stop/resume cannot replace original monitor scope")
            binding = archive.load_binding(directory, sha, check_policy=args.command != "stop-monitor")
            if args.command == "stop-monitor":
                (directory / "stop").touch(exist_ok=True)
                print(exact_json({"status": "MONITOR_STOP_REQUESTED", "original_deadline": plan["ends_at"],
                                  "orders_submitted": False}))
                return 0
            if args.command == "status-monitor":
                path = directory / "progress.json"
                progress = archive.read(path) if path.exists() else None
                terminal_path = directory / "state.json"
                terminal = archive.read(terminal_path) if progress is None and terminal_path.exists() else None
                pid = binding["legacy_pid"] if progress is None else progress["worker_pid"]
                created = binding["legacy_created_at"] if progress is None else progress["worker_created_at"]
                if created is None:
                    if pid != binding["legacy_pid"]:
                        raise ValueError("monitor owner creation time missing")
                    created = binding["legacy_created_at"]
                birth = archive.at(created)
                legacy = progress is None or progress["mode"] == "LEGACY"
                matches = (lambda a: archive.matches_legacy(a, directory, binding["spec"], plan)) if legacy else (
                    lambda a: archive.matches_resumed(a, directory, sha))
                def owner_probe(owner_pid: int, **kwargs: Any) -> ProcessObservation:
                    return observe_process(owner_pid, matches=matches, not_before=birth, not_after=birth)
                used: int | None
                if progress is not None:
                    pulse, reported, used = progress["updated_at"], progress["state"], progress["checks"]
                elif terminal is not None:
                    pulse, reported, used = terminal["ended_at"], terminal["state"], terminal["checks"]
                else:
                    pulse = archive.read(directory / "latest.json")["evaluated_at"]
                    reported, used = "RUNNING", None
                if (reported not in ("RUNNING", "STOPPED", "INTERRUPTED", "FINISHED")
                        or used is not None and (type(used) is not int or not 0 <= used <= plan["max_checks"])):
                    raise ValueError("invalid monitor lifecycle or count")
                now = utc_now()
                owner = _owner(directory / "monitor.lock", pid, birth, archive.at(pulse), now,
                    timedelta(seconds=binding["spec"]["max_heartbeat_age_seconds"]), matches,
                    owner_probe, lock_is_held, reported)
                ended = now >= archive.at(plan["ends_at"]) or used is not None and used >= plan["max_checks"]
                latest_path = directory / "latest.json"
                latest = archive.read(latest_path) if latest_path.exists() else None
                check_attention = latest is None or bool(latest.get("attention_required")) or any(
                    item.get("status") in ("UNAVAILABLE", "OWNER_UNVERIFIED", "STOPPED_CONFIRMED", "RUNNING_STALE_HEARTBEAT")
                    for name in ("research", "testnet") if latest is not None and (item := latest.get(name)) is not None
                )
                print(exact_json({**owner, "evaluated_at": now, "original_plan_sha256": sha,
                    "original_deadline": plan["ends_at"], "budget_consumed": used, "budget_limit": plan["max_checks"],
                    "last_check_attention_required": check_attention,
                    "attention_required": owner["status"] not in (RuntimeState.RUNNING, RuntimeState.FINISHED)
                        or owner["status"] is RuntimeState.RUNNING and ended or check_attention,
                    "orders_submitted": False, "credentials_read": False}))
                return 0
            # Only this actual process may become the new owner. A failed
            # native identity query is not authority to restart an old owner.
            now = utc_now()
            observed = observe_process(os.getpid(), matches=lambda a: archive.matches_resumed(a, directory, sha),
                                       not_before=archive.EPOCH, not_after=now)
            if observed.state is not ProcessState.MATCHED or observed.created_at is None:
                raise ValueError("new monitor native identity unconfirmed")
            spec = binding["spec"]
            research = None if spec["research_config"] is None else load_settings(Path(spec["research_config"]))
            testnet = None if spec["testnet_config"] is None else load_settings(Path(spec["testnet_config"]))
            def resumed_check() -> dict[str, Any]:
                if archive.policy(spec) != binding["policy"]:
                    raise ValueError("monitor policy changed while running")
                return runtime_status(research_settings=research,
                    research_directory=None if spec["research_directory"] is None else Path(spec["research_directory"]),
                    testnet_settings=testnet, max_heartbeat_age=timedelta(seconds=spec["max_heartbeat_age_seconds"]))
            result = monitor_status(resumed_check, directory, duration_seconds=None, interval_seconds=None,
                max_checks=None, resume=True, original_plan_sha256=sha, worker_created_at=observed.created_at)
            print(exact_json(result))
            return 0
        if args.max_heartbeat_age_seconds is None:
            parser.error("status/monitor requires explicit maximum heartbeat age")
        research = None if args.research_config is None else load_settings(args.research_config)
        testnet = None if args.testnet_config is None else load_settings(args.testnet_config)

        def check() -> dict[str, Any]:
            return runtime_status(
                research_settings=research,
                research_directory=args.research_dir,
                testnet_settings=testnet,
                max_heartbeat_age=timedelta(seconds=args.max_heartbeat_age_seconds),
            )

        if args.command == "status":
            print(exact_json(check()))
        else:
            if any(
                v is None
                for v in (
                    args.output_dir,
                    args.duration_seconds,
                    args.interval_seconds,
                    args.max_checks,
                )
            ):
                parser.error(
                    "monitor requires output directory, original duration, interval and max checks"
                )
            print(
                exact_json(
                    monitor_status(
                        check,
                        args.output_dir,
                        duration_seconds=args.duration_seconds,
                        interval_seconds=args.interval_seconds,
                        max_checks=args.max_checks,
                        monitor_spec={
                            "research_config": None if args.research_config is None else str(args.research_config.resolve()),
                            "research_directory": None if args.research_dir is None else str(args.research_dir.resolve()),
                            "testnet_config": None if args.testnet_config is None else str(args.testnet_config.resolve()),
                            "max_heartbeat_age_seconds": args.max_heartbeat_age_seconds,
                            "working_directory": str(Path.cwd().resolve()),
                        },
                    )
                )
            )
        return 0
    except Exception as error:
        print(
            exact_json(
                {
                    "status": "SUPERVISION_FAILED",
                    "error_type": type(error).__name__,
                    "orders_submitted": False,
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
