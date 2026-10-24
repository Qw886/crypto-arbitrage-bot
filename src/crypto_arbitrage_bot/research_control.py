"""Bounded public research jobs: start, status, cooperative stop and resume.

This controller never imports the testnet executor, loads credentials, submits
orders or changes a financial policy. Bound legacy validation watches can resume
their original budget; unbound existing workers are never silently adopted.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event
from time import monotonic, monotonic_ns
from typing import Any
from uuid import uuid4

from crypto_arbitrage_bot.config.settings import Settings, load_settings
from crypto_arbitrage_bot.operational_report import read_recent_quality
from crypto_arbitrage_bot.runner import ResearchRunner, create_runner
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.study_registry import full_research_policy
from crypto_arbitrage_bot.utils.atomic_publish import publish_state
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.file_lock import ControlError as ControlError
from crypto_arbitrage_bot.utils.file_lock import LockBusyError, file_lock, lock_is_held
from crypto_arbitrage_bot.utils.logging import configure_logging

_launched_processes: list[subprocess.Popen[bytes]] = []


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _exclusive(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(exact_json(value).encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _save_state(directory: Path, value: object) -> None:
    temporary = directory / "state.tmp"
    with temporary.open("wb") as stream:
        stream.write(exact_json(value).encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    publish_state(temporary, directory / "state.json")


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            body = stream.read(256_001)
        data = json.loads(body)
        if len(body) > 256_000 or not isinstance(data, dict) or exact_json(data).encode() != body:
            raise ValueError("canonical bounded object required")
        return dict(data)
    except (OSError, ValueError, TypeError, RecursionError) as error:
        raise ControlError("INVALID_LOCAL_CONTROL_RECORD") from error


def _at(value: str) -> datetime:
    return utc_timestamp(datetime.fromisoformat(value))


def _database_lock(database: Path) -> Path:
    return database.with_suffix(database.suffix + ".research-control.lock")


def prepare_job(config_path: Path, directory: Path, *, duration_seconds: int, max_cycles: int,
                clock: Callable[[], datetime] = utc_now) -> dict[str, Any]:
    """A new job requires a new DB; never silently take over an old worker."""
    if (type(duration_seconds) is not int or not 1 <= duration_seconds <= 8*86400
            or type(max_cycles) is not int or not 1 <= max_cycles <= 30000):
        raise ControlError("BOUNDED_DURATION_AND_CYCLE_LIMIT_REQUIRED")
    settings = load_settings(config_path)
    if settings.research is None or settings.testnet is not None:
        raise ControlError("PUBLIC_RESEARCH_CONFIGURATION_REQUIRED")
    database = settings.research.sqlite_path.resolve()
    directory = directory.resolve()
    if directory == database or database in directory.parents:
        raise ControlError("CONTROL_DIRECTORY_MUST_BE_SEPARATE")
    with file_lock(_database_lock(database)):
        if database.exists():
            raise ControlError("EXISTING_DATABASE_REQUIRES_ORIGINAL_JOB_RESUME")
        marker = database.with_suffix(database.suffix + ".research-job.json")
        if marker.exists():
            raise ControlError("DATABASE_ALREADY_BOUND_TO_A_JOB")
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise ControlError("NEW_CONTROL_DIRECTORY_REQUIRED")
        now = utc_timestamp(clock())
        reference = settings.research.study_registration
        if reference is not None:
            from crypto_arbitrage_bot.candidate_study import is_candidate_index, load_index
            if is_candidate_index(reference.path, reference.sha256):
                _, studies = load_index(settings)
                protocol = next(iter(studies.values())).protocol
                required_end = protocol.study_end+protocol.max_holding_period+protocol.max_exit_lateness
                if (now > protocol.study_start or now+timedelta(seconds=duration_seconds) < required_end
                        or max_cycles*settings.poll_interval_seconds < duration_seconds):
                    raise ControlError("CANDIDATE_JOB_MUST_COVER_ORIGINAL_PLANNED_EXITS")
        plan = {"schema_version": 1, "job_id": uuid4().hex, "mode": "PUBLIC_RESEARCH",
                "config_path": str(config_path.resolve()), "working_directory": str(Path.cwd().resolve()),
                "database": str(database), "created_at": now, "ends_at": now+timedelta(seconds=duration_seconds),
                "max_cycles": max_cycles, "poll_interval_seconds": settings.poll_interval_seconds,
                "policy_sha256": _sha(full_research_policy(settings))}
        if reference is not None:
            plan["study_reference"] = {"path": str(reference.path.resolve()), "sha256": reference.sha256}
        _exclusive(directory / "plan.json", plan)
        _exclusive(marker, {"job_directory": str(directory), "plan_sha256": _sha(exact_json(plan))})
        _save_state(directory, {"job_id": plan["job_id"], "state": "PREPARED", "cycles_started": 0,
            "cycles_completed": 0, "failed_cycles": 0, "interrupted_cycles": 0, "active_cycle": None,
            "updated_at": now, "worker_pid": None, "last_error_type": None})
        return plan


def load_job(directory: Path) -> tuple[dict[str, Any], Settings]:
    try:
        plan = _read(directory / "plan.json")
        settings = load_settings(Path(plan["config_path"]))
        if settings.research is None or settings.testnet is not None:
            raise ControlError("PUBLIC_RESEARCH_CONFIGURATION_REQUIRED")
        database = Path(plan["database"])
        marker = _read(database.with_suffix(database.suffix + ".research-job.json"))
        reference = settings.research.study_registration
        actual_reference = None if reference is None else {"path": str(reference.path.resolve()), "sha256": reference.sha256}
        if plan.get("study_reference") != actual_reference:
            raise ControlError("PINNED_STUDY_REFERENCE_CHANGED")
        if (plan["schema_version"] != 1 or plan["mode"] != "PUBLIC_RESEARCH" or not database.is_absolute()
                or database != settings.research.sqlite_path.resolve()
                or plan["policy_sha256"] != _sha(full_research_policy(settings))
                or marker != {"job_directory": str(directory.resolve()), "plan_sha256": _sha(exact_json(plan))}
                or type(plan["max_cycles"]) is not int or not 1 <= plan["max_cycles"] <= 30000
                or plan["poll_interval_seconds"] != settings.poll_interval_seconds
                or not timedelta(0) < _at(plan["ends_at"]) - _at(plan["created_at"]) <= timedelta(days=8)):
            raise ControlError("JOB_IDENTITY_OR_POLICY_CHANGED")
        return plan, settings
    except ControlError:
        raise
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        raise ControlError("JOB_IDENTITY_OR_POLICY_CHANGED") from error


class FileStopSignal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.event = Event()

    def set(self) -> None:
        self.event.set()

    def is_set(self) -> bool:
        return self.event.is_set() or self.path.exists()

    def wait(self, timeout: float | None = None) -> bool:
        end = None if timeout is None else monotonic()+timeout
        while not self.is_set():
            remaining = 1 if end is None else end-monotonic()
            if remaining <= 0:
                return False
            self.event.wait(min(1, remaining))
        return True


def _cycle(runner: ResearchRunner) -> dict[str, Any]:
    values = runner.run_once()  # Existing math, Builder, SQL and scheduler.
    return {"symbols": [{"cycle_id": v.cycle_id, "symbol": v.symbol, "status": v.status,
        "opportunity_id": v.opportunity_id, "decision": v.decision, "failure_stage": v.failure_stage}
                        for v in values], "failed": any(v.status.value != "SAVED" for v in values)}


def run_job(directory: Path, *, clock: Callable[[], datetime] = utc_now,
            steady_ns: Callable[[], int] = monotonic_ns,
            cycle: Callable[[ResearchRunner], dict[str, Any]] = _cycle,
            wait: Callable[[float], bool] | None = None) -> dict[str, Any]:
    """Resume remaining original time/cycles, with a durable pre-I/O claim."""
    directory = directory.resolve()
    plan, settings = load_job(directory)
    with file_lock(directory / "worker.lock"), file_lock(_database_lock(Path(plan["database"]))):
        state = _read(directory / "state.json")
        counters = ("cycles_started", "cycles_completed", "failed_cycles", "interrupted_cycles")
        if (state["job_id"] != plan["job_id"]
                or any(type(state[n]) is not int or not 0 <= state[n] <= plan["max_cycles"] for n in counters)
                or state["cycles_completed"] > state["cycles_started"]):
            raise ControlError("JOB_STATE_MISMATCH")
        now, end = utc_timestamp(clock()), _at(plan["ends_at"])
        if now < _at(plan["created_at"]):
            raise ControlError("CLOCK_BEFORE_REGISTERED_START")
        if now >= end or state["cycles_started"] >= plan["max_cycles"]:
            state.update(state="FINISHED", updated_at=now, worker_pid=None)
            _save_state(directory, state)
            return state
        if (directory / "stop").exists():
            state.update(state="STOPPED", updated_at=now, worker_pid=None)
            _save_state(directory, state)
            return state
        if state["active_cycle"] is not None:
            state["interrupted_cycles"] += 1
            state["active_cycle"] = None  # Never replay the old capture/claim.
            state["next_cycle_at"] = now+timedelta(seconds=settings.poll_interval_seconds)
        signal = FileStopSignal(directory / "stop")
        runner = create_runner(settings)
        runner.stop_signal = signal
        if runner.scheduler is not None:
            runner.scheduler.stopped = signal.is_set
        wait = signal.wait if wait is None else wait
        delta = end-now
        remaining_ns = ((delta.days*86400+delta.seconds)*1_000_000+delta.microseconds)*1000
        deadline = steady_ns()+remaining_ns
        last_pulse_ns = steady_ns()
        deadline_reached = False
        def progress() -> None:
            nonlocal last_pulse_ns, deadline_reached
            stamp, at = steady_ns(), utc_timestamp(clock())
            if at >= end or stamp >= deadline:
                deadline_reached = True
                signal.set()
            if stamp-last_pulse_ns >= 5_000_000_000:
                state["updated_at"] = at
                _save_state(directory, state)
                last_pulse_ns = stamp
        reporter = getattr(runner, "set_progress", None)
        if callable(reporter):
            reporter(progress)
        state.update(state="RUNNING", worker_pid=os.getpid(), updated_at=now)
        _save_state(directory, state)
        try:
            while not signal.is_set() and state["cycles_started"] < plan["max_cycles"]:
                now = utc_timestamp(clock())
                remaining = min((deadline-steady_ns())/1_000_000_000, (end-now).total_seconds())
                if remaining <= 0:
                    break
                next_at = state.get("next_cycle_at")
                if next_at is not None and now < _at(str(next_at)):
                    if wait(min((_at(str(next_at))-now).total_seconds(), remaining)):
                        signal.set()
                    continue
                index = state["cycles_started"]
                state.update(active_cycle=index, cycles_started=index+1, updated_at=now)
                _save_state(directory, state)  # Claim before market/SQLite I/O.
                # A bounded publication retry can cross the original deadline
                # or a stop request. The durable claim stays consumed; no IO.
                if utc_timestamp(clock()) >= end or steady_ns() >= deadline or signal.is_set():
                    deadline_reached = utc_timestamp(clock()) >= end or steady_ns() >= deadline
                    state["interrupted_cycles"] += 1
                    state["active_cycle"] = None
                    break
                try:
                    outcome = cycle(runner)
                    if not isinstance(outcome, dict) or type(outcome.get("failed")) is not bool:
                        raise ValueError("structured cycle outcome required")
                    (directory / "cycles").mkdir(exist_ok=True)
                    _exclusive(directory / "cycles" / f"{index:05d}.json", {"index": index,
                        "started_at": now, "ended_at": utc_timestamp(clock()), "outcome": outcome})
                    state["cycles_completed"] += 1
                    state["failed_cycles"] += int(outcome["failed"])
                    state["last_error_type"] = None
                except Exception as error:
                    state["failed_cycles"] += 1
                    state["last_error_type"] = type(error).__name__
                state.update(active_cycle=None, updated_at=utc_timestamp(clock()),
                    next_cycle_at=utc_timestamp(clock())+timedelta(seconds=settings.poll_interval_seconds))
                _save_state(directory, state)
            state["state"] = "FINISHED" if deadline_reached else ("STOPPED" if signal.is_set() else "FINISHED")
        except KeyboardInterrupt:
            signal.set()
            state["state"] = "INTERRUPTED"
        except BaseException as error:
            state.update(state="INTERRUPTED", last_error_type=type(error).__name__)
            raise
        finally:
            state.update(updated_at=utc_timestamp(clock()), worker_pid=None)
            _save_state(directory, state)
        return state


def request_stop(directory: Path) -> dict[str, Any]:
    # A changed config must block more work, never prevent a stop request.
    plan = _read(directory / "plan.json")
    marker = _read(Path(plan["database"]).with_suffix(Path(plan["database"]).suffix + ".research-job.json"))
    if marker != {"job_directory": str(directory.resolve()), "plan_sha256": _sha(exact_json(plan))}:
        raise ControlError("JOB_IDENTITY_OR_POLICY_CHANGED")
    (directory / "stop").touch(exist_ok=True)
    return {"job_id": plan["job_id"], "status": "STOP_REQUESTED", "orders_submitted": False}


def resume_job(directory: Path) -> None:
    plan, _ = load_job(directory)
    with file_lock(directory / "worker.lock"), file_lock(_database_lock(Path(plan["database"]))):
        (directory / "stop").unlink(missing_ok=True)


def database_status(database: Path) -> dict[str, Any]:
    """Counts and unfinished UTC time bounds in one strictly read-only snapshot.

    The eight-row UI preview is not complete coverage. Decode every unfinished
    due time before taking its bounds: SQL TEXT order can differ from UTC order.
    No financial SUM/AVG, writes, task transition or database initialization.
    """
    if not database.is_file():
        return {"status": "NOT_CREATED"}
    try:
        with closing(sqlite3.connect(database.resolve().as_uri()+"?mode=ro", uri=True, timeout=0)) as db:
            db.execute("PRAGMA query_only=ON")
            db.execute("PRAGMA busy_timeout=1000")
            db.execute("BEGIN")
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            result: dict[str, Any] = {"status": "READ"}
            if "opportunities" in tables:
                result["decisions"] = dict(db.execute("SELECT decision,COUNT(*) FROM opportunities GROUP BY decision"))
                result["recent_saved_quality"] = read_recent_quality(db)
            if "forward_validations" in tables:
                result["validations"] = dict(db.execute("SELECT status,COUNT(*) FROM forward_validations GROUP BY status"))
            if "validation_tasks" in tables:
                result["tasks"] = dict(db.execute("SELECT status,COUNT(*) FROM validation_tasks GROUP BY status"))
                if set(result["tasks"]) - {"PENDING", "RUNNING", "COMPLETED", "FAILED"}:
                    raise ValueError("unsupported validation task state")
                unfinished_times = tuple(utc_timestamp(datetime.fromisoformat(row[0])) for row in db.execute(
                    "SELECT due_at FROM validation_tasks WHERE status IN ('PENDING','RUNNING')"))
                result["earliest_unfinished_due_at"] = min(unfinished_times).isoformat() if unfinished_times else None
                result["latest_unfinished_due_at"] = max(unfinished_times).isoformat() if unfinished_times else None
                result["next_tasks"] = [dict(zip(("symbol", "horizon_seconds", "due_at", "status"), r, strict=True)) for r in db.execute(
                    "SELECT symbol,horizon_seconds,due_at,status FROM validation_tasks WHERE status IN ('PENDING','RUNNING') ORDER BY due_at,id LIMIT 8")]
            return result
    except (sqlite3.Error, ValueError, TypeError, OverflowError) as error:
        return {"status": "UNAVAILABLE", "error_type": type(error).__name__}


def job_status(directory: Path) -> dict[str, Any]:
    plan, _ = load_job(directory)
    state = _read(directory / "state.json")
    held = lock_is_held(directory / "worker.lock")
    return {"plan": plan, "state": state, "process_observation": "WORKER_LOCK_HELD" if held else
            ("INTERRUPTED_OR_NOT_RUNNING" if state["state"] == "RUNNING" else "NOT_RUNNING"),
            "stop_requested": (directory / "stop").exists(), "database": database_status(Path(plan["database"]))}


def _launch(directory: Path) -> int:
    plan, _ = load_job(directory)
    if lock_is_held(directory / "worker.lock"):
        raise ControlError("WORKER_ALREADY_RUNNING")
    command = [sys.executable, "-m", "crypto_arbitrage_bot.research_control", "_worker", "--job-dir", str(directory.resolve())]
    with (directory / "worker.log").open("ab") as log:
        process = subprocess.Popen(command, cwd=plan["working_directory"], stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            start_new_session=sys.platform != "win32")
    _launched_processes.append(process)
    return process.pid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Public research control; no orders or credentials")
    parser.add_argument("command", choices=("start", "resume", "stop", "status", "_worker", "_legacy_worker"))
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--duration-seconds", type=int)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "status" and args.job_dir is None and args.config is not None:
            settings = load_settings(args.config)
            if settings.research is None:
                raise ControlError("PUBLIC_RESEARCH_CONFIGURATION_REQUIRED")
            print(exact_json({"process_observation": "UNMANAGED_PROCESS_STATE_UNKNOWN",
                              "database": database_status(settings.research.sqlite_path)}))
            return 0
        if args.job_dir is None:
            parser.error("--job-dir required")
        directory = args.job_dir.resolve()
        if (directory / "recovery-binding.json").is_file() and not (directory / "plan.json").exists():
            from crypto_arbitrage_bot.legacy_research_control import (
                launch_legacy_job,
                load_legacy_job,
                request_legacy_stop,
                resume_legacy_job,
                run_legacy_job,
            )
            if args.command not in ("resume", "stop", "status", "_legacy_worker"):
                raise ControlError("BOUND_LEGACY_JOB_REQUIRES_RESUME")
            if args.config is not None or args.duration_seconds is not None or args.max_cycles is not None:
                raise ControlError("LEGACY_PARAMETERS_ARE_FROZEN")
            if args.command == "stop":
                print(exact_json(request_legacy_stop(directory)))
                return 0
            job = load_legacy_job(directory)
            if args.command == "status":
                from crypto_arbitrage_bot.runtime_status import research_status
                print(exact_json(research_status(job.settings, directory, at=utc_now(),
                                                max_heartbeat_age=timedelta(seconds=90))))
                return 0
            if args.command == "resume":
                resume_legacy_job(directory)
            if args.background:
                print(exact_json({"status": "LAUNCHED_NOT_YET_CONFIRMED", "pid": launch_legacy_job(directory),
                                  "job_directory": str(directory), "orders_submitted": False}))
                return 0
            configure_logging(job.settings.log_dir, job.settings.log_level)
            print(exact_json(run_legacy_job(directory)))
            return 0
        if args.command == "_legacy_worker":
            raise ControlError("EXISTING_LEGACY_BINDING_REQUIRED")
        if args.command == "start":
            if args.config is None or args.duration_seconds is None or args.max_cycles is None:
                parser.error("start requires config, original duration and cycle limit")
            prepare_job(args.config, directory, duration_seconds=args.duration_seconds, max_cycles=args.max_cycles)
        elif args.command == "stop":
            print(exact_json(request_stop(directory)))
            return 0
        elif args.command == "status":
            print(exact_json(job_status(directory)))
            return 0
        elif args.command == "resume":
            resume_job(directory)
        if args.background:
            print(exact_json({"status": "LAUNCHED_NOT_YET_CONFIRMED", "pid": _launch(directory), "job_directory": str(directory)}))
            return 0
        _, settings = load_job(directory)
        configure_logging(settings.log_dir, settings.log_level)
        print(exact_json(run_job(directory)))
        return 0
    except (ControlError, LockBusyError, OSError, ValueError, TypeError) as error:
        print(exact_json({"status": "CONTROL_FAILED", "reason": str(error) if isinstance(error, (ControlError, LockBusyError))
                         else type(error).__name__, "orders_submitted": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
