"""Product lifecycle for an already bound legacy public validation watch.

Never create a replacement study, extend its original window, replay a consumed
batch or qualify a strategy. The existing scheduler owns all market/financial
work. Operational migration is not retrospective research preregistration.
"""

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic_ns
from typing import Any

from crypto_arbitrage_bot.candidate_forward_research import run_due_once
from crypto_arbitrage_bot.config.settings import Settings, load_settings
from crypto_arbitrage_bot.research_control import FileStopSignal, _database_lock
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.study_registry import full_research_policy
from crypto_arbitrage_bot.utils.atomic_publish import publish_state
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.file_lock import ControlError, file_lock
from crypto_arbitrage_bot.utils.process_probe import (
    ProcessObservation,
    ProcessState,
    observe_process,
)

Probe = Callable[..., ProcessObservation]
Cycle = Callable[[Settings, Path], dict[str, object]]
_launched_processes: list[subprocess.Popen[bytes]] = []


@dataclass(frozen=True)
class LegacyJob:
    directory: Path
    binding: dict[str, Any]
    state: dict[str, Any]
    settings: Settings
    started: datetime
    deadline: datetime
    database: Path


def _at(value: str) -> datetime:
    return utc_timestamp(datetime.fromisoformat(value))


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = dict(items)
        if len(result) != len(items):
            raise ValueError("duplicate operational fields")
        return result

    try:
        with path.open("rb") as stream:
            body = stream.read(256001)
        value = json.loads(body, object_pairs_hook=unique)
        if len(body) > 256000 or not isinstance(value, dict):
            raise ValueError("bounded operational object required")
        return dict(value)
    except (OSError, ValueError, TypeError, RecursionError) as error:
        raise ControlError("INVALID_LEGACY_CONTROL_RECORD") from error


def _save(directory: Path, state: dict[str, Any]) -> None:
    with (directory / "state.tmp").open("wb") as stream:
        stream.write((exact_json(state) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    publish_state(directory / "state.tmp", directory / "state.json")


def _binding(directory: Path) -> dict[str, Any]:
    binding = _read(directory / "recovery-binding.json")
    try:
        duration, poll, limit = (
            binding[n]
            for n in ("original_duration_seconds", "poll_seconds", "original_max_batches")
        )
        digest = binding["config_sha256"]
        if (
            type(binding["schema_version"]) is not int
            or binding["schema_version"] != 1
            or type(duration) is not int
            or not 1 <= duration <= 8 * 86400
            or type(poll) is not int
            or not 1 <= poll <= 60
            or type(limit) is not int
            or not 1 <= limit <= 30000
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or not Path(binding["config_path"]).is_absolute()
            or not Path(binding["sqlite_path"]).is_absolute()
            or _at(binding["original_deadline"]) - _at(binding["original_started_at"])
            != timedelta(seconds=duration)
        ):
            raise ValueError("original identity and bounded budget required")
        if (directory / "plan.json").exists():
            raise ValueError("a managed research job is not a legacy watch")
        return binding
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise ControlError("LEGACY_IDENTITY_OR_BUDGET_CHANGED") from error


def _receipt(job: LegacyJob) -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "BOUND_LEGACY_PUBLIC_VALIDATION",
        "binding_sha256": _sha(exact_json(job.binding).encode("utf-8")),
        "job_directory": str(job.directory),
        "database": str(job.database),
        "working_directory": str(Path.cwd().resolve()),
        "policy_sha256": _sha(full_research_policy(job.settings).encode("utf-8")),
        "meaning": "Operational recovery only; not a preregistered strategy or trading permission.",
    }


def load_legacy_job(directory: Path) -> LegacyJob:
    """Require existing binding/state/database; never infer a new time or budget."""
    directory = directory.resolve()
    binding = _binding(directory)
    try:
        config = Path(binding["config_path"])
        if _sha(config.read_bytes()) != binding["config_sha256"]:
            raise ValueError("original config changed")
        settings = load_settings(config)
        research = settings.research
        database = Path(binding["sqlite_path"])
        if (
            settings.testnet is not None
            or research is None
            or research.candidate_catalog is None
            or research.validation_scheduler is None
            or research.sqlite_path.resolve() != database
            or not database.is_file()
        ):
            raise ValueError("original public candidate validation database required")
        state = _read(directory / "state.json")
        started, deadline = _at(binding["original_started_at"]), _at(binding["original_deadline"])
        used = state["completed_batches"]
        if (
            state["state"] not in ("RUNNING", "STOPPED", "INTERRUPTED", "FINISHED")
            or _at(state["started_at"]) != started
            or _at(state["original_deadline"]) != deadline
            or type(state["duration_seconds"]) is not int
            or type(state["poll_seconds"]) is not int
            or state["duration_seconds"] != binding["original_duration_seconds"]
            or state["poll_seconds"] != binding["poll_seconds"]
            or Path(state["sqlite_path"]).resolve() != database
            or type(used) is not int
            or not 0 <= used <= binding["original_max_batches"]
            or _at(state["updated_at"]) < started
        ):
            raise ValueError("legacy state differs from its binding")
        for name in ("failed_batches", "task_events", "interrupted_batches", "recovery_count"):
            value = state.get(name, 0)
            if type(value) is not int or value < 0:
                raise ValueError("invalid legacy counter")
        active = state.get("active_batch")
        if active is not None and (type(active) is not int or used == 0 or active != used - 1):
            raise ValueError("invalid consumed active batch")
        pid = state["worker_pid"]
        if pid is not None and (type(pid) is not int or not 1 <= pid <= 2**32 - 1):
            raise ValueError("invalid reported worker PID")
        job = LegacyJob(directory, binding, state, settings, started, deadline, database)
        path = directory / "product-control.json"
        if path.exists() and _read(path) != _receipt(job):
            raise ValueError("operational policy or working directory changed")
        return job
    except ControlError:
        raise
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        raise ControlError("LEGACY_IDENTITY_OR_POLICY_CHANGED") from error


def matches_legacy_worker(args: tuple[str, ...], directory: Path, binding: dict[str, Any]) -> bool:
    """Recognize both the old helper and its product replacement, never a reused PID."""
    try:
        if Path(args[0]).resolve() != Path(sys.executable).resolve():
            return False
        product = (
            "-m",
            "crypto_arbitrage_bot.research_control",
            "_legacy_worker",
            "--job-dir",
            str(directory.resolve()),
        )
        foreground = product[:2] + ("resume",) + product[3:]
        if args[1:] in (product, foreground):
            return True
        expected = (
            "-u",
            str(Path("work/candidate-forward/resume.py").resolve()),
            "--config",
            str(Path(binding["config_path"]).resolve()),
            "--output-dir",
            str(directory.resolve()),
            "--original-config-sha256",
            binding["config_sha256"],
            "--original-max-batches",
            str(binding["original_max_batches"]),
        )
        # Paths in the already-running helper can be relative to its original project cwd.
        if len(args[1:]) != len(expected):
            return False
        actual = list(args[1:])
        for index in (1, 3, 5):
            actual[index] = str(Path(actual[index]).resolve())
        return tuple(actual) == expected
    except (IndexError, KeyError, TypeError, OSError, ValueError):
        return False


def _unowned(job: LegacyJob, probe: Probe) -> None:
    pid = job.state.get("worker_pid")
    if pid is None and job.state["state"] == "RUNNING":
        raise ControlError("LEGACY_OWNER_NOT_CONFIRMED_MISSING")
    if pid is not None:
        observation = probe(
            pid,
            matches=lambda args: matches_legacy_worker(args, job.directory, job.binding),
            not_before=job.started,
            not_after=_at(job.state["updated_at"]),
        )
        if observation.state is not ProcessState.MISSING:
            raise ControlError("LEGACY_OWNER_NOT_CONFIRMED_MISSING")
    pending = job.directory / "launch.json"
    if pending.exists():
        receipt = _read(pending)
        try:
            pending_pid, requested = receipt["pid"], _at(receipt["requested_at"])
            if (
                type(pending_pid) is not int
                or not 1 <= pending_pid <= 2**32 - 1
                or requested < job.started
                or requested > utc_now()
            ):
                raise ValueError("invalid launch receipt")
        except (ValueError, TypeError, KeyError) as error:
            raise ControlError("INVALID_LEGACY_LAUNCH_RECEIPT") from error
        if pending_pid != os.getpid():
            observation = probe(
                pending_pid,
                matches=lambda args: matches_legacy_worker(args, job.directory, job.binding),
                not_before=requested,
                not_after=utc_now(),
            )
            # Windows venv redirectors remain the actual worker's parent. They
            # are not a second worker; permit only our own matched native parent.
            own_launcher = pending_pid == os.getppid() and observation.state is ProcessState.MATCHED
            if observation.state is not ProcessState.MISSING and not own_launcher:
                raise ControlError("LEGACY_LAUNCH_NOT_CONFIRMED_MISSING")


def request_legacy_stop(directory: Path) -> dict[str, object]:
    """Changed or missing config must not prevent a cooperative stop request."""
    binding = _binding(directory.resolve())
    (directory / "stop").touch(exist_ok=True)
    return {
        "status": "STOP_REQUESTED",
        "kind": "LEGACY_PUBLIC_VALIDATION",
        "original_deadline": binding["original_deadline"],
        "orders_submitted": False,
    }


def _pin(job: LegacyJob) -> None:
    path = job.directory / "product-control.json"
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(exact_json(_receipt(job)).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())


def resume_legacy_job(directory: Path, *, probe: Probe = observe_process) -> None:
    """Explicit resume clears stop only after ownership/identity checks; no IO cycle."""
    initial = load_legacy_job(directory)
    with file_lock(initial.directory / "worker.lock"), file_lock(_database_lock(initial.database)):
        job = load_legacy_job(directory)
        _unowned(job, probe)
        _pin(job)
        (directory / "stop").unlink(missing_ok=True)


def run_legacy_job(
    directory: Path,
    *,
    clock: Callable[[], datetime] = utc_now,
    steady_ns: Callable[[], int] = monotonic_ns,
    probe: Probe = observe_process,
    cycle: Cycle = run_due_once,
    wait: Callable[[float], bool] | None = None,
) -> dict[str, Any]:
    initial = load_legacy_job(directory)
    with file_lock(initial.directory / "worker.lock"), file_lock(_database_lock(initial.database)):
        job = load_legacy_job(directory)
        _unowned(job, probe)
        _pin(job)
        state = job.state
        now = utc_timestamp(clock())
        if now < job.started or _at(state["updated_at"]) > now:
            raise ControlError("CLOCK_BEFORE_ORIGINAL_RESEARCH_STATE")
        limit, poll = job.binding["original_max_batches"], job.binding["poll_seconds"]
        signal = FileStopSignal(job.directory / "stop")
        used = state["completed_batches"]
        # Retain pre-crash output, even when the old heartbeat did not claim it.
        while used < limit and (job.directory / f"batch-{used:05d}.json").exists():
            used += 1
        state["completed_batches"] = used
        if now >= job.deadline or used >= limit or signal.is_set():
            state.update(
                state="STOPPED" if signal.is_set() else "FINISHED",
                updated_at=now,
                ended_at=now,
                worker_pid=None,
            )
            _save(job.directory, state)
            return state
        if state.get("active_batch") is not None:
            state["interrupted_batches"] = state.get("interrupted_batches", 0) + 1
            state["active_batch"] = None
            state["next_batch_at"] = now + timedelta(seconds=poll)
        next_at = state.get("next_batch_at")
        if next_at is not None:
            next_at = _at(str(next_at))
            if not job.started <= next_at <= now + timedelta(seconds=poll):
                raise ControlError("INVALID_LEGACY_NEXT_BATCH_TIME")
        delta = job.deadline - now
        remaining_ns = (
            (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
        ) * 1000
        monotonic_deadline = steady_ns() + remaining_ns
        wait = signal.wait if wait is None else wait
        state.update(
            state="RUNNING",
            ended_at=None,
            updated_at=now,
            resumed_at=now,
            worker_pid=os.getpid(),
            original_deadline=job.deadline,
            recovery_count=state.get("recovery_count", 0) + 1,
        )
        _save(job.directory, state)
        try:
            while not signal.is_set() and state["completed_batches"] < limit:
                now = utc_timestamp(clock())
                if now < job.started:
                    raise ControlError("CLOCK_BEFORE_ORIGINAL_START")
                remaining = min(
                    (job.deadline - now).total_seconds(),
                    (monotonic_deadline - steady_ns()) / 1_000_000_000,
                )
                if remaining <= 0:
                    break
                if next_at is not None and now < next_at:
                    if wait(min((next_at - now).total_seconds(), remaining)):
                        signal.set()
                    continue
                index = state["completed_batches"]
                state.update(
                    completed_batches=index + 1,
                    active_batch=index,
                    updated_at=now,
                    next_batch_at=now + timedelta(seconds=poll),
                )
                _save(job.directory, state)  # Consumption before scheduler/network/SQLite IO.
                if (utc_timestamp(clock()) >= job.deadline or steady_ns() >= monotonic_deadline
                        or signal.is_set()):
                    state["interrupted_batches"] = state.get("interrupted_batches", 0) + 1
                    state["active_batch"] = None
                    break
                try:
                    result = cycle(job.settings, job.directory / f"batch-{index:05d}.json")
                    events = result.get("task_events")
                    if not isinstance(events, (tuple, list)):
                        raise ValueError("structured task events required")
                    state["task_events"] = state.get("task_events", 0) + len(events)
                    state["last_error"] = None
                except Exception as error:
                    state["failed_batches"] = state.get("failed_batches", 0) + 1
                    state["last_error"] = {"type": type(error).__name__}
                # Ordinary failures also consumed their batch; keep a full poll interval.
                state["active_batch"] = None
                now = utc_timestamp(clock())
                next_at = now + timedelta(seconds=poll)
                state.update(updated_at=now, next_batch_at=next_at)
                _save(job.directory, state)
            state["state"] = "STOPPED" if signal.is_set() else "FINISHED"
        except KeyboardInterrupt:
            state["state"] = "INTERRUPTED"  # Preserve any active pre-IO claim.
        except BaseException:
            state["state"] = "INTERRUPTED"
            raise
        finally:
            state.update(
                ended_at=utc_timestamp(clock()), updated_at=utc_timestamp(clock()), worker_pid=None
            )
            _save(job.directory, state)
        return state


def launch_legacy_job(directory: Path, *, probe: Probe = observe_process) -> int:
    """No shell, credentials, new budget or automatic stop-marker removal."""
    initial = load_legacy_job(directory)
    with file_lock(initial.directory / "launch.lock"):
        with (
            file_lock(initial.directory / "worker.lock"),
            file_lock(_database_lock(initial.database)),
        ):
            job = load_legacy_job(directory)
            _unowned(job, probe)
            _pin(job)
            if (job.directory / "stop").exists():
                raise ControlError("LEGACY_STOP_REQUESTED")
        # Release worker/DB locks before Popen: the child must acquire them itself.
        command = [
            sys.executable,
            "-m",
            "crypto_arbitrage_bot.research_control",
            "_legacy_worker",
            "--job-dir",
            str(job.directory),
        ]
        requested_at = utc_now()
        with (job.directory / "worker.log").open("ab") as log:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                start_new_session=sys.platform != "win32",
            )
        _launched_processes.append(process)
        with (job.directory / "launch.tmp").open("wb") as stream:
            stream.write(
                exact_json({"pid": process.pid, "requested_at": requested_at}).encode("utf-8")
            )
            stream.flush()
            os.fsync(stream.fileno())
        (job.directory / "launch.tmp").replace(job.directory / "launch.json")
        return process.pid
