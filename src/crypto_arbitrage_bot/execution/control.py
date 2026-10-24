"""Finite TESTNET lifecycle; original bot owns orders, recovery and admission.

Control budgets never replace the original journal's entry/recovery budgets.
Status/stop and an empty managed-only run require no credentials or network.
"""

import argparse
import hashlib
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from threading import Event
from time import monotonic_ns
from typing import Any
from uuid import uuid4

from crypto_arbitrage_bot.config.settings import Settings, load_settings
from crypto_arbitrage_bot.execution.credential_store import (
    DEFAULT_CREDENTIAL_FILE,
    execution_environment,
)
from crypto_arbitrage_bot.execution.risk_review import execution_policy_fingerprint
from crypto_arbitrage_bot.execution.runner import (
    TestnetBot,
    create_testnet_bot,
    process_lock,
)
from crypto_arbitrage_bot.research_control import (
    FileStopSignal,
    _at,
    _exclusive,
    _read,
    _save_state,
    _sha,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.file_lock import (
    ControlError,
    file_lock,
    lock_is_held,
)
from crypto_arbitrage_bot.utils.logging import configure_logging
from crypto_arbitrage_bot.utils.process_probe import (
    ProcessObservation,
    ProcessState,
    observe_process,
)


class ControlMode(StrEnum):
    MANAGE_ONLY = "MANAGE_ONLY"
    AUTOMATIC = "AUTOMATIC"


TERMINAL = frozenset({"FINISHED", "STOPPED", "INTERRUPTED", "RECOVERY_REQUIRED", "FAILED"})
Probe = Callable[..., ProcessObservation]
Factory = Callable[[Settings, Path], TestnetBot]
_launched_processes: list[subprocess.Popen[bytes]] = []


def _marker(journal: Path) -> Path:
    return journal.with_suffix(journal.suffix + ".testnet-control.json")


def _journal(settings: Settings) -> dict[str, Any]:
    from crypto_arbitrage_bot.runtime_status import testnet_status

    value = testnet_status(settings)
    if value["status"] != "READ" or "UNKNOWN_ATTEMPT_STATE" in value.get("reason_codes", ()):
        raise ControlError("UNREADABLE_OR_UNKNOWN_TESTNET_JOURNAL")
    return value


def _command(directory: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "crypto_arbitrage_bot.execution.control",
        "_worker",
        "--job-dir",
        str(directory.resolve()),
    ]


def _unowned(directory: Path, plan: dict[str, Any], state: dict[str, Any], probe: Probe) -> None:
    def matches(args: tuple[str, ...]) -> bool:
        return matches_worker(args, directory)

    pid = state["worker_pid"]
    if pid is None and state["status"] in ("RUNNING", "DRAINING"):
        raise ControlError("TESTNET_OWNER_UNVERIFIED")
    if pid is not None:
        observation = probe(
            pid,
            matches=matches,
            not_before=_at(plan["created_at"]),
            not_after=_at(state["updated_at"]),
        )
        if observation.state is not ProcessState.MISSING:
            raise ControlError("TESTNET_OWNER_NOT_CONFIRMED_MISSING")
    pending = directory / "launch.json"
    if pending.exists():
        receipt = _read(pending)
        requested = _at(receipt["requested_at"])
        pid = receipt["pid"]
        if (
            type(pid) is not int
            or not 1 <= pid <= 2**32 - 1
            or not _at(plan["created_at"]) <= requested <= utc_now()
        ):
            raise ControlError("INVALID_TESTNET_LAUNCH_RECEIPT")
        if pid != os.getpid():
            observation = probe(pid, matches=matches, not_before=requested, not_after=utc_now())
            own_parent = pid == os.getppid() and observation.state is ProcessState.MATCHED
            if observation.state is not ProcessState.MISSING and not own_parent:
                raise ControlError("TESTNET_LAUNCH_UNVERIFIED")


def matches_worker(arguments: tuple[str, ...], directory: Path) -> bool:
    """Single exact command identity shared with read-only supervision."""
    return arguments == tuple(_command(directory))


def load_job(directory: Path) -> tuple[dict[str, Any], dict[str, Any], Settings]:
    directory = directory.resolve()
    try:
        plan, state = _read(directory / "plan.json"), _read(directory / "state.json")
        if (
            hashlib.sha256(Path(plan["config_path"]).read_bytes()).hexdigest()
            != plan["config_sha256"]
        ):
            raise ValueError("original complete configuration changed")
        settings = load_settings(Path(plan["config_path"]))
        config = settings.testnet
        if config is None or settings.research is None:
            raise ValueError("explicit TESTNET configuration required")
        marker = _read(_marker(config.journal_path.resolve()))
        if (
            plan["schema_version"] != 1
            or plan["environment"] != "TESTNET"
            or plan["mode"] not in ControlMode
            or state["job_id"] != plan["job_id"]
            or plan["journal"] != str(config.journal_path.resolve())
            or plan["research_database"] != str(settings.research.sqlite_path.resolve())
            or plan["working_directory"] != str(Path.cwd().resolve())
            or plan["policy_sha256"] != execution_policy_fingerprint(settings)
            or plan["max_total_attempts"] != config.max_total_attempts
            or plan["poll_seconds"] != settings.poll_interval_seconds
            or marker != {"directory": str(directory), "plan_sha256": _sha(exact_json(plan))}
            or type(plan["max_cycles"]) is not int
            or not 1 <= plan["max_cycles"] <= 30000
            or not timedelta(0)
            < _at(plan["ends_at"]) - _at(plan["created_at"])
            <= timedelta(days=8)
            or state["status"] not in TERMINAL | {"PREPARED", "RUNNING", "DRAINING"}
            or type(state["cycles_started"]) is not int
            or not 0 <= state["cycles_started"] <= plan["max_cycles"]
            or type(state["cycles_completed"]) is not int
            or not 0 <= state["cycles_completed"] <= state["cycles_started"]
            or type(state["failed_cycles"]) is not int
            or not 0 <= state["failed_cycles"] <= state["cycles_started"]
            or type(state["interrupted_cycles"]) is not int
            or not 0 <= state["interrupted_cycles"] <= state["cycles_started"]
            or state["active_cycle"] is not None
            and (
                type(state["active_cycle"]) is not int
                or state["active_cycle"] != state["cycles_started"] - 1
            )
            or state["cycles_started"]
            != state["cycles_completed"]
            + state["failed_cycles"]
            + state["interrupted_cycles"]
            + (state["active_cycle"] is not None)
            or _at(state["updated_at"]) < _at(plan["created_at"])
        ):
            raise ValueError("job/config/state identity differs")
        if state["worker_pid"] is not None and (
            type(state["worker_pid"]) is not int or not 1 <= state["worker_pid"] <= 2**32 - 1
        ):
            raise ValueError("invalid owner PID")
        return plan, state, settings
    except ControlError:
        raise
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        raise ControlError("TESTNET_JOB_IDENTITY_OR_POLICY_CHANGED") from error


def prepare_job(
    config_path: Path,
    directory: Path,
    *,
    duration_seconds: int,
    max_cycles: int,
    mode: ControlMode = ControlMode.MANAGE_ONLY,
    credentials_file: Path = DEFAULT_CREDENTIAL_FILE,
    clock: Callable[[], datetime] = utc_now,
    probe: Probe = observe_process,
) -> dict[str, Any]:
    if (
        not isinstance(mode, ControlMode)
        or type(duration_seconds) is not int
        or not 1 <= duration_seconds <= 8 * 86400
        or type(max_cycles) is not int
        or not 1 <= max_cycles <= 30000
    ):
        raise ControlError("EXPLICIT_BOUNDED_TESTNET_CONTROL_REQUIRED")
    settings = load_settings(config_path)
    config = settings.testnet
    if config is None or settings.research is None or not config.journal_path.is_file():
        raise ControlError("EXISTING_TESTNET_JOURNAL_REQUIRED")
    directory = directory.resolve()
    with process_lock(config.journal_path.with_suffix(".lock")):
        journal = _journal(settings)
        if mode is ControlMode.AUTOMATIC:
            if config.admission is None:
                raise ControlError("AUTOMATIC_STRATEGY_ADMISSION_NOT_CONFIGURED")
            if journal["budget_consumed"] >= config.max_total_attempts:
                raise ControlError("ORIGINAL_TESTNET_ATTEMPT_BUDGET_EXHAUSTED")
        marker = _marker(config.journal_path.resolve())
        if marker.exists():
            old_directory = Path(_read(marker)["directory"])
            old_plan, old_state, old_settings = load_job(old_directory)
            if old_state["status"] not in TERMINAL:
                raise ControlError("EXISTING_TESTNET_JOB_REQUIRES_RESUME")
            _unowned(old_directory, old_plan, old_state, probe)
            if execution_policy_fingerprint(old_settings) != execution_policy_fingerprint(settings):
                raise ControlError("ORIGINAL_TESTNET_POLICY_CHANGED")
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise ControlError("NEW_TESTNET_CONTROL_DIRECTORY_REQUIRED")
        now = utc_timestamp(clock())
        plan = {
            "schema_version": 1,
            "environment": "TESTNET",
            "job_id": uuid4().hex,
            "mode": mode.value,
            "config_path": str(config_path.resolve()),
            "working_directory": str(Path.cwd().resolve()),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "journal": str(config.journal_path.resolve()),
            "research_database": str(settings.research.sqlite_path.resolve()),
            "credentials_file": str(credentials_file.resolve()),
            "created_at": now,
            "ends_at": now + timedelta(seconds=duration_seconds),
            "max_cycles": max_cycles,
            "poll_seconds": settings.poll_interval_seconds,
            "max_total_attempts": config.max_total_attempts,
            "policy_sha256": execution_policy_fingerprint(settings),
        }
        _exclusive(directory / "plan.json", plan)
        _save_state(
            directory,
            {
                "job_id": plan["job_id"],
                "status": "PREPARED",
                "worker_pid": None,
                "updated_at": now,
                "cycles_started": 0,
                "cycles_completed": 0,
                "active_cycle": None,
                "interrupted_cycles": 0,
                "failed_cycles": 0,
                "last_error_type": None,
            },
        )
        temporary = marker.with_suffix(marker.suffix + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(
                exact_json(
                    {"directory": str(directory), "plan_sha256": _sha(exact_json(plan))}
                ).encode("utf-8")
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(marker)
        return plan


def status(
    directory: Path,
    *,
    max_heartbeat_age_seconds: int,
    clock: Callable[[], datetime] = utc_now,
    probe: Probe = observe_process,
) -> dict[str, Any]:
    from crypto_arbitrage_bot.runtime_status import _owner

    if type(max_heartbeat_age_seconds) is not int or not 1 <= max_heartbeat_age_seconds <= 86400:
        raise ControlError("EXPLICIT_TESTNET_HEARTBEAT_THRESHOLD_REQUIRED")
    plan, state, settings = load_job(directory)
    directory = directory.resolve()
    owner = _owner(
        directory / "worker.lock",
        state["worker_pid"],
        _at(plan["created_at"]),
        _at(state["updated_at"]),
        utc_timestamp(clock()),
        timedelta(seconds=max_heartbeat_age_seconds),
        lambda args: tuple(_command(directory)) == args,
        probe,
        lock_is_held,
        "RUNNING" if state["status"] in ("RUNNING", "DRAINING") else state["status"],
    )
    return {
        "plan": plan,
        "state": state,
        "owner": owner,
        "stop_requested": (directory / "stop").exists(),
        "journal": _journal(settings),
        "credentials_read": False,
        "orders_submitted": False,
    }


def request_stop(directory: Path) -> dict[str, object]:
    """Stop entry, drain managed plans within the original deadline; never kill."""
    directory = directory.resolve()
    plan = _read(directory / "plan.json")
    marker = _read(_marker(Path(plan["journal"])))
    if marker != {"directory": str(directory), "plan_sha256": _sha(exact_json(plan))}:
        raise ControlError("TESTNET_CONTROL_IDENTITY_CHANGED")
    (directory / "stop").touch(exist_ok=True)
    return {
        "status": "STOP_REQUESTED",
        "job_id": plan["job_id"],
        "meaning": "New entries blocked; existing plans remain managed within original control budget.",
        "orders_submitted": False,
    }


def resume_job(directory: Path, *, probe: Probe = observe_process) -> None:
    plan, _, settings = load_job(directory)
    assert settings.testnet is not None
    with (
        file_lock(directory / "worker.lock"),
        process_lock(settings.testnet.journal_path.with_suffix(".lock")),
    ):
        plan, state, _ = load_job(directory)
        _unowned(directory, plan, state, probe)
        (directory / "stop").unlink(missing_ok=True)


def _factory(settings: Settings, credentials_file: Path) -> TestnetBot:
    if os.environ.get("CAB_TESTNET_ORDERS_ENABLED") != "true":
        raise ControlError("TESTNET_ORDERS_DISABLED")
    return create_testnet_bot(settings, environ=execution_environment(credentials_file, os.environ))


def launch_job(directory: Path, *, probe: Probe = observe_process) -> int:
    """Hidden child inherits the caller's proxy environment; no order gate changes."""
    directory = directory.resolve()
    with file_lock(directory / "launch.lock"):
        with file_lock(directory / "worker.lock"):
            plan, state, settings = load_job(directory)
            assert settings.testnet is not None
            with process_lock(settings.testnet.journal_path.with_suffix(".lock")):
                _unowned(directory, plan, state, probe)
        # Child acquires its own worker/journal locks; launch.lock serializes launchers.
        requested = utc_now()
        with (directory / "worker.log").open("ab") as log:
            process = subprocess.Popen(
                _command(directory),
                cwd=plan["working_directory"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                start_new_session=sys.platform != "win32",
            )
        _launched_processes.append(process)
        receipt = {"pid": process.pid, "requested_at": requested}
        temporary = directory / "launch.tmp"
        with temporary.open("wb") as stream:
            stream.write(exact_json(receipt).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(directory / "launch.json")
        return process.pid


def run_job(
    directory: Path,
    *,
    clock: Callable[[], datetime] = utc_now,
    steady_ns: Callable[[], int] = monotonic_ns,
    probe: Probe = observe_process,
    factory: Factory = _factory,
    wait: Callable[[float], bool] | None = None,
) -> dict[str, Any]:
    directory = directory.resolve()
    plan, _, settings = load_job(directory)
    assert settings.testnet is not None
    with file_lock(directory / "worker.lock"):
        plan, state, settings = load_job(directory)
        _unowned(directory, plan, state, probe)
        now = utc_timestamp(clock())
        if now < _at(state["updated_at"]):
            raise ControlError("CLOCK_BEFORE_TESTNET_JOB_STATE")
        # Before constructing clients, verify the original journal is available.
        with process_lock(Path(plan["journal"]).with_suffix(".lock")):
            journal = _journal(settings)
            stopped = (directory / "stop").exists()
            if (
                (plan["mode"] == ControlMode.MANAGE_ONLY or stopped)
                and not journal["active_attempts"]
                or now >= _at(plan["ends_at"])
                or state["cycles_started"] >= plan["max_cycles"]
            ):
                state.update(
                    status="RECOVERY_REQUIRED"
                    if journal["active_attempts"]
                    else ("STOPPED" if stopped else "FINISHED"),
                    completion_reason="LOCAL_JOURNAL_NO_ACTIVE_ATTEMPTS"
                    if not journal["active_attempts"]
                    else "CONTROL_BUDGET_ENDED_WITH_ACTIVE_ATTEMPTS",
                    worker_pid=None,
                    updated_at=now,
                )
                _save_state(directory, state)
                return state
        end, start = _at(plan["ends_at"]), _at(plan["created_at"])
        delta = end - now
        deadline = (
            steady_ns()
            + ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1000
        )
        signal = FileStopSignal(directory / "stop")
        wait = Event().wait if wait is None else wait
        next_at = state.get("next_cycle_at")
        next_at = None if next_at is None else _at(str(next_at))
        if state["active_cycle"] is not None:
            state["interrupted_cycles"] += 1
            state["active_cycle"] = None
            next_at = now + timedelta(seconds=plan["poll_seconds"])
        if next_at is not None and not start <= next_at <= now + timedelta(
            seconds=plan["poll_seconds"]
        ):
            raise ControlError("INVALID_TESTNET_NEXT_CYCLE_TIME")

        def entry_stopped() -> bool:
            at = utc_timestamp(clock())
            return signal.is_set() or not start <= at < end or steady_ns() >= deadline

        try:
            bot = factory(settings, Path(plan["credentials_file"]))
            if (
                bot.config.journal_path.resolve() != Path(plan["journal"])
                or execution_policy_fingerprint(bot.settings) != plan["policy_sha256"]
            ):
                raise ControlError("TESTNET_BOT_DOES_NOT_MATCH_JOB")
            with bot.controlled_session(
                manage_only=plan["mode"] == ControlMode.MANAGE_ONLY, entry_stop=entry_stopped
            ):
                state.update(
                    status="RUNNING", worker_pid=os.getpid(), updated_at=utc_timestamp(clock())
                )
                _save_state(directory, state)
                while state["cycles_started"] < plan["max_cycles"]:
                    now = utc_timestamp(clock())
                    remaining = min(
                        (end - now).total_seconds(), (deadline - steady_ns()) / 1_000_000_000
                    )
                    if now < start or remaining <= 0:
                        break
                    journal = _journal(settings)
                    draining = (
                        signal.is_set()
                        or plan["mode"] == ControlMode.MANAGE_ONLY
                        or journal["budget_consumed"] >= plan["max_total_attempts"]
                    )
                    if draining and not journal["active_attempts"]:
                        break
                    if next_at is not None and now < next_at:
                        if wait(min((next_at - now).total_seconds(), remaining)):
                            signal.set()
                        continue
                    index = state["cycles_started"]
                    state.update(
                        status="DRAINING" if draining else "RUNNING",
                        active_cycle=index,
                        cycles_started=index + 1,
                        updated_at=now,
                    )
                    _save_state(directory, state)
                    try:
                        results = bot.manage_existing_once() if draining else bot.run_once()
                        (directory / "cycles").mkdir(exist_ok=True)
                        _exclusive(
                            directory / "cycles" / f"{index:05d}.json",
                            {
                                "started_at": now,
                                "ended_at": utc_timestamp(clock()),
                                "mode": "MANAGE_ONLY" if draining else "ORIGINAL_AUTOMATIC_GATES",
                                "results": [
                                    {
                                        "symbol": result.symbol,
                                        "status": result.status,
                                        "opportunity_id": result.opportunity_id,
                                        "reason": result.reason,
                                        "attempt_id": result.pair.attempt_id
                                        if result.pair
                                        else None,
                                    }
                                    for result in results
                                ],
                            },
                        )
                        state["cycles_completed"] += 1
                        state["last_error_type"] = None
                    except Exception as error:
                        state["failed_cycles"] += 1
                        state["last_error_type"] = type(error).__name__
                    state.update(active_cycle=None, updated_at=utc_timestamp(clock()))
                    next_at = utc_timestamp(clock()) + timedelta(seconds=plan["poll_seconds"])
                    state["next_cycle_at"] = next_at
                    _save_state(directory, state)
                journal = _journal(settings)
                state.update(
                    status="RECOVERY_REQUIRED"
                    if journal["active_attempts"]
                    else ("STOPPED" if signal.is_set() else "FINISHED"),
                    completion_reason="CONTROL_BUDGET_ENDED_WITH_ACTIVE_ATTEMPTS"
                    if journal["active_attempts"]
                    else "LOCAL_JOURNAL_NO_ACTIVE_ATTEMPTS",
                )
        except KeyboardInterrupt:
            state.update(
                status="INTERRUPTED", completion_reason="INTERRUPTED_EXISTING_ATTEMPTS_PRESERVED"
            )
        except Exception as error:
            state.update(
                status="FAILED",
                last_error_type=type(error).__name__,
                last_error_code=str(error)
                if isinstance(error, ControlError)
                else "TESTNET_CONTROL_FAILED",
                completion_reason="CONTROL_FAILURE_REQUIRES_JOURNAL_REVIEW",
            )
        finally:
            state.update(worker_pid=None, updated_at=utc_timestamp(clock()))
            _save_state(directory, state)
        return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded TESTNET control; original strategy/order budgets remain authoritative"
    )
    parser.add_argument("command", choices=("start", "resume", "stop", "status", "_worker"))
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--credentials-file", type=Path, default=DEFAULT_CREDENTIAL_FILE)
    parser.add_argument("--mode", choices=tuple(ControlMode), default=ControlMode.MANAGE_ONLY.value)
    parser.add_argument("--duration-seconds", type=int)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--max-heartbeat-age-seconds", type=int)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args(argv)
    try:
        directory = args.job_dir.resolve()
        if args.command == "status":
            if args.max_heartbeat_age_seconds is None:
                raise ControlError("EXPLICIT_TESTNET_HEARTBEAT_THRESHOLD_REQUIRED")
            print(
                exact_json(
                    status(directory, max_heartbeat_age_seconds=args.max_heartbeat_age_seconds)
                )
            )
            return 0
        if args.command == "stop":
            print(exact_json(request_stop(directory)))
            return 0
        if args.command == "start":
            if args.config is None or args.duration_seconds is None or args.max_cycles is None:
                raise ControlError("EXPLICIT_BOUNDED_TESTNET_CONTROL_REQUIRED")
            prepare_job(
                args.config,
                directory,
                duration_seconds=args.duration_seconds,
                max_cycles=args.max_cycles,
                mode=ControlMode(args.mode),
                credentials_file=args.credentials_file,
            )
        else:
            if (
                args.config is not None
                or args.duration_seconds is not None
                or args.max_cycles is not None
            ):
                raise ControlError("ORIGINAL_TESTNET_JOB_PARAMETERS_FROZEN")
            if args.command == "resume":
                resume_job(directory)
        if args.background:
            if args.command == "_worker":
                raise ControlError("WORKER_CANNOT_LAUNCH_ANOTHER_WORKER")
            print(
                exact_json(
                    {
                        "status": "LAUNCHED",
                        "pid": launch_job(directory),
                        "meaning": "Launch receipt only; inspect status for worker completion.",
                    }
                )
            )
            return 0
        _, _, settings = load_job(directory)
        configure_logging(settings.log_dir, settings.log_level)
        result = run_job(directory)
        print(exact_json(result))
        return 0 if result["status"] in ("FINISHED", "STOPPED") else 1
    except Exception as error:
        print(
            exact_json(
                {
                    "status": "CONTROL_FAILED",
                    "reason": str(error)
                    if isinstance(error, ControlError)
                    else type(error).__name__,
                    "control_command_submitted_orders": False,
                    "worker_execution_requires_status_check": True,
                    "credentials_exposed": False,
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
