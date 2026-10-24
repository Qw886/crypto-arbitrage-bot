"""Finite AC-only sleep-inhibition guard attached to one existing monitor owner.

No research restart, task replay, orders, global policy changes or display request.
Manual sleep and network availability remain outside this request's guarantee.
"""

import argparse
import os
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event
from time import monotonic_ns
from typing import Any, Protocol

from crypto_arbitrage_bot import monitor_archive as archive
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.file_lock import file_lock, lock_is_held
from crypto_arbitrage_bot.utils.process_probe import ProcessState, observe_process
from crypto_arbitrage_bot.utils.windows_power import WindowsPower

INTERVAL = 15
MAX_GAP = 60


class Power(Protocol):
    def ac_available(self) -> bool: ...
    def system_request(self, reason: str) -> AbstractContextManager[None]: ...


def target(directory: Path, sha: str, now: datetime, *,
           probe: archive.Probe = observe_process) -> dict[str, Any]:
    plan, _ = archive.load_plan(directory, sha)
    binding = archive.load_binding(directory, sha)
    progress = archive.read(directory / "progress.json")
    if (progress["plan_sha256"] != sha or progress["state"] != "RUNNING"
            or progress["mode"] not in ("LEGACY", "RESUMED")
            or type(progress["checks"]) is not int or not 0 <= progress["checks"] < plan["max_checks"]
            or not archive.at(plan["started_at"]) <= now < archive.at(plan["ends_at"])
            or not timedelta(0) <= now-archive.at(progress["updated_at"]) <= timedelta(
                seconds=binding["spec"]["max_heartbeat_age_seconds"])
            or type(progress["worker_pid"]) is not int or not 1 <= progress["worker_pid"] <= 2**32-1
            or not lock_is_held(directory / "monitor.lock")):
        raise ValueError("MONITOR_NOT_LIVE_WITHIN_ORIGINAL_BUDGET")
    created = progress["worker_created_at"]
    if created is None and progress["mode"] == "LEGACY" and progress["worker_pid"] == binding["legacy_pid"]:
        created = binding["legacy_created_at"]
    birth = archive.at(created)
    if birth > now:
        raise ValueError("MONITOR_BIRTH_IN_FUTURE")
    matches = (lambda args: archive.matches_resumed(args, directory, sha)) if progress["mode"] == "RESUMED" else (
        lambda args: archive.matches_legacy(args, directory, binding["spec"], plan))
    observation = probe(progress["worker_pid"], matches=matches, not_before=birth, not_after=birth)
    if observation.state is not ProcessState.MATCHED or observation.created_at != birth:
        raise ValueError("MONITOR_NATIVE_OWNER_UNCONFIRMED")
    return {"pid": progress["worker_pid"], "created_at": birth.isoformat(),
            "ends_at": plan["ends_at"], "binding_sha256": archive.digest(
                (directory / "monitor-binding.json").read_bytes())}


def hold(directory: Path, sha: str, output: Path, *, power: Power,
         clock: Callable[[], datetime] = utc_now,
         steady_ns: Callable[[], int] = monotonic_ns,
         wait: Callable[[float], object] | None = None,
         probe: archive.Probe = observe_process) -> dict[str, Any]:
    directory, output = directory.resolve(), output.resolve()
    if output == directory or output in directory.parents or directory in output.parents:
        raise ValueError("SEPARATE_NEW_GUARD_DIRECTORY_REQUIRED")
    now = utc_timestamp(clock())
    pinned = target(directory, sha, now, probe=probe)
    end = archive.at(pinned["ends_at"])
    start_ns = steady_ns()
    deadline_ns = start_ns + int((end-now).total_seconds()*1_000_000_000)
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / "guard.lock"):
        if any(p.name != "guard.lock" for p in output.iterdir()):
            raise ValueError("NEW_GUARD_DIRECTORY_REQUIRED_NO_RESUME")
        registered = utc_timestamp(clock())
        owner = probe(os.getpid(), matches=lambda args: matches_guard(args, directory, sha, output),
                      not_before=archive.EPOCH, not_after=registered)
        if owner.state is not ProcessState.MATCHED or owner.created_at is None:
            raise ValueError("GUARD_NATIVE_OWNER_UNCONFIRMED")
        binding = {"schema_version": 1, "monitor_directory": str(directory), "plan_sha256": sha,
                   "target": pinned, "started_at": now, "worker_pid": os.getpid(),
                   "worker_created_at": None if owner.created_at is None else owner.created_at.isoformat(),
                   "native_owner_confirmed": owner.state is ProcessState.MATCHED,
                   "request_type": "SYSTEM_REQUIRED", "ac_only": True, "interval_seconds": INTERVAL,
                   "max_gap_seconds": MAX_GAP, "display_required": False, "global_policy_changed": False}
        archive.write(output / "binding.json", binding, exclusive=True)
        reason = "AC_POWER_UNAVAILABLE"
        pulses = 0
        state: dict[str, Any] = {}
        previous = now
        def save(status: str, code: str) -> dict[str, Any]:
            value = {"status": status, "reason": code, "updated_at": utc_timestamp(clock()),
                     "checks": pulses, "original_deadline": end, "plan_sha256": sha,
                     "worker_pid": os.getpid(), "request_type": "SYSTEM_REQUIRED",
                     "os_request_verified": False, "network_guaranteed": False,
                     "orders_submitted": False, "credentials_read": False}
            archive.write(output / "state.json", value)
            return value
        if wait is None:
            wait = Event().wait
        try:
            if utc_timestamp(clock()) >= end or steady_ns() >= deadline_ns:
                reason = "ORIGINAL_DEADLINE_REACHED"
            elif power.ac_available():
                with power.system_request(f"crypto-arbitrage-bot bounded public research {sha[:12]}"):
                    while True:
                        current = utc_timestamp(clock())
                        elapsed = current-previous
                        if elapsed < timedelta(0) or elapsed > timedelta(seconds=MAX_GAP):
                            reason = "CLOCK_OR_EXECUTION_GAP"
                            break
                        remaining = min((end-current).total_seconds(), (deadline_ns-steady_ns())/1e9)
                        if remaining <= 0:
                            reason = "ORIGINAL_DEADLINE_REACHED"
                            break
                        if (output / "stop").exists():
                            reason = "STOP_REQUESTED"
                            break
                        if not power.ac_available():
                            reason = "AC_POWER_UNAVAILABLE"
                            break
                        if target(directory, sha, current, probe=probe) != pinned:
                            reason = "ORIGINAL_MONITOR_OWNER_CHANGED"
                            break
                        # Native observation may take up to five seconds. Check
                        # AC/deadline again before publishing a held request.
                        if (not power.ac_available() or utc_timestamp(clock()) >= end
                                or steady_ns() >= deadline_ns):
                            reason = "POWER_OR_DEADLINE_CHANGED_DURING_CHECK"
                            break
                        pulses += 1
                        state = save("REQUEST_CREATED", "AC_ONLY_SYSTEM_REQUEST")
                        previous = current
                        # Observation/writing time consumes the original lease;
                        # it is never added to the final wait.
                        remaining = min((end-utc_timestamp(clock())).total_seconds(),
                                        (deadline_ns-steady_ns())/1e9)
                        wait(max(0, min(INTERVAL, remaining)))
            state = save("RELEASED", reason)
        except KeyboardInterrupt:
            state = save("RELEASED", "INTERRUPTED")
        except Exception as error:
            # Context-manager cleanup happens before publishing failure.
            state = save("FAILED", type(error).__name__)
            raise
        return state


def matches_guard(args: tuple[str, ...], directory: Path, sha: str, output: Path) -> bool:
    try:
        parts = args[1:]
        if parts[:1] == ("-u",):
            parts = parts[1:]
        if Path(args[0]).resolve() != Path(sys.executable).resolve() or parts[:3] != (
                "-m", "crypto_arbitrage_bot.power_guard", "hold"):
            return False
        options = parts[3:]
        pairs = tuple(zip(options[::2], options[1::2], strict=True))
        values = dict(pairs)
        if len(values) != len(pairs):
            return False
        return set(values) == {"--monitor-dir", "--original-plan-sha256", "--output-dir"} and (
            Path(values["--monitor-dir"]).resolve() == directory and
            Path(values["--output-dir"]).resolve() == output and values["--original-plan-sha256"] == sha)
    except (IndexError, KeyError, ValueError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finite AC-only system request for an existing monitor")
    parser.add_argument("command", choices=("hold", "stop", "status"))
    parser.add_argument("--monitor-dir", required=True, type=Path)
    parser.add_argument("--original-plan-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    directory, output, sha = args.monitor_dir.resolve(), args.output_dir.resolve(), args.original_plan_sha256
    try:
        if args.command == "hold":
            result = hold(directory, sha, output, power=WindowsPower())
        else:
            binding = archive.read(output / "binding.json")
            if binding["plan_sha256"] != sha or binding["monitor_directory"] != str(directory):
                raise ValueError("GUARD_SCOPE_DIFFERS")
            archive.load_plan(directory, sha)
            if args.command == "stop":
                (output / "stop").touch(exist_ok=True)
                result = {"status": "GUARD_STOP_REQUESTED", "orders_submitted": False}
            else:
                state = archive.read(output / "state.json")
                birth = archive.at(binding["worker_created_at"])
                observed = observe_process(binding["worker_pid"],
                    matches=lambda a: matches_guard(a, directory, sha, output), not_before=birth, not_after=birth)
                now = utc_now()
                alive = observed.state is ProcessState.MATCHED and lock_is_held(output / "guard.lock")
                fresh = timedelta(0) <= now-archive.at(state["updated_at"]) <= timedelta(seconds=MAX_GAP)
                running = (alive and fresh and state["status"] == "REQUEST_CREATED"
                           and now < archive.at(binding["target"]["ends_at"]))
                result = {**state, "evaluated_at": now, "native_owner_confirmed": alive,
                          "heartbeat_fresh": fresh, "status": "RUNNING_CONFIRMED" if running else "OWNER_UNCONFIRMED",
                          "reported_status": state["status"], "process_observation": observed.state}
        print(exact_json(result))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(exact_json({"status": "UNAVAILABLE", "error_type": type(error).__name__,
                          "orders_submitted": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
