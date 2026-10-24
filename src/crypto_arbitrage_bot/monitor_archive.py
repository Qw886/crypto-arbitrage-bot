"""Pinned monitor continuation and durable check claims, no research/SQL writes.

Legacy counters are accepted only after a natively identified monitor exits.
Local hashes detect accidental evidence changes, not a malicious operator
rewriting all archives. No missing counter is inferred from elapsed time.
"""

import errno
import hashlib
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Any

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.file_lock import lock_is_held
from crypto_arbitrage_bot.utils.process_probe import (
    ProcessObservation,
    ProcessState,
    observe_process,
)

Probe = Callable[..., ProcessObservation]
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
MAX_READ_ATTEMPTS = 10
READ_WAIT_SECONDS = .01
MAX_ARCHIVE_BYTES = 256000


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read(path: Path) -> dict[str, Any]:
    # CPython's Windows buffered open can report EACCES without winerror for
    # a short-lived sharing denial. Retry only these OS read failures, never
    # malformed content or business work. Permanent denial still propagates.
    for attempt in range(MAX_READ_ATTEMPTS):
        try:
            with path.open("rb") as stream:
                body = stream.read(MAX_ARCHIVE_BYTES+1)
            break
        except OSError as error:
            code = getattr(error, "winerror", None)
            if (sys.platform != "win32" or not (code in (5, 32, 33) or
                    code is None and error.errno == errno.EACCES) or attempt == MAX_READ_ATTEMPTS-1):
                raise
            sleep(READ_WAIT_SECONDS)
    if len(body) > MAX_ARCHIVE_BYTES:
        raise ValueError("bounded monitor archive required")
    value = json.loads(body)
    if not isinstance(value, dict) or exact_json(value).encode() != body:
        raise ValueError("canonical monitor archive required")
    return dict(value)


def write(path: Path, value: object, *, exclusive: bool = False) -> None:
    target = path if exclusive else path.with_suffix(path.suffix + ".tmp")
    with target.open("xb" if exclusive else "wb") as output:
        output.write(exact_json(value).encode())
        output.flush()
        os.fsync(output.fileno())
    if not exclusive:
        # Windows readers can briefly deny FILE_SHARE_DELETE. Keep the same
        # durable temporary file; never re-run a check or fall back to unlink.
        for attempt in range(10):
            try:
                target.replace(path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                sleep(.01)


def at(value: str) -> datetime:
    return utc_timestamp(datetime.fromisoformat(value))


def load_plan(directory: Path, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    path = directory / "plan.json"
    sha = digest(path.read_bytes())
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError("original monitor plan hash differs")
    plan = read(path)
    if (set(plan) != {"started_at", "ends_at", "interval_seconds", "max_checks", "orders_submitted"}
            or plan["orders_submitted"] is not False
            or not timedelta(0) < at(plan["ends_at"])-at(plan["started_at"]) <= timedelta(days=8)
            or type(plan["interval_seconds"]) is not int or not 1 <= plan["interval_seconds"] <= 3600
            or type(plan["max_checks"]) is not int or not 1 <= plan["max_checks"] <= 30000):
        raise ValueError("invalid original monitor budget")
    return plan, sha


def policy(spec: dict[str, Any]) -> dict[str, Any]:
    if (set(spec) != {"research_config", "research_directory", "testnet_config", "max_heartbeat_age_seconds",
                      "working_directory"}
            or type(spec["max_heartbeat_age_seconds"]) is not int
            or not 1 <= spec["max_heartbeat_age_seconds"] <= 86400
            or (spec["research_config"] is None) != (spec["research_directory"] is None)
            or spec["research_config"] is spec["testnet_config"] is None
            or spec["working_directory"] != str(Path.cwd().resolve())):
        raise ValueError("explicit monitor scope required")
    result = {}
    for name in ("research_config", "testnet_config"):
        target = spec[name]
        if target is not None:
            path = Path(target)
            if not path.is_absolute() or path != path.resolve():
                raise ValueError("absolute monitor configuration required")
            settings = load_settings(path)
            if (name == "research_config" and (settings.research is None or settings.testnet is not None)
                    or name == "testnet_config" and settings.testnet is None):
                raise ValueError("monitor configuration scope differs")
            result[name] = {"file_sha256": digest(path.read_bytes()),
                            "settings_sha256": digest(exact_json(settings.model_dump(mode="json")).encode())}
    if spec["research_directory"] is not None:
        path = Path(spec["research_directory"])
        if not path.is_absolute() or path != path.resolve():
            raise ValueError("absolute research directory required")
    return result


def _arguments(args: tuple[str, ...], command: str) -> dict[str, str]:
    if not args or Path(args[0]).resolve() != Path(sys.executable).resolve():
        raise ValueError("monitor interpreter differs")
    parts = args[1:]
    if parts[:1] == ("-u",):
        parts = parts[1:]
    if parts[:3] != ("-m", "crypto_arbitrage_bot.runtime_status", command):
        raise ValueError("monitor command differs")
    options = parts[3:]
    if len(options) % 2:
        raise ValueError("single monitor option values required")
    pairs = tuple(zip(options[::2], options[1::2], strict=True))
    values = dict(pairs)
    if len(values) != len(pairs):
        raise ValueError("duplicate monitor options")
    return values


def matches_legacy(args: tuple[str, ...], directory: Path, spec: dict[str, Any], plan: dict[str, Any]) -> bool:
    try:
        actual = _arguments(args, "monitor")
        expected = {"--output-dir": str(directory.resolve()),
                    "--duration-seconds": str(int((at(plan["ends_at"])-at(plan["started_at"])).total_seconds())),
                    "--interval-seconds": str(plan["interval_seconds"]), "--max-checks": str(plan["max_checks"]),
                    "--max-heartbeat-age-seconds": str(spec["max_heartbeat_age_seconds"])}
        for key, option in (("research_config", "--research-config"), ("research_directory", "--research-dir"),
                            ("testnet_config", "--testnet-config")):
            if spec[key] is not None:
                expected[option] = spec[key]
        for key in ("--output-dir", "--research-config", "--research-dir", "--testnet-config"):
            if key in actual:
                actual[key] = str(Path(actual[key]).resolve())
        return actual == expected
    except (ValueError, TypeError, IndexError, OSError):
        return False


def matches_resumed(args: tuple[str, ...], directory: Path, sha: str) -> bool:
    try:
        actual = _arguments(args, "resume-monitor")
        actual["--output-dir"] = str(Path(actual["--output-dir"]).resolve())
        return actual == {"--output-dir": str(directory.resolve()), "--original-plan-sha256": sha}
    except (ValueError, TypeError, KeyError, OSError):
        return False


def bind_live(directory: Path, sha: str, pid: int, spec: dict[str, Any], *,
              clock: Callable[[], datetime] = utc_now, probe: Probe = observe_process) -> dict[str, Any]:
    directory = directory.resolve()
    plan, sha = load_plan(directory, sha)
    current_policy = policy(spec)
    if type(pid) is not int or not 1 <= pid <= 2**32-1 or not lock_is_held(directory / "monitor.lock"):
        raise ValueError("actual live monitor owner required")
    now = utc_timestamp(clock())
    observation = probe(pid, matches=lambda args: matches_legacy(args, directory, spec, plan),
                        not_before=EPOCH, not_after=now)
    if observation.state is not ProcessState.MATCHED or observation.created_at is None:
        raise ValueError("legacy monitor identity unconfirmed")
    path = directory / "monitor-binding.json"
    fixed = {"schema_version": 1, "plan_sha256": sha, "spec": spec, "policy": current_policy,
             "legacy_pid": pid, "legacy_created_at": observation.created_at}
    if path.exists():
        original = read(path)
        if any(exact_json(original.get(k)) != exact_json(v) for k, v in fixed.items()):
            raise ValueError("existing monitor binding differs")
        return original
    value = {**fixed, "registered_at": now}
    write(path, value, exclusive=True)
    return value


def load_binding(directory: Path, sha: str, *, check_policy: bool = True) -> dict[str, Any]:
    value = read(directory / "monitor-binding.json")
    if (set(value) != {"schema_version", "plan_sha256", "spec", "policy", "legacy_pid", "legacy_created_at",
                      "registered_at"} or type(value["schema_version"]) is not int
            or value["schema_version"] != 1 or value["plan_sha256"] != sha
            or type(value["legacy_pid"]) is not int or not 1 <= value["legacy_pid"] <= 2**32-1
            or at(value["legacy_created_at"]) > at(value["registered_at"])):
        raise ValueError("monitor binding differs")
    if check_policy and policy(value["spec"]) != value["policy"]:
        raise ValueError("monitor scope or policy changed")
    return value


def require_missing_owner(directory: Path, sha: str, binding: dict[str, Any], *,
                          probe: Probe = observe_process) -> None:
    path = directory / "progress.json"
    progress = read(path) if path.exists() else None
    pid = binding["legacy_pid"] if progress is None else progress["worker_pid"]
    created = binding["legacy_created_at"] if progress is None else progress["worker_created_at"]
    if created is None:
        if pid != binding["legacy_pid"]:
            raise ValueError("monitor owner creation time missing")
        created = binding["legacy_created_at"]
    birth = at(created)
    legacy = progress is None or progress["mode"] == "LEGACY"
    plan, _ = load_plan(directory, sha)
    matches = (lambda args: matches_legacy(args, directory, binding["spec"], plan)) if legacy else (
        lambda args: matches_resumed(args, directory, sha))
    observation = probe(pid, matches=matches, not_before=birth, not_after=birth)
    if observation.state is not ProcessState.MISSING:
        raise ValueError("prior monitor owner not confirmed missing")


def event_state(directory: Path, expected_count: int, *, allow_extra: bool = False) -> tuple[int, str | None]:
    paths = sorted(directory.glob("event-*.json"))
    if len(paths) not in ({expected_count, expected_count+1} if allow_extra else {expected_count}):
        raise ValueError("missing or extra monitor event")
    previous = None
    for index, path in enumerate(paths):
        value = read(path)
        if path.name != f"event-{index:05d}.json" or value["change_index"] != index:
            raise ValueError("monitor event sequence differs")
        at(value["observed_at"])
        previous = exact_json(value["status"])
    return len(paths), previous


def resume_progress(directory: Path, sha: str, *, now: datetime, probe: Probe = observe_process) -> dict[str, Any]:
    plan, sha = load_plan(directory, sha)
    binding = load_binding(directory, sha)
    require_missing_owner(directory, sha, binding, probe=probe)
    path = directory / "progress.json"
    migration = directory / "monitor-migration.json"
    if not path.exists():
        terminal = read(directory / "state.json")
        if (set(terminal) != {"state", "checks", "changes", "ended_at", "orders_submitted"}
                or terminal["state"] not in ("STOPPED", "INTERRUPTED", "FINISHED")
                or terminal["orders_submitted"] is not False
                or not at(binding["registered_at"]) <= at(terminal["ended_at"]) <= now
                or type(terminal["checks"]) is not int or not 0 <= terminal["checks"] <= plan["max_checks"]
                or type(terminal["changes"]) is not int or not 0 <= terminal["changes"] <= terminal["checks"]):
            raise ValueError("exact legacy terminal counters required")
        changes, previous = event_state(directory, terminal["changes"])
        value: dict[str, Any] = {"schema_version": 1, "plan_sha256": sha,
                 "binding_sha256": digest((directory / "monitor-binding.json").read_bytes()),
                 "original_state": terminal}
        if migration.exists():
            if read(migration) != value:
                raise ValueError("original migration evidence differs")
        else:
            write(migration, value, exclusive=True)
        latest = read(directory / "latest.json") if terminal["checks"] else None
        next_at = at(terminal["ended_at"]) if latest is None else max(
            at(terminal["ended_at"]), at(latest["evaluated_at"])+timedelta(seconds=plan["interval_seconds"]))
        return {"schema_version": 1, "plan_sha256": sha, "initial_checks": terminal["checks"],
                "checks": terminal["checks"], "changes": changes, "previous": previous,
                "active_check": None, "next_check_at": next_at.isoformat(), "interrupted_checks": 0}
    progress = read(path)
    if (set(progress) != {"schema_version", "plan_sha256", "initial_checks", "checks", "changes", "previous",
                         "active_check", "next_check_at", "interrupted_checks", "state", "worker_pid",
                         "worker_created_at", "mode", "updated_at"}
            or type(progress["schema_version"]) is not int or progress["schema_version"] != 1
            or progress["plan_sha256"] != sha or progress["mode"] not in ("LEGACY", "RESUMED")
            or progress["state"] not in ("RUNNING", "STOPPED", "FINISHED", "INTERRUPTED")
            or type(progress["checks"]) is not int or type(progress["changes"]) is not int
            or type(progress["initial_checks"]) is not int
            or type(progress["interrupted_checks"]) is not int
            or not 0 <= progress["interrupted_checks"] <= progress["checks"]
            or progress["active_check"] is not None and type(progress["active_check"]) is not int
            or not 0 <= progress["initial_checks"] <= progress["checks"] <= plan["max_checks"]
            or not 0 <= progress["changes"] <= progress["checks"] or at(progress["updated_at"]) > now
            or progress["active_check"] not in (None, progress["checks"]-1)):
        raise ValueError("invalid monitor checkpoint")
    if migration.exists():
        saved = read(migration)
        if (saved["plan_sha256"] != sha or saved["original_state"]["checks"] != progress["initial_checks"]
                or saved["binding_sha256"] != digest((directory / "monitor-binding.json").read_bytes())):
            raise ValueError("migration baseline differs")
    elif progress["initial_checks"] != 0:
        raise ValueError("missing original counter baseline")
    claims = sorted((directory / "check-claims").glob("*.json"))
    expected = progress["checks"]-progress["initial_checks"]
    if len(claims) not in (expected, expected+1):
        raise ValueError("lost monitor check claims")
    for index, claim in enumerate(claims, start=progress["initial_checks"]):
        value = read(claim)
        if (claim.name != f"{index:05d}.json" or value["index"] != index or value["plan_sha256"] != sha
                or not at(plan["started_at"]) <= at(value["claimed_at"]) < at(plan["ends_at"])
                or at(value["claimed_at"]) > now):
            raise ValueError("invalid monitor check claim")
    extra = len(claims)-expected
    interrupted = progress["active_check"] is not None or bool(extra)
    changes, previous = event_state(directory, progress["changes"], allow_extra=interrupted)
    progress.update(checks=progress["checks"]+extra, changes=changes, previous=previous, active_check=None,
                    interrupted_checks=progress["interrupted_checks"]+int(interrupted))
    if progress["checks"] > plan["max_checks"]:
        raise ValueError("original monitor count exhausted")
    return progress
