"""Exact finite monitor continuation; synthetic archives, no exchange clients."""

import copy
import json
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_offline_estimate import OPENED

from crypto_arbitrage_bot import monitor_archive as archive
from crypto_arbitrage_bot import runtime_status as module
from crypto_arbitrage_bot.utils.file_lock import LockBusyError, file_lock
from crypto_arbitrage_bot.utils.process_probe import ProcessObservation, ProcessState


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("monitor is local only")))


def missing(*args, **kwargs):
    return ProcessObservation(ProcessState.MISSING, args[0])


@pytest.fixture
def legacy(tmp_path):
    directory = tmp_path / "monitor"
    directory.mkdir()
    config = tmp_path / "testnet.toml"
    config.write_text(Path("docs/testnet-funding-acceptance.toml").read_text().replace(
        "data/testnet/execution.sqlite", (tmp_path / "absent-journal.sqlite").as_posix()), encoding="utf-8")
    spec = {"research_config": None, "research_directory": None, "testnet_config": str(config.resolve()),
            "max_heartbeat_age_seconds": 60, "working_directory": str(Path.cwd().resolve())}
    plan = {"started_at": OPENED, "ends_at": OPENED+timedelta(seconds=100),
            "interval_seconds": 10, "max_checks": 20, "orders_submitted": False}
    archive.write(directory / "plan.json", plan, exclusive=True)
    plan, sha = archive.load_plan(directory)
    with file_lock(directory / "monitor.lock"):
        archive.bind_live(directory, sha, 4242, spec, clock=lambda: OPENED+timedelta(seconds=5),
            probe=lambda *a, **k: ProcessObservation(ProcessState.MATCHED, 4242, OPENED-timedelta(seconds=1)))
    report = {"evaluated_at": OPENED+timedelta(seconds=29),
              "research": {"status": "RUNNING_CONFIRMED", "reason_codes": ()}}
    archive.write(directory / "latest.json", report)
    archive.write(directory / "event-00000.json", {"observed_at": OPENED+timedelta(seconds=10),
        "change_index": 0, "status": json.loads(module._meaningful(report))}, exclusive=True)
    archive.write(directory / "state.json", {"state": "STOPPED", "checks": 3, "changes": 1,
        "ended_at": OPENED+timedelta(seconds=30), "orders_submitted": False})
    return directory, sha, spec, report


def resume(legacy, *, start=31, check=None, probe=missing, stop_after=2):
    directory, sha, _, original = legacy
    elapsed = [start]
    observed = []
    def observe():
        checkpoint = archive.read(directory / "progress.json")
        claim = directory / "check-claims" / f"{checkpoint['checks']-1:05d}.json"
        assert claim.is_file() and checkpoint["active_check"] == checkpoint["checks"]-1
        observed.append(checkpoint["active_check"])
        if check is not None:
            return check()
        report = copy.deepcopy(original)
        report["evaluated_at"] = OPENED+timedelta(seconds=elapsed[0])
        if len(observed) >= stop_after:
            (directory / "stop").touch()
        return report
    def wait(seconds):
        elapsed[0] += seconds
        return (directory / "stop").exists()
    result = module.monitor_status(observe, directory, duration_seconds=None, interval_seconds=None,
        max_checks=None, resume=True, original_plan_sha256=sha, probe=probe,
        worker_created_at=OPENED+timedelta(seconds=start),
        clock=lambda: OPENED+timedelta(seconds=elapsed[0]),
        steady_ns=lambda: int(elapsed[0]*1_000_000_000), wait=wait)
    return result, observed


def test_legacy_resume_preserves_original_budget_count_events_and_quiet_state(legacy):
    directory, sha, _, _ = legacy
    plan_before = (directory / "plan.json").read_bytes()
    event_before = (directory / "event-00000.json").read_bytes()
    result, observed = resume(legacy)
    assert observed == [3, 4] and result["checks"] == 5 and result["changes"] == 1
    assert result["state"] == "STOPPED"
    assert (directory / "plan.json").read_bytes() == plan_before
    assert (directory / "event-00000.json").read_bytes() == event_before
    migrated = archive.read(directory / "monitor-migration.json")
    assert migrated["original_state"]["checks"] == 3 and migrated["plan_sha256"] == sha
    progress = archive.read(directory / "progress.json")
    assert progress["initial_checks"] == 3 and progress["checks"] == 5


@pytest.mark.parametrize("state", [ProcessState.MATCHED, ProcessState.UNKNOWN, ProcessState.DIFFERENT_IDENTITY])
def test_alive_unknown_or_reused_pid_cannot_resume_even_when_lock_is_free(legacy, state):
    directory, *_ = legacy
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="not confirmed missing"):
        resume(legacy, probe=lambda *a, **k: ProcessObservation(state, a[0]))
    assert before == {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


def test_held_monitor_lock_blocks_duplicate_owner_before_any_check(legacy):
    with file_lock(legacy[0] / "monitor.lock"), pytest.raises(LockBusyError):
        resume(legacy)
    assert not (legacy[0] / "progress.json").exists()


@pytest.mark.parametrize("case", ["missing_state", "missing_count", "float_count", "negative_count", "too_many",
                                  "running_state", "future_end", "missing_event", "extra_event", "changed_plan",
                                  "changed_scope", "changed_policy", "missing_binding"])
def test_legacy_missing_or_corrupt_proof_never_guesses_remaining_count(legacy, case):
    directory, sha, spec, _ = legacy
    path = directory / "state.json"
    value = archive.read(path)
    if case == "missing_state":
        path.unlink()
    elif case == "missing_binding":
        (directory / "monitor-binding.json").unlink()
    elif case == "changed_plan":
        plan = archive.read(directory / "plan.json")
        plan["max_checks"] += 1
        archive.write(directory / "plan.json", plan)
    elif case == "changed_scope":
        binding = archive.read(directory / "monitor-binding.json")
        binding["spec"]["working_directory"] = str(directory)
        archive.write(directory / "monitor-binding.json", binding)
    elif case == "changed_policy":
        Path(spec["testnet_config"]).write_text("# changed", encoding="utf-8")
    elif case == "missing_event":
        (directory / "event-00000.json").unlink()
    elif case == "extra_event":
        archive.write(directory / "event-00001.json", archive.read(directory / "event-00000.json"), exclusive=True)
    else:
        if case == "missing_count":
            del value["checks"]
        elif case == "float_count":
            path.write_text(json.dumps({**value, "checks": 3.0}))
        elif case == "negative_count":
            value["checks"] = -1
        elif case == "too_many":
            value["checks"] = 21
        elif case == "running_state":
            value["state"] = "RUNNING"
        elif case == "future_end":
            value["ended_at"] = (OPENED+timedelta(seconds=99)).isoformat()
        if case != "float_count":
            archive.write(path, value)
    with pytest.raises((ValueError, OSError, KeyError, TypeError)):
        resume(legacy)
    assert not (directory / "progress.json").exists()


@pytest.mark.parametrize("budget", ["time", "count"])
def test_exhausted_original_budget_is_not_reopened(legacy, budget):
    directory, *_ = legacy
    if budget == "count":
        terminal = archive.read(directory / "state.json")
        terminal["checks"] = 20
        archive.write(directory / "state.json", terminal)
    with pytest.raises(ValueError, match="budget ended"):
        resume(legacy, start=100 if budget == "time" else 31)
    assert not (directory / "progress.json").exists()


def test_interrupted_durable_check_is_consumed_not_replayed(legacy):
    interrupted = Mock(side_effect=KeyboardInterrupt)
    result, observed = resume(legacy, check=interrupted)
    assert result["state"] == "INTERRUPTED" and result["checks"] == 4 and observed == [3]
    result, observed = resume(legacy, start=50, stop_after=1)
    assert observed == [4] and result["checks"] == 5
    assert archive.read(legacy[0] / "progress.json")["interrupted_checks"] == 1


def test_claim_written_before_checkpoint_failure_is_consumed_on_resume(legacy):
    result, _ = resume(legacy)
    directory, sha, *_ = legacy
    archive.write(directory / "check-claims" / "00005.json",
                  {"index": 5, "plan_sha256": sha, "claimed_at": OPENED+timedelta(seconds=55)}, exclusive=True)
    result, observed = resume(legacy, start=60, stop_after=1)
    assert observed == [6] and result["checks"] == 7


@pytest.mark.parametrize("case", ["deleted_claim", "changed_claim", "lost_event", "changed_migration", "changed_count"])
def test_new_checkpoint_or_archive_corruption_blocks_resume(legacy, case):
    resume(legacy)
    directory, *_ = legacy
    if case == "deleted_claim":
        (directory / "check-claims/00003.json").unlink()
    elif case == "changed_claim":
        value = archive.read(directory / "check-claims/00003.json")
        value["plan_sha256"] = "0"*64
        archive.write(directory / "check-claims/00003.json", value)
    elif case == "lost_event":
        (directory / "event-00000.json").unlink()
    elif case == "changed_migration":
        value = archive.read(directory / "monitor-migration.json")
        value["original_state"]["checks"] = 0
        archive.write(directory / "monitor-migration.json", value)
    else:
        value = archive.read(directory / "progress.json")
        value["checks"] = 0
        archive.write(directory / "progress.json", value)
    with pytest.raises((ValueError, KeyError, OSError)):
        resume(legacy, start=60)


def test_monitor_stop_does_not_require_unchanged_configuration(legacy, capsys):
    directory, sha, spec, _ = legacy
    Path(spec["testnet_config"]).unlink()
    assert module.main(["stop-monitor", "--output-dir", str(directory), "--original-plan-sha256", sha]) == 0
    assert (directory / "stop").exists()
    assert json.loads(capsys.readouterr().out)["status"] == "MONITOR_STOP_REQUESTED"


def test_status_monitor_reports_original_count_and_no_owner_without_restart(legacy, monkeypatch, capsys):
    directory, sha, *_ = legacy
    monkeypatch.setattr(module, "observe_process", missing)
    monkeypatch.setattr(module, "utc_now", lambda: OPENED+timedelta(seconds=31))
    assert module.main(["status-monitor", "--output-dir", str(directory), "--original-plan-sha256", sha]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "FINISHED_NO_OWNER" and result["budget_consumed"] == 3
    assert result["budget_limit"] == 20 and not result["orders_submitted"]
    assert not (directory / "progress.json").exists()


def test_status_monitor_does_not_hide_a_failed_check_behind_healthy_owner(legacy, monkeypatch, capsys):
    directory, sha, *_ = legacy
    archive.write(directory / "latest.json", {"evaluated_at": OPENED+timedelta(seconds=30),
                  "research": {"status": "UNAVAILABLE", "reason_codes": ["CHECK_FAILED"]}})
    monkeypatch.setattr(module, "observe_process", missing)
    monkeypatch.setattr(module, "utc_now", lambda: OPENED+timedelta(seconds=31))
    assert module.main(["status-monitor", "--output-dir", str(directory), "--original-plan-sha256", sha]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["last_check_attention_required"] and result["attention_required"]


@pytest.mark.parametrize("field,value", [("checks", True), ("active_check", True), ("mode", "UNKNOWN"),
                                         ("state", "UNKNOWN"), ("interrupted_checks", -1)])
def test_checkpoint_types_and_lifecycle_cannot_be_silently_repaired(legacy, field, value):
    resume(legacy)
    path = legacy[0] / "progress.json"
    progress = archive.read(path)
    progress[field] = value
    archive.write(path, progress)
    with pytest.raises((ValueError, TypeError)):
        resume(legacy, start=60)


@pytest.mark.parametrize("option", ["--duration-seconds", "--interval-seconds", "--max-checks", "--research-config",
                                   "--testnet-config", "--max-heartbeat-age-seconds", "--pid"])
def test_cli_resume_rejects_scope_and_budget_overrides_before_launch(legacy, option):
    directory, sha, *_ = legacy
    with pytest.raises(SystemExit):
        module.main(["resume-monitor", "--output-dir", str(directory), "--original-plan-sha256", sha, option, "1"])
    assert not (directory / "progress.json").exists()


@pytest.mark.parametrize("change", ["none", "interpreter", "module", "directory", "duration", "age", "duplicate", "extra"])
def test_legacy_identity_covers_exact_original_arguments(legacy, change):
    directory, sha, spec, _ = legacy
    plan, _ = archive.load_plan(directory, sha)
    command = [sys.executable, "-u", "-m", "crypto_arbitrage_bot.runtime_status", "monitor",
        "--testnet-config", spec["testnet_config"], "--max-heartbeat-age-seconds", "60",
        "--output-dir", str(directory), "--duration-seconds", "100", "--interval-seconds", "10", "--max-checks", "20"]
    if change == "interpreter":
        command[0] = "other.exe"
    elif change == "module":
        command[3] = "crypto_arbitrage_bot.research_control"
    elif change == "directory":
        command[command.index("--output-dir")+1] = str(directory.parent)
    elif change == "duration":
        command[command.index("--duration-seconds")+1] = "101"
    elif change == "age":
        command[command.index("--max-heartbeat-age-seconds")+1] = "90"
    elif change == "duplicate":
        command += ["--max-checks", "20"]
    elif change == "extra":
        command += ["--unreviewed-option", "1"]
    assert archive.matches_legacy(tuple(command), directory, spec, plan) is (change == "none")


def test_resume_recovers_an_event_appended_before_checkpoint_update(legacy):
    resume(legacy)
    directory, *_ = legacy
    progress = archive.read(directory / "progress.json")
    progress["active_check"] = progress["checks"]-1
    archive.write(directory / "progress.json", progress)
    report = {"research": {"status": "STOPPED_CONFIRMED", "reason_codes": ["WORKER_MISSING"]}}
    archive.write(directory / "event-00001.json", {"observed_at": OPENED+timedelta(seconds=55),
                  "change_index": 1, "status": json.loads(module._meaningful(report))}, exclusive=True)
    result, observed = resume(legacy, start=60, check=lambda: {**report, "evaluated_at": OPENED+timedelta(seconds=60)},
                              stop_after=1)
    # This callback cannot request stop, so the original deadline ends the run.
    assert observed[0] == 5 and result["changes"] == 2
    assert (directory / "event-00001.json").is_file()


def test_brief_windows_reader_sharing_violation_retries_same_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    archive.write(path, {"checks": 1})
    original = Path.replace
    attempts = []
    def replace(source, target):
        attempts.append((source, target))
        if len(attempts) == 1:
            raise PermissionError("synthetic reader holds delete sharing")
        return original(source, target)
    monkeypatch.setattr(Path, "replace", replace)
    delay = Mock()
    monkeypatch.setattr(archive, "sleep", delay)
    archive.write(path, {"checks": 2})
    assert archive.read(path) == {"checks": 2} and len(attempts) == 2
    delay.assert_called_once_with(.01)


def test_persistent_checkpoint_permission_failure_is_bounded_and_preserves_original(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    archive.write(path, {"checks": 1})
    replace = Mock(side_effect=PermissionError("synthetic persistent permission failure"))
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(archive, "sleep", Mock())
    with pytest.raises(PermissionError):
        archive.write(path, {"checks": 2})
    assert replace.call_count == 10 and archive.read(path) == {"checks": 1}
    assert path.with_suffix(".json.tmp").is_file()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows monitor identity required")
def test_native_hidden_monitor_stop_and_resume_preserve_plan_and_counter(tmp_path):
    directory = tmp_path / "native-monitor"
    config = tmp_path / "testnet.toml"
    config.write_text(Path("docs/testnet-funding-acceptance.toml").read_text().replace(
        "data/testnet/execution.sqlite", (tmp_path / "absent-journal.sqlite").as_posix()), encoding="utf-8")
    spec = {"research_config": None, "research_directory": None, "testnet_config": str(config.resolve()),
            "max_heartbeat_age_seconds": 60, "working_directory": str(Path.cwd().resolve())}
    command = [sys.executable, "-u", "-m", "crypto_arbitrage_bot.runtime_status", "monitor",
               "--testnet-config", str(config), "--max-heartbeat-age-seconds", "60", "--output-dir", str(directory),
               "--duration-seconds", "60", "--interval-seconds", "1", "--max-checks", "40"]
    processes = []
    def launch(args):
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        processes.append(process)
        return process
    def await_progress(predicate):
        deadline = time.monotonic()+12
        while time.monotonic() < deadline:
            path = directory / "progress.json"
            if path.exists():
                value = archive.read(path)
                if predicate(value):
                    return value
            time.sleep(.05)
        raise AssertionError("native monitor checkpoint not observed")
    try:
        original = launch(command)
        progress = await_progress(lambda p: p["checks"] >= 1)
        plan_before = (directory / "plan.json").read_bytes()
        _, sha = archive.load_plan(directory)
        assert archive.lock_is_held(directory / "monitor.lock"), (original.poll(), progress)
        bound = archive.bind_live(directory, sha, progress["worker_pid"], spec)
        assert bound["legacy_pid"] == progress["worker_pid"]
        assert module.main(["stop-monitor", "--output-dir", str(directory), "--original-plan-sha256", sha]) == 0
        stdout, stderr = original.communicate(timeout=12)
        assert original.returncode == 0, (stdout, stderr)
        terminal = archive.read(directory / "state.json")
        assert terminal["state"] == "STOPPED" and terminal["checks"] >= 1
        second = launch([sys.executable, "-u", "-m", "crypto_arbitrage_bot.runtime_status", "resume-monitor",
                         "--output-dir", str(directory), "--original-plan-sha256", sha])
        newer = await_progress(lambda p: p["mode"] == "RESUMED" and p["checks"] > terminal["checks"])
        assert newer["worker_pid"] != progress["worker_pid"]
        (directory / "stop").touch()
        stdout, stderr = second.communicate(timeout=12)
        assert second.returncode == 0, (stdout, stderr)
        final = archive.read(directory / "state.json")
        assert terminal["checks"] < final["checks"] <= 40
        assert (directory / "plan.json").read_bytes() == plan_before
        assert len(tuple((directory / "check-claims").glob("*.json"))) == final["checks"]
    finally:
        for process in processes:
            if process.poll() is None:
                (directory / "stop").touch()
                try:
                    stdout, stderr = process.communicate(timeout=12)
                except subprocess.TimeoutExpired:
                    # This test owns these temporary children exclusively.
                    process.kill()
                    stdout, stderr = process.communicate(timeout=12)
            else:
                stdout, stderr = process.communicate(timeout=12)
            (tmp_path / f"child-{process.pid}.stdout").write_bytes(stdout)
            (tmp_path / f"child-{process.pid}.stderr").write_bytes(stderr)
