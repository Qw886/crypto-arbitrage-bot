"""Existing public study recovery; synthetic clocks/SQLite, never orders or Binance."""

import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_offline_estimate import OPENED

from crypto_arbitrage_bot import legacy_research_control as module
from crypto_arbitrage_bot import research_control, runtime_status
from crypto_arbitrage_bot.research_control import ControlError
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.file_lock import LockBusyError, file_lock, lock_is_held
from crypto_arbitrage_bot.utils.process_probe import ProcessObservation, ProcessState


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(
        "socket.create_connection", Mock(side_effect=AssertionError("offline recovery"))
    )


@pytest.fixture
def legacy(tmp_path):
    config = tmp_path / "public.toml"
    database = tmp_path / "original.sqlite"
    config.write_text(
        Path("docs/research-multisettlement.toml")
        .read_text(encoding="utf-8")
        .replace("data/research/multisettlement.sqlite", database.as_posix()),
        encoding="utf-8",
    )
    with closing(sqlite3.connect(database)) as db, db:
        db.executescript(
            "CREATE TABLE opportunities(id TEXT,decision TEXT);"
            "INSERT INTO opportunities VALUES('original','REJECT');"
        )
    directory = tmp_path / "original-worker"
    directory.mkdir()
    binding = {
        "schema_version": 1,
        "config_path": str(config),
        "sqlite_path": str(database),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "original_started_at": OPENED,
        "original_deadline": OPENED + timedelta(seconds=100),
        "original_duration_seconds": 100,
        "poll_seconds": 30,
        "original_max_batches": 10,
        "meaning": "Legacy recovery, not preregistered strategy evidence.",
    }
    state = {
        "state": "RUNNING",
        "worker_pid": 4242,
        "started_at": OPENED,
        "ended_at": None,
        "original_deadline": binding["original_deadline"],
        "duration_seconds": 100,
        "poll_seconds": 30,
        "completed_batches": 3,
        "failed_batches": 1,
        "task_events": 2,
        "active_batch": None,
        "recovery_count": 2,
        "sqlite_path": str(database),
        "updated_at": OPENED + timedelta(seconds=35),
    }
    (directory / "state.json").write_text(exact_json(state) + "\n", encoding="utf-8")
    (directory / "recovery-binding.json").write_text(exact_json(binding), encoding="utf-8")
    for index in range(3):
        (directory / f"batch-{index:05d}.json").write_bytes(b"original evidence")
    elapsed = [40]
    waits = []

    def now():
        return OPENED + timedelta(seconds=elapsed[0])

    def wait(seconds):
        waits.append(seconds)
        elapsed[0] += seconds
        return False

    return directory, config, database, state, binding, elapsed, now, waits, wait


def probe(value=ProcessState.MISSING):
    return Mock(return_value=ProcessObservation(value, 4242, OPENED))


def run(legacy, cycle=None, **kwargs):
    return module.run_legacy_job(
        legacy[0],
        clock=legacy[6],
        steady_ns=lambda: int(legacy[5][0] * 1_000_000_000),
        wait=legacy[8],
        probe=probe(),
        cycle=cycle or Mock(return_value={"task_events": []}),
        **kwargs,
    )


def state(legacy):
    return json.loads((legacy[0] / "state.json").read_text(encoding="utf-8"))


def change_state(legacy, **values):
    saved = state(legacy)
    saved.update(values)
    (legacy[0] / "state.json").write_text(exact_json(saved) + "\n", encoding="utf-8")


def test_remaining_original_time_and_counters_no_database_or_evidence_changes(legacy):
    before = legacy[2].read_bytes()
    evidence = {p.name: p.read_bytes() for p in legacy[0].glob("batch-*.json")}
    result = run(legacy)
    assert result["state"] == "FINISHED" and result["completed_batches"] == 5
    assert result["started_at"] == OPENED.isoformat(timespec="microseconds")
    assert result["original_deadline"] == OPENED + timedelta(seconds=100)
    assert legacy[7] == [30, 30] and result["failed_batches"] == 1
    assert result["recovery_count"] == 3 and result["worker_pid"] is None
    assert legacy[2].read_bytes() == before
    assert {p.name: p.read_bytes() for p in legacy[0].glob("batch-*.json")} == evidence
    receipt = json.loads((legacy[0] / "product-control.json").read_text())
    assert receipt["mode"] == "BOUND_LEGACY_PUBLIC_VALIDATION"
    assert "trading permission" in receipt["meaning"]


def test_claim_before_io_uses_original_numbered_output_and_scheduler(legacy):
    seen = []

    def cycle(settings, output):
        saved = state(legacy)
        index = saved["active_batch"]
        assert saved["completed_batches"] == index + 1
        assert settings.research.sqlite_path == legacy[2]
        assert output.name == f"batch-{index:05d}.json"
        seen.append(index)
        output.write_bytes(b"new public mock observation")
        return {"task_events": [{"status": "COMPLETED"}]}

    assert run(legacy, cycle)["task_events"] == 4
    assert seen == [3, 4]


def test_orphan_capture_is_never_overwritten_or_replayed(legacy):
    orphan = legacy[0] / "batch-00003.json"
    orphan.write_bytes(b"previous capture written before crash")
    cycle = Mock(return_value={"task_events": []})
    final = run(legacy, cycle)
    assert [args[0][1].name for args in cycle.call_args_list] == [
        "batch-00004.json",
        "batch-00005.json",
    ]
    assert final["completed_batches"] == 6
    assert orphan.read_bytes() == b"previous capture written before crash"


def test_failure_waits_is_compact_and_cannot_reset_budget(legacy):
    cycle = Mock(side_effect=RuntimeError("SECRET_RAW_ORDER_BOOK"))
    final = run(legacy, cycle)
    assert final["failed_batches"] == 3 and final["completed_batches"] == 5
    assert legacy[7] == [30, 30]
    assert "SECRET_RAW" not in (legacy[0] / "state.json").read_text()
    assert final["last_error"] == {"type": "RuntimeError"}


@pytest.mark.parametrize("at,used", [(100, 3), (200, 3), (40, 10)])
def test_expired_or_exhausted_study_never_makes_public_io(legacy, at, used):
    legacy[5][0] = at
    change_state(legacy, completed_batches=used)
    cycle = Mock()
    assert run(legacy, cycle)["state"] == "FINISHED"
    cycle.assert_not_called()
    assert not legacy[7]


def test_stop_resume_retains_binding_budget_and_all_completed_evidence(legacy):
    binding_before = (legacy[0] / "recovery-binding.json").read_bytes()
    module.request_legacy_stop(legacy[0])
    cycle = Mock()
    assert run(legacy, cycle)["state"] == "STOPPED"
    cycle.assert_not_called()
    module.resume_legacy_job(legacy[0], probe=probe())
    assert not (legacy[0] / "stop").exists()
    assert run(legacy)["completed_batches"] == 5
    assert (legacy[0] / "recovery-binding.json").read_bytes() == binding_before


def test_stop_works_even_when_config_changed_but_resume_is_rejected(legacy):
    legacy[1].write_text("# deleted original policy", encoding="utf-8")
    assert module.request_legacy_stop(legacy[0])["status"] == "STOP_REQUESTED"
    with pytest.raises(ControlError, match="POLICY_CHANGED"):
        module.resume_legacy_job(legacy[0], probe=probe())
    assert (legacy[0] / "stop").exists()


def test_cooperative_stop_during_capture_keeps_current_result_and_no_next_batch(legacy):
    def cycle(settings, output):
        module.request_legacy_stop(legacy[0])
        output.write_bytes(b"last complete capture")
        return {"task_events": []}

    final = run(legacy, cycle)
    assert final["state"] == "STOPPED" and final["completed_batches"] == 4
    assert (
        not legacy[7] and (legacy[0] / "batch-00003.json").read_bytes() == b"last complete capture"
    )


def test_interrupt_keeps_pre_io_claim_and_resume_waits_before_next_batch(legacy):
    final = run(legacy, Mock(side_effect=KeyboardInterrupt))
    assert final["state"] == "INTERRUPTED" and final["active_batch"] == 3
    assert final["completed_batches"] == 4 and final["worker_pid"] is None
    cycle = Mock(return_value={"task_events": []})
    result = run(legacy, cycle)
    assert result["interrupted_batches"] == 1
    assert [args[0][1].name for args in cycle.call_args_list] == ["batch-00004.json"]
    assert legacy[7] == [30, 30]


def test_native_wall_and_monotonic_deadlines_cannot_be_extended(legacy):
    ticks = iter((0, 0, 61_000_000_000))
    cycle = Mock(return_value={"task_events": []})
    result = module.run_legacy_job(
        legacy[0],
        clock=legacy[6],
        steady_ns=lambda: next(ticks),
        cycle=cycle,
        wait=lambda _: False,
        probe=probe(),
    )
    # The deadline reaches its bound while the pre-IO claim is published.
    # Retain that consumed claim, but do not start a market capture afterwards.
    assert result["state"] == "FINISHED"
    cycle.assert_not_called()
    assert result["completed_batches"] == 4 and result["interrupted_batches"] == 1


@pytest.mark.parametrize("lock", ["worker", "database"])
def test_actual_os_lock_denies_duplicate_resume_or_execution(legacy, lock):
    path = (
        legacy[0] / "worker.lock"
        if lock == "worker"
        else research_control._database_lock(legacy[2])
    )
    before = (legacy[0] / "state.json").read_bytes()
    with file_lock(path), pytest.raises(LockBusyError):
        run(legacy)
    with file_lock(path), pytest.raises(LockBusyError):
        module.resume_legacy_job(legacy[0], probe=probe())
    assert (legacy[0] / "state.json").read_bytes() == before


@pytest.mark.parametrize(
    "observation", [ProcessState.MATCHED, ProcessState.UNKNOWN, ProcessState.DIFFERENT_IDENTITY]
)
def test_observation_failure_or_live_pid_never_allows_takeover(legacy, observation):
    (legacy[0] / "stop").touch()
    before = (legacy[0] / "state.json").read_bytes()
    for action in (module.resume_legacy_job, module.launch_legacy_job, module.run_legacy_job):
        with pytest.raises(ControlError, match="OWNER_NOT_CONFIRMED_MISSING"):
            action(legacy[0], probe=probe(observation))
    assert (legacy[0] / "state.json").read_bytes() == before
    assert (legacy[0] / "stop").exists()


@pytest.mark.parametrize(
    "change",
    [
        "config",
        "database",
        "counter",
        "deadline",
        "poll",
        "active",
        "duplicate",
        "size",
        "missing_binding",
        "private",
    ],
)
def test_corrupt_or_unbound_legacy_job_cannot_run(legacy, change):
    if change == "config":
        legacy[1].write_text("# changed", encoding="utf-8")
    elif change == "database":
        legacy[2].unlink()
    elif change == "counter":
        change_state(legacy, completed_batches=True)
    elif change == "deadline":
        change_state(legacy, original_deadline=OPENED + timedelta(seconds=101))
    elif change == "poll":
        change_state(legacy, poll_seconds=31)
    elif change == "active":
        change_state(legacy, active_batch=3)
    elif change == "duplicate":
        path = legacy[0] / "state.json"
        path.write_text(path.read_text().strip()[:-1] + ',"completed_batches":3}', encoding="utf-8")
    elif change == "size":
        (legacy[0] / "state.json").write_text(" " * 256001, encoding="utf-8")
    elif change == "missing_binding":
        (legacy[0] / "recovery-binding.json").unlink()
    elif change == "private":
        body = Path("docs/testnet-funding-acceptance.toml").read_text(encoding="utf-8")
        legacy[1].write_text(body, encoding="utf-8")
        binding = json.loads((legacy[0] / "recovery-binding.json").read_text())
        binding["config_sha256"] = hashlib.sha256(legacy[1].read_bytes()).hexdigest()
        (legacy[0] / "recovery-binding.json").write_text(exact_json(binding), encoding="utf-8")
    cycle = Mock()
    with pytest.raises(ControlError):
        run(legacy, cycle)
    cycle.assert_not_called()


def test_operational_policy_receipt_cannot_change_on_restart(legacy, monkeypatch):
    module.resume_legacy_job(legacy[0], probe=probe())
    original = module.load_legacy_job(legacy[0]).settings
    monkeypatch.setattr(
        module, "load_settings", lambda p: original.model_copy(update={"poll_interval_seconds": 5})
    )
    with pytest.raises(ControlError, match="POLICY_CHANGED"):
        run(legacy)


def test_old_and_product_worker_identities_match_and_unrelated_pid_does_not(legacy):
    directory, config, _, _, binding, *_ = legacy
    product = (
        sys.executable,
        "-m",
        "crypto_arbitrage_bot.research_control",
        "_legacy_worker",
        "--job-dir",
        str(directory),
    )
    old = (
        sys.executable,
        "-u",
        str(Path("work/candidate-forward/resume.py")),
        "--config",
        str(config),
        "--output-dir",
        str(directory),
        "--original-config-sha256",
        binding["config_sha256"],
        "--original-max-batches",
        "10",
    )
    assert module.matches_legacy_worker(product, directory, binding)
    assert module.matches_legacy_worker(old, directory, binding)
    for argv in (
        (),
        (sys.executable,),
        product[:-1] + ("other-job",),
        old[:-1] + ("11",),
        product + ("--bad",),
    ):
        assert not module.matches_legacy_worker(argv, directory, binding)


def test_runtime_supervision_recognizes_product_worker_without_requiring_old_script(legacy):
    job = module.load_legacy_job(legacy[0])
    (legacy[0] / "worker.lock").touch()
    product = (
        sys.executable,
        "-m",
        "crypto_arbitrage_bot.research_control",
        "_legacy_worker",
        "--job-dir",
        str(job.directory),
    )

    def observation(pid, *, matches, not_before, not_after):
        assert matches(product)
        return ProcessObservation(ProcessState.MATCHED, pid, OPENED)

    result = runtime_status.research_status(
        job.settings,
        job.directory,
        at=legacy[6](),
        max_heartbeat_age=timedelta(seconds=90),
        probe=observation,
        lock_probe=lambda _: True,
    )
    assert result["status"] == runtime_status.RuntimeState.RUNNING
    assert result["budget_consumed"] == 3


def test_background_launch_releases_worker_locks_before_hidden_child_and_inherits_proxy(
    legacy, monkeypatch
):
    process = Mock(pid=123)

    def launch(argv, **kwargs):
        assert not lock_is_held(legacy[0] / "worker.lock")
        assert not lock_is_held(research_control._database_lock(legacy[2]))
        assert argv[:4] == [
            sys.executable,
            "-m",
            "crypto_arbitrage_bot.research_control",
            "_legacy_worker",
        ]
        assert "env" not in kwargs and "shell" not in kwargs
        if sys.platform == "win32":
            assert kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW
        return process

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    try:
        assert module.launch_legacy_job(legacy[0], probe=probe()) == 123
        saved = json.loads((legacy[0] / "launch.json").read_text())
        assert saved["pid"] == 123
    finally:
        module._launched_processes.remove(process)


def test_pending_launch_blocks_second_child_even_before_worker_claim(legacy, monkeypatch):
    (legacy[0] / "launch.json").write_text(
        exact_json({"pid": 777, "requested_at": OPENED}), encoding="utf-8"
    )
    observer = Mock(
        side_effect=[
            ProcessObservation(ProcessState.MISSING, 4242),
            ProcessObservation(ProcessState.UNKNOWN, 777),
        ]
    )
    child = Mock()
    monkeypatch.setattr(module.subprocess, "Popen", child)
    with pytest.raises(ControlError, match="LAUNCH_NOT_CONFIRMED_MISSING"):
        module.launch_legacy_job(legacy[0], probe=observer)
    child.assert_not_called()


@pytest.mark.parametrize("action", [module.resume_legacy_job, module.run_legacy_job])
def test_pending_child_is_not_bypassed_by_foreground_resume_or_run(legacy, action):
    change_state(legacy, state="STOPPED", worker_pid=None)
    (legacy[0] / "launch.json").write_text(
        exact_json({"pid": 777, "requested_at": OPENED}), encoding="utf-8"
    )
    observer = Mock(return_value=ProcessObservation(ProcessState.UNKNOWN, 777))
    (legacy[0] / "stop").touch()
    with pytest.raises(ControlError, match="LAUNCH_NOT_CONFIRMED_MISSING"):
        action(legacy[0], probe=observer)
    assert (legacy[0] / "stop").exists()


def test_windows_venv_matched_parent_launcher_is_not_a_duplicate_worker(legacy, monkeypatch):
    change_state(legacy, state="STOPPED", worker_pid=None)
    (legacy[0] / "launch.json").write_text(
        exact_json({"pid": 777, "requested_at": OPENED}), encoding="utf-8"
    )
    monkeypatch.setattr(module.os, "getppid", lambda: 777)
    observer = Mock(return_value=ProcessObservation(ProcessState.MATCHED, 777))
    cycle = Mock()
    final = module.run_legacy_job(legacy[0], probe=observer, cycle=cycle)
    assert final["state"] == "FINISHED"
    cycle.assert_not_called()


@pytest.mark.parametrize(
    "receipt",
    [
        {},
        {"pid": True, "requested_at": OPENED},
        {"pid": 777, "requested_at": "2026-01-01T07:00:00"},
    ],
)
def test_bad_launch_receipt_is_a_structured_control_failure(legacy, receipt):
    (legacy[0] / "launch.json").write_text(exact_json(receipt), encoding="utf-8")
    with pytest.raises(ControlError, match="INVALID_LEGACY_LAUNCH_RECEIPT"):
        module.resume_legacy_job(legacy[0], probe=probe())


def test_explicit_stop_marker_is_never_removed_by_launch_or_worker(legacy):
    (legacy[0] / "stop").touch()
    with pytest.raises(ControlError, match="STOP_REQUESTED"):
        module.launch_legacy_job(legacy[0], probe=probe())
    assert run(legacy)["state"] == "STOPPED" and (legacy[0] / "stop").exists()


@pytest.mark.parametrize("command", ["start", "_worker"])
def test_cli_never_treats_bound_legacy_job_as_a_new_scanner(legacy, capsys, command):
    assert research_control.main([command, "--job-dir", str(legacy[0])]) == 1
    assert "BOUND_LEGACY_JOB_REQUIRES_RESUME" in capsys.readouterr().out


def test_cli_routes_legacy_stop_and_status_without_exchange_clients(legacy, monkeypatch, capsys):
    constructor = Mock(side_effect=AssertionError("no runner for status/stop"))
    monkeypatch.setattr(research_control, "create_runner", constructor)
    assert research_control.main(["status", "--job-dir", str(legacy[0])]) == 0
    assert research_control.main(["stop", "--job-dir", str(legacy[0])]) == 0
    output = capsys.readouterr().out
    assert "LEGACY_PUBLIC_VALIDATION" in output and "STOP_REQUESTED" in output
    constructor.assert_not_called()


def test_cli_does_not_accept_new_legacy_time_or_cycle_budget(legacy, capsys):
    assert (
        research_control.main(["resume", "--job-dir", str(legacy[0]), "--duration-seconds", "200"])
        == 1
    )
    assert "LEGACY_PARAMETERS_ARE_FROZEN" in capsys.readouterr().out


def test_cli_resume_uses_bound_product_worker(legacy, monkeypatch, capsys):
    recover = Mock()
    launch = Mock(return_value=321)
    monkeypatch.setattr(module, "resume_legacy_job", recover)
    monkeypatch.setattr(module, "launch_legacy_job", launch)
    assert research_control.main(["resume", "--job-dir", str(legacy[0]), "--background"]) == 0
    recover.assert_called_once_with(legacy[0])
    launch.assert_called_once_with(legacy[0])
    assert "LAUNCHED_NOT_YET_CONFIRMED" in capsys.readouterr().out


def test_real_product_background_cli_expired_study_finishes_without_network(legacy):
    # OPENED is historical, so this real child exits before creating any market client.
    change_state(legacy, state="STOPPED", worker_pid=None)
    database_before = legacy[2].read_bytes()
    pid = module.launch_legacy_job(legacy[0])
    process = next(p for p in module._launched_processes if p.pid == pid)
    try:
        assert process.wait(timeout=15) == 0
        assert state(legacy)["state"] == "FINISHED" and state(legacy)["completed_batches"] == 3
        assert legacy[2].read_bytes() == database_before
        assert not lock_is_held(legacy[0] / "worker.lock")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        module._launched_processes.remove(process)


def test_native_worker_lock_stop_and_restart_skip_existing_capture_without_network(legacy):
    change_state(legacy, state="STOPPED", worker_pid=None)
    code = """
import sys,time
from pathlib import Path
from datetime import timedelta
from crypto_arbitrage_bot import legacy_research_control as m
p=Path(sys.argv[1])
opened=m._at(m._read(p/'state.json')['started_at'])
def cycle(settings, output):
    (p/'test-ready').touch()
    for _ in range(100):
        if (p/'stop').exists(): break
        time.sleep(.05)
    output.write_bytes(b'offline native helper evidence')
    return {'task_events':[]}
m.run_legacy_job(p,clock=lambda:opened+timedelta(seconds=40),cycle=cycle)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(legacy[0])],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    try:
        deadline = time.monotonic() + 15
        while (
            not (legacy[0] / "test-ready").exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert process.poll() is None and (legacy[0] / "test-ready").exists()
        with pytest.raises(LockBusyError):
            module.resume_legacy_job(legacy[0])
        module.request_legacy_stop(legacy[0])
        _, error = process.communicate(timeout=10)
        assert process.returncode == 0, error.decode(errors="replace")
        assert state(legacy)["state"] == "STOPPED" and state(legacy)["completed_batches"] == 4
        module.resume_legacy_job(legacy[0])
        cycle = Mock(return_value={"task_events": []})
        run(legacy, cycle)
        assert cycle.call_args_list[0].args[1].name == "batch-00004.json"
        assert (legacy[0] / "batch-00003.json").read_bytes() == b"offline native helper evidence"
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def test_product_control_does_not_import_credentials_or_private_execution():
    import ast

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert all("execution" not in name and "credentials" not in name for name in imports)
    assert "work/candidate-forward/resume.py" in Path(module.__file__).read_text(encoding="utf-8")
    assert module.run_due_once.__module__ == "crypto_arbitrage_bot.candidate_forward_research"


@pytest.mark.parametrize("missing", [False, True])
def test_running_without_pid_is_unverified_and_cannot_be_taken_over(legacy, missing):
    saved = state(legacy)
    if missing:
        saved.pop("worker_pid")
    else:
        saved["worker_pid"] = None
    (legacy[0] / "state.json").write_text(exact_json(saved), encoding="utf-8")
    (legacy[0] / "stop").touch()
    for action in (module.resume_legacy_job, module.launch_legacy_job, module.run_legacy_job):
        with pytest.raises(ControlError):
            action(legacy[0], probe=probe())
    assert (legacy[0] / "stop").exists()
    assert not (legacy[0] / "product-control.json").exists()


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("missing_owner", "LEGACY_OWNER_NOT_CONFIRMED_MISSING"),
        ("changed_config", "LEGACY_IDENTITY_OR_POLICY_CHANGED"),
        ("bad_launch", "INVALID_LEGACY_LAUNCH_RECEIPT"),
    ],
)
def test_native_module_cli_control_errors_are_structured_without_traceback(legacy, fault, reason):
    change_state(legacy, state="STOPPED", worker_pid=None)
    if fault == "missing_owner":
        change_state(legacy, state="RUNNING")
    elif fault == "changed_config":
        legacy[1].write_text("# changed policy", encoding="utf-8")
    else:
        (legacy[0] / "launch.json").write_text(
            exact_json({"pid": True, "requested_at": OPENED}), encoding="utf-8"
        )
    before = legacy[2].read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crypto_arbitrage_bot.research_control",
            "resume",
            "--job-dir",
            str(legacy[0]),
            "--background",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    assert result.returncode == 1 and not result.stderr
    failure = json.loads(result.stdout)
    assert failure == {"status": "CONTROL_FAILED", "reason": reason, "orders_submitted": False}
    assert legacy[2].read_bytes() == before and not (legacy[0] / "product-control.json").exists()
