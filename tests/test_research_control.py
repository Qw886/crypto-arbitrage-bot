"""Bounded local lifecycle uses mocks; no Binance, accounts or orders."""

import hashlib
import subprocess
import sys
import time
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_offline_estimate import OPENED

from crypto_arbitrage_bot import research_control as module
from crypto_arbitrage_bot.research_control import ControlError
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.file_lock import LockBusyError, file_lock


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("offline controller")))


@pytest.fixture
def job(tmp_path, monkeypatch):
    text = Path("docs/research-soak.toml").read_text(encoding="utf-8")
    text = text.replace("data/research/soak.sqlite", (tmp_path / "research.sqlite").as_posix())
    config = tmp_path / "research.toml"
    config.write_text(text, encoding="utf-8")
    directory = tmp_path / "job"
    elapsed = [0]
    def now():
        return OPENED+timedelta(seconds=elapsed[0])
    plan = module.prepare_job(config, directory, duration_seconds=65, max_cycles=10, clock=now)
    runner = Mock()
    runner.scheduler = None
    factory = Mock(return_value=runner)
    monkeypatch.setattr(module, "create_runner", factory)
    waits = []
    def wait(seconds):
        waits.append(seconds)
        elapsed[0] += seconds
        return False
    return config, directory, plan, elapsed, now, waits, wait, factory


def run(job, cycle=None, wait=None):
    _, directory, _, elapsed, now, _, original_wait, _ = job
    return module.run_job(directory, clock=now, steady_ns=lambda: int(elapsed[0]*1_000_000_000),
                          cycle=cycle or Mock(return_value={"failed": False, "symbols": []}),
                          wait=original_wait if wait is None else wait)


def state(job):
    return module._read(job[1] / "state.json")


def test_prepare_freezes_original_policy_deadline_limits_and_does_no_network(job):
    _, directory, plan, _, _, _, _, factory = job
    assert plan["ends_at"] == OPENED+timedelta(seconds=65)
    assert plan["poll_interval_seconds"] == 30
    assert state(job)["state"] == "PREPARED"
    assert not Path(plan["database"]).exists()
    assert module.load_job(directory)[0]["job_id"] == plan["job_id"]
    factory.assert_not_called()


def test_bounded_worker_waits_at_failures_and_finishes_at_original_deadline(job):
    cycle = Mock(side_effect=[RuntimeError("NEVER PRINT PRIVATE BODY"), {"failed": True}, {"failed": False}])
    final = run(job, cycle)
    assert final["state"] == "FINISHED" and final["cycles_started"] == 3
    assert final["cycles_completed"] == 2 and final["failed_cycles"] == 2
    assert job[5] == [30, 30, 5]
    assert "PRIVATE BODY" not in (job[1] / "state.json").read_text(encoding="utf-8")


def test_claim_is_durable_before_io_and_each_cycle_has_unique_evidence(job):
    seen = []
    def cycle(runner):
        current = state(job)
        seen.append(current["active_cycle"])
        assert current["cycles_started"] == current["active_cycle"]+1
        return {"failed": False, "symbols": []}
    run(job, cycle)
    assert seen == [0, 1, 2]
    assert sorted(p.name for p in (job[1] / "cycles").iterdir()) == ["00000.json", "00001.json", "00002.json"]


def test_finished_job_resume_cannot_extend_original_window(job):
    run(job)
    cycle = Mock()
    module.resume_job(job[1])
    assert run(job, cycle)["state"] == "FINISHED"
    cycle.assert_not_called()


def test_cycle_limit_and_budget_survive_restart(job):
    plan = module._read(job[1] / "plan.json")
    plan["max_cycles"] = 1
    (job[1] / "plan.json").write_text(exact_json(plan), encoding="utf-8")
    marker = Path(plan["database"]).with_suffix(".sqlite.research-job.json")
    marker.write_text(exact_json({"job_directory": str(job[1].resolve()),
        "plan_sha256": hashlib.sha256(exact_json(plan).encode()).hexdigest()}), encoding="utf-8")
    assert run(job)["cycles_started"] == 1
    cycle = Mock()
    assert run(job, cycle)["cycles_started"] == 1
    cycle.assert_not_called()


def test_stop_before_first_cycle_and_resume_keep_same_job_deadline(job):
    module.request_stop(job[1])
    assert run(job)["state"] == "STOPPED"
    assert state(job)["cycles_started"] == 0
    module.resume_job(job[1])
    final = run(job)
    assert final["job_id"] == job[2]["job_id"] and final["cycles_started"] == 3


def test_keyboard_interrupt_keeps_claim_and_restart_skips_old_cycle(job):
    cycle = Mock(side_effect=KeyboardInterrupt)
    assert run(job, cycle)["state"] == "INTERRUPTED"
    assert state(job)["active_cycle"] == 0 and state(job)["cycles_started"] == 1
    module.resume_job(job[1])
    final = run(job)
    assert final["interrupted_cycles"] == 1 and final["cycles_started"] == 3
    assert not (job[1] / "cycles" / "00000.json").exists()
    assert job[5] == [30, 30, 5]


def test_stop_during_work_finishes_current_record_before_idle(job):
    def cycle(runner):
        module.request_stop(job[1])
        return {"failed": False}
    final = run(job, cycle)
    assert final["state"] == "STOPPED" and final["cycles_completed"] == 1 and job[5] == []


def test_stop_while_waiting_has_no_more_cycles(job):
    cycle = Mock(return_value={"failed": False})
    def stop_wait(seconds):
        module.request_stop(job[1])
        return True
    assert run(job, cycle, stop_wait)["state"] == "STOPPED" and cycle.call_count == 1


def test_changed_policy_blocks_work_but_does_not_block_stop(job):
    config, directory = job[:2]
    config.write_text(config.read_text(encoding="utf-8").replace("poll_interval_seconds = 30", "poll_interval_seconds = 20"), encoding="utf-8")
    with pytest.raises(ControlError, match="POLICY_CHANGED"):
        run(job)
    assert module.request_stop(directory)["status"] == "STOP_REQUESTED"
    with pytest.raises(ControlError):
        module.resume_job(directory)


def test_os_owned_lock_prevents_duplicate_worker_and_resume(job):
    with file_lock(job[1] / "worker.lock"):
        assert module.job_status(job[1])["process_observation"] == "WORKER_LOCK_HELD"
        with pytest.raises(LockBusyError):
            run(job)
        with pytest.raises(LockBusyError):
            module.resume_job(job[1])
    assert module.job_status(job[1])["process_observation"] == "NOT_RUNNING"


def test_database_lock_protects_a_managed_job_from_another_worker(job):
    with file_lock(module._database_lock(Path(job[2]["database"]))), pytest.raises(LockBusyError):
        run(job)


def test_old_running_state_without_os_lock_does_not_claim_liveness(job):
    data = state(job)
    data.update(state="RUNNING", worker_pid=1)
    module._save_state(job[1], data)
    assert module.job_status(job[1])["process_observation"] == "INTERRUPTED_OR_NOT_RUNNING"


def test_old_existing_database_cannot_be_taken_over(job):
    Path(job[2]["database"]).touch()
    with pytest.raises(ControlError, match="EXISTING_DATABASE"):
        module.prepare_job(job[0], job[1].parent / "other-job", duration_seconds=30, max_cycles=1)


def test_database_binding_prevents_duplicate_registration_before_db_creation(job):
    with pytest.raises(ControlError, match="ALREADY_BOUND"):
        module.prepare_job(job[0], job[1].parent / "other-job", duration_seconds=30, max_cycles=1)


@pytest.mark.parametrize("seconds,cycles", [(0, 1), (691201, 1), (True, 1), (1, 0), (1, 30001), (1, True)])
def test_unbounded_inputs_rejected_without_registering(job, seconds, cycles):
    with pytest.raises(ControlError, match="BOUNDED"):
        module.prepare_job(job[0], job[1].parent / "invalid-job", duration_seconds=seconds, max_cycles=cycles)


def test_testnet_configuration_is_rejected_without_credentials(job, monkeypatch):
    monkeypatch.setattr(module, "load_settings", lambda path: Mock(research=Mock(), testnet=Mock()))
    with pytest.raises(ControlError, match="PUBLIC_RESEARCH"):
        module.prepare_job(job[0], job[1].parent / "invalid-job", duration_seconds=30, max_cycles=1)


def test_status_reads_old_database_without_claiming_worker_liveness(job, tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE opportunities(id TEXT,decision TEXT)")
        db.execute("INSERT INTO opportunities VALUES('1','REJECT')")
    before = path.read_bytes()
    assert module.database_status(path)["decisions"] == {"REJECT": 1}
    assert path.read_bytes() == before
    assert module.database_status(tmp_path / "missing.sqlite") == {"status": "NOT_CREATED"}


def test_background_launch_uses_only_public_worker_hidden_window_and_no_shell(job, monkeypatch):
    process = Mock(pid=123)
    launch = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, "Popen", launch)
    assert module._launch(job[1]) == 123
    args, kwargs = launch.call_args
    assert args[0][:3] == [module.sys.executable, "-m", "crypto_arbitrage_bot.research_control"]
    assert "shell" not in kwargs and "env" not in kwargs
    assert kwargs["cwd"] == job[2]["working_directory"]
    if module.sys.platform == "win32":
        assert kwargs["creationflags"] == module.subprocess.CREATE_NO_WINDOW
    module._launched_processes.remove(process)


def test_cli_status_and_stop_never_construct_network_runner(job, capsys):
    assert module.main(["status", "--job-dir", str(job[1])]) == 0
    assert module.main(["stop", "--job-dir", str(job[1])]) == 0
    job[7].assert_not_called()
    assert "STOP_REQUESTED" in capsys.readouterr().out


def test_cli_old_db_status_is_explicitly_unmanaged(job, capsys):
    assert module.main(["status", "--config", str(job[0])]) == 0
    assert "UNMANAGED_PROCESS_STATE_UNKNOWN" in capsys.readouterr().out
    job[7].assert_not_called()


@pytest.mark.parametrize("value", [True, 1.5, -1, 30001])
def test_corrupt_claim_counter_never_restarts_work(job, value):
    data = state(job)
    if type(value) is float:
        # Deliberately corrupt the file; production exact_json forbids floats.
        (job[1] / "state.json").write_text(exact_json(data).replace('"cycles_started":0', '"cycles_started":1.5'), encoding="utf-8")
    else:
        data["cycles_started"] = value
        module._save_state(job[1], data)
    with pytest.raises(ControlError, match="STATE_MISMATCH|INVALID_LOCAL_CONTROL"):
        run(job)
    job[7].assert_not_called()


def test_native_separate_process_lock_and_cooperative_stop_without_network(job):
    # Only this test child is controlled. No production worker or external PID
    # is signalled; its injected cycle never constructs a market adapter.
    code = '''
import sys,time
from pathlib import Path
from types import SimpleNamespace
from crypto_arbitrage_bot import research_control as m
p=Path(sys.argv[1])
m.create_runner=lambda settings: SimpleNamespace(scheduler=None, stop_signal=None)
at=m._at(m._read(p/"plan.json")["created_at"])
def cycle(runner):
    (p/"test-ready").touch()
    for _ in range(100):
        if runner.stop_signal.is_set(): break
        time.sleep(.05)
    return {"failed":False}
m.run_job(p,clock=lambda:at,cycle=cycle)
'''
    process = subprocess.Popen([sys.executable, "-c", code, str(job[1])], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    try:
        deadline = time.monotonic()+15
        while not (job[1] / "test-ready").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert process.poll() is None and (job[1] / "test-ready").exists()
        observed = module.job_status(job[1])
        assert observed["process_observation"] == "WORKER_LOCK_HELD"
        with pytest.raises(LockBusyError):
            module.resume_job(job[1])
        module.request_stop(job[1])
        output, error = process.communicate(timeout=10)
        assert process.returncode == 0, error.decode(errors="replace")
        assert module.job_status(job[1])["process_observation"] == "NOT_RUNNING"
        assert state(job)["state"] == "STOPPED" and state(job)["cycles_completed"] == 1
    finally:
        if process.poll() is None:
            process.kill()  # Only our offline test child, identified by its live Popen handle.
        process.communicate(timeout=10)
