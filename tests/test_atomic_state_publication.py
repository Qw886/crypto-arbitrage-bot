"""Same-file publication retries and durable budget/stop/deadline behavior."""

import hashlib
import json
import sys
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_legacy_research_control import legacy as legacy
from test_legacy_research_control import run as run_legacy
from test_legacy_research_control import state as legacy_state
from test_research_control import job as job
from test_research_control import run as run_new
from test_research_control import state as new_state

from crypto_arbitrage_bot import legacy_research_control, research_control
from crypto_arbitrage_bot.utils import atomic_publish as module


def sharing_error(code=5):
    error = PermissionError(13, "synthetic sharing failure")
    error.winerror = code
    return error


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_windows_replace_publishes_exact_same_bytes_once(tmp_path, monkeypatch, code):
    source = tmp_path / "state.tmp"
    target = tmp_path / "state.json"
    source.write_bytes(b"next durable claim")
    target.write_bytes(b"previous claim")
    original = Path.replace
    attempts = []

    def replace(path, destination):
        attempts.append((path, destination, hashlib.sha256(path.read_bytes()).hexdigest()))
        if len(attempts) <= 2:
            raise sharing_error(code)
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="win32"))
    delay = Mock()
    monkeypatch.setattr(module, "sleep", delay)
    module.publish_state(source, target)
    assert target.read_bytes() == b"next durable claim" and not source.exists()
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert delay.call_count == 2 and all(call.args == (0.01,) for call in delay.call_args_list)


@pytest.mark.parametrize("code", [5, 32, 33])
def test_persistent_windows_failure_keeps_old_target_and_temp_with_finite_attempts(
    tmp_path, monkeypatch, code
):
    source = tmp_path / "state.tmp"
    target = tmp_path / "state.json"
    source.write_bytes(b"next")
    target.write_bytes(b"previous")
    error = sharing_error(code)
    replace = Mock(side_effect=error)
    delay = Mock()
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(module, "sleep", delay)
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="win32"))
    with pytest.raises(PermissionError) as caught:
        module.publish_state(source, target)
    assert caught.value is error and replace.call_count == 10 and delay.call_count == 9
    assert target.read_bytes() == b"previous" and source.read_bytes() == b"next"


@pytest.mark.parametrize("code", [None, 2, 112, 123])
def test_other_os_failures_are_not_retried(tmp_path, monkeypatch, code):
    error = OSError("synthetic permanent failure")
    if code is not None:
        error.winerror = code
    replace = Mock(side_effect=error)
    delay = Mock()
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(module, "sleep", delay)
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="win32"))
    with pytest.raises(OSError):
        module.publish_state(tmp_path / "from", tmp_path / "to")
    assert replace.call_count == 1
    delay.assert_not_called()


def test_non_windows_permission_failure_is_immediate(tmp_path, monkeypatch):
    replace = Mock(side_effect=sharing_error())
    delay = Mock()
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(module, "sleep", delay)
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux"))
    with pytest.raises(PermissionError):
        module.publish_state(tmp_path / "from", tmp_path / "to")
    assert replace.call_count == 1
    delay.assert_not_called()


@pytest.mark.parametrize("writer", [research_control._save_state, legacy_research_control._save])
def test_both_controllers_keep_original_state_encoding_and_one_claim(tmp_path, monkeypatch, writer):
    state = {"state": "RUNNING", "active_cycle": 4, "cycles_started": 5}
    writer(tmp_path, state)
    original = Path.replace
    attempts = []

    def replace(source, target):
        attempts.append(source.read_bytes())
        if len(attempts) == 1:
            raise sharing_error()
        return original(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(module, "sleep", Mock())
    writer(tmp_path, state)
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8")) == state
    assert attempts[0].endswith(b"\n") is (writer is legacy_research_control._save)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows file sharing semantics")
@pytest.mark.parametrize("writer", [research_control._save_state, legacy_research_control._save])
def test_native_reader_release_allows_same_durable_publication(tmp_path, monkeypatch, writer):
    writer(tmp_path, {"claim": 1})
    reader = (tmp_path / "state.json").open("rb")
    released = Event()
    failed = Event()
    observed = []
    original = Path.replace

    def release():
        if failed.wait(2):
            reader.close()
            released.set()

    thread = Thread(target=release)
    thread.start()

    def replace(source, target):
        try:
            return original(source, target)
        except PermissionError as error:
            observed.append(error.winerror)
            failed.set()
            assert released.wait(2)
            raise

    monkeypatch.setattr(Path, "replace", replace)
    try:
        writer(tmp_path, {"claim": 2})
    finally:
        failed.set()
        reader.close()
        thread.join(3)
    assert not thread.is_alive() and observed == [5]
    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8")) == {"claim": 2}


def install_claim_conflict(monkeypatch, *, directory, active_key, change=None, persistent=False):
    original = Path.replace
    failed = [False]
    seen = []

    def replace(source, target):
        if source == directory / "state.tmp":
            data = json.loads(source.read_text(encoding="utf-8"))
            if data.get(active_key) is not None and (persistent or not failed[0]):
                seen.append(data)
                failed[0] = True
                raise sharing_error()
        return original(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(module, "sleep", Mock(side_effect=change))
    return seen


def test_new_worker_publication_retry_never_replays_business_cycle(job, monkeypatch):
    seen = install_claim_conflict(monkeypatch, directory=job[1], active_key="active_cycle")
    calls = []

    def cycle(_):
        calls.append(new_state(job)["active_cycle"])
        return {"failed": False}

    final = run_new(job, cycle)
    assert calls == [0, 1, 2] and final["cycles_started"] == final["cycles_completed"] == 3
    assert len(seen) == 1 and final["state"] == "FINISHED"


def test_legacy_publication_retry_never_replays_original_batch_or_refunds_budget(
    legacy, monkeypatch
):
    seen = install_claim_conflict(monkeypatch, directory=legacy[0], active_key="active_batch")
    calls = []

    def cycle(_, output):
        calls.append(output.name)
        assert legacy_state(legacy)["active_batch"] == int(output.stem.split("-")[1])
        return {"task_events": []}

    final = run_legacy(legacy, cycle)
    assert calls == ["batch-00003.json", "batch-00004.json"]
    assert final["completed_batches"] == 5 and len(seen) == 1 and final["state"] == "FINISHED"


@pytest.mark.parametrize("clock_kind", ["utc", "steady", "stop"])
def test_new_worker_deadline_or_stop_during_retry_consumes_claim_but_starts_no_io(
    job, monkeypatch, clock_kind
):
    steady = [0]

    def change(_):
        if clock_kind == "utc":
            job[3][0] = 65
        elif clock_kind == "steady":
            steady[0] = 65_000_000_000
        else:
            (job[1] / "stop").touch()

    install_claim_conflict(monkeypatch, directory=job[1], active_key="active_cycle", change=change)
    cycle = Mock(return_value={"failed": False})
    final = research_control.run_job(
        job[1], clock=job[4], steady_ns=lambda: steady[0], cycle=cycle, wait=job[6]
    )
    cycle.assert_not_called()
    assert final["cycles_started"] == 1 and final["cycles_completed"] == 0
    assert final["interrupted_cycles"] == 1 and final["active_cycle"] is None
    assert final["state"] == ("STOPPED" if clock_kind == "stop" else "FINISHED")


@pytest.mark.parametrize("clock_kind", ["utc", "steady", "stop"])
def test_legacy_deadline_or_stop_during_retry_retains_original_claim_without_io(
    legacy, monkeypatch, clock_kind
):
    steady = [0]

    def change(_):
        if clock_kind == "utc":
            legacy[5][0] = 100
        elif clock_kind == "steady":
            steady[0] = 60_000_000_000
        else:
            (legacy[0] / "stop").touch()

    install_claim_conflict(
        monkeypatch, directory=legacy[0], active_key="active_batch", change=change
    )
    from test_legacy_research_control import probe

    cycle = Mock(return_value={"task_events": []})
    final = legacy_research_control.run_legacy_job(
        legacy[0],
        clock=legacy[6],
        steady_ns=lambda: steady[0],
        cycle=cycle,
        probe=probe(),
        wait=legacy[8],
    )
    cycle.assert_not_called()
    assert final["completed_batches"] == 4 and final["interrupted_batches"] == 1
    assert final["active_batch"] is None and final["state"] == (
        "STOPPED" if clock_kind == "stop" else "FINISHED"
    )


def test_persistent_claim_failure_never_starts_market_work_and_remains_interrupted(
    job, monkeypatch
):
    seen = install_claim_conflict(
        monkeypatch, directory=job[1], active_key="active_cycle", persistent=True
    )
    cycle = Mock()
    with pytest.raises(PermissionError):
        run_new(job, cycle)
    cycle.assert_not_called()
    assert len(seen) == 20  # Ten claim attempts and ten failed terminal publications.
    assert new_state(job)["cycles_started"] == 0
    assert json.loads((job[1] / "state.tmp").read_text(encoding="utf-8"))["state"] == "INTERRUPTED"


def test_post_io_publication_failure_recovers_next_cycle_without_replaying_capture(
    job, monkeypatch
):
    original = Path.replace
    failed = [False]

    def replace(source, target):
        if source == job[1] / "state.tmp":
            value = json.loads(source.read_text(encoding="utf-8"))
            if (
                value["state"] == "RUNNING"
                and value["cycles_completed"] == 1
                and value["active_cycle"] is None
                and not failed[0]
            ):
                failed[0] = True
                raise OSError("synthetic permanent publish failure after capture")
        return original(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    calls = []

    def cycle(_):
        calls.append(new_state(job)["active_cycle"])
        return {"failed": False}

    with pytest.raises(OSError):
        run_new(job, cycle)
    interrupted = new_state(job)
    assert interrupted["state"] == "INTERRUPTED" and interrupted["cycles_started"] == 1
    original_capture = (job[1] / "cycles/00000.json").read_bytes()
    final = run_new(job, cycle)
    assert calls == [0, 1, 2] and final["cycles_started"] == final["cycles_completed"] == 3
    assert (job[1] / "cycles/00000.json").read_bytes() == original_capture
