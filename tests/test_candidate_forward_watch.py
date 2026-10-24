"""Acceptance worker is finite and delegates to original candidate scheduler."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location("candidate_acceptance_watch", Path("work/candidate-forward/watch.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_finite_worker_sleeps_and_stops_at_deadline(tmp_path):
    clock = [0]
    waits = []
    def wait(seconds):
        waits.append(seconds)
        clock[0] += seconds
    cycle = Mock(return_value={"task_events": []})
    result = module.watch(SimpleNamespace(research=SimpleNamespace(sqlite_path=tmp_path / "db")),
                          tmp_path, duration_seconds=65, poll_seconds=30, max_batches=10,
                          cycle=cycle, monotonic=lambda: clock[0], wait=wait, stop=lambda: False)
    assert waits == [30, 30, 5] and cycle.call_count == 3
    assert result["state"] == "FINISHED" and result["failed_batches"] == 0


def test_worker_failure_waits_and_does_not_crash_loop(tmp_path):
    clock = [0]
    cycle = Mock(side_effect=[TimeoutError("offline timeout"), {"task_events": [1]}])
    result = module.watch(SimpleNamespace(research=SimpleNamespace(sqlite_path=tmp_path / "db")),
                          tmp_path, duration_seconds=100, poll_seconds=30, max_batches=2,
                          cycle=cycle, monotonic=lambda: clock[0],
                          wait=lambda seconds: clock.__setitem__(0, clock[0] + seconds), stop=lambda: False)
    assert clock[0] == 30 and result["failed_batches"] == 1
    assert result["task_events"] == 1 and result["completed_batches"] == 2


def test_worker_keyboard_interrupt_retains_state(tmp_path):
    cycle = Mock(side_effect=KeyboardInterrupt)
    result = module.watch(SimpleNamespace(research=SimpleNamespace(sqlite_path=tmp_path / "db")),
                          tmp_path, duration_seconds=60, poll_seconds=30, max_batches=2,
                          cycle=cycle, wait=Mock(), stop=lambda: False)
    assert result["state"] == "INTERRUPTED" and (tmp_path / "state.json").is_file()


def test_stop_file_prevents_requests(tmp_path):
    (tmp_path / "stop").touch()
    cycle = Mock()
    result = module.watch(SimpleNamespace(research=SimpleNamespace(sqlite_path=tmp_path / "db")),
                          tmp_path, duration_seconds=60, poll_seconds=30, max_batches=2, cycle=cycle)
    assert result["state"] == "STOPPED" and not cycle.called


def test_fresh_watch_cannot_reset_existing_original_state(tmp_path):
    (tmp_path / "state.json").write_text("original worker evidence", encoding="utf-8")
    cycle = Mock()
    with pytest.raises(ValueError, match="bounded resume"):
        module.watch(SimpleNamespace(research=SimpleNamespace(sqlite_path=tmp_path / "db")),
                     tmp_path, duration_seconds=60, poll_seconds=30, max_batches=2, cycle=cycle)
    cycle.assert_not_called()
    assert (tmp_path / "state.json").read_text(encoding="utf-8") == "original worker evidence"


@pytest.mark.parametrize("duration,poll,batches", [(0, 30, 1), (60, 0, 1), (60, 30, 10001)])
def test_unbounded_or_busy_worker_configuration_is_rejected(tmp_path, duration, poll, batches):
    with pytest.raises(ValueError):
        module.watch(None, tmp_path, duration_seconds=duration, poll_seconds=poll, max_batches=batches)
