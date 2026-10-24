"""Restore old public study deadlines/counters; no network or evidence reset."""
import hashlib
import importlib.util
import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_offline_estimate import OPENED

from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.file_lock import LockBusyError, file_lock

spec = importlib.util.spec_from_file_location("candidate_forward_resume", Path("work/candidate-forward/resume.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("resume tests are offline")))


@pytest.fixture
def legacy(tmp_path):
    config = tmp_path / "original.toml"
    text = Path("docs/research-multisettlement.toml").read_text(encoding="utf-8")
    text = text.replace("data/research/multisettlement.sqlite", (tmp_path / "original.sqlite").as_posix())
    config.write_text(text, encoding="utf-8")
    state = {"state": "RUNNING", "started_at": OPENED, "ended_at": None, "completed_batches": 3,
             "failed_batches": 1, "task_events": 2, "duration_seconds": 100, "poll_seconds": 30,
             "extended_duration": True, "sqlite_path": str(tmp_path / "original.sqlite")}
    directory = tmp_path / "old-worker"
    directory.mkdir()
    (directory / "state.json").write_text(exact_json(state)+"\n", encoding="utf-8")
    for index in range(3):
        (directory / f"batch-{index:05d}.json").write_text("original evidence", encoding="utf-8")
    return config, directory, hashlib.sha256(config.read_bytes()).hexdigest(), state


def restore(legacy, watch=None, cycle=None, **changes):
    config, directory, sha, _ = legacy
    return module.restore(changes.get("config", config), directory,
        expected_config_sha256=changes.get("sha", sha), original_max_batches=changes.get("max_batches", 10),
        clock=changes.get("clock", lambda: OPENED+timedelta(seconds=40)),
        watch=watch or Mock(return_value={"state": "FINISHED"}),
        cycle=cycle or Mock(return_value={"task_events": []}))


def test_restore_uses_original_remaining_time_batch_budget_and_unchanged_files(legacy):
    watch = Mock(return_value={"state": "FINISHED"})
    before = {p.name: p.read_bytes() for p in legacy[1].glob("batch-*.json")}
    result = restore(legacy, watch)
    assert watch.call_args.kwargs["duration_seconds"] == 60
    assert watch.call_args.kwargs["max_batches"] == 7
    assert result["started_at"] == OPENED.isoformat(timespec="microseconds")
    assert result["original_deadline"] == OPENED+timedelta(seconds=100)
    assert before == {p.name: p.read_bytes() for p in legacy[1].glob("batch-*.json")}


def test_delegated_cycle_claims_original_next_index_before_io(legacy):
    def cycle(settings, output):
        assert output.name == "batch-00003.json"
        saved = json.loads((legacy[1] / "state.json").read_text(encoding="utf-8"))
        assert saved["completed_batches"] == 4 and saved["active_batch"] == 3
        output.write_text("new offline observation", encoding="utf-8")
        return {"task_events": ["fake-task-event"]}
    def watch(settings, directory, **kw):
        kw["cycle"](settings, directory / "ignored.json")
        return {"state": "FINISHED"}
    result = restore(legacy, watch, cycle)
    assert result["completed_batches"] == 4 and result["task_events"] == 3
    assert (legacy[1] / "batch-00003.json").exists()


def test_crash_after_evidence_write_but_before_old_state_update_is_not_overwritten(legacy):
    orphan = legacy[1] / "batch-00003.json"
    orphan.write_text("evidence after last state", encoding="utf-8")
    watch = Mock(return_value={"state": "FINISHED"})
    result = restore(legacy, watch)
    assert result["completed_batches"] == 4 and watch.call_args.kwargs["max_batches"] == 6
    assert orphan.read_text(encoding="utf-8") == "evidence after last state"


@pytest.mark.parametrize("at", [100, 200])
def test_expired_deadline_must_not_restart_worker(legacy, at):
    watch = Mock()
    assert restore(legacy, watch, clock=lambda: OPENED+timedelta(seconds=at))["state"] == "FINISHED"
    watch.assert_not_called()


def test_budget_exhaustion_must_not_reset_count(legacy):
    watch = Mock()
    result = restore(legacy, watch, max_batches=3)
    assert result["state"] == "FINISHED" and result["completed_batches"] == 3
    watch.assert_not_called()


def test_explicit_stop_marker_is_never_cleared_by_restore(legacy):
    (legacy[1] / "stop").touch()
    watch = Mock()
    assert restore(legacy, watch)["state"] == "STOPPED"
    assert (legacy[1] / "stop").exists()
    watch.assert_not_called()


def test_restart_cannot_change_previously_bound_budget(legacy):
    restore(legacy)
    with pytest.raises(ValueError, match="binding changed"):
        restore(legacy, max_batches=11)


def test_source_configuration_hash_must_match(legacy):
    before = (legacy[1] / "state.json").read_bytes()
    with pytest.raises(ValueError, match="configuration changed"):
        restore(legacy, sha="0"*64)
    assert (legacy[1] / "state.json").read_bytes() == before


def test_actual_database_identity_cannot_be_substituted(legacy):
    data = json.loads((legacy[1] / "state.json").read_text(encoding="utf-8"))
    data["sqlite_path"] = "another-database.sqlite"
    (legacy[1] / "state.json").write_text(exact_json(data), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        restore(legacy)


def test_worker_lock_prevents_duplicate_resume(legacy):
    with file_lock(legacy[1] / "worker.lock"), pytest.raises(LockBusyError):
        restore(legacy)


def test_prior_active_claim_survives_restart_as_interrupted_not_success(legacy):
    data = json.loads((legacy[1] / "state.json").read_text(encoding="utf-8"))
    data.update(completed_batches=4, active_batch=3)
    (legacy[1] / "state.json").write_text(exact_json(data), encoding="utf-8")
    result = restore(legacy)
    assert result["completed_batches"] == 4 and result["interrupted_batches"] == 1
    assert result["active_batch"] is None


def test_wall_deadline_and_user_stop_are_checked_after_resume(legacy):
    now = [OPENED+timedelta(seconds=40)]
    def watch(settings, directory, **kw):
        assert not kw["stop"]()
        now[0] = OPENED+timedelta(seconds=100)
        assert kw["stop"]()
        return {"state": "STOPPED"}
    assert restore(legacy, watch, clock=lambda: now[0])["state"] == "FINISHED"


def test_whole_worker_interruption_retains_original_deadline_and_evidence(legacy):
    before = (legacy[1] / "batch-00000.json").read_bytes()
    with pytest.raises(KeyboardInterrupt):
        restore(legacy, watch=Mock(side_effect=KeyboardInterrupt))
    data = json.loads((legacy[1] / "state.json").read_text(encoding="utf-8"))
    assert data["state"] == "INTERRUPTED" and data["worker_pid"] is None
    assert module._at(data["original_deadline"]) == OPENED+timedelta(seconds=100)
    assert (legacy[1] / "batch-00000.json").read_bytes() == before
