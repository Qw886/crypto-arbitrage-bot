"""Local status and finite supervision; all archives synthetic and offline."""

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_offline_estimate import OPENED
from test_research_control import job as job_fixture

from crypto_arbitrage_bot import runtime_status as module
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.process_probe import ProcessObservation, ProcessState


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(
        "socket.create_connection", Mock(side_effect=AssertionError("offline supervision"))
    )


@pytest.fixture(name="job")
def local_job(tmp_path, monkeypatch):
    return job_fixture.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture
def legacy(tmp_path):
    database = tmp_path / "research.sqlite"
    config = tmp_path / "research.toml"
    config.write_text(
        Path("docs/research-multisettlement.toml")
        .read_text()
        .replace("data/research/multisettlement.sqlite", database.as_posix()),
        encoding="utf-8",
    )
    settings = load_settings(config)
    directory = tmp_path / "legacy"
    directory.mkdir()
    binding = {
        "schema_version": 1,
        "config_path": str(config),
        "sqlite_path": str(database),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "original_started_at": OPENED,
        "original_deadline": OPENED + timedelta(days=2),
        "original_duration_seconds": 172800,
        "poll_seconds": 30,
        "original_max_batches": 1000,
    }
    state = {
        "state": "RUNNING",
        "worker_pid": 4242,
        "started_at": OPENED,
        "original_deadline": OPENED + timedelta(days=2),
        "duration_seconds": 172800,
        "poll_seconds": 30,
        "completed_batches": 20,
        "sqlite_path": str(database),
        "updated_at": OPENED + timedelta(hours=1),
    }
    (directory / "state.json").write_text(exact_json(state), encoding="utf-8")
    (directory / "recovery-binding.json").write_text(exact_json(binding), encoding="utf-8")
    (directory / "worker.lock").write_bytes(b"0")
    with closing(sqlite3.connect(database)) as db, db:
        db.executescript(
            "CREATE TABLE opportunities(decision TEXT);INSERT INTO opportunities VALUES('REJECT');"
            "CREATE TABLE validation_tasks(id TEXT,status TEXT,symbol TEXT,due_at TEXT,horizon_seconds INTEGER);"
            "CREATE TABLE forward_validations(status TEXT);"
        )
        db.execute(
            "INSERT INTO validation_tasks VALUES(?,?,?,?,?)",
            ("task", "PENDING", "BTCUSDT", (OPENED + timedelta(days=1)).isoformat(), 86400),
        )
    return settings, directory, state


def status(legacy, state=ProcessState.MATCHED, held=True, at=None):
    settings, directory, saved = legacy
    return module.runtime_status(
        research_settings=settings,
        research_directory=directory,
        max_heartbeat_age=timedelta(seconds=60),
        clock=lambda: at or OPENED + timedelta(hours=1, seconds=5),
        probe=lambda *a, **k: ProcessObservation(
            state, saved["worker_pid"], OPENED + timedelta(minutes=1)
        ),
        lock_probe=lambda p: held,
    )


def test_combined_actual_owner_evidence_not_state_file_alone(legacy):
    result = status(legacy)
    assert result["research"]["status"] is module.RuntimeState.RUNNING
    assert not result["attention_required"] and not result["orders_submitted"]
    assert result["research"]["budget_consumed"] == 20
    assert result["research"]["database"]["tasks"] == {"PENDING": 1}


def test_missing_process_and_released_lock_overrides_fresh_running_file(legacy):
    result = status(legacy, ProcessState.MISSING, False)
    assert result["research"]["reported_status"] == "RUNNING"
    assert result["research"]["status"] is module.RuntimeState.STOPPED
    assert module.RuntimeReason.WORKER_MISSING in result["research"]["reason_codes"]
    assert module.RuntimeReason.PENDING_WITHOUT_OWNER in result["research"]["reason_codes"]
    assert result["attention_required"]


@pytest.mark.parametrize(
    "state,held",
    [
        (ProcessState.UNKNOWN, True),
        (ProcessState.UNKNOWN, False),
        (ProcessState.DIFFERENT_IDENTITY, True),
        (ProcessState.DIFFERENT_IDENTITY, False),
        (ProcessState.MISSING, True),
        (ProcessState.MATCHED, False),
    ],
)
def test_partial_or_failed_observation_never_declares_safe_restart(legacy, state, held):
    result = status(legacy, state, held)
    assert (
        result["research"]["status"] is module.RuntimeState.UNKNOWN and result["attention_required"]
    )
    assert "restart_requested" not in result and "worker_started" not in result


def test_alive_stale_heartbeat_is_not_process_death(legacy):
    result = status(legacy, at=OPENED + timedelta(hours=2))
    assert result["research"]["status"] is module.RuntimeState.STALE
    assert module.RuntimeReason.STALE_HEARTBEAT in result["research"]["reason_codes"]
    assert module.RuntimeReason.WORKER_MISSING not in result["research"]["reason_codes"]


@pytest.mark.parametrize(
    "change",
    [
        "config",
        "database",
        "deadline",
        "counter",
        "duplicate_json",
        "future_timestamp",
        "lifecycle",
    ],
)
def test_corrupt_legacy_binding_or_state_cannot_be_running(legacy, change):
    settings, directory, state = legacy
    if change == "config":
        binding = json.loads((directory / "recovery-binding.json").read_text())
        Path(binding["config_path"]).write_text("# changed", encoding="utf-8")
    else:
        if change == "database":
            state["sqlite_path"] = "other.sqlite"
        elif change == "deadline":
            state["original_deadline"] = OPENED + timedelta(days=3)
        elif change == "counter":
            state["completed_batches"] = True
        elif change == "future_timestamp":
            state["updated_at"] = OPENED + timedelta(days=5)
        elif change == "lifecycle":
            state["state"] = "MYSTERY"
        body = exact_json(state)
        if change == "duplicate_json":
            body = body[:-1] + ',"worker_pid":4242}'
        (directory / "state.json").write_text(body, encoding="utf-8")
    result = status(legacy)
    assert (
        result["research"]["status"] is module.RuntimeState.UNAVAILABLE
        and result["attention_required"]
    )


def test_managed_new_job_and_original_budget_stay_unchanged(job):
    config, directory, plan, *_ = job
    settings = load_settings(config)
    original = json.loads((directory / "state.json").read_text())
    original.update(state="RUNNING", worker_pid=4242, updated_at=OPENED + timedelta(seconds=1))
    (directory / "state.json").write_text(exact_json(original), encoding="utf-8")
    (directory / "worker.lock").write_bytes(b"0")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    result = module.runtime_status(
        research_settings=settings,
        research_directory=directory,
        max_heartbeat_age=timedelta(seconds=60),
        clock=lambda: OPENED + timedelta(seconds=2),
        probe=lambda *a, **k: ProcessObservation(ProcessState.MATCHED, 4242, OPENED),
        lock_probe=lambda p: True,
    )
    assert result["research"]["kind"] == "MANAGED_PUBLIC_RESEARCH"
    assert result["research"]["original_deadline"] == OPENED + timedelta(seconds=65)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_managed_changed_policy_returns_unavailable_without_network(job):
    config, directory, plan, *_ = job
    settings = load_settings(config).model_copy(update={"poll_interval_seconds": 5})
    result = module.runtime_status(
        research_settings=settings,
        research_directory=directory,
        max_heartbeat_age=timedelta(seconds=60),
        clock=lambda: OPENED,
    )
    assert result["research"]["status"] is module.RuntimeState.UNAVAILABLE


@pytest.mark.parametrize("kind", ["legacy", "managed"])
@pytest.mark.parametrize("boundary", ["deadline", "count"])
@pytest.mark.parametrize("reached", [False, True])
def test_alive_research_reports_ended_budget_without_relabeling_or_restarting(
    request, kind, boundary, reached
):
    if kind == "legacy":
        settings, directory, state = request.getfixturevalue("legacy")
        ends = OPENED + timedelta(days=2)
        limit, counter = 1000, "completed_batches"
    else:
        config, directory, plan, *_ = request.getfixturevalue("job")
        settings = load_settings(config)
        state = json.loads((directory / "state.json").read_text())
        ends = OPENED + timedelta(seconds=65)
        limit, counter = 10, "cycles_started"
        (directory / "worker.lock").write_bytes(b"0")
    now = ends + timedelta(seconds=0 if reached else -1) if boundary == "deadline" else OPENED + timedelta(seconds=2)
    state.update(state="RUNNING", worker_pid=4242, updated_at=now-timedelta(seconds=1))
    state[counter] = limit - (0 if reached else 1) if boundary == "count" else 1
    (directory / "state.json").write_text(exact_json(state), encoding="utf-8")
    before = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    result = module.runtime_status(
        research_settings=settings, research_directory=directory,
        max_heartbeat_age=timedelta(seconds=60), clock=lambda: now,
        probe=lambda *a, **k: ProcessObservation(ProcessState.MATCHED, 4242, OPENED),
        lock_probe=lambda _: True,
    )
    assert result["research"]["status"] is module.RuntimeState.RUNNING
    assert ("CONTROL_BUDGET_ENDED" in result["research"]["reason_codes"]) is reached
    if reached:
        assert result["attention_required"]
    assert {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()} == before


@pytest.mark.parametrize("task_state", ["PENDING", "RUNNING"])
@pytest.mark.parametrize("late", [300, 301])
def test_unhandled_validation_uses_existing_scheduler_lateness(legacy, task_state, late):
    settings, directory, state = legacy
    now = OPENED + timedelta(hours=1, seconds=5)
    due = now - timedelta(seconds=late)
    with closing(sqlite3.connect(settings.research.sqlite_path)) as db, db:
        db.execute("UPDATE validation_tasks SET status=?,due_at=?", (task_state, due.isoformat()))
    before = settings.research.sqlite_path.read_bytes()
    result = status(legacy)
    assert result["research"]["status"] is module.RuntimeState.RUNNING
    assert ("OVERDUE_VALIDATIONS_PRESENT" in result["research"]["reason_codes"]) is (late > 300)
    assert result["attention_required"] is (late > 300)
    assert settings.research.sqlite_path.read_bytes() == before


def test_overdue_task_beyond_eight_item_preview_is_still_detected_in_utc(legacy):
    settings, _, _ = legacy
    now = OPENED + timedelta(hours=1, seconds=5)
    # Deliberately different offsets: SQL text ordering puts the older UTC
    # task after the eight future rows. Health checks must consume all tasks.
    hidden_due = (now-timedelta(seconds=301)).astimezone(
        timezone(timedelta(hours=12)))
    with closing(sqlite3.connect(settings.research.sqlite_path)) as db, db:
        db.execute("UPDATE validation_tasks SET due_at=?", (hidden_due.isoformat(),))
        for i in range(8):
            db.execute("INSERT INTO validation_tasks VALUES(?,?,?,?,?)",
                       (f"future-{i}", "PENDING", "BTCUSDT", (now+timedelta(minutes=1)).isoformat(), 86400))
    result = status(legacy)
    assert len(result["research"]["database"]["next_tasks"]) == 8
    assert all(task["due_at"] != hidden_due.isoformat() for task in result["research"]["database"]["next_tasks"])
    assert "OVERDUE_VALIDATIONS_PRESENT" in result["research"]["reason_codes"]
    assert result["attention_required"]


@pytest.mark.parametrize("beyond", [False, True])
def test_unfinished_task_at_or_after_deadline_cannot_be_observed_by_this_job(legacy, beyond):
    settings, _, _ = legacy
    deadline = OPENED + timedelta(days=2)
    due = deadline + timedelta(seconds=0 if beyond else -1)
    with closing(sqlite3.connect(settings.research.sqlite_path)) as db, db:
        db.execute("UPDATE validation_tasks SET due_at=?", (due.isoformat(),))
    result = status(legacy)
    assert ("UNFINISHED_VALIDATIONS_BEYOND_DEADLINE" in result["research"]["reason_codes"]) is beyond
    assert result["attention_required"] is beyond


@pytest.mark.parametrize("due", ["bad-date", "2026-10-08T00:00:00", None])
def test_invalid_unfinished_task_time_is_unavailable_not_healthy(legacy, due):
    settings, _, _ = legacy
    with closing(sqlite3.connect(settings.research.sqlite_path)) as db, db:
        db.execute("UPDATE validation_tasks SET due_at=?", (due,))
    result = status(legacy)
    assert result["research"]["database"]["status"] == "UNAVAILABLE"
    assert result["attention_required"]


@pytest.mark.parametrize("task_state", ["COMPLETED", "FAILED"])
def test_old_terminal_task_is_not_an_unhandled_due_task(legacy, task_state):
    settings, _, _ = legacy
    with closing(sqlite3.connect(settings.research.sqlite_path)) as db, db:
        db.execute("UPDATE validation_tasks SET status=?,due_at=?", (task_state, OPENED.isoformat()))
    result = status(legacy)
    assert "OVERDUE_VALIDATIONS_PRESENT" not in result["research"]["reason_codes"]
    assert not result["attention_required"]


def change_legacy_config(legacy, transform):
    settings, directory, state = legacy
    binding = json.loads((directory / "recovery-binding.json").read_text())
    config = Path(binding["config_path"])
    config.write_text(transform(config.read_text()), encoding="utf-8")
    binding["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    (directory / "recovery-binding.json").write_text(exact_json(binding), encoding="utf-8")
    return load_settings(config), directory, state


@pytest.mark.parametrize("late", [600, 601])
def test_monitor_does_not_replace_explicit_scheduler_tolerance_with_300(legacy, late):
    configured = change_legacy_config(legacy, lambda text: text.replace(
        "max_lateness_seconds = 300", "max_lateness_seconds = 600"))
    settings = configured[0]
    due = OPENED + timedelta(hours=1, seconds=5-late)
    with closing(sqlite3.connect(settings.research.sqlite_path)) as db, db:
        db.execute("UPDATE validation_tasks SET due_at=?", (due.isoformat(),))
    result = status(configured)
    assert ("OVERDUE_VALIDATIONS_PRESENT" in result["research"]["reason_codes"]) is (late > 600)


def test_queued_tasks_without_configured_scheduler_require_attention(legacy):
    def without_scheduler(text):
        start = text.index("[research.validation_scheduler]")
        end = text.index("[research.candidate_catalog]", start)
        return text[:start] + text[end:]
    configured = change_legacy_config(legacy, without_scheduler)
    result = status(configured)
    assert result["research"]["status"] is module.RuntimeState.RUNNING
    assert "UNFINISHED_VALIDATIONS_WITHOUT_SCHEDULER" in result["research"]["reason_codes"]
    assert result["attention_required"]


def test_legacy_unbounded_lateness_is_not_replaced_with_a_monitor_default(legacy):
    configured = change_legacy_config(legacy, lambda text: text.replace("max_lateness_seconds = 300\n", ""))
    settings = configured[0]
    with closing(sqlite3.connect(settings.research.sqlite_path)) as db, db:
        db.execute("UPDATE validation_tasks SET due_at=?", ((OPENED+timedelta(seconds=4)).isoformat(),))
    result = status(configured)
    assert result["research"]["validation_lateness_limit_seconds"] is None
    assert "OVERDUE_VALIDATIONS_PRESENT" not in result["research"]["reason_codes"]
    assert not result["attention_required"]


@pytest.mark.parametrize("late", [0, 1])
def test_explicit_zero_lateness_boundary_is_preserved(legacy, late):
    configured = change_legacy_config(legacy, lambda text: text.replace("max_lateness_seconds = 300", "max_lateness_seconds = 0"))
    now = OPENED + timedelta(hours=1, seconds=5)
    with closing(sqlite3.connect(configured[0].research.sqlite_path)) as db, db:
        db.execute("UPDATE validation_tasks SET due_at=?", ((now-timedelta(seconds=late)).isoformat(),))
    result = status(configured)
    assert ("OVERDUE_VALIDATIONS_PRESENT" in result["research"]["reason_codes"]) is bool(late)


def test_unknown_task_state_does_not_disappear_from_health_coverage(legacy):
    settings, _, _ = legacy
    with closing(sqlite3.connect(settings.research.sqlite_path)) as db, db:
        db.execute("UPDATE validation_tasks SET status='UNRECOGNIZED'")
    result = status(legacy)
    assert result["research"]["database"]["status"] == "UNAVAILABLE"
    assert result["attention_required"]


def test_overdue_transition_emits_one_event_while_heartbeat_and_owner_stay_healthy(legacy, tmp_path):
    settings, _, _ = legacy
    base = OPENED + timedelta(hours=1, seconds=5)
    with closing(sqlite3.connect(settings.research.sqlite_path)) as db, db:
        db.execute("UPDATE validation_tasks SET due_at=?", ((base-timedelta(seconds=299)).isoformat(),))
    elapsed = [0]
    observed = []
    def check():
        report = status(legacy, at=base+timedelta(seconds=elapsed[0]))
        observed.append(report)
        return report
    def wait(seconds):
        elapsed[0] += seconds
        return False
    directory = tmp_path / "due-monitor"
    result = module.monitor_status(check, directory, duration_seconds=11, interval_seconds=5,
        max_checks=3, clock=lambda: base+timedelta(seconds=elapsed[0]),
        steady_ns=lambda: int(elapsed[0]*1_000_000_000), wait=wait)
    assert [report["attention_required"] for report in observed] == [False, True, True]
    assert result["checks"] == 3 and result["changes"] == 2
    event = json.loads((directory / "event-00001.json").read_text())
    assert "OVERDUE_VALIDATIONS_PRESENT" in event["status"]["research"]["reason_codes"]
    assert all(report["research"]["status"] is module.RuntimeState.RUNNING for report in observed)


def test_research_database_failure_requires_attention_even_when_process_alive(legacy):
    path = legacy[0].research.sqlite_path
    path.write_bytes(b"not a database")
    result = status(legacy)
    assert result["research"]["status"] is module.RuntimeState.RUNNING
    assert result["research"]["database"]["status"] == "UNAVAILABLE"
    assert module.RuntimeReason.RESEARCH_DATABASE_UNAVAILABLE in result["research"]["reason_codes"]
    assert result["attention_required"]


@pytest.fixture
def journal(tmp_path):
    path = tmp_path / "journal.sqlite"
    base = load_settings(Path("docs/testnet-funding-acceptance.toml"))
    settings = base.model_copy(
        update={
            "testnet": base.testnet.model_copy(
                update={"journal_path": path, "max_total_attempts": 2}
            )
        }
    )
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(
            "CREATE TABLE testnet_pair_attempts(attempt_id TEXT,plan_json TEXT,state TEXT,failure_code TEXT,created_at TEXT)"
        )
        for i, state in enumerate(("CLOSED", "FLATTENED")):
            db.execute(
                "INSERT INTO testnet_pair_attempts VALUES(?,?,?,?,?)",
                (
                    f"id{i}",
                    exact_json({"spot_entry": {"symbol": "ETHUSDT"}}),
                    state,
                    None,
                    OPENED.isoformat(),
                ),
            )
    return settings, path


def test_testnet_local_terminal_is_not_current_exchange_flatness(journal):
    settings, path = journal
    before = path.read_bytes()
    result = module.runtime_status(
        testnet_settings=settings, max_heartbeat_age=timedelta(seconds=60)
    )
    data = result["testnet"]
    assert (
        data["scope"] == "LOCAL_ARCHIVE_NOT_EXCHANGE_ACCOUNT"
        and not data["exchange_account_flat_verified"]
    )
    assert data["budget_consumed"] == data["budget_limit"] == 2
    assert module.RuntimeReason.ORDER_BUDGET_EXHAUSTED in data["reason_codes"]
    assert not result["attention_required"] and path.read_bytes() == before


@pytest.mark.parametrize(
    "state", ["HEDGED", "RECOVERY_REQUIRED", "OPENING_PERP", "UNKNOWN_FUTURE_STATE"]
)
def test_active_and_unknown_attempts_remain_visible_and_require_attention(journal, state):
    settings, path = journal
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(
            "UPDATE testnet_pair_attempts SET state=?,failure_code=? WHERE attempt_id=?",
            (state, "SYNTHETIC_FAILURE", "id0"),
        )
    data = module.runtime_status(testnet_settings=settings, max_heartbeat_age=timedelta(seconds=60))
    assert data["attention_required"] and len(data["testnet"]["active_attempts"]) == 1
    assert data["testnet"]["active_attempts"][0]["state"] == state
    if state == "UNKNOWN_FUTURE_STATE":
        assert module.RuntimeReason.UNKNOWN_ATTEMPT_STATE in data["testnet"]["reason_codes"]
    elif state == "OPENING_PERP":
        assert module.RuntimeReason.UNKNOWN_ATTEMPT_STATE not in data["testnet"]["reason_codes"]


def test_held_testnet_lock_cannot_be_named_verified_worker(journal):
    settings, path = journal
    path.with_suffix(".lock").write_bytes(b"0")
    data = module.testnet_status(settings, lock_probe=lambda p: True)
    assert data["process_observation"] == "LOCK_HELD_OWNER_UNVERIFIED"
    assert "worker_pid" not in data


@pytest.mark.parametrize("kind", ["missing", "invalid", "locked"])
def test_journal_failure_is_explicit_and_does_not_create_or_repair(journal, kind):
    settings, path = journal
    connection = None
    if kind == "missing":
        path.unlink()
    elif kind == "invalid":
        path.write_bytes(b"not sqlite")
    else:
        connection = sqlite3.connect(path)
        connection.execute("BEGIN EXCLUSIVE")
    try:
        result = module.runtime_status(
            testnet_settings=settings, max_heartbeat_age=timedelta(seconds=60)
        )
        assert (
            result["attention_required"]
            and module.RuntimeReason.JOURNAL_UNAVAILABLE in result["testnet"]["reason_codes"]
        )
        if kind == "missing":
            assert not path.exists()
    finally:
        if connection is not None:
            connection.close()


@pytest.mark.parametrize("age", [timedelta(0), timedelta(seconds=-1), timedelta(days=2), 60])
def test_explicit_operational_age_limits_only(legacy, age):
    with pytest.raises(ValueError):
        module.runtime_status(
            research_settings=legacy[0], research_directory=legacy[1], max_heartbeat_age=age
        )


def monitor(tmp_path, check, wait=None, **changes):
    elapsed = [0]
    waits = []

    def default_wait(seconds):
        waits.append(seconds)
        elapsed[0] += seconds
        return False

    result = module.monitor_status(
        check,
        tmp_path / "monitor",
        duration_seconds=changes.get("duration", 11),
        interval_seconds=5,
        max_checks=changes.get("max_checks", 3),
        clock=lambda: OPENED + timedelta(seconds=elapsed[0]),
        steady_ns=lambda: int(elapsed[0] * 1_000_000_000),
        wait=wait or default_wait,
    )
    return result, waits


def test_monitor_reports_meaningful_change_not_every_heartbeat(tmp_path):
    step = [0]

    def check():
        step[0] += 1
        return {
            "evaluated_at": OPENED + timedelta(seconds=step[0]),
            "research": {
                "status": "RUNNING_CONFIRMED",
                "updated_at": OPENED + timedelta(seconds=step[0]),
                "budget_consumed": step[0],
                "database": {"decisions": {"REJECT": step[0]}, "tasks": {"PENDING": 1}},
                "reason_codes": (),
            },
        }

    result, waits = monitor(tmp_path, check)
    assert result["checks"] == 3 and result["changes"] == 1 and waits == [5, 5]
    assert len(list((tmp_path / "monitor").glob("event-*.json"))) == 1


def test_process_death_emits_single_local_change_and_no_restart(tmp_path):
    check = Mock(
        side_effect=[
            {"research": {"status": "RUNNING_CONFIRMED"}},
            {"research": {"status": "STOPPED_CONFIRMED", "reason_codes": ("WORKER_MISSING",)}},
            {"research": {"status": "STOPPED_CONFIRMED", "reason_codes": ("WORKER_MISSING",)}},
        ]
    )
    result, waits = monitor(tmp_path, check)
    assert result["changes"] == 2 and check.call_count == 3 and waits == [5, 5]
    event = json.loads((tmp_path / "monitor/event-00001.json").read_text())
    assert event["status"]["research"]["status"] == "STOPPED_CONFIRMED"
    assert "worker_started" not in event and not (tmp_path / "research").exists()


def test_validator_completion_is_meaningful_even_when_owner_unchanged(tmp_path):
    check = Mock(
        side_effect=[
            {"research": {"status": "RUNNING_CONFIRMED", "database": {"tasks": {"PENDING": 1}}}},
            {"research": {"status": "RUNNING_CONFIRMED", "database": {"tasks": {"COMPLETED": 1}}}},
        ]
    )
    result, waits = monitor(tmp_path, check, max_checks=2)
    assert result["changes"] == 2 and result["checks"] == 2


def test_monitor_check_failure_redacted_and_finite(tmp_path):
    check = Mock(side_effect=RuntimeError("SECRET_RAW_ACCOUNT"))
    result, waits = monitor(tmp_path, check)
    assert result["checks"] == 3 and result["changes"] == 1 and waits == [5, 5]
    assert all(
        "SECRET_RAW_ACCOUNT" not in p.read_text() for p in (tmp_path / "monitor").glob("*.json")
    )


def test_monitor_stop_and_keyboard_interrupt_are_bounded(tmp_path):
    def wait(seconds):
        (tmp_path / "monitor/stop").touch()
        return True

    assert monitor(tmp_path, Mock(return_value={}), wait=wait)[0]["state"] == "STOPPED"
    other = tmp_path / "second"

    def interrupted():
        raise KeyboardInterrupt()

    assert monitor(other, interrupted)[0]["state"] == "INTERRUPTED"


def test_monitor_deadline_uses_monotonic_even_if_wall_clock_frozen(tmp_path):
    ticks = iter((0, 0, 12_000_000_000))
    check = Mock(return_value={})
    result = module.monitor_status(
        check,
        tmp_path / "monitor",
        duration_seconds=11,
        interval_seconds=5,
        max_checks=100,
        clock=lambda: OPENED,
        steady_ns=lambda: next(ticks),
        wait=lambda _: False,
    )
    assert result["checks"] == 1


def test_monitor_cannot_overwrite_prior_plan_or_reset_count(tmp_path):
    monitor(tmp_path, Mock(return_value={}), max_checks=1)
    before = (tmp_path / "monitor/plan.json").read_bytes()
    with pytest.raises(ValueError):
        monitor(tmp_path, Mock(return_value={}), max_checks=1)
    assert (tmp_path / "monitor/plan.json").read_bytes() == before


def test_one_cli_reports_both_scopes_without_credentials(legacy, journal, monkeypatch, capsys):
    settings, directory, state = legacy
    config = json.loads((directory / "recovery-binding.json").read_text())["config_path"]
    monkeypatch.setattr(
        module, "load_settings", lambda p: settings if str(p) == config else journal[0]
    )
    result = module.main(
        [
            "status",
            "--research-config",
            config,
            "--research-dir",
            str(directory),
            "--testnet-config",
            "synthetic.toml",
            "--max-heartbeat-age-seconds",
            "60",
        ]
    )
    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert "research" in report and "testnet" in report and not report["credentials_read"]
