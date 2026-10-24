"""Synthetic preregistered research: original math, temporary DB, no network."""

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_forward_validation import later_snapshot, shift_fixture
from test_offline_estimate import OPENED, assembly_for, books
from test_runner import settings_for
from test_scenario import event

from crypto_arbitrage_bot.analytics.forward_validation import validate_forward_outcome
from crypto_arbitrage_bot.analytics.registered_study import StudyReadStatus, read_registered_study
from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
from crypto_arbitrage_bot.models.strategy_study import (
    StrategyStudyProtocol,
    StudyReason,
    StudyStatus,
)
from crypto_arbitrage_bot.models.validation_task import TaskFailure
from crypto_arbitrage_bot.runner import CycleStage, CycleStatus, ResearchRunner
from crypto_arbitrage_bot.storage.forward_validation_repository import (
    SQLiteForwardValidationRepository,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import (
    SQLiteConfig,
    SQLiteOpportunityRepository,
    _columns,
    _encode,
)
from crypto_arbitrage_bot.storage.study_attempts import finish_attempt, start_attempt
from crypto_arbitrage_bot.storage.study_registry import (
    StrategyStudyRegistry,
    StudyArchiveError,
    StudyArchiveFailure,
    full_research_policy,
    load_registered_study,
    study_cost_fingerprints,
)
from crypto_arbitrage_bot.storage.validation_tasks import SQLiteValidationTaskRepository

D = Decimal
R = StudyReason
F = StudyArchiveFailure


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("registry tests cannot contact an exchange"))
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("http.client.HTTPSConnection", forbidden)


class Cohort:
    def __init__(self, tmp_path, symbols=("BTCUSDT",)):
        base = settings_for(tmp_path, symbols=symbols)
        scenarios = {s: base.research.scenarios[s].model_copy(update={
            "funding_events": tuple(replace(event(".1", symbol=s), event_id=f"synthetic-{day}",
                                   settled_at=event(".1", symbol=s).settled_at+timedelta(days=day))
                                    for day in range(5))}) for s in symbols}
        self.settings = base.model_copy(update={"research": base.research.model_copy(update={"scenarios": scenarios})})
        self.clock = OPENED-timedelta(hours=1)
        self.registry = StrategyStudyRegistry(tmp_path / "registry", clock=lambda: self.clock)
        self.protocol = StrategyStudyProtocol(
            id="synthetic-fixed-study", registered_at=OPENED-timedelta(days=1), study_start=OPENED,
            study_end=OPENED+timedelta(days=3), symbols=symbols, formula_version="synthetic-1",
            config_version="synthetic-1", config_fingerprints=study_cost_fingerprints(self.settings),
            max_holding_period=timedelta(hours=2), min_entry_separation=timedelta(days=1),
            max_exit_lateness=timedelta(seconds=300), min_entries_per_symbol=2, min_entry_days_per_symbol=2,
            min_mean_net_return=D(".001"), min_median_net_return=D(".001"), max_loss_return=D(".1"))
        # Sparse synthetic clocks only; not a production sampling recommendation.
        self.study = self.registry.register(self.protocol, self.settings, max_capture_delay=timedelta(seconds=60),
                                            max_sampling_gap=timedelta(days=2))
        self.db = SQLiteConfig(self.settings.research.sqlite_path, busy_timeout_ms=0)
        self.repository = SQLiteOpportunityRepository(self.db)
        self.repository.initialize()
        self.validations = SQLiteForwardValidationRepository(self.db)
        self.validations.initialize()
        self.tasks = SQLiteValidationTaskRepository(self.db)
        self.tasks.initialize()

    def add(self, day=0, *, capture=True, validate=True, symbol="BTCUSDT", exit_ask="112"):
        self.clock = OPENED+timedelta(days=day)
        settings = self.settings.model_copy(update={"symbols": (symbol,)})
        collector = Mock()
        collector.collect.return_value = shift_fixture(assembly_for((books(symbol)[0], books(symbol)[2])), timedelta(days=day))
        runner = ResearchRunner(settings, collector, self.repository, clock=lambda: self.clock, logger=Mock())
        in_window = self.protocol.study_start <= self.clock < self.protocol.study_end
        attempt_id = f"fixture-{symbol}-{day}-{len(self.repository.list_recent_opportunities())}"
        if in_window:
            start_attempt(self.study, attempt_id, symbol, self.clock, recorded_at=self.clock)
        cycle, = runner.run_once()
        assert cycle.status.value == "SAVED" and cycle.decision == "PASS"
        record = self.repository.get_opportunity_by_id(cycle.opportunity_id)
        if in_window:
            finish_attempt(self.study, attempt_id, symbol, ended_at=self.clock, status="SAVED", failure_stage=None,
                           reason_codes=(), record=record, recorded_at=self.clock)
        if capture:
            self.clock += timedelta(seconds=1)
            self.registry.capture(self.study, record, self.settings)
        self.tasks.create_tasks(record.id, (7200, 86400), now=record.created_at)
        if validate:
            at = record.created_at+timedelta(hours=2)
            task = self.tasks.list_for_opportunity(record.id)[0]
            claimed = self.tasks.mark_running(task.id, now=at, lease_seconds=300)
            result = validate_forward_outcome(record, later_snapshot(symbol, perp_ask=exit_ask, at=at),
                                             validation_time=at, funding_events=())
            assert result.status.value == "VALIDATED"
            assert self.validations.save_forward_validation(result, validation_id=task.validation_id, created_at=at).saved_id
            assert self.tasks.finish(claimed, now=at)
        return record

    def read(self, at=None):
        return read_registered_study(self.study.path, self.study.sha256,
                                     evaluated_at=at or OPENED+timedelta(days=4), busy_timeout_ms=0)

    def sql(self, statement, values=()):
        with closing(sqlite3.connect(self.db.database_path)) as db, db:
            db.execute(statement, values)


@pytest.fixture
def cohort(tmp_path):
    return Cohort(tmp_path)


def test_registered_archive_pins_full_scenario_policy_and_exact_utc(cohort):
    saved = load_registered_study(cohort.study.path, cohort.study.sha256)
    assert saved.protocol == cohort.protocol
    assert saved.registered_at == OPENED-timedelta(hours=1)
    policy = json.loads(saved.policy_json)
    assert policy["research"]["scenarios"]["BTCUSDT"]["adverse_exit_basis_pct"] == "0.16"
    assert "funding_events" in policy["research"]["scenarios"]["BTCUSDT"]
    assert policy["fees"]["spot_entry"] == "0.001"
    assert saved.database_path == cohort.db.database_path and "testnet" not in policy


def test_identical_registration_after_start_is_idempotent_not_new_timestamp(cohort):
    before = cohort.study.path.read_bytes()
    cohort.clock = OPENED+timedelta(days=5)
    same = cohort.registry.register(cohort.protocol, cohort.settings, max_capture_delay=timedelta(seconds=60))
    assert same == cohort.study and cohort.study.path.read_bytes() == before


@pytest.mark.parametrize("change", ["threshold", "scenario", "interval", "database", "delay"])
def test_same_id_cannot_change_registered_policy(cohort, change):
    settings = cohort.settings
    protocol = cohort.protocol
    delay = timedelta(seconds=60)
    if change == "threshold":
        protocol = replace(protocol, min_mean_net_return=D(".0001"))
    elif change == "scenario":
        scenarios = {"BTCUSDT": settings.research.scenarios["BTCUSDT"].model_copy(update={"adverse_exit_basis_pct": D(".17")})}
        settings = settings.model_copy(update={"research": settings.research.model_copy(update={"scenarios": scenarios})})
    elif change == "interval":
        settings = settings.model_copy(update={"poll_interval_seconds": 30})
    elif change == "database":
        settings = settings.model_copy(update={"research": settings.research.model_copy(update={"sqlite_path": Path("other.sqlite")})})
    else:
        delay = timedelta(seconds=61)
    before = cohort.study.path.read_bytes()
    with pytest.raises(StudyArchiveError) as error:
        cohort.registry.register(protocol, settings, max_capture_delay=delay)
    assert error.value.code is F.REGISTRATION_CONFLICT
    assert cohort.study.path.read_bytes() == before


def test_declared_old_date_cannot_register_after_study_start(cohort):
    cohort.clock = OPENED+timedelta(seconds=1)
    with pytest.raises(StudyArchiveError) as error:
        cohort.registry.register(replace(cohort.protocol, id="new-retrodated"), cohort.settings,
                                 max_capture_delay=timedelta(seconds=60))
    assert error.value.code is F.REGISTRATION_NOT_PRIOR


def test_forged_cost_fingerprint_cannot_be_registered(cohort):
    p = replace(cohort.protocol, id="bad-cost", config_fingerprints=(("BTCUSDT", "0"*64),))
    with pytest.raises(StudyArchiveError) as error:
        cohort.registry.register(p, cohort.settings, max_capture_delay=timedelta(seconds=60))
    assert error.value.code is F.INVALID_REGISTRATION


def test_unsafe_id_is_only_hashed_filename(cohort):
    study = cohort.registry.register(replace(cohort.protocol, id="../../escaping"), cohort.settings,
                                    max_capture_delay=timedelta(seconds=60))
    assert study.path.parent == cohort.registry.directory and len(study.path.stem) == 64


def test_empty_initialized_database_is_insufficient_without_fake_zero(cohort):
    result = cohort.read()
    assert result.status is StudyReadStatus.READ
    assert result.assessment.status is StudyStatus.INSUFFICIENT_EVIDENCE
    assert not result.opportunity_ids and result.assessment.symbols[0].profit.mean is None


def test_full_positive_cohort_uses_original_outputs_and_database_unchanged(cohort):
    entries = (cohort.add(), cohort.add(1))
    before = cohort.db.database_path.read_bytes()
    files = {p.name: p.read_bytes() for p in cohort.study.receipt_directory.iterdir()}
    result = cohort.read()
    assert result.status is StudyReadStatus.READ and result.assessment.status is StudyStatus.READY_FOR_EXECUTION_REVIEW
    assert result.opportunity_ids == tuple(r.id for r in entries)
    assert result.assessment.symbols[0].profit.mean == D("6.9688")
    assert isinstance(result.assessment.symbols[0].net_return.mean, D)
    assert len(result.tasks) == 4 and [t.status for t in result.tasks].count("PENDING") == 2
    assert cohort.db.database_path.read_bytes() == before
    assert {p.name: p.read_bytes() for p in cohort.study.receipt_directory.iterdir()} == files
    assert not hasattr(result, "trade_allowed")


def test_failed_exit_retained_no_selection_of_only_successes(cohort):
    first = cohort.add(validate=False)
    cohort.add(1)
    at = first.created_at+timedelta(hours=2)
    task = cohort.tasks.list_for_opportunity(first.id)[0]
    claimed = cohort.tasks.mark_running(task.id, now=at, lease_seconds=300)
    assert cohort.tasks.finish(claimed, now=at, failure=TaskFailure.FUNDING_UNAVAILABLE, reason="synthetic failure")
    result = cohort.read()
    assert len(result.opportunity_ids) == 2 and len(result.validation_ids) == 1
    assert any(t.failure_code == "FUNDING_UNAVAILABLE" for t in result.tasks)
    assert R.MISSING_PLANNED_EXIT_VALIDATION in result.assessment.reason_codes
    assert result.assessment.status is StudyStatus.INSUFFICIENT_EVIDENCE


def test_good_manual_exit_cannot_replace_uncompleted_planned_task(cohort):
    record = cohort.add(validate=False)
    cohort.add(1)
    at = record.created_at+timedelta(hours=2)
    result = validate_forward_outcome(record, later_snapshot(at=at), validation_time=at, funding_events=())
    cohort.validations.save_forward_validation(result, validation_id="manual-good-exit", created_at=at)
    evidence = cohort.read()
    assert R.PLANNED_TASK_NOT_COMPLETED in evidence.assessment.reason_codes
    assert evidence.assessment.status is StudyStatus.INSUFFICIENT_EVIDENCE


def test_missing_capture_is_not_repaired_or_discarded(cohort):
    record = cohort.add(capture=False)
    cohort.add(1)
    result = cohort.read()
    assert record.id in result.opportunity_ids and R.MISSING_CAPTURE_RECEIPT in result.assessment.reason_codes
    assert len(tuple(cohort.study.receipt_directory.iterdir())) == 1


def test_capture_is_compact_idempotent_and_no_book_duplication(cohort):
    record = cohort.add()
    path = cohort.registry.capture(cohort.study, record, cohort.settings)
    before = path.read_bytes()
    cohort.clock = OPENED+timedelta(days=10)
    assert cohort.registry.capture(cohort.study, record, cohort.settings) == path
    assert path.read_bytes() == before and len(before) < 700
    assert "raw_response" not in before.decode() and "snapshot" not in before.decode()


def test_capture_cannot_happen_after_outcome_is_known(cohort):
    record = cohort.add(capture=False)
    cohort.clock = record.created_at+timedelta(hours=2)
    with pytest.raises(StudyArchiveError) as error:
        cohort.registry.capture(cohort.study, record, cohort.settings)
    assert error.value.code is F.CAPTURE_TOO_LATE


@pytest.mark.parametrize("changed", ["record", "policy", "timestamp"])
def test_changed_capture_evidence_is_insufficient(cohort, changed):
    record = cohort.add()
    cohort.add(1)
    path = cohort.registry.capture(cohort.study, record, cohort.settings)
    value = json.loads(path.read_text(encoding="utf-8"))
    if changed == "record":
        value["record_sha256"] = "0"*64
    elif changed == "policy":
        value["policy_sha256"] = "0"*64
    else:
        value["captured_at"] = (OPENED+timedelta(hours=2)).isoformat()
    path.write_text(exact_json(value), encoding="utf-8")
    result = cohort.read()
    assert R.CAPTURE_RECEIPT_MISMATCH in result.assessment.reason_codes
    assert result.assessment.status is StudyStatus.INSUFFICIENT_EVIDENCE


def test_losing_original_record_cannot_hide_bad_outcome(cohort):
    record = cohort.add(validate=False)
    cohort.add(1)
    cohort.sql("DELETE FROM validation_tasks WHERE opportunity_id=?", (record.id,))
    cohort.sql("DELETE FROM opportunity_sources WHERE opportunity_id=?", (record.id,))
    cohort.sql("DELETE FROM opportunity_scenarios WHERE opportunity_id=?", (record.id,))
    cohort.sql("DELETE FROM opportunities WHERE id=?", (record.id,))
    result = cohort.read()
    assert R.ORPHAN_CAPTURE_RECEIPT in result.assessment.reason_codes


def test_rejected_record_stays_in_full_cohort(cohort):
    record = cohort.add(capture=False, validate=False)
    details = json.loads(record.details_json)
    # Storage alteration is not a real decision. Its missing hash receipt must
    # be visible rather than excluded by a SELECT decision='PASS' shortcut.
    cohort.sql("UPDATE opportunities SET decision='REJECT' WHERE id=?", (record.id,))
    result = cohort.read()
    assert result.opportunity_ids == (record.id,) and R.MISSING_CAPTURE_RECEIPT in result.assessment.reason_codes
    assert details["estimate"]["calculation"] is not None


def test_original_cost_version_mismatch_not_filtered_out(cohort):
    record = cohort.add()
    cohort.sql("UPDATE opportunities SET config_version='changed' WHERE id=?", (record.id,))
    result = cohort.read()
    assert record.id in result.opportunity_ids
    assert {R.CONFIG_MISMATCH, R.CAPTURE_RECEIPT_MISMATCH}.issubset(result.assessment.reason_codes)


def test_registry_tamper_fails_pinned_reference(cohort):
    value = json.loads(cohort.study.path.read_text(encoding="utf-8"))
    value["protocol"]["min_mean_net_return"] = "0"
    cohort.study.path.write_text(exact_json(value), encoding="utf-8")
    result = cohort.read()
    assert result.status is StudyReadStatus.FAILED and result.failure is F.ARCHIVE_HASH_MISMATCH
    assert result.assessment is None


@pytest.mark.parametrize("mode", ["missing", "schema", "no_tables", "lock"])
def test_read_failure_never_creates_or_repairs_database(cohort, mode):
    if mode == "missing":
        cohort.db.database_path.unlink()
    elif mode == "schema":
        cohort.sql("PRAGMA user_version=77")
    elif mode == "no_tables":
        cohort.sql("DROP TABLE forward_validations")
    if mode == "lock":
        with closing(sqlite3.connect(cohort.db.database_path)) as holder, holder:
            holder.execute("BEGIN EXCLUSIVE")
            result = cohort.read()
    else:
        result = cohort.read()
    assert result.status is StudyReadStatus.FAILED and result.assessment is None
    assert result.failure is F.DATABASE_READ_FAILED
    if mode == "missing":
        assert not cohort.db.database_path.exists()


def test_only_readonly_sqlite_and_no_repository_initialize(cohort, monkeypatch):
    cohort.add()
    connection = sqlite3.connect
    seen = []
    def open_readonly(database_uri, **kwargs):
        assert database_uri.endswith("?mode=ro") and kwargs["uri"] is True
        db = connection(database_uri, **kwargs)
        db.set_trace_callback(seen.append)
        return db
    monkeypatch.setattr(sqlite3, "connect", open_readonly)
    monkeypatch.setattr(SQLiteOpportunityRepository, "initialize", Mock(side_effect=AssertionError("no initialization")))
    monkeypatch.setattr(SQLiteOpportunityRepository, "list_recent_opportunities", Mock(side_effect=AssertionError("no partial query")))
    assert cohort.read().status is StudyReadStatus.READ
    assert "PRAGMA query_only=ON" in seen and "BEGIN" in seen
    assert not any(s.upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER")) for s in seen)


def test_separate_symbols_require_separate_evidence(tmp_path):
    cohort = Cohort(tmp_path, symbols=("BTCUSDT", "ETHUSDT"))
    cohort.add()
    cohort.add(1)
    cohort.add(symbol="ETHUSDT")
    result = cohort.read()
    assert len(result.opportunity_ids) == 3 and result.assessment.status is StudyStatus.INSUFFICIENT_EVIDENCE
    assert any(i.symbol == "ETHUSDT" and i.code is R.TOO_FEW_SEPARATED_ENTRIES for i in result.assessment.issues)


def test_policy_excludes_paths_but_includes_complete_scenario(cohort):
    settings = cohort.settings
    policy = full_research_policy(settings)
    changed = settings.model_copy(update={"log_dir": Path("another-log"), "log_level": "DEBUG"})
    assert full_research_policy(changed) == policy
    research = settings.research.model_copy(update={"scenarios": {
        "BTCUSDT": settings.research.scenarios["BTCUSDT"].model_copy(update={"holding_seconds": 3600})}})
    assert full_research_policy(settings.model_copy(update={"research": research})) != policy
    # This explains why the old record cost hash alone could not pin an exit policy.
    assert study_cost_fingerprints(settings.model_copy(update={"research": research})) == study_cost_fingerprints(settings)


def test_receipt_write_failure_does_not_mutate_original(cohort, monkeypatch):
    record = cohort.add(capture=False)
    before = cohort.db.database_path.read_bytes()
    original = exact_json(record)
    monkeypatch.setattr("crypto_arbitrage_bot.storage.study_registry._exclusive_write", Mock(side_effect=OSError("synthetic failure")))
    with pytest.raises(StudyArchiveError) as error:
        cohort.registry.capture(cohort.study, record, cohort.settings)
    assert error.value.code is F.ARCHIVE_IO_ERROR
    assert cohort.db.database_path.read_bytes() == before and exact_json(record) == original


def test_no_overwrite_if_original_receipt_conflicts(cohort):
    record = cohort.add()
    before = {p.name: p.read_bytes() for p in cohort.study.receipt_directory.iterdir()}
    with pytest.raises(StudyArchiveError) as error:
        cohort.registry.capture(cohort.study, replace(record, net_profit=D("999")), cohort.settings)
    assert error.value.code is F.RECEIPT_CONFLICT
    assert {p.name: p.read_bytes() for p in cohort.study.receipt_directory.iterdir()} == before


def test_negative_complete_cohort_is_unsupported_not_positive_selection(cohort):
    cohort.add(exit_ask="124")
    cohort.add(1, exit_ask="124")
    result = cohort.read()
    assert result.assessment.status is StudyStatus.UNSUPPORTED
    assert result.assessment.symbols[0].profit.mean == D("-17.0504")


def test_reader_does_not_apply_recent_repository_thousand_row_limit(cohort):
    source = cohort.add(capture=False, validate=False)
    columns = _columns(type(source))
    with closing(sqlite3.connect(cohort.db.database_path)) as db, db:
        db.executemany(f"INSERT INTO opportunities ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                       (_encode(replace(source, id=f"saved-rejected-{number}", decision="REJECT")) for number in range(1001)))
    result = cohort.read()
    assert result.status is StudyReadStatus.READ and len(result.opportunity_ids) == 1002
    assert len(result.assessment.reason_codes) > 0 and result.assessment.status is StudyStatus.INSUFFICIENT_EVIDENCE


def test_completed_task_missing_its_fixed_validation_is_not_success(cohort):
    record = cohort.add()
    cohort.sql("UPDATE validation_tasks SET validation_id='missing-fixed-id' WHERE opportunity_id=? AND horizon_seconds=7200", (record.id,))
    result = cohort.read()
    assert R.TASK_VALIDATION_CONFLICT in result.assessment.reason_codes
    assert R.PLANNED_TASK_NOT_COMPLETED in result.assessment.reason_codes


def test_task_cannot_claim_completion_before_validation_capture(cohort):
    record = cohort.add()
    cohort.add(1)
    cohort.sql("UPDATE validation_tasks SET updated_at=? WHERE opportunity_id=? AND horizon_seconds=7200",
               (OPENED.isoformat(timespec="microseconds"), record.id))
    result = cohort.read()
    assert R.PLANNED_TASK_NOT_COMPLETED in result.assessment.reason_codes
    assert result.assessment.status is StudyStatus.INSUFFICIENT_EVIDENCE


def test_receipt_actual_registration_later_than_source_rejected(cohort):
    record = cohort.add()
    value = json.loads(cohort.study.path.read_text(encoding="utf-8"))
    value["registered_at"] = OPENED.isoformat(timespec="microseconds")
    text = exact_json(value)
    cohort.study.path.write_text(text, encoding="utf-8")
    # Even with a newly pinned local reference, earlier public captures cannot
    # become registered evidence simply because the declared date was old.
    result = read_registered_study(cohort.study.path, hashlib.sha256(text.encode()).hexdigest(),
                                   evaluated_at=OPENED+timedelta(days=4))
    assert result.status is StudyReadStatus.READ
    assert R.NON_PRODUCTION_SOURCE in result.assessment.reason_codes
    assert record.id in result.opportunity_ids


def registered_runner(cohort, *, scheduler=None):
    reference = StudyRegistrationSettings(path=cohort.study.path, sha256=cohort.study.sha256)
    settings = cohort.settings.model_copy(update={"research": cohort.settings.research.model_copy(
        update={"study_registration": reference})})
    collector = Mock()
    collector.collect.side_effect = lambda s: assembly_for((books(s)[0], books(s)[2]))
    cohort.clock = OPENED
    runner = ResearchRunner(settings, collector, cohort.repository, clock=lambda: cohort.clock,
                            scheduler=scheduler, logger=Mock())
    return runner, collector


def scheduler_for(cohort):
    scheduler = Mock()
    scheduler.opportunities = cohort.repository
    return scheduler


def test_runner_automatically_captures_the_exact_saved_record(cohort):
    runner, collector = registered_runner(cohort)
    cycle, = runner.run_once()
    assert cycle.status is CycleStatus.SAVED and collector.collect.call_count == 1
    record = cohort.repository.get_opportunity_by_id(cycle.opportunity_id)
    files = tuple(cohort.study.receipt_directory.iterdir())
    assert len(files) == 1
    receipt = json.loads(files[0].read_text(encoding="utf-8"))
    assert receipt["record_sha256"] == hashlib.sha256(exact_json(record).encode()).hexdigest()
    assert receipt["opportunity_id"] == record.id and receipt["captured_at"] == exact_json(OPENED).strip('"')


def test_runner_registration_failure_does_not_destroy_decision_or_skip_scheduler(cohort, monkeypatch):
    scheduler = scheduler_for(cohort)
    runner, _ = registered_runner(cohort, scheduler=scheduler)
    monkeypatch.setattr(runner.study_registry, "capture", Mock(side_effect=StudyArchiveError(F.ARCHIVE_IO_ERROR)))
    cycle, = runner.run_once()
    assert cycle.status is CycleStatus.FAILED and cycle.failure_stage is CycleStage.STUDY_REGISTRATION
    record = cohort.repository.get_opportunity_by_id(cycle.opportunity_id)
    assert record.decision == cycle.decision == "PASS" and record.net_profit == D("9.0144")
    scheduler.schedule.assert_called_once_with(record.id)
    scheduler.process_due.assert_called_once()
    assert R.MISSING_CAPTURE_RECEIPT in cohort.read().assessment.reason_codes


def test_runner_btc_receipt_failure_does_not_block_eth(tmp_path, monkeypatch):
    cohort = Cohort(tmp_path, ("BTCUSDT", "ETHUSDT"))
    runner, _ = registered_runner(cohort)
    original = runner.study_registry.capture
    def capture(study, record, settings):
        if record.symbol == "BTCUSDT":
            raise StudyArchiveError(F.ARCHIVE_IO_ERROR)
        return original(study, record, settings)
    monkeypatch.setattr(runner.study_registry, "capture", capture)
    first, second = runner.run_once()
    assert first.status is CycleStatus.FAILED and first.opportunity_id is not None
    assert second.status is CycleStatus.SAVED and second.symbol == "ETHUSDT"
    assert len(cohort.repository.list_recent_opportunities()) == 2


def test_registered_runner_refuses_full_policy_mismatch_at_construction(cohort):
    reference = StudyRegistrationSettings(path=cohort.study.path, sha256=cohort.study.sha256)
    changed = cohort.settings.model_copy(update={"poll_interval_seconds": 30,
        "research": cohort.settings.research.model_copy(update={"study_registration": reference})})
    collector = Mock()
    with pytest.raises(StudyArchiveError) as error:
        ResearchRunner(changed, collector, cohort.repository)
    assert error.value.code is F.POLICY_MISMATCH
    collector.collect.assert_not_called()


def test_registry_modified_during_run_blocks_before_collection(cohort):
    runner, collector = registered_runner(cohort)
    cohort.study.path.write_text("{}", encoding="utf-8")
    cycle, = runner.run_once()
    assert cycle.failure_stage is CycleStage.STUDY_REGISTRATION and cycle.opportunity_id is None
    assert "ARCHIVE_HASH_MISMATCH" in cycle.failure_reason
    collector.collect.assert_not_called()


@pytest.mark.parametrize("at", [OPENED-timedelta(seconds=1), OPENED+timedelta(days=3)])
def test_runner_no_new_capture_outside_fixed_window(cohort, at):
    scheduler = scheduler_for(cohort)
    runner, collector = registered_runner(cohort, scheduler=scheduler)
    cohort.clock = at
    assert runner.run_once() == ()
    collector.collect.assert_not_called()
    scheduler.process_due.assert_called_once()
    assert not runner.stop_signal.is_set()


def test_registered_runner_stops_after_final_bounded_exit_allowance(cohort):
    scheduler = scheduler_for(cohort)
    runner, collector = registered_runner(cohort, scheduler=scheduler)
    cohort.clock = cohort.protocol.study_end+cohort.protocol.max_holding_period+cohort.protocol.max_exit_lateness
    runner.run_forever()
    assert runner.stop_signal.is_set()
    collector.collect.assert_not_called()
    scheduler.process_due.assert_not_called()


@pytest.mark.parametrize("digest", ["", "a"*63, "G"*64, "A"*64, "0"*65])
def test_config_requires_pinned_registration_digest(digest):
    with pytest.raises(ValueError):
        StudyRegistrationSettings(path=Path("study.json"), sha256=digest)
