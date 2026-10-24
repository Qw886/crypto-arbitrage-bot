"""Candidate registration/attempts/planned exits with original math, no network."""

import gzip
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_candidate_catalog import NOW, instrument, source
from test_candidate_market import collector, install
from test_candidate_market import offline as offline

from crypto_arbitrage_bot import candidate_study as module
from crypto_arbitrage_bot.analytics.registered_study import StudyReadStatus, read_registered_study
from crypto_arbitrage_bot.config.research import (
    FundingProjectionSettings,
    StudyRegistrationSettings,
)
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.market_data.candidate_catalog import build_candidate_catalog
from crypto_arbitrage_bot.models.validation_task import TaskFailure, TaskStatus
from crypto_arbitrage_bot.runner import CycleStatus, create_runner
from crypto_arbitrage_bot.storage.study_attempts import read_attempt_coverage
from crypto_arbitrage_bot.storage.study_registry import StudyArchiveError

D = Decimal


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("no network")))
    install(monkeypatch)
    settings = load_settings(Path("docs/research-candidate-costs.toml"))
    research = settings.research
    criteria = research.candidate_catalog.model_copy(update={"min_perp_age_seconds": 86400,
        "min_remaining_lifetime_seconds": 86400})
    profile = research.candidate_economics.windows["8h"].model_copy(update={
        "funding_projection": FundingProjectionSettings(mode="next_settlement", positive_rate_multiplier="0.5",
            max_wait_seconds=7200, exit_delay_seconds=30, min_entry_lead_seconds=5, entry_window_seconds=7200)})
    research = research.model_copy(update={"sqlite_path": tmp_path / "new.sqlite",
        "candidate_catalog": criteria, "candidate_economics": research.candidate_economics.model_copy(update={"windows": {"next": profile}})})
    settings = settings.model_copy(update={"research": research})
    assets = ("BTC", "ETH", "THE", "TST")
    catalog = build_candidate_catalog(source([instrument(a) for a in assets]),
        source([instrument(a, True) for a in assets], True), criteria, now=NOW)
    kwargs = {"symbols": tuple(a+"USDT" for a in assets), "id_prefix": "fixed-four", "study_start": NOW,
        "study_end": NOW+timedelta(days=3), "min_entry_separation": timedelta(seconds=7530), "min_entries": 6,
        "min_days": 3, "min_mean_return": D(".001"), "min_median_return": D(".001"), "max_loss_return": D(".02"),
        "max_capture_delay": timedelta(seconds=30), "max_sampling_gap": timedelta(minutes=30),
        "entry_interval_seconds": 900, "clock": lambda: NOW, "catalog": catalog}
    path, sha = module.prepare_candidate_index(settings, tmp_path / "registry", **kwargs)
    settings = settings.model_copy(update={"research": research.model_copy(update={
        "study_registration": StudyRegistrationSettings(path=path, sha256=sha)})})
    feed = collector(catalog, criteria)
    def observed(symbol):
        return catalog, feed.collect(symbol)
    runner = module.CandidateStudyRunner(settings, clock=lambda: NOW,
        logger=Mock(), observation=observed)
    return settings, catalog, runner, kwargs, tmp_path


def test_four_independent_registrations_keep_original_execution_symbols_and_no_old_data(study):
    settings, catalog, runner, _, tmp_path = study
    assert settings.symbols == ("BTCUSDT", "ETHUSDT")
    assert set(runner.studies) == {"BTCUSDT", "ETHUSDT", "THEUSDT", "TSTUSDT"}
    assert not settings.research.sqlite_path.exists()
    assert all(s.protocol.symbols == (name,) and s.candidate_context_json is not None
               for name, s in runner.studies.items())
    assert (tmp_path / "registry/intent.json").exists()


def test_all_starts_precede_first_collection_and_round_restarts_never_repeat(study):
    settings, _, runner, _, _ = study
    original = runner.observation
    seen = []
    def observe(symbol):
        for study in runner.studies.values():
            coverage = read_attempt_coverage(study, (), evaluated_at=NOW)
            assert coverage.started == 1
        seen.append(symbol)
        return original(symbol)
    runner.observation = observe
    result = runner.run_once()
    assert len(result) == 4 and all(r.status is CycleStatus.SAVED for r in result), result
    assert seen == list(runner.index["symbols"])
    records = runner.repository.list_recent_opportunities()
    assert len(records) == 4
    for record in records:
        assert record.config_fingerprint == dict(runner.studies[record.symbol].protocol.config_fingerprints)[record.symbol]
        assert record.config_version.endswith(":next")
    assert runner.run_once() == ()
    another = module.CandidateStudyRunner(settings, clock=lambda: NOW, logger=Mock(), observation=original)
    assert another.run_once() == () and len(another.repository.list_recent_opportunities()) == 4


def test_planned_exit_only_uses_ceiling_seconds_not_legacy_fixed_horizons(study):
    settings, _, runner, _, _ = study
    result = runner.run_once()
    assert all(row.status is CycleStatus.SAVED for row in result), result
    with closing(sqlite3.connect(settings.research.sqlite_path)) as db:
        tasks = db.execute("SELECT opportunity_id,horizon_seconds,due_at FROM validation_tasks").fetchall()
    assert len(tasks) == 4
    for identifier, horizon, due in tasks:
        record = runner.repository.get_opportunity_by_id(identifier)
        assert horizon == module.planned_horizon(record) == 3630
        assert due == (record.created_at+timedelta(seconds=horizon)).isoformat(timespec="microseconds")
    assert 3600 not in {row[1] for row in tasks} and 86400 not in {row[1] for row in tasks}


@pytest.mark.parametrize("failure", ["collection", "identity", "rules", "quality", "persistence", "receipt", "scheduler", "interrupt"])
def test_failed_attempts_kept_and_other_symbols_continue(study, failure, monkeypatch):
    _, _, runner, _, _ = study
    original = runner.observation
    def observe(symbol):
        if symbol != "THEUSDT":
            return original(symbol)
        if failure == "collection":
            raise OSError("synthetic offline failure")
        if failure == "interrupt":
            raise KeyboardInterrupt
        catalog, result = original(symbol)
        if failure == "identity":
            result = replace(result, candidate=replace(result.candidate, base_asset="OTHER"))
        elif failure == "rules":
            result = replace(result, candidate=replace(result.candidate, spot_tick_size=D(".2")))
        elif failure == "quality":
            result = replace(result, assembly=replace(result.assembly, symbol="BTCUSDT"))
        return catalog, result
    runner.observation = observe
    if failure == "persistence":
        original_save = module.build_and_save_opportunity
        def save(repo, estimate, *args, **kwargs):
            if estimate.symbol == "THEUSDT":
                raise OSError("fake storage error")
            return original_save(repo, estimate, *args, **kwargs)
        monkeypatch.setattr(module, "build_and_save_opportunity", save)
    elif failure == "receipt":
        original_capture = runner.study_registry.capture
        def capture(study, record, settings):
            if record.symbol == "THEUSDT":
                from crypto_arbitrage_bot.storage.study_registry import StudyArchiveFailure
                raise StudyArchiveError(StudyArchiveFailure.ARCHIVE_IO_ERROR)
            return original_capture(study, record, settings)
        runner.study_registry.capture = capture
    elif failure == "scheduler":
        original_schedule = runner.scheduler.schedule
        def schedule(identifier):
            if runner.repository.get_opportunity_by_id(identifier).symbol == "THEUSDT":
                raise OSError("fake schedule error")
            original_schedule(identifier)
        runner.scheduler.schedule = schedule
    result = runner.run_once()
    assert len(result) == 4
    the = next(r for r in result if r.symbol == "THEUSDT")
    assert the.status is not CycleStatus.SAVED
    records = runner.repository.list_recent_opportunities()
    for name, saved in runner.studies.items():
        rows = tuple(r for r in records if r.symbol == name)
        coverage = read_attempt_coverage(saved, rows, evaluated_at=NOW)
        assert coverage.started == coverage.finished == 1
    if failure in ("receipt", "scheduler"):
        assert the.opportunity_id is not None
    if failure == "interrupt":
        assert result[-1].status is CycleStatus.INTERRUPTED


@pytest.mark.parametrize("field", ["fees", "economics", "profile", "database", "symbol", "index_bytes"])
def test_bound_policy_changes_fail_before_collection(study, field):
    settings, _, runner, _, _ = study
    research = settings.research
    if field == "fees":
        settings = settings.model_copy(update={"fees": settings.fees.model_copy(update={"spot_entry": D(".2")})})
    elif field == "economics":
        research = research.model_copy(update={"candidate_economics": research.candidate_economics.model_copy(update={"spot_budget": D("102")})})
    elif field == "profile":
        research = research.model_copy(update={"config_version": "changed"})
    elif field == "database":
        research = research.model_copy(update={"sqlite_path": research.sqlite_path.with_name("other.sqlite")})
    elif field == "symbol":
        settings = settings.model_copy(update={"symbols": ("BTCUSDT",)})
    else:
        research.study_registration.path.write_text("{}", encoding="utf-8")
    if field in ("economics", "profile", "database"):
        settings = settings.model_copy(update={"research": research})
    with pytest.raises((ValueError, StudyArchiveError)):
        module.CandidateStudyRunner(settings, observation=Mock(side_effect=AssertionError("before network")))


def test_complete_read_uses_existing_evaluator_for_all_symbols_without_qualifying(study):
    _, _, runner, _, _ = study
    runner.run_once()
    for saved in runner.studies.values():
        report = read_registered_study(saved.path, saved.sha256, evaluated_at=NOW)
        assert report.status is StudyReadStatus.READ
        assert len(report.opportunity_ids) == 1 and report.attempt_coverage.started == 1
        assert report.assessment.status.value == "INSUFFICIENT_EVIDENCE"


def test_standard_factory_selects_candidate_mode_without_constructing_execution_clients(study, monkeypatch):
    settings, _, _, _, _ = study
    import crypto_arbitrage_bot.runner as factory
    monkeypatch.setattr(factory, "SnapshotCollector", Mock(side_effect=AssertionError("different mode")))
    runner = create_runner(settings)
    assert isinstance(runner, module.CandidateStudyRunner)


def test_no_registration_can_be_reused_for_existing_research_data(study):
    settings, catalog, runner, kwargs, tmp_path = study
    runner.run_once()
    unbound = settings.model_copy(update={"research": settings.research.model_copy(update={"study_registration": None})})
    with pytest.raises(ValueError, match="NEW_FIXED"):
        module.prepare_candidate_index(unbound, tmp_path / "another", **kwargs)


def test_fractional_saved_time_ceil_and_lateness_never_adds_fractional_grace(study):
    _, _, runner, _, _ = study
    runner.run_once()
    original = runner.repository.list_recent_opportunities()[0]
    record = replace(original, id="fractional-save", created_at=NOW+timedelta(microseconds=400000))
    runner.repository.save_opportunity(record)
    assert module.planned_horizon(record) == 3630
    runner.scheduler.schedule(record.id)
    task, = runner.scheduler.tasks.list_for_opportunity(record.id)
    closed = NOW+timedelta(seconds=3630)
    assert task.due_at == closed+timedelta(microseconds=400000)
    at = closed+timedelta(seconds=300, microseconds=1)
    claimed = runner.scheduler.tasks.mark_running(task.id, now=at, lease_seconds=120)
    assert runner.scheduler._window_missed(claimed, at)
    failed = runner.scheduler.tasks.get(task.id)
    assert failed.status is TaskStatus.FAILED and failed.failure_code is TaskFailure.VALIDATION_WINDOW_MISSED


def test_missing_planned_task_reconciles_once_and_failed_tasks_are_not_retried(study):
    settings, _, runner, _, _ = study
    runner.run_once()
    record = runner.repository.list_recent_opportunities()[0]
    with closing(sqlite3.connect(settings.research.sqlite_path)) as db:
        db.execute("DELETE FROM validation_tasks WHERE opportunity_id=?", (record.id,))
        db.commit()
    runner.scheduler.process_due()
    task, = runner.scheduler.tasks.list_for_opportunity(record.id)
    runner.scheduler.clock = lambda: task.due_at+timedelta(seconds=301)
    runner.scheduler.process_due()
    before = runner.scheduler.tasks.get(task.id)
    assert before.status is TaskStatus.FAILED
    runner.scheduler.process_due()
    assert runner.scheduler.tasks.list_for_opportunity(record.id) == (before,)


@pytest.mark.parametrize("body", [b"{", b"{}", b'{"slot":true}', b'null'])
def test_corrupt_consumed_round_cannot_trigger_another_market_read(study, body):
    settings, _, runner, _, _ = study
    runner.run_once()
    marker = settings.research.study_registration.path.with_suffix(".rounds") / "000000.json"
    marker.write_bytes(body)
    runner.observation = Mock(side_effect=AssertionError("never recollect consumed slot"))
    with pytest.raises((ValueError, TypeError, KeyError)):
        runner.run_once()
    runner.observation.assert_not_called()


def test_start_receipt_failure_before_io_retains_partial_attempt_without_replay(study, monkeypatch):
    _, _, runner, _, _ = study
    original = module.start_attempt
    def start(study, *args, **kwargs):
        if study.protocol.symbols == ("THEUSDT",):
            raise OSError("synthetic disk failure")
        return original(study, *args, **kwargs)
    monkeypatch.setattr(module, "start_attempt", start)
    runner.observation = Mock(side_effect=AssertionError("all START receipts must persist first"))
    with pytest.raises(OSError):
        runner.run_once()
    assert runner.run_once() == ()
    runner.observation.assert_not_called()
    coverage = read_attempt_coverage(runner.studies["BTCUSDT"], (), evaluated_at=NOW)
    assert coverage.started == 1 and coverage.finished == 0 and not coverage.complete


def test_drain_period_never_adds_entry_attempts_or_market_reads(study):
    _, _, runner, _, _ = study
    end = next(iter(runner.studies.values())).protocol.study_end
    runner.clock = runner.scheduler.clock = lambda: end
    runner.observation = Mock(side_effect=AssertionError("entry window ended"))
    assert runner.run_once() == ()
    runner.observation.assert_not_called()
    assert all(read_attempt_coverage(s, (), evaluated_at=end).started == 0 for s in runner.studies.values())
    runner.clock = lambda: end+timedelta(seconds=7530)
    assert runner.run_once() == () and runner.stop_signal.is_set()


def test_reference_metadata_is_retained_before_entry_data_quality_rejection(study):
    settings, _, runner, _, _ = study
    result = runner.run_once()
    archives = list((settings.research.study_registration.path.parent / "entry-evidence").glob("*.json.gz"))
    assert len(archives) == 4
    by_cycle = {r.cycle_id: r for r in result}
    for path in archives:
        saved = json.loads(gzip.decompress(path.read_bytes()))
        identifier = path.name.removesuffix(".json.gz")
        assert saved["cycle_id"] == identifier
        assert saved["symbol"] == by_cycle[identifier].symbol
        assert saved["registration_sha256"] == runner.studies[saved["symbol"]].sha256


@pytest.mark.parametrize("change", ["duration", "separation", "days", "old_job_marker"])
def test_infeasible_or_already_bound_study_is_rejected_before_metadata_io(study, change, monkeypatch):
    settings, _, _, kwargs, tmp_path = study
    kwargs = dict(kwargs)
    if change == "duration":
        kwargs["study_end"] = NOW+timedelta(days=9)
    elif change == "separation":
        kwargs["min_entries"] = 1000
    elif change == "days":
        kwargs["study_end"] = NOW+timedelta(days=2)
    else:
        path = settings.research.sqlite_path
        path.with_suffix(path.suffix+".research-job.json").write_text("{}")
    kwargs.pop("catalog")
    monkeypatch.setattr(module, "collect_catalog", Mock(side_effect=AssertionError("metadata forbidden")))
    unbound = settings.model_copy(update={"research": settings.research.model_copy(update={"study_registration": None})})
    with pytest.raises(ValueError):
        module.prepare_candidate_index(unbound, tmp_path / "other-new", **kwargs)
    module.collect_catalog.assert_not_called()


@pytest.mark.parametrize("failure_stage", ["write", "directory"])
def test_raw_exit_capture_is_durable_before_return_and_archive_failure_blocks_qualification(study, monkeypatch, failure_stage):
    _, catalog, runner, _, _ = study
    runner.run_once()
    record = runner.repository.list_recent_opportunities()[0]
    task, = runner.scheduler.tasks.list_for_opportunity(record.id)
    sources = runner.scheduler.sources
    sources.task = task
    _, observation = runner.observation(record.symbol)
    def collect(self, symbol):
        self.catalogs.append(catalog)
        self.captures.append({"phase": "EXIT_MARKET", "symbol": symbol, "observation": observation})
        return observation.assembly
    monkeypatch.setattr(module.CandidateValidationSources, "collect", collect)
    assert sources.collect(record.symbol) == observation.assembly
    files = list((runner.studies[record.symbol].path.parent / "exit-evidence").glob("*.json.gz"))
    assert len(files) == 1
    saved = json.loads(gzip.decompress(files[0].read_bytes()))
    assert saved["opportunity_id"] == record.id and saved["task_id"] == task.id
    assert not sources.catalogs and not sources.captures
    if failure_stage == "write":
        monkeypatch.setattr(module, "write_public_archive", Mock(side_effect=OSError("offline disk failure")))
    else:
        original_mkdir = Path.mkdir
        def mkdir(path, *args, **kwargs):
            if path.name == "exit-evidence":
                raise OSError("offline directory failure")
            return original_mkdir(path, *args, **kwargs)
        monkeypatch.setattr(Path, "mkdir", mkdir)
    with pytest.raises(OSError):
        sources.collect(record.symbol)
    assert sources.archive_failed
    runner.scheduler.clock = lambda: task.due_at
    claimed = runner.scheduler.tasks.mark_running(task.id, now=task.due_at, lease_seconds=120)
    runner.scheduler._execute(claimed)
    failed = runner.scheduler.tasks.get(task.id)
    assert failed.failure_code is TaskFailure.PERSISTENCE_FAILED
    assert runner.scheduler.validations.list_validations_for_opportunity(record.id) == ()


@pytest.mark.parametrize("mode", ["valid", "short_duration", "short_budget", "changed_reference"])
def test_native_job_pins_candidate_reference_and_requires_exit_coverage(study, monkeypatch, mode):
    from crypto_arbitrage_bot import research_control as control
    from crypto_arbitrage_bot.storage.study_registry import full_research_policy
    settings, _, _, _, tmp_path = study
    live = [settings]
    config = tmp_path / "new-policy.toml"
    config.write_text("# synthetic config; patched loader, no IO")
    monkeypatch.setattr(control, "load_settings", lambda path: live[0])
    duration = 3*86400+7530
    cycles = duration//30+1
    if mode == "short_duration":
        duration -= 1
    elif mode == "short_budget":
        cycles = 1
    directory = tmp_path / "native-job"
    if mode in ("short_duration", "short_budget"):
        with pytest.raises(control.ControlError, match="CANDIDATE_JOB"):
            control.prepare_job(config, directory, duration_seconds=duration, max_cycles=cycles, clock=lambda: NOW)
        assert not (directory / "plan.json").exists()
        return
    plan = control.prepare_job(config, directory, duration_seconds=duration, max_cycles=cycles, clock=lambda: NOW)
    assert plan["study_reference"]["sha256"] == settings.research.study_registration.sha256
    assert module.exact_json(control.load_job(directory)[0]) == module.exact_json(plan)
    if mode == "changed_reference":
        research = settings.research.model_copy(update={"study_registration":
            settings.research.study_registration.model_copy(update={"sha256": "1"*64})})
        live[0] = settings.model_copy(update={"research": research})
        assert full_research_policy(live[0]) == full_research_policy(settings)
        with pytest.raises(control.ControlError, match="PINNED_STUDY_REFERENCE_CHANGED"):
            control.load_job(directory)


def test_failed_reference_catalog_is_retained_without_registering_a_smaller_cohort(study):
    settings, catalog, _, kwargs, tmp_path = study
    kwargs = dict(kwargs)
    kwargs["catalog"] = replace(catalog, status=module.CatalogStatus.INVALID)
    unbound = settings.model_copy(update={"research": settings.research.model_copy(update={"study_registration": None})})
    directory = tmp_path / "failed-prepare"
    with pytest.raises(ValueError, match="ALL_PREDECLARED"):
        module.prepare_candidate_index(unbound, directory, **kwargs)
    assert (directory / "intent.json").exists()
    raw = json.loads((directory / "reference-catalog.json").read_text())
    assert raw["status"] == "INVALID" and len(raw["sources"]) == 2
    assert not (directory / "index.json").exists()
    assert not settings.research.sqlite_path.exists()


def test_public_archive_lossless_and_existing_target_never_overwritten(tmp_path):
    data = {"raw": b"\x00raw official bytes\xff", "rate": D("0.00001"), "at": NOW}
    path = tmp_path/"capture.json"
    expected = module.exact_json(data).encode()
    import hashlib
    assert module.write_public_archive(path, data) == hashlib.sha256(expected).hexdigest()
    saved = path.with_suffix(".json.gz")
    assert gzip.decompress(saved.read_bytes()) == expected
    before = saved.read_bytes()
    with pytest.raises(FileExistsError):
        module.write_public_archive(path, {"replacement": True})
    assert saved.read_bytes() == before


@pytest.mark.parametrize("reason", ["size", "space", "fsync"])
def test_public_archive_limits_fail_closed_without_replacing_prior_evidence(tmp_path, monkeypatch, reason):
    path = tmp_path/"new.json"
    if reason == "size":
        monkeypatch.setattr(module, "MAX_RAW_ARCHIVE_BYTES", 1)
    elif reason == "space":
        monkeypatch.setattr(module.shutil, "disk_usage", Mock(return_value=Mock(free=0)))
    else:
        monkeypatch.setattr(module.os, "fsync", Mock(side_effect=OSError("offline disk error")))
    with pytest.raises((ValueError, OSError)):
        module.write_public_archive(path, {"complete": True})
    saved = path.with_suffix(".json.gz")
    assert saved.exists() == (reason == "fsync")
