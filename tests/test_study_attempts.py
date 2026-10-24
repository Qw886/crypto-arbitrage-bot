"""Offline attempt coverage; synthetic earnings never qualify a real strategy."""

import hashlib
import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest
from test_forward_validation import shift_fixture
from test_offline_estimate import OPENED
from test_registered_study import Cohort, registered_runner, scheduler_for

from crypto_arbitrage_bot.models.strategy_study import StudyReason, StudyStatus
from crypto_arbitrage_bot.runner import CycleStage, CycleStatus
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.study_attempts import (
    finish_attempt,
    read_attempt_coverage,
    start_attempt,
)
from crypto_arbitrage_bot.storage.study_registry import StudyArchiveError, StudyArchiveFailure


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("offline attempt tests"))
    monkeypatch.setattr("socket.create_connection", forbidden)
    return Cohort(tmp_path)


def coverage(cohort, records=None, at=None):
    return read_attempt_coverage(cohort.study,
        tuple(cohort.repository.list_recent_opportunities()) if records is None else records,
        evaluated_at=at or OPENED+timedelta(days=4))


def events(cohort):
    return sorted(cohort.study.path.with_suffix(".attempts").glob("*.json"))


def failed_attempt(cohort, status="FAILED", identifier="failure", at=OPENED):
    start_attempt(cohort.study, identifier, "BTCUSDT", at, recorded_at=at)
    finish_attempt(cohort.study, identifier, "BTCUSDT", ended_at=at, status=status,
                   failure_stage="COLLECTION", reason_codes=("COLLECTION_FAILED",), record=None, recorded_at=at)


def test_attempt_starts_before_any_public_io(cohort):
    runner, collector = registered_runner(cohort)
    original = collector.collect.side_effect
    def collect(symbol):
        files = events(cohort)
        assert len(files) == 1 and json.loads(files[0].read_text())["kind"] == "START"
        return original(symbol)
    collector.collect.side_effect = collect
    result, = runner.run_once()
    audit = coverage(cohort, at=OPENED)
    assert result.status is CycleStatus.SAVED and audit.complete
    assert (audit.started, audit.finished, audit.failed) == (1, 1, 0)
    assert json.loads(events(cohort)[1].read_text())["opportunity_id"] == result.opportunity_id


@pytest.mark.parametrize("status", ["FAILED", "INTERRUPTED"])
def test_finished_failed_attempt_preserved_without_inventing_record(cohort, status):
    failed_attempt(cohort, status)
    audit = coverage(cohort, at=OPENED)
    assert audit.complete and audit.started == audit.finished == 1
    assert audit.failed == (status == "FAILED") and audit.interrupted == (status == "INTERRUPTED")
    terminal = json.loads(events(cohort)[1].read_text())
    assert terminal["record_sha256"] is terminal["opportunity_id"] is None
    assert "net_profit" not in terminal


def test_crash_leaves_unfinished_attempt_and_blocks_otherwise_good_cohort(cohort):
    cohort.add()
    cohort.add(1)
    start_attempt(cohort.study, "power-loss", "BTCUSDT", OPENED+timedelta(days=2),
                  recorded_at=OPENED+timedelta(days=2))
    result = cohort.read()
    assert result.assessment.status is StudyStatus.INSUFFICIENT_EVIDENCE
    assert StudyReason.INCOMPLETE_ATTEMPT in result.assessment.reason_codes
    assert result.attempt_coverage.started == 3 and result.attempt_coverage.finished == 2


@pytest.mark.parametrize("index", [0, 1, 3])
def test_delete_first_middle_or_final_event_is_detected(cohort, index):
    cohort.add()
    cohort.add(1)
    events(cohort)[index].unlink()
    assert StudyReason.CORRUPT_ATTEMPT_LOG in tuple(i.code for i in coverage(cohort).issues)
    assert cohort.read().assessment.status is StudyStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize("damage", ["changed", "partial", "extra", "head_deleted", "wrong_identity"])
def test_archive_corruption_fails_closed(cohort, damage):
    failed_attempt(cohort)
    path = events(cohort)[0]
    if damage == "head_deleted":
        cohort.study.path.with_suffix(".attempt-head.json").unlink()
    elif damage == "extra":
        path.with_name("extra.json").write_bytes(path.read_bytes())
    elif damage == "partial":
        path.write_text('{"kind":', encoding="utf-8")
    else:
        value = json.loads(path.read_text())
        value["symbol" if damage == "changed" else "registration_sha256"] = "ETHUSDT" if damage == "changed" else "0"*64
        path.write_text(exact_json(value), encoding="utf-8")
    assert not coverage(cohort, at=OPENED).complete
    assert coverage(cohort, at=OPENED).issues[0].code is StudyReason.CORRUPT_ATTEMPT_LOG


def test_partial_terminal_write_not_healed_or_overwritten(cohort, monkeypatch):
    from crypto_arbitrage_bot.storage import study_attempts as module
    start_attempt(cohort.study, "partial", "BTCUSDT", OPENED, recorded_at=OPENED)
    replace_file = Mock(side_effect=OSError("synthetic head replace interruption"))
    monkeypatch.setattr(module.Path, "replace", replace_file)
    with pytest.raises(StudyArchiveError):
        finish_attempt(cohort.study, "partial", "BTCUSDT", ended_at=OPENED, status="FAILED",
                       failure_stage="COLLECTION", reason_codes=(), record=None, recorded_at=OPENED)
    retained = [p.read_bytes() for p in events(cohort)]
    with pytest.raises(StudyArchiveError):
        start_attempt(cohort.study, "next", "BTCUSDT", OPENED, recorded_at=OPENED)
    assert [p.read_bytes() for p in events(cohort)] == retained
    assert not coverage(cohort, at=OPENED).complete


@pytest.mark.parametrize("stage", ["collection", "quality", "persistence", "logging", "interrupt"])
def test_runner_failure_stages_record_no_fabricated_success(cohort, monkeypatch, stage):
    runner, collector = registered_runner(cohort)
    if stage == "collection":
        collector.collect.side_effect = RuntimeError("SECRET_RAW_BODY"*10000)
    elif stage == "quality":
        cohort.clock += timedelta(seconds=11)
    elif stage == "persistence":
        monkeypatch.setattr(runner.repository, "save_opportunity", Mock(side_effect=OSError("full")))
    elif stage == "logging":
        runner.logger.info.side_effect = OSError("logging broken")
    else:
        collector.collect.side_effect = KeyboardInterrupt()
    result, = runner.run_once()
    terminal = json.loads(events(cohort)[1].read_text())
    assert terminal["status"] == result.status.value != "SAVED"
    assert len(events(cohort)[1].read_bytes()) < 1500
    assert "SECRET_RAW_BODY" not in events(cohort)[1].read_text()
    if stage in ("collection", "quality", "interrupt"):
        assert terminal["opportunity_id"] is None and result.decision is None
    if stage == "quality":
        assert terminal["failure_stage"] == "DATA_QUALITY" and terminal["reason_codes"]
    if stage == "logging":
        assert terminal["opportunity_id"] is not None and result.decision == "PASS"
    assert coverage(cohort, at=cohort.clock).complete


def test_unrecordable_start_blocks_io(cohort, monkeypatch):
    runner, collector = registered_runner(cohort)
    cohort.study.path.with_suffix(".attempt-head.json").unlink()
    result, = runner.run_once()
    assert result.failure_stage is CycleStage.STUDY_REGISTRATION
    collector.collect.assert_not_called()
    assert not coverage(cohort, at=OPENED).complete


def test_unrecordable_finish_preserves_saved_pnl_decision_and_scheduling(cohort, monkeypatch):
    scheduler = scheduler_for(cohort)
    runner, _ = registered_runner(cohort, scheduler=scheduler)
    monkeypatch.setattr("crypto_arbitrage_bot.runner.finish_attempt", Mock(side_effect=OSError("private body")))
    result, = runner.run_once()
    record = cohort.repository.get_opportunity_by_id(result.opportunity_id)
    assert result.status is CycleStatus.FAILED and result.decision == record.decision == "PASS"
    assert record.net_profit == Decimal("9.0144")
    assert result.failure_reason == "ATTEMPT_CAPTURE_FAILED:OSError"
    scheduler.schedule.assert_called_once_with(record.id)
    assert StudyReason.INCOMPLETE_ATTEMPT in tuple(i.code for i in coverage(cohort).issues)


def test_btc_failure_and_eth_attempt_are_both_retained(tmp_path):
    cohort = Cohort(tmp_path, ("BTCUSDT", "ETHUSDT"))
    runner, collector = registered_runner(cohort)
    original = collector.collect.side_effect
    collector.collect.side_effect = lambda s: (_ for _ in ()).throw(OSError("down")) if s == "BTCUSDT" else original(s)
    first, second = runner.run_once()
    assert first.status is CycleStatus.FAILED and second.status is CycleStatus.SAVED
    audit = coverage(cohort, at=OPENED)
    assert audit.complete and (audit.started, audit.finished, audit.failed) == (2, 2, 1)


def test_failed_attempt_does_not_make_positive_samples_disappear(cohort):
    cohort.add()
    failed_attempt(cohort, at=OPENED+timedelta(hours=3))
    cohort.add(1)
    result = cohort.read()
    assert result.attempt_coverage.failed == 1 and result.attempt_coverage.complete
    assert len(result.opportunity_ids) == 2 and result.assessment.symbols[0].profit.mean == Decimal("6.9688")
    assert result.assessment.status is StudyStatus.READY_FOR_EXECUTION_REVIEW


@pytest.mark.parametrize("change", ["record", "missing", "extra"])
def test_attempt_record_link_uses_exact_original_not_symbol_only(cohort, change):
    first = cohort.add()
    records = (first,)
    if change == "record":
        records = (replace(first, net_profit=first.net_profit+1),)
    elif change == "missing":
        records = ()
    else:
        records = (first, replace(first, id="uncaptured-record"))
    result = coverage(cohort, records)
    assert not result.complete
    code = StudyReason.MISSING_ATTEMPT_COVERAGE if change == "extra" else StudyReason.ATTEMPT_RECORD_MISMATCH
    assert code in tuple(i.code for i in result.issues)


def test_sampling_gap_uses_predeclared_bound_including_window_edges(cohort):
    cohort.clock = OPENED-timedelta(hours=1)
    cohort.study = cohort.registry.register(replace(cohort.protocol, id="short-gap"), cohort.settings,
        max_capture_delay=timedelta(seconds=60), max_sampling_gap=timedelta(hours=12))
    cohort.add()
    cohort.add(1)
    result = coverage(cohort)
    assert not result.complete and result.maximum_gaps == (("BTCUSDT", timedelta(days=2)),)
    assert StudyReason.SAMPLING_GAP_TOO_LARGE in tuple(i.code for i in result.issues)


@pytest.mark.parametrize("gap", [timedelta(0), timedelta(seconds=-1), timedelta(days=4), 30])
def test_invalid_sampling_gap_rejected_before_registration(cohort, gap):
    with pytest.raises(StudyArchiveError) as error:
        cohort.registry.register(replace(cohort.protocol, id="invalid-gap"), cohort.settings,
            max_capture_delay=timedelta(seconds=60), max_sampling_gap=gap)
    assert error.value.code is StudyArchiveFailure.INVALID_REGISTRATION


def test_legacy_v1_is_readable_but_cannot_qualify(cohort):
    cohort.study = cohort.registry.register(replace(cohort.protocol, id="legacy"), cohort.settings,
                                           max_capture_delay=timedelta(seconds=60))
    runner, _ = registered_runner(cohort)
    result, = runner.run_once()
    assert result.status is CycleStatus.SAVED and not events(cohort)
    review = cohort.read()
    assert review.status.value == "READ" and review.opportunity_ids == (result.opportunity_id,)
    assert StudyReason.MISSING_ATTEMPT_COVERAGE in review.assessment.reason_codes
    assert review.assessment.status is StudyStatus.INSUFFICIENT_EVIDENCE


def test_duplicate_start_finish_and_future_evidence_rejected(cohort):
    failed_attempt(cohort)
    start_attempt(cohort.study, "failure", "BTCUSDT", OPENED, recorded_at=OPENED)
    assert not coverage(cohort, at=OPENED).complete
    assert not coverage(cohort, at=OPENED-timedelta(seconds=1)).complete


def test_deleting_terminal_head_cannot_be_repaired_by_idempotent_registration(cohort):
    failed_attempt(cohort)
    cohort.study.path.with_suffix(".attempt-head.json").unlink()
    same = cohort.registry.register(cohort.protocol, cohort.settings, max_capture_delay=timedelta(seconds=60),
                                    max_sampling_gap=timedelta(days=2))
    assert same == cohort.study and not coverage(cohort).complete


def test_explicit_gap_change_cannot_reuse_study_identity(cohort):
    with pytest.raises(StudyArchiveError) as error:
        cohort.registry.register(cohort.protocol, cohort.settings, max_capture_delay=timedelta(seconds=60),
                                 max_sampling_gap=timedelta(days=1))
    assert error.value.code is StudyArchiveFailure.REGISTRATION_CONFLICT


def test_restart_appends_original_head_not_fresh_history(cohort):
    failed_attempt(cohort)
    before = [p.read_bytes() for p in events(cohort)]
    runner, _ = registered_runner(cohort)
    runner.run_once()
    assert [p.read_bytes() for p in events(cohort)[:2]] == before
    audit = coverage(cohort, at=OPENED)
    assert audit.complete and audit.started == audit.finished == 2


def test_start_timestamps_and_finished_record_are_utc(cohort):
    cohort.add()
    value = json.loads(events(cohort)[0].read_text())
    assert value["started_at"].endswith("+00:00") and value["recorded_at"].endswith("+00:00")
    head = json.loads(cohort.study.path.with_suffix(".attempt-head.json").read_text())
    assert head["last_sha256"] == hashlib.sha256(events(cohort)[-1].read_bytes()).hexdigest()


def test_saved_reject_is_captured_without_becoming_collection_failure(cohort):
    runner, collector = registered_runner(cohort)
    original = collector.collect.side_effect
    offset = timedelta(hours=12)
    cohort.clock = OPENED+offset
    collector.collect.side_effect = lambda s: shift_fixture(original(s), offset)
    result, = runner.run_once()
    assert result.status is CycleStatus.SAVED and result.decision == "REJECT"
    terminal = json.loads(events(cohort)[1].read_text())
    assert terminal["status"] == "SAVED" and terminal["failure_stage"] is None
    record = cohort.repository.get_opportunity_by_id(result.opportunity_id)
    assert record.net_profit < 0 and record.estimated_funding == 0
    assert coverage(cohort, at=cohort.clock).complete
