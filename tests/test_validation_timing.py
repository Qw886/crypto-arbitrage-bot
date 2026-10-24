"""Observed exit time, restart deadlines and funding timestamps stay distinct."""

import json
from datetime import timedelta
from unittest.mock import Mock

import pytest
from test_decision import research_estimate, thresholds
from test_forward_validation import OPENED, event, later_snapshot, shift_fixture, validate
from test_funding_history import client, row, transport
from test_validation_scheduler import setup_scheduler

from crypto_arbitrage_bot import validation_scheduler as module
from crypto_arbitrage_bot.arbitrage.decision import decide
from crypto_arbitrage_bot.config.research import ValidationSchedulerSettings
from crypto_arbitrage_bot.exchanges.funding_history import FundingWindow
from crypto_arbitrage_bot.models.validation_task import TaskFailure as F
from crypto_arbitrage_bot.models.validation_task import TaskStatus as S
from crypto_arbitrage_bot.storage.opportunity_records import build_opportunity


@pytest.mark.parametrize("horizon", [3600, 28800, 86400])
@pytest.mark.parametrize("delay", [0, 30, 300, 301, 15 * 3600])
def test_horizon_deadline_uses_actual_observation_time(tmp_path, horizon, delay):
    due = OPENED + timedelta(seconds=horizon)
    scheduler, clock, (record,) = setup_scheduler(
        tmp_path, horizons=(horizon,), at=due - timedelta(microseconds=1), max_lateness_seconds=300,
    )
    scheduler.process_due()
    scheduler.collector.collect.assert_not_called()
    clock[0] = due + timedelta(seconds=delay)
    scheduler.process_due()
    task, = scheduler.tasks.list_for_opportunity(record.id)
    saved = scheduler.validations.list_validations_for_opportunity(record.id)
    if delay <= 300:
        assert task.status is S.COMPLETED
        assert saved[0].validation_time == clock[0]
        assert saved[0].holding_duration == timedelta(seconds=horizon + delay)
        expected = validate(record, later_snapshot(at=clock[0]), events=(event(),), at=clock[0])
        assert saved[0].validated_simulated_net_profit == expected.validated_simulated_net_profit
    else:
        assert task.failure_code is F.VALIDATION_WINDOW_MISSED and not saved
        assert json.loads(task.failure_reason)["horizon_seconds"] == horizon
        scheduler.collector.collect.assert_not_called()
        scheduler.funding.fetch_window.assert_not_called()
    calls = scheduler.collector.collect.call_count
    scheduler.process_due()
    assert scheduler.collector.collect.call_count == calls
    assert scheduler.tasks.get(task.id).attempts == 1


def test_restart_backlog_cannot_starve_current_due_task(tmp_path):
    now = OPENED + timedelta(hours=23)
    scheduler, _, _ = setup_scheduler(
        tmp_path, horizons=(3600, 28800, 86400), at=now,
        batch_size=1, max_lateness_seconds=300,
    )
    current_entry = shift_fixture(research_estimate("ETHUSDT"), timedelta(hours=22))
    current = build_opportunity(current_entry, decide(current_entry, thresholds()),
                                opportunity_id="current", created_at=now-timedelta(hours=1))
    scheduler.funding.fetch_window.side_effect = lambda s, o, c: FundingWindow(
        s, o, c, (event(symbol=s, at=o+timedelta(hours=1)),))
    scheduler.opportunities.save_opportunity(current)
    scheduler.schedule(current.id)
    restarted = module.ValidationScheduler(
        scheduler.settings, scheduler.opportunities, scheduler.validations, scheduler.tasks,
        scheduler.collector, scheduler.funding, clock=lambda: now, logger=Mock(),
    )
    restarted.process_due()
    assert scheduler.tasks.list_for_opportunity("current")[0].status is S.COMPLETED
    assert scheduler.validations.list_validations_for_opportunity("current")[0].holding_duration == timedelta(hours=1)
    scheduler.collector.collect.assert_called_once_with("ETHUSDT")
    # Old tasks are retired in bounded batches, without public requests.
    restarted.process_due()
    restarted.process_due()
    old = scheduler.tasks.list_for_opportunity("BTCUSDT")
    assert [t.failure_code for t in old[:2]] == [F.VALIDATION_WINDOW_MISSED] * 2
    assert old[2].status is S.PENDING
    assert scheduler.collector.collect.call_count == 1


def test_slow_collection_crossing_deadline_never_validates(tmp_path):
    scheduler, clock, (record,) = setup_scheduler(
        tmp_path, horizons=(3600,), at=OPENED+timedelta(hours=1), max_lateness_seconds=30,
    )
    def delayed(symbol):
        clock[0] += timedelta(seconds=31)
        return later_snapshot(symbol, at=clock[0])
    scheduler.collector.collect.side_effect = delayed
    scheduler.process_due()
    assert scheduler.tasks.list_for_opportunity(record.id)[0].failure_code is F.VALIDATION_WINDOW_MISSED
    scheduler.funding.fetch_window.assert_not_called()
    assert not scheduler.validations.list_validations_for_opportunity(record.id)


def test_saved_validation_recovers_completion_even_after_deadline(tmp_path, monkeypatch):
    scheduler, clock, (record,) = setup_scheduler(
        tmp_path, horizons=(3600,), at=OPENED+timedelta(hours=1), max_lateness_seconds=30,
    )
    finish = scheduler.tasks.finish
    monkeypatch.setattr(scheduler.tasks, "finish", Mock(side_effect=OSError("status write interrupted")))
    scheduler.process_due()
    clock[0] += timedelta(hours=22)
    monkeypatch.setattr(scheduler.tasks, "finish", finish)
    scheduler.process_due()
    assert scheduler.tasks.list_for_opportunity(record.id)[0].status is S.COMPLETED
    assert len(scheduler.validations.list_validations_for_opportunity(record.id)) == 1
    scheduler.collector.collect.assert_called_once()


@pytest.mark.parametrize("horizon", [3600, 28800, 86400])
@pytest.mark.parametrize("offset_ms", [-1, 1, 999, 1000, 1001])
def test_funding_scheduled_time_matches_actual_history_without_rewriting_it(tmp_path, monkeypatch, horizon, offset_ms):
    at = OPENED + timedelta(seconds=horizon + 5)
    scheduler, _, (record,) = setup_scheduler(tmp_path, horizons=(horizon,), at=at, max_lateness_seconds=300)
    actual = event().settled_at + timedelta(milliseconds=offset_ms)
    transport(monkeypatch, [[row(at=actual)]])
    scheduler.funding = client()
    scheduler.process_due()
    task, = scheduler.tasks.list_for_opportunity(record.id)
    saved = scheduler.validations.list_validations_for_opportunity(record.id)
    if abs(offset_ms) <= 1000:
        assert task.status is S.COMPLETED
        evidence = json.loads(saved[0].details_json)
        assert evidence["funding_events"][0]["settled_at"] == actual.isoformat(timespec="microseconds")
    else:
        assert task.failure_code is F.FUNDING_UNAVAILABLE and not saved
        assert "event_times" in task.failure_reason and "expected_at" in task.failure_reason


@pytest.mark.parametrize("kind", ["empty", "ambiguous", "outside_exit", "outside_entry"])
def test_tolerance_never_accepts_missing_ambiguous_or_out_of_window_funding(tmp_path, kind):
    at = OPENED + timedelta(hours=1)
    scheduler, _, (record,) = setup_scheduler(tmp_path, horizons=(3600,), at=at)
    events = {
        "empty": (),
        "ambiguous": (event(), event(at=at-timedelta(milliseconds=1), event_id="other")),
        "outside_exit": (event(at=at+timedelta(milliseconds=1)),),
        "outside_entry": (event(), event(at=OPENED, event_id="entry")),
    }[kind]
    scheduler.funding.fetch_window.side_effect = lambda s, o, c: FundingWindow(s, o, c, events)
    scheduler.process_due()
    assert scheduler.tasks.list_for_opportunity(record.id)[0].failure_code is F.FUNDING_UNAVAILABLE
    assert not scheduler.validations.list_validations_for_opportunity(record.id)


@pytest.mark.parametrize("field,value", [
    ("max_lateness_seconds", -1), ("max_lateness_seconds", True),
    ("funding_time_tolerance_ms", -1), ("funding_time_tolerance_ms", 5001),
])
def test_timing_configuration_is_explicit_and_bounded(field, value):
    with pytest.raises(ValueError):
        ValidationSchedulerSettings(validation_horizons_seconds=(3600,), **{field: value})
