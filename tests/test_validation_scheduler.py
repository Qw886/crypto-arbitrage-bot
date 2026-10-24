"""Scheduler integration uses actual existing engines/SQLite with offline quotes."""

import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from threading import Event
from unittest.mock import Mock

import pytest
from test_forward_validation import AT, OPENED, event, later_snapshot, research_record, validate
from test_runner import make_runner, settings_for

from crypto_arbitrage_bot import runner as runner_module
from crypto_arbitrage_bot import validation_scheduler as module
from crypto_arbitrage_bot.config.research import ValidationSchedulerSettings
from crypto_arbitrage_bot.exchanges.funding_history import FundingFailure, FundingWindow
from crypto_arbitrage_bot.market_data.snapshot import assemble_snapshot
from crypto_arbitrage_bot.models.validation_task import TaskFailure as F
from crypto_arbitrage_bot.models.validation_task import TaskStatus as S
from crypto_arbitrage_bot.storage.forward_validation_repository import (
    SQLiteForwardValidationRepository,
    ValidationPersistenceStatus,
    ValidationSaveResult,
)
from crypto_arbitrage_bot.storage.sqlite_repository import (
    RepositoryError,
    SQLiteConfig,
    SQLiteOpportunityRepository,
)
from crypto_arbitrage_bot.storage.validation_tasks import SQLiteValidationTaskRepository


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("offline only")))


def setup_scheduler(tmp_path, *, horizons=(3600, 28800, 86400), symbols=("BTCUSDT",), at=AT, **config):
    settings = ValidationSchedulerSettings(validation_horizons_seconds=horizons, **config)
    db = SQLiteConfig(tmp_path / "scheduler.sqlite", busy_timeout_ms=0)
    clock = [at]
    collector, funding = Mock(), Mock()
    collector.collect.side_effect = lambda symbol: later_snapshot(symbol, at=clock[0])
    funding.fetch_window.side_effect = lambda symbol, opened, closed: FundingWindow(
        symbol, opened, closed, (event(symbol=symbol),) if closed >= event().settled_at else ())
    scheduler = module.ValidationScheduler(settings, SQLiteOpportunityRepository(db),
                                          SQLiteForwardValidationRepository(db), SQLiteValidationTaskRepository(db),
                                          collector, funding, clock=lambda: clock[0], logger=Mock())
    scheduler.initialize()
    records = []
    for symbol in symbols:
        record = replace(research_record(symbol), id=symbol)
        scheduler.opportunities.save_opportunity(record)
        scheduler.schedule(record.id)
        records.append(record)
    return scheduler, clock, records


def test_create_horizons_idempotent_and_schema_does_not_destroy_records(tmp_path):
    scheduler, _, records = setup_scheduler(tmp_path)
    scheduler.initialize()
    scheduler.schedule(records[0].id)
    tasks = scheduler.tasks.list_for_opportunity(records[0].id)
    assert [task.horizon_seconds for task in tasks] == [3600, 28800, 86400]
    assert all(task.due_at == OPENED + timedelta(seconds=task.horizon_seconds) for task in tasks)
    assert all(task.status is S.PENDING and task.attempts == 0 for task in tasks)
    assert scheduler.opportunities.get_opportunity_by_id(records[0].id) == records[0]


def test_not_due_never_fetches(tmp_path):
    scheduler, _, _ = setup_scheduler(tmp_path, at=OPENED + timedelta(minutes=59))
    scheduler.process_due()
    scheduler.collector.collect.assert_not_called()
    scheduler.funding.fetch_window.assert_not_called()


@pytest.mark.parametrize("horizon", [3600, 28800, 86400])
@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_each_horizon_due_at_equality_saves_exact_existing_validator_result(tmp_path, horizon, symbol):
    at = OPENED + timedelta(seconds=horizon)
    scheduler, _, (record,) = setup_scheduler(tmp_path, horizons=(horizon,), symbols=(symbol,), at=at)
    scheduler.process_due()
    task, = scheduler.tasks.list_for_opportunity(record.id)
    assert task.status is S.COMPLETED and task.attempts == 1
    saved, = scheduler.validations.list_validations_for_opportunity(record.id)
    expected = validate(record, later_snapshot(symbol, at=at), events=(event(symbol=symbol),), at=at)
    assert saved.validated_simulated_net_profit == expected.validated_simulated_net_profit
    assert saved.validated_simulated_net_return == expected.validated_simulated_net_return
    assert saved.validation_time == at and isinstance(saved.validated_simulated_net_profit, Decimal)
    assert scheduler.opportunities.get_opportunity_by_id(record.id) == record
    scheduler.process_due()
    scheduler.collector.collect.assert_called_once_with(symbol)
    assert len(scheduler.validations.list_validations_for_opportunity(record.id)) == 1


@pytest.mark.parametrize("rate", [".01", "-.01", "0"])
def test_actual_funding_signed_events_only_once(tmp_path, rate):
    scheduler, _, (record,) = setup_scheduler(tmp_path, horizons=(3600,))
    events = (event(rate),)
    scheduler.funding.fetch_window.side_effect = lambda s, o, c: FundingWindow(s, o, c, events)
    scheduler.process_due()
    saved, = scheduler.validations.list_validations_for_opportunity(record.id)
    assert saved.funding_total == Decimal("220") * Decimal(rate)
    assert saved.validated_simulated_net_profit == validate(record, events=events).validated_simulated_net_profit


def test_no_settlement_before_known_due_time_is_zero_not_forecast(tmp_path):
    scheduler, _, (record,) = setup_scheduler(tmp_path, horizons=(60,), at=OPENED + timedelta(minutes=1))
    scheduler.process_due()
    saved, = scheduler.validations.list_validations_for_opportunity(record.id)
    assert saved.funding_total == Decimal(0)


@pytest.mark.parametrize("bad_time", [OPENED, OPENED - timedelta(seconds=1), AT + timedelta(microseconds=1)])
def test_outside_window_fails_without_any_validation_profit(tmp_path, bad_time):
    scheduler, _, (record,) = setup_scheduler(tmp_path, horizons=(3600,))
    scheduler.funding.fetch_window.side_effect = lambda s, o, c: FundingWindow(s, o, c, (event(), event(at=bad_time)))
    scheduler.process_due()
    task, = scheduler.tasks.list_for_opportunity(record.id)
    assert task.status is S.FAILED and task.failure_code is F.FUNDING_UNAVAILABLE
    assert scheduler.validations.list_validations_for_opportunity(record.id) == ()


@pytest.mark.parametrize("window_kind", ["missing", "network", "wrong_symbol", "wrong_window"])
def test_funding_absent_or_untrusted_fails_closed(tmp_path, window_kind):
    scheduler, _, (record,) = setup_scheduler(tmp_path, horizons=(3600,))
    window = FundingWindow(record.symbol, OPENED, AT)
    if window_kind == "network":
        window = replace(window, failure=FundingFailure.TIMEOUT, message="timeout")
    elif window_kind == "wrong_symbol":
        window = replace(window, symbol="ETHUSDT")
    elif window_kind == "wrong_window":
        window = replace(window, opened_at=OPENED + timedelta(seconds=1))
    scheduler.funding.fetch_window.side_effect = None
    scheduler.funding.fetch_window.return_value = window
    scheduler.process_due()
    task, = scheduler.tasks.list_for_opportunity(record.id)
    assert task.failure_code is F.FUNDING_UNAVAILABLE
    assert not scheduler.validations.list_validations_for_opportunity(record.id)


@pytest.mark.parametrize("source", ["spot_depth", "perp_depth"])
def test_api_failure_does_not_stop_eth_or_create_fake_result(tmp_path, source):
    scheduler, _, records = setup_scheduler(tmp_path, horizons=(3600,), symbols=("BTCUSDT", "ETHUSDT"))
    original = later_snapshot()
    incomplete = assemble_snapshot("BTCUSDT", replace(original.sources, **{source: None}), assembled_at=AT)
    scheduler.collector.collect.side_effect = lambda s: incomplete if s == "BTCUSDT" else later_snapshot(s)
    scheduler.process_due()
    btc, = scheduler.tasks.list_for_opportunity("BTCUSDT")
    eth, = scheduler.tasks.list_for_opportunity("ETHUSDT")
    assert btc.status is S.FAILED and btc.failure_code is F.COLLECTION_FAILED
    assert eth.status is S.COMPLETED
    assert scheduler.validations.list_validations_for_opportunity("BTCUSDT") == ()
    assert all(scheduler.opportunities.get_opportunity_by_id(r.id) == r for r in records)


def test_data_quality_invalid_does_not_reach_validator(tmp_path, monkeypatch):
    scheduler, clock, (record,) = setup_scheduler(tmp_path, horizons=(3600,))
    scheduler.collector.collect.side_effect = lambda s: later_snapshot(s, at=clock[0] - timedelta(minutes=1))
    validator = Mock(side_effect=AssertionError("must be blocked"))
    monkeypatch.setattr(module, "validate_forward_outcome", validator)
    scheduler.process_due()
    assert scheduler.tasks.list_for_opportunity(record.id)[0].failure_code is F.INVALID_EXIT_MARKET_DATA
    validator.assert_not_called()
    scheduler.funding.fetch_window.assert_not_called()


def test_insufficient_exit_depth_stores_incomplete_and_marks_failed(tmp_path):
    scheduler, _, (record,) = setup_scheduler(tmp_path, horizons=(3600,))
    scheduler.collector.collect.side_effect = lambda s: later_snapshot(s, spot_depth=".1")
    scheduler.process_due()
    saved, = scheduler.validations.list_validations_for_opportunity(record.id)
    assert saved.status == "INCOMPLETE" and saved.validated_simulated_net_profit is None
    assert scheduler.tasks.list_for_opportunity(record.id)[0].failure_code is F.VALIDATION_FAILED


def test_persistence_failure_keeps_original_and_other_tasks_run(tmp_path, monkeypatch):
    scheduler, _, records = setup_scheduler(tmp_path, horizons=(3600,), symbols=("BTCUSDT", "ETHUSDT"))
    save = scheduler.validations.save_forward_validation
    monkeypatch.setattr(scheduler.validations, "save_forward_validation", lambda result, **kw:
                        ValidationSaveResult(ValidationPersistenceStatus.PERSISTENCE_FAILED)
                        if result.symbol == "BTCUSDT" else save(result, **kw))
    scheduler.process_due()
    assert scheduler.tasks.list_for_opportunity("BTCUSDT")[0].failure_code is F.PERSISTENCE_FAILED
    assert scheduler.tasks.list_for_opportunity("ETHUSDT")[0].status is S.COMPLETED
    assert all(scheduler.opportunities.get_opportunity_by_id(r.id) == r for r in records)


def test_restart_pending_and_completed_with_same_database(tmp_path):
    scheduler, clock, (record,) = setup_scheduler(tmp_path)
    scheduler.process_due()
    restarted = module.ValidationScheduler(scheduler.settings, scheduler.opportunities, scheduler.validations,
                                          SQLiteValidationTaskRepository(scheduler.tasks.config),
                                          scheduler.collector, scheduler.funding, clock=lambda: clock[0])
    clock[0] = OPENED + timedelta(hours=8)
    restarted.process_due()
    statuses = [t.status for t in restarted.tasks.list_for_opportunity(record.id)]
    assert statuses == [S.COMPLETED, S.COMPLETED, S.PENDING]
    assert len(scheduler.validations.list_validations_for_opportunity(record.id)) == 2
    assert scheduler.collector.collect.call_count == 2


def test_reconcile_opportunity_saved_before_scheduling_crash(tmp_path):
    scheduler, _, _ = setup_scheduler(tmp_path, symbols=(), horizons=(3600,))
    record = research_record()
    scheduler.opportunities.save_opportunity(record)
    scheduler.process_due()
    assert scheduler.tasks.list_for_opportunity(record.id)[0].status is S.COMPLETED


def test_crash_between_validation_save_and_task_complete_never_saves_again(tmp_path, monkeypatch):
    scheduler, clock, (record,) = setup_scheduler(tmp_path, horizons=(3600,))
    original = scheduler.tasks.finish
    monkeypatch.setattr(scheduler.tasks, "finish", Mock(side_effect=OSError("simulated status write failure")))
    scheduler.process_due()
    task, = scheduler.tasks.list_for_opportunity(record.id)
    assert task.status is S.RUNNING
    assert len(scheduler.validations.list_validations_for_opportunity(record.id)) == 1
    clock[0] += timedelta(seconds=scheduler.settings.lease_seconds)
    monkeypatch.setattr(scheduler.tasks, "finish", original)
    scheduler.process_due()
    assert scheduler.tasks.get(task.id).status is S.COMPLETED
    scheduler.collector.collect.assert_called_once()
    assert len(scheduler.validations.list_validations_for_opportunity(record.id)) == 1


def test_claim_cas_and_stale_owner_cannot_finish(tmp_path):
    scheduler, clock, (record,) = setup_scheduler(tmp_path, horizons=(3600,))
    task, = scheduler.tasks.list_for_opportunity(record.id)
    first = scheduler.tasks.mark_running(task.id, now=clock[0], lease_seconds=1)
    assert scheduler.tasks.mark_running(task.id, now=clock[0], lease_seconds=1) is None
    clock[0] += timedelta(seconds=1)
    second = scheduler.tasks.mark_running(task.id, now=clock[0], lease_seconds=1)
    assert not scheduler.tasks.finish(first, now=clock[0])
    assert scheduler.tasks.get(task.id).claim_token == second.claim_token


def test_interrupted_recovery_is_bounded(tmp_path):
    scheduler, clock, (record,) = setup_scheduler(tmp_path, horizons=(3600,), max_attempts=1)
    task, = scheduler.tasks.list_for_opportunity(record.id)
    scheduler.tasks.mark_running(task.id, now=clock[0], lease_seconds=1)
    clock[0] += timedelta(seconds=1)
    scheduler.process_due()
    assert scheduler.tasks.get(task.id).failure_code is F.RECOVERY_LIMIT_EXCEEDED
    scheduler.collector.collect.assert_not_called()
    scheduler.process_due()
    assert scheduler.tasks.get(task.id).attempts == 2


def test_restart_recovers_interrupted_running_without_saved_validation(tmp_path):
    scheduler, clock, (record,) = setup_scheduler(tmp_path, horizons=(3600,))
    task, = scheduler.tasks.list_for_opportunity(record.id)
    scheduler.tasks.mark_running(task.id, now=clock[0], lease_seconds=1)
    scheduler.process_due()
    scheduler.collector.collect.assert_not_called()
    clock[0] += timedelta(seconds=1)
    restarted = module.ValidationScheduler(
        scheduler.settings, scheduler.opportunities, scheduler.validations,
        SQLiteValidationTaskRepository(scheduler.tasks.config), scheduler.collector,
        scheduler.funding, clock=lambda: clock[0], logger=Mock())
    restarted.process_due()
    recovered = restarted.tasks.get(task.id)
    assert recovered.status is S.COMPLETED and recovered.attempts == 2
    saved, = scheduler.validations.list_validations_for_opportunity(record.id)
    # Recovery uses the actual later exit, never the originally scheduled time.
    assert saved.validation_time == clock[0] and saved.validation_time > task.due_at
    restarted.process_due()
    scheduler.collector.collect.assert_called_once()


def test_bounded_batch_and_failed_tasks_never_busy_retry(tmp_path):
    scheduler, _, _ = setup_scheduler(tmp_path, horizons=(3600, 3601), batch_size=1)
    scheduler.collector.collect.side_effect = TimeoutError("connect timeout")
    scheduler.process_due()
    assert scheduler.collector.collect.call_count == 1
    scheduler.process_due()
    scheduler.process_due()
    assert scheduler.collector.collect.call_count == 2
    assert all(t.status is S.FAILED for t in scheduler.tasks.list_for_opportunity("BTCUSDT"))


def test_keyboard_interrupt_leaves_claim_recoverable_and_stops_batch(tmp_path):
    scheduler, _, _ = setup_scheduler(tmp_path, horizons=(3600, 3601))
    scheduler.collector.collect.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        scheduler.process_due()
    assert [t.status for t in scheduler.tasks.list_for_opportunity("BTCUSDT")] == [S.RUNNING, S.PENDING]
    scheduler.stopped = lambda: True
    scheduler.process_due()
    assert scheduler.collector.collect.call_count == 1


def test_fk_and_missing_parent(tmp_path):
    scheduler, _, _ = setup_scheduler(tmp_path)
    with pytest.raises(RepositoryError, match="does not exist"):
        scheduler.schedule("missing")
    with pytest.raises(RepositoryError, match="FOREIGN KEY"):
        from crypto_arbitrage_bot.storage.sqlite_repository import RepositoryOperation
        with scheduler.tasks._connection(RepositoryOperation.SAVE) as db:
            db.execute("UPDATE validation_tasks SET symbol='ETHUSDT'")
    with closing(sqlite3.connect(scheduler.tasks.config.database_path)) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1


@pytest.mark.parametrize("horizons", [[], [0], [-1], [True], [1.0], ["3600"], [3600, 3600]])
def test_invalid_horizon_config(horizons):
    with pytest.raises(ValueError):
        ValidationSchedulerSettings(validation_horizons_seconds=horizons)


def test_runner_creates_tasks_on_save_and_processes_even_when_scans_fail(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path, monkeypatch)
    cfg = ValidationSchedulerSettings(validation_horizons_seconds=(3600, 28800, 86400))
    scheduler = module.ValidationScheduler(cfg, runner.repository, SQLiteForwardValidationRepository(runner.repository.config),
                                          SQLiteValidationTaskRepository(runner.repository.config), runner.collector, Mock(),
                                          clock=runner.clock, logger=Mock())
    runner.scheduler = scheduler
    cycles = runner.run_once()
    assert all(len(scheduler.tasks.list_for_opportunity(c.opportunity_id)) == 3 for c in cycles)
    due = Mock()
    monkeypatch.setattr(scheduler, "process_due", due)
    runner.collector.collect.side_effect = TimeoutError("scan failure")
    runner.run_once()
    due.assert_called_once()


def test_runner_validation_failure_does_not_stop_next_round_and_waits(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path, monkeypatch, symbols=("BTCUSDT",))
    runner.scheduler = Mock()
    runner.scheduler.process_due.side_effect = RuntimeError("storage unavailable")
    stop = Mock(spec=Event)
    stop.is_set.return_value = False
    stop.wait.side_effect = [False, True]
    runner.stop_signal = stop
    runner.run_forever()
    assert runner.scheduler.process_due.call_count == 2
    assert runner.collector.collect.call_count == 2
    assert [call.args for call in stop.wait.call_args_list] == [(5,), (5,)]


def test_runner_keyboard_interrupt_from_scheduler_stops(tmp_path, monkeypatch):
    runner, _, _ = make_runner(tmp_path, monkeypatch)
    runner.scheduler = Mock()
    runner.scheduler.process_due.side_effect = KeyboardInterrupt
    runner.run_forever()
    assert runner.stop_signal.is_set()


def test_factory_opt_in_uses_existing_settings_and_has_no_network(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    cfg = ValidationSchedulerSettings(validation_horizons_seconds=(60, 7200))
    settings = settings.model_copy(update={"research": settings.research.model_copy(update={"validation_scheduler": cfg})})
    monkeypatch.setattr(runner_module, "configure_logging", Mock(return_value=Mock()))
    runner = runner_module.create_runner(settings)
    assert runner.scheduler.settings == cfg
    assert not settings.research.sqlite_path.exists()
    runner.stop()
    assert runner.scheduler.stopped()
