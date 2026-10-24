"""Synthetic clustered exits with original SQL/math; no actual-market claims."""

import sqlite3
import time
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from threading import Lock
from unittest.mock import Mock

import pytest
from test_candidate_catalog import NOW
from test_candidate_forward import exit_books
from test_candidate_market import offline as offline
from test_candidate_study import study as study

from crypto_arbitrage_bot import candidate_study as module
from crypto_arbitrage_bot.analytics.forward_validation import validate_forward_outcome
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.exchanges.funding_history import FundingWindow
from crypto_arbitrage_bot.models.validation_task import TaskFailure, TaskStatus
from crypto_arbitrage_bot.storage.opportunity_records import exact_json

AT = NOW+timedelta(seconds=3630)


def clustered(study, count):
    settings, _, runner, _, _ = study
    runner.run_once()
    originals=runner.repository.list_recent_opportunities()
    records=[]
    for i in range(count):
        parent=originals[i % 4]
        record=replace(parent,id=f'load-{i:03d}')
        runner.repository.save_opportunity(record)
        runner.scheduler.schedule(record.id)
        records.append(record)
    with closing(sqlite3.connect(settings.research.sqlite_path)) as db:
        # Keep original parents/tasks terminal in this synthetic load DB; a
        # deleted task would correctly be reconstructed by reconciliation.
        db.execute("UPDATE validation_tasks SET status='FAILED',failure_code='INTERRUPTED' "
                   "WHERE opportunity_id NOT LIKE ?",('load-%',))
        db.commit()
    return runner, records


def attach_sources(runner, *, fault=None):
    observations={symbol:runner.observation(symbol)[1] for symbol in runner.studies}
    guard=Lock()
    created=[]
    def event(symbol):
        return FundingEvent('synthetic-official-settlement', NOW+timedelta(hours=1),Decimal('100'),Decimal('.001'),symbol)
    class Source(module.ArchivedCandidateSources):
        def _current(self):
            owner=self
            result=Mock()
            def collect(symbol):
                assert owner.task.symbol==symbol
                time.sleep(.003)  # Real overlap, not a logical concurrency counter.
                if fault=='network' and symbol=='THEUSDT':
                    raise OSError('synthetic public timeout')
                observation=observations[symbol]
                if fault=='rules' and symbol=='THEUSDT':
                    observation=replace(observation,candidate=replace(observation.candidate,spot_tick_size=Decimal('.2')))
                assembly=exit_books(observation,AT)
                if fault=='quality' and symbol=='THEUSDT':
                    assembly=exit_books(observation,AT-timedelta(hours=1))
                return replace(observation,assembly=assembly)
            def funding(symbol,opened,closed):
                assert owner.task.symbol==symbol
                return FundingWindow(symbol,opened,closed, () if fault=='funding' and symbol=='THEUSDT' else (event(symbol),))
            result.collect.side_effect=collect
            result.fetch_funding_window.side_effect=funding
            return result
        def _archive(self,phase):
            if fault=='archive' and self.task.symbol=='THEUSDT':
                self.archive_failed=True
                raise OSError('synthetic disk failure')
            super()._archive(phase)
    def factory():
        item=Source(runner.settings,runner.studies,lambda:AT)
        with guard:
            created.append(item)
        return item
    runner.scheduler.source_factory=factory
    runner.scheduler.clock=lambda:AT
    return observations,event,created


def test_ninety_six_clustered_tasks_drain_without_poll_wait_and_match_original_cashflows(study):
    runner,records=clustered(study,96)
    before={r.id:exact_json(r) for r in records}
    observations,event,sources=attach_sources(runner)
    runner.scheduler.drain_due()
    assert len(sources)==96 and len({id(item) for item in sources})==96
    for record in records:
        task,=runner.scheduler.tasks.list_for_opportunity(record.id)
        assert task.status is TaskStatus.COMPLETED and task.attempts==1
        saved,=runner.scheduler.validations.list_validations_for_opportunity(record.id)
        expected=validate_forward_outcome(record,exit_books(observations[record.symbol],AT),
                                         validation_time=AT,funding_events=(event(record.symbol),))
        for name in ('validated_simulated_net_profit','validated_simulated_net_return','funding_total',
                     'price_pnl','capital_occupied','matched_quantity'):
            assert getattr(saved,name)==getattr(expected,name),(record.id,name)
        for name in ('spot_entry','perp_entry','spot_exit','perp_exit'):
            assert getattr(saved,name+'_fee')==getattr(expected.fees,name)
        assert before[record.id]==exact_json(runner.repository.get_opportunity_by_id(record.id))
    assert all(item.task is None and not item.captures and not item.catalogs for item in sources)


def test_maximum_original_batch_size_is_not_expanded(study):
    runner,records=clustered(study,16)
    _,_,sources=attach_sources(runner)
    runner.scheduler.process_due()
    assert len(sources)==runner.scheduler.settings.batch_size==10
    assert sum(runner.scheduler.tasks.list_for_opportunity(r.id)[0].status is TaskStatus.COMPLETED for r in records)==10
    runner.scheduler.drain_due()
    assert len(sources)==16


@pytest.mark.parametrize('fault',['network','quality','funding','archive','rules'])
def test_one_symbol_failure_kept_without_cross_symbol_source_state_or_retry(study,fault):
    runner,records=clustered(study,12)
    attach_sources(runner,fault=fault)
    runner.scheduler.drain_due()
    for record in records:
        task,=runner.scheduler.tasks.list_for_opportunity(record.id)
        assert task.attempts==1
        if record.symbol=='THEUSDT':
            assert task.status is TaskStatus.FAILED
            assert not runner.scheduler.validations.list_validations_for_opportunity(record.id)
        else:
            assert task.status is TaskStatus.COMPLETED
    states={r.id:runner.scheduler.tasks.list_for_opportunity(r.id) for r in records}
    runner.scheduler.drain_due()
    assert states=={r.id:runner.scheduler.tasks.list_for_opportunity(r.id) for r in records}


def test_late_cluster_fails_before_any_sources_are_constructed(study):
    runner,records=clustered(study,12)
    runner.scheduler.clock=lambda:AT+timedelta(seconds=301)
    factory=Mock(side_effect=AssertionError('late task must not collect'))
    runner.scheduler.source_factory=factory
    runner.scheduler.drain_due()
    factory.assert_not_called()
    for record in records:
        task,=runner.scheduler.tasks.list_for_opportunity(record.id)
        assert task.status is TaskStatus.FAILED and task.failure_code is TaskFailure.VALIDATION_WINDOW_MISSED


def test_drain_total_task_batch_and_monotonic_time_budgets_are_finite(study):
    _,_,runner,_,_=study
    scheduler=runner.scheduler
    limits=[]
    scheduler._process_batch=lambda limit,**kw:limits.append(limit) or limit
    scheduler.drain_due()
    assert limits==[10]*16 and sum(limits)==module.DRAIN_MAX_TASKS
    limits.clear()
    tick=[0]
    def steady():
        tick[0]+=241
        return tick[0]
    scheduler.steady=steady
    scheduler.drain_due()
    assert limits==[]


def test_stop_before_entry_or_after_drain_does_not_consume_another_round(study):
    _,_,runner,_,_=study
    runner.stop()
    runner.observation=Mock(side_effect=AssertionError('stopped before collection'))
    assert runner.run_once()==()
    runner.observation.assert_not_called()
    assert not runner.settings.research.study_registration.path.with_suffix('.rounds').exists()


def test_saved_on_time_validation_recovers_expired_lease_after_window_without_new_io(study):
    runner,records=clustered(study,4)
    attach_sources(runner)
    runner.scheduler.drain_due()
    record=records[0]
    task,=runner.scheduler.tasks.list_for_opportunity(record.id)
    saved,=runner.scheduler.validations.list_validations_for_opportunity(record.id)
    with closing(sqlite3.connect(runner.settings.research.sqlite_path)) as db:
        db.execute("UPDATE validation_tasks SET status='RUNNING',claim_token='interrupted',lease_until=? WHERE id=?",
                   ((AT-timedelta(seconds=1)).isoformat(timespec='microseconds'),task.id))
        db.commit()
    runner.scheduler.clock=lambda:AT+timedelta(seconds=301)
    runner.scheduler.source_factory=Mock(side_effect=AssertionError('recover saved evidence without IO'))
    runner.scheduler.drain_due()
    restored=runner.scheduler.tasks.get(task.id)
    assert restored.status is TaskStatus.COMPLETED and restored.attempts==2
    assert runner.scheduler.validations.list_validations_for_opportunity(record.id)==(saved,)
    runner.scheduler.source_factory.assert_not_called()
