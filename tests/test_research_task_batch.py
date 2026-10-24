"""Real threads with isolated offline tasks; no exchange access or orders."""

from dataclasses import replace
from datetime import UTC, datetime
from threading import Barrier, Event, Lock, get_ident
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot.models.validation_task import TaskStatus, ValidationTask
from crypto_arbitrage_bot.research_task_batch import process_task_batch

AT = datetime(2026, 10, 9, tzinfo=UTC)


def tasks(count):
    return tuple(ValidationTask(str(i), str(i), 'BTCUSDT', 60, AT, TaskStatus.PENDING,
                               str(i), AT, AT) for i in range(count))


def test_four_real_workers_claim_only_available_slots_and_pulse_from_owner_thread():
    owner = get_ident()
    gate, guard = Barrier(4), Lock()
    active, claimed, seen, pulses = set(), [], [], []
    max_active = [0]
    def claim(task):
        assert get_ident() == owner
        with guard:
            # No leases consumed while waiting in an executor queue.
            assert len(claimed)-len(seen) < 4
            claimed.append(task.id)
        return replace(task, status=TaskStatus.RUNNING)
    def execute(task):
        with guard:
            active.add(task.id)
            max_active[0] = max(max_active[0], len(active))
        if int(task.id) < 4:
            gate.wait(timeout=10)
        with guard:
            active.remove(task.id)
            seen.append(task.id)
    def pulse():
        pulses.append(get_ident())
    assert process_task_batch(tasks(96), claim=claim, execute=execute,
        failure=Mock(side_effect=AssertionError('unexpected failure')), stopped=lambda: False, progress=pulse) == 96
    assert sorted(claimed) == sorted(seen) == sorted(str(i) for i in range(96))
    assert max_active[0] == 4 and pulses and set(pulses) == {owner}


def test_stop_finishes_inflight_tasks_without_claiming_the_remainder():
    stop = Event()
    gate = Barrier(4)
    claims, completed = [], []
    def claim(task):
        claims.append(task.id)
        return task
    def execute(task):
        gate.wait(timeout=10)
        stop.set()
        completed.append(task.id)
    count = process_task_batch(tasks(30),claim=claim,execute=execute,
        failure=Mock(side_effect=AssertionError('unexpected failure')),stopped=stop.is_set,progress=lambda:None)
    assert count == len(claims) == len(completed) == 4


def test_failure_is_reported_once_by_owner_and_other_tasks_continue():
    owner = get_ident()
    seen, errors = [], []
    def execute(task):
        if task.id=='2':
            raise OSError('synthetic offline failure')
        seen.append(task.id)
    def failure(task,error):
        assert get_ident()==owner
        errors.append((task.id,type(error).__name__))
    assert process_task_batch(tasks(10),claim=lambda t:t,execute=execute,
        failure=failure,stopped=lambda:False,progress=lambda:None)==10
    assert len(seen)==9 and errors==[('2','OSError')]


def test_preexisting_stop_does_not_claim_or_execute():
    claim,execute=Mock(),Mock()
    assert process_task_batch(tasks(10),claim=claim,execute=execute,failure=Mock(),
                              stopped=lambda:True,progress=Mock())==0
    claim.assert_not_called()
    execute.assert_not_called()


def test_lost_claim_is_skipped_without_fabricating_a_completion():
    execute=Mock()
    assert process_task_batch(tasks(3),claim=lambda t:None,execute=execute,failure=Mock(),
                              stopped=lambda:False,progress=Mock())==0
    execute.assert_not_called()


@pytest.mark.parametrize('workers',[0,5,True,-1])
def test_bad_worker_bound_rejected_before_claiming(workers):
    claim=Mock()
    with pytest.raises(ValueError):
        process_task_batch(tasks(2),claim=claim,execute=Mock(),failure=Mock(),stopped=Mock(),progress=Mock(),workers=workers)
    claim.assert_not_called()


@pytest.mark.parametrize('fault',['progress','interrupt'])
def test_progress_storage_failure_or_interrupt_cannot_start_another_batch(fault):
    claims=[]
    def claim(task):
        claims.append(task.id)
        return task
    def execute(task):
        if fault=='interrupt':
            raise KeyboardInterrupt
    def progress():
        if fault=='progress':
            raise OSError('offline progress disk failure')
    with pytest.raises((OSError,KeyboardInterrupt)):
        process_task_batch(tasks(40),claim=claim,execute=execute,failure=Mock(),stopped=lambda:False,progress=progress)
    assert len(claims)<=4
