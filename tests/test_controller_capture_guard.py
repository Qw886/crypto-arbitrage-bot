"""Damaged witness metadata never replaces an entry or prevents bounded exits."""

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import native_case_control as control
import pytest

from crypto_arbitrage_bot.config.research import StudyRegistrationSettings as Reference
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.native_identity import (
    NativeIdentity,
    NativeState,
    current_native_identity,
    observe_native_identity,
)

fixture_path = Path(__file__).resolve().parents[1] / 'work/native-controller-20261014/test_controller.py'
spec = importlib.util.spec_from_file_location('capture_controller_fixtures', fixture_path)
assert spec is not None and spec.loader is not None
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)
prepared = fixture_module.prepared


@pytest.fixture
def assigned(prepared, monkeypatch, request):
    original, transport, reference = prepared
    plan, settings = control.load(reference)
    moment = {'now': utc_now()}
    def clock():
        return moment['now']
    bot = Bot(settings, original.spot, original.perp, clock=clock)
    transport.clock = bot.collector.clock = bot.research_runner.clock = clock
    original_pair = original.journal.plans()[0][0]
    attempt = 'capture-' + uuid4().hex[:16]
    legs = {name: replace(getattr(original_pair, name), client_order_id=attempt + suffix)
            for name, suffix in (('spot_entry', '-se'), ('perp_entry', '-pe'),
                                 ('spot_exit', '-sx'), ('perp_exit', '-px'))}
    pair = replace(original_pair, attempt_id=attempt, **legs,
                   scheduled_exit_at=clock() + timedelta(seconds=getattr(request, 'param', 1)),
                   funding_settlement_at=clock(),
                   opportunity_id='OFFLINE_EXISTING_EXPOSURE_FIXTURE')
    assert bot.coordinator.open_pair(pair).state.value == 'HEDGED'
    moment['now'] += timedelta(seconds=2)
    folder = reference.path.parent / 'cases/07'
    folder.mkdir()
    control.protocol.exclusive(reference.path.parent / 'claims/07.json',
                               {'controller_sha256': reference.sha256, 'slot': 7})
    report = {'symbol': 'ETHUSDT', 'journal_path': str(bot.config.journal_path.resolve()),
              'sample_type': 'TESTNET_CONNECTIVITY_ACCEPTANCE', 'attempt_id': pair.attempt_id,
              'status': 'NEEDS_ATTENTION', 'witness_registration': None}
    control.protocol.exclusive(folder / 'acceptance.json', report)
    state = {'controller_sha256': reference.sha256, 'slot': 7, 'case': plan['cases'][7]['case'],
             'symbol': 'ETHUSDT', 'status': 'CAPTURING', 'attempt_id': pair.attempt_id,
             'deadline': clock() + timedelta(minutes=10), 'elapsed_ns': 0,
             'updated_at': clock(), 'owner': current_native_identity().payload()}
    control.write_state(folder / 'state.json', state)
    monkeypatch.setattr(control, 'observe_native_identity', lambda *args: NativeState.EXITED)
    approved = fixture_module.offline_approval(reference)

    def factory(config):
        fresh = Bot(config, original.spot, original.perp, clock=clock)
        fresh.collector.clock = fresh.research_runner.clock = clock
        return fresh

    def wait(seconds):
        moment['now'] += timedelta(seconds=30)

    return bot, transport, reference, approved, folder, clock, factory, wait


@pytest.mark.parametrize('registration', [None, [], 'invalid', False, {}])
def test_invalid_witness_registration_still_manages_original_due_exit(assigned, registration):
    bot, transport, reference, approved, folder, clock, factory, wait = assigned
    report = json.loads((folder / 'acceptance.json').read_text())
    report['witness_registration'] = registration
    (folder / 'acceptance.json').write_text(json.dumps(report), encoding='utf-8')
    original_claim = (reference.path.parent / 'claims/07.json').read_bytes()
    original_report = (folder / 'acceptance.json').read_bytes()
    before_orders = set(transport.orders)
    result = control.run_offline_slot(reference, 7, approval=approved, bot_factory=factory,
                                      clock=clock, wait=wait)
    assert result['status'] == 'NOT_PROVEN' and result['capture_failed'] is True
    assert bot.journal.get(result['attempt_id']).state.value == 'CLOSED'
    assert len(set(transport.orders) - before_orders) == 2
    assert transport.base['ETHUSDT'] == transport.positions['ETHUSDT'] == 0
    assert (reference.path.parent / 'claims/07.json').read_bytes() == original_claim
    assert (folder / 'acceptance.json').read_bytes() == original_report
    assert not (folder / 'proof.json').exists()


def test_missing_registration_still_manages_original_due_exit(assigned):
    bot, transport, reference, approved, folder, clock, factory, wait = assigned
    report = json.loads((folder / 'acceptance.json').read_text())
    del report['witness_registration']
    (folder / 'acceptance.json').write_text(json.dumps(report), encoding='utf-8')
    original = (folder / 'acceptance.json').read_bytes()
    before_orders = set(transport.orders)
    result = control.run_offline_slot(reference, 7, approval=approved, bot_factory=factory,
                                      clock=clock, wait=wait)
    assert result['status'] == 'NOT_PROVEN' and result['capture_failed'] is True
    assert bot.journal.get(result['attempt_id']).state.value == 'CLOSED'
    assert len(set(transport.orders) - before_orders) == 2
    assert (folder / 'acceptance.json').read_bytes() == original
    assert not (folder / 'proof.json').exists()


@pytest.mark.parametrize('assigned', [1800], indirect=True)
def test_damaged_capture_does_not_exit_early_or_extend_the_original_controller_window(assigned):
    bot, transport, reference, approved, folder, clock, factory, wait = assigned
    original = json.loads((folder / 'state.json').read_text())
    claim = (reference.path.parent / 'claims/07.json').read_bytes()
    before = set(transport.orders)
    result = control.run_offline_slot(reference, 7, approval=approved, bot_factory=factory,
                                      clock=clock, wait=wait)
    assert result['status'] == 'NEEDS_ATTENTION' and result['capture_failed'] is True
    assert bot.journal.get(result['attempt_id']).state.value == 'HEDGED'
    assert set(transport.orders) == before
    assert result['deadline'] == original['deadline']
    assert (reference.path.parent / 'claims/07.json').read_bytes() == claim
    assert not (folder / 'proof.json').exists()


def test_registration_reference_outside_case_is_not_opened(assigned):
    bot, transport, reference, approved, folder, clock, factory, wait = assigned
    foreign = folder.parent / 'outside-registration.json'
    foreign.write_text('{"source":"OFFLINE_FIXTURE_ONLY"}', encoding='utf-8')
    digest = hashlib.sha256(foreign.read_bytes()).hexdigest()
    report = json.loads((folder / 'acceptance.json').read_text())
    report['witness_registration'] = {'path': str(foreign), 'sha256': digest}
    (folder / 'acceptance.json').write_text(json.dumps(report), encoding='utf-8')
    reads = []
    recording = True

    def watch(event, args):
        if (recording and event == 'open' and isinstance(args[0], (str, bytes))
                and Path(os.fsdecode(args[0])).resolve() == foreign):
            reads.append(True)

    sys.addaudithook(watch)
    try:
        result = control.run_offline_slot(reference, 7, approval=approved, bot_factory=factory,
                                          clock=clock, wait=wait)
    finally:
        recording = False
    foreign.read_bytes()  # An unrelated access after the controller must not be observed.
    sys.audit('open', os.fsencode(foreign), 'r', 0)
    assert not reads
    assert result['status'] == 'NOT_PROVEN' and result['capture_failed'] is True
    assert bot.journal.get(result['attempt_id']).state.value == 'CLOSED'
    assert not (folder / 'proof.json').exists()


@pytest.mark.parametrize('owner', [NativeState.RUNNING, NativeState.UNKNOWN])
def test_damaged_capture_does_not_override_live_or_unknown_previous_owner(assigned, monkeypatch, owner):
    bot, transport, reference, approved, folder, clock, _, wait = assigned
    monkeypatch.setattr(control, 'observe_native_identity', lambda *args: owner)
    before = set(transport.orders)

    def forbidden_factory(config):
        pytest.fail('unverified owner must be rejected before a gateway is created')

    with pytest.raises(control.ControlError, match='PREVIOUS_OWNER_NOT_CONFIRMED_GONE'):
        control.run_offline_slot(reference, 7, approval=approved, bot_factory=forbidden_factory,
                                 clock=clock, wait=wait)
    assert set(transport.orders) == before
    assert bot.journal.get(json.loads((folder / 'state.json').read_text())['attempt_id']).state.value == 'HEDGED'


@pytest.mark.parametrize('damage', ['expired', 'unknown_budget', 'elapsed_exhausted', 'clock_backwards'])
def test_damaged_capture_preserves_original_deadline_and_consumed_time(assigned, damage):
    _, transport, reference, approved, folder, clock, _, wait = assigned
    state = json.loads((folder / 'state.json').read_text())
    if damage == 'expired':
        state['deadline'] = clock().isoformat()
    elif damage == 'unknown_budget':
        state['elapsed_ns'] = None
    elif damage == 'elapsed_exhausted':
        state['elapsed_ns'] = 29640 * 1_000_000_000
    else:
        state['updated_at'] = (clock() + timedelta(seconds=1)).isoformat()
    control.write_state(folder / 'state.json', state)
    before_state = (folder / 'state.json').read_bytes()
    before = set(transport.orders)

    def forbidden_factory(config):
        pytest.fail('invalid original budget must be rejected before a gateway is created')

    reasons = {'expired': 'CASE_DEADLINE_EXPIRED', 'unknown_budget': 'UNKNOWN_CONSUMED_TIME_BUDGET',
               'elapsed_exhausted': 'CASE_TIME_BUDGET_EXHAUSTED',
               'clock_backwards': 'CLOCK_MOVED_BACKWARDS_NO_BUDGET_REFUND'}
    with pytest.raises(control.ControlError, match=reasons[damage]):
        control.run_offline_slot(reference, 7, approval=approved, bot_factory=forbidden_factory,
                                 clock=clock, wait=wait)
    assert set(transport.orders) == before
    assert (folder / 'state.json').read_bytes() == before_state


@pytest.mark.parametrize('field,value', [('symbol', 'BTCUSDT'), ('journal_path', 'foreign.sqlite'),
                                         ('attempt_id', 'foreign-attempt')])
def test_damaged_capture_does_not_override_report_assignment(assigned, field, value):
    _, transport, reference, approved, folder, clock, _, wait = assigned
    report = json.loads((folder / 'acceptance.json').read_text())
    report[field] = value
    (folder / 'acceptance.json').write_text(json.dumps(report), encoding='utf-8')
    before = set(transport.orders)

    def forbidden_factory(config):
        pytest.fail('mismatched assignment must be rejected before a gateway is created')

    with pytest.raises(control.ControlError):
        control.run_offline_slot(reference, 7, approval=approved, bot_factory=forbidden_factory,
                                 clock=clock, wait=wait)
    assert set(transport.orders) == before


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows native independent-process recovery')
@pytest.mark.parametrize('damage', ['null', 'missing', 'outside_reference'])
def test_real_new_process_exits_damaged_capture_without_rearming(tmp_path, damage):
    worker_tests = fixture_path.parent.parent / 'controller-recovery-20261014/test_process_recovery.py'
    worker_spec = importlib.util.spec_from_file_location('capture_guard_process_fixtures', worker_tests)
    assert worker_spec is not None and worker_spec.loader is not None
    worker = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker)
    setup = worker.run('setup', tmp_path)
    assert setup.returncode == 0, setup.stderr
    reference = Reference(**json.loads((tmp_path / 'reference.json').read_text()))
    crashed = worker.run('crash', tmp_path)
    assert crashed.returncode == 23, crashed.stderr
    folder = reference.path.parent / 'cases/07'
    old = json.loads((folder / 'state.json').read_text())
    identity = NativeIdentity.parse(old['owner'])
    assert observe_native_identity(identity) in (NativeState.EXITED, NativeState.MISSING, NativeState.REPLACED)
    claim = (reference.path.parent / 'claims/07.json').read_bytes()
    exchange = json.loads((tmp_path / 'exchange.json').read_text())
    assert len(exchange['orders']) == 6 and exchange['positions']['ETHUSDT'] == '-0.01'
    report = json.loads((folder / 'acceptance.json').read_text())
    if damage == 'null':
        report['witness_registration'] = None
    elif damage == 'missing':
        del report['witness_registration']
    else:
        foreign = tmp_path / 'foreign-registration.json'
        foreign.write_text('{"source":"OFFLINE_FIXTURE"}', encoding='utf-8')
        report['witness_registration'] = {'path': str(foreign),
                                          'sha256': hashlib.sha256(foreign.read_bytes()).hexdigest()}
    (folder / 'acceptance.json').write_text(json.dumps(report), encoding='utf-8')
    damaged_report = (folder / 'acceptance.json').read_bytes()
    resumed = worker.run('resume', tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads((tmp_path / 'resume-result.json').read_text())
    assert result['state']['status'] == 'NOT_PROVEN' and result['state']['capture_failed'] is True
    assert result['mock_order_count'] == 8 and result['base'] == result['perp'] == '0.00'
    assert result['state']['attempt_id'] == old['attempt_id']
    assert result['state']['deadline'] == old['deadline']
    assert result['state']['elapsed_ns'] >= old['elapsed_ns']
    assert NativeIdentity.parse(result['state']['owner']) != identity
    assert (reference.path.parent / 'claims/07.json').read_bytes() == claim
    assert (folder / 'acceptance.json').read_bytes() == damaged_report
    assert control.status(reference)['claims_consumed'] == 1
    assert result['actual_network_requests'] == 0
    assert not (folder / 'proof.json').exists()
