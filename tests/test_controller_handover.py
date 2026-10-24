"""Explicit handover manages a known original exposure without a new entry."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import native_case_control as control
import pytest
from test_controller_capture_guard import assigned as assigned
from test_controller_capture_guard import prepared as prepared

from crypto_arbitrage_bot.config.research import StudyRegistrationSettings as Reference
from crypto_arbitrage_bot.utils.native_identity import (
    NativeIdentity,
    NativeState,
    observe_native_identity,
)


@pytest.fixture
def attention(assigned):
    bot, transport, reference, approved, folder, clock, factory, wait = assigned
    state = json.loads((folder / 'state.json').read_text())
    state['status'] = 'NEEDS_ATTENTION'  # Constructed checkpoint; not a natural-path claim.
    control.write_state(folder / 'state.json', state)
    claim_path = reference.path.parent / 'claims/07.json'
    claim = json.loads(claim_path.read_text())
    claim.update(case=state['case'], symbol=state['symbol'],
                 consumed_at=(clock() - timedelta(seconds=2)).isoformat())
    claim_path.write_text(json.dumps(claim), encoding='utf-8')  # Synthetic fixture setup only.
    return assigned


def test_explicit_management_exits_existing_attention_checkpoint_without_rearming(attention):
    bot, transport, reference, approved, folder, clock, factory, wait = attention
    original = json.loads((folder / 'state.json').read_text())
    claim = (reference.path.parent / 'claims/07.json').read_bytes()
    report = (folder / 'acceptance.json').read_bytes()
    orders = set(transport.orders)
    # Ordinary resume remains a cached status read until management is explicit.
    cached = control.run_offline_slot(reference, 7, approval=approved, bot_factory=factory,
                                     clock=clock, wait=wait)
    assert cached['status'] == 'NEEDS_ATTENTION' and set(transport.orders) == orders
    result = control.run_offline_slot(reference, 7, approval=approved, bot_factory=factory,
                                     clock=clock, wait=wait, manage_existing=True)
    assert result['status'] == 'NOT_PROVEN' and result['management_only'] is True
    assert bot.journal.get(original['attempt_id']).state.value == 'CLOSED'
    assert len(set(transport.orders) - orders) == 2
    assert result['attempt_id'] == original['attempt_id'] and result['deadline'] == original['deadline']
    assert result['elapsed_ns'] >= original['elapsed_ns']
    assert (reference.path.parent / 'claims/07.json').read_bytes() == claim
    assert (folder / 'acceptance.json').read_bytes() == report
    assert control.status(reference)['claims_consumed'] == 1
    assert not (folder / 'proof.json').exists()


def test_management_cli_is_recognized_and_refuses_without_current_authorization(attention, monkeypatch, capsys):
    _, transport, reference, _, _, _, _, _ = attention
    orders = set(transport.orders)
    monkeypatch.setattr(sys, 'argv', ['native-case-control', 'manage-slot', '--control', str(reference.path),
                                    '--control-sha256', reference.sha256, '--slot', '7'])
    assert control.main() == 1
    assert json.loads(capsys.readouterr().out)['code'] == 'CURRENT_EXPLICIT_AUTHORIZATION_REQUIRED'
    assert set(transport.orders) == orders
    assert control.status(reference)['claims_consumed'] == 1


@pytest.mark.parametrize('damage,reason', [
    ('elapsed_unknown', 'UNKNOWN_CONSUMED_TIME_BUDGET'),
    ('elapsed_exhausted', 'CASE_TIME_BUDGET_EXHAUSTED'),
    ('expired', 'CASE_DEADLINE_EXPIRED'),
    ('extended_deadline', 'ORIGINAL_CLAIM_TIME_AND_DEADLINE_REQUIRED'),
    ('clock_backwards', 'CLOCK_MOVED_BACKWARDS_NO_BUDGET_REFUND'),
    ('claim_time_missing', 'ORIGINAL_CLAIM_TIME_AND_DEADLINE_REQUIRED'),
    ('claim_time_future', 'ORIGINAL_CLAIM_TIME_AND_DEADLINE_REQUIRED'),
    ('claim_case_differs', 'MANAGEMENT_ORIGINAL_ASSIGNMENT_REQUIRED'),
    ('claim_symbol_differs', 'MANAGEMENT_ORIGINAL_ASSIGNMENT_REQUIRED'),
    ('state_attempt_differs', 'MANAGEMENT_ORIGINAL_ASSIGNMENT_REQUIRED'),
    ('report_type_differs', 'MANAGEMENT_ORIGINAL_ASSIGNMENT_REQUIRED'),
    ('saved_mode_unknown', 'UNKNOWN_SAVED_MANAGEMENT_MODE'),
    ('sealed_status', 'MANAGEABLE_ORIGINAL_CHECKPOINT_REQUIRED'),
])
def test_management_refuses_unconfirmed_original_context_before_factory(attention, damage, reason):
    _, transport, reference, approved, folder, clock, _, wait = attention
    state_path = folder / 'state.json'
    claim_path = reference.path.parent / 'claims/07.json'
    report_path = folder / 'acceptance.json'
    state = json.loads(state_path.read_text())
    claim = json.loads(claim_path.read_text())
    report = json.loads(report_path.read_text())
    if damage == 'elapsed_unknown':
        state['elapsed_ns'] = None
    elif damage == 'elapsed_exhausted':
        state['elapsed_ns'] = 29640 * 1_000_000_000
    elif damage == 'expired':
        state['deadline'] = clock().isoformat()
    elif damage == 'extended_deadline':
        state['deadline'] = (clock() + timedelta(days=8)).isoformat()
    elif damage == 'clock_backwards':
        state['updated_at'] = (clock() + timedelta(seconds=1)).isoformat()
    elif damage == 'claim_time_missing':
        del claim['consumed_at']
    elif damage == 'claim_time_future':
        claim['consumed_at'] = (clock() + timedelta(seconds=1)).isoformat()
    elif damage == 'claim_case_differs':
        claim['case'] = 'INVALID_CASE'
    elif damage == 'claim_symbol_differs':
        claim['symbol'] = 'BTCUSDT'
    elif damage == 'state_attempt_differs':
        state['attempt_id'] = 'unassigned-attempt'
    elif damage == 'report_type_differs':
        report['sample_type'] = 'STRATEGY_QUALIFICATION'
    elif damage == 'saved_mode_unknown':
        state['management_only'] = 'true'
    else:
        state['status'] = 'NOT_PROVEN'
    control.write_state(state_path, state)
    claim_path.write_text(json.dumps(claim), encoding='utf-8')
    report_path.write_text(json.dumps(report), encoding='utf-8')
    originals = {path: path.read_bytes() for path in (state_path, claim_path, report_path)}
    orders = set(transport.orders)

    def forbidden(config):
        pytest.fail('an unconfirmed context must not create a gateway')

    with pytest.raises(control.ControlError, match=reason):
        control.run_offline_slot(reference, 7, approval=approved, bot_factory=forbidden,
                                 clock=clock, wait=wait, manage_existing=True)
    assert set(transport.orders) == orders
    assert all(path.read_bytes() == value for path, value in originals.items())
    assert control.status(reference)['claims_consumed'] == 1


@pytest.mark.parametrize('owner', [NativeState.RUNNING, NativeState.UNKNOWN])
def test_management_refuses_previous_live_or_unknown_owner(attention, monkeypatch, owner):
    _, transport, reference, approved, _, clock, _, wait = attention
    monkeypatch.setattr(control, 'observe_native_identity', lambda *args: owner)
    orders = set(transport.orders)

    def forbidden(config):
        pytest.fail('an unconfirmed previous owner must not create a gateway')

    with pytest.raises(control.ControlError, match='PREVIOUS_OWNER_NOT_CONFIRMED_GONE'):
        control.run_offline_slot(reference, 7, approval=approved, bot_factory=forbidden,
                                 clock=clock, wait=wait, manage_existing=True)
    assert set(transport.orders) == orders


def test_management_without_original_claim_cannot_occupy_a_new_slot(prepared):
    _, transport, reference = prepared
    from test_controller_capture_guard import fixture_module

    from crypto_arbitrage_bot.utils.clock import utc_now

    approved = fixture_module.offline_approval(reference)
    orders = set(transport.orders)

    def forbidden(config):
        pytest.fail('management cannot create a fresh gateway or entry')

    with pytest.raises(control.ControlError, match='EXISTING_CLAIM_REQUIRED_NO_NEW_SLOT'):
        control.run_offline_slot(reference, 7, approval=approved, bot_factory=forbidden,
                                 clock=utc_now, wait=lambda seconds: None, manage_existing=True)
    assert set(transport.orders) == orders
    assert control.status(reference)['claims_consumed'] == 0


def test_stop_request_still_allows_only_original_due_exit(attention):
    bot, transport, reference, approved, folder, clock, factory, wait = attention
    assert control.stop(reference)['orders_submitted'] is False
    orders = set(transport.orders)
    result = control.run_offline_slot(reference, 7, approval=approved, bot_factory=factory,
                                     clock=clock, wait=wait, manage_existing=True)
    assert result['status'] == 'NOT_PROVEN'
    assert bot.journal.get(result['attempt_id']).state.value == 'CLOSED'
    assert len(set(transport.orders) - orders) == 2
    assert not (folder / 'proof.json').exists()


def run_process(mode, directory, *, tick_controller=False):
    worker = Path('work/controller-handover-20261015/fixture_worker.py').resolve()
    command = [sys.executable, str(worker), mode, str(directory)]
    if tick_controller:
        command.append('--tick-controller')
    return subprocess.run(command,
                          env=dict(os.environ, PYTHONPATH=os.pathsep.join(
                              str(Path(name).resolve()) for name in ('src', 'work'))),
                          capture_output=True, encoding='utf-8', errors='replace', timeout=45,
                          creationflags=subprocess.CREATE_NO_WINDOW)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows native-process acceptance')
@pytest.mark.parametrize('tick_controller', [False, True])
def test_new_process_manages_naturally_saved_attention_after_synthetic_clock_rollback(tmp_path, tick_controller):
    setup = run_process('setup', tmp_path)
    assert setup.returncode == 0, setup.stderr
    attention_result = run_process('attention', tmp_path, tick_controller=tick_controller)
    assert attention_result.returncode == 0, attention_result.stderr
    reference = Reference(**json.loads((tmp_path / 'reference.json').read_text()))
    before = json.loads((tmp_path / 'attention-result.json').read_text())
    assert before['state']['status'] == 'NEEDS_ATTENTION' and before['mock_order_count'] == 6
    owner = NativeIdentity.parse(before['state']['owner'])
    assert observe_native_identity(owner) in (NativeState.MISSING, NativeState.EXITED, NativeState.REPLACED)
    folder = reference.path.parent / 'cases/07'
    claim = (reference.path.parent / 'claims/07.json').read_bytes()
    report = (folder / 'acceptance.json').read_bytes()
    managed = run_process('manage', tmp_path)
    assert managed.returncode == 0, managed.stderr
    result = json.loads((tmp_path / 'manage-result.json').read_text())
    assert result['state']['status'] == 'NOT_PROVEN' and result['state']['management_only'] is True
    assert result['mock_order_count'] == 8 and result['base'] == result['perp'] == '0.00'
    assert NativeIdentity.parse(result['state']['owner']) != owner
    assert result['state']['attempt_id'] == before['state']['attempt_id']
    assert result['state']['deadline'] == before['state']['deadline']
    assert result['state']['elapsed_ns'] >= before['state']['elapsed_ns']
    assert (reference.path.parent / 'claims/07.json').read_bytes() == claim
    assert (folder / 'acceptance.json').read_bytes() == report
    assert result['network_mode'] == 'OFFLINE_TRANSPORT_AND_PYTHON_GUARD'
    assert not (folder / 'proof.json').exists()


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows native-process acceptance')
def test_management_mode_survives_real_crash_and_ordinary_new_process_resume(tmp_path):
    assert run_process('setup', tmp_path).returncode == 0
    initial = run_process('attention', tmp_path)
    assert initial.returncode == 0, initial.stderr
    reference = Reference(**json.loads((tmp_path / 'reference.json').read_text()))
    folder = reference.path.parent / 'cases/07'
    original = json.loads((folder / 'state.json').read_text())
    claim = (reference.path.parent / 'claims/07.json').read_bytes()
    report = (folder / 'acceptance.json').read_bytes()
    crash = run_process('manage-crash', tmp_path)
    assert crash.returncode == 29, crash.stderr
    interrupted = json.loads((folder / 'state.json').read_text())
    assert interrupted['status'] == 'MANAGING_WITHOUT_VALID_CAPTURE'
    assert interrupted['management_only'] is True
    identity = NativeIdentity.parse(interrupted['owner'])
    assert observe_native_identity(identity) in (NativeState.MISSING, NativeState.EXITED, NativeState.REPLACED)
    assert len(json.loads((tmp_path / 'exchange.json').read_text())['orders']) == 6
    resumed = run_process('resume', tmp_path)  # No management flag in this process.
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads((tmp_path / 'resume-result.json').read_text())
    assert result['state']['status'] == 'NOT_PROVEN' and result['state']['management_only'] is True
    assert result['state']['attempt_id'] == original['attempt_id']
    assert result['state']['deadline'] == original['deadline']
    assert result['state']['elapsed_ns'] >= interrupted['elapsed_ns']
    assert NativeIdentity.parse(result['state']['owner']) != identity
    assert result['mock_order_count'] == 8 and result['base'] == result['perp'] == '0.00'
    assert (reference.path.parent / 'claims/07.json').read_bytes() == claim
    assert (folder / 'acceptance.json').read_bytes() == report
    assert control.status(reference)['claims_consumed'] == 1
    assert not (folder / 'proof.json').exists()


@pytest.mark.parametrize('mode', [1, 'true', None])
def test_management_flag_requires_an_explicit_boolean(attention, mode):
    _, _, reference, approved, folder, clock, factory, wait = attention
    before = (folder / 'state.json').read_bytes()
    with pytest.raises(control.ControlError, match='EXPLICIT_MANAGEMENT_MODE_REQUIRED'):
        control.run_offline_slot(reference, 7, approval=approved, bot_factory=factory,
                                 clock=clock, wait=wait, manage_existing=mode)
    assert (folder / 'state.json').read_bytes() == before


def test_management_charges_the_original_consumption_and_recovery_gap(attention):
    _, _, reference, approved, folder, clock, factory, wait = attention
    state = json.loads((folder / 'state.json').read_text())
    state.update(elapsed_ns=5_000_000_000, updated_at=(clock() - timedelta(seconds=60)).isoformat())
    control.write_state(folder / 'state.json', state)
    result = control.run_offline_slot(reference, 7, approval=approved, bot_factory=factory,
                                     clock=clock, wait=wait, manage_existing=True)
    assert result['status'] == 'NOT_PROVEN'
    assert result['elapsed_ns'] >= 65_000_000_000
    assert result['deadline'] == state['deadline']


def test_management_will_not_reopen_a_terminal_original_attempt(attention):
    bot, transport, reference, approved, _, clock, _, wait = attention
    bot.manage_existing_once()
    orders = set(transport.orders)

    def forbidden(config):
        pytest.fail('a terminal attempt does not need a gateway')

    with pytest.raises(control.ControlError, match='EXISTING_NONTERMINAL_ATTEMPT_REQUIRED'):
        control.run_offline_slot(reference, 7, approval=approved, bot_factory=forbidden,
                                 clock=clock, wait=wait, manage_existing=True)
    assert set(transport.orders) == orders


def test_management_cli_keeps_the_native_review_gate(attention, monkeypatch, capsys):
    _, transport, reference, approved, _, _, _, _ = attention
    orders = set(transport.orders)
    monkeypatch.setattr(sys, 'argv', ['native-case-control', 'manage-slot', '--control', str(reference.path),
                                    '--control-sha256', reference.sha256, '--slot', '7',
                                    '--approval', str(approved.path), '--approval-sha256', approved.sha256])
    assert control.main() == 1
    assert json.loads(capsys.readouterr().out)['code'] == 'NATIVE_EXECUTION_NOT_REVIEWED'
    assert set(transport.orders) == orders


@pytest.mark.parametrize('explicit', [False, True])
def test_single_case_handover_refuses_another_assigned_nonterminal_exposure(attention, explicit):
    bot, transport, reference, approved, folder, clock, _, wait = attention
    target = json.loads((folder / 'state.json').read_text())['attempt_id']
    original = next(pair for pair, _ in bot.journal.plans() if pair.attempt_id == target)
    attempt = 'other-' + uuid4().hex[:16]
    legs = {name: replace(getattr(original, name), symbol='BTCUSDT', quantity=Decimal('.001'),
                          limit_price=Decimal(price), client_order_id=attempt + suffix)
            for name, price, suffix in (('spot_entry', '50001', '-se'), ('perp_entry', '50009', '-pe'),
                                        ('spot_exit', '49999', '-sx'), ('perp_exit', '50011', '-px'))}
    second = replace(original, attempt_id=attempt, **legs, opportunity_id='OFFLINE_SECOND_CASE_FIXTURE')
    assert bot.coordinator.open_pair(second).state.value == 'HEDGED'
    other = reference.path.parent / 'cases/00'
    other.mkdir()
    control.protocol.exclusive(other / 'acceptance.json', {
        'symbol': 'BTCUSDT', 'journal_path': str(bot.config.journal_path.resolve()),
        'sample_type': 'TESTNET_CONNECTIVITY_ACCEPTANCE', 'attempt_id': attempt})
    control.protocol.exclusive(reference.path.parent / 'claims/00.json',
                               {'controller_sha256': reference.sha256, 'slot': 0})
    state = json.loads((folder / 'state.json').read_text())
    state['status'] = 'CAPTURING'
    control.write_state(folder / 'state.json', state)
    orders = set(transport.orders)

    def forbidden(config):
        pytest.fail('a one-case handover must not operate another nonterminal case')

    with pytest.raises(control.ControlError, match='OTHER_NONTERMINAL_ATTEMPT_BLOCKS_MANAGEMENT'):
        control.run_offline_slot(reference, 7, approval=approved, bot_factory=forbidden,
                                 clock=clock, wait=wait, manage_existing=explicit)
    assert set(transport.orders) == orders
    assert control.status(reference)['claims_consumed'] == 2


def test_new_claim_and_case_deadline_use_the_same_utc_sample(prepared):
    from test_controller_capture_guard import fixture_module

    from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot
    from crypto_arbitrage_bot.utils.clock import utc_now

    old, transport, reference = prepared
    approved = fixture_module.offline_approval(reference)
    now = [utc_now() + timedelta(seconds=1)]

    def advancing_clock():
        now[0] += timedelta(microseconds=1)
        return now[0]

    result = control.run_offline_slot(
        reference, 7, approval=approved, clock=advancing_clock,
        bot_factory=lambda settings: Bot(settings, old.spot, old.perp),
        wait=lambda seconds: control.stop(reference))
    assert result['status'] == 'CANCELLED' and len(transport.orders) == 4
    claim = json.loads((reference.path.parent / 'claims/07.json').read_text())
    from datetime import datetime

    deadline = datetime.fromisoformat(result['deadline']) if isinstance(result['deadline'], str) else result['deadline']
    assert deadline <= datetime.fromisoformat(claim['consumed_at']) + timedelta(seconds=29640)
