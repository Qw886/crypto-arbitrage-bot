"""Fixed native-case control. A prepared plan never authorizes a private request."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic_ns, sleep
from typing import Any

import native_case_protocol as protocol

from crypto_arbitrage_bot.config.research import StudyRegistrationSettings as Reference
from crypto_arbitrage_bot.config.settings import Settings, load_settings
from crypto_arbitrage_bot.execution.balance_ledger import reconcile_balance_ledger
from crypto_arbitrage_bot.execution.captured_case_proof import export_captured_case_proof
from crypto_arbitrage_bot.execution.funding_acceptance import FundingAcceptance
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState
from crypto_arbitrage_bot.execution.risk_review import implementation_fingerprint
from crypto_arbitrage_bot.execution.risk_witness import RiskWitness, load_witness
from crypto_arbitrage_bot.execution.runner import TestnetBot, lease_process_lock, process_lock
from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind, TestnetExecutionError
from crypto_arbitrage_bot.execution.types import ExecutionMarket, IocLimitOrder, OrderReceipt
from crypto_arbitrage_bot.models.market import Side
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.atomic_publish import publish_state
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp
from crypto_arbitrage_bot.utils.native_identity import (
    NativeIdentity,
    NativeState,
    current_native_identity,
    observe_native_identity,
)


class ControlError(RuntimeError):
    pass


def prepare(settings: Settings,directory: Path,*,workspace: Path | None = None,
            clock: Callable[[],datetime] = utc_now) -> Reference:
    reference = protocol.prepare(settings,directory,workspace=workspace,clock=clock)
    body = {'schema':'native-controller-v1','protocol':reference.model_dump(mode='json'),
            'controller_sha256':protocol.digest(Path(__file__)),'normal_trading_allowed':False,
            'native_execution_reviewed':False}
    protocol.exclusive(reference.path.parent/'control.json',body)
    frozen = reference.path.parent/'control-src'
    frozen.mkdir()
    for source in (Path(__file__),Path(protocol.__file__)):
        with (frozen/source.name).open('xb') as output:
            output.write(source.read_bytes())
    return Reference(path=reference.path.parent/'control.json',sha256=protocol.digest(reference.path.parent/'control.json'))


def load(reference: Reference) -> tuple[dict[str,Any],Settings]:
    control = protocol.document(reference.path,reference.sha256)
    if (control['schema']!='native-controller-v1' or control['normal_trading_allowed'] is not False
            or control['controller_sha256']!=protocol.digest(Path(__file__))):
        raise ControlError('FIXED_CONTROLLER_DIFFERS')
    for source in (Path(__file__),Path(protocol.__file__)):
        copied = reference.path.parent/'control-src'/source.name
        if copied.is_symlink() or protocol.digest(copied)!=protocol.digest(source):
            raise ControlError('FROZEN_CONTROLLER_DIFFERS')
    plan,settings = protocol.load(Reference(**control['protocol']))
    if reference.path.parent.resolve()!=Path(plan['directory']):
        raise ControlError('CONTROLLER_DIRECTORY_DIFFERS')
    for name in ('claims','cases'):
        directory = reference.path.parent/name
        if directory.is_symlink() or directory.is_junction() or not directory.is_dir():
            raise ControlError('REGULAR_CONTROL_DIRECTORIES_REQUIRED')
    allowed = {f'{number:02d}.json' for number in range(14)}
    if any(path.name not in allowed or not path.is_file() or path.is_symlink() for path in (reference.path.parent/'claims').iterdir()):
        raise ControlError('UNKNOWN_CASE_CLAIM')
    return plan,settings


def status(reference: Reference) -> dict[str,Any]:
    plan,_ = load(reference)
    claims = list((reference.path.parent/'claims').iterdir())
    if any(path.name not in {f'{number:02d}.json' for number in range(14)} or not path.is_file() or path.is_symlink() for path in claims):
        raise ControlError('UNKNOWN_CASE_CLAIM')
    cases = []
    for item in plan['cases']:
        path = reference.path.parent/'cases'/f"{item['slot']:02d}"/'state.json'
        value = read_state(path) if path.exists() else None
        observed = None if value is None else observe_native_identity(NativeIdentity.parse(value['owner'])).value
        cases.append({**item,'claim_consumed':(reference.path.parent/'claims'/f"{item['slot']:02d}.json").exists(),
                      'saved_status':None if value is None else value['status'],'native_owner_observation':observed})
    return {'status':'STOP_REQUESTED' if (reference.path.parent/'stop').exists() else 'INSPECTED_NOT_AUTHORIZED',
            'claims_consumed':len(claims),'cases':cases,
            'native_case_slots':14,'normal_trading_allowed':False,'inspection_private_requests':0,
            'proposed_global_limit':plan['proposed_global_limit']}


def read_state(path: Path) -> dict[str,Any]:
    if path.is_symlink() or path.is_junction():
        raise ControlError('REGULAR_STATE_REQUIRED')
    with path.open('rb') as source:
        body = source.read(protocol.MAX_DOCUMENT+1)
    if len(body)>protocol.MAX_DOCUMENT:
        raise ControlError('STATE_LIMIT')
    try:
        data = json.loads(body)
    except (ValueError,UnicodeError) as error:
        raise ControlError('PARTIAL_OR_INVALID_STATE_NOT_REPAIRED') from error
    if not isinstance(data,dict):
        raise ControlError('INVALID_STATE')
    return data


def write_state(path: Path,value: dict[str,Any]) -> None:
    body = exact_json(value).encode('utf-8')
    if len(body)>protocol.MAX_DOCUMENT:
        raise ControlError('STATE_LIMIT')
    temporary = path.with_suffix('.tmp')
    with temporary.open('xb') as output:
        output.write(body)
        output.flush()
        os.fsync(output.fileno())
    publish_state(temporary,path)


def stop(reference: Reference) -> dict[str,Any]:
    load(reference)
    path = reference.path.parent/'stop'
    if not path.exists():
        protocol.exclusive(path,{'controller_sha256':reference.sha256,'request':'BLOCK_NEW_ENTRY_KEEP_ORIGINAL_EXITS'})
    return {'status':'STOP_REQUESTED','orders_submitted':False,'workers_killed':False}


def attempt_state(journal: Path,attempt: str | None) -> str | None:
    if attempt is None:
        return None
    with closing(sqlite3.connect(journal.resolve().as_uri()+'?mode=ro',uri=True,timeout=0)) as db:
        row = db.execute('SELECT state FROM testnet_pair_attempts WHERE attempt_id=?',(attempt,)).fetchone()
    return None if row is None else row[0]


def verify_assignments(plan: dict[str,Any],directory: Path) -> None:
    current,_ = protocol.original_records(Path(plan['journal_path']),plan['original_attempt_ids'])
    assigned = []
    for item in plan['cases']:
        path = directory/'cases'/f"{item['slot']:02d}"/'acceptance.json'
        if path.exists():
            report = read_state(path)
            if report.get('symbol')!=item['symbol'] or report.get('journal_path')!=plan['journal_path']:
                raise ControlError('CLAIM_REPORT_IDENTITY_DIFFERS')
            attempt = report.get('attempt_id')
            if not isinstance(attempt,str) or attempt in plan['original_attempt_ids']:
                raise ControlError('ORIGINAL_OR_UNKNOWN_ATTEMPT_ASSIGNMENT')
            assigned.append(attempt)
    if len(set(assigned))!=len(assigned) or not set(current)-set(plan['original_attempt_ids'])<=set(assigned):
        raise ControlError('UNASSIGNED_EXECUTION_NO_NEW_PRIVATE_ACCESS')


def authorization(reference: Reference,plan: dict[str,Any],approval: Reference | None,
                  *,clock: Callable[[],datetime] = utc_now) -> dict[str,Any]:
    if approval is None:
        raise ControlError('CURRENT_EXPLICIT_AUTHORIZATION_REQUIRED')
    document = protocol.document(approval.path,approval.sha256)
    expected = {'schema':'native-case-current-authorization-v1','controller_sha256':reference.sha256,
                'scope':'TESTNET_NATIVE_CASES_ONLY','native_case_slots':14,'normal_trading_allowed':False}
    if any(document.get(key)!=value for key,value in expected.items()):
        raise ControlError('EXACT_NATIVE_CASE_AUTHORIZATION_REQUIRED')
    if type(document['native_case_slots']) is not int or document['normal_trading_allowed'] is not False:
        raise ControlError('EXACT_NATIVE_CASE_AUTHORIZATION_REQUIRED')
    at = utc_timestamp(clock())
    approved = datetime.fromisoformat(document['approved_at'])
    if not datetime.fromisoformat(plan['created_at'])<=approved<=at<datetime.fromisoformat(plan['ends_at']):
        raise ControlError('AUTHORIZATION_OUTSIDE_FIXED_WINDOW')
    if document.get('current_user_confirmed') is not True or not isinstance(document.get('authorization_record'),str) or not document['authorization_record']:
        raise ControlError('CURRENT_EXPLICIT_AUTHORIZATION_REQUIRED')
    return document


def run_slot(reference: Reference,slot: int,*,approval: Reference | None,
             bot_factory: Callable[[Settings],TestnetBot] | None = None,
             clock: Callable[[],datetime] = utc_now,manage_existing: bool = False) -> dict[str,Any]:
    plan,settings = load(reference)
    authorization(reference,plan,approval,clock=clock)
    if type(slot) is not int or not 0<=slot<14:
        raise ControlError('EXACT_CASE_SLOT_REQUIRED')
    control = protocol.document(reference.path,reference.sha256)
    if control.get('native_execution_reviewed') is not True:
        raise ControlError('NATIVE_EXECUTION_NOT_REVIEWED')
    if implementation_fingerprint()!=plan['source_sha256']:
        raise ControlError('PINNED_IMPLEMENTATION_REQUIRED')
    if (reference.path.parent/'stop').exists() and not (reference.path.parent/'claims'/f'{slot:02d}.json').exists():
        raise ControlError('STOP_REQUESTED')
    return _run_slot(reference,slot,approval=approval,bot_factory=bot_factory,clock=clock,native=True,
                     manage_existing=manage_existing)


def run_offline_slot(reference: Reference,slot: int,*,approval: Reference,
                     bot_factory: Callable[[Settings],TestnetBot],clock: Callable[[],datetime],
                     wait: Callable[[float],None],manage_existing: bool = False) -> dict[str,Any]:
    """Explicit offline acceptance seam. This path refuses default live gateways."""
    plan,_ = load(reference)
    approved = authorization(reference,plan,approval,clock=clock)
    if approved['authorization_record']!='OFFLINE_FIXTURE':
        raise ControlError('EXPLICIT_OFFLINE_FIXTURE_REQUIRED')
    return _run_slot(reference,slot,approval=approval,bot_factory=bot_factory,clock=clock,native=False,
                     wait=wait,manage_existing=manage_existing)


def _run_slot(reference: Reference,slot: int,*,approval: Reference | None,
              bot_factory: Callable[[Settings],TestnetBot] | None,clock: Callable[[],datetime],
              native: bool,wait: Callable[[float],None] = sleep,
              manage_existing: bool = False) -> dict[str,Any]:
    if type(manage_existing) is not bool:
        raise ControlError('EXPLICIT_MANAGEMENT_MODE_REQUIRED')
    plan,settings = load(reference)
    authorization(reference,plan,approval,clock=clock)
    if type(slot) is not int or not 0<=slot<14:
        raise ControlError('EXACT_CASE_SLOT_REQUIRED')
    if implementation_fingerprint()!=plan['source_sha256']:
        raise ControlError('PINNED_IMPLEMENTATION_REQUIRED')
    if (reference.path.parent/'stop').exists() and not (reference.path.parent/'claims'/f'{slot:02d}.json').exists():
        raise ControlError('STOP_REQUESTED')
    assert settings.testnet is not None
    with process_lock(reference.path.parent/'control.lock'),lease_process_lock(settings.testnet.journal_path.with_suffix('.lock')) as journal_lease:
        # Recheck under both original OS locks before any credential factory.
        plan,settings = load(reference)
        authorization(reference,plan,approval,clock=clock)
        assert settings.testnet is not None
        verify_assignments(plan,reference.path.parent)
        claim_path = reference.path.parent/'claims'/f'{slot:02d}.json'
        folder = reference.path.parent/'cases'/f'{slot:02d}'
        item = plan['cases'][slot]
        if manage_existing and not claim_path.exists():
            raise ControlError('EXISTING_CLAIM_REQUIRED_NO_NEW_SLOT')
        if claim_path.exists():
            claim = read_state(claim_path)
            if claim['controller_sha256']!=reference.sha256 or claim['slot']!=slot:
                raise ControlError('PARTIAL_OR_CONFLICTING_CLAIM_NOT_REFUNDED')
            if not (folder/'state.json').exists():
                raise ControlError('MISSING_STATE_NO_NEW_ENTRY')
            state = read_state(folder/'state.json')
            if state['controller_sha256']!=reference.sha256 or state['slot']!=slot or state['case']!=item['case']:
                raise ControlError('CASE_STATE_IDENTITY_DIFFERS')
            saved_management = state.get('management_only',False)
            if type(saved_management) is not bool:
                raise ControlError('UNKNOWN_SAVED_MANAGEMENT_MODE')
            if not manage_existing and state['status'] in ('EXPORTED','NOT_PROVEN','CANCELLED','NEEDS_ATTENTION'):
                return state
            manage_existing = manage_existing or saved_management
            if manage_existing and state['status'] not in (
                    'NEEDS_ATTENTION','CAPTURING','STARTING','MANAGING_WITHOUT_VALID_CAPTURE'):
                raise ControlError('MANAGEABLE_ORIGINAL_CHECKPOINT_REQUIRED')
            observed = observe_native_identity(NativeIdentity.parse(state['owner']))
            if observed not in (NativeState.EXITED,NativeState.MISSING,NativeState.REPLACED):
                raise ControlError('PREVIOUS_OWNER_NOT_CONFIRMED_GONE')
            if not (folder/'acceptance.json').exists():
                raise ControlError('MISSING_REPORT_NO_NEW_ENTRY')
            deadline = datetime.fromisoformat(state['deadline'])
            if manage_existing:
                report = read_state(folder/'acceptance.json')
                if (state.get('symbol')!=item['symbol'] or claim.get('symbol')!=item['symbol']
                        or claim.get('case')!=item['case'] or report.get('attempt_id')!=state.get('attempt_id')
                        or report.get('sample_type')!='TESTNET_CONNECTIVITY_ACCEPTANCE'):
                    raise ControlError('MANAGEMENT_ORIGINAL_ASSIGNMENT_REQUIRED')
                try:
                    consumed = utc_timestamp(datetime.fromisoformat(claim['consumed_at']))
                    if not datetime.fromisoformat(plan['created_at'])<=consumed<=utc_timestamp(clock()):
                        raise ValueError('claim time outside original window')
                    if utc_timestamp(deadline)>min(datetime.fromisoformat(plan['ends_at']),consumed+timedelta(seconds=29640)):
                        raise ValueError('deadline outside original claim cap')
                except (KeyError,TypeError,ValueError):
                    raise ControlError('ORIGINAL_CLAIM_TIME_AND_DEADLINE_REQUIRED') from None
                active = {value.value for value in PairExecutionState} - {'CLOSED','FLATTENED','FAILED'}
                current_attempt = attempt_state(settings.testnet.journal_path,state.get('attempt_id'))
                if current_attempt is None or current_attempt not in active:
                    raise ControlError('EXISTING_NONTERMINAL_ATTEMPT_REQUIRED')
            if utc_timestamp(clock())>=deadline:
                raise ControlError('CASE_DEADLINE_EXPIRED')
            if utc_timestamp(clock())<datetime.fromisoformat(state['updated_at']):
                raise ControlError('CLOCK_MOVED_BACKWARDS_NO_BUDGET_REFUND')
            used = state['elapsed_ns']
            if type(used) is not int or not 0<=used<=29640*1_000_000_000:
                raise ControlError('UNKNOWN_CONSUMED_TIME_BUDGET')
            gap = utc_timestamp(clock())-datetime.fromisoformat(state['updated_at'])
            state['elapsed_ns']=used+((gap.days*86400+gap.seconds)*1_000_000+gap.microseconds)*1000
            known_attempts,_ = protocol.original_records(settings.testnet.journal_path,plan['original_attempt_ids'])
            if any(attempt!=state['attempt_id'] and attempt_state(settings.testnet.journal_path,attempt)
                   not in ('CLOSED','FLATTENED','FAILED') for attempt in known_attempts):
                raise ControlError('OTHER_NONTERMINAL_ATTEMPT_BLOCKS_MANAGEMENT')
        else:
            if (reference.path.parent/'stop').exists():
                raise ControlError('STOP_REQUESTED')
            claim_at = utc_timestamp(clock())
            deadline = min(datetime.fromisoformat(plan['ends_at']),claim_at+timedelta(seconds=29640))
            # Occupy before a factory, arming GET, entry claim or POST. Never refund.
            protocol.exclusive(claim_path,{'controller_sha256':reference.sha256,'slot':slot,'case':item['case'],
                'symbol':item['symbol'],'consumed_at':claim_at})
            folder.mkdir(exist_ok=False)
            state = {'controller_sha256':reference.sha256,'slot':slot,'case':item['case'],'symbol':item['symbol'],
                'status':'CLAIMED','deadline':deadline,'owner':current_native_identity().payload(),'attempt_id':None,
                'elapsed_ns':0,
                'normal_trading_allowed':False,'origin':'NATIVE_TESTNET' if native else 'OFFLINE_FIXTURE'}
        remaining_ns = min(29640*1_000_000_000-state['elapsed_ns'],max(0,int((deadline-utc_timestamp(clock())).total_seconds()))*1_000_000_000)
        if remaining_ns<=0:
            raise ControlError('CASE_TIME_BUDGET_EXHAUSTED')
        timer_start = monotonic_ns()
        timer_deadline = timer_start+remaining_ns
        state.update(owner=current_native_identity().payload(),status='STARTING',updated_at=utc_timestamp(clock()))
        if manage_existing:
            state['management_only'] = True
        write_state(folder/'state.json',state)
        if bot_factory is None:
            if os.environ.get('CAB_TESTNET_ORDERS_ENABLED','').lower()!='true':
                raise ControlError('EXPLICIT_TESTNET_ENABLE_REQUIRED')
            from crypto_arbitrage_bot.execution.credential_store import (
                DEFAULT_CREDENTIAL_FILE,
                execution_environment,
            )
            from crypto_arbitrage_bot.execution.runner import create_testnet_bot
            bot = create_testnet_bot(settings,environ=execution_environment(DEFAULT_CREDENTIAL_FILE,os.environ))
        else:
            bot = bot_factory(settings)
        origins = (bot.spot.uses_default_transport,bot.perp.uses_default_transport)
        if native and not all(origins) or not native and any(origins):
            raise ControlError('GATEWAY_ORIGIN_DIFFERS')
        if bot.settings.model_dump(mode='json')!=settings.model_dump(mode='json'):
            raise ControlError('FACTORY_POLICY_DIFFERS')
        with bot.controlled_session(manage_only=manage_existing, lock_lease=journal_lease):
            return capture_session(reference,plan,item,folder,state,bot,clock=clock,wait=wait,
                                   timer_deadline=timer_deadline,timer_start=timer_start,
                                   manage_existing=manage_existing)


def replay_slot(reference: Reference,slot: int,*,clock: Callable[[],datetime] = utc_now) -> dict[str,Any]:
    """Original terminal restart capture in this new process, no credential factory."""
    plan,settings = load(reference)
    if type(slot) is not int or not 0<=slot<14 or plan['cases'][slot]['case']!='RESTART_IDEMPOTENCY':
        raise ControlError('EXACT_RESTART_SLOT_REQUIRED')
    if implementation_fingerprint()!=plan['source_sha256']:
        raise ControlError('PINNED_IMPLEMENTATION_REQUIRED')
    folder = reference.path.parent/'cases'/f'{slot:02d}'
    with process_lock(reference.path.parent/'control.lock'):
        state = read_state(folder/'state.json')
        if state['status']!='RESTART_PENDING_NEW_PROCESS' or state['controller_sha256']!=reference.sha256:
            raise ControlError('ORIGINAL_TERMINAL_RESTART_REQUIRED')
        if observe_native_identity(NativeIdentity.parse(state['owner'])) not in (
            NativeState.MISSING,NativeState.EXITED,NativeState.REPLACED):
            raise ControlError('PREVIOUS_OWNER_NOT_CONFIRMED_GONE')
        witness = RiskWitness(load_witness(Reference(**state['registration'])),settings,clock=clock)
        witness.capture_terminal_restart()
        trace = witness.export_trace(folder/'restart-trace.json')
        result = export_captured_case_proof(settings,witness.registration.reference,trace,folder/'proof.json',clock=clock)
        state.update(status='EXPORTED' if result.success else 'NOT_PROVEN',owner=current_native_identity().payload(),
                     trace=trace.model_dump(mode='json'),proof=None if result.reference is None else result.reference.model_dump(mode='json'))
        write_state(folder/'state.json',state)
        return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','status','stop','run-slot','manage-slot','replay-slot'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--control',type=Path)
    parser.add_argument('--control-sha256')
    parser.add_argument('--approval',type=Path)
    parser.add_argument('--approval-sha256')
    parser.add_argument('--slot',type=int)
    args = parser.parse_args()
    try:
        if args.command=='prepare':
            if args.config is None or args.directory is None:
                raise ControlError('PREPARATION_ARGUMENTS_REQUIRED')
            ref = prepare(load_settings(args.config),args.directory)
            result: dict[str,Any] = {'status':'PREPARED_NOT_AUTHORIZED','control':ref.model_dump(mode='json')}
        else:
            ref = Reference(path=args.control,sha256=args.control_sha256)
            if args.command=='status':
                result = status(ref)
            elif args.command=='stop':
                result = stop(ref)
            elif args.command=='replay-slot':
                state = replay_slot(ref,args.slot)
                result = {'status':state['status'],'slot':state['slot'],'normal_trading_allowed':False}
            else:
                approved = None if args.approval is None else Reference(path=args.approval,sha256=args.approval_sha256)
                state = run_slot(ref,args.slot,approval=approved,manage_existing=args.command=='manage-slot')
                result = {'status':state['status'],'slot':state['slot'],'normal_trading_allowed':False}
        print(exact_json(result))
        return 0
    except Exception as error:
        print(exact_json({'status':'CONTROL_REJECTED','code':str(error) if isinstance(error,ControlError) else type(error).__name__,
                          'normal_trading_allowed':False}))
        return 1


@contextmanager
def guarded_entries(bot: TestnetBot,blocked: Callable[[],bool]) -> Iterator[None]:
    originals = []
    for client in (bot.spot,bot.perp):
        original = client.place_ioc_limit_order
        def place(order: IocLimitOrder,original: Callable[[IocLimitOrder],OrderReceipt] = original) -> OrderReceipt:
            entry = not order.reduce_only and (order.market is ExecutionMarket.SPOT and order.side is Side.BUY
                                               or order.market is ExecutionMarket.USDTM and order.side is Side.SELL)
            if entry and blocked():
                raise TestnetExecutionError(ExecutionErrorKind.ORDER_GUARD_REJECTED,'controller blocks new entry')
            return original(order)
        originals.append((client,original))
        client.place_ioc_limit_order = place  # type: ignore[method-assign]
    try:
        yield
    finally:
        for client,original in originals:
            client.place_ioc_limit_order = original  # type: ignore[method-assign]


def capture_session(reference: Reference,plan: dict[str,Any],item: dict[str,Any],folder: Path,
                    state: dict[str,Any],bot: TestnetBot,*,clock: Callable[[],datetime],
                    wait: Callable[[float],None],timer_deadline: int,timer_start: int,
                    manage_existing: bool = False) -> dict[str,Any]:
    deadline = datetime.fromisoformat(str(state['deadline'])) if isinstance(state['deadline'],str) else state['deadline']
    monotonic_start = timer_start
    used_ns = state['elapsed_ns']
    minimum_clock = datetime.fromisoformat(state['updated_at']) if isinstance(state['updated_at'],str) else state['updated_at']
    def stopped() -> bool:
        return (reference.path.parent/'stop').exists()
    def expired() -> bool:
        return utc_timestamp(clock())>=deadline or utc_timestamp(clock())<minimum_clock or monotonic_ns()>=timer_deadline
    def persist() -> None:
        state.update(elapsed_ns=used_ns+monotonic_ns()-monotonic_start,updated_at=utc_timestamp(clock()))
        write_state(folder/'state.json',state)
    def manage_without_capture() -> dict[str,Any]:
        # Losing witness provenance must never grant a replacement entry/proof.
        # Only original persisted exposure can be reduced within the old cap.
        with guarded_entries(bot,lambda:True):
            while not expired() and attempt_state(bot.config.journal_path,state['attempt_id']) not in (
                None,'CLOSED','FLATTENED','FAILED'):
                bot.manage_existing_once()
                state['status']='MANAGING_WITHOUT_VALID_CAPTURE'
                persist()
                wait(bot.settings.poll_interval_seconds)
        state['status']='NOT_PROVEN' if attempt_state(bot.config.journal_path,state['attempt_id']) in (
            None,'CLOSED','FLATTENED','FAILED') else 'NEEDS_ATTENTION'
        state['capture_failed']=True
        persist()
        return state
    if manage_existing:
        return manage_without_capture()
    acceptance = FundingAcceptance(bot,folder/'acceptance.json',symbol=item['symbol'],max_wait_seconds=28800)
    witness = None
    if not acceptance.state:
        if stopped() or expired():
            state.update(status='CANCELLED')
            write_state(folder/'state.json',state)
            return state
        try:
            acceptance.tick()  # Original arming, exact journal budget and flat checks.
        except Exception as error:
            acceptance._handle_failure(error)
        if acceptance.state.get('status')!='WAITING':
            state.update(status='NOT_PROVEN')
            write_state(folder/'state.json',state)
            return state
        state['attempt_id']=str(acceptance.state['attempt_id'])
        write_state(folder/'state.json',state)
        witness = RiskWitness.register(folder/'witness',bot.settings,bot.spot,bot.perp,
            attempt_id=state['attempt_id'],symbol=item['symbol'],case=ExecutionRiskCase(item['case']),
            duration=deadline-utc_timestamp(clock()),clock=clock)
        acceptance._save('WAITING',witness_registration=witness.registration.reference.model_dump(mode='json'))
    else:
        # A partially saved report cannot silently become a second arming/entry.
        if acceptance.state.get('attempt_id')!=state['attempt_id']:
            raise ControlError('CLAIM_REPORT_IDENTITY_DIFFERS')
        registered = acceptance.state.get('witness_registration')
        if not isinstance(registered,dict):
            return manage_without_capture()
        try:
            witness_reference = Reference(**registered)
            namespace = hashlib.sha256((str(state['attempt_id'])+':'+item['case']).encode()).hexdigest()
            expected = folder/'witness'/namespace/'registration.json'
            if (witness_reference.path.resolve()!=expected.resolve() or any(
                    path.is_symlink() or path.is_junction() for path in (expected,*expected.parents)
                    if path.is_relative_to(folder))):
                return manage_without_capture()
            witness = RiskWitness(load_witness(witness_reference),bot.settings,clock=clock)
            if (witness.registration.case.value!=item['case']
                    or witness.registration.attempt_id!=state['attempt_id']
                    or witness.registration.symbol!=item['symbol']):
                return manage_without_capture()
        except Exception:
            return manage_without_capture()
    observer = witness.observe(bot.spot,bot.perp)
    try:
        observer.__enter__()
    except Exception:
        return manage_without_capture()
    try:
      with guarded_entries(bot,lambda:stopped() or expired()):
        while not expired():
            current = attempt_state(bot.config.journal_path,state['attempt_id'])
            if stopped() and current is None:
                state['status']='CANCELLED'
                break
            if current in ('CLOSED','FLATTENED','FAILED'):
                reconcile_balance_ledger(bot)
                state['status']='CAPTURED'
                break
            if acceptance.state.get('status') in ('ABORTED','NEEDS_ATTENTION'):
                if current is None:
                    state['status']='NOT_PROVEN'
                    break
                bot.manage_existing_once()  # Original unknown/partial/exit rules; no entries.
            else:
                try:
                    acceptance.tick()
                except Exception as error:
                    acceptance._handle_failure(error)
            state.update(status='CAPTURING')
            persist()
            wait(bot.settings.poll_interval_seconds)
        else:
            state['status']='NEEDS_ATTENTION' if attempt_state(bot.config.journal_path,state['attempt_id']) not in (
                None,'CLOSED','FLATTENED','FAILED') else 'NOT_PROVEN'
    except BaseException as error:
        observer.__exit__(type(error),error,error.__traceback__)
        raise
    else:
        observer.__exit__(None,None,None)
    trace_path = folder/f"trace-{state['owner']['pid']}-{state['owner']['creation_filetime']}.json"
    try:
        trace = witness.export_trace(trace_path)
    except Exception:
        state['capture_failed']=True
        state['status']='NOT_PROVEN' if attempt_state(bot.config.journal_path,state['attempt_id']) in (
            None,'CLOSED','FLATTENED','FAILED') else 'NEEDS_ATTENTION'
        persist()
        return state
    state.update(registration=witness.registration.reference.model_dump(mode='json'),
                 trace=trace.model_dump(mode='json'),updated_at=utc_timestamp(clock()))
    if item['case']==ExecutionRiskCase.RESTART_IDEMPOTENCY.value and state['status']=='CAPTURED':
        state['status']='RESTART_PENDING_NEW_PROCESS'
    elif state['status']=='CAPTURED':
        result = export_captured_case_proof(bot.settings,witness.registration.reference,trace,folder/'proof.json',clock=clock)
        state['status']='EXPORTED' if result.success else 'NOT_PROVEN'
        state['proof']=None if result.reference is None else result.reference.model_dump(mode='json')
        state['proof_failure']=None if result.failure is None else result.failure.value
    persist()
    return state


if __name__=='__main__':
    raise SystemExit(main())
