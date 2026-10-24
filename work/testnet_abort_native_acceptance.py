"""One controlled TESTNET ACK interruption and native abort accounting proof.

This is a local fault-injection acceptance, not an actual upstream timeout or
strategy trade. Passive IOC is the default. Explicit bounded_marketable mode
uses the original configured entry limit to observe filled-inventory recovery;
a zero fill in that mode is CASE_NOT_OBSERVED, never proof of a filled abort.
The existing unhedged recovery must safely flatten fills. No Perp entry is sent.
Original planning, account/quantity/notional/price guards and recovery budgets
remain in force. Resume queries the same durable ID and never sends an entry.
"""

import argparse
import json
import os
import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from restart_acceptance import save

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.abort_balance_proof import (
    ABORT_SCOPE,
    AbortBalanceProofRepository,
    supported_abort,
)
from crypto_arbitrage_bot.execution.balance_audit_repository import BalanceAuditRepository
from crypto_arbitrage_bot.execution.balance_check_repository import BalanceCheckRepository
from crypto_arbitrage_bot.execution.balance_ledger import balance_entry_blocker, reconcile_balance_ledger
from crypto_arbitrage_bot.execution.credential_store import DEFAULT_CREDENTIAL_FILE, execution_environment
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState
from crypto_arbitrage_bot.execution.reconciliation_repository import ReconciliationRepository
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError, create_testnet_bot, process_lock
from crypto_arbitrage_bot.execution.types import ExecutionMarket
from crypto_arbitrage_bot.models.calculation import TradeLeg
from crypto_arbitrage_bot.models.market import SUPPORTED_SYMBOLS
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.logging import configure_logging


class LocalAckInterruption(RuntimeError):
    """Deliberate local boundary failure, never called a Binance/network error."""


def interrupted_gateway(bot, plan, ack_path):
    original=bot.spot.place_ioc_limit_order
    calls=[0]
    def submit(order):
        if order != plan.spot_entry or calls[0] != 0:
            raise ExecutionGuardError('ACCEPTANCE_ENTRY_REPLAY_FORBIDDEN')
        calls[0]+=1
        receipt=original(order)
        save(ack_path,{'scope':'TESTNET_ORIGINAL_ACK_BEFORE_LOCAL_INTERRUPTION',
            'at':bot.clock(),'receipt':receipt,'meaning':'Actual TESTNET response, preserved unmodified.'})
        raise LocalAckInterruption('INTENTIONAL_LOCAL_ACK_INTERRUPTION')
    return original,submit,calls


def start_once(bot, report, state):
    mode=state.get('entry_mode','passive')
    if mode not in ('passive','bounded_marketable'):
        raise ValueError('explicit bounded acceptance entry mode required')
    if state['phase'] != 'PREFLIGHT':
        raise ExecutionGuardError('ACCEPTANCE_ENTRY_REPLAY_FORBIDDEN')
    for symbol in SUPPORTED_SYMBOLS:
        account=bot.perp.get_account_state(symbol)
        if (not account.can_trade or account.open_order_count or account.position_quantity != 0
                or not account.one_way or not account.single_asset or account.leverage != 1):
            raise ExecutionGuardError('TESTNET_ACCOUNT_NOT_ALREADY_FLAT_1X')
    quantity=bot.research.sizing['ETHUSDT'].target_quantity
    plan=bot._plan('ETHUSDT',quantity,state['attempt_id'])
    if mode=='passive':
        assembly=bot.collector.collect(plan.symbol)
        bot._require_quality(assembly,'INVALID_TESTNET_PASSIVE_PROBE_DATA')
        rules=bot.spot.get_symbol_rules(plan.symbol)
        # Reuse the existing bounded bid-side price function for a passive BUY.
        price=bot._limit(assembly,TradeLeg.SPOT_EXIT,quantity,rules)
        if assembly.snapshot is None or price >= assembly.snapshot.spot_book.best_ask:
            raise ExecutionGuardError('PASSIVE_PROBE_WOULD_CROSS_ASK')
        plan=replace(plan,spot_entry=replace(plan.spot_entry,limit_price=price))
        bot.spot._validate_order(plan.spot_entry,rules)
        bot._require_quality(assembly,'PASSIVE_PROBE_MARKET_EXPIRED')
    # Mark receipt time and fill case explicitly. No marketable price is
    # invented: _plan's current book, price cap and quantity guards stay intact.
    state.update(phase='OPEN_INTENT',plan=plan,started_at=bot.clock(),
                 entry_mode=mode,probe=mode.upper()+'_IOC_WITH_INTENTIONAL_LOCAL_ACK_INTERRUPTION')
    save(report,state) # Durable phase precedes the one entry call.
    original,submit,calls=interrupted_gateway(bot,plan,report.with_suffix('.ack.json'))
    bot.spot.place_ioc_limit_order=submit
    try:
        result=bot.coordinator.open_pair(plan)
    finally:
        bot.spot.place_ioc_limit_order=original
    state.update(phase='RECOVERING',initial_pair=result,local_gateway_entry_calls=calls[0])
    save(report,state)
    return plan


def outcome(bot, plan):
    pair=bot.journal.get(plan.attempt_id)
    check=BalanceCheckRepository(bot.journal).get(plan.attempt_id)
    audit=None if check is None or check.audit_id is None else BalanceAuditRepository(bot.journal).get(check.audit_id)
    marker=next((i for i in BalanceCheckRepository(bot.journal).before_intents() if i.attempt_id==plan.attempt_id),None)
    proof=None if marker is None else AbortBalanceProofRepository(bot.journal).get_verified(plan,marker.frame_sha256)
    accounting=ReconciliationRepository(bot.journal).get_report(plan.attempt_id)
    intents=bot.journal.intents(plan.attempt_id)
    valid=(supported_abort(pair) and check is not None and check.status.value=='ARCHIVED' and proof is not None
        and audit is not None and audit['status']=='MATCHED' and audit['issues']==[] and bool(audit['movements'])
        and all(m['unexplained_delta'] is not None and Decimal(m['unexplained_delta'])==0 for m in audit['movements'])
        and accounting is not None and accounting['accounting_scope']==ABORT_SCOPE
        and accounting['calculation'] is None and accounting['net_return'] is None
        and not any(order.market is ExecutionMarket.USDTM for order in intents)
        and balance_entry_blocker(bot) is None)
    return {'status':'PASS' if valid else 'NEEDS_ATTENTION','pair':pair,'balance_check':check,
        'native_currency_audit':audit,'abort_proof':proof,'accounting':accounting,
        'perp_intent_count':sum(order.market is ExecutionMarket.USDTM for order in intents),
        'meaning':'Controlled local ACK interruption and native currency proof; not a strategy PASS, closed arbitrage trade, or profit.'}


def classify_result(result, mode):
    """Keep safe zero-fill abort distinct from observed inventory restoration."""
    output=dict(result)
    entry=getattr(result.get('pair'),'spot_entry',None)
    executed=getattr(entry,'executed_quantity',None)
    original=getattr(entry,'original_quantity',None)
    case='UNKNOWN'
    if (isinstance(executed,Decimal) and isinstance(original,Decimal)
            and executed.is_finite() and original.is_finite() and original>0 and 0<=executed<=original):
        case='NO_FILL' if executed==0 else 'FULL_FILL' if executed==original else 'PARTIAL_FILL'
    output.update(entry_mode=mode,observed_entry_case=case)
    if result['status']=='PASS' and mode=='bounded_marketable' and case in ('UNKNOWN','NO_FILL'):
        output['status']='CASE_NOT_OBSERVED' if case=='NO_FILL' else 'NEEDS_ATTENTION'
    return output


def run(bot, report, *, duration_seconds=180, wait=time.sleep, monotonic=time.monotonic,entry_mode=None):
    if type(duration_seconds) is not int or not 1 <= duration_seconds <= 300:
        raise ValueError('bounded acceptance duration required')
    if entry_mode not in (None,'passive','bounded_marketable'):
        raise ValueError('explicit bounded acceptance entry mode required')
    if report.exists():
        state=json.loads(report.read_text(encoding='utf-8'))
        if (state.get('scope')!='TESTNET_CONTROLLED_ACK_INTERRUPTION_NATIVE_ABORT' or
                not isinstance(state.get('attempt_id'),str) or state.get('symbol')!='ETHUSDT'):
            raise ExecutionGuardError('INVALID_ABORT_ACCEPTANCE_REPORT')
        saved_mode=state.get('entry_mode','passive')
        if saved_mode not in ('passive','bounded_marketable'):
            raise ExecutionGuardError('INVALID_ABORT_ACCEPTANCE_REPORT')
        if entry_mode is not None and entry_mode!=saved_mode:
            raise ExecutionGuardError('ABORT_ACCEPTANCE_MODE_MISMATCH')
        # A preflight report cannot prove that no hidden submission occurred.
        # All restarts are recovery-only, even if planning never finished.
        plans=[p for p,_ in bot.journal.plans() if p.attempt_id==state['attempt_id']]
        if len(plans)!=1:
            raise ExecutionGuardError('NO_DURABLE_ABORT_PLAN_RECOVERY_ONLY')
        plan=plans[0]
    else:
        state={'scope':'TESTNET_CONTROLLED_ACK_INTERRUPTION_NATIVE_ABORT','symbol':'ETHUSDT',
            'attempt_id':uuid4().hex,'phase':'PREFLIGHT','new_entry_limit':1,'orders_are_virtual':True,
            'entry_mode':entry_mode or 'passive',
            'meaning':'Explicit controlled local failure acceptance, not a profitable strategy or actual network failure.'}
        save(report,state)
        plan=start_once(bot,report,state)
    started=monotonic()
    for _ in range(20):
        if bot.stop.is_set() or monotonic()-started >= duration_seconds:
            break
        bot._managed_exits()
        reconcile_balance_ledger(bot)
        result=classify_result(outcome(bot,plan),state.get('entry_mode','passive'))
        state.update(phase=result['status'] if result['status'] in ('PASS','CASE_NOT_OBSERVED') else
                     'RECOVERING',last_checked_at=bot.clock())
        save(report,state)
        save(report.with_suffix('.balance.json'),result)
        if result['status'] in ('PASS','CASE_NOT_OBSERVED'):
            return result
        check=result['balance_check']
        if check is not None and check.status.value=='FAILED':
            break
        remaining=duration_seconds-(monotonic()-started)
        if remaining>0:
            wait(min(10,remaining))
    state.update(phase='NEEDS_ATTENTION',ended_at=bot.clock())
    save(report,state)
    result=classify_result(outcome(bot,plan),state.get('entry_mode','passive'))
    save(report.with_suffix('.balance.json'),result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True,type=Path)
    parser.add_argument('--entry-mode',choices=('passive','bounded_marketable'))
    args=parser.parse_args()
    try:
        if os.environ.get('CAB_TESTNET_ORDERS_ENABLED')!='true':
            raise ExecutionGuardError('TESTNET_ORDERS_DISABLED')
        settings=load_settings(Path('docs/testnet-balance-research.toml'))
        configure_logging(Path('logs/testnet/native-abort-acceptance'),settings.log_level)
        bot=create_testnet_bot(settings,environ=execution_environment(DEFAULT_CREDENTIAL_FILE,os.environ))
        with process_lock(bot.config.journal_path.with_suffix('.lock')):
            result=run(bot,args.report,entry_mode=args.entry_mode)
        print(exact_json({'status':result['status'],'pair_state':None if result['pair'] is None else result['pair'].state,
            'perp_intent_count':result['perp_intent_count'],'observed_entry_case':result['observed_entry_case'],
            'entry_mode':result['entry_mode'],'report':str(args.report)}),flush=True)
        return 0 if result['status']=='PASS' else 1
    except Exception as error:
        result={'status':'NEEDS_ATTENTION','at':utc_now(),'error_type':type(error).__name__,
                'meaning':'No entry retry, no fake proof.'}
        if hasattr(error,'diagnostic_fields'): result['diagnostics']=error.diagnostic_fields()
        elif isinstance(error,ExecutionGuardError): result.update(reason=error.reason,diagnostics=error.diagnostics)
        save(args.report.with_suffix('.error.json'),result)
        print(exact_json(result),flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
