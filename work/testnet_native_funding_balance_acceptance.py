"""One durable virtual Funding settlement plus original full-wallet evidence.

Uses existing FundingAcceptance and native balance ledger. No new financial
formula, trade selection, private production endpoint or submission retry.
"""

import argparse
import json
import os
from pathlib import Path
from time import monotonic

from restart_acceptance import save
from testnet_continuous_acceptance import safe_error

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.balance_audit_repository import BalanceAuditRepository
from crypto_arbitrage_bot.execution.balance_check_repository import BalanceCheckRepository
from crypto_arbitrage_bot.execution.balance_ledger import reconcile_balance_ledger
from crypto_arbitrage_bot.execution.credential_store import (
    DEFAULT_CREDENTIAL_FILE,
    execution_environment,
)
from crypto_arbitrage_bot.execution.funding_acceptance import FundingAcceptance
from crypto_arbitrage_bot.execution.reconciliation_repository import ReconciliationRepository
from crypto_arbitrage_bot.execution.runner import (
    ExecutionGuardError,
    create_testnet_bot,
    process_lock,
)
from crypto_arbitrage_bot.execution.trade_reconciliation import reconcile_closed_attempt
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.logging import configure_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    outcome = args.report.with_suffix('.balance.json')
    if outcome.exists():
        raise ExecutionGuardError('FUNDING_BALANCE_OUTCOME_ALREADY_EXISTS')
    result = {'scope': 'TESTNET_FUNDING_FULL_WALLET_EVIDENCE', 'orders_are_virtual': True,
              'new_entry_limit': 1, 'native_entry_planning': True,
              'meaning': 'Execution and test-account evidence, not strategy profitability'}
    try:
        if os.environ.get('CAB_TESTNET_ORDERS_ENABLED') != 'true':
            raise ExecutionGuardError('TESTNET_ORDERS_DISABLED')
        settings = load_settings(Path('docs/testnet-balance-research.toml'))
        configure_logging(Path('logs/testnet/native-funding-balance'), settings.log_level)
        bot = create_testnet_bot(settings, environ=execution_environment(DEFAULT_CREDENTIAL_FILE, os.environ))
        # Original report/journal persist identity, retries and exact settlement.
        acceptance = FundingAcceptance(bot, args.report, symbol='ETHUSDT',
            max_wait_seconds=14400, max_preflight_attempts=3, preflight_retry_seconds=15)
        with process_lock(args.report.with_suffix('.supervisor.lock')):
            status = acceptance.run()
            result['funding_acceptance_status'] = status
            evidence = json.loads(args.report.read_text(encoding='utf-8'))
            attempt = evidence.get('attempt_id')
            result['attempt_id'] = attempt
            if status not in ('VERIFIED', 'UNAVAILABLE') or attempt is None:
                result.update(status='NEEDS_ATTENTION', reason='SETTLEMENT_NOT_ACCEPTED')
            else:
                # No run_once(), research scan or new entry in the followup.
                deadline = monotonic() + 240
                check = None
                while not bot.stop.is_set() and monotonic() < deadline:
                    with process_lock(bot.config.journal_path.with_suffix('.lock')):
                        reconcile_balance_ledger(bot)
                    check = BalanceCheckRepository(bot.journal).get(attempt)
                    if check is not None and check.status.value in ('ARCHIVED', 'FAILED'):
                        break
                    if bot.stop.wait(bot.config.balance_check_interval_seconds):
                        break
                result['balance_check'] = check
                audit = (None if check is None or check.audit_id is None
                         else BalanceAuditRepository(bot.journal).get(check.audit_id))
                result['balance_audit'] = audit
                report = reconcile_closed_attempt(bot, attempt)
                result['trade_accounting'] = report
                result['trade_archive'] = ReconciliationRepository(bot.journal).save(report)
                # MATCHED means event/wallet agreement. Capital/NetReturn remain
                # unknown without actual allocation evidence; never infer them.
                result['status'] = ('PASS' if status == 'VERIFIED' and audit is not None
                                    and audit['status'] == 'MATCHED' and report.status.value == 'RECONCILED'
                                    and result['trade_archive'].success else 'NEEDS_ATTENTION')
    except Exception as error:
        result.update(status='NEEDS_ATTENTION', **safe_error(error))
    save(outcome, result)
    print(exact_json({'status': result['status'], 'attempt_id': result.get('attempt_id'),
                     'funding_acceptance_status': result.get('funding_acceptance_status'),
                     'balance_check': result.get('balance_check'), 'reason': result.get('reason'),
                     'report': str(outcome)}), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
