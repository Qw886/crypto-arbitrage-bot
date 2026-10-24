"""One bounded native Runner balance-evidence acceptance using virtual funds.

No monkeypatch of entry planning. The existing controlled connectivity entry
bypasses opportunity selection explicitly, never the financial safety gates.
This is execution evidence, not a profitable strategy sample.
"""

import argparse
import json
import os
from decimal import Decimal
from pathlib import Path

from restart_acceptance import save
from testnet_continuous_acceptance import run_acceptance, safe_error

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.balance_audit_repository import BalanceAuditRepository
from crypto_arbitrage_bot.execution.balance_check_repository import BalanceCheckRepository
from crypto_arbitrage_bot.execution.credential_store import (
    DEFAULT_CREDENTIAL_FILE,
    execution_environment,
)
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError, create_testnet_bot
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.logging import configure_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    outcome = args.report.with_suffix('.balance.json')
    if args.report.exists() or outcome.exists():
        raise ExecutionGuardError('NATIVE_ACCEPTANCE_REPORT_ALREADY_EXISTS')
    result = {'scope': 'TESTNET_NATIVE_BALANCE_EVIDENCE', 'orders_are_virtual': True,
              'native_entry_planning': True, 'new_entry_limit': 1,
              'meaning': 'Virtual execution and currency accounting, not profitability'}
    try:
        if os.environ.get('CAB_TESTNET_ORDERS_ENABLED') != 'true':
            raise ExecutionGuardError('TESTNET_ORDERS_DISABLED')
        settings = load_settings(Path('docs/testnet-balance-research.toml'))
        configure_logging(Path('logs/testnet/native-balance-acceptance'), settings.log_level)
        bot = create_testnet_bot(settings, environ=execution_environment(DEFAULT_CREDENTIAL_FILE, os.environ))
        execution_code = run_acceptance(bot, args.report, duration_seconds=120, entry_symbol='ETHUSDT')
        evidence = json.loads(args.report.read_text(encoding='utf-8'))
        attempt = evidence.get('attempt_id')
        result.update(execution_status=evidence['status'], attempt_id=attempt)
        check = None if attempt is None else BalanceCheckRepository(bot.journal).get(attempt)
        result['balance_check'] = check
        audit = None if check is None or check.audit_id is None else BalanceAuditRepository(bot.journal).get(check.audit_id)
        result['balance_audit'] = audit
        result['status'] = ('PASS' if execution_code == 0 and check is not None
                            and check.status.value == 'ARCHIVED' and audit is not None
                            and audit['status'] in ('MATCHED', 'INCOMPLETE')
                            and bool(audit['movements'])
                            and all(i['code'] == 'FUNDING_EVIDENCE_INCOMPLETE' for i in audit['issues'])
                            and all(m['unexplained_delta'] is not None
                                    and Decimal(m['unexplained_delta']) == 0 for m in audit['movements'])
                            else 'NEEDS_ATTENTION')
    except Exception as error:
        result.update(status='NEEDS_ATTENTION', **safe_error(error))
    save(outcome, result)
    print(exact_json({'status': result['status'], 'execution_status': result.get('execution_status'),
                     'balance_check': result.get('balance_check'), 'report': str(outcome),
                     'reason': result.get('reason')}), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
