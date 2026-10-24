"""One bounded ETH Demo pair with durable pre/post wallet evidence.

Virtual connectivity/accounting sample; not a strategy opportunity or profit
proof. Uses the existing guarded 120-second continuous acceptance, one entry
only, and the unchanged original managed exit. Do not run beside another bot.
"""

import argparse
import os
from pathlib import Path

from restart_acceptance import save
from testnet_continuous_acceptance import run_acceptance, safe_error

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.balance_accounting import (
    plan_with_balance_evidence,
    reconcile_balance_evidence,
)
from crypto_arbitrage_bot.execution.credential_store import (
    DEFAULT_CREDENTIAL_FILE,
    execution_environment,
)
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError, create_testnet_bot
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.logging import configure_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix('.balance.json').exists():
        raise ValueError('existing acceptance must not be repeated')
    if os.environ.get('CAB_TESTNET_ORDERS_ENABLED') != 'true':
        raise ExecutionGuardError('TESTNET_ORDERS_DISABLED')
    settings = load_settings(Path('docs/testnet-funding-acceptance.toml'))
    configure_logging(Path('logs/testnet/balance-acceptance'), settings.log_level)
    bot = create_testnet_bot(settings, environ=execution_environment(DEFAULT_CREDENTIAL_FILE, os.environ))
    before_directory = args.report.with_suffix('.before-balances')
    original_plan = bot._plan

    class PlanningView:
        def __getattr__(self, name):
            return getattr(bot, name)

        def _plan(self, *a, **kw):
            return original_plan(*a, **kw)

    def captured_plan(symbol, quantity, attempt_id, **kwargs):
        if kwargs:
            raise ValueError('controlled acceptance cannot consume a strategy opportunity')
        return plan_with_balance_evidence(PlanningView(), before_directory, symbol, quantity, attempt_id)

    bot._plan = captured_plan
    result = {'scope': 'TESTNET_FULL_WINDOW_BALANCE_EVIDENCE_NOT_PROFIT', 'orders_are_virtual': True}
    try:
        execution_code = run_acceptance(bot, args.report, duration_seconds=120, entry_symbol='ETHUSDT')
        bot._plan = original_plan
        import json
        evidence = json.loads(args.report.read_text(encoding='utf-8'))
        result['execution_status'] = evidence['status']
        result['attempt_id'] = evidence.get('attempt_id')
        if execution_code == 0 and result['attempt_id']:
            result['balance_followup'] = reconcile_balance_evidence(bot, before_directory, result['attempt_id'])
        else:
            result['balance_followup'] = None
            result['error_code'] = 'EXECUTION_NOT_ACCEPTED'
    except Exception as error:
        result.update(**safe_error(error), error_code='BALANCE_ACCEPTANCE_FAILED')
    finally:
        bot._plan = original_plan
    outcome = args.report.with_suffix('.balance.json')
    save(outcome, result)
    print(exact_json({'execution_status': result.get('execution_status'),
                     'balance_followup': result.get('balance_followup'), 'report': str(outcome),
                     'error_code': result.get('error_code')}), flush=True)


if __name__ == '__main__':
    main()
