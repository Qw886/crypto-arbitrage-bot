"""Two real Windows processes, explicitly OFFLINE_FIXTURE exchange simulation.

Only use a fresh isolated directory. No keys, network, actual orders or permits.
"""
import argparse
import json
import re
import socket
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution import main
from crypto_arbitrage_bot.execution.demo import demo_bot
from crypto_arbitrage_bot.execution.risk_witness import RiskWitness
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase


def forbidden(*args, **kwargs):
    raise AssertionError('OFFLINE_PROBE_NO_NETWORK_OR_CREDENTIALS')


socket.create_connection = forbidden
parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['first', 'second'])
parser.add_argument('directory', type=Path)
args = parser.parse_args()
directory = args.directory.resolve()
config = directory / 'bound-offline.toml'
if args.stage == 'first':
    directory.mkdir(parents=True, exist_ok=False)
    text = Path('docs/testnet-bot.toml').read_text(encoding='utf-8')
    text = text.replace('sqlite_path = "data/testnet/research.sqlite"', 'sqlite_path = ' + json.dumps(str(directory/'research.sqlite')))
    text = re.sub(r'(?m)^journal_path\s*=.*$', lambda match: 'journal_path = ' + json.dumps(str(directory/'execution.sqlite')), text, count=1)
    config.write_text(text, encoding='utf-8')
    bot, transport = demo_bot(load_settings(config), directory)
    witness = RiskWitness.register(directory/'witness', bot.settings, bot.spot, bot.perp,
        attempt_id='offline-real-process-restart', symbol='ETHUSDT',
        case=ExecutionRiskCase.RESTART_IDEMPOTENCY, duration=timedelta(minutes=10))
    with witness.observe(bot.spot, bot.perp):
        plan = bot._plan('ETHUSDT', Decimal('.01'), witness.registration.attempt_id)
        bot.coordinator.open_pair(plan)
        bot.coordinator.close_pair(plan)
    result = {'registration': witness.registration.reference.model_dump(mode='json'),
        'origin': witness.registration.origin, 'mock_order_count': len(transport.orders),
        'actual_exchange_orders': 0, 'credentials_read': False}
    (directory/'first-result.json').write_text(json.dumps(result), encoding='utf-8')
    print(json.dumps(result))
else:
    import sys

    result = json.loads((directory/'first-result.json').read_text(encoding='utf-8'))
    main.execution_environment = forbidden
    main.create_testnet_bot = forbidden
    sys.argv = ['bot', '--config', str(config), '--capture-terminal-restart',
        '--capture-registration', result['registration']['path'],
        '--capture-registration-sha256', result['registration']['sha256'],
        '--capture-trace-output', str(directory/'restart-trace.json')]
    raise SystemExit(main.main())
