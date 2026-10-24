"""Disposable processes with a synthetic controller clock and offline gateways."""

import argparse
import json
import os
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import native_case_control as control

from crypto_arbitrage_bot.config.research import StudyRegistrationSettings as Reference
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.demo import DemoResponse, demo_bot
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot
from crypto_arbitrage_bot.utils.clock import utc_now

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['setup', 'attention', 'manage', 'manage-crash', 'resume'])
parser.add_argument('directory', type=Path)
parser.add_argument('--tick-controller', action='store_true')
args = parser.parse_args()
directory = args.directory.resolve()
settings = load_settings(Path('docs/testnet-funding-acceptance.toml'))
settings = settings.model_copy(update={'testnet': settings.testnet.model_copy(update={'max_total_attempts': 1})})
old, transport = demo_bot(settings, directory / 'original')
controller_start = utc_now()
moment = {'controller': controller_start, 'bot': controller_start,
          'settlement': controller_start + timedelta(seconds=300)}
wait_count = [0]
opener = transport.open


def response(request, **kwargs):
    row = opener(request, **kwargs)
    payload = json.loads(row.body)
    if '/premiumIndex?' in request.full_url:
        payload.update(lastFundingRate='0', nextFundingTime=int(moment['settlement'].timestamp() * 1000),
                       time=int(moment['bot'].timestamp() * 1000))
    if '/account?' in request.full_url:
        if 'balances' in payload:
            for asset in payload['balances']:
                asset['locked'] = '0'
        else:
            payload['assets'] = [{'asset': 'USDT', 'walletBalance': '10000'}]
    return DemoResponse(payload)


def bot_clock():
    return moment['bot']


def controller_clock():
    if args.tick_controller:
        moment['controller'] += timedelta(microseconds=1)
    return moment['controller']


def save_exchange():
    (directory / 'exchange.json').write_text(json.dumps({
        'orders': transport.orders, 'base': {key: str(value) for key, value in transport.base.items()},
        'positions': {key: str(value) for key, value in transport.positions.items()}}), encoding='utf-8')


transport.open = response
transport.clock = old.clock = old.collector.clock = old.research_runner.clock = bot_clock
for client in (old.spot, old.perp):
    client._clock_ms = lambda: int(bot_clock().timestamp() * 1000)

if args.mode == 'setup':
    assert old.round_trip('ETHUSDT', Decimal('.01')).state.value == 'CLOSED'
    reference = control.prepare(old.settings, directory / 'controller', workspace=directory)
    control.protocol.exclusive(directory / 'reference.json', reference.model_dump(mode='json'))
    control.protocol.exclusive(directory / 'approval.json', {
        'schema': 'native-case-current-authorization-v1', 'controller_sha256': reference.sha256,
        'scope': 'TESTNET_NATIVE_CASES_ONLY', 'native_case_slots': 14, 'normal_trading_allowed': False,
        'current_user_confirmed': True, 'authorization_record': 'OFFLINE_FIXTURE', 'approved_at': utc_now()})
    save_exchange()
else:
    reference = Reference(**json.loads((directory / 'reference.json').read_text()))
    approved = Reference(path=directory / 'approval.json', sha256=control.protocol.digest(directory / 'approval.json'))
    exchange = json.loads((directory / 'exchange.json').read_text())
    transport.orders = exchange['orders']
    transport.base = {key: Decimal(value) for key, value in exchange['base'].items()}
    transport.positions = {key: Decimal(value) for key, value in exchange['positions'].items()}
    folder = reference.path.parent / 'cases/07'
    if args.mode != 'attention':
        report = json.loads((folder / 'acceptance.json').read_text())
        moment['settlement'] = datetime.fromisoformat(report['settlement_at'])
        moment['bot'] = moment['settlement'] + timedelta(seconds=-1 if args.mode == 'manage-crash' else 61)

    def factory(config):
        bot = Bot(config, old.spot, old.perp, clock=bot_clock)
        bot.collector.clock = bot.research_runner.clock = bot_clock
        assert not bot.spot.uses_default_transport and not bot.perp.uses_default_transport
        return bot

    def wait(seconds):
        wait_count[0] += 1
        assert wait_count[0] <= 12, 'Offline fixture failed to progress within its own bound'
        if args.mode == 'attention':
            if len(transport.orders) == 4:
                moment['bot'] = moment['settlement'] - timedelta(seconds=180)
                moment['controller'] += timedelta(seconds=1)
            elif len(transport.orders) == 6:
                moment['controller'] = controller_start - timedelta(seconds=1)
        elif args.mode == 'manage-crash':
            save_exchange()
            os._exit(29)
        else:
            raise AssertionError('The original due exit should finish in its first management cycle')

    # The fallback waits once even after a terminal transition. Preserve that public contract.
    def management_wait(seconds):
        if args.mode == 'manage-crash' or args.mode == 'attention':
            wait(seconds)
        else:
            assert transport.positions['ETHUSDT'] == 0, 'Original exit was not completed'

    result = control.run_offline_slot(reference, 7, approval=approved, bot_factory=factory,
                                      clock=controller_clock, wait=management_wait,
                                      manage_existing=args.mode in ('manage', 'manage-crash'))
    save_exchange()
    control.protocol.exclusive(directory / (args.mode + '-result.json'), {
        'state': result, 'mock_order_count': len(transport.orders), 'base': str(transport.base['ETHUSDT']),
        'perp': str(transport.positions['ETHUSDT']), 'network_mode': 'OFFLINE_TRANSPORT_AND_PYTHON_GUARD'})
