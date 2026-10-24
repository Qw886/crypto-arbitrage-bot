"""Disposable fake venue worker, never a product or real-account entry point."""
import hashlib
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.balance_ledger import balance_entry_blocker
from crypto_arbitrage_bot.execution.demo import DemoResponse, DemoTransport
from crypto_arbitrage_bot.execution.runner import TestnetBot
from crypto_arbitrage_bot.execution.testnet_client import (
    TestnetCredentials,
    TestnetExecutionClient,
    TestnetExecutionPolicy,
)
from crypto_arbitrage_bot.execution.trade_reconciliation import reconcile_closed_attempt
from crypto_arbitrage_bot.execution.types import ExecutionMarket
from crypto_arbitrage_bot.utils.native_identity import current_native_identity


class PersistentVenue(DemoTransport):
    """Persist external venue effects before responding, including deliberate faults."""

    def __init__(self, directory, symbol, case, phase):
        super().__init__()
        self.directory, self.symbol, self.case, self.phase = directory, symbol, case, phase
        self.path = directory / 'synthetic-venue.json'
        self.events = []
        self.checkpoint = None
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding='utf-8'))
            assert data['origin'] == 'SYNTHETIC_OFFLINE'
            self.base = {key: Decimal(value) for key, value in data['base'].items()}
            self.positions = {key: Decimal(value) for key, value in data['positions'].items()}
            self.orders, self.events = data['orders'], data['events']
        else:
            assert phase == 'entry'
            self.base[symbol] = Decimal('1')  # Old inventory must never be sold.

    def save(self):
        self.path.write_text(json.dumps({'origin': 'SYNTHETIC_OFFLINE',
            'base': {key: str(value) for key, value in self.base.items()},
            'positions': {key: str(value) for key, value in self.positions.items()},
            'orders': self.orders, 'events': self.events}), encoding='utf-8')

    def open(self, fullurl, data=None, timeout=5):
        url = urlsplit(fullurl.full_url)
        method = fullurl.get_method()
        params = parse_qs((fullurl.data or b'').decode() if method == 'POST' else url.query)
        is_order = url.path.endswith('/order')
        spot = url.hostname == 'testnet.binance.vision'
        self.events.append({'method': method, 'route': url.path, 'spot': spot,
                            'cid': params.get('newClientOrderId', params.get('origClientOrderId', ['']))[0],
                            'side': params.get('side', [''])[0]})
        self.save()
        if is_order and method == 'GET' and self.phase == 'blocked':
            raise TimeoutError('SYNTHETIC_ORDER_QUERY_UNAVAILABLE')
        response = super().open(fullurl, data, timeout)
        payload = json.loads(response.body)
        if url.path.endswith('/account'):
            if spot:
                for balance in payload['balances']:
                    balance['locked'] = '0'
            else:
                payload['assets'] = [{'asset': 'USDT', 'walletBalance': '10000'}]
        if is_order and method == 'POST':
            side = params['side'][0]
            if side == 'SELL' and self.case in ('partial_exit', 'repair_ack_loss'):
                sells = [event for event in self.events if event['method'] == 'POST' and event['side'] == 'SELL']
                if len(sells) == 1:
                    quantity = Decimal(params['quantity'][0])
                    executed = quantity * Decimal('.5')
                    self.base[self.symbol] += quantity - executed
                    payload.update(status='EXPIRED', executedQty=str(executed),
                                   cummulativeQuoteQty=str(executed * Decimal(params['price'][0])))
                    payload['fills'][0]['qty'] = str(executed)
                    self.orders[params['newClientOrderId'][0]] = payload
            self.save()
            if self.phase == 'entry':
                assert self.checkpoint is not None
                self.checkpoint()
                os._exit(29)  # External effect durable, original client has not received any ACK.
            if self.phase == 'exit' and side == 'SELL':
                sells = [event for event in self.events if event['method'] == 'POST' and event['side'] == 'SELL']
                if self.case == 'exit_ack_loss' or self.case == 'repair_ack_loss' and len(sells) == 2:
                    assert self.checkpoint is not None
                    self.checkpoint()
                    os._exit(31)
        return DemoResponse(payload)


def main():
    directory = Path(sys.argv[1]).resolve()
    symbol, case, phase = sys.argv[2:]
    assert directory.is_relative_to(Path('D:/CodexValidation').resolve())
    assert symbol in ('BTCUSDT', 'ETHUSDT') and phase in ('entry', 'blocked', 'exit', 'resume')
    output = directory / f'{phase}.json'
    assert not output.exists()
    directory.mkdir(parents=True, exist_ok=True)
    settings = load_settings(Path(__file__).resolve().parents[1] / 'docs/testnet-bot.toml')
    settings = settings.model_copy(update={
        'testnet': settings.testnet.model_copy(update={'journal_path': directory / 'execution.sqlite',
            'balance_evidence_enabled': True, 'balance_evidence_directory': directory / 'frames'}),
        'research': settings.research.model_copy(update={'sqlite_path': directory / 'research.sqlite'})})
    venue = PersistentVenue(directory, symbol, case, phase)
    clients = [TestnetExecutionClient(market, TestnetCredentials('offline-key', 'offline-secret'),
        TestnetExecutionPolicy(settings.testnet.max_order_notional), opener=venue) for market in ExecutionMarket]
    bot = TestnetBot(settings, *clients)
    identity = current_native_identity().payload()
    attempt = 'native-pre-ack-' + symbol

    def checkpoint():
        result = bot.journal.get(attempt)
        frame = directory / 'frames' / (attempt + '.before.json')
        value = {'origin': 'SYNTHETIC_OFFLINE', 'identity': identity, 'state': result.state.value,
                 'orders': len(venue.orders), 'base': str(venue.base[symbol]), 'perp': str(venue.positions[symbol]),
                 'entry_frame_sha256': hashlib.sha256(frame.read_bytes()).hexdigest()}
        if phase == 'resume':
            before = len(venue.events)
            bot.manage_existing_once()
            value['terminal_extra_requests'] = len(venue.events) - before
            value['entry_blocker'] = balance_entry_blocker(bot)
            report = reconcile_closed_attempt(bot, attempt)
            value.update(accounting=report.status.value, profit=None if report.calculation is None else 'UNEXPECTED',
                entry_posts=sum(event['method'] == 'POST' and event['side'] == 'BUY' for event in venue.events),
                perp_posts=sum(event['method'] == 'POST' and not event['spot'] for event in venue.events))
        output.write_text(json.dumps(value, indent=2), encoding='utf-8')

    venue.checkpoint = checkpoint
    if phase == 'entry':
        quantity = Decimal('.002') if symbol == 'BTCUSDT' else Decimal('.01')
        bot.coordinator.open_pair(bot._plan(symbol, quantity, attempt))
        raise AssertionError('Expected actual pre-ACK process death')
    for _ in range(6):
        bot.manage_existing_once()
        if bot.journal.get(attempt).state.value == 'FLATTENED':
            break
    checkpoint()


if __name__ == '__main__':
    main()
