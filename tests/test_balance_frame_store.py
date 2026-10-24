"""Offline restart-safe pre-entry evidence; exclusively test-account GETs."""

import json
from dataclasses import replace
from decimal import Decimal

import pytest
from test_account_balance_audit import AT, snapshot
from test_testnet_execution import Opener, Response, client

from crypto_arbitrage_bot.execution.balance_frame_store import (
    BalanceFrameCode as Code,
)
from crypto_arbitrage_bot.execution.balance_frame_store import (
    BalanceFrameStore,
    BeforeBalanceFrame,
)
from crypto_arbitrage_bot.execution.types import ExecutionMarket as Market


@pytest.fixture
def frame():
    return BeforeBalanceFrame('demo-attempt', 'ETHUSDT', Decimal('.01'), (
        replace(snapshot(Market.SPOT, 0, {'USDT': '123.123456789123456789', '中文币': '2'}),
                raw_balances_json='{"asset":"中文币"}'),
        snapshot(Market.USDTM, 0, {'USDT': '1000'})))


def test_exact_utf8_roundtrip_and_restart(tmp_path, frame):
    store = BalanceFrameStore(tmp_path / 'new-directory')
    assert store.save(frame).success
    result = BalanceFrameStore(store.directory).get(frame.attempt_id, frame.symbol, frame.quantity)
    assert result.success and result.frame == frame
    assert result.frame.snapshots[0].received_at == AT
    assert result.frame.snapshots[0].balances[0].amount == Decimal('123.123456789123456789')
    assert '中文币' in (store.directory / 'demo-attempt.before.json').read_text(encoding='utf-8')


def test_same_evidence_idempotent_different_evidence_blocked(tmp_path, frame):
    store = BalanceFrameStore(tmp_path)
    assert store.save(frame).success
    path = tmp_path / 'demo-attempt.before.json'
    original = path.read_bytes()
    assert store.save(frame).success
    other = replace(frame, snapshots=(snapshot(Market.SPOT, 1, {'USDT': '999'}), frame.snapshots[1]))
    assert store.save(other).error_code is Code.EVIDENCE_CONFLICT
    assert path.read_bytes() == original


@pytest.mark.parametrize(('symbol', 'quantity'), [('BTCUSDT', '.01'), ('ETHUSDT', '.02')])
def test_wrong_symbol_or_quantity_cannot_reuse_evidence(tmp_path, frame, symbol, quantity):
    store = BalanceFrameStore(tmp_path)
    assert store.save(frame).success
    assert store.get(frame.attempt_id, symbol, Decimal(quantity)).error_code is Code.IDENTITY_MISMATCH


@pytest.mark.parametrize('identity', ['../escape', 'bad/name', '', 'a' * 65])
def test_identity_prevents_path_escape(tmp_path, identity):
    assert BalanceFrameStore(tmp_path).get(identity, 'ETHUSDT', Decimal('.01')).error_code is Code.INVALID_FRAME


def test_missing_does_not_make_before_balance_zero(tmp_path):
    assert BalanceFrameStore(tmp_path).get('unused', 'ETHUSDT', Decimal('.01')).error_code is Code.MISSING_FRAME
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('corruption', ['partial', 'version', 'float', 'duplicate-market', 'naive-time'])
def test_corrupt_frame_blocks_recapture_and_is_not_overwritten(tmp_path, frame, corruption):
    store = BalanceFrameStore(tmp_path)
    assert store.save(frame).success
    path = tmp_path / 'demo-attempt.before.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if corruption == 'version':
        data['schema_version'] = 2
    elif corruption == 'float':
        data['snapshots'][0]['balances'][0]['amount'] = 123.0
    elif corruption == 'duplicate-market':
        data['snapshots'][1]['market'] = 'SPOT'
    elif corruption == 'naive-time':
        data['snapshots'][0]['requested_at'] = '2026-10-05T00:00:00'
    path.write_text('{' if corruption == 'partial' else json.dumps(data), encoding='utf-8')
    old = path.read_bytes()
    spot, perp = client(Market.SPOT, Opener([])), client(Market.USDTM, Opener([]))
    result = store.capture_before(frame.attempt_id, frame.symbol, frame.quantity, spot, perp)
    assert result.error_code is Code.INVALID_FRAME
    assert not spot._opener.requests and not perp._opener.requests
    assert path.read_bytes() == old


def test_capture_only_gets_both_wallets_and_restart_does_not_recapture(tmp_path):
    spot = client(Market.SPOT, Opener([Response({'serverTime': 10000}), Response({
        'balances': [{'asset': 'USDT', 'free': '1', 'locked': '.1'}]})]))
    perp = client(Market.USDTM, Opener([Response({'serverTime': 10000}), Response({
        'multiAssetsMargin': False, 'assets': [{'asset': 'USDT', 'walletBalance': '2'}]})]))
    store = BalanceFrameStore(tmp_path)
    result = store.capture_before('capture', 'BTCUSDT', Decimal('.001'), spot, perp, clock=lambda: AT)
    assert result.success
    assert result.frame.snapshots[0].balances[0].amount == Decimal('1.1')
    assert len(spot._opener.requests) == len(perp._opener.requests) == 2
    assert all(r.get_method() == 'GET' for c in (spot, perp) for r, _ in c._opener.requests)
    assert BalanceFrameStore(tmp_path).capture_before('capture', 'BTCUSDT', Decimal('.001'),
                                                     spot, perp).frame == result.frame
    assert len(spot._opener.requests) == len(perp._opener.requests) == 2


def test_api_failure_does_not_write_partial_two_market_frame(tmp_path):
    spot = client(Market.SPOT, Opener([TimeoutError()]))
    perp = client(Market.USDTM, Opener([]))
    result = BalanceFrameStore(tmp_path).capture_before('capture', 'BTCUSDT', Decimal('.001'), spot, perp)
    assert result.error_code is Code.CAPTURE_FAILED and result.error_kind == 'TestnetExecutionError'
    assert not list(tmp_path.iterdir()) and not perp._opener.requests


def test_write_failure_does_not_claim_durable_frame(tmp_path, frame):
    blocked = tmp_path / 'file'
    blocked.write_text('not a directory', encoding='utf-8')
    assert BalanceFrameStore(blocked).save(frame).error_code is Code.PERSISTENCE_FAILED
    assert blocked.read_text(encoding='utf-8') == 'not a directory'


def test_existing_before_is_reused_without_account_requests(tmp_path, frame):
    store = BalanceFrameStore(tmp_path)
    assert store.save(frame).success
    spot, perp = client(Market.SPOT, Opener([])), client(Market.USDTM, Opener([]))
    assert store.capture_before(frame.attempt_id, frame.symbol, frame.quantity, spot, perp).frame == frame
    assert not spot._opener.requests and not perp._opener.requests
