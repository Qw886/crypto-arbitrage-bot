"""One original Funding acceptance, bounded read-only native balance followup."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import testnet_native_funding_balance_acceptance as tool

from crypto_arbitrage_bot.execution.runner import ExecutionGuardError


def test_completed_outcome_never_repeats_orders(tmp_path, monkeypatch):
    report = tmp_path / 'run.json'
    report.with_suffix('.balance.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['tool', '--report', str(report)])
    monkeypatch.setattr(tool, 'load_settings', lambda *a: pytest.fail('no further work'))
    with pytest.raises(ExecutionGuardError, match='ALREADY_EXISTS'):
        tool.main()


def test_disabled_has_no_account_or_key_reads(tmp_path, monkeypatch):
    report = tmp_path / 'run.json'
    monkeypatch.setattr('sys.argv', ['tool', '--report', str(report)])
    monkeypatch.delenv('CAB_TESTNET_ORDERS_ENABLED', raising=False)
    monkeypatch.setattr(tool, 'load_settings', lambda *a: pytest.fail('disabled'))
    assert tool.main() == 1
    assert json.loads(report.with_suffix('.balance.json').read_text(encoding='utf-8'))['reason'] == 'TESTNET_ORDERS_DISABLED'


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    path = tmp_path / 'funding.json'
    calls, outcomes = [], []
    stop = SimpleNamespace(is_set=lambda: False, wait=lambda seconds: calls.append(('wait', seconds)) or False)
    bot = SimpleNamespace(journal=object(), stop=stop,
        config=SimpleNamespace(journal_path=tmp_path / 'journal.sqlite', balance_check_interval_seconds=60))
    monkeypatch.setattr('sys.argv', ['tool', '--report', str(path)])
    monkeypatch.setenv('CAB_TESTNET_ORDERS_ENABLED', 'true')
    monkeypatch.setattr(tool, 'load_settings', lambda *a: SimpleNamespace(log_level='INFO'))
    monkeypatch.setattr(tool, 'configure_logging', lambda *a: None)
    monkeypatch.setattr(tool, 'execution_environment', lambda *a: {})
    monkeypatch.setattr(tool, 'create_testnet_bot', lambda *a, **kw: bot)
    monkeypatch.setattr(tool, 'process_lock', lambda *a: nullcontext())
    monkeypatch.setattr(tool, 'reconcile_balance_ledger', lambda b: calls.append('readonly balance'))
    monkeypatch.setattr(tool, 'reconcile_closed_attempt', lambda *a: SimpleNamespace(status=SimpleNamespace(value='RECONCILED')))
    monkeypatch.setattr(tool, 'ReconciliationRepository', lambda *a: SimpleNamespace(save=lambda r: SimpleNamespace(success=True)))
    monkeypatch.setattr(tool, 'save', lambda p, data: outcomes.append(data))
    monkeypatch.setattr(tool, 'exact_json', lambda data: '{}')
    return path, bot, calls, outcomes


@pytest.mark.parametrize('status', ['VERIFIED', 'UNAVAILABLE', 'ABORTED'])
def test_original_settlement_called_once_with_native_followup(prepared, monkeypatch, status):
    path, bot, calls, outcomes = prepared
    def factory(current, report, **kw):
        assert current is bot and report == path
        assert kw == {'symbol': 'ETHUSDT', 'max_wait_seconds': 14400,
                      'max_preflight_attempts': 3, 'preflight_retry_seconds': 15}
        def run():
            calls.append('original acceptance')
            path.write_text(json.dumps({'attempt_id': 'original-attempt'}), encoding='utf-8')
            return status
        return SimpleNamespace(run=run)
    monkeypatch.setattr(tool, 'FundingAcceptance', factory)
    check = SimpleNamespace(status=SimpleNamespace(value='ARCHIVED'), audit_id='audit')
    monkeypatch.setattr(tool, 'BalanceCheckRepository', lambda *a: SimpleNamespace(get=lambda a: check))
    monkeypatch.setattr(tool, 'BalanceAuditRepository', lambda *a: SimpleNamespace(get=lambda a: {'status': 'MATCHED'}))
    assert tool.main() == (0 if status == 'VERIFIED' else 1)
    assert calls == (['original acceptance'] if status == 'ABORTED' else ['original acceptance', 'readonly balance'])
    assert outcomes[0]['attempt_id'] == 'original-attempt'
    assert 'net_return' not in outcomes[0]  # Never invent allocated capital.


def test_pending_followup_waits_and_stops_at_durable_failed_state(prepared, monkeypatch):
    path, _, calls, outcomes = prepared
    path.write_text(json.dumps({'attempt_id': 'original-attempt'}), encoding='utf-8')
    monkeypatch.setattr(tool, 'FundingAcceptance', lambda *a, **kw: SimpleNamespace(run=lambda: 'VERIFIED'))
    statuses = iter(['PENDING', 'PENDING', 'FAILED'])
    monkeypatch.setattr(tool, 'BalanceCheckRepository', lambda *a: SimpleNamespace(
        get=lambda a: SimpleNamespace(status=SimpleNamespace(value=next(statuses)), audit_id=None)))
    monkeypatch.setattr(tool, 'BalanceAuditRepository', Mock(side_effect=AssertionError('no complete audit')))
    assert tool.main() == 1
    assert calls == ['readonly balance', ('wait', 60), 'readonly balance', ('wait', 60), 'readonly balance']
    assert outcomes[0]['balance_audit'] is None


def test_source_error_never_embeds_unsafe_payload(prepared, monkeypatch):
    from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind
    from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as GatewayError
    _, _, _, outcomes = prepared
    error = GatewayError(ExecutionErrorKind.NETWORK_ERROR, 'secret-signed-url')
    monkeypatch.setattr(tool, 'FundingAcceptance', Mock(side_effect=error))
    assert tool.main() == 1
    assert outcomes[0]['reason'] == 'NETWORK_ERROR'
    assert 'secret-signed-url' not in str(outcomes[0])
