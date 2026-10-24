"""Bounded orchestration harness tests; no credentials or HTTP."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import testnet_balance_acceptance as tool

from crypto_arbitrage_bot.execution.runner import ExecutionGuardError


def test_disabled_never_loads_keys_or_accounts(tmp_path, monkeypatch):
    monkeypatch.delenv('CAB_TESTNET_ORDERS_ENABLED', raising=False)
    monkeypatch.setattr('sys.argv', ['tool', '--report', str(tmp_path / 'run.json')])
    monkeypatch.setattr(tool, 'load_settings', lambda *a: pytest.fail('disabled'))
    with pytest.raises(ExecutionGuardError, match='TESTNET_ORDERS_DISABLED'):
        tool.main()


def test_existing_report_not_repeated(tmp_path, monkeypatch):
    report = tmp_path / 'existing.json'
    report.write_text('{}', encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['tool', '--report', str(report)])
    with pytest.raises(ValueError, match='must not be repeated'):
        tool.main()


def test_failed_preflight_evidence_not_overwritten(tmp_path, monkeypatch):
    report = tmp_path / 'existing.json'
    report.with_suffix('.balance.json').write_text('{"error_code":"PRIOR_FAILURE"}', encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['tool', '--report', str(report)])
    with pytest.raises(ValueError, match='must not be repeated'):
        tool.main()


def test_one_bounded_existing_acceptance_then_account_followup(tmp_path, monkeypatch):
    report = tmp_path / 'run.json'
    calls = []
    def original_plan(*a, **kw):
        return 'existing guarded plan'
    bot = SimpleNamespace(_plan=original_plan)
    monkeypatch.setattr('sys.argv', ['tool', '--report', str(report)])
    monkeypatch.setenv('CAB_TESTNET_ORDERS_ENABLED', 'true')
    monkeypatch.setattr(tool, 'load_settings', lambda *a: SimpleNamespace(log_level='INFO'))
    monkeypatch.setattr(tool, 'configure_logging', lambda directory, level:
                        calls.append(('logging', isinstance(directory, Path), level)))
    monkeypatch.setattr(tool, 'execution_environment', lambda *a: {})
    monkeypatch.setattr(tool, 'create_testnet_bot', lambda *a, **kw: bot)
    def plan(view, directory, symbol, q, attempt):
        calls.append(('balances-before-plan', symbol))
        return view._plan(symbol, q, attempt)
    monkeypatch.setattr(tool, 'plan_with_balance_evidence', plan)
    def accept(current, path, **kw):
        assert kw == {'duration_seconds': 120, 'entry_symbol': 'ETHUSDT'}
        assert current._plan('ETHUSDT', '.01', 'attempt') == 'existing guarded plan'
        path.write_text(json.dumps({'status': 'PASS', 'attempt_id': 'attempt'}), encoding='utf-8')
        calls.append(('accepted',))
        return 0
    monkeypatch.setattr(tool, 'run_acceptance', accept)
    monkeypatch.setattr(tool, 'reconcile_balance_evidence', lambda *a:
                        calls.append(('account-followup',)) or {'status': 'ARCHIVED', 'balance_status': 'INCOMPLETE'})
    tool.main()
    assert bot._plan is original_plan
    assert calls == [('logging', True, 'INFO'), ('balances-before-plan', 'ETHUSDT'),
                     ('accepted',), ('account-followup',)]
    outcome = json.loads(report.with_suffix('.balance.json').read_text(encoding='utf-8'))
    assert outcome['scope'] == 'TESTNET_FULL_WINDOW_BALANCE_EVIDENCE_NOT_PROFIT'
    assert outcome['balance_followup']['balance_status'] == 'INCOMPLETE'
