"""Native acceptance wiring/NULL evidence semantics without HTTP or keys."""

import json
from types import SimpleNamespace

import pytest
import testnet_native_balance_acceptance as tool

from crypto_arbitrage_bot.execution.runner import ExecutionGuardError


@pytest.mark.parametrize('suffix', ['.json', '.balance.json'])
def test_existing_evidence_never_repeats_entry(tmp_path, monkeypatch, suffix):
    report = tmp_path / 'run.json'
    report.with_suffix(suffix).write_text('{}', encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['tool', '--report', str(report)])
    monkeypatch.setattr(tool, 'load_settings', lambda *a: pytest.fail('must not reach credentials'))
    with pytest.raises(ExecutionGuardError, match='ALREADY_EXISTS'):
        tool.main()


def test_disabled_has_no_credentials_or_http(tmp_path, monkeypatch):
    report = tmp_path / 'run.json'
    monkeypatch.setattr('sys.argv', ['tool', '--report', str(report)])
    monkeypatch.delenv('CAB_TESTNET_ORDERS_ENABLED', raising=False)
    monkeypatch.setattr(tool, 'load_settings', lambda *a: pytest.fail('must stay disabled'))
    assert tool.main() == 1
    result = json.loads(report.with_suffix('.balance.json').read_text(encoding='utf-8'))
    assert result['reason'] == 'TESTNET_ORDERS_DISABLED'


@pytest.mark.parametrize('audit_status', ['MATCHED', 'INCOMPLETE', 'MISMATCH', None])
def test_native_planning_unchanged_and_account_status_not_profit(tmp_path, monkeypatch, audit_status):
    report = tmp_path / 'run.json'
    monkeypatch.setattr('sys.argv', ['tool', '--report', str(report)])
    monkeypatch.setenv('CAB_TESTNET_ORDERS_ENABLED', 'true')
    def original_plan():
        return None
    bot = SimpleNamespace(_plan=original_plan, journal=object())
    monkeypatch.setattr(tool, 'load_settings', lambda path: SimpleNamespace(log_level='INFO'))
    monkeypatch.setattr(tool, 'configure_logging', lambda *a: None)
    monkeypatch.setattr(tool, 'execution_environment', lambda *a: {})
    monkeypatch.setattr(tool, 'create_testnet_bot', lambda *a, **kw: bot)
    calls = []
    def accept(current, path, **kw):
        assert current._plan is original_plan
        assert kw == {'duration_seconds': 120, 'entry_symbol': 'ETHUSDT'}
        calls.append('one bounded entry')
        path.write_text(json.dumps({'status': 'PASS', 'attempt_id': 'existing'}), encoding='utf-8')
        return 0
    monkeypatch.setattr(tool, 'run_acceptance', accept)
    check = SimpleNamespace(status=SimpleNamespace(value='ARCHIVED'), audit_id='audit')
    monkeypatch.setattr(tool, 'BalanceCheckRepository', lambda journal: SimpleNamespace(get=lambda aid: check))
    audit = None if audit_status is None else {'status': audit_status,
        'issues': [{'code': 'FUNDING_EVIDENCE_INCOMPLETE'}] if audit_status == 'INCOMPLETE' else [],
        'movements': [{'unexplained_delta': '0E-8'}]}
    monkeypatch.setattr(tool, 'BalanceAuditRepository', lambda journal: SimpleNamespace(get=lambda aid: audit))
    # Fake objects need no synthetic financial serialization or new formulas.
    monkeypatch.setattr(tool, 'save', lambda path, data: calls.append(data))
    monkeypatch.setattr(tool, 'exact_json', lambda data: '{}')
    expected = audit_status in ('MATCHED', 'INCOMPLETE')
    assert tool.main() == (0 if expected else 1)
    assert bot._plan is original_plan and calls[0] == 'one bounded entry'
    assert calls[1]['status'] == ('PASS' if expected else 'NEEDS_ATTENTION')
    assert 'net_profit' not in calls[1]
