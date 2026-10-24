"""Public manage-existing entry point after actual pre-ACK process death."""
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / 'work/pre_ack_management_fixture.py'


def run_phase(directory, symbol, case, phase, expected):
    result = subprocess.run(
        [sys.executable, str(HELPER), str(directory), symbol, case, phase],
        capture_output=True, encoding='utf-8', timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
    )
    assert result.returncode == expected, result.stderr
    return json.loads((directory / f'{phase}.json').read_text(encoding='utf-8'))


@pytest.mark.skipif(sys.platform != 'win32', reason='native Windows process birth proof')
@pytest.mark.parametrize('symbol', ['BTCUSDT', 'ETHUSDT'])
@pytest.mark.parametrize('case', ['normal', 'partial_exit', 'query_timeout', 'exit_ack_loss', 'repair_ack_loss'])
def test_native_pre_ack_management_restores_only_own_inventory(tmp_path, symbol, case):
    start = run_phase(tmp_path, symbol, case, 'entry', 29)
    assert start['state'] == 'OPENING_SPOT'
    assert start['orders'] == 1 and Decimal(start['base']) > 1
    if case == 'query_timeout':
        blocked = run_phase(tmp_path, symbol, case, 'blocked', 0)
        assert blocked['state'] == 'RECOVERY_REQUIRED' and blocked['orders'] == 1
        assert blocked['base'] == start['base']
    if case in ('exit_ack_loss', 'repair_ack_loss'):
        lost = run_phase(tmp_path, symbol, case, 'exit', 31)
        assert lost['identity'] != start['identity']
    end = run_phase(tmp_path, symbol, case, 'resume', 0)
    assert end['identity'] != start['identity']
    assert end['state'] == 'FLATTENED'
    assert Decimal(end['base']) == 1 and Decimal(end['perp']) == 0
    assert end['orders'] == (3 if case in ('partial_exit', 'repair_ack_loss') else 2)
    assert end['entry_posts'] == 1 and end['perp_posts'] == 0
    assert end['terminal_extra_requests'] == 0
    assert end['entry_frame_sha256'] == start['entry_frame_sha256']
    assert end['entry_blocker'] == 'BALANCE_NON_CLOSED_ATTEMPT_REQUIRES_REVIEW'
    assert end['accounting'] == 'INCOMPLETE' and end['profit'] is None
