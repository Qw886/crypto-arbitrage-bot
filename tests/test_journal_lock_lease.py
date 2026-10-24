"""An explicit live lock lease permits one controller to delegate its journal."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution import runner
from crypto_arbitrage_bot.execution.demo import demo_bot


def test_managed_session_borrows_held_lock_without_unlocking_for_other_bot(tmp_path):
    bot, transport = demo_bot(load_settings(Path('docs/testnet-bot.toml')), tmp_path)
    other = runner.TestnetBot(bot.settings, bot.spot, bot.perp)
    with runner.lease_process_lock(bot.config.journal_path.with_suffix('.lock')) as lease:
        with bot.controlled_session(manage_only=True, lock_lease=lease):
            assert bot.manage_existing_once() == ()
            with pytest.raises(runner.ExecutionGuardError):
                other.manage_existing_once()
        with pytest.raises(runner.ExecutionGuardError):
            other.manage_existing_once()
    assert other.manage_existing_once() == ()
    assert not transport.orders


def test_wrong_journal_lease_cannot_be_borrowed(tmp_path):
    bot, transport = demo_bot(load_settings(Path('docs/testnet-bot.toml')), tmp_path / 'bot')
    with runner.lease_process_lock(tmp_path / 'different.lock') as lease:
        with pytest.raises(runner.ExecutionGuardError):
            with bot.controlled_session(manage_only=True, lock_lease=lease):
                bot.manage_existing_once()
    assert not transport.orders


def test_lease_is_valid_only_on_the_owning_thread(tmp_path):
    bot, transport = demo_bot(load_settings(Path('docs/testnet-bot.toml')), tmp_path)
    def other_thread(lease):
        with bot.controlled_session(manage_only=True, lock_lease=lease):
            bot.manage_existing_once()
    with runner.lease_process_lock(bot.config.journal_path.with_suffix('.lock')) as lease:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(runner.ExecutionGuardError):
                pool.submit(other_thread, lease).result(timeout=10)
    assert not transport.orders


def test_unissued_and_released_lease_cannot_be_borrowed(tmp_path):
    bot, transport = demo_bot(load_settings(Path('docs/testnet-bot.toml')), tmp_path)
    with runner.lease_process_lock(bot.config.journal_path.with_suffix('.lock')) as expired:
        pass
    for lease in (runner.ProcessLockLease(), expired):
        with pytest.raises(runner.ExecutionGuardError):
            with bot.controlled_session(manage_only=True, lock_lease=lease):
                bot.manage_existing_once()
    assert not transport.orders


def test_self_claimed_lock_object_cannot_grant_a_controlled_session(tmp_path):
    bot, transport = demo_bot(load_settings(Path('docs/testnet-bot.toml')), tmp_path)
    forged = SimpleNamespace(require_owned=lambda path: None)
    with pytest.raises(runner.ExecutionGuardError):
        with bot.controlled_session(manage_only=True, lock_lease=forged):
            bot.manage_existing_once()
    assert not transport.orders


def test_released_parent_lease_blocks_further_cycles_inside_borrowed_session(tmp_path):
    bot, transport = demo_bot(load_settings(Path('docs/testnet-bot.toml')), tmp_path)
    parent = runner.lease_process_lock(bot.config.journal_path.with_suffix('.lock'))
    lease = parent.__enter__()
    try:
        with bot.controlled_session(manage_only=True, lock_lease=lease):
            parent.__exit__(None, None, None)
            with pytest.raises(runner.ExecutionGuardError):
                bot.manage_existing_once()
    finally:
        parent.__exit__(None, None, None)
    assert not transport.orders


def test_nested_borrowed_session_cannot_replace_the_original_management_policy(tmp_path):
    bot, transport = demo_bot(load_settings(Path('docs/testnet-bot.toml')), tmp_path)
    with runner.lease_process_lock(bot.config.journal_path.with_suffix('.lock')) as lease:
        with bot.controlled_session(manage_only=True, lock_lease=lease):
            with pytest.raises(runner.ExecutionGuardError, match='CONTROLLED_SESSION_ALREADY_ACTIVE'):
                with bot.controlled_session(manage_only=False, lock_lease=lease):
                    pass
            assert bot.run_once() == ()
    assert not transport.orders


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows native process-lock acceptance')
def test_independent_process_is_excluded_until_parent_lease_releases(tmp_path):
    bot, transport = demo_bot(load_settings(Path('docs/testnet-bot.toml')), tmp_path)
    script = '''
import sys
from pathlib import Path
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError, process_lock
try:
    with process_lock(Path(sys.argv[1])):
        pass
except ExecutionGuardError:
    raise SystemExit(0)
raise SystemExit(3)
'''

    def probe():
        result = subprocess.run(
            [sys.executable, '-c', script, str(bot.config.journal_path.with_suffix('.lock'))],
            env=dict(os.environ, PYTHONPATH=str(Path('src').resolve())),
            capture_output=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
        assert not result.stderr and not result.stdout
        return result.returncode

    with runner.lease_process_lock(bot.config.journal_path.with_suffix('.lock')) as lease:
        with bot.controlled_session(manage_only=True, lock_lease=lease):
            assert bot.manage_existing_once() == ()
            assert probe() == 0
        assert probe() == 0
    assert probe() == 3
    assert not transport.orders
