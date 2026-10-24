"""Real local Windows ownership probe and unavailable observations, offline."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils import process_probe as module
from crypto_arbitrage_bot.utils.file_lock import lock_is_held
from crypto_arbitrage_bot.utils.process_probe import ProcessState


@pytest.mark.parametrize(
    "failure", ["timeout", "permission", "invalid_json", "failed_command", "oversize"]
)
def test_probe_failure_is_unknown_not_missing(monkeypatch, failure):
    call = Mock()
    if failure == "timeout":
        call.side_effect = subprocess.TimeoutExpired("probe", 5)
    elif failure == "permission":
        call.side_effect = PermissionError("PRIVATE_TEXT")
    else:
        call.return_value = Mock(
            returncode=1 if failure == "failed_command" else 0,
            stdout="{" if failure == "invalid_json" else "x" * 64001,
        )
    monkeypatch.setattr(module.subprocess, "run", call)
    now = datetime.now(UTC)
    result = module.observe_process(
        123, matches=lambda a: True, not_before=now - timedelta(days=1), not_after=now
    )
    assert result.state is ProcessState.UNKNOWN and "PRIVATE_TEXT" not in exact_json(result)


@pytest.mark.parametrize("pid", [None, True, 0, -1, 2**32, "123; Get-Secret"])
def test_invalid_pid_cannot_enter_shell_code(monkeypatch, pid):
    run = Mock(side_effect=AssertionError("unsafe shell input"))
    monkeypatch.setattr(module.subprocess, "run", run)
    now = datetime.now(UTC)
    assert (
        module.observe_process(pid, matches=lambda a: True, not_before=now, not_after=now).state
        is ProcessState.UNKNOWN
    )
    run.assert_not_called()


def test_reused_pid_and_unrelated_command_are_distinct(monkeypatch):
    now = datetime.now(UTC)
    run = Mock(
        return_value=Mock(
            returncode=0,
            stdout=json.dumps(
                {
                    "command": "python.exe --api-secret PRIVATE_TEXT",
                    "created": (now + timedelta(seconds=1)).isoformat(),
                }
            ),
        )
    )
    monkeypatch.setattr(module.subprocess, "run", run)
    result = module.observe_process(
        123, matches=lambda a: True, not_before=now - timedelta(days=1), not_after=now
    )
    assert result.state is ProcessState.DIFFERENT_IDENTITY
    assert "PRIVATE_TEXT" not in exact_json(result)
    run.return_value.stdout = json.dumps(
        {"command": "python.exe --api-secret PRIVATE_TEXT", "created": now.isoformat()}
    )
    result = module.observe_process(
        123, matches=lambda a: False, not_before=now - timedelta(days=1), not_after=now
    )
    assert result.state is ProcessState.DIFFERENT_IDENTITY
    assert "PRIVATE_TEXT" not in exact_json(result)


def test_native_owned_child_and_termination_with_stale_pid(tmp_path):
    if sys.platform != "win32":
        pytest.skip("native Windows observation")
    lock = tmp_path / "child.lock"
    code = "from pathlib import Path; import sys,os,threading; from crypto_arbitrage_bot.utils.file_lock import file_lock\ntimer=threading.Timer(15,lambda:os._exit(1));timer.daemon=True;timer.start()\nwith file_lock(Path(sys.argv[1])):\n print('READY',flush=True)\n sys.stdin.readline()"
    environment = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    started = datetime.now(UTC)
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", code, str(lock)],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        assert child.stdout.readline().strip() == "READY"
        updated = datetime.now(UTC)

        def match(a):
            return "-c" in a and str(lock) in a

        actual = module.observe_process(
            child.pid, matches=match, not_before=started, not_after=updated
        )
        assert actual.state is ProcessState.MATCHED and lock_is_held(lock)
        assert (
            module.observe_process(
                child.pid, matches=lambda a: False, not_before=started, not_after=updated
            ).state
            is ProcessState.DIFFERENT_IDENTITY
        )
        child.stdin.write("\n")
        child.stdin.flush()
        assert child.wait(timeout=5) == 0
        stopped = module.observe_process(
            child.pid, matches=match, not_before=started, not_after=updated
        )
        assert stopped.state is ProcessState.MISSING and not lock_is_held(lock)
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)
        for stream in (child.stdin, child.stdout, child.stderr):
            stream.close()
