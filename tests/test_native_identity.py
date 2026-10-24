"""Native process probes and simulated Windows API failures; no trading clients."""

import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot.utils import native_identity as module
from crypto_arbitrage_bot.utils.native_identity import NativeIdentity, NativeState

TICKS = 134360000000000000


class Kernel:
    def __init__(self, *, opened=1, birth=TICKS, times=True, waited=258):
        self.opened, self.birth, self.times, self.waited = opened, birth, times, waited
        self.CloseHandle = Mock(return_value=True)
        self.OpenProcess = Mock(return_value=opened)
        self.WaitForSingleObject = Mock(return_value=waited)
        self.GetCurrentProcess = Mock(return_value=-1)

    def GetProcessTimes(self, handle, *values):
        values[0]._obj.dwHighDateTime = self.birth >> 32
        values[0]._obj.dwLowDateTime = self.birth & (2**32 - 1)
        return self.times


@pytest.mark.parametrize(
    "opened,birth,times,waited,error,expected",
    [
        (1, TICKS, True, 258, 0, NativeState.RUNNING),
        (1, TICKS, True, 0, 0, NativeState.EXITED),
        (1, TICKS + 1, True, 258, 0, NativeState.REPLACED),
        (1, TICKS, False, 258, 0, NativeState.UNKNOWN),
        (1, TICKS, True, 4294967295, 0, NativeState.UNKNOWN),
        (0, TICKS, True, 258, 87, NativeState.MISSING),
        (0, TICKS, True, 258, 5, NativeState.UNKNOWN),
    ],
)
def test_native_states_keep_unavailable_distinct_from_terminal(
    monkeypatch, opened, birth, times, waited, error, expected
):
    kernel = Kernel(opened=opened, birth=birth, times=times, waited=waited)
    monkeypatch.setattr(module, "_kernel", lambda: kernel)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: error, raising=False)
    assert module.observe_native_identity(NativeIdentity(42, TICKS)) is expected
    assert kernel.CloseHandle.call_count == bool(opened)
    if opened:
        kernel.OpenProcess.assert_called_once_with(0x101000, False, 42)


def test_current_identity_is_native_pid_and_exact_birth(monkeypatch):
    kernel = Kernel()
    monkeypatch.setattr(module, "_kernel", lambda: kernel)
    result = module.current_native_identity()
    assert result.pid == os.getpid() and result.creation_filetime == TICKS
    assert NativeIdentity.parse(result.payload()) == result
    assert kernel.CloseHandle.call_count == 0


@pytest.mark.parametrize(
    "change",
    [
        "boolean_pid",
        "zero_pid",
        "big_pid",
        "boolean_ticks",
        "zero_ticks",
        "negative_ticks",
        "too_big_ticks",
        "wrong_date",
        "leading_zero",
        "unicode_ticks",
    ],
)
def test_identity_values_and_utc_projection_must_be_exact(change):
    value = NativeIdentity(42, TICKS).payload()
    key, content = {
        "boolean_pid": ("pid", True),
        "zero_pid": ("pid", 0),
        "big_pid": ("pid", 2**32),
        "boolean_ticks": ("creation_filetime", True),
        "zero_ticks": ("creation_filetime", "0"),
        "negative_ticks": ("creation_filetime", "-1"),
        "too_big_ticks": ("creation_filetime", str(2**64)),
        "wrong_date": ("created_at", "2020-01-01T00:00:00+00:00"),
        "leading_zero": ("creation_filetime", "0" + str(TICKS)),
        "unicode_ticks": ("creation_filetime", "１２３"),
    }[change]
    value[key] = content
    with pytest.raises((ValueError, TypeError, OverflowError)):
        NativeIdentity.parse(value)


def test_native_api_unavailable_is_not_a_missing_process(monkeypatch):
    monkeypatch.setattr(module, "_kernel", Mock(side_effect=OSError("PRIVATE_DETAIL")))
    assert module.observe_native_identity(NativeIdentity(42, TICKS)) is NativeState.UNKNOWN
    with pytest.raises(OSError):
        module.current_native_identity()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows process evidence")
def test_actual_child_birth_liveness_and_exit(tmp_path):
    path = tmp_path / "native-identity.json"
    code = "from pathlib import Path; import json,sys; from crypto_arbitrage_bot.utils.native_identity import current_native_identity; p=Path(sys.argv[1]); t=p.with_suffix('.tmp'); t.write_text(json.dumps(current_native_identity().payload()),encoding='utf-8'); t.replace(p); input()"
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(__import__("pathlib").Path("src").resolve())},
    )
    try:
        deadline = time.monotonic() + 10
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.025)
        assert path.exists()
        identity = NativeIdentity.parse(json.loads(path.read_text(encoding="utf-8")))
        assert identity.pid != os.getpid() and identity.created_at <= datetime.now(UTC)
        assert module.observe_native_identity(identity) is NativeState.RUNNING
        assert (
            module.observe_native_identity(
                NativeIdentity(identity.pid, identity.creation_filetime + 1)
            )
            is NativeState.REPLACED
        )
        stdout, stderr = process.communicate("\n", timeout=10)
        assert process.returncode == 0 and not stderr
        assert module.observe_native_identity(identity) in (NativeState.EXITED, NativeState.MISSING)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
