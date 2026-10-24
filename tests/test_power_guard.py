"""Finite lifecycle and Win32 cleanup; no exchange clients or real orders."""

import ctypes
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_offline_estimate import OPENED

from crypto_arbitrage_bot import monitor_archive as archive
from crypto_arbitrage_bot import power_guard as guard
from crypto_arbitrage_bot.utils import windows_power as native
from crypto_arbitrage_bot.utils.file_lock import file_lock
from crypto_arbitrage_bot.utils.process_probe import ProcessObservation, ProcessState


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("local power only")))


@pytest.fixture
def monitor(tmp_path):
    directory = tmp_path / "monitor"
    directory.mkdir()
    config = tmp_path / "testnet.toml"
    config.write_text(Path("docs/testnet-funding-acceptance.toml").read_text().replace(
        "data/testnet/execution.sqlite", (tmp_path / "absent.sqlite").as_posix()), encoding="utf-8")
    spec = {"research_config": None, "research_directory": None, "testnet_config": str(config.resolve()),
            "max_heartbeat_age_seconds": 60, "working_directory": str(Path.cwd().resolve())}
    archive.write(directory / "plan.json", {"started_at": OPENED, "ends_at": OPENED+timedelta(seconds=50),
        "interval_seconds": 10, "max_checks": 20, "orders_submitted": False}, exclusive=True)
    _, sha = archive.load_plan(directory)
    born = OPENED-timedelta(seconds=1)
    with file_lock(directory / "monitor.lock"):
        archive.bind_live(directory, sha, 4242, spec, clock=lambda: OPENED,
            probe=lambda *a, **k: ProcessObservation(ProcessState.MATCHED, 4242, born))
        archive.write(directory / "progress.json", {"plan_sha256": sha, "state": "RUNNING", "checks": 1,
            "mode": "RESUMED", "worker_pid": 4242, "worker_created_at": born.isoformat(),
            "updated_at": (OPENED+timedelta(seconds=10)).isoformat()})
        yield directory, sha, tmp_path / "guard", born


class FakePower:
    def __init__(self):
        self.ac = True
        self.active = False
        self.starts = 0
        self.closed = 0

    def ac_available(self):
        return self.ac

    @contextmanager
    def system_request(self, reason):
        assert not self.active
        self.starts += 1
        self.active = True
        try:
            yield
        finally:
            self.active = False
            self.closed += 1


def run(monitor, *, wait_action=None, power=None, probe_action=None):
    directory, sha, output, born = monitor
    seconds = [10]
    steady = [10]
    power = power or FakePower()
    before = {p: p.read_bytes() for p in (directory / "plan.json", directory / "monitor-binding.json")}
    def probe(pid, **kwargs):
        if probe_action is not None:
            custom = probe_action(pid, kwargs)
            if custom is not None:
                return custom
        return ProcessObservation(ProcessState.MATCHED, pid, born)
    def wait(duration):
        assert power.active
        if wait_action is not None:
            wait_action(seconds, steady, power, directory, output)
        else:
            seconds[0] += duration
            steady[0] += duration
    try:
        result = guard.hold(directory, sha, output, power=power, clock=lambda: OPENED+timedelta(seconds=seconds[0]),
            steady_ns=lambda: int(steady[0]*1e9), wait=wait, probe=probe)
        return result, power
    finally:
        assert all(p.read_bytes() == body for p, body in before.items())
        assert not power.active


def test_original_deadline_and_short_final_wait_release_without_extension(monitor):
    result, power = run(monitor)
    assert result["reason"] == "ORIGINAL_DEADLINE_REACHED"
    assert result["updated_at"] == OPENED+timedelta(seconds=50)
    assert result["checks"] == 3 and power.starts == power.closed == 1
    assert result["os_request_verified"] is result["network_guaranteed"] is False


@pytest.mark.parametrize("action,reason", [
    ("battery", "AC_POWER_UNAVAILABLE"), ("stop", "STOP_REQUESTED"),
    ("gap", "CLOCK_OR_EXECUTION_GAP"), ("rollback", "CLOCK_OR_EXECUTION_GAP"),
    ("steady_deadline", "ORIGINAL_DEADLINE_REACHED")])
def test_release_on_power_stop_gap_clock_and_monotonic_budget(monitor, action, reason):
    def mutate(seconds, steady, power, directory, output):
        if action == "battery":
            power.ac = False
        elif action == "stop":
            (output / "stop").touch()
        elif action == "gap":
            seconds[0] += 61
        elif action == "rollback":
            seconds[0] -= 1
        else:
            steady[0] += 41
    result, power = run(monitor, wait_action=mutate)
    assert result["reason"] == reason and power.closed == 1


def test_no_request_on_battery(monitor):
    power = FakePower()
    power.ac = False
    result, _ = run(monitor, power=power)
    assert result["status"] == "RELEASED" and power.starts == power.closed == 0


@pytest.mark.parametrize("field,value", [("state", "STOPPED"), ("checks", 20), ("checks", True),
    ("worker_pid", True), ("updated_at", "2020-01-01T00:00:00+00:00"),
    ("updated_at", "2100-01-01T00:00:00+00:00"), ("worker_created_at", "2100-01-01T00:00:00+00:00")])
def test_target_invalid_checkpoint_rejected_before_request(monitor, field, value):
    directory, _, output, _ = monitor
    path = directory / "progress.json"
    progress = archive.read(path)
    progress[field] = value
    archive.write(path, progress)
    with pytest.raises(ValueError):
        run(monitor)
    assert not (output / "binding.json").exists()


@pytest.mark.parametrize("state", [ProcessState.MISSING, ProcessState.UNKNOWN, ProcessState.DIFFERENT_IDENTITY])
def test_unknown_or_reused_target_pid_does_not_acquire(monitor, state):
    with pytest.raises(ValueError, match="NATIVE_OWNER"):
        run(monitor, probe_action=lambda pid, k: ProcessObservation(state, pid))


def test_existing_archive_cannot_restart_or_rearm(monitor):
    run(monitor)
    with pytest.raises(ValueError, match="NO_RESUME"):
        run(monitor)


@pytest.mark.parametrize("failure", ["corrupt", "stopped", "interrupted", "checkpoint_io"])
def test_failures_after_acquisition_release_before_report(monitor, failure, monkeypatch):
    def mutate(seconds, steady, power, directory, output):
        if failure == "interrupted":
            raise KeyboardInterrupt
        if failure == "checkpoint_io":
            original = archive.write
            def write(path, *args, **kwargs):
                if path == output / "state.json":
                    raise PermissionError("synthetic disk failure")
                return original(path, *args, **kwargs)
            monkeypatch.setattr(archive, "write", write)
        elif failure == "corrupt":
            (directory / "progress.json").write_text("broken", encoding="utf-8")
        else:
            value = archive.read(directory / "progress.json")
            value["state"] = "STOPPED"
            archive.write(directory / "progress.json", value)
        seconds[0] += 15
        steady[0] += 15
    power = FakePower()
    if failure == "interrupted":
        result, _ = run(monitor, power=power, wait_action=mutate)
        assert result["reason"] == "INTERRUPTED"
    else:
        with pytest.raises((ValueError, PermissionError)):
            run(monitor, power=power, wait_action=mutate)
    assert power.closed == 1 and not power.active


def test_monitor_replacement_does_not_transfer_lease(monitor):
    def mutate(seconds, steady, power, directory, output):
        progress = archive.read(directory / "progress.json")
        progress["worker_pid"] = 4343
        archive.write(directory / "progress.json", progress)
    result, power = run(monitor, wait_action=mutate)
    assert result["reason"] == "ORIGINAL_MONITOR_OWNER_CHANGED" and power.closed == 1


@pytest.fixture
def kernel(monkeypatch):
    mock = Mock()
    mock.PowerCreateRequest.return_value = 123
    mock.PowerSetRequest.return_value = 1
    mock.PowerClearRequest.return_value = 1
    mock.CloseHandle.return_value = 1
    def power_status(pointer):
        status = ctypes.cast(pointer, ctypes.POINTER(native._PowerStatus)).contents
        status.ac, status.saver = 1, 0
        return 1
    mock.GetSystemPowerStatus.side_effect = power_status
    monkeypatch.setattr(native.sys, "platform", "win32")
    monkeypatch.setattr(native.ctypes, "WinDLL", Mock(return_value=mock), raising=False)
    return mock


def test_win32_request_only_system_and_cleanup_after_body_failure(kernel):
    with pytest.raises(RuntimeError):
        with native.WindowsPower().system_request("bounded test"):
            raise RuntimeError("body")
    kernel.PowerSetRequest.assert_called_once_with(123, 1)
    kernel.PowerClearRequest.assert_called_once_with(123, 1)
    kernel.CloseHandle.assert_called_once_with(123)


@pytest.mark.parametrize("operation", ["PowerSetRequest", "PowerClearRequest", "CloseHandle"])
def test_win32_failures_are_explicit_and_close_is_attempted(kernel, operation):
    getattr(kernel, operation).return_value = 0
    with pytest.raises(OSError):
        with native.WindowsPower().system_request("bounded test"):
            pass
    kernel.CloseHandle.assert_called_once_with(123)


@pytest.mark.parametrize("handle", [None, 0, ctypes.c_void_p(-1).value])
def test_invalid_create_handle_is_never_used(kernel, handle):
    kernel.PowerCreateRequest.return_value = handle
    with pytest.raises(OSError, match="CREATE"):
        with native.WindowsPower().system_request("bounded test"):
            pass
    kernel.PowerSetRequest.assert_not_called()
    kernel.CloseHandle.assert_not_called()


@pytest.mark.parametrize("ac,saver", [(0, 0), (255, 0), (1, 1), (1, 255)])
def test_dc_unknown_or_saver_does_not_create_request(kernel, ac, saver):
    def status(pointer):
        value = ctypes.cast(pointer, ctypes.POINTER(native._PowerStatus)).contents
        value.ac, value.saver = ac, saver
        return 1
    kernel.GetSystemPowerStatus.side_effect = status
    with pytest.raises(OSError, match="AC_POWER"):
        with native.WindowsPower().system_request("bounded test"):
            pass
    kernel.PowerCreateRequest.assert_not_called()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows API and owner required")
def test_native_guard_short_monitor_stop_and_os_request_cleanup(tmp_path, capsys):
    if not native.WindowsPower().ac_available():
        pytest.skip("AC required for native sleep request")
    directory, output = tmp_path / "monitor", tmp_path / "guard"
    config = tmp_path / "testnet.toml"
    config.write_text(Path("docs/testnet-funding-acceptance.toml").read_text().replace(
        "data/testnet/execution.sqlite", (tmp_path / "absent.sqlite").as_posix()), encoding="utf-8")
    processes = []
    def launch(args):
        proc = subprocess.Popen([sys.executable, "-u", "-m", *args], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        processes.append(proc)
        return proc
    def await_file(path, predicate):
        deadline = time.monotonic()+15
        while time.monotonic() < deadline:
            if path.is_file():
                value = archive.read(path)
                if predicate(value):
                    return value
            time.sleep(.05)
        raise AssertionError("native guard/checkpoint not observed")
    try:
        monitor = launch(["crypto_arbitrage_bot.runtime_status", "monitor", "--testnet-config", str(config),
            "--max-heartbeat-age-seconds", "60", "--output-dir", str(directory), "--duration-seconds", "50",
            "--interval-seconds", "1", "--max-checks", "40"])
        await_file(directory / "progress.json", lambda p: p["checks"] >= 1)
        _, sha = archive.load_plan(directory)
        before = (directory / "plan.json").read_bytes()
        worker = launch(["crypto_arbitrage_bot.power_guard", "hold", "--monitor-dir", str(directory),
            "--original-plan-sha256", sha, "--output-dir", str(output)])
        state = await_file(output / "state.json", lambda p: p["status"] == "REQUEST_CREATED")
        assert state["checks"] >= 1 and archive.lock_is_held(output / "guard.lock")
        assert guard.main(["status", "--monitor-dir", str(directory), "--original-plan-sha256", sha,
            "--output-dir", str(output)]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "RUNNING_CONFIRMED" and report["native_owner_confirmed"]
        assert report["os_request_verified"] is False
        assert guard.main(["stop", "--monitor-dir", str(directory), "--original-plan-sha256", sha,
            "--output-dir", str(output)]) == 0
        stdout, stderr = worker.communicate(timeout=20)
        assert worker.returncode == 0, (stdout, stderr)
        assert archive.read(output / "state.json")["status"] == "RELEASED"
        assert not archive.lock_is_held(output / "guard.lock")
        assert (directory / "plan.json").read_bytes() == before and monitor.poll() is None
    finally:
        for path in (directory / "stop", output / "stop"):
            if path.parent.exists():
                path.touch(exist_ok=True)
        for proc in processes:
            try:
                proc.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()  # This test owns only these temporary children.
                proc.communicate(timeout=5)


@pytest.mark.parametrize("change", ["wrong_hash", "unlocked", "changed_config", "own_unknown"])
def test_scope_lock_policy_and_actual_guard_identity_required(monitor, monkeypatch, change):
    directory, sha, output, born = monitor
    if change == "wrong_hash":
        monitor = directory, "0"*64, output, born
    elif change == "unlocked":
        monkeypatch.setattr(guard, "lock_is_held", lambda path: False)
    elif change == "changed_config":
        config = Path(archive.read(directory / "monitor-binding.json")["spec"]["testnet_config"])
        config.write_text(config.read_text()+"\n# changed policy file\n", encoding="utf-8")
    with pytest.raises(ValueError):
        run(monitor, probe_action=(lambda pid, k: ProcessObservation(ProcessState.UNKNOWN, pid)
            if pid != 4242 else None) if change == "own_unknown" else None)
    assert not (output / "binding.json").exists()


@pytest.mark.parametrize("mutation", ["extra", "duplicate", "odd", "other_interpreter", "other_hash"])
def test_exact_guard_command_does_not_accept_scope_overrides(monitor, mutation):
    directory, sha, output, _ = monitor
    args = [sys.executable, "-u", "-m", "crypto_arbitrage_bot.power_guard", "hold",
            "--monitor-dir", str(directory), "--original-plan-sha256", sha, "--output-dir", str(output)]
    assert guard.matches_guard(tuple(args), directory, sha, output)
    if mutation == "extra":
        args.extend(["--duration-seconds", "999999"])
    elif mutation == "duplicate":
        args.extend(["--output-dir", str(output)])
    elif mutation == "odd":
        args.append("--output-dir")
    elif mutation == "other_interpreter":
        args[0] = "different.exe"
    else:
        args[-3] = "0"*64
    assert not guard.matches_guard(tuple(args), directory, sha, output)


def test_power_status_api_failure_is_not_assumed_ac(kernel):
    kernel.GetSystemPowerStatus.side_effect = None
    kernel.GetSystemPowerStatus.return_value = 0
    with pytest.raises(OSError, match="STATUS_UNAVAILABLE"):
        native.WindowsPower().ac_available()
    kernel.PowerCreateRequest.assert_not_called()


@pytest.mark.parametrize("reason", ["", "x"*201, "embedded\x00nul"])
def test_invalid_reason_cannot_reach_native_create(kernel, reason):
    with pytest.raises(ValueError):
        with native.WindowsPower().system_request(reason):
            pass
    kernel.PowerCreateRequest.assert_not_called()
