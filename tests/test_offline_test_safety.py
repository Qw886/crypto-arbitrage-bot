"""The default test runtime blocks real boundaries in disposable Python processes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).with_name("offline_runtime").resolve()


def probe(directory):
    fake_root = directory / "isolated-project"
    secret = fake_root / "data/testnet/credentials.dpapi"
    secret.parent.mkdir(parents=True)
    secret.write_bytes(b"OFFLINE_FIXTURE_ONLY_NOT_A_REAL_CREDENTIAL")
    script = """import json,os,sys
from pathlib import Path
result={}
try:
    Path(os.environ['CAB_TEST_EXTRA_PROTECTED_ROOT']).joinpath('data/testnet/credentials.dpapi').read_bytes()
except PermissionError:
    result['credential']='BLOCKED'
else:
    result['credential']='READ_ALLOWED'
try:
    sys.audit('socket.connect',None,('synthetic.invalid',443))
except PermissionError:
    result['network']='BLOCKED'
else:
    result['network']='CONNECT_ALLOWED'
print(json.dumps(result))
"""
    env = dict(os.environ, PYTHONPATH=str(GUARD), CAB_TEST_EXTRA_PROTECTED_ROOT=str(fake_root))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=directory,
        env=env,
        capture_output=True,
        encoding="utf-8",
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_python_startup_blocks_fixture_credential_and_network_event(tmp_path):
    # Red is safe: no actual credential, DNS lookup or socket connection is made.
    assert probe(tmp_path) == {"credential": "BLOCKED", "network": "BLOCKED"}


def test_default_pytest_process_blocks_synthetic_network_before_socket_call():
    # Public CPython event is safe even while this regression is red.
    import pytest

    with pytest.raises(PermissionError, match="OFFLINE_TEST_NETWORK_BLOCKED"):
        sys.audit("socket.connect", None, ("synthetic.invalid", 443))


def test_child_explicit_environment_without_pythonpath_keeps_network_denial(tmp_path):
    script = """import json,sys
try:
    sys.audit('socket.connect',None,('synthetic.invalid',443))
except PermissionError:
    print(json.dumps({'network':'BLOCKED'}))
else:
    print(json.dumps({'network':'CONNECT_ALLOWED'}))
"""
    env = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        encoding="utf-8",
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"network": "BLOCKED"}


def test_python_without_site_startup_is_rejected_before_launch(tmp_path):
    import pytest

    marker = tmp_path / "must-not-run"
    with pytest.raises(PermissionError, match="OFFLINE_TEST_UNGUARDED_PYTHON_STARTUP"):
        subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "from pathlib import Path;Path(__import__('sys').argv[1]).touch()",
                str(marker),
            ],
            cwd=tmp_path,
            check=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    assert not marker.exists()


def test_child_does_not_inherit_account_access_or_order_enable(tmp_path):
    keys = (
        "CAB_TESTNET_SPOT_API_KEY",
        "CAB_TESTNET_SPOT_API_SECRET",
        "CAB_TESTNET_USDTM_API_KEY",
        "CAB_TESTNET_USDTM_API_SECRET",
        "CAB_EXECUTION_MODE",
        "CAB_TESTNET_ORDERS_ENABLED",
    )
    env = dict(os.environ)
    env.update({key: "OFFLINE_SENTINEL_NOT_A_REAL_KEY" for key in keys})
    script = (
        "import json,os;print(json.dumps({'account_environment_present':any(k in os.environ for k in "
        + repr(keys)
        + ")}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        encoding="utf-8",
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"account_environment_present": False}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows case-insensitive .env alias")
def test_windows_uppercase_env_alias_remains_protected(tmp_path):
    fake_root = tmp_path / "isolated-project"
    fake_root.mkdir()
    (fake_root / ".env").write_text("OFFLINE_FIXTURE_ONLY", encoding="utf-8")
    script = """import json,os
from pathlib import Path
try:
    (Path(os.environ['CAB_TEST_EXTRA_PROTECTED_ROOT'])/'.ENV').read_text()
except PermissionError:
    print(json.dumps({'env_file':'BLOCKED'}))
else:
    print(json.dumps({'env_file':'READ_ALLOWED'}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, CAB_TEST_EXTRA_PROTECTED_ROOT=str(fake_root)),
        cwd=tmp_path,
        capture_output=True,
        encoding="utf-8",
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"env_file": "BLOCKED"}


def test_subprocess_audit_refuses_environment_without_guard_path(tmp_path):
    with pytest.raises(PermissionError, match="OFFLINE_TEST_CHILD_STARTUP_PATH_MISSING"):
        sys.audit(
            "subprocess.Popen",
            sys.executable,
            [sys.executable, "-c", "pass"],
            str(tmp_path),
            {"PYTHONPATH": str(tmp_path)},
        )


@pytest.mark.parametrize("flags", [["-E"], ["-I"], ["-IE"], ["-W", "ignore", "-S"]])
def test_unsupported_python_startup_flags_never_run_program(tmp_path, flags):
    marker = tmp_path / "must-not-run"
    with pytest.raises(PermissionError, match="OFFLINE_TEST_UNGUARDED_PYTHON_STARTUP"):
        subprocess.run(
            [
                sys.executable,
                *flags,
                "-c",
                "from pathlib import Path;Path(__import__('sys').argv[1]).touch()",
                str(marker),
            ],
            cwd=tmp_path,
            check=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    assert not marker.exists()


@pytest.mark.parametrize("flags", [["-s"], ["-W", "ignore"], ["-X", "dev"]])
def test_supported_python_flags_preserve_denial_and_script_arguments(tmp_path, flags):
    script = """import json,sys
try:
    sys.audit('socket.connect',None,('synthetic.invalid',443))
except PermissionError:
    print(json.dumps({'network':'BLOCKED','script_argument':sys.argv[1]}))
else:
    print(json.dumps({'network':'CONNECT_ALLOWED'}))
"""
    result = subprocess.run(
        [sys.executable, *flags, "-c", script, "-S"],
        cwd=tmp_path,
        capture_output=True,
        encoding="utf-8",
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"network": "BLOCKED", "script_argument": "-S"}


def test_grandchild_with_replaced_environment_keeps_startup_guard(tmp_path):
    inner = """import json,sys
try:
    sys.audit('socket.connect',None,('synthetic.invalid',443))
except PermissionError:
    print(json.dumps({'network':'BLOCKED'}))
else:
    print(json.dumps({'network':'CONNECT_ALLOWED'}))
"""
    outer = (
        "import os,subprocess,sys;env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'};p=subprocess.run([sys.executable,'-c',"
        + repr(inner)
        + "],env=env,capture_output=True,encoding='utf-8',timeout=5);print(p.stdout.strip());sys.exit(p.returncode)"
    )
    result = subprocess.run(
        [sys.executable, "-c", outer],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        encoding="utf-8",
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"network": "BLOCKED"}


@pytest.mark.parametrize("event", ["socket.getaddrinfo", "socket.sendto", "socket.sendmsg"])
def test_network_events_are_denied_even_with_test_socket_mocks(event):
    with pytest.raises(PermissionError, match="OFFLINE_TEST_NETWORK_BLOCKED"):
        sys.audit(event, None)


def test_known_actual_credential_path_denied_without_opening_it():
    actual = Path(__file__).resolve().parents[1] / "data/testnet/credentials.dpapi"
    with pytest.raises(PermissionError, match="OFFLINE_TEST_ACTUAL_CREDENTIAL_BLOCKED"):
        sys.audit("open", str(actual), "r", 0)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Popen command-line quoting")
def test_windows_nested_quotes_preserve_arguments_and_startup_guard(tmp_path):
    script = """import json,sys
quoted = 'single "double"'
try:
    sys.audit('socket.connect',None,('synthetic.invalid',443))
except PermissionError:
    print(json.dumps({'network':'BLOCKED','argument':sys.argv[1],'quoted':quoted}))
"""
    argument = 'argument with "double quotes" and space'
    result = subprocess.run(
        subprocess.list2cmdline([sys.executable, "-c", script, argument]),
        cwd=tmp_path,
        capture_output=True,
        encoding="utf-8",
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "network": "BLOCKED",
        "argument": argument,
        "quoted": 'single "double"',
    }
