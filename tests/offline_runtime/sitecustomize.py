"""Python test boundary protection; never imported by the product entry point.

The hook blocks CPython network events and access to known actual credentials.
An extra disposable protected root lets tests verify denial without real keys.
This is test isolation, not an operating-system sandbox for hostile native code.
"""

import ctypes
import os
import re
import shlex
import subprocess
import sys
from functools import wraps
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_ROOTS = (PROJECT_ROOT,)
# Delete names only in this disposable Python environment, never fetch or print
# their values. Tests can explicitly supply synthetic values after startup.
if not getattr(sys, "_cab_test_account_environment_cleared", False):
    for key in (
        "CAB_TESTNET_SPOT_API_KEY",
        "CAB_TESTNET_SPOT_API_SECRET",
        "CAB_TESTNET_USDTM_API_KEY",
        "CAB_TESTNET_USDTM_API_SECRET",
        "CAB_EXECUTION_MODE",
        "CAB_TESTNET_ORDERS_ENABLED",
    ):
        if key in os.environ:
            del os.environ[key]
    sys._cab_test_account_environment_cleared = True
extra_root = os.environ.get("CAB_TEST_EXTRA_PROTECTED_ROOT")
if extra_root:
    PROTECTED_ROOTS += (Path(extra_root).resolve(),)


def require_guarded_python_startup(command, executable=None):
    pattern = r"(?:python(?:\d+(?:\.\d+)*)?w?|py)(?:\.exe)?"
    if executable is not None and not re.fullmatch(
        pattern, Path(os.fsdecode(executable)).name, re.IGNORECASE
    ):
        return False
    if isinstance(command, (str, bytes)):
        text = os.fsdecode(command)
        if len(text) > 32768:
            raise PermissionError("OFFLINE_TEST_CHILD_COMMAND_LIMIT")
        if sys.platform == "win32":
            shell = ctypes.WinDLL("shell32", use_last_error=True)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            split = shell.CommandLineToArgvW
            split.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
            split.restype = ctypes.POINTER(ctypes.c_wchar_p)
            kernel.LocalFree.argtypes = [ctypes.c_void_p]
            kernel.LocalFree.restype = ctypes.c_void_p
            count = ctypes.c_int()
            values = split(text, ctypes.byref(count))
            if not values:
                raise PermissionError("OFFLINE_TEST_CHILD_COMMAND_UNAVAILABLE")
            try:
                tokens = [str(values[index]) for index in range(count.value)]
            finally:
                kernel.LocalFree(values)
        else:
            tokens = shlex.split(text)
    elif isinstance(command, (tuple, list)):
        tokens = [os.fsdecode(part) for part in command]
    else:
        tokens = [os.fsdecode(command)]
    program = os.fsdecode(executable) if executable is not None else tokens[0].strip('"')
    if not re.fullmatch(pattern, Path(program).name, re.IGNORECASE):
        return False
    index = 1
    while index < len(tokens):
        option = tokens[index]
        if option in {"-c", "-m", "--", "-"} or not option.startswith("-"):
            break
        if option in {"-W", "-X", "--check-hash-based-pycs"}:
            index += 2
            continue
        if option.startswith(("-c", "-m")):
            break
        if option.startswith(("-W", "-X", "--")):
            index += 1
            continue
        if set(option[1:]) & {"S", "I", "E"}:
            raise PermissionError("OFFLINE_TEST_UNGUARDED_PYTHON_STARTUP")
        index += 1
    return True


def test_boundary_event(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.sendmsg"}:
        raise PermissionError("OFFLINE_TEST_NETWORK_BLOCKED")
    if event == "subprocess.Popen":
        if require_guarded_python_startup(args[1], args[0]):
            environment = os.environ if args[3] is None else args[3]
            supplied = environment.get("PYTHONPATH", "")
            paths = supplied.split(os.pathsep) if isinstance(supplied, str) else []
            if not paths or Path(paths[0]).resolve() != Path(__file__).resolve().parent:
                raise PermissionError("OFFLINE_TEST_CHILD_STARTUP_PATH_MISSING")
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        name = Path(os.fsdecode(args[0]))
        if name.name.lower() != ".env" and name.suffix.lower() != ".dpapi":
            return
        candidate = name.resolve()
        for root in PROTECTED_ROOTS:
            if candidate == root / ".env" or candidate.is_relative_to(root / "data/testnet"):
                raise PermissionError("OFFLINE_TEST_ACTUAL_CREDENTIAL_BLOCKED")


sys.addaudithook(test_boundary_event)

# Keep the same Popen class and generic annotations. Test-specific environment
# dictionaries must not drop Python startup protection in actual subprocesses.
if not getattr(subprocess.Popen.__init__, "_cab_offline_guard", False):
    _popen_init = subprocess.Popen.__init__
    _guard_directory = str(Path(__file__).resolve().parent)

    @wraps(_popen_init)
    def guarded_popen_init(self, *positional, **keywords):
        command = positional[0] if positional else keywords["args"]
        executable = positional[2] if len(positional) > 2 else keywords.get("executable")
        require_guarded_python_startup(command, executable)
        requested = positional[10] if len(positional) > 10 else keywords.get("env")
        environment = dict(os.environ if requested is None else requested)
        supplied = environment.get("PYTHONPATH", "")
        if not isinstance(supplied, str):
            raise PermissionError("OFFLINE_TEST_CHILD_PYTHONPATH_INVALID")
        paths = [_guard_directory, *(part for part in supplied.split(os.pathsep) if part)]
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
        if len(positional) > 10:
            positional = (*positional[:10], environment, *positional[11:])
        else:
            keywords["env"] = environment
        return _popen_init(self, *positional, **keywords)

    guarded_popen_init._cab_offline_guard = True
    subprocess.Popen.__init__ = guarded_popen_init
