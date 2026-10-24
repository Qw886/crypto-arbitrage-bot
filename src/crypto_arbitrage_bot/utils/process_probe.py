"""Bounded local Windows process identity probe; never expose command lines.

Missing process, mismatched PID identity and unavailable observation differ.
This module never sends a signal or starts an exchange/worker client.
"""

import ctypes
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


class ProcessState(StrEnum):
    MATCHED = "MATCHED"
    MISSING = "MISSING"
    DIFFERENT_IDENTITY = "DIFFERENT_IDENTITY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ProcessObservation:
    state: ProcessState
    pid: int | None
    created_at: datetime | None = None


def _windows_arguments(command: str) -> tuple[str, ...]:
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    split = shell.CommandLineToArgvW
    split.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    split.restype = ctypes.POINTER(ctypes.c_wchar_p)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    count = ctypes.c_int()
    values = split(command, ctypes.byref(count))
    if not values:
        raise OSError("command identity unavailable")
    try:
        return tuple(str(values[index]) for index in range(count.value))
    finally:
        kernel.LocalFree(values)


def observe_process(
    pid: int | None,
    *,
    matches: Callable[[tuple[str, ...]], bool],
    not_before: datetime,
    not_after: datetime,
) -> ProcessObservation:
    if type(pid) is not int or not 1 <= pid <= 2**32 - 1 or sys.platform != "win32":
        return ProcessObservation(ProcessState.UNKNOWN, pid)
    try:
        # Only an integer PID enters shell code. Targets/configs/secret strings
        # are never interpolated into PowerShell. No shell=True or killing.
        command = (
            "[Console]::OutputEncoding=[Text.UTF8Encoding]::new();"
            f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' -ErrorAction Stop;"
            "if($null -eq $p){'{\"missing\":true}'}else{"
            "[ordered]@{command=$p.CommandLine;created=$p.CreationDate.ToUniversalTime().ToString('o')}"
            "|ConvertTo-Json -Compress}"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode != 0 or len(result.stdout) > 64000:
            return ProcessObservation(ProcessState.UNKNOWN, pid)
        data = json.loads(result.stdout)
        if data == {"missing": True}:
            return ProcessObservation(ProcessState.MISSING, pid)
        created = utc_timestamp(datetime.fromisoformat(data["created"]))
        if not utc_timestamp(not_before) <= created <= utc_timestamp(not_after):
            return ProcessObservation(ProcessState.DIFFERENT_IDENTITY, pid, created)
        arguments = _windows_arguments(data["command"])
        return ProcessObservation(
            ProcessState.MATCHED if matches(arguments) else ProcessState.DIFFERENT_IDENTITY,
            pid,
            created,
        )
    except Exception:
        # Timeout/permissions/malformed response are observation failure, not
        # proof the target stopped. Command/exception body is never returned.
        return ProcessObservation(ProcessState.UNKNOWN, pid)
