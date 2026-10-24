"""Read Windows process birth and liveness; no shells, command lines or signals."""

import ctypes
import os
import sys
from ctypes import wintypes
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


class NativeState(StrEnum):
    RUNNING = "RUNNING"
    EXITED = "EXITED"
    MISSING = "MISSING"
    REPLACED = "REPLACED"
    UNKNOWN = "UNKNOWN"


def _date(ticks: int) -> datetime:
    return datetime(1601, 1, 1, tzinfo=UTC) + timedelta(microseconds=ticks // 10)


@dataclass(frozen=True)
class NativeIdentity:
    pid: int
    creation_filetime: int

    def __post_init__(self) -> None:
        if type(self.pid) is not int or not 1 <= self.pid < 2**32 or (
            type(self.creation_filetime) is not int or not 0 < self.creation_filetime < 2**64
        ):
            raise ValueError("valid native process identity required")
        _date(self.creation_filetime)

    @property
    def created_at(self) -> datetime:
        return _date(self.creation_filetime)

    def payload(self) -> dict[str, Any]:
        return {"pid": self.pid, "creation_filetime": str(self.creation_filetime),
                "created_at": self.created_at.isoformat(timespec="microseconds")}

    @classmethod
    def parse(cls, value: dict[str, Any]) -> "NativeIdentity":
        ticks = value["creation_filetime"]
        if not isinstance(ticks, str) or not ticks.isascii() or not ticks.isdecimal() or len(ticks) > 20:
            raise ValueError("exact native birth ticks required")
        result = cls(value["pid"], int(ticks))
        if str(result.creation_filetime) != ticks or utc_timestamp(datetime.fromisoformat(value["created_at"])) != result.created_at:
            raise ValueError("native birth projection differs")
        return result


def _kernel() -> Any:
    if sys.platform != "win32":
        raise OSError("native Windows identity unavailable")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


def _birth(kernel: Any, handle: Any) -> int:
    values = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
        raise OSError("native birth unavailable")
    return (int(values[0].dwHighDateTime) << 32) | int(values[0].dwLowDateTime)


def current_native_identity() -> NativeIdentity:
    kernel = _kernel()
    return NativeIdentity(os.getpid(), _birth(kernel, kernel.GetCurrentProcess()))


def observe_native_identity(identity: NativeIdentity) -> NativeState:
    handle = None
    try:
        kernel = _kernel()
        # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, no write/kill rights.
        handle = kernel.OpenProcess(0x100000 | 0x1000, False, identity.pid)
        if not handle:
            return NativeState.MISSING if ctypes.get_last_error() == 87 else NativeState.UNKNOWN
        if _birth(kernel, handle) != identity.creation_filetime:
            return NativeState.REPLACED
        status = kernel.WaitForSingleObject(handle, 0)
        return {0: NativeState.EXITED, 258: NativeState.RUNNING}.get(status, NativeState.UNKNOWN)
    except Exception:
        return NativeState.UNKNOWN
    finally:
        if handle:
            kernel.CloseHandle(handle)
