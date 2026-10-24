"""Process-owned SYSTEM_REQUIRED request; no display, power-plan or registry edits."""

import ctypes
import sys
from collections.abc import Iterator
from contextlib import contextmanager


class _Detailed(ctypes.Structure):
    _fields_ = [("module", ctypes.c_void_p), ("id", ctypes.c_uint32),
                ("count", ctypes.c_uint32), ("strings", ctypes.c_void_p)]


class _Reason(ctypes.Union):
    _fields_ = [("detailed", _Detailed), ("simple", ctypes.c_wchar_p)]


class _Context(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("reason", _Reason)]


class _PowerStatus(ctypes.Structure):
    _fields_ = [("ac", ctypes.c_ubyte), ("battery", ctypes.c_ubyte),
                ("percent", ctypes.c_ubyte), ("saver", ctypes.c_ubyte),
                ("remaining", ctypes.c_uint32), ("full", ctypes.c_uint32)]


class WindowsPower:
    """Failure is explicit; an API acknowledgement is not network/sleep proof."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("WINDOWS_POWER_API_REQUIRED")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.GetSystemPowerStatus.argtypes = [ctypes.POINTER(_PowerStatus)]
        self.kernel.GetSystemPowerStatus.restype = ctypes.c_int
        self.kernel.PowerCreateRequest.argtypes = [ctypes.POINTER(_Context)]
        self.kernel.PowerCreateRequest.restype = ctypes.c_void_p
        self.kernel.PowerSetRequest.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.kernel.PowerSetRequest.restype = ctypes.c_int
        self.kernel.PowerClearRequest.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.kernel.PowerClearRequest.restype = ctypes.c_int
        self.kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel.CloseHandle.restype = ctypes.c_int

    def ac_available(self) -> bool:
        status = _PowerStatus()
        if not self.kernel.GetSystemPowerStatus(ctypes.byref(status)):
            raise OSError("POWER_STATUS_UNAVAILABLE")
        # Unknown AC or battery-saver status is not permission to inhibit sleep.
        return bool(status.ac == 1 and status.saver == 0)

    @contextmanager
    def system_request(self, reason: str) -> Iterator[None]:
        if not 1 <= len(reason) <= 200 or "\x00" in reason:
            raise ValueError("BOUNDED_POWER_REASON_REQUIRED")
        if not self.ac_available():
            raise OSError("AC_POWER_REQUIRED")
        context = _Context(0, 1, _Reason(simple=reason))
        handle = self.kernel.PowerCreateRequest(ctypes.byref(context))
        if not handle or handle == ctypes.c_void_p(-1).value:
            raise OSError("POWER_CREATE_FAILED")
        active = False
        try:
            if not self.kernel.PowerSetRequest(handle, 1):  # PowerRequestSystemRequired
                raise OSError("POWER_SET_FAILED")
            active = True
            yield
        finally:
            # Close even if clearing fails; process exit also destroys the handle.
            cleared = not active or bool(self.kernel.PowerClearRequest(handle, 1))
            closed = bool(self.kernel.CloseHandle(handle))
            if not cleared or not closed:
                raise OSError("POWER_RELEASE_FAILED")
