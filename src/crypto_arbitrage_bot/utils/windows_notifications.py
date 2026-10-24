"""Native notification-area messages; acceptance, display and clicks differ."""

import ctypes
import sys
import winreg
from collections import deque
from collections.abc import Callable
from threading import Event
from time import monotonic
from typing import Protocol


class NotificationAPI(Protocol):
    def available(self) -> bool: ...
    def open(self) -> None: ...
    def show(self, title: str, body: str) -> bool: ...
    def events(self) -> tuple[int, ...]: ...
    def close(self) -> None: ...


class _Guid(ctypes.Structure):
    _fields_ = [('a', ctypes.c_uint32), ('b', ctypes.c_uint16), ('c', ctypes.c_uint16),
                ('d', ctypes.c_ubyte * 8)]


class _IconData(ctypes.Structure):
    _fields_ = [('size', ctypes.c_uint32), ('window', ctypes.c_void_p), ('id', ctypes.c_uint32),
                ('flags', ctypes.c_uint32), ('callback', ctypes.c_uint32), ('icon', ctypes.c_void_p),
                ('tip', ctypes.c_wchar * 128), ('state', ctypes.c_uint32), ('mask', ctypes.c_uint32),
                ('body', ctypes.c_wchar * 256), ('version', ctypes.c_uint32),
                ('title', ctypes.c_wchar * 64), ('info_flags', ctypes.c_uint32),
                ('guid', _Guid), ('balloon_icon', ctypes.c_void_p)]


class _Message(ctypes.Structure):
    _fields_ = [('window', ctypes.c_void_p), ('message', ctypes.c_uint32),
                ('wparam', ctypes.c_size_t), ('lparam', ctypes.c_ssize_t), ('time', ctypes.c_uint32),
                ('x', ctypes.c_int32), ('y', ctypes.c_int32), ('private', ctypes.c_uint32)]


class WindowsNotification:
    """One message-only window and temporary icon, no visible console/dialog."""

    def __init__(self) -> None:
        if sys.platform != 'win32':
            raise OSError('WINDOWS_NOTIFICATION_API_REQUIRED')
        self.user = ctypes.WinDLL('user32', use_last_error=True)
        self.shell = ctypes.WinDLL('shell32', use_last_error=True)
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        self.kernel.GetModuleHandleW.restype = ctypes.c_void_p
        self.user.CreateWindowExW.argtypes = [ctypes.c_uint32, ctypes.c_wchar_p, ctypes.c_wchar_p,
            ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.user.CreateWindowExW.restype = ctypes.c_void_p
        self.user.DestroyWindow.argtypes = [ctypes.c_void_p]
        self.user.DestroyWindow.restype = ctypes.c_int
        self.user.LoadIconW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.user.LoadIconW.restype = ctypes.c_void_p
        self.user.PeekMessageW.argtypes = [ctypes.POINTER(_Message), ctypes.c_void_p,
                                          ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self.user.PeekMessageW.restype = ctypes.c_int
        self.user.TranslateMessage.argtypes = [ctypes.POINTER(_Message)]
        self.user.TranslateMessage.restype = ctypes.c_int
        self.user.DispatchMessageW.argtypes = [ctypes.POINTER(_Message)]
        self.user.DispatchMessageW.restype = ctypes.c_ssize_t
        setter = 'SetWindowLongPtrW' if ctypes.sizeof(ctypes.c_void_p) == 8 else 'SetWindowLongW'
        self._set_procedure = getattr(self.user, setter)
        self._set_procedure.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self._set_procedure.restype = ctypes.c_void_p
        self.user.CallWindowProcW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                            ctypes.c_size_t, ctypes.c_ssize_t]
        self.user.CallWindowProcW.restype = ctypes.c_ssize_t
        self._pending: deque[int] = deque(maxlen=64)
        self._previous: int | None = None
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_uint32,
                                          ctypes.c_size_t, ctypes.c_ssize_t)
        self._procedure = callback_type(self._receive)
        self.shell.Shell_NotifyIconW.argtypes = [ctypes.c_uint32, ctypes.POINTER(_IconData)]
        self.shell.Shell_NotifyIconW.restype = ctypes.c_int
        self.shell.SHQueryUserNotificationState.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.shell.SHQueryUserNotificationState.restype = ctypes.c_long
        self.data = _IconData()
        self.added = False

    def available(self) -> bool:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                    r'Software\Microsoft\Windows\CurrentVersion\PushNotifications') as key:
                enabled, kind = winreg.QueryValueEx(key, 'ToastEnabled')
                if kind != winreg.REG_DWORD or type(enabled) is not int or enabled != 1:
                    return False
        except FileNotFoundError:
            pass  # No explicit user override; still require the native availability query.
        except OSError:
            return False
        state = ctypes.c_int(-1)
        return self.shell.SHQueryUserNotificationState(ctypes.byref(state)) == 0 and state.value == 5

    def _receive(self, window: int, message: int, wparam: int, lparam: int) -> int:
        if message == self.data.callback:
            self._pending.append(lparam & 0xffff)
            return 0
        return int(self.user.CallWindowProcW(self._previous, window, message, wparam, lparam))

    def open(self) -> None:
        if self.data.window:
            raise ValueError('NOTIFICATION_ALREADY_OPEN')
        window = self.user.CreateWindowExW(0, 'STATIC', 'crypto-arbitrage-bot', 0, 0, 0, 0, 0,
            ctypes.c_void_p(-3), None, self.kernel.GetModuleHandleW(None), None)
        if not window:
            raise OSError('NOTIFICATION_WINDOW_FAILED')
        self.data.size = ctypes.sizeof(_IconData)
        self.data.window = window
        self.data.callback = 0x8007
        self._previous = self._set_procedure(window, -4,
            ctypes.cast(self._procedure, ctypes.c_void_p))
        if not self._previous:
            self.close()
            raise OSError('NOTIFICATION_CALLBACK_FAILED')
        self.data.id = 1
        self.data.flags = 1 | 2 | 4  # NIF_MESSAGE/ICON/TIP, no notification yet.
        self.data.callback = 0x8007
        self.data.icon = self.user.LoadIconW(None, ctypes.c_void_p(32516))
        self.data.tip = 'crypto-arbitrage-bot'
        if not self.data.icon or not self.shell.Shell_NotifyIconW(0, ctypes.byref(self.data)):
            self.close()
            raise OSError('NOTIFICATION_ICON_FAILED')
        self.added = True
        self.data.version = 4
        if not self.shell.Shell_NotifyIconW(4, ctypes.byref(self.data)):
            self.close()
            raise OSError('NOTIFICATION_VERSION_FAILED')

    def show(self, title: str, body: str) -> bool:
        self.data.flags = 0x10 | 0x40  # INFO/REALTIME: discard obsolete queued messages.
        self.data.title = title
        self.data.body = body
        self.data.info_flags = 2 | 0x10 | 0x80  # Warning, no sound, respect quiet time.
        return bool(self.shell.Shell_NotifyIconW(1, ctypes.byref(self.data)))

    def events(self) -> tuple[int, ...]:
        message = _Message()
        for _ in range(100):
            if not self.user.PeekMessageW(ctypes.byref(message), self.data.window, 0, 0, 1):
                break
            self.user.TranslateMessage(ctypes.byref(message))
            self.user.DispatchMessageW(ctypes.byref(message))
        values = tuple(self._pending)
        self._pending.clear()
        return values

    def close(self) -> None:
        deleted = not self.added or bool(self.shell.Shell_NotifyIconW(2, ctypes.byref(self.data)))
        restored = (not self._previous or not self.data.window
                    or bool(self._set_procedure(self.data.window, -4, self._previous)))
        destroyed = not self.data.window or bool(self.user.DestroyWindow(self.data.window))
        self.added = False
        if destroyed:
            self.data.window = None
            self._previous = None
            self._pending.clear()
        if not deleted or not restored or not destroyed:
            raise OSError('NOTIFICATION_CLEANUP_FAILED')


def _text(value: str, units: int) -> None:
    if (not isinstance(value, str) or not value or '\x00' in value
            or len(value.encode('utf-16-le', errors='strict')) // 2 > units):
        raise ValueError('BOUNDED_NOTIFICATION_TEXT_REQUIRED')


def notification_available() -> bool:
    return WindowsNotification().available()


def notify(title: str, body: str, *, api: NotificationAPI | None = None,
           seconds: int = 12, steady: Callable[[], float] = monotonic,
           wait: Callable[[float], object] | None = None) -> dict[str, bool]:
    _text(title, 48)
    _text(body, 200)
    if type(seconds) is not int or not 1 <= seconds <= 12:
        raise ValueError('BOUNDED_NOTIFICATION_WAIT_REQUIRED')
    native = WindowsNotification() if api is None else api
    wait = Event().wait if wait is None else wait
    result = {'api_attempted': False, 'api_accepted': False, 'shell_reported_shown': False, 'user_clicked': False,
              'shell_reported_hidden': False, 'shell_reported_timeout': False,
              'user_read_verified': False}
    try:
        if not native.available():
            return result
        native.open()
        result['api_attempted'] = True
        result['api_accepted'] = native.show(title, body)
        if not result['api_accepted']:
            return result
        deadline = steady() + seconds
        while steady() < deadline:
            for event in native.events():
                field = {0x402: 'shell_reported_shown', 0x403: 'shell_reported_hidden',
                         0x404: 'shell_reported_timeout', 0x405: 'user_clicked'}.get(event)
                if field is not None:
                    result[field] = True
            wait(max(0, min(.05, deadline - steady())))
        return result
    finally:
        native.close()
