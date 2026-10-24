import ctypes
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from crypto_arbitrage_bot.utils import windows_notifications as module
from crypto_arbitrage_bot.utils.windows_notifications import WindowsNotification, _IconData, notify


class FakeAPI:
    def __init__(self, accepted=True, events=(), fail=None):
        self.accepted, self.messages, self.fail = accepted, events, fail
        self.calls = []

    def open(self):
        self.calls.append('open')
        if self.fail == 'open':
            raise OSError('private detail')

    def available(self):
        return True

    def show(self, title, body):
        self.calls.append('show')
        if self.fail == 'show':
            raise OSError('private detail')
        return self.accepted

    def events(self):
        value, self.messages = self.messages, ()
        return value

    def close(self):
        self.calls.append('close')
        if self.fail == 'close':
            raise OSError('cleanup failure')


def run(api):
    ticks = iter([0, 0, 0, 2])
    return notify('提醒', '本次检查需要查看状态。', api=api, seconds=1,
                  steady=lambda: next(ticks), wait=lambda delay: None)


def test_native_windows_x64_layout():
    assert ctypes.sizeof(ctypes.c_void_p) == 8
    assert ctypes.sizeof(_IconData) == 976


def test_acceptance_does_not_claim_display_or_reading():
    api = FakeAPI()
    result = run(api)
    assert result['api_accepted']
    assert not result['shell_reported_shown'] and not result['user_read_verified']
    assert api.calls == ['open', 'show', 'close']


def test_display_and_click_are_distinct_from_reading():
    result = run(FakeAPI(events=(0x402, 0x405, 999)))
    assert result['shell_reported_shown'] and result['user_clicked']
    assert not result['user_read_verified']


@pytest.mark.parametrize('event,key', [(0x403,'shell_reported_hidden'), (0x404,'shell_reported_timeout')])
def test_suppression_and_timeout_are_not_display(event, key):
    result = run(FakeAPI(events=(event,)))
    assert result[key] and not result['shell_reported_shown']


def test_api_rejection_still_cleans_up():
    api = FakeAPI(accepted=False)
    result = run(api)
    assert not result['api_accepted'] and api.calls[-1]=='close'


def test_busy_os_does_not_create_icon_or_send():
    api = FakeAPI()
    api.available = lambda: False
    result = run(api)
    assert not result['api_attempted'] and not result['api_accepted']
    assert api.calls == ['close']


@pytest.mark.parametrize('failure', ['open','show','close'])
def test_native_errors_and_cleanup_are_explicit(failure):
    api = FakeAPI(fail=failure)
    with pytest.raises(OSError):
        run(api)
    assert api.calls[-1]=='close'


@pytest.mark.parametrize('title,body', [('', 'x'), ('a\x00b','x'), ('a'*49,'x'),
                                      ('x','b'*201), ('x','\U0001f600'*101)])
def test_utf16_bounds_before_native_api(title, body):
    api = FakeAPI()
    with pytest.raises(ValueError):
        notify(title,body,api=api)
    assert api.calls==[]


@pytest.mark.parametrize('seconds', [0,13,True,1.5])
def test_finite_wait_before_native_api(seconds):
    api = FakeAPI()
    with pytest.raises(ValueError):
        notify('x','y',api=api,seconds=seconds)
    assert api.calls==[]


def receiver():
    native = WindowsNotification.__new__(WindowsNotification)
    native.data = _IconData()
    native.data.callback = 0x8007
    native.data.window = 11
    native._pending = deque(maxlen=64)
    native._previous = 22
    return native


def test_sent_callback_is_received_even_without_a_queued_message():
    native = receiver()
    def peek(*args):
        native._receive(11, 0x8007, 0, (1 << 16) | 0x402)
        return 0
    native.user = SimpleNamespace(PeekMessageW=peek)
    assert native.events() == (0x402,)


def test_posted_callback_is_dispatched_once_and_consumed():
    native = receiver()
    peeks = iter([1, 0, 0])
    native.user = SimpleNamespace(PeekMessageW=lambda *args: next(peeks),
        TranslateMessage=lambda *args: 1,
        DispatchMessageW=lambda *args: native._receive(11, 0x8007, 0, 0x405))
    assert native.events() == (0x405,)
    assert native.events() == ()


def test_other_window_messages_are_forwarded_to_original_procedure():
    native = receiver()
    calls = []
    def forward(*args):
        calls.append(args)
        return 77
    native.user = SimpleNamespace(CallWindowProcW=forward)
    assert native._receive(11, 123, 4, 5) == 77
    assert calls == [(22, 11, 123, 4, 5)] and not native._pending


def test_callback_buffer_is_bounded():
    native = receiver()
    for number in range(1000):
        native._receive(11, 0x8007, 0, number)
    assert len(native._pending) == 64
    assert tuple(native._pending) == tuple(range(936, 1000))


@pytest.mark.parametrize('value,kind,expected', [
    (0, module.winreg.REG_DWORD, False), (1, module.winreg.REG_DWORD, True),
    (2, module.winreg.REG_DWORD, False), ('1', module.winreg.REG_SZ, False),
])
def test_global_user_setting_is_checked_before_native_state(monkeypatch, value, kind, expected):
    native = receiver()
    calls = []
    def query_state(pointer):
        calls.append('state')
        pointer._obj.value = 5
        return 0
    native.shell = SimpleNamespace(SHQueryUserNotificationState=query_state)
    monkeypatch.setattr(module.winreg, 'OpenKey', lambda *args: nullcontext('key'))
    monkeypatch.setattr(module.winreg, 'QueryValueEx', lambda *args: (value, kind))
    assert native.available() is expected
    assert calls == (['state'] if expected else [])


@pytest.mark.parametrize('error,expected', [(FileNotFoundError(), True), (PermissionError(), False)])
def test_missing_override_and_unreadable_global_state_differ(monkeypatch, error, expected):
    native = receiver()
    def unavailable(*args):
        raise error
    def query_state(pointer):
        pointer._obj.value = 5
        return 0
    monkeypatch.setattr(module.winreg, 'OpenKey', unavailable)
    native.shell = SimpleNamespace(SHQueryUserNotificationState=query_state)
    assert native.available() is expected


def test_callback_is_restored_before_window_destruction():
    native = receiver()
    native.added = True
    calls = []
    native.shell = SimpleNamespace(Shell_NotifyIconW=lambda *args: calls.append('delete') or 1)
    native._set_procedure = lambda *args: calls.append('restore') or 99
    native.user = SimpleNamespace(DestroyWindow=lambda *args: calls.append('destroy') or 1)
    native.close()
    assert calls == ['delete', 'restore', 'destroy']
    assert native._previous is None and native.data.window is None


def test_failed_restoration_still_destroys_window_and_reports_cleanup_failure():
    native = receiver()
    native.added = False
    calls = []
    native._set_procedure = lambda *args: calls.append('restore') or 0
    native.user = SimpleNamespace(DestroyWindow=lambda *args: calls.append('destroy') or 1)
    with pytest.raises(OSError, match='NOTIFICATION_CLEANUP_FAILED'):
        native.close()
    assert calls == ['restore', 'destroy']
