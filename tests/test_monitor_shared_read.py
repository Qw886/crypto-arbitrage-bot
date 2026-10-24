"""Bounded Windows read-sharing recovery without rewriting monitor evidence."""

import ctypes
import errno
import io
import sys
from ctypes import wintypes as W
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot import monitor_archive as module
from crypto_arbitrage_bot.storage.opportunity_records import exact_json


def permission(winerror=None, number=errno.EACCES):
    error = PermissionError(number, 'private-looking OS message')
    if winerror is not None:
        error.winerror = winerror
    return error


class Stream(io.BytesIO):
    def __init__(self, body, sizes):
        super().__init__(body)
        self.sizes = sizes

    def read(self, size=-1):
        self.sizes.append(size)
        return super().read(size)


@pytest.fixture
def io_context(monkeypatch, tmp_path):
    path = tmp_path/'progress.json'
    path.write_text(exact_json({'checks':7, 'state':'RUNNING'}),encoding='utf-8')
    body = path.read_bytes()
    sleeps = Mock()
    monkeypatch.setattr(module, 'sleep', sleeps)
    monkeypatch.setattr(module, 'sys', SimpleNamespace(platform='win32'))
    return path, body, sleeps


@pytest.mark.parametrize('code', [None, 5, 32, 33])
@pytest.mark.parametrize('failures', [1, 3, 9])
def test_temporary_windows_sharing_reopens_and_returns_only_full_canonical_bytes(io_context, monkeypatch, code, failures):
    path, body, sleeps = io_context
    sizes = []
    calls = []
    streams = []
    def opened(value, mode='r', **kwargs):
        assert value == path and mode == 'rb'
        calls.append(value)
        if len(calls) <= failures:
            raise permission(code)
        stream = Stream(body, sizes)
        streams.append(stream)
        return stream
    monkeypatch.setattr(Path, 'open', opened)
    assert module.read(path) == {'checks':7, 'state':'RUNNING'}
    assert len(calls) == failures+1 and sleeps.call_count == failures
    assert all(call.args == (.01,) for call in sleeps.call_args_list)
    assert sizes == [256001] and all(stream.closed for stream in streams)


@pytest.mark.parametrize('code', [None, 5, 32, 33])
def test_persistent_windows_permission_remains_failure_after_ten_attempts(io_context, monkeypatch, code):
    path, _, sleeps = io_context
    error = permission(code)
    opened = Mock(side_effect=error)
    monkeypatch.setattr(Path, 'open', opened)
    with pytest.raises(PermissionError) as caught:
        module.read(path)
    assert caught.value is error and opened.call_count == 10 and sleeps.call_count == 9


@pytest.mark.parametrize('platform,code,number', [('linux',None,errno.EACCES), ('linux',32,errno.EACCES),
    ('win32',87,errno.EACCES), ('win32',None,errno.EPERM), ('win32',2,errno.EACCES)])
def test_other_platforms_or_unrelated_errors_propagate_immediately(io_context, monkeypatch, platform, code, number):
    path, _, sleeps = io_context
    monkeypatch.setattr(module, 'sys', SimpleNamespace(platform=platform))
    error = permission(code,number)
    opened = Mock(side_effect=error)
    monkeypatch.setattr(Path, 'open', opened)
    with pytest.raises(PermissionError) as caught:
        module.read(path)
    assert caught.value is error and opened.call_count == 1
    sleeps.assert_not_called()


@pytest.mark.parametrize('error', [FileNotFoundError('missing'), IsADirectoryError('directory'),
    OSError(errno.EIO,'disk error'), KeyboardInterrupt()])
def test_missing_disk_and_interrupt_failures_are_not_retried_or_inferred_zero(io_context, monkeypatch, error):
    path, _, sleeps = io_context
    opened = Mock(side_effect=error)
    monkeypatch.setattr(Path, 'open', opened)
    with pytest.raises(type(error)) as caught:
        module.read(path)
    assert caught.value is error and opened.call_count == 1
    sleeps.assert_not_called()


@pytest.mark.parametrize('body', [b'{',b'null',b'[]',b'{"checks": 0}',
    b'{"checks":1,"checks":2}',b'x'*256001],
    ids=['malformed','null','array','noncanonical','duplicate','oversize'])
def test_corrupt_noncanonical_or_large_documents_fail_without_retry(io_context, monkeypatch, body):
    path, _, sleeps = io_context
    sizes = []
    stream = Stream(body,sizes)
    opened = Mock(return_value=stream)
    monkeypatch.setattr(Path,'open',opened)
    with pytest.raises(ValueError):
        module.read(path)
    assert sizes == [256001] and opened.call_count == 1 and stream.closed
    sleeps.assert_not_called()


def test_read_failure_closes_old_handle_and_starts_fresh_not_partial_data(io_context, monkeypatch):
    path, body, sleeps = io_context
    class InterruptedStream(io.BytesIO):
        def read(self,size=-1):
            super().read(5)
            raise permission(32)
    first = InterruptedStream(body)
    second = Stream(body,[])
    opened = Mock(side_effect=[first,second])
    monkeypatch.setattr(Path,'open',opened)
    assert module.read(path)['checks'] == 7
    assert first.closed and second.closed and opened.call_count == 2 and sleeps.call_count == 1


@pytest.mark.skipif(sys.platform != 'win32',reason='native Windows sharing semantics')
def test_native_nonshared_handle_release_recovers_original_bytes_without_write(tmp_path,monkeypatch):
    path = tmp_path/'progress.json'
    original = exact_json({'checks':7,'state':'RUNNING'}).encode()
    path.write_bytes(original)
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes = [W.LPCWSTR,W.DWORD,W.DWORD,ctypes.c_void_p,W.DWORD,W.DWORD,W.HANDLE]
    kernel.CreateFileW.restype = W.HANDLE
    kernel.CloseHandle.argtypes = [W.HANDLE]
    kernel.CloseHandle.restype = W.BOOL
    handle = kernel.CreateFileW(str(path),0x80000000,0,None,3,0x80,None)
    assert handle not in (None,ctypes.c_void_p(-1).value)
    calls = []
    def release(seconds):
        nonlocal handle
        calls.append(seconds)
        if len(calls) == 2:
            assert kernel.CloseHandle(handle)
            handle = None
    monkeypatch.setattr(module,'sleep',release)
    try:
        assert module.read(path) == {'checks':7,'state':'RUNNING'}
        assert calls == [.01,.01]
    finally:
        if handle is not None:
            kernel.CloseHandle(handle)
    assert path.read_bytes() == original


@pytest.mark.skipif(sys.platform != 'win32',reason='native Windows sharing semantics')
def test_native_permanently_held_handle_is_still_refused(tmp_path,monkeypatch):
    path = tmp_path/'progress.json'
    original = exact_json({'checks':7}).encode()
    path.write_bytes(original)
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes = [W.LPCWSTR,W.DWORD,W.DWORD,ctypes.c_void_p,W.DWORD,W.DWORD,W.HANDLE]
    kernel.CreateFileW.restype = W.HANDLE
    kernel.CloseHandle.argtypes = [W.HANDLE]
    kernel.CloseHandle.restype = W.BOOL
    handle = kernel.CreateFileW(str(path),0x80000000,0,None,3,0x80,None)
    assert handle not in (None,ctypes.c_void_p(-1).value)
    waits = Mock()
    monkeypatch.setattr(module,'sleep',waits)
    try:
        with pytest.raises(PermissionError):
            module.read(path)
        assert waits.call_count == 9
    finally:
        kernel.CloseHandle(handle)
    assert module.read(path) == {'checks':7} and path.read_bytes() == original
