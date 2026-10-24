"""Explicit testnet credential setup, encrypted for the current Windows user.

The public research entry point never imports this module. DPAPI is not a
defence against software already running as the same Windows user.
"""

import ctypes
import getpass
import json
import os
import sys
import warnings
from collections.abc import Mapping
from ctypes import wintypes
from pathlib import Path

from crypto_arbitrage_bot.execution.testnet_client import TestnetCredentials
from crypto_arbitrage_bot.execution.types import ExecutionMarket

DEFAULT_CREDENTIAL_FILE = Path("data/testnet/credentials.dpapi")
_HEADER = b"CAB-TESTNET-DPAPI-1\n"
_MAX_FILE_BYTES = 65536
_FIELDS = (
    "CAB_TESTNET_SPOT_API_KEY", "CAB_TESTNET_SPOT_API_SECRET",
    "CAB_TESTNET_USDTM_API_KEY", "CAB_TESTNET_USDTM_API_SECRET",
)


class CredentialStoreError(RuntimeError):
    """A stable message without a credential, plaintext payload or file contents."""


class _DataBlob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _crypt(data: bytes, *, decrypt: bool) -> bytes:
    if sys.platform != "win32":
        raise CredentialStoreError("WINDOWS_DPAPI_REQUIRED: use testnet environment variables on other systems")
    # User-scoped encryption; no machine-wide scope and no UI/prompt fallback.
    library = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    function = library.CryptUnprotectData if decrypt else library.CryptProtectData
    function.argtypes = [
        ctypes.POINTER(_DataBlob), ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob),
    ]
    function.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    source = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = _DataBlob()
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise CredentialStoreError("CREDENTIAL_ENCRYPTION_FAILED" if not decrypt else "CREDENTIAL_DECRYPTION_FAILED")
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        kernel.LocalFree(ctypes.cast(result.data, ctypes.c_void_p))


def _validated(values: object) -> dict[str, str]:
    if not isinstance(values, dict) or set(values) != set(_FIELDS):
        raise CredentialStoreError("INVALID_TESTNET_CREDENTIAL_FILE")
    if any(type(value) is not str for value in values.values()):
        raise CredentialStoreError("INVALID_TESTNET_CREDENTIAL_FILE")
    try:
        for market in ExecutionMarket:
            TestnetCredentials.from_env(market, values)
    except Exception:
        raise CredentialStoreError("INVALID_TESTNET_CREDENTIAL_FILE") from None
    return dict(values)


def save_credentials(path: Path, values: Mapping[str, str]) -> None:
    """Never overwrite an existing file; never write a plaintext temporary file."""
    checked = _validated(dict(values))
    plaintext = json.dumps(checked, ensure_ascii=True).encode("utf-8")
    if len(plaintext) > _MAX_FILE_BYTES // 2:
        raise CredentialStoreError("CREDENTIAL_FILE_TOO_LARGE")
    encrypted = _HEADER + _crypt(plaintext, decrypt=False)
    if len(encrypted) > _MAX_FILE_BYTES:
        raise CredentialStoreError("CREDENTIAL_FILE_TOO_LARGE")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(encrypted)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise CredentialStoreError("CREDENTIAL_FILE_ALREADY_EXISTS: existing credentials were not overwritten") from None
    except OSError:
        # A partial encrypted file fails closed on load; never expose plaintext.
        raise CredentialStoreError("CREDENTIAL_FILE_WRITE_FAILED") from None


def load_credentials(path: Path) -> dict[str, str]:
    try:
        with path.open("rb") as stream:
            payload = stream.read(_MAX_FILE_BYTES + 1)
    except OSError:
        raise CredentialStoreError("CREDENTIAL_FILE_READ_FAILED") from None
    if len(payload) > _MAX_FILE_BYTES or not payload.startswith(_HEADER):
        raise CredentialStoreError("INVALID_TESTNET_CREDENTIAL_FILE")
    plaintext = _crypt(payload[len(_HEADER):], decrypt=True)
    try:
        values = json.loads(plaintext)
    except (ValueError, UnicodeError):
        raise CredentialStoreError("INVALID_TESTNET_CREDENTIAL_FILE") from None
    return _validated(values)


def execution_environment(path: Path, environ: Mapping[str, str]) -> dict[str, str]:
    """Complete environment pairs override the file; partial pairs fail closed.

    Loading the explicit testnet store only enables testnet account access. It
    never enables the separate virtual-order gate or modifies os.environ.
    """
    environment = dict(environ)
    if any(key in environment for key in _FIELDS):
        for market in ExecutionMarket:
            TestnetCredentials.from_env(market, environment)
        return environment
    if path.exists():
        environment.update(load_credentials(path))
        environment.setdefault("CAB_EXECUTION_MODE", "TESTNET")
    return environment


def setup_credentials(path: Path) -> None:
    if sys.platform != "win32":
        raise CredentialStoreError("WINDOWS_DPAPI_REQUIRED")
    if not sys.stdin.isatty():
        raise CredentialStoreError("INTERACTIVE_TERMINAL_REQUIRED: credential input must be hidden")
    if path.exists():
        raise CredentialStoreError("CREDENTIAL_FILE_ALREADY_EXISTS: existing credentials were not overwritten")
    print("TESTNET ONLY: enter HMAC keys from Spot Testnet and USDT-M Demo. No network requests or orders.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            values = {name: getpass.getpass(name + " (hidden): ") for name in _FIELDS}
    except getpass.GetPassWarning:
        raise CredentialStoreError("HIDDEN_INPUT_UNAVAILABLE: refusing visible credential input") from None
    save_credentials(path, values)
    print("TESTNET_CREDENTIALS_SAVED: encrypted for this Windows user; orders remain disabled.")
