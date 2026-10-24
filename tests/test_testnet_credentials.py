"""Local credential setup has no exchange calls; all test values are synthetic."""

import json
import os
import sys
import warnings
from pathlib import Path
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot.execution import credential_store as store
from crypto_arbitrage_bot.execution import main as cli
from crypto_arbitrage_bot.execution.testnet_client import (
    ExecutionErrorKind,
)
from crypto_arbitrage_bot.execution.testnet_client import (
    TestnetExecutionError as GatewayError,
)


@pytest.fixture
def keys():
    return {field: "synthetic_" + field for field in store._FIELDS}


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows DPAPI round trip")
def test_actual_dpapi_round_trip_has_no_plaintext_on_disk(tmp_path, keys):
    path = tmp_path / "nested" / "keys.dpapi"
    store.save_credentials(path, keys)
    assert store.load_credentials(path) == keys
    ciphertext = path.read_bytes()
    assert ciphertext.startswith(store._HEADER)
    assert all(value.encode() not in ciphertext for value in keys.values())
    env = store.execution_environment(path, {})
    assert env["CAB_EXECUTION_MODE"] == "TESTNET"
    assert "CAB_TESTNET_ORDERS_ENABLED" not in env


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows DPAPI integrity")
def test_actual_dpapi_corrupt_ciphertext_fails_closed(tmp_path, keys):
    path = tmp_path / "keys.dpapi"
    store.save_credentials(path, keys)
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(store.CredentialStoreError, match="DECRYPTION_FAILED"):
        store.load_credentials(path)


def test_existing_file_never_overwritten(tmp_path, monkeypatch, keys):
    monkeypatch.setattr(store, "_crypt", lambda data, **kwargs: b"encrypted-placeholder")
    path = tmp_path / "existing.dpapi"
    path.write_bytes(b"original")
    with pytest.raises(store.CredentialStoreError, match="ALREADY_EXISTS"):
        store.save_credentials(path, keys)
    assert path.read_bytes() == b"original"


def test_file_credentials_do_not_modify_process_environment(tmp_path, monkeypatch, keys):
    path = tmp_path / "keys.dpapi"
    path.write_bytes(b"mock")
    loader = Mock(return_value=keys)
    monkeypatch.setattr(store, "load_credentials", loader)
    before = dict(os.environ)
    supplied = {"HTTP_PROXY": "http://proxy.invalid:1234"}
    result = store.execution_environment(path, supplied)
    assert supplied == {"HTTP_PROXY": "http://proxy.invalid:1234"}
    assert dict(os.environ) == before
    assert result["HTTP_PROXY"] == supplied["HTTP_PROXY"]
    assert result["CAB_EXECUTION_MODE"] == "TESTNET"
    assert "CAB_TESTNET_ORDERS_ENABLED" not in result


def test_environment_pairs_take_priority_without_reading_file(tmp_path, monkeypatch, keys):
    loader = Mock(side_effect=AssertionError("not read"))
    monkeypatch.setattr(store, "load_credentials", loader)
    environment = dict(keys, CAB_EXECUTION_MODE="TESTNET")
    assert store.execution_environment(tmp_path / "unused", environment) == environment
    loader.assert_not_called()


@pytest.mark.parametrize("field", store._FIELDS)
def test_partial_environment_does_not_mix_with_stored_credentials(tmp_path, monkeypatch, keys, field):
    loader = Mock(side_effect=AssertionError("not read"))
    monkeypatch.setattr(store, "load_credentials", loader)
    with pytest.raises(GatewayError) as error:
        store.execution_environment(tmp_path / "unused", {field: keys[field]})
    assert error.value.kind is ExecutionErrorKind.CREDENTIALS_MISSING
    loader.assert_not_called()


def test_explicit_other_mode_is_not_overridden(tmp_path, monkeypatch, keys):
    path = tmp_path / "keys.dpapi"
    path.write_bytes(b"mock")
    monkeypatch.setattr(store, "load_credentials", lambda _: keys)
    assert store.execution_environment(path, {"CAB_EXECUTION_MODE": "DISABLED"})["CAB_EXECUTION_MODE"] == "DISABLED"


def test_no_file_no_keys_remains_disabled(tmp_path):
    assert store.execution_environment(tmp_path / "absent", {}) == {}


@pytest.mark.parametrize("payload", [b"", b"plaintext-secret", b"CAB-TESTNET-DPAPI-2\nblob", b"X" * 65537],
                         ids=["empty", "plaintext", "wrong-version", "oversize"])
def test_malformed_or_large_file_does_not_decrypt(tmp_path, monkeypatch, payload):
    path = tmp_path / "bad.dpapi"
    path.write_bytes(payload)
    decrypt = Mock(side_effect=AssertionError("never decrypt invalid format"))
    monkeypatch.setattr(store, "_crypt", decrypt)
    with pytest.raises(store.CredentialStoreError, match="INVALID_TESTNET_CREDENTIAL_FILE"):
        store.load_credentials(path)
    decrypt.assert_not_called()


@pytest.mark.parametrize("payload", [b"not-json", b"{}", b"[]", b'{"production_key":"sensitive"}'])
def test_decrypted_malformed_payload_returns_only_safe_message(tmp_path, monkeypatch, payload):
    path = tmp_path / "bad.dpapi"
    path.write_bytes(store._HEADER + b"mock")
    monkeypatch.setattr(store, "_crypt", lambda *args, **kwargs: payload)
    with pytest.raises(store.CredentialStoreError) as error:
        store.load_credentials(path)
    assert str(error.value) == "INVALID_TESTNET_CREDENTIAL_FILE"


def test_invalid_key_shape_never_reaches_encryption(tmp_path, monkeypatch, keys):
    keys[store._FIELDS[0]] = "header\ninjection"
    crypt = Mock()
    monkeypatch.setattr(store, "_crypt", crypt)
    with pytest.raises(store.CredentialStoreError):
        store.save_credentials(tmp_path / "keys", keys)
    crypt.assert_not_called()


def test_encryption_failure_never_writes_plaintext(tmp_path, monkeypatch, keys):
    monkeypatch.setattr(store, "_crypt", Mock(side_effect=store.CredentialStoreError("CREDENTIAL_ENCRYPTION_FAILED")))
    path = tmp_path / "keys.dpapi"
    with pytest.raises(store.CredentialStoreError):
        store.save_credentials(path, keys)
    assert not path.exists()


def test_write_failure_is_structured(tmp_path, monkeypatch, keys):
    monkeypatch.setattr(store, "_crypt", lambda *args, **kwargs: b"ciphertext")
    monkeypatch.setattr(Path, "open", Mock(side_effect=PermissionError("sensitive")))
    with pytest.raises(store.CredentialStoreError, match="^CREDENTIAL_FILE_WRITE_FAILED$"):
        store.save_credentials(tmp_path / "keys", keys)


def test_read_failure_is_structured(tmp_path):
    with pytest.raises(store.CredentialStoreError, match="^CREDENTIAL_FILE_READ_FAILED$"):
        store.load_credentials(tmp_path / "absent")


def test_noninteractive_setup_never_prompts_or_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(store.sys, "platform", "win32")
    monkeypatch.setattr(store.sys.stdin, "isatty", lambda: False)
    prompt = Mock(side_effect=AssertionError("no echo fallback"))
    monkeypatch.setattr(store.getpass, "getpass", prompt)
    with pytest.raises(store.CredentialStoreError, match="INTERACTIVE_TERMINAL_REQUIRED"):
        store.setup_credentials(tmp_path / "keys")
    prompt.assert_not_called()


def test_setup_uses_hidden_prompts_and_does_not_print_credentials(tmp_path, monkeypatch, keys, capsys):
    monkeypatch.setattr(store.sys, "platform", "win32")
    monkeypatch.setattr(store.sys.stdin, "isatty", lambda: True)
    prompt = Mock(side_effect=list(keys.values()))
    writer = Mock()
    monkeypatch.setattr(store.getpass, "getpass", prompt)
    monkeypatch.setattr(store, "save_credentials", writer)
    path = tmp_path / "keys.dpapi"
    store.setup_credentials(path)
    writer.assert_called_once_with(path, keys)
    assert prompt.call_count == 4
    output = capsys.readouterr().out
    assert all(value not in output for value in keys.values())


def test_non_windows_file_setup_refuses_plaintext_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(store.sys, "platform", "linux")
    with pytest.raises(store.CredentialStoreError, match="WINDOWS_DPAPI_REQUIRED"):
        store.setup_credentials(tmp_path / "keys")


def test_hidden_input_warning_is_error_before_visible_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(store.sys, "platform", "win32")
    monkeypatch.setattr(store.sys.stdin, "isatty", lambda: True)
    def unavailable(prompt):
        warnings.warn("No hidden input", store.getpass.GetPassWarning, stacklevel=1)
        raise AssertionError("must stop before an echoing fallback")
    monkeypatch.setattr(store.getpass, "getpass", unavailable)
    with pytest.raises(store.CredentialStoreError, match="HIDDEN_INPUT_UNAVAILABLE"):
        store.setup_credentials(tmp_path / "keys")
    assert not (tmp_path / "keys").exists()


def test_cli_setup_does_not_load_research_or_construct_clients(tmp_path, monkeypatch):
    path = tmp_path / "keys.dpapi"
    monkeypatch.setattr(sys, "argv", ["testnet", "--setup-credentials", "--credentials-file", str(path)])
    setup = Mock()
    settings = Mock(side_effect=AssertionError("no research config required"))
    factory = Mock(side_effect=AssertionError("no network/client"))
    monkeypatch.setattr(cli, "setup_credentials", setup)
    monkeypatch.setattr(cli, "load_settings", settings)
    monkeypatch.setattr(cli, "create_testnet_bot", factory)
    assert cli.main() == 0
    setup.assert_called_once_with(path)
    settings.assert_not_called()
    factory.assert_not_called()


def test_cli_setup_error_is_sanitized(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["testnet", "--setup-credentials"])
    monkeypatch.setattr(cli, "setup_credentials", Mock(side_effect=store.CredentialStoreError("CREDENTIAL_DECRYPTION_FAILED")))
    assert cli.main() == 1
    result = json.loads(capsys.readouterr().out)
    assert result["reason"] == "CREDENTIAL_DECRYPTION_FAILED"


def test_stored_credentials_cannot_bypass_virtual_order_gate(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["testnet", "--round-trip"])
    monkeypatch.delenv("CAB_TESTNET_ORDERS_ENABLED", raising=False)
    loader = Mock(side_effect=AssertionError("no decrypt before gate"))
    factory = Mock(side_effect=AssertionError("no client before gate"))
    monkeypatch.setattr(cli, "execution_environment", loader)
    monkeypatch.setattr(cli, "create_testnet_bot", factory)
    assert cli.main() == 2
    assert "TESTNET_ORDERS_DISABLED" in capsys.readouterr().out
    loader.assert_not_called()
    factory.assert_not_called()
