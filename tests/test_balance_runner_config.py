"""Configuration injection preserves previously accepted research parameters."""

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.config.testnet import TestnetSettings as Config

FIELDS = {'balance_evidence_enabled', 'balance_evidence_directory', 'balance_check_interval_seconds',
          'balance_check_max_attempts', 'balance_check_lease_seconds'}


def test_dedicated_configuration_keeps_financial_inputs_and_old_defaults():
    original = load_settings(Path('docs/testnet-funding-acceptance.toml'))
    updated = load_settings(Path('docs/testnet-balance-research.toml'))
    assert not original.testnet.balance_evidence_enabled
    assert original.testnet.balance_evidence_directory is None
    assert updated.testnet.balance_evidence_enabled
    assert updated.testnet.balance_evidence_directory == Path('data/testnet/balance-evidence')
    assert updated.testnet.balance_check_interval_seconds == 60
    assert updated.testnet.balance_check_max_attempts == 3
    assert updated.testnet.balance_check_lease_seconds == 300
    assert updated.research == original.research and updated.fees == original.fees
    assert updated.symbols == original.symbols and updated.poll_interval_seconds == original.poll_interval_seconds
    assert updated.testnet.model_dump(exclude=FIELDS) == original.testnet.model_dump(exclude=FIELDS)


def test_original_accepted_configuration_bytes_unchanged():
    expected = '0084f6d35a432419a2ff320026395422b4aca4024afcfc4c51790a9ed62f4685'
    assert hashlib.sha256(Path('docs/testnet-funding-acceptance.toml').read_bytes()).hexdigest() == expected


@pytest.mark.parametrize('field,value', [
    ('balance_evidence_enabled', 'true'), ('balance_evidence_enabled', 1),
    ('balance_check_max_attempts', 0), ('balance_check_max_attempts', 11),
    ('balance_check_interval_seconds', 0), ('balance_check_interval_seconds', 3601),
    ('balance_check_lease_seconds', 29), ('balance_check_lease_seconds', 3601),
])
def test_new_limits_validate_strictly(field, value):
    config = load_settings(Path('docs/testnet-balance-research.toml')).testnet.model_dump()
    config[field] = value
    with pytest.raises(ValidationError):
        Config.model_validate(config)


def test_lease_must_cover_configured_http_timeouts():
    config = load_settings(Path('docs/testnet-balance-research.toml')).testnet.model_dump()
    config.update(timeout_seconds=30, balance_check_lease_seconds=300)
    with pytest.raises(ValidationError, match='cover bounded'):
        Config.model_validate(config)


def test_configuration_cannot_enable_mainnet_execution():
    config = load_settings(Path('docs/testnet-balance-research.toml')).testnet.model_dump()
    config['environment'] = 'MAINNET'
    with pytest.raises(ValidationError):
        Config.model_validate(config)
