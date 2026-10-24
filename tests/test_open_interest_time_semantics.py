"""Explicit transaction-time semantics never repair timestamps or stale GETs."""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
from test_data_quality import assembly as assembly
from test_data_quality import corrupt, shift_source
from test_offline_estimate import OPENED

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.models.data_quality import (
    DataQualityConfig,
    DataQualityReason,
    OpenInterestTimeSemantics,
)

POLICY = DataQualityConfig(open_interest_time_semantics=OpenInterestTimeSemantics.LAST_TRANSACTION)
R = DataQualityReason


def check(assembly, config=POLICY):
    return validate_market_data(assembly, config, evaluated_at=OPENED)


def test_fresh_oi_get_keeps_old_transaction_timestamp_without_mutation(assembly):
    corrupt(assembly.sources.open_interest.data, event_time=OPENED - timedelta(minutes=5))
    original = deepcopy(assembly)
    assert check(assembly).valid
    assert assembly == original
    assert not check(assembly, DataQualityConfig()).valid  # Historical default preserved.
    assert R.STALE_OPEN_INTEREST_DATA in check(assembly, DataQualityConfig()).reason_codes


@pytest.mark.parametrize("field", ["started_at", "received_at", "finished_at"])
def test_old_request_receive_or_finish_still_invalid(assembly, field):
    corrupt(assembly.sources.open_interest, **{field: OPENED - timedelta(seconds=11)})
    assert R.STALE_OPEN_INTEREST_DATA in check(assembly).reason_codes


def test_old_observed_at_still_invalid(assembly):
    corrupt(assembly.sources.open_interest.data, observed_at=OPENED - timedelta(seconds=11))
    assert R.STALE_OPEN_INTEREST_DATA in check(assembly).reason_codes


def test_reused_old_oi_source_still_invalid(assembly):
    shift_source(assembly, "open_interest", -timedelta(seconds=11))
    assert R.STALE_OPEN_INTEREST_DATA in check(assembly).reason_codes


@pytest.mark.parametrize("value,reason", [(None, R.MISSING_TIMESTAMP),
                                        (OPENED + timedelta(seconds=1), R.FUTURE_TIMESTAMP),
                                        (OPENED.replace(tzinfo=None), R.INVALID_TIMESTAMP)])
def test_transaction_timestamp_cannot_be_missing_future_or_naive(assembly, value, reason):
    corrupt(assembly.sources.open_interest.data, event_time=value)
    assert reason in check(assembly).reason_codes


@pytest.mark.parametrize("source,reason", [("spot_depth", R.STALE_SPOT_DATA),
                                         ("perp_depth", R.STALE_PERP_DATA),
                                         ("premium_index", R.STALE_PREMIUM_DATA)])
def test_other_sources_keep_original_freshness_gates(assembly, source, reason):
    shift_source(assembly, source, -timedelta(seconds=11))
    assert reason in check(assembly).reason_codes


def test_missing_or_negative_oi_is_not_excused(assembly):
    corrupt(assembly.sources.open_interest.data, open_interest=None)
    corrupt(assembly.snapshot, open_interest=None)
    assert R.MISSING_OPEN_INTEREST in check(assembly).reason_codes


def test_testnet_explicit_semantics_and_unchanged_thresholds():
    settings = load_settings(Path("docs/testnet-bot.toml"))
    cfg = settings.research.data_quality
    assert cfg.open_interest_time_semantics is OpenInterestTimeSemantics.LAST_TRANSACTION
    assert cfg.max_data_age == timedelta(seconds=10)
    assert cfg.max_sampling_skew == timedelta(seconds=2)
    main = load_settings(Path("docs/research-soak.toml"))
    assert main.research.data_quality.open_interest_time_semantics is OpenInterestTimeSemantics.EXCHANGE_EVENT


@pytest.mark.parametrize("value", [None, True, "last_transaction", "ignore_all_times"])
def test_invalid_policy_not_silently_enabled(value):
    with pytest.raises(TypeError):
        DataQualityConfig(open_interest_time_semantics=value)
