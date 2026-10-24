"""Restore the recorded OI clock policy without weakening other freshness checks."""

import json
from dataclasses import replace
from datetime import timedelta

import pytest
from test_forward_validation import AT, OPENED, later_snapshot, research_record, validate

from crypto_arbitrage_bot.analytics.entry_evidence import decode_entry
from crypto_arbitrage_bot.models.data_quality import OpenInterestTimeSemantics as Policy
from crypto_arbitrage_bot.models.forward_validation import ValidationReason as R
from crypto_arbitrage_bot.models.forward_validation import ValidationStatus as S
from crypto_arbitrage_bot.storage.opportunity_records import exact_json


def with_policy(record, value=Policy.LAST_TRANSACTION):
    config = json.loads(record.config_json)
    quality = config["estimate"]["data_quality"]
    if value is None:
        quality.pop("open_interest_time_semantics", None)
    else:
        quality["open_interest_time_semantics"] = value
    return replace(record, config_json=exact_json(config))


def old_transaction(snapshot):
    source = snapshot.sources.open_interest
    return replace(snapshot, sources=replace(snapshot.sources, open_interest=replace(
        source, data=replace(source.data, event_time=OPENED - timedelta(seconds=20)))))


@pytest.mark.parametrize("policy", tuple(Policy))
def test_saved_quality_policy_restored_exactly(policy):
    record = with_policy(research_record(), policy)
    assert decode_entry(record).quality_config.open_interest_time_semantics is policy


def test_legacy_missing_policy_keeps_original_exchange_event_default():
    assert decode_entry(with_policy(research_record(), None)).quality_config.open_interest_time_semantics is Policy.EXCHANGE_EVENT


@pytest.mark.parametrize("value", ["unknown", 0, {}, []])
def test_malformed_policy_cannot_validate(value):
    result = validate(with_policy(research_record(), value))
    assert result.status is S.INVALID and result.reason_codes == (R.INVALID_ENTRY_EVIDENCE,)
    assert result.validated_simulated_net_profit is None


def test_fresh_current_oi_with_older_transaction_uses_saved_policy_and_same_cashflow():
    record = with_policy(research_record())
    before = exact_json(record)
    expected = validate(record)
    result = validate(record, old_transaction(later_snapshot()))
    assert result.status is S.VALIDATED and result.exit_data_quality.valid
    assert result.calculation == expected.calculation and result.fees == expected.fees
    assert result.decomposition == expected.decomposition
    assert result.validated_simulated_net_return == expected.validated_simulated_net_return
    assert exact_json(record) == before


def test_exchange_event_policy_still_rejects_older_official_oi_time():
    result = validate(with_policy(research_record(), Policy.EXCHANGE_EVENT), old_transaction(later_snapshot()))
    assert result.status is S.INVALID
    assert R.VALIDATION_TIME_NOT_AFTER_ENTRY in result.reason_codes
    assert R.INVALID_EXIT_MARKET_DATA in result.reason_codes
    assert result.validated_simulated_net_profit is None


@pytest.mark.parametrize("fault", ["stale_receive", "old_receive", "future_transaction", "old_premium", "old_perp"])
def test_last_transaction_does_not_allow_stale_or_non_subsequent_market_observations(fault):
    snapshot = old_transaction(later_snapshot())
    sources = snapshot.sources
    if fault in ("stale_receive", "old_receive"):
        timestamp = AT - timedelta(seconds=11) if fault == "stale_receive" else OPENED
        oi = replace(sources.open_interest, received_at=timestamp,
                     started_at=timestamp - timedelta(milliseconds=10), finished_at=timestamp,
                     data=replace(sources.open_interest.data, observed_at=timestamp))
        sources = replace(sources, open_interest=oi)
    elif fault == "future_transaction":
        sources = replace(sources, open_interest=replace(sources.open_interest,
            data=replace(sources.open_interest.data, event_time=AT + timedelta(seconds=1))))
    elif fault == "old_premium":
        sources = replace(sources, premium_index=replace(sources.premium_index,
            data=replace(sources.premium_index.data, event_time=OPENED)))
    else:
        source = sources.perp_depth
        sources = replace(sources, perp_depth=replace(source,
            data=replace(source.data, order_book=replace(source.data.order_book, event_time=OPENED))))
    result = validate(with_policy(research_record()), replace(snapshot, sources=sources))
    assert result.status is S.INVALID and result.validated_simulated_net_profit is None
    assert R.INVALID_EXIT_MARKET_DATA in result.reason_codes
