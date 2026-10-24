"""Common-window descriptive comparisons reuse actual signed settlements."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, Inexact, localcontext

import pytest

from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.exchanges.funding_history import FundingFailure, FundingWindow

NOW = datetime(2026, 10, 8, 12, tzinfo=UTC)
D = Decimal


def event(symbol, hours, suffix):
    return FundingEvent(f"{symbol}:{suffix}", NOW - timedelta(hours=hours), D("100"), D("0.001"), symbol)


def test_common_eight_hours_compare_actual_event_counts_not_single_rate_or_hourly_extrapolation():
    from crypto_arbitrage_bot.analytics.funding_comparison import (
        FundingCostReference,
        compare_historical_funding,
    )

    references = tuple(FundingCostReference(s, D("1"), D("0.3"), f"entry-{s}", NOW-timedelta(days=2))
                       for s in ("BTCUSDT", "ETHUSDT"))
    windows = (FundingWindow("BTCUSDT", NOW-timedelta(hours=8), NOW, (event("BTCUSDT", 1, "a"),)),
               FundingWindow("ETHUSDT", NOW-timedelta(hours=8), NOW,
                             (event("ETHUSDT", 5, "a"), event("ETHUSDT", 1, "b"))))
    result = compare_historical_funding(references, windows, lookbacks=(timedelta(hours=8),), evaluated_at=NOW)
    assert result.status == "COMPLETE"
    btc, eth = (row.study.slices[0] for row in result.rows)
    assert btc.settlement_count == 1 and eth.settlement_count == 2
    assert btc.historical_short_funding_at_reference_quantity == D("0.1")
    assert eth.historical_short_funding_at_reference_quantity == D("0.2")
    assert btc.historical_funding_to_reference_cost_ratio == D("0.33333333333333333333333333333333333333333333333333")
    assert eth.historical_funding_to_reference_cost_ratio == D("0.66666666666666666666666666666666666666666666666667")
    assert result.horizons[0].ranked_symbols == ("ETHUSDT", "BTCUSDT")
    assert "not NetProfit" in result.interpretation


def test_missing_reference_cost_keeps_row_unranked_and_comparison_partial():
    from crypto_arbitrage_bot.analytics.funding_comparison import (
        FundingCostReference,
        compare_historical_funding,
    )

    refs = (FundingCostReference("BTCUSDT", D(1), None, "a", NOW),
            FundingCostReference("ETHUSDT", D(1), D("0.3"), "b", NOW))
    windows = tuple(FundingWindow(r.symbol, NOW-timedelta(hours=8), NOW, (event(r.symbol, 1, "a"),)) for r in refs)
    result = compare_historical_funding(refs, windows, lookbacks=(timedelta(hours=8),), evaluated_at=NOW)
    assert result.status == "PARTIAL"
    assert result.horizons[0].unranked_symbols == ("BTCUSDT",)
    assert result.rows[0].study.slices[0].historical_funding_to_reference_cost_ratio is None


def test_reference_cost_mapping_preserves_all_four_fees_and_costs_once_without_basis_or_funding():
    from test_forward_validation import research_record

    from crypto_arbitrage_bot.analytics.funding_comparison import cost_reference_from_opportunity
    from crypto_arbitrage_bot.storage.opportunity_records import exact_json

    original = research_record()
    before = exact_json(original)
    evidence = cost_reference_from_opportunity(original)
    # Original fixture: .204+.408+.108+.176 fees, 8 spread, 8 slippage, .1 other.
    # Its +12 mid-basis and +2.2 assumed Funding must not reduce the reference costs.
    assert evidence.reference.round_trip_costs == D("16.996")
    assert evidence.reference.quantity == D(2)
    assert evidence.fees.spot_entry == D(".204") and evidence.fees.perp_exit == D(".176")
    assert evidence.spread_costs == D(8) and evidence.slippage_costs == D(8)
    assert evidence.other_costs == D(".1")
    assert evidence.reference.opportunity_id == original.id
    assert exact_json(original) == before


@pytest.mark.parametrize("failure", [FundingFailure.TIMEOUT, FundingFailure.RATE_LIMIT, FundingFailure.INVALID_RESPONSE])
def test_failed_symbol_stays_visible_and_unranked_without_using_partial_events(failure):
    from crypto_arbitrage_bot.analytics.funding_comparison import (
        FundingCostReference,
        compare_historical_funding,
    )

    refs = tuple(FundingCostReference(s, D(1), D(".3"), s, NOW) for s in ("BTCUSDT", "ETHUSDT"))
    windows = (FundingWindow("BTCUSDT", NOW-timedelta(hours=8), NOW,
                            (event("BTCUSDT", 1, "partial"),), failure=failure),
               FundingWindow("ETHUSDT", NOW-timedelta(hours=8), NOW, (event("ETHUSDT", 1, "actual"),)))
    result = compare_historical_funding(refs, windows, lookbacks=(timedelta(hours=8),), evaluated_at=NOW)
    assert result.status == "PARTIAL" and len(result.rows) == 2
    assert result.rows[0].study.slices == ()
    assert result.horizons[0].ranked_symbols == ("ETHUSDT",) and result.horizons[0].unranked_symbols == ("BTCUSDT",)


@pytest.mark.parametrize("change", ["start", "end", "symbol", "duplicate"])
def test_mismatched_query_bounds_or_symbol_set_cannot_be_silently_compared(change):
    from crypto_arbitrage_bot.analytics.funding_comparison import (
        FundingCostReference,
        compare_historical_funding,
    )

    refs = tuple(FundingCostReference(s, D(1), D(".3"), s, NOW) for s in ("BTCUSDT", "ETHUSDT"))
    windows = [FundingWindow(r.symbol, NOW-timedelta(hours=8), NOW) for r in refs]
    if change == "start":
        windows[1] = replace(windows[1], opened_at=NOW-timedelta(hours=7))
    elif change == "end":
        windows[1] = replace(windows[1], closed_at=NOW-timedelta(seconds=1))
    elif change == "symbol":
        windows[1] = replace(windows[1], symbol="SOLUSDT")
    else:
        refs = (refs[0], refs[0])
    with pytest.raises(ValueError):
        compare_historical_funding(refs, tuple(windows), lookbacks=(timedelta(hours=8),), evaluated_at=NOW)


@pytest.mark.parametrize("rate,expected", [("-.001", "-.1"), ("0", "0"), (".001", ".1")])
def test_signed_funding_is_preserved_in_descriptive_coverage(rate, expected):
    from crypto_arbitrage_bot.analytics.funding_comparison import (
        FundingCostReference,
        compare_historical_funding,
    )

    ref = FundingCostReference("BTCUSDT", D(1), D(".3"), "a", NOW)
    actual = replace(event("BTCUSDT", 1, "a"), funding_rate=D(rate))
    window = FundingWindow("BTCUSDT", NOW-timedelta(hours=8), NOW, (actual,))
    result = compare_historical_funding((ref,), (window,), lookbacks=(timedelta(hours=8),), evaluated_at=NOW)
    assert result.rows[0].study.slices[0].historical_short_funding_at_reference_quantity == D(expected)


def test_observed_four_hour_metadata_does_not_invent_any_historical_events():
    from crypto_arbitrage_bot.analytics.funding_comparison import (
        FundingCostReference,
        compare_historical_funding,
    )
    from crypto_arbitrage_bot.models.funding_info import FundingAdjustment

    ref = FundingCostReference("BTCUSDT", D(1), D(".3"), "a", NOW)
    metadata = FundingAdjustment("BTCUSDT", D(".01"), D("-.01"), 4, NOW)
    window = FundingWindow("BTCUSDT", NOW-timedelta(days=7), NOW, (event("BTCUSDT", 1, "a"),))
    result = compare_historical_funding((ref,), (window,), lookbacks=(timedelta(days=7),), evaluated_at=NOW, adjustments=(metadata,))
    value = result.rows[0].study.slices[0]
    assert value.settlement_count == 1 and value.historical_short_funding_at_reference_quantity == D(".1")
    assert result.rows[0].observed_adjustment == metadata


def test_low_external_precision_cannot_change_ratios_or_ranking():
    with localcontext() as context:
        context.prec = 2
        context.traps[Inexact] = True
        test_common_eight_hours_compare_actual_event_counts_not_single_rate_or_hourly_extrapolation()


@pytest.mark.parametrize("cost", [None, D(0)])
def test_all_unknown_or_zero_costs_are_unavailable_for_ranking(cost):
    from crypto_arbitrage_bot.analytics.funding_comparison import (
        FundingCostReference,
        compare_historical_funding,
    )

    ref = FundingCostReference("BTCUSDT", D(1), cost, "a", NOW)
    window = FundingWindow("BTCUSDT", NOW-timedelta(hours=8), NOW)
    result = compare_historical_funding((ref,), (window,), lookbacks=(timedelta(hours=8),), evaluated_at=NOW)
    assert result.status == "UNAVAILABLE" and result.horizons[0].unranked_symbols == ("BTCUSDT",)


def test_future_reference_is_not_observed_evidence():
    from crypto_arbitrage_bot.analytics.funding_comparison import (
        FundingCostReference,
        compare_historical_funding,
    )

    ref = FundingCostReference("BTCUSDT", D(1), D(".3"), "a", NOW+timedelta(seconds=1))
    with pytest.raises(ValueError):
        compare_historical_funding((ref,), (FundingWindow("BTCUSDT", NOW-timedelta(hours=8), NOW),),
                                   lookbacks=(timedelta(hours=8),), evaluated_at=NOW)
