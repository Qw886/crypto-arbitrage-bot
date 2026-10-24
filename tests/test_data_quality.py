"""Offline quality checks over complete, incomplete and deliberately corrupt data."""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest
from test_offline_estimate import OPENED, SETTLED, assembly_for, books, config, scenario_for

from crypto_arbitrage_bot.arbitrage import estimate as engine
from crypto_arbitrage_bot.arbitrage.profit import decomposed_net_profit, net_return
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import prepare_get
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.market_data.snapshot import assemble_snapshot
from crypto_arbitrage_bot.models.data_quality import (
    DataQualityConfig,
    DataQualityReason,
    DataQualityStatus,
)
from crypto_arbitrage_bot.models.estimate import EstimateStatus
from crypto_arbitrage_bot.models.spot import SpotErrorKind, SpotFetchError

D = Decimal
R = DataQualityReason


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    monkeypatch.setattr(
        "socket.create_connection", Mock(side_effect=AssertionError("quality tests are offline")),
    )


@pytest.fixture
def assembly():
    source_books = books()
    return deepcopy(assembly_for((source_books[0], source_books[2])))


def corrupt(target, **changes):
    """Bypass constructor guards only to exercise the defensive input boundary."""
    for name, value in changes.items():
        object.__setattr__(target, name, value)


def validate(assembly, **kwargs):
    return validate_market_data(assembly, evaluated_at=OPENED, **kwargs)


def assert_invalid(result, *codes):
    assert result.status is DataQualityStatus.INVALID and not result.valid
    assert set(codes) <= set(result.reason_codes)
    assert len(result.reason_codes) == len(set(result.reason_codes))
    assert all(issue.source and issue.field for issue in result.issues)


def shift_source(assembly, source_name, delta):
    """Move a source's actual times together without inventing Spot event time."""
    source = getattr(assembly.sources, source_name)
    for name in ("started_at", "received_at", "finished_at"):
        corrupt(source, **{name: getattr(source, name) + delta})
    if source_name == "spot_depth":
        observed = source.order_book
    elif source_name == "perp_depth":
        observed = source.data.order_book
        corrupt(source.data, transaction_time=source.data.transaction_time + delta)
    else:
        observed = source.data
    corrupt(observed, observed_at=observed.observed_at + delta)
    if observed.event_time is not None:
        corrupt(observed, event_time=observed.event_time + delta)


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_complete_snapshot_is_valid_without_mutating_provenance(symbol):
    source_books = books(symbol)
    assembly = assembly_for((source_books[0], source_books[2]))
    before = deepcopy(assembly)
    result = validate(assembly)
    assert result.valid and result.status is DataQualityStatus.VALID
    assert result.issues == result.reason_codes == ()
    assert result.evaluated_at == OPENED
    assert result.sampling_skew == timedelta(0)
    assert assembly == before
    assert assembly.snapshot.spot_book.event_time is None
    assert assembly.snapshot.perp_book.event_time == OPENED


@pytest.mark.parametrize(
    "source_name,code",
    [
        ("spot_depth", R.MISSING_SPOT_DEPTH),
        ("perp_depth", R.MISSING_PERP_DEPTH),
        ("premium_index", R.MISSING_FUNDING),
        ("open_interest", R.MISSING_OPEN_INTEREST),
    ],
)
def test_missing_source_is_invalid(assembly, source_name, code):
    incomplete = assemble_snapshot(
        assembly.symbol, replace(assembly.sources, **{source_name: None}), assembled_at=OPENED,
    )
    assert_invalid(validate(incomplete), R.INCOMPLETE_SNAPSHOT, code)


@pytest.mark.parametrize(
    "field,code",
    [
        ("mark_price", R.MISSING_MARK_PRICE),
        ("index_price", R.MISSING_INDEX_PRICE),
        ("funding_rate", R.MISSING_FUNDING),
        ("next_funding_time", R.MISSING_NEXT_FUNDING_TIME),
        ("open_interest", R.MISSING_OPEN_INTEREST),
    ],
)
def test_missing_unified_field_is_invalid_even_when_source_is_complete(assembly, field, code):
    corrupt(assembly.snapshot, **{field: None})
    assert_invalid(validate(assembly), code)


@pytest.mark.parametrize(
    "book_name,side,code",
    [
        ("spot_book", "bids", R.MISSING_SPOT_BID),
        ("spot_book", "asks", R.MISSING_SPOT_ASK),
        ("perp_book", "bids", R.MISSING_PERP_BID),
        ("perp_book", "asks", R.MISSING_PERP_ASK),
    ],
)
def test_empty_depth_side_has_no_usable_best_price(assembly, book_name, side, code):
    corrupt(getattr(assembly.snapshot, book_name), **{side: ()})
    assert_invalid(validate(assembly), code)


@pytest.mark.parametrize(
    "book_name,code", [("spot_book", R.CROSSED_SPOT_BOOK), ("perp_book", R.CROSSED_PERP_BOOK)],
)
def test_crossed_books_are_invalid(assembly, book_name, code):
    book = getattr(assembly.snapshot, book_name)
    corrupt(book.bids[0], price=D("1000"))
    assert_invalid(validate(assembly), code)


@pytest.mark.parametrize("value", [D("0"), D("-1"), D("NaN"), D("Infinity"), 1.5])
@pytest.mark.parametrize("field,code", [("price", R.INVALID_PRICE), ("quantity", R.INVALID_QUANTITY)])
def test_invalid_order_book_numbers_are_rejected(assembly, field, code, value):
    # A deeper level must be checked too, not just the best bid and ask.
    corrupt(assembly.snapshot.spot_book.asks[1], **{field: value})
    assert_invalid(validate(assembly), code)


@pytest.mark.parametrize("field", ["mark_price", "index_price"])
@pytest.mark.parametrize("value", [D("0"), D("-1")])
def test_mark_and_index_prices_must_be_positive(assembly, field, value):
    corrupt(assembly.snapshot, **{field: value})
    corrupt(assembly.sources.premium_index.data, **{field: value})
    assert_invalid(validate(assembly), R.INVALID_PRICE)


def test_unsorted_book_is_rejected(assembly):
    corrupt(assembly.snapshot.spot_book, asks=tuple(reversed(assembly.snapshot.spot_book.asks)))
    assert_invalid(validate(assembly), R.UNSORTED_BOOK)


@pytest.mark.parametrize("funding", [D("0.01"), D("0"), D("-0.01")])
def test_funding_sign_is_not_a_data_error(assembly, funding):
    corrupt(assembly.snapshot, funding_rate=funding)
    corrupt(assembly.sources.premium_index.data, last_funding_rate=funding)
    assert validate(assembly).valid


@pytest.mark.parametrize("funding", [D("NaN"), D("Infinity"), 0.01])
def test_nonfinite_or_float_funding_is_invalid(assembly, funding):
    corrupt(assembly.snapshot, funding_rate=funding)
    corrupt(assembly.sources.premium_index.data, last_funding_rate=funding)
    assert_invalid(validate(assembly), R.INVALID_FUNDING)


@pytest.mark.parametrize("oi", [D("-1"), D("NaN"), D("Infinity"), 1.5])
def test_invalid_open_interest(assembly, oi):
    corrupt(assembly.snapshot, open_interest=oi)
    corrupt(assembly.sources.open_interest.data, open_interest=oi)
    assert_invalid(validate(assembly), R.INVALID_OPEN_INTEREST)


def test_zero_open_interest_is_valid(assembly):
    corrupt(assembly.snapshot, open_interest=D("0"))
    corrupt(assembly.sources.open_interest.data, open_interest=D("0"))
    assert validate(assembly).valid


@pytest.mark.parametrize("next_time", [None, 0, OPENED, OPENED - timedelta(seconds=1), SETTLED.replace(tzinfo=None)])
def test_next_funding_time_must_exist_and_be_future_aware_time(assembly, next_time):
    corrupt(assembly.snapshot, next_funding_time=next_time)
    corrupt(assembly.sources.premium_index.data, next_funding_time=next_time)
    code = R.MISSING_NEXT_FUNDING_TIME if next_time is None else R.INVALID_NEXT_FUNDING_TIME
    assert_invalid(validate(assembly), code)


@pytest.mark.parametrize("location", ["spot_book", "perp_book", "premium", "oi", "snapshot", "request"])
def test_symbols_must_match_at_every_layer(assembly, location):
    if location in ("spot_book", "perp_book"):
        corrupt(getattr(assembly.snapshot, location), symbol="ETHUSDT")
    elif location == "premium":
        corrupt(assembly.sources.premium_index.data, symbol="ETHUSDT")
    elif location == "oi":
        corrupt(assembly.sources.open_interest.data, symbol="ETHUSDT")
    elif location == "snapshot":
        corrupt(assembly.snapshot, symbol="ETHUSDT")
    else:
        corrupt(assembly.sources.spot_depth, request=prepare_get(
            PublicEndpoint.SPOT_DEPTH, {"symbol": "ETHUSDT", "limit": 100},
        ))
    assert_invalid(validate(assembly), R.SYMBOL_MISMATCH)


@pytest.mark.parametrize(
    "source_name,code",
    [
        ("spot_depth", R.STALE_SPOT_DATA),
        ("perp_depth", R.STALE_PERP_DATA),
        ("premium_index", R.STALE_PREMIUM_DATA),
        ("open_interest", R.STALE_OPEN_INTEREST_DATA),
    ],
)
def test_stale_sources_are_invalid(assembly, source_name, code):
    shift_source(assembly, source_name, -timedelta(seconds=11))
    assert_invalid(validate(assembly), code)


@pytest.mark.parametrize(
    "source_name,code",
    [
        ("perp_depth", R.STALE_PERP_DATA),
        ("premium_index", R.STALE_PREMIUM_DATA),
        ("open_interest", R.STALE_OPEN_INTEREST_DATA),
    ],
)
def test_fresh_receive_does_not_hide_stale_official_time(assembly, source_name, code):
    source = getattr(assembly.sources, source_name)
    data = source.data.order_book if source_name == "perp_depth" else source.data
    corrupt(data, event_time=OPENED - timedelta(seconds=11))
    assert source.received_at == OPENED
    assert_invalid(validate(assembly), code)


def test_stale_perp_transaction_time_is_also_checked(assembly):
    corrupt(assembly.sources.perp_depth.data, transaction_time=OPENED - timedelta(seconds=11))
    assert_invalid(validate(assembly), R.STALE_PERP_DATA)


@pytest.mark.parametrize("seconds,valid", [(2, True), (3, False), (-3, False)])
def test_sampling_skew_is_recomputed_from_source_receive_times(assembly, seconds, valid):
    source_name = "perp_depth" if seconds >= 0 else "spot_depth"
    shift_source(assembly, source_name, -timedelta(seconds=abs(seconds)))
    corrupt(assembly, spot_perp_sample_delta=timedelta(0))
    result = validate(assembly)
    assert result.sampling_skew == timedelta(seconds=abs(seconds))
    assert result.valid is valid
    if not valid:
        assert_invalid(result, R.SAMPLING_SKEW_TOO_LARGE)


@pytest.mark.parametrize("excess", [timedelta(0), timedelta(microseconds=1)])
def test_max_age_boundary_is_inclusive(assembly, excess):
    # All observations are exactly ten seconds old; requests begin at receive.
    for source_name in ("spot_depth", "perp_depth", "premium_index", "open_interest"):
        corrupt(getattr(assembly.sources, source_name), started_at=OPENED)
        shift_source(assembly, source_name, -timedelta(seconds=10) - excess)
    assert validate(assembly).valid is (excess == timedelta(0))


@pytest.mark.parametrize("field", ["max_data_age", "max_sampling_skew"])
@pytest.mark.parametrize("value,exception", [(1, TypeError), (0.5, TypeError), (timedelta(seconds=-1), ValueError)])
def test_thresholds_require_nonnegative_timedeltas(field, value, exception):
    with pytest.raises(exception):
        DataQualityConfig(**{field: value})


@pytest.mark.parametrize("field", ["started_at", "received_at", "finished_at"])
def test_required_source_times_cannot_be_missing(assembly, field):
    corrupt(assembly.sources.spot_depth, **{field: None})
    assert_invalid(validate(assembly), R.MISSING_TIMESTAMP)


def test_naive_source_time_is_invalid(assembly):
    corrupt(assembly.sources.spot_depth, received_at=OPENED.replace(tzinfo=None))
    assert_invalid(validate(assembly), R.INVALID_TIMESTAMP)


def test_reversed_request_times_are_invalid(assembly):
    corrupt(assembly.sources.spot_depth, finished_at=OPENED - timedelta(seconds=1))
    assert_invalid(validate(assembly), R.INVALID_TIMESTAMP)


def test_future_receive_time_is_invalid(assembly):
    shift_source(assembly, "spot_depth", timedelta(seconds=1))
    assert_invalid(validate(assembly), R.FUTURE_TIMESTAMP)


@pytest.mark.parametrize("source_name", ["perp_depth", "premium_index", "open_interest"])
def test_required_official_times_are_not_fabricated(assembly, source_name):
    source = getattr(assembly.sources, source_name)
    data = source.data.order_book if source_name == "perp_depth" else source.data
    corrupt(data, event_time=None)
    assert_invalid(validate(assembly), R.MISSING_TIMESTAMP)
    assert data.event_time is None


def test_source_failure_cannot_hide_behind_complete_snapshot(assembly):
    corrupt(assembly.sources.spot_depth, error=SpotFetchError(SpotErrorKind.TIMEOUT, "timeout"))
    assert assembly.complete
    assert_invalid(validate(assembly), R.SOURCE_ERROR)


def test_unified_values_must_match_original_source(assembly):
    corrupt(assembly.snapshot, mark_price=D("120"))
    assert_invalid(validate(assembly), R.SOURCE_MISMATCH)
    assert assembly.sources.premium_index.data.mark_price == D("110")


def test_multiple_errors_are_collected_together(assembly):
    shift_source(assembly, "spot_depth", -timedelta(seconds=11))
    corrupt(assembly.snapshot.spot_book.bids[0], price=D("200"))
    corrupt(assembly.snapshot.perp_book.asks[0], quantity=D("0"))
    corrupt(assembly.snapshot, funding_rate=None, open_interest=None)
    assert_invalid(
        validate(assembly), R.STALE_SPOT_DATA, R.SAMPLING_SKEW_TOO_LARGE,
        R.CROSSED_SPOT_BOOK, R.INVALID_QUANTITY, R.MISSING_FUNDING,
        R.MISSING_OPEN_INTEREST,
    )


def test_invalid_data_stops_estimate_before_any_math(assembly, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("invalid market data reached math"))
    for name in ("size_position", "simulate_fill", "accrue_short_funding", "round_trip_from_fills", "decompose_fills"):
        monkeypatch.setattr(engine, name, forbidden)
    result = engine.estimate_once(
        assembly, config(), scenario_for(books()), evaluated_at=OPENED + timedelta(seconds=11),
    )
    assert result.status is EstimateStatus.INVALID_MARKET_DATA and not result.complete
    assert_invalid(result.data_quality, R.STALE_SPOT_DATA, R.STALE_PERP_DATA)
    assert result.fills == () and result.sizing is None
    assert result.calculation is result.decomposition is result.fees is None
    assert result.net_profit is result.net_return is None
    forbidden.assert_not_called()


def test_incomplete_estimate_also_contains_quality_reasons(assembly):
    incomplete = assemble_snapshot(
        assembly.symbol, replace(assembly.sources, open_interest=None), assembled_at=OPENED,
    )
    result = engine.estimate_once(incomplete, config(), scenario_for(books()))
    assert result.status is EstimateStatus.INCOMPLETE_SNAPSHOT
    assert_invalid(result.data_quality, R.INCOMPLETE_SNAPSHOT, R.MISSING_OPEN_INTEREST)
    assert result.net_profit is result.net_return is None


def test_valid_estimate_keeps_prior_cashflow_and_decomposition_results(assembly):
    result = engine.estimate_once(assembly, config(), scenario_for(books()))
    assert result.complete and result.data_quality.valid
    assert result.data_quality.evaluated_at == OPENED
    assert result.net_profit == D("-2.796")
    assert result.capital_occupied == D("420.996")
    assert result.net_return == net_return(D("-2.796"), D("420.996"))
    assert result.net_profit == decomposed_net_profit(
        result.decomposition, D("2.2"), result.fees, D(".1"),
    )
    assert all(isinstance(value, Decimal) for value in (
        result.net_profit, result.net_return, result.capital_occupied,
        result.spot_entry_vwap, result.perp_entry_vwap, result.funding_rate,
    ))


def test_estimate_uses_configured_quality_thresholds(assembly):
    quality_config = DataQualityConfig(max_data_age=timedelta(seconds=20))
    result = engine.estimate_once(
        assembly, config(data_quality=quality_config), scenario_for(books()),
        evaluated_at=OPENED + timedelta(seconds=12),
    )
    assert result.complete and result.net_profit == D("-2.796")
    assert result.data_quality.config == quality_config
