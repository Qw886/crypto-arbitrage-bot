from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, localcontext

import pytest

from crypto_arbitrage_bot.arbitrage.funding import (
    FundingEvent,
    accrue_short_funding,
    short_funding_cashflow,
)

D = Decimal
OPEN = datetime(2026, 9, 14, 0, 0, tzinfo=UTC)
SETTLEMENT = OPEN + timedelta(hours=8)
CLOSE = OPEN + timedelta(hours=9)


def event(rate="0.0001", *, when=SETTLEMENT, event_id="btc-1", price="60000", symbol="BTCUSDT"):
    return FundingEvent(event_id, when, D(price), D(rate), symbol)


@pytest.mark.parametrize("rate, expected", [("0.0001", "1.2"), ("-0.0001", "-1.2"), ("0", "0")])
def test_funding_sign_for_short(rate, expected):
    assert short_funding_cashflow(D("0.2"), D("60000"), D(rate)) == D(expected)
    assert accrue_short_funding(D("0.2"), OPEN, CLOSE, [event(rate)], "BTCUSDT") == D(expected)


def test_no_crossed_settlement_means_zero_not_prorated_funding():
    assert accrue_short_funding(D("0.2"), OPEN, SETTLEMENT - timedelta(seconds=1), [event()], "BTCUSDT") == D("0")


def test_no_provided_events_means_zero_even_for_long_holding_window():
    assert accrue_short_funding(D("0.2"), OPEN, CLOSE + timedelta(days=365), [], "BTCUSDT") == D("0")


def test_open_boundary_excluded_close_boundary_included():
    assert accrue_short_funding(D("0.2"), SETTLEMENT, CLOSE, [event()], "BTCUSDT") == D("0")
    assert accrue_short_funding(D("0.2"), OPEN, SETTLEMENT, [event()], "BTCUSDT") == D("1.2")


def test_zero_duration_and_zero_position_receive_nothing():
    assert accrue_short_funding(D("0.2"), SETTLEMENT, SETTLEMENT, [event()], "BTCUSDT") == D("0")
    assert accrue_short_funding(D("0"), OPEN, CLOSE, [event()], "BTCUSDT") == D("0")


def test_each_event_uses_its_actual_settlement_mark_price():
    events = [
        event("0.0001", price="60000"),
        event("-0.0002", when=SETTLEMENT + timedelta(hours=8), event_id="btc-2", price="65000"),
    ]
    # 0.2 * 60000 * 0.0001 - 0.2 * 65000 * 0.0002 = -1.4 USDT.
    assert accrue_short_funding(D("0.2"), OPEN, CLOSE + timedelta(hours=8), events, "BTCUSDT") == D("-1.4")


def test_other_symbols_are_not_accrued():
    events = [event(), event(symbol="ETHUSDT", event_id="eth-1", price="3000")]
    assert accrue_short_funding(D("0.2"), OPEN, CLOSE, events, "BTCUSDT") == D("1.2")


def test_event_datetime_is_normalized_to_utc():
    local_time = datetime(2026, 9, 14, 16, 0, tzinfo=timezone(timedelta(hours=8)))
    local_event = event(when=local_time)
    assert local_event.settled_at == SETTLEMENT
    assert local_event.settled_at.tzinfo == UTC
    assert accrue_short_funding(D("0.2"), OPEN, CLOSE, [local_event], "BTCUSDT") == D("1.2")


def test_naive_event_timestamp_rejected():
    with pytest.raises(ValueError):
        event(when=datetime(2026, 9, 14, 8, 0))


def test_reversed_holding_window_rejected():
    with pytest.raises(ValueError, match="closed_at"):
        accrue_short_funding(D("0.2"), CLOSE, OPEN, [], "BTCUSDT")


def test_duplicate_event_id_rejected_even_at_different_times():
    events = [event(), event(when=SETTLEMENT + timedelta(hours=8))]
    with pytest.raises(ValueError, match="duplicate"):
        accrue_short_funding(D("0.2"), OPEN, CLOSE, events, "BTCUSDT")


def test_duplicate_settlement_rejected_even_with_different_event_ids():
    with pytest.raises(ValueError, match="duplicate"):
        accrue_short_funding(D("0.2"), OPEN, CLOSE, [event(), event(event_id="btc-copy")], "BTCUSDT")


@pytest.mark.parametrize("bad_rate", [0.0001, "0.0001", 0])
def test_float_or_nondecimal_funding_rate_rejected(bad_rate):
    with pytest.raises(TypeError):
        FundingEvent("btc-1", SETTLEMENT, D("60000"), bad_rate, "BTCUSDT")


@pytest.mark.parametrize("bad", [D("NaN"), D("Infinity"), D("-Infinity")])
def test_nonfinite_funding_rate_rejected(bad):
    with pytest.raises(ValueError):
        short_funding_cashflow(D("0.2"), D("60000"), bad)


def test_float_quantity_rejected():
    with pytest.raises(TypeError):
        accrue_short_funding(0.2, OPEN, CLOSE, [event()], "BTCUSDT")


def test_negative_quantity_rejected():
    with pytest.raises(ValueError):
        short_funding_cashflow(D("-0.2"), D("60000"), D("0.0001"))


def test_decimal_multiplication_is_exact_and_context_is_restored():
    with localcontext() as context:
        context.prec = 3
        value = short_funding_cashflow(D("0.123456789"), D("60000.123456789"), D("0.000123456789"))
        assert context.prec == 3
    assert value == D("0.914496606687803049154860897069")
