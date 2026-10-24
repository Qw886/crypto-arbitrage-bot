"""Actual historical settlement descriptions never generate a strategy return."""

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, Inexact, localcontext
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot.analytics.funding_history_summary import (
    FundingStudyStatus,
    summarize_observed_funding,
)
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent, accrue_short_funding
from crypto_arbitrage_bot.exchanges.funding_history import FundingFailure, FundingWindow
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.storage.opportunity_records import exact_json

D=Decimal
AT=datetime(2026,10,5,12,tzinfo=UTC)
OPENED=AT-timedelta(days=7)
SOL=ResearchInstrument("SOLUSDT","SOL")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("socket.create_connection",Mock(side_effect=AssertionError("offline summary")))


def event(name,hours,rate,price="100",symbol="BTCUSDT"):
    return FundingEvent(name,AT-timedelta(hours=hours),D(price),D(rate),symbol)


def summarize(events=(),**overrides):
    window=overrides.pop("window",FundingWindow("BTCUSDT",OPENED,AT,events))
    kwargs={"reference_quantity":D(2),"reference_round_trip_costs":D(".6"),
            "lookback_windows":(timedelta(hours=8),timedelta(days=1),timedelta(days=3),timedelta(days=7)),
            "evaluated_at":AT}
    kwargs.update(overrides)
    return summarize_observed_funding(window,**kwargs)


def test_hand_calculation_uses_each_actual_mark_and_sign_without_extrapolation():
    events=(event("a",20,".002","100"),event("b",5,"-.001","120"),
            event("c",2,"0","150"),event("d",0,".001","110"))
    result=summarize(events)
    short,long=result.slices[:2]
    assert short.settlement_count==3
    assert (short.positive_rate_count,short.zero_rate_count,short.negative_rate_count)==(1,1,1)
    # 2*(120*(-.001)+150*0+110*.001)=-.02; do not apply one final mark to all rates.
    assert short.historical_short_funding_at_reference_quantity==D("-.02")
    assert long.historical_short_funding_at_reference_quantity==D(".38")
    assert long.historical_short_funding_at_reference_quantity==accrue_short_funding(D(2),AT-timedelta(days=1),AT,events,"BTCUSDT")
    assert long.observed_rates.mean==D(".0005")
    assert long.first_settlement_at==AT-timedelta(hours=20) and long.last_settlement_at==AT
    assert result.slices[2].settlement_count==4 # No manufactured 4h/8h settlement count.
    assert result.status is FundingStudyStatus.COMPLETE
    assert not hasattr(result,"net_profit") and not hasattr(result,"net_return")
    assert "not NetProfit" in result.interpretation


def test_empty_complete_history_is_distinct_from_failed_history():
    result=summarize()
    assert result.status is FundingStudyStatus.COMPLETE
    assert all(s.settlement_count==0 and s.historical_short_funding_at_reference_quantity==0 for s in result.slices)
    assert all(s.first_settlement_at is None and s.observed_rates.mean is None for s in result.slices)


@pytest.mark.parametrize("failure",list(FundingFailure))
def test_incomplete_history_has_no_statistical_or_cashflow_output(failure):
    # Even a caller-retained partial event cannot become apparent valid income.
    window=FundingWindow("BTCUSDT",OPENED,AT,(event("partial",1,".1"),),failure=failure)
    result=summarize(window=window)
    assert result.status is FundingStudyStatus.UNAVAILABLE and result.failure is failure
    assert result.slices==()


def test_open_boundary_excluded_closed_boundary_included():
    events=(event("open8",8,".01"),event("inside8",7,".02"),event("closed",0,".03"))
    result=summarize(events)
    assert result.slices[0].settlement_count==2
    assert result.slices[0].historical_short_funding_at_reference_quantity==D(10)
    assert result.slices[1].historical_short_funding_at_reference_quantity==D(12)


@pytest.mark.parametrize("events",[
    (event("outside",169,".01"),),(event("open",168,".01"),),(event("future",-1,".01"),),
    (event("other",1,".01",symbol="ETHUSDT"),),(event("FORECAST_ONLY:next",1,".01"),),
    (event("duplicate",1,".01"),event("duplicate",2,".01")),
    (event("first",1,".01"),event("second",1,".01")),
])
def test_malformed_duplicate_forecast_and_outside_events_are_not_silent(events):
    with pytest.raises(ValueError):
        summarize(events)


@pytest.mark.parametrize("cost",[None,D(0)])
def test_missing_or_zero_reference_cost_does_not_invent_a_ratio(cost):
    result=summarize((event("payment",1,".01"),),reference_round_trip_costs=cost)
    assert result.slices[0].historical_funding_to_reference_cost_ratio is None
    assert result.reference_round_trip_costs is cost


def test_cost_comparison_is_dimensionless_only_and_never_a_net_pnl():
    result=summarize((event("payment",1,".01"),),reference_round_trip_costs=D(4))
    assert result.slices[0].historical_funding_to_reference_cost_ratio==D(".5")
    assert "not historical executions" in result.interpretation
    assert "No hourly extrapolation" in result.interpretation


@pytest.mark.parametrize("kwargs,error",[
    ({"reference_quantity":0.1},TypeError),({"reference_quantity":D(0)},ValueError),
    ({"reference_quantity":D("NaN")},ValueError),({"reference_round_trip_costs":.1},TypeError),
    ({"reference_round_trip_costs":D(-1)},ValueError),
    ({"lookback_windows":[]},TypeError),({"lookback_windows":()},TypeError),
    ({"lookback_windows":(1,)},TypeError),({"lookback_windows":(timedelta(0),)},ValueError),
    ({"lookback_windows":(timedelta(hours=1),timedelta(hours=1))},ValueError),
    ({"lookback_windows":(timedelta(days=8),)},ValueError),
    ({"evaluated_at":AT-timedelta(seconds=1)},ValueError),
])
def test_units_numeric_types_and_source_coverage_are_explicit(kwargs,error):
    with pytest.raises(error):
        summarize(**kwargs)


def test_candidate_identity_is_explicit_and_not_a_trade_permission():
    window=FundingWindow("SOLUSDT",OPENED,AT,(event("sol",1,".01",symbol="SOLUSDT"),))
    with pytest.raises(ValueError):
        summarize(window=window)
    result=summarize(window=window,research_instrument=SOL)
    assert result.symbol=="SOLUSDT" and result.research_instrument==SOL
    with pytest.raises(ValueError):
        summarize(window=window,research_instrument=ResearchInstrument("ETHUSDT","ETH"))


def test_utc_exact_decimal_and_source_record_immutability():
    events=(event("a",2,".000000123456789123456789","98765.4321"),)
    window=FundingWindow("BTCUSDT",OPENED.astimezone(timezone(timedelta(hours=8))),AT,events)
    before=exact_json(window)
    expected=accrue_short_funding(D(".00123456789123456789"),OPENED,AT,events,"BTCUSDT")
    with localcontext() as context:
        context.prec=2
        context.traps[Inexact]=True
        result=summarize(window=window,reference_quantity=D(".00123456789123456789"))
    assert result.source_opened_at.tzinfo is UTC
    assert result.slices[0].historical_short_funding_at_reference_quantity==expected
    assert exact_json(window)==before


def test_unordered_events_do_not_fake_first_last_times():
    result=summarize((event("last",0,".01"),event("first",3,".02")))
    assert result.slices[0].first_settlement_at==AT-timedelta(hours=3)
    assert result.slices[0].last_settlement_at==AT


@pytest.mark.parametrize("window,error",[
    (FundingWindow("BTCUSDT",AT,AT),ValueError),
    (FundingWindow("BTCUSDT",AT,OPENED),ValueError),
    (FundingWindow("BTCUSDT",OPENED,AT,(None,)),TypeError),
    (FundingWindow("BTCUSDT",OPENED,AT,failure="TIMEOUT"),TypeError),
])
def test_bad_window_is_not_an_empty_success(window,error):
    with pytest.raises(error):
        summarize(window=window)
