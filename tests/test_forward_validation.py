"""Synthetic later observations validate hypothetical exits entirely offline."""

import json
from dataclasses import fields, is_dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, Inexact, localcontext
from unittest.mock import Mock

import pytest
from test_decision import thresholds
from test_offline_estimate import OPENED, assembly_for, book, books, config, scenario_for

from crypto_arbitrage_bot.analytics.forward_validation import validate_forward_outcome
from crypto_arbitrage_bot.arbitrage.decision import decide
from crypto_arbitrage_bot.arbitrage.estimate import estimate_once
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.arbitrage.profit import (
    decomposed_net_profit,
    net_return,
    round_trip_from_fills,
)
from crypto_arbitrage_bot.models.forward_validation import ValidationReason as R
from crypto_arbitrage_bot.models.forward_validation import ValidationStatus as S
from crypto_arbitrage_bot.models.market import Market
from crypto_arbitrage_bot.storage.opportunity_records import build_opportunity, exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import SQLiteConfig, SQLiteOpportunityRepository

D = Decimal
AT = OPENED + timedelta(hours=2)


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("forward validation must remain offline"))
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("http.client.HTTPSConnection", forbidden)


def shift_fixture(value, delta):
    """Construct time-consistent synthetic observations, not production history."""
    if isinstance(value, datetime):
        return value + delta
    if is_dataclass(value):
        return replace(value, **{field.name: shift_fixture(getattr(value, field.name), delta)
                                 for field in fields(value)})
    if isinstance(value, tuple):
        return tuple(shift_fixture(item, delta) for item in value)
    return value


def later_snapshot(symbol="BTCUSDT", perp_ask="112", spot_depth="5", perp_depth="5", at=AT):
    spot = book(Market.SPOT, (("110", spot_depth),), (("112", "5"),), symbol)
    perp = book(Market.USDTM, ((str(D(perp_ask) - D("2")), "5"),), ((perp_ask, perp_depth),), symbol)
    return shift_fixture(assembly_for((spot, perp)), at - OPENED)


def research_record(symbol="BTCUSDT", recurring=False):
    source = books(symbol, recurring=recurring)
    estimate = estimate_once(assembly_for((source[0], source[2])),
                             config(target_quantity=D("3") if recurring else D("2")),
                             scenario_for(source))
    return build_opportunity(estimate, decide(estimate, thresholds()), created_at=OPENED,
                             opportunity_id="historical-research")


@pytest.fixture
def repository(tmp_path):
    result = SQLiteOpportunityRepository(SQLiteConfig(tmp_path / "original.sqlite"))
    result.initialize()
    return result


@pytest.fixture
def record(repository):
    source = research_record()
    repository.save_opportunity(source)
    return repository.get_opportunity_by_id(source.id)


def validate(record, snapshot=None, events=(), at=AT):
    return validate_forward_outcome(record, later_snapshot() if snapshot is None else snapshot,
                                    validation_time=at, funding_events=events)


def event(rate=".01", at=OPENED + timedelta(hours=1), symbol="BTCUSDT", event_id="settlement-1"):
    return FundingEvent(event_id, at, D("110"), D(rate), symbol)


@pytest.mark.parametrize("exit_ask,expected_pnl,expected_profit", [
    ("112", "8", "6.9688"),
    ("116", "0", "-1.0376"),
    ("124", "-16", "-17.0504"),
])
def test_convergence_unchanged_and_adverse_basis(record, exit_ask, expected_pnl, expected_profit):
    result = validate(record, later_snapshot(perp_ask=exit_ask))
    assert result.status is S.VALIDATED and result.reason_codes == ()
    assert result.price_pnl == D(expected_pnl)
    assert result.validated_simulated_net_profit == D(expected_profit)
    assert result.original_entry_basis == D("6")  # entry VWAP basis, best quote basis was 8
    assert result.validated_exit_basis == D(exit_ask) - D("110")
    assert result.basis_change == result.validated_exit_basis - D("6")
    assert result.spot_pnl == D("16")
    assert result.spot_pnl + result.perp_pnl == result.price_pnl
    assert result.opportunity_id == record.id and result.symbol == record.symbol
    assert result.entry_time == OPENED and result.validation_time == AT
    assert result.holding_duration == timedelta(hours=2)
    assert result.capital_occupied == record.capital_occupied
    assert result.validated_simulated_net_return == net_return(D(expected_profit), record.capital_occupied)
    assert result.original_estimated_net_profit == record.net_profit
    assert result.original_estimated_net_return == record.net_return
    with localcontext() as ctx:
        ctx.prec = 50
        assert result.profit_error == result.validated_simulated_net_profit - record.net_profit
        assert result.return_error == result.validated_simulated_net_return - record.net_return


@pytest.mark.parametrize("rate,expected", [(".01", "2.2"), ("-.01", "-2.2"), ("0", "0")])
def test_funding_signs(record, rate, expected):
    result = validate(record, events=(event(rate),))
    assert result.status is S.VALIDATED
    assert result.funding_total == D(expected)
    assert result.validated_simulated_net_profit == D("6.9688") + D(expected)


def test_no_event_does_not_reuse_forecast_funding(record):
    assert record.estimated_funding > 0 and record.funding_rate > 0
    result = validate(record)
    assert result.funding_total == D(0)


@pytest.mark.parametrize("settled_at", [OPENED - timedelta(seconds=1), OPENED, AT + timedelta(microseconds=1)])
def test_out_of_window_funding_is_invalid_without_pnl(record, settled_at):
    result = validate(record, events=(event(at=settled_at),))
    assert result.status is S.INVALID
    assert R.FUNDING_EVENT_OUTSIDE_HOLDING_WINDOW in result.reason_codes
    assert result.funding_total is result.validated_simulated_net_profit is result.validated_simulated_net_return is None


def test_event_at_exit_included_once_and_multiple_events_sum(record):
    result = validate(record, events=(event(), event("-.005", at=AT, event_id="second")))
    assert result.status is S.VALIDATED and result.funding_total == D("1.1")
    assert result.validated_simulated_net_profit == D("8.0688")


@pytest.mark.parametrize("second", [event(), event(event_id="different-id")])
def test_duplicate_funding_is_rejected(record, second):
    result = validate(record, events=(event(), second))
    assert result.status is S.INVALID
    assert result.reason_codes == (R.INVALID_FUNDING_EVENTS,)
    assert result.validated_simulated_net_profit is None


@pytest.mark.parametrize("changes,leg", [({"spot_depth": "1"}, 0), ({"perp_depth": "1"}, 1)])
def test_insufficient_exit_depth_never_extrapolates(record, changes, leg):
    result = validate(record, later_snapshot(**changes))
    assert result.status is S.INCOMPLETE
    assert result.reason_codes == (R.INSUFFICIENT_EXIT_DEPTH,)
    assert not result.exit_fills[leg].complete and result.exit_fills[leg].vwap is None
    assert result.exit_fills[leg].filled_quantity == D(1)
    assert result.validated_simulated_net_profit is result.validated_simulated_net_return is None


def test_wrong_symbol_snapshot_and_funding_rejected(record):
    for result in (validate(record, later_snapshot("ETHUSDT")),
                   validate(record, events=(event(symbol="ETHUSDT"),))):
        assert result.status is S.INVALID and R.SYMBOL_MISMATCH in result.reason_codes
        assert result.validated_simulated_net_profit is None


@pytest.mark.parametrize("at", [OPENED, OPENED - timedelta(seconds=1)])
def test_validation_time_must_follow_entry(record, at):
    result = validate(record, later_snapshot(at=at), at=at)
    assert result.status is S.INVALID
    assert R.VALIDATION_TIME_NOT_AFTER_ENTRY in result.reason_codes


def test_old_snapshot_cannot_be_used_with_later_validation_timestamp(record):
    result = validate(record, later_snapshot(at=OPENED), at=OPENED + timedelta(seconds=1))
    assert result.status is S.INVALID
    assert R.VALIDATION_TIME_NOT_AFTER_ENTRY in result.reason_codes


def test_old_official_time_is_not_hidden_by_fresh_receive_time(record):
    at = OPENED + timedelta(seconds=1)
    snapshot = later_snapshot(at=at)
    premium = snapshot.sources.premium_index
    snapshot = replace(snapshot, sources=replace(snapshot.sources, premium_index=replace(
        premium, data=replace(premium.data, event_time=OPENED),
    )))
    result = validate(record, snapshot, at=at)
    assert result.exit_data_quality.valid  # Within age limit, but not a subsequent observation.
    assert R.VALIDATION_TIME_NOT_AFTER_ENTRY in result.reason_codes
    assert result.validated_simulated_net_profit is None


def test_eth_record_can_be_validated_offline():
    result = validate(research_record("ETHUSDT"), later_snapshot("ETHUSDT"),
                      events=(event(symbol="ETHUSDT"),))
    assert result.status is S.VALIDATED and result.symbol == "ETHUSDT"


def test_multiple_validation_reasons_are_retained(record):
    result = validate(replace(record, matched_quantity=D("1")), later_snapshot("ETHUSDT"),
                      events=(event(at=OPENED),))
    assert {R.SYMBOL_MISMATCH, R.QUANTITY_MISMATCH,
            R.FUNDING_EVENT_OUTSIDE_HOLDING_WINDOW} <= set(result.reason_codes)
    assert result.validated_simulated_net_profit is None


def test_snapshot_must_postdate_record_creation(record):
    result = validate(replace(record, created_at=AT))
    assert R.VALIDATION_TIME_NOT_AFTER_ENTRY in result.reason_codes


def test_naive_timestamp_is_invalid_and_aware_timestamp_normalized(record):
    assert validate(record, at=AT.replace(tzinfo=None)).reason_codes == (R.INVALID_VALIDATION_TIME,)
    result = validate(record, at=AT.astimezone(timezone(timedelta(hours=8))))
    assert result.status is S.VALIDATED and result.validation_time.tzinfo is UTC


@pytest.mark.parametrize("failure", ["stale", "incomplete", "crossed"])
def test_exit_data_quality_is_required(record, failure):
    snapshot = later_snapshot()
    at = AT
    if failure == "stale":
        at += timedelta(seconds=11)
    elif failure == "incomplete":
        snapshot = replace(snapshot, snapshot=None)
    else:
        spot = snapshot.snapshot.spot_book
        crossed = replace(spot, bids=(replace(spot.bids[0], price=D("120")),))
        snapshot = replace(snapshot, snapshot=replace(snapshot.snapshot, spot_book=crossed))
    result = validate(record, snapshot, at=at)
    assert result.status is S.INVALID and R.INVALID_EXIT_MARKET_DATA in result.reason_codes
    assert not result.exit_data_quality.valid
    assert result.validated_simulated_net_profit is None


def test_original_quantity_is_required_and_never_resized(record):
    assert validate(record).matched_quantity == record.matched_quantity
    result = validate(replace(record, matched_quantity=D("1")))
    assert result.status is S.INVALID and R.QUANTITY_MISMATCH in result.reason_codes


def test_cashflow_and_decomposition_agree_without_recharging_costs(record):
    result = validate(record, events=(event(),))
    se, pe = result.entry_fills
    sx, px = result.exit_fills
    expected = round_trip_from_fills(se, sx, pe, px, result.fees, D("2.2"), D(".1"))
    assert result.calculation == expected
    assert result.fees.spot_entry == D(".204") and result.fees.perp_entry == D(".108")
    assert result.fees.spot_exit == D(".44") and result.fees.perp_exit == D(".1792")
    assert result.other_costs == D(".1")
    assert result.validated_simulated_net_profit == decomposed_net_profit(
        result.decomposition, result.funding_total, result.fees, result.other_costs,
    )


def test_recurring_vwap_precision_and_multilevel_exits(repository):
    record = research_record(recurring=True)
    repository.save_opportunity(record)
    record = repository.get_opportunity_by_id(record.id)
    snapshot = later_snapshot()
    spot = snapshot.snapshot.spot_book
    perp = snapshot.snapshot.perp_book
    spot = replace(spot, bids=(replace(spot.bids[0], quantity=D(1)),
                               replace(spot.bids[0], price=D("109"), quantity=D(3))))
    perp = replace(perp, asks=(replace(perp.asks[0], quantity=D(1)),
                              replace(perp.asks[0], price=D("113"), quantity=D(3))))
    # Build a consistent synthetic assembly from these books with original fixture times.
    snapshot = shift_fixture(assembly_for((shift_fixture(spot, OPENED - AT),
                                          shift_fixture(perp, OPENED - AT))), AT - OPENED)
    with localcontext() as ctx:
        ctx.prec = 2
        ctx.traps[Inexact] = True
        result = validate(record, snapshot)
    assert result.status is S.VALIDATED
    assert result.entry_fills[0].quote_notional == D("305")
    assert result.exit_fills[0].quote_notional == D("328")
    assert result.exit_fills[1].quote_notional == D("338")
    assert result.price_pnl == D("10")
    assert all(isinstance(value, Decimal) for value in (
        result.validated_simulated_net_profit, result.validated_simulated_net_return, result.funding_total,
    ))
    assert result.validated_simulated_net_profit == decomposed_net_profit(
        result.decomposition, result.funding_total, result.fees, result.other_costs,
    )


def test_original_record_decision_and_database_unchanged(repository, record):
    before = exact_json(record)
    database = repository.config.database_path.read_bytes()
    result = validate(record)
    assert result.status is S.VALIDATED
    assert exact_json(record) == before
    assert repository.get_opportunity_by_id(record.id) == record
    assert repository.config.database_path.read_bytes() == database
    assert not hasattr(result, "realized_profit")


@pytest.mark.parametrize("case", ["missing", "bad_json", "float_price", "bad_notional"])
def test_missing_or_corrupt_original_execution_cannot_generate_profit(record, case):
    details = json.loads(record.details_json)
    if case == "missing":
        details["estimate"]["fills"] = []
    elif case == "float_price":
        details["estimate"]["fills"][0]["vwap"] = 102.0
    elif case == "bad_notional":
        details["estimate"]["fills"][0]["quote_notional"] = "1"
    record = replace(record, details_json="invalid" if case == "bad_json" else json.dumps(details))
    result = validate(record)
    assert result.status is (S.INCOMPLETE if case == "missing" else S.INVALID)
    assert result.validated_simulated_net_profit is result.validated_simulated_net_return is None


def test_forecast_exit_and_funding_are_not_used(record):
    details = json.loads(record.details_json)
    details["estimate"]["scenario"]["funding_events"] = [{"not": "an observed event"}]
    details["estimate"]["scenario"]["spot_exit_book"] = None
    details["estimate"]["scenario"]["perp_exit_book"] = None
    result = validate(replace(record, details_json=json.dumps(details)))
    assert result.status is S.VALIDATED and result.funding_total == 0
    assert result.validated_simulated_net_profit == D("6.9688")
