"""Offline forward outcome summaries; all financial examples are synthetic."""

import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, timedelta
from decimal import Decimal, Inexact, localcontext
from unittest.mock import Mock

import pytest
from test_forward_validation import later_snapshot, research_record, validate
from test_offline_estimate import OPENED

from crypto_arbitrage_bot.analytics.forward_sqlite_summary import summarize_forward_database
from crypto_arbitrage_bot.analytics.forward_summary import (
    ForwardAnalyticsConfig,
    ValidationObservation,
    summarize_forward_observations,
)
from crypto_arbitrage_bot.analytics.sqlite_summary import AnalyticsReadError
from crypto_arbitrage_bot.models.decision import Decision
from crypto_arbitrage_bot.models.forward_validation import ValidationStatus as S
from crypto_arbitrage_bot.storage.forward_validation_repository import (
    SQLiteForwardValidationRepository,
    ValidationPersistenceStatus,
)
from crypto_arbitrage_bot.storage.sqlite_repository import SQLiteConfig, SQLiteOpportunityRepository

D = Decimal
CONFIG = ForwardAnalyticsConfig(D(".1"), (timedelta(hours=1), timedelta(hours=8), timedelta(hours=24)))


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("analytics must be offline"))
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("http.client.HTTPSConnection", forbidden)


@pytest.fixture
def store(tmp_path):
    settings = SQLiteConfig(tmp_path / "research.sqlite", busy_timeout_ms=0)
    opportunities = SQLiteOpportunityRepository(settings)
    opportunities.initialize()
    validations = SQLiteForwardValidationRepository(settings)
    validations.initialize()
    return opportunities, validations


def parent(store, id="one", decision="PASS", symbol="BTCUSDT"):
    record = replace(research_record(symbol), id=id, decision=decision)
    store[0].save_opportunity(record)
    return record


def sample(store, record, *, status=S.VALIDATED, profit="2", net_return=".02", hours=2,
           error="1", return_error=".01", basis="-1", funding=".2", duration=None):
    delta = timedelta(hours=hours) if duration is None else duration
    at = OPENED + delta
    if status is S.INCOMPLETE:
        result = validate(record, later_snapshot(record.symbol, at=at, spot_depth="1"), at=at)
    elif status is S.INVALID:
        result = validate(record, later_snapshot(record.symbol, at=at), at=at + timedelta(seconds=11))
    else:
        result = validate(record, later_snapshot(record.symbol, at=at), at=at)
        result = replace(result, calculation=replace(result.calculation, net_profit=D(profit), funding_total=D(funding)),
                         validated_simulated_net_return=D(net_return), profit_error=None if error is None else D(error),
                         return_error=None if return_error is None else D(return_error), basis_change=None if basis is None else D(basis))
    assert result.status is status
    saved = store[1].save_forward_validation(result)
    assert saved.status is ValidationPersistenceStatus.SAVED
    return saved


def summary(store, config=CONFIG):
    return summarize_forward_database(store[0].config, config)


def group(groups, key):
    return next(item.outcomes for item in groups if item.key == key)


def test_empty_initialized_database(store):
    result = summary(store)
    assert result.total_opportunities == result.opportunities_with_validation == result.unvalidated_opportunities == 0
    assert result.overall.validation_count == 0
    assert all(item.count == 0 and item.proportion is None for item in result.overall.statuses)
    assert result.overall.simulated_net_profit.mean is result.overall.positive_outcome_rate is None
    assert len(result.by_decision) == 3 and len(result.by_symbol) == 2 and len(result.horizons) == 4
    assert result.earliest_validation_time is result.latest_validation_time is None


def test_opportunities_without_validation_are_not_dropped(store):
    first = parent(store)
    parent(store, "unvalidated-eth", "OBSERVE", "ETHUSDT")
    result = summary(store)
    assert result.total_opportunities == result.unvalidated_opportunities == 2
    assert result.opportunities_with_validation == 0
    sample(store, first, status=S.INVALID)
    result = summary(store)
    assert result.opportunities_with_validation == result.unvalidated_opportunities == 1
    assert result.overall.simulated_net_profit.count == 0


def test_mixed_decisions_statuses_symbols_and_outcome_signs(store):
    passed = parent(store)
    observed = parent(store, "observe", "OBSERVE", "ETHUSDT")
    rejected = parent(store, "reject", "REJECT")
    sample(store, passed, profit="4", net_return=".04")
    sample(store, passed, profit="-2", net_return="-.02")
    sample(store, passed, status=S.INCOMPLETE)
    sample(store, passed, status=S.INVALID)
    sample(store, observed, profit="0", net_return="0")
    sample(store, rejected, profit="3", net_return=".03")
    result = summary(store)
    assert result.overall.validation_count == 6 and result.opportunities_with_validation == 3
    assert [item.count for item in result.overall.statuses] == [4, 1, 1]
    assert result.overall.positive_count == 2 and result.overall.zero_count == result.overall.negative_count == 1
    pass_group = group(result.by_decision, "PASS")
    assert pass_group.validation_count == 4
    assert pass_group.positive_outcome_rate == D(".25")
    assert pass_group.simulated_net_profit.count == 2 and pass_group.simulated_net_profit.null_count == 0
    assert (pass_group.simulated_net_profit.minimum, pass_group.simulated_net_profit.maximum,
            pass_group.simulated_net_profit.mean, pass_group.simulated_net_profit.median) == (D(-2), D(4), D(1), D(1))
    assert pass_group.simulated_net_return.mean == D(".01")
    assert group(result.by_decision, "OBSERVE").zero_count == 1
    assert group(result.by_decision, "REJECT").positive_count == 1
    assert group(result.by_symbol, "BTCUSDT").validation_count == 5
    assert group(result.by_symbol, "ETHUSDT").validation_count == 1
    assert group(result.by_symbol, "ETHUSDT").simulated_net_profit.mean == 0


def test_prediction_and_absolute_error_use_saved_fields_only(store):
    record = parent(store)
    sample(store, record, error="-3", return_error="-.03")
    sample(store, record, error="1", return_error=".01")
    sample(store, record, error=None, return_error=None)
    sample(store, record, status=S.INVALID)
    errors = summary(store).prediction_errors
    assert errors.profit_error.mean == errors.profit_error.median == D(-1)
    assert errors.return_error.mean == errors.return_error.median == D("-.01")
    assert errors.absolute_profit_error.mean == errors.absolute_profit_error.median == D(2)
    assert errors.absolute_return_error.mean == errors.absolute_return_error.median == D(".02")
    assert errors.profit_error.count == 2 and errors.profit_error.null_count == 1
    assert errors.absolute_profit_error.null_count == 1


def test_basis_threshold_boundaries_and_funding_sign_groups(store):
    record = parent(store)
    for index, (basis, funding) in enumerate((("-.1001", "1"), ("-.1", "0"), ("0", "-1"),
                                             (".1", "0"), (".1001", "1"), (None, "-1"))):
        sample(store, record, profit=str(index), basis=basis, funding=funding)
    sample(store, record, status=S.INCOMPLETE)
    result = summary(store)
    assert [item.outcomes.validation_count for item in result.by_basis] == [1, 3, 1, 1]
    assert group(result.by_basis, "UNCHANGED").simulated_net_profit.mean == D(2)
    assert group(result.by_basis, "ADVERSE").simulated_net_profit.mean == D(4)
    assert [item.outcomes.validation_count for item in result.by_funding] == [2, 2, 2, 0]
    assert group(result.by_funding, "NEGATIVE").simulated_net_profit.mean == D("3.5")
    changed = summary(store, replace(CONFIG, basis_change_tolerance=D(".2")))
    assert group(changed.by_basis, "UNCHANGED").validation_count == 5


def test_null_funding_is_not_zero(store):
    record = parent(store)
    saved = sample(store, record)
    with closing(sqlite3.connect(store[0].config.database_path)) as connection:
        connection.execute("UPDATE forward_validations SET funding_total=NULL WHERE id=?", (saved.saved_id,))
        connection.commit()
    result = summary(store)
    assert group(result.by_funding, "MISSING").validation_count == 1
    assert group(result.by_funding, "ZERO").validation_count == 0


def test_horizon_boundaries_nonstandard_times_and_missing_duration(store):
    record = parent(store)
    durations = (timedelta(minutes=30), timedelta(hours=1), timedelta(hours=1, microseconds=1),
                 timedelta(hours=8), timedelta(hours=8, microseconds=1), timedelta(hours=24),
                 timedelta(hours=24, microseconds=1), timedelta(hours=48))
    for index, duration in enumerate(durations):
        sample(store, record, duration=duration, profit=str(index - 3))
    sample(store, record, status=S.INVALID)  # Validator leaves duration NULL for this failed attempt.
    result = summary(store)
    assert [item.outcomes.validation_count for item in result.horizons] == [2, 2, 2, 2]
    assert result.missing_horizon.validation_count == 1
    assert result.horizons[0].outcomes.negative_count == 2
    assert result.horizons[1].outcomes.zero_count == 1
    assert result.horizons[2].outcomes.positive_count == 2
    assert result.horizons[3].outcomes.simulated_net_profit.mean == D("3.5")
    custom = summary(store, ForwardAnalyticsConfig(D(0), (timedelta(hours=2),)))
    assert [item.outcomes.validation_count for item in custom.horizons] == [3, 5]
    unbounded = summary(store, ForwardAnalyticsConfig(D(0), ()))
    assert len(unbounded.horizons) == 1 and unbounded.horizons[0].outcomes.validation_count == 8


def test_decimal_precision_and_absolute_values_survive_low_caller_precision(store):
    record = parent(store)
    first = "1.0000000000000000000000000000000000000000000000000001"
    second = "1.0000000000000000000000000000000000000000000000000003"
    for value in (first, second):
        sample(store, record, profit=value, net_return=value, error="-" + value, return_error="-" + value)
    with localcontext() as ctx:
        ctx.prec = 2
        ctx.traps[Inexact] = True
        result = summary(store)
    expected = D("1.0000000000000000000000000000000000000000000000000002")
    assert result.overall.simulated_net_profit.mean == result.overall.simulated_net_profit.median == expected
    assert result.overall.simulated_net_return.mean == expected
    assert result.prediction_errors.absolute_profit_error.mean == expected
    assert result.prediction_errors.absolute_return_error.median == expected


def test_only_validated_financial_fields_are_used_and_null_profit_not_zero():
    valid = ValidationObservation("1", "op", "BTCUSDT", Decision.PASS, S.VALIDATED,
                                  OPENED, timedelta(hours=1), None, None, None, None, None, None)
    invalid = replace(valid, id="2", status=S.INVALID, net_profit=D(999), net_return=D(999),
                      profit_error=D(999), funding_total=D(999), basis_change=D(999))
    result = summarize_forward_observations(frozenset({"op"}), (valid, invalid), CONFIG)
    assert result.overall.simulated_net_profit.count == 0
    assert result.overall.simulated_net_profit.null_count == 1
    assert result.overall.simulated_net_profit.mean is None
    assert result.overall.zero_count == result.overall.positive_count == 0
    assert result.prediction_errors.profit_error.count == 0
    assert group(result.by_basis, "MISSING").validation_count == 1
    assert group(result.by_funding, "POSITIVE").validation_count == 0


def test_original_database_and_utc_times_unchanged(store):
    record = parent(store)
    saved = sample(store, record)
    config = store[0].config
    before = config.database_path.read_bytes()
    original = store[0].get_opportunity_by_id(record.id)
    validation = store[1].get_forward_validation_by_id(saved.saved_id)
    with closing(sqlite3.connect(config.database_path)) as connection:
        dump = tuple(connection.iterdump())
    result = summary(store)
    assert config.database_path.read_bytes() == before
    assert store[0].get_opportunity_by_id(record.id) == original
    assert store[1].get_forward_validation_by_id(saved.saved_id) == validation
    with closing(sqlite3.connect(config.database_path)) as connection:
        assert tuple(connection.iterdump()) == dump
    assert result.earliest_validation_time == result.latest_validation_time == validation.validation_time
    assert result.earliest_validation_time.tzinfo is UTC
    assert "not actual execution" in result.interpretation
    assert not hasattr(result, "win_rate")


def test_analytics_connection_is_read_only(store, monkeypatch):
    original = sqlite3.connect
    def connect(database, *args, **kwargs):
        assert database.endswith("?mode=ro")
        connection = original(database, *args, **kwargs)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM opportunities")
        connection.rollback()
        return connection
    monkeypatch.setattr(sqlite3, "connect", connect)
    assert summary(store).overall.validation_count == 0


def test_missing_database_or_validation_table_is_not_reported_as_empty(tmp_path):
    config = SQLiteConfig(tmp_path / "missing.sqlite")
    with pytest.raises(AnalyticsReadError):
        summarize_forward_database(config, CONFIG)
    assert not config.database_path.exists()
    SQLiteOpportunityRepository(config).initialize()
    with pytest.raises(AnalyticsReadError):
        summarize_forward_database(config, CONFIG)


@pytest.mark.parametrize("sql", [
    "UPDATE forward_validations SET symbol='ETHUSDT'",
    "UPDATE forward_validations SET opportunity_id='missing'",
    "UPDATE forward_validations SET validated_simulated_net_profit='NaN'",
    "UPDATE forward_validations SET holding_duration='-1'",
    "UPDATE forward_validations SET validation_time='2026-01-01T09:00:00'",
])
def test_corrupt_relationship_or_metrics_are_explicit_read_failure(store, sql):
    sample(store, parent(store))
    with closing(sqlite3.connect(store[0].config.database_path)) as connection:
        connection.execute(sql)
        connection.commit()
    with pytest.raises(AnalyticsReadError):
        summary(store)


@pytest.mark.parametrize("tolerance,bounds,exception", [
    (0.1, (), TypeError), (D(-1), (), ValueError), (D("NaN"), (), ValueError),
    (D(0), [timedelta(hours=1)], TypeError), (D(0), (1,), TypeError),
    (D(0), (timedelta(0),), ValueError),
    (D(0), (timedelta(hours=1), timedelta(hours=1)), ValueError),
    (D(0), (timedelta(hours=8), timedelta(hours=1)), ValueError),
])
def test_config_rejects_ambiguous_thresholds(tolerance, bounds, exception):
    with pytest.raises(exception):
        ForwardAnalyticsConfig(tolerance, bounds)
