"""Offline historical research statistics over temporary SQLite databases."""

import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, timedelta
from decimal import ROUND_DOWN, Decimal, Inexact, localcontext
from unittest.mock import Mock

import pytest
from test_offline_estimate import OPENED
from test_scenario_decision import limits, scenarios

from crypto_arbitrage_bot.analytics.sqlite_summary import (
    AnalyticsReadError,
    read_observations,
    summarize_database,
)
from crypto_arbitrage_bot.analytics.summary import numeric_summary
from crypto_arbitrage_bot.arbitrage.scenario_decision import decide_scenarios
from crypto_arbitrage_bot.models.scenario import ScenarioType
from crypto_arbitrage_bot.storage.opportunity_records import build_opportunity
from crypto_arbitrage_bot.storage.sqlite_repository import SQLiteConfig, SQLiteOpportunityRepository

D = Decimal


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("analytics is offline"))
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("http.client.HTTPSConnection", forbidden)


@pytest.fixture
def repository(tmp_path):
    repo = SQLiteOpportunityRepository(SQLiteConfig(tmp_path / "history.sqlite", busy_timeout_ms=0))
    repo.initialize()
    return repo


@pytest.fixture
def record():
    evidence = scenarios()
    return build_opportunity(evidence[0].estimate, decide_scenarios(evidence, limits()),
                             opportunity_id="one", created_at=OPENED)


def test_empty_initialized_database(repository):
    result = summarize_database(repository.config)
    assert result.total_opportunities == 0
    assert all(item.count == 0 and item.proportion is None for item in result.decisions)
    assert result.symbols == result.decision_reasons == result.data_quality_reasons == ()
    assert result.net_profit.count == result.net_profit.null_count == 0
    assert result.net_profit.mean is result.net_profit.median is None
    assert result.earliest_created_at is result.latest_created_at is None
    assert len(result.scenarios) == 3
    assert all(item.count == item.missing_opportunity_count == 0 for item in result.scenarios)


def test_single_record_and_utc_provenance(repository, record):
    repository.save_opportunity(record)
    result = summarize_database(repository.config)
    assert result.total_opportunities == 1
    assert [(item.decision.value, item.count, item.proportion) for item in result.decisions] == [
        ("PASS", 1, D(1)), ("OBSERVE", 0, D(0)), ("REJECT", 0, D(0)),
    ]
    for metric in ("net_profit", "net_return"):
        summary = getattr(result, metric)
        value = getattr(record, metric)
        assert summary.count == 1 and summary.null_count == 0
        assert summary.minimum == summary.maximum == summary.mean == summary.median == value
    assert result.earliest_created_at == result.latest_created_at == record.created_at
    assert result.earliest_created_at.tzinfo is UTC
    assert read_observations(repository.config)[0].created_at == record.created_at
    assert "not realized" in result.interpretation
    assert not hasattr(result, "win_rate")


def test_mixed_decisions_symbols_nulls_funding_and_reason_frequencies(repository, record):
    rows = (
        replace(record, id="1", net_profit=D(".1"), net_return=D(".01"), funding_rate=D(".001"),
                decision_reason_codes=("A", "B", "A")),
        replace(record, id="2", decision="OBSERVE", symbol="ETHUSDT", net_profit=D(".2"),
                net_return=D(".02"), funding_rate=D("0"), decision_reason_codes=("B",)),
        replace(record, id="3", decision="REJECT", net_profit=D("-.6"), net_return=D("-.03"),
                funding_rate=D("-.002"), data_quality_status="INVALID",
                data_quality_reason_codes=("STALE_SPOT_DATA", "MISSING_FUNDING", "MISSING_FUNDING"),
                decision_reason_codes=("A",)),
        replace(record, id="4", decision="REJECT", symbol="ETHUSDT", net_profit=None,
                net_return=None, funding_rate=None, data_quality_status=None,
                data_quality_reason_codes=("MISSING_FUNDING",), decision_reason_codes=()),
    )
    for row in rows:
        repository.save_opportunity(row)
    result = summarize_database(repository.config)
    assert result.total_opportunities == 4
    assert [item.count for item in result.decisions] == [1, 1, 2]
    assert [item.proportion for item in result.decisions] == [D(".25"), D(".25"), D(".5")]
    assert [(item.value, item.count) for item in result.symbols] == [("BTCUSDT", 2), ("ETHUSDT", 2)]
    assert result.net_profit.count == 3 and result.net_profit.null_count == 1
    assert (result.net_profit.minimum, result.net_profit.maximum, result.net_profit.mean,
            result.net_profit.median) == (D("-.6"), D(".2"), D("-.1"), D(".1"))
    assert result.net_return.mean == D(0) and result.net_return.median == D(".01")
    assert result.net_return.null_count == 1
    assert (result.funding_rates.positive, result.funding_rates.zero,
            result.funding_rates.negative, result.funding_rates.missing) == (1, 1, 1, 1)
    assert result.invalid_data_quality_count == result.missing_data_quality_count == 1
    assert [(item.value, item.count) for item in result.decision_reasons] == [("A", 2), ("B", 2)]
    assert [(item.value, item.count) for item in result.data_quality_reasons] == [
        ("MISSING_FUNDING", 2), ("STALE_SPOT_DATA", 1),
    ]


@pytest.mark.parametrize("values,mean,median", [
    ([".1", ".2"], ".15", ".15"),
    (["9", "-4", "1"], "2", "1"),
    (["-5", "-1", "4", "10"], "2", "1.5"),
    (["1E60", "1", "-1E60"], None, "1"),
    (["1.0000000000000000000000000000000000000000000000000001",
      "1.0000000000000000000000000000000000000000000000000003"],
     "1.0000000000000000000000000000000000000000000000000002",
     "1.0000000000000000000000000000000000000000000000000002"),
])
def test_decimal_mean_median_and_cancellation(values, mean, median):
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        result = numeric_summary(map(D, values))
    assert result.median == D(median)
    if mean is not None:
        assert result.mean == D(mean)
    else:
        assert str(result.mean).startswith("0." + "3" * 50)
    assert isinstance(result.mean, Decimal) and isinstance(result.median, Decimal)


def test_all_nulls_do_not_become_zero(repository, record):
    repository.save_opportunity(replace(record, net_profit=None, net_return=None,
                                         funding_rate=None, scenarios=()))
    result = summarize_database(repository.config)
    assert result.net_profit.count == result.net_return.count == 0
    assert result.net_profit.null_count == result.net_return.null_count == 1
    assert result.net_profit.minimum is result.net_profit.maximum is result.net_profit.mean is None
    assert result.funding_rates.zero == 0 and result.funding_rates.missing == 1
    assert all(item.missing_opportunity_count == 1 and item.count == 0 for item in result.scenarios)


def test_each_scenario_variant_is_counted_with_its_own_nulls(repository, record):
    baseline, convergence, stress = record.scenarios
    repository.save_opportunity(replace(record, scenarios=(
        replace(baseline, net_profit=D("2"), net_return=D(".02")),
        replace(convergence, net_profit=D("10"), net_return=D(".1")),
        replace(convergence, name="second", net_profit=D("20"), net_return=D(".2")),
        replace(stress, net_profit=D("-6"), net_return=D("-.06")),
    )))
    repository.save_opportunity(replace(record, id="two", scenarios=(
        replace(baseline, net_profit=None, net_return=None),
        replace(stress, net_profit=D("-2"), net_return=None),
    )))
    summary = summarize_database(repository.config)
    base, converge, adverse = summary.scenarios
    assert [item.scenario_type for item in summary.scenarios] == list(ScenarioType)
    assert base.count == base.opportunity_count == 2
    assert base.net_profit.count == base.net_profit.null_count == 1
    assert base.net_profit.mean == D("2")
    assert converge.count == 2 and converge.opportunity_count == 1 and converge.missing_opportunity_count == 1
    assert converge.net_profit.mean == converge.net_profit.median == D("15")
    assert converge.net_return.mean == D(".15")
    assert adverse.net_profit.mean == D("-4") and adverse.net_profit.minimum == D("-6")
    assert adverse.net_return.count == adverse.net_return.null_count == 1
    assert adverse.net_return.mean == D("-.06")


def test_analytics_preserves_database_bytes_records_schema_and_utc(repository, record):
    repository.save_opportunity(record)
    repository.save_opportunity(replace(record, id="old", created_at=OPENED - timedelta(microseconds=1)))
    before_bytes = repository.config.database_path.read_bytes()
    before_records = repository.list_recent_opportunities()
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        before_dump = tuple(connection.iterdump())
    result = summarize_database(repository.config)
    assert result.earliest_created_at == OPENED - timedelta(microseconds=1)
    assert result.latest_created_at == OPENED
    assert repository.config.database_path.read_bytes() == before_bytes
    assert repository.list_recent_opportunities() == before_records
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        assert tuple(connection.iterdump()) == before_dump


def test_reader_enforces_read_only_connection(repository, record, monkeypatch):
    repository.save_opportunity(record)
    original = sqlite3.connect
    def connect(database, *args, **kwargs):
        assert database.endswith("?mode=ro") and kwargs["uri"] is True
        connection = original(database, *args, **kwargs)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM opportunities")
        connection.rollback()
        return connection
    monkeypatch.setattr(sqlite3, "connect", connect)
    assert summarize_database(repository.config).total_opportunities == 1


def test_exact_decimal_statistics_from_sqlite_ignore_caller_precision(repository, record):
    values = (D("1.0000000000000000000000000000000000000000000000000001"),
              D("1.0000000000000000000000000000000000000000000000000003"))
    for index, value in enumerate(values):
        repository.save_opportunity(replace(record, id=str(index), net_profit=value,
                                             net_return=value, funding_rate=D("-0.000")))
    with localcontext() as context:
        context.prec = 2
        context.traps[Inexact] = True
        result = summarize_database(repository.config)
    expected = D("1.0000000000000000000000000000000000000000000000000002")
    assert result.net_profit.mean == result.net_profit.median == expected
    assert result.net_return.mean == result.net_return.median == expected
    assert result.funding_rates.zero == 2 and result.funding_rates.negative == 0


def test_all_history_is_read_beyond_recent_repository_limit(repository, record):
    # Save one full record then duplicate parent observations inside the fixture.
    repository.save_opportunity(replace(record, scenarios=()))
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(opportunities)")]
        values = list(connection.execute("SELECT * FROM opportunities").fetchone())
        for index in range(1001):
            values[columns.index("id")] = f"extra-{index}"
            connection.execute(f"INSERT INTO opportunities VALUES ({','.join('?' for _ in columns)})", values)
        connection.commit()
    assert summarize_database(repository.config).total_opportunities == 1002


def test_missing_database_is_explicit_failure_and_not_created(tmp_path):
    settings = SQLiteConfig(tmp_path / "missing.sqlite")
    with pytest.raises(AnalyticsReadError) as error:
        summarize_database(settings)
    assert error.value.sqlite_errorcode is not None
    assert not settings.database_path.exists()


@pytest.mark.parametrize("sql", [
    "PRAGMA user_version=99",
    "UPDATE opportunities SET net_profit='NaN'",
    "UPDATE opportunities SET net_return='Infinity'",
    "UPDATE opportunities SET decision_reason_codes='{}'",
    "UPDATE opportunities SET created_at='2026-01-01T00:00:00'",
    "UPDATE opportunities SET data_quality_status='UNKNOWN'",
    "UPDATE opportunity_scenarios SET scenario_type='UNKNOWN'",
])
def test_corrupt_history_is_not_silently_ignored(repository, record, sql):
    repository.save_opportunity(record)
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        connection.execute(sql)
        connection.commit()
    with pytest.raises(AnalyticsReadError):
        summarize_database(repository.config)


@pytest.mark.parametrize("value", [0.1, D("NaN"), D("Infinity")])
def test_numeric_statistics_reject_float_and_nonfinite(value):
    with pytest.raises((ValueError, TypeError)):
        numeric_summary([value])
