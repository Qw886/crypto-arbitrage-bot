"""Offline persistence of existing research evidence, using temporary SQLite files."""

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import Mock

import pytest
from test_decision import research_estimate, thresholds
from test_offline_estimate import OPENED, assembly_for, books, config, scenario_for
from test_scenario_decision import limits, scenarios

from crypto_arbitrage_bot.arbitrage import estimate as estimate_engine
from crypto_arbitrage_bot.arbitrage.decision import decide
from crypto_arbitrage_bot.arbitrage.scenario_decision import decide_scenarios
from crypto_arbitrage_bot.market_data.snapshot import assemble_snapshot
from crypto_arbitrage_bot.models.decision import Decision
from crypto_arbitrage_bot.models.opportunity import OpportunityRecord
from crypto_arbitrage_bot.storage.opportunity_records import build_opportunity
from crypto_arbitrage_bot.storage.sqlite_repository import (
    SCHEMA_VERSION,
    RepositoryError,
    RepositoryOperation,
    SQLiteConfig,
    SQLiteOpportunityRepository,
)

D = Decimal


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("opportunity storage must remain offline"))
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("http.client.HTTPSConnection", forbidden)


@pytest.fixture
def repository(tmp_path):
    result = SQLiteOpportunityRepository(SQLiteConfig(tmp_path / "research.sqlite", busy_timeout_ms=0))
    result.initialize()
    return result


@pytest.fixture
def record():
    evidence = scenarios()
    decision = decide_scenarios(evidence, limits())
    assert decision.status is Decision.PASS
    return build_opportunity(
        evidence[0].estimate, decision, created_at=OPENED, opportunity_id="research-1",
        formula_version="synthetic-formula-v1", config_version="synthetic-config-v1",
    )


def test_initialize_is_idempotent_and_preserves_records_and_children(repository, record):
    repository.save_opportunity(record)
    repository.initialize()
    repository.initialize()
    assert repository.get_opportunity_by_id(record.id) == record
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'",
        )}
        assert {"opportunities", "opportunity_sources", "opportunity_scenarios"} <= tables
        columns = {row[1]: row[2] for row in connection.execute("PRAGMA table_info(opportunities)")}
        assert all(columns[name] == "TEXT" for name in OpportunityRecord.DECIMAL_FIELDS)
        assert connection.execute("SELECT count(*) FROM opportunities").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM opportunity_sources").fetchone()[0] == 4
        assert connection.execute("SELECT count(*) FROM opportunity_scenarios").fetchone()[0] == 3


def test_initialize_uses_configured_path_and_creates_parent_directories(tmp_path):
    location = tmp_path / "nested" / "custom research.sqlite"
    repository = SQLiteOpportunityRepository(SQLiteConfig(location))
    assert not location.exists()
    repository.initialize()
    assert location.is_file()
    assert repository.list_recent_opportunities() == ()


@pytest.mark.parametrize("status,changes", [
    (Decision.PASS, {}),
    (Decision.OBSERVE, {"min_convergence_profit": D("26")}),
    (Decision.REJECT, {"max_stress_loss": D("1")}),
])
def test_save_all_scenario_decision_statuses(repository, status, changes):
    evidence = scenarios()
    decision = decide_scenarios(evidence, limits(**changes))
    assert decision.status is status
    record = build_opportunity(evidence[0].estimate, decision, created_at=OPENED)
    assert repository.save_opportunity(record) == record.id
    loaded = repository.get_opportunity_by_id(record.id)
    assert loaded == record
    assert loaded.decision == status.value
    assert loaded.decision_reason_codes == tuple(reason.value for reason in decision.reason_codes)
    assert loaded.data_quality_status == "VALID"
    assert loaded.data_quality_reason_codes == ()


def test_complete_market_execution_and_scenario_fields_are_preserved(repository, record):
    repository.save_opportunity(record)
    loaded = repository.get_opportunity_by_id(record.id)
    assert loaded == record
    assert loaded.symbol == "BTCUSDT"
    assert (loaded.spot_bid, loaded.spot_ask, loaded.perp_bid, loaded.perp_ask) == (
        D("99"), D("101"), D("109"), D("111"),
    )
    assert (loaded.mark_price, loaded.index_price, loaded.open_interest) == (
        D("110"), D("100"), D("123.456"),
    )
    assert (loaded.matched_quantity, loaded.matched_notional, loaded.residual_delta) == (
        D("2"), D("204"), D("0"),
    )
    assert loaded.executable_entry_basis == D("8")
    assert loaded.estimated_funding == D("22")
    assert loaded.spot_fees == D(".600") and loaded.perp_fees == D(".2856")
    assert loaded.price_pnl == D("-12") and loaded.net_profit == D("9.0144")
    assert loaded.formula_version == "synthetic-formula-v1"
    assert loaded.config_version == "synthetic-config-v1"
    assert loaded.config_fingerprint == hashlib.sha256(loaded.config_json.encode("utf-8")).hexdigest()
    assert json.loads(loaded.config_json)["estimate"]["target_quantity"] == "2"
    assert [item.net_profit for item in loaded.scenarios] == [
        D("9.0144"), D("25.0272"), D("-2.9952"),
    ]
    assert loaded.primary_scenario_name == loaded.scenarios[0].name
    for item, expected in zip(loaded.scenarios, scenarios(), strict=True):
        assert item.scenario_type == expected.scenario_type.value
        assert item.net_return == expected.net_return
        assert item.capital_occupied == expected.capital_occupied
        assert item.funding_total == expected.funding_total
        assert item.opened_at == expected.definition.exit_scenario.opened_at
        assert item.closed_at == expected.definition.exit_scenario.closed_at
        details = json.loads(item.details_json)
        assert details["definition"]["name"] == expected.name
        assert details["estimate"]["calculation"]["net_profit"] == str(expected.net_profit)


def test_source_request_receive_and_official_times_are_saved_without_invented_spot_event(repository, record):
    repository.save_opportunity(record)
    loaded = repository.get_opportunity_by_id(record.id)
    assert [item.source for item in loaded.source_timings] == [
        "spot_depth", "perp_depth", "premium_index", "open_interest",
    ]
    for source in loaded.source_timings:
        assert source.request_started_at == OPENED - timedelta(milliseconds=10)
        assert source.request_finished_at == source.received_at == source.observed_at == OPENED
        assert json.loads(source.details_json)["status_code"] == 200
    spot, perp, premium, interest = loaded.source_timings
    assert spot.event_time is spot.transaction_time is None
    assert perp.event_time == perp.transaction_time == OPENED
    assert premium.event_time == interest.event_time == OPENED


@pytest.mark.parametrize("value", [
    D("123456789012345678901234567890.123456789012345678901234567890"),
    D("0.0000000000000000000000000000000000000000000012300"),
    D("-0.0000"),
])
def test_decimal_round_trip_retains_all_digits_exponent_and_signed_zero(repository, record, value):
    # Persistence must preserve supplied evidence exactly, without new calculations.
    record = replace(
        record, **dict.fromkeys(OpportunityRecord.DECIMAL_FIELDS, value),
        scenarios=tuple(replace(item, net_profit=value, net_return=value) for item in record.scenarios),
    )
    repository.save_opportunity(record)
    loaded = repository.get_opportunity_by_id(record.id)
    for name in OpportunityRecord.DECIMAL_FIELDS:
        actual = getattr(loaded, name)
        assert isinstance(actual, Decimal) and actual.as_tuple() == value.as_tuple()
    assert all(item.net_profit.as_tuple() == value.as_tuple() for item in loaded.scenarios)
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        stored, storage_type = connection.execute(
            "SELECT net_profit, typeof(net_profit) FROM opportunities WHERE id=?", (record.id,),
        ).fetchone()
    assert storage_type == "text" and stored == str(value)


def test_datetime_round_trip_normalizes_to_utc_without_losing_microseconds(repository, record):
    local = datetime(2026, 9, 25, 16, 2, 3, 123456, tzinfo=timezone(timedelta(hours=8)))
    expected = local.astimezone(UTC)
    source = replace(record.source_timings[0], request_started_at=local, received_at=local)
    scenario = replace(record.scenarios[0], opened_at=local, closed_at=local + timedelta(hours=1))
    record = replace(record, created_at=local, next_funding_time=local,
                     source_timings=(source, *record.source_timings[1:]),
                     scenarios=(scenario, *record.scenarios[1:]))
    repository.save_opportunity(record)
    loaded = repository.get_opportunity_by_id(record.id)
    for value in (loaded.created_at, loaded.next_funding_time,
                  loaded.source_timings[0].request_started_at,
                  loaded.source_timings[0].received_at, loaded.scenarios[0].opened_at):
        assert value == expected and value.tzinfo is UTC and value.microsecond == 123456
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        text = connection.execute("SELECT created_at FROM opportunities").fetchone()[0]
    assert text == "2026-09-25T08:02:03.123456+00:00"


@pytest.mark.parametrize("rate", [".0100", "-.0100", "0.0000"])
def test_positive_negative_and_zero_funding_round_trip(repository, rate):
    source = books()
    estimate = estimate_engine.estimate_once(
        assembly_for((source[0], source[2]), indicative_rate=rate), config(),
        scenario_for(source, rate=rate),
    )
    record = build_opportunity(estimate, decide(estimate, thresholds()), created_at=OPENED)
    repository.save_opportunity(record)
    loaded = repository.get_opportunity_by_id(record.id)
    assert loaded.funding_rate.as_tuple() == D(rate).as_tuple()
    assert loaded.estimated_funding.as_tuple() == estimate.estimated_funding.as_tuple()
    assert loaded.net_profit == estimate.net_profit


def test_rejected_incomplete_research_retains_multiple_reasons_and_nulls(repository):
    source = books()
    complete = assembly_for((source[0], source[2]))
    incomplete = assemble_snapshot(
        complete.symbol, replace(complete.sources, spot_depth=None, open_interest=None),
        assembled_at=OPENED,
    )
    estimate = estimate_engine.estimate_once(incomplete, config(), scenario_for(source))
    decision = decide(estimate, thresholds())
    assert decision.status is Decision.REJECT
    record = build_opportunity(estimate, decision, created_at=OPENED)
    repository.save_opportunity(record)
    loaded = repository.get_opportunity_by_id(record.id)
    assert loaded == record
    assert len(loaded.decision_reason_codes) >= 2
    assert len(loaded.data_quality_reason_codes) >= 2
    assert loaded.data_quality_status == "INVALID"
    assert loaded.snapshot_status == "INCOMPLETE"
    assert loaded.spot_bid is loaded.open_interest is loaded.funding_rate is None
    assert loaded.net_profit is loaded.net_return is loaded.matched_notional is None
    assert loaded.source_timings[0].received_at is None
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        assert connection.execute("SELECT net_profit, net_return FROM opportunities").fetchone() == (None, None)


def test_multiple_scenarios_of_same_type_are_not_overwritten(repository):
    evidence = scenarios()
    extra = replace(evidence[1], definition=replace(evidence[1].definition, name="Second convergence"))
    evidence = (*evidence, extra)
    decision = decide_scenarios(evidence, limits())
    record = build_opportunity(evidence[0].estimate, decision)
    repository.save_opportunity(record)
    loaded = repository.get_opportunity_by_id(record.id)
    assert len(loaded.scenarios) == 4
    assert loaded.scenarios[1].scenario_type == loaded.scenarios[3].scenario_type
    assert loaded.scenarios[1].name != loaded.scenarios[3].name
    assert loaded.scenarios == record.scenarios


def test_single_estimate_decision_can_record_explicit_scenarios(repository):
    evidence = scenarios()
    estimate = evidence[0].estimate
    record = build_opportunity(estimate, decide(estimate, thresholds()), scenarios=evidence)
    assert record.formula_version is record.config_version is None
    repository.save_opportunity(record)
    assert repository.get_opportunity_by_id(record.id) == record


def test_get_missing_id_and_parameterized_id_with_quotes(repository, record):
    record = replace(record, id="opportunity'); DROP TABLE opportunities; --")
    assert repository.get_opportunity_by_id(record.id) is None
    repository.save_opportunity(record)
    assert repository.get_opportunity_by_id(record.id) == record
    assert repository.get_opportunity_by_id("' OR 1=1 --") is None
    assert repository.list_recent_opportunities() == (record,)


def test_recent_records_have_stable_descending_order_and_limit(repository, record):
    old = replace(record, id="old", created_at=OPENED - timedelta(microseconds=1))
    first = replace(record, id="a", created_at=OPENED)
    last = replace(record, id="z", created_at=OPENED)
    for item in (last, old, first):
        repository.save_opportunity(item)
    assert repository.list_recent_opportunities() == (last, first, old)
    assert repository.list_recent_opportunities(limit=2) == (last, first)
    assert repository.list_recent_opportunities(limit=1) == (last,)


@pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5])
def test_invalid_recent_limit_is_rejected(repository, limit):
    with pytest.raises(ValueError, match="limit"):
        repository.list_recent_opportunities(limit)


def test_duplicate_id_fails_explicitly_without_overwriting_previous_evidence(repository, record):
    repository.save_opportunity(record)
    changed = replace(record, net_profit=D("999999"))
    with pytest.raises(RepositoryError) as captured:
        repository.save_opportunity(changed)
    assert captured.value.operation is RepositoryOperation.SAVE
    assert captured.value.sqlite_errorname == "SQLITE_CONSTRAINT_PRIMARYKEY"
    assert isinstance(captured.value.__cause__, sqlite3.IntegrityError)
    assert repository.list_recent_opportunities() == (record,)


def test_locked_database_write_fails_without_mutating_existing_research(repository):
    evidence = scenarios()
    decision = decide_scenarios(evidence, limits())
    before = tuple((item.net_profit, item.net_return, item.estimate.calculation) for item in evidence)
    record = build_opportunity(evidence[0].estimate, decision)
    lock = sqlite3.connect(repository.config.database_path)
    try:
        lock.execute("BEGIN IMMEDIATE")
        with pytest.raises(RepositoryError) as captured:
            repository.save_opportunity(record)
        assert captured.value.operation is RepositoryOperation.SAVE
        assert captured.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
        assert isinstance(captured.value.__cause__, sqlite3.OperationalError)
    finally:
        lock.rollback()
        lock.close()
    assert decision.status is Decision.PASS
    assert tuple((item.net_profit, item.net_return, item.estimate.calculation) for item in evidence) == before
    assert repository.list_recent_opportunities() == ()
    assert repository.save_opportunity(record) == record.id


def test_failed_child_insert_rolls_back_parent_and_all_previous_children(repository, record):
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        connection.execute(
            "CREATE TRIGGER fail_second_scenario BEFORE INSERT ON opportunity_scenarios "
            "WHEN NEW.position=1 BEGIN SELECT RAISE(ABORT, 'synthetic child failure'); END",
        )
    with pytest.raises(RepositoryError, match="synthetic child failure") as captured:
        repository.save_opportunity(record)
    assert captured.value.operation is RepositoryOperation.SAVE
    assert captured.value.sqlite_errorname == "SQLITE_CONSTRAINT_TRIGGER"
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        for table in ("opportunities", "opportunity_sources", "opportunity_scenarios"):
            assert connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
        connection.execute("DROP TRIGGER fail_second_scenario")
    repository.save_opportunity(record)
    assert repository.get_opportunity_by_id(record.id) == record


def test_save_without_initialization_fails_instead_of_creating_empty_database(tmp_path, record):
    path = tmp_path / "missing.sqlite"
    repository = SQLiteOpportunityRepository(SQLiteConfig(path))
    with pytest.raises(RepositoryError) as captured:
        repository.save_opportunity(record)
    assert captured.value.operation is RepositoryOperation.SAVE
    assert captured.value.sqlite_errorcode == sqlite3.SQLITE_CANTOPEN
    assert not path.exists()


def test_bad_database_parent_is_structured_initialize_failure(tmp_path):
    parent = tmp_path / "file-not-directory"
    parent.write_text("existing content", encoding="utf-8")
    repository = SQLiteOpportunityRepository(SQLiteConfig(parent / "research.sqlite"))
    with pytest.raises(RepositoryError) as captured:
        repository.initialize()
    assert captured.value.operation is RepositoryOperation.INITIALIZE
    assert isinstance(captured.value.__cause__, OSError)
    assert parent.read_text(encoding="utf-8") == "existing content"


def test_repository_rejects_float_instead_of_rounding_it(repository, record):
    with pytest.raises(RepositoryError, match="Decimal, never float") as captured:
        repository.save_opportunity(replace(record, net_profit=0.1))
    assert captured.value.operation is RepositoryOperation.SAVE
    assert repository.list_recent_opportunities() == ()


def test_build_reuses_evidence_and_does_not_invoke_engines(monkeypatch):
    estimate = research_estimate()
    decision = decide(estimate, thresholds())
    before = (estimate.net_profit, estimate.net_return, estimate.fees, estimate.decomposition)
    forbidden = Mock(side_effect=AssertionError("logger must not calculate research outcomes"))
    for name in ("estimate_once", "size_position", "simulate_fill", "accrue_short_funding",
                 "round_trip_from_fills", "decompose_fills"):
        monkeypatch.setattr(estimate_engine, name, forbidden)
    record = build_opportunity(estimate, decision)
    assert record.net_profit is estimate.net_profit
    assert record.net_return is estimate.net_return
    assert (estimate.net_profit, estimate.net_return, estimate.fees, estimate.decomposition) == before
    forbidden.assert_not_called()


def test_build_rejects_mismatched_decision_provenance():
    estimate = research_estimate()
    other = research_estimate("ETHUSDT")
    with pytest.raises(ValueError, match="match"):
        build_opportunity(other, decide(estimate, thresholds()))
    evidence = scenarios()
    decision = decide_scenarios(evidence, limits())
    with pytest.raises(ValueError, match="belong"):
        build_opportunity(other, decision)
    with pytest.raises(ValueError, match="scenarios must match"):
        build_opportunity(evidence[0].estimate, decision, scenarios=evidence[:2])
