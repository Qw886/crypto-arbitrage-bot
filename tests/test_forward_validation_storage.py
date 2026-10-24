"""Offline persistence of validation results without altering historical evidence."""

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, timedelta, timezone
from decimal import Decimal
from unittest.mock import Mock

import pytest
from test_forward_validation import AT, later_snapshot, research_record, validate
from test_offline_estimate import OPENED

from crypto_arbitrage_bot.models.forward_validation import ValidationReason as R
from crypto_arbitrage_bot.models.forward_validation import ValidationStatus as S
from crypto_arbitrage_bot.storage.forward_validation_records import build_forward_validation_record
from crypto_arbitrage_bot.storage.forward_validation_repository import (
    SQLiteForwardValidationRepository,
)
from crypto_arbitrage_bot.storage.forward_validation_repository import (
    ValidationPersistenceStatus as P,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import (
    RepositoryError,
    RepositoryOperation,
    SQLiteConfig,
    SQLiteOpportunityRepository,
)

D = Decimal


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("validation persistence is offline"))
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("http.client.HTTPSConnection", forbidden)


@pytest.fixture
def setup(tmp_path):
    config = SQLiteConfig(tmp_path / "research.sqlite", busy_timeout_ms=0)
    opportunities = SQLiteOpportunityRepository(config)
    opportunities.initialize()
    record = research_record()
    opportunities.save_opportunity(record)
    validations = SQLiteForwardValidationRepository(config)
    validations.initialize()
    return opportunities, validations, record


@pytest.mark.parametrize("status", list(S))
def test_save_all_statuses_and_map_fields(setup, status):
    _, repository, record = setup
    if status is S.VALIDATED:
        result = validate(record)
    elif status is S.INCOMPLETE:
        result = validate(record, later_snapshot(spot_depth="1"))
    else:
        result = validate(record, at=AT + timedelta(seconds=11))
    assert result.status is status
    saved = repository.save_forward_validation(result, validation_id="result", created_at=AT)
    assert saved.status is P.SAVED and saved.failure is None
    loaded = repository.get_forward_validation_by_id(saved.saved_id)
    assert loaded == saved.record == build_forward_validation_record(result, validation_id="result", created_at=AT)
    for name in ("opportunity_id", "symbol", "entry_time", "validation_time", "holding_duration",
                 "original_estimated_net_profit", "original_estimated_net_return",
                 "validated_simulated_net_profit", "validated_simulated_net_return", "profit_error",
                 "return_error", "original_entry_basis", "validated_exit_basis", "basis_change",
                 "funding_total", "spot_pnl", "perp_pnl", "price_pnl", "other_costs",
                 "matched_quantity", "capital_occupied"):
        assert getattr(loaded, name) == getattr(result, name)
    assert loaded.reason_codes == tuple(code.value for code in result.reason_codes)
    assert loaded.status == result.status.value
    assert loaded.details_json == exact_json(result)
    if status is S.VALIDATED:
        assert loaded.spot_exit_vwap == result.exit_fills[0].vwap
        assert loaded.perp_exit_vwap == result.exit_fills[1].vwap
        assert loaded.spot_entry_fee == result.fees.spot_entry
        assert loaded.perp_entry_fee == result.fees.perp_entry
        assert loaded.spot_exit_fee == result.fees.spot_exit
        assert loaded.perp_exit_fee == result.fees.perp_exit
    else:
        assert loaded.validated_simulated_net_profit is loaded.validated_simulated_net_return is None
        assert loaded.spot_exit_fee is loaded.funding_total is None


def test_repeated_initialization_preserves_parents_children_and_schema_version(setup):
    opportunities, repository, record = setup
    saved = repository.save_forward_validation(validate(record))
    repository.initialize()
    repository.initialize()
    opportunities.initialize()
    assert opportunities.get_opportunity_by_id(record.id) == record
    assert repository.get_forward_validation_by_id(saved.saved_id) == saved.record
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        foreign_keys = connection.execute("PRAGMA foreign_key_list(forward_validations)").fetchall()
        assert {(item[2], item[3], item[4]) for item in foreign_keys} == {
            ("opportunities", "opportunity_id", "id"), ("opportunities", "symbol", "symbol"),
        }


def test_multiple_times_and_multiple_opportunities_have_correct_query_scope(setup):
    opportunities, repository, record = setup
    saved = []
    for hours in (24, 1, 8):
        at = OPENED + timedelta(hours=hours)
        saved.append(repository.save_forward_validation(
            validate(record, later_snapshot(at=at), at=at), validation_id=str(hours), created_at=at,
        ))
    eth = replace(research_record("ETHUSDT"), id="eth-research")
    opportunities.save_opportunity(eth)
    other = repository.save_forward_validation(validate(eth, later_snapshot("ETHUSDT")), created_at=AT)
    assert all(item.status is P.SAVED for item in (*saved, other))
    assert [item.id for item in repository.list_validations_for_opportunity(record.id)] == ["1", "8", "24"]
    assert repository.list_validations_for_opportunity(eth.id) == (other.record,)
    assert [item.id for item in repository.list_recent_validations(limit=2)] == ["24", "8"]
    assert repository.list_validations_for_opportunity("missing") == ()
    assert repository.get_forward_validation_by_id("missing") is None


def test_same_time_distinct_attempts_do_not_overwrite_and_queries_have_stable_ties(setup):
    _, repository, record = setup
    for identifier in ("b", "a"):
        assert repository.save_forward_validation(validate(record), validation_id=identifier,
                                                   created_at=AT).status is P.SAVED
    assert [item.id for item in repository.list_recent_validations()] == ["b", "a"]
    assert [item.id for item in repository.list_validations_for_opportunity(record.id)] == ["a", "b"]


@pytest.mark.parametrize("value", [
    D("123456789012345678901234567890.123456789012345678901234567890"),
    D("0.00000000000000000000000000000000000000000000000012300"),
    D("-0.0000"),
])
def test_decimal_roundtrip_retains_digits_exponent_and_signed_zero(setup, value):
    _, repository, record = setup
    result = validate(record)
    result = replace(result, original_estimated_net_profit=value, original_estimated_net_return=value,
                     validated_simulated_net_return=value, profit_error=value, return_error=value,
                     calculation=replace(result.calculation, net_profit=value, funding_total=value))
    saved = repository.save_forward_validation(result)
    assert saved.status is P.SAVED
    loaded = repository.get_forward_validation_by_id(saved.saved_id)
    for name in ("original_estimated_net_profit", "original_estimated_net_return", "profit_error",
                 "return_error", "validated_simulated_net_profit", "validated_simulated_net_return", "funding_total"):
        assert getattr(loaded, name).as_tuple() == value.as_tuple()
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        text, kind = connection.execute("SELECT validated_simulated_net_profit, "
                                        "typeof(validated_simulated_net_profit) FROM forward_validations").fetchone()
    assert kind == "text" and text == str(value)


def test_utc_datetime_and_duration_roundtrip(setup):
    _, repository, record = setup
    at = AT + timedelta(microseconds=123456)
    local = timezone(timedelta(hours=8))
    result = validate(record, later_snapshot(at=at), at=at)
    result = replace(result, entry_time=OPENED.astimezone(local), validation_time=at.astimezone(local))
    saved = repository.save_forward_validation(result, created_at=at.astimezone(local))
    loaded = repository.get_forward_validation_by_id(saved.saved_id)
    assert loaded.entry_time == OPENED and loaded.entry_time.tzinfo is UTC
    assert loaded.validation_time == at and loaded.validation_time.tzinfo is UTC
    assert loaded.created_at == at and loaded.created_at.microsecond == 123456
    assert loaded.holding_duration == timedelta(hours=2, microseconds=123456)
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        assert connection.execute("SELECT holding_duration FROM forward_validations").fetchone()[0] == "7200123456"


def test_multiple_reasons_and_null_fields_remain_null(setup):
    _, repository, record = setup
    result = validate(record, later_snapshot("ETHUSDT"), at=AT + timedelta(seconds=11))
    assert len(result.reason_codes) >= 2
    saved = repository.save_forward_validation(result)
    assert saved.status is P.SAVED
    loaded = repository.get_forward_validation_by_id(saved.saved_id)
    assert loaded.reason_codes == tuple(code.value for code in result.reason_codes)
    assert loaded.holding_duration is loaded.spot_exit_vwap is loaded.perp_exit_vwap is None
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        assert connection.execute("SELECT validated_simulated_net_profit, holding_duration, "
                                  "spot_exit_fee FROM forward_validations").fetchone() == (None, None, None)


def test_missing_invalid_timestamps_can_be_recorded_without_invention(setup):
    _, repository, record = setup
    result = validate(record, at=AT.replace(tzinfo=None))
    saved = repository.save_forward_validation(result)
    assert saved.status is P.SAVED
    loaded = repository.get_forward_validation_by_id(saved.saved_id)
    assert loaded.entry_time is loaded.validation_time is loaded.holding_duration is None
    assert loaded.reason_codes == (R.INVALID_VALIDATION_TIME.value,)


@pytest.mark.parametrize("case", ["missing_parent", "foreign_symbol_parent"])
def test_foreign_key_rejects_wrong_parent_relationship(setup, case):
    opportunities, repository, record = setup
    if case == "foreign_symbol_parent":
        eth = replace(research_record("ETHUSDT"), id="other")
        opportunities.save_opportunity(eth)
    result = replace(validate(record), opportunity_id="other")
    saved = repository.save_forward_validation(result)
    assert saved.status is P.PERSISTENCE_FAILED
    assert saved.failure.sqlite_errorname == "SQLITE_CONSTRAINT_FOREIGNKEY"
    assert repository.list_recent_validations() == ()


def test_foreign_keys_enabled_on_every_repository_connection_and_protect_parent(setup):
    _, repository, record = setup
    saved = repository.save_forward_validation(validate(record))
    assert saved.status is P.SAVED
    with repository._connection(RepositoryOperation.GET) as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE opportunities SET symbol='ETHUSDT' WHERE id=?", (record.id,))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM opportunities WHERE id=?", (record.id,))


@pytest.mark.parametrize("at", [OPENED, OPENED - timedelta(seconds=1)])
def test_time_not_after_entry_rejected_by_builder_and_database(setup, at):
    _, repository, record = setup
    result = replace(validate(record), validation_time=at, holding_duration=None)
    saved = repository.save_forward_validation(result)
    assert saved.status is P.BUILD_FAILED and "after entry" in saved.failure.message
    assert repository.list_recent_validations() == ()
    saved = repository.save_forward_validation(validate(record))
    with repository._connection(RepositoryOperation.GET) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE forward_validations SET validation_time=entry_time WHERE id=?", (saved.saved_id,))


@pytest.mark.parametrize("failure", ["locked", "trigger", "duplicate"])
def test_write_failure_preserves_validation_and_original_opportunity(setup, failure):
    opportunities, repository, record = setup
    result = validate(record)
    before = exact_json(result)
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        if failure == "locked":
            connection.execute("BEGIN IMMEDIATE")
        elif failure == "trigger":
            connection.execute("CREATE TRIGGER fail_validation BEFORE INSERT ON forward_validations "
                               "BEGIN SELECT RAISE(ABORT,'synthetic write failure'); END")
        else:
            assert repository.save_forward_validation(result, validation_id="same").status is P.SAVED
        saved = repository.save_forward_validation(result, validation_id="same")
        connection.rollback()
    assert saved.status is P.PERSISTENCE_FAILED and saved.saved_id is None
    assert saved.failure.sqlite_errorcode is not None and saved.failure.operation is RepositoryOperation.SAVE
    assert saved.record.status == "VALIDATED" and result.status is S.VALIDATED
    assert exact_json(result) == before
    assert opportunities.get_opportunity_by_id(record.id) == record
    assert len(repository.list_recent_validations()) == (1 if failure == "duplicate" else 0)


def test_init_requires_existing_opportunity_database_and_does_not_create_it(tmp_path):
    path = tmp_path / "missing.sqlite"
    repository = SQLiteForwardValidationRepository(SQLiteConfig(path))
    with pytest.raises(RepositoryError):
        repository.initialize()
    assert not path.exists()


def test_save_requires_explicit_initialization(tmp_path):
    config = SQLiteConfig(tmp_path / "existing.sqlite")
    parent = SQLiteOpportunityRepository(config)
    parent.initialize()
    record = research_record()
    parent.save_opportunity(record)
    repository = SQLiteForwardValidationRepository(config)
    result = repository.save_forward_validation(validate(record))
    assert result.status is P.PERSISTENCE_FAILED
    assert parent.get_opportunity_by_id(record.id) == record


def test_bound_query_parameters_and_missing_ids(setup):
    _, repository, record = setup
    identifier = "'); DROP TABLE forward_validations; --"
    saved = repository.save_forward_validation(validate(record), validation_id=identifier)
    assert saved.status is P.SAVED
    assert repository.get_forward_validation_by_id(identifier) == saved.record
    assert repository.get_forward_validation_by_id("' OR 1=1 --") is None
    assert repository.list_validations_for_opportunity("' OR 1=1 --") == ()


@pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5])
def test_invalid_query_limit(setup, limit):
    with pytest.raises(ValueError):
        setup[1].list_recent_validations(limit)


def test_float_rejected_without_mutating_input(setup):
    _, repository, record = setup
    result = replace(validate(record), profit_error=0.1)
    saved = repository.save_forward_validation(result)
    assert saved.status is P.BUILD_FAILED
    assert repository.list_recent_validations() == () and result.profit_error == 0.1


def test_saving_does_not_call_validator_or_math(setup, monkeypatch):
    _, repository, record = setup
    result = validate(record)
    forbidden = Mock(side_effect=AssertionError("persistence must not calculate"))
    for name in ("validate_forward_outcome", "round_trip_from_fills", "decompose_fills", "net_return",
                 "accrue_short_funding", "simulate_fill", "calculate_fee"):
        monkeypatch.setattr(f"crypto_arbitrage_bot.analytics.forward_validation.{name}", forbidden)
    saved = repository.save_forward_validation(result)
    assert saved.status is P.SAVED
    assert json.loads(saved.record.details_json)["calculation"]["net_profit"] == str(result.validated_simulated_net_profit)
    forbidden.assert_not_called()
