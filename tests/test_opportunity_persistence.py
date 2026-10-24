"""Offline one-shot research persistence; no recalculation or live requests."""

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, timedelta, timezone
from decimal import Decimal, Inexact, localcontext
from unittest.mock import Mock

import pytest
from test_decision import thresholds
from test_offline_estimate import OPENED, assembly_for, books, config, scenario_for
from test_scenario_decision import limits, scenarios

from crypto_arbitrage_bot.arbitrage.decision import decide
from crypto_arbitrage_bot.arbitrage.estimate import estimate_once
from crypto_arbitrage_bot.arbitrage.scenario import evaluate_scenarios
from crypto_arbitrage_bot.arbitrage.scenario_decision import decide_scenarios
from crypto_arbitrage_bot.market_data.snapshot import assemble_snapshot
from crypto_arbitrage_bot.models.decision import Decision
from crypto_arbitrage_bot.storage.opportunity_records import build_opportunity, exact_json
from crypto_arbitrage_bot.storage.persistence import PersistenceStatus as S
from crypto_arbitrage_bot.storage.persistence import build_and_save_opportunity
from crypto_arbitrage_bot.storage.record_validation import RecordBuildError
from crypto_arbitrage_bot.storage.record_validation import RecordBuildReason as R
from crypto_arbitrage_bot.storage.sqlite_repository import (
    RepositoryOperation,
    SQLiteConfig,
    SQLiteOpportunityRepository,
)

D = Decimal


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("persistence must stay offline"))
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("http.client.HTTPSConnection", forbidden)


@pytest.fixture
def repository(tmp_path):
    repo = SQLiteOpportunityRepository(SQLiteConfig(tmp_path / "research.sqlite", busy_timeout_ms=0))
    repo.initialize()
    return repo


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
@pytest.mark.parametrize("status,changes", [
    (Decision.PASS, {}),
    (Decision.OBSERVE, {"min_convergence_profit": D("26")}),
    (Decision.REJECT, {"max_stress_loss": D("1")}),
])
def test_full_chain_saved_and_read_without_changing_any_evidence(repository, symbol, status, changes):
    evidence = scenarios(symbol)
    primary = evidence[0].estimate
    decision = decide_scenarios(evidence, limits(**changes))
    before = exact_json((primary, evidence, decision))
    local_time = OPENED.astimezone(timezone(timedelta(hours=8))).replace(microsecond=123456)
    result = build_and_save_opportunity(
        repository, primary, decision, created_at=local_time,
        formula_version="synthetic-formula-1", config_version="synthetic-config-1",
    )
    assert result.status is S.SAVED and result.failure is None
    loaded = repository.get_opportunity_by_id(result.saved_id)
    assert loaded == result.record
    assert loaded.symbol == symbol and loaded.decision == status.value
    assert loaded.created_at == local_time and loaded.created_at.tzinfo is UTC
    assert loaded.created_at.microsecond == 123456
    assert loaded.formula_version == "synthetic-formula-1"
    assert loaded.config_version == "synthetic-config-1"
    assert loaded.decision_reason_codes == tuple(code.value for code in decision.reason_codes)
    assert loaded.data_quality_status == primary.data_quality.status.value
    assert loaded.data_quality_reason_codes == tuple(code.value for code in primary.data_quality.reason_codes)
    for name in ("matched_quantity", "residual_delta", "executable_entry_basis", "basis_pct",
                 "estimated_funding", "price_pnl", "capital_occupied", "net_profit", "net_return"):
        assert getattr(loaded, name).as_tuple() == getattr(primary, name).as_tuple()
    assert loaded.matched_notional == min(primary.fills[0].quote_notional, primary.fills[2].quote_notional)
    assert loaded.spot_fees == primary.calculation.spot_fees
    assert loaded.perp_fees == primary.calculation.perp_fees
    snapshot = primary.assembly.snapshot
    for name in ("mark_price", "index_price", "funding_rate", "open_interest", "next_funding_time"):
        assert getattr(loaded, name) == getattr(snapshot, name)
    assert (loaded.spot_bid, loaded.spot_ask) == (snapshot.spot_book.best_bid, snapshot.spot_book.best_ask)
    assert (loaded.perp_bid, loaded.perp_ask) == (snapshot.perp_book.best_bid, snapshot.perp_book.best_ask)
    for saved, source in zip(loaded.scenarios, evidence, strict=True):
        assert saved.name == source.name and saved.scenario_type == source.scenario_type.value
        assert saved.net_profit.as_tuple() == source.net_profit.as_tuple()
        assert saved.net_return.as_tuple() == source.net_return.as_tuple()
        assert saved.funding_total == source.funding_total
        assert json.loads(saved.details_json)["estimate"]["calculation"]["price_pnl"] == str(source.price_pnl)
    for saved in loaded.source_timings:
        source = getattr(primary.assembly.sources, saved.source)
        assert saved.request_started_at == source.started_at
        assert saved.request_finished_at == source.finished_at
        assert saved.received_at == source.received_at
        assert saved.request_started_at.tzinfo is UTC
    assert loaded.source_timings[0].event_time is None
    assert loaded.source_timings[1].event_time == snapshot.perp_book.event_time
    assert loaded.source_timings[1].transaction_time == primary.assembly.sources.perp_depth.data.transaction_time
    assert exact_json((primary, evidence, decision)) == before


@pytest.mark.parametrize("rate", [".010000000000000000000000000000123", "-.010000000000000000000000000000123", "0.0000"])
def test_decimal_mapping_uses_no_arithmetic_or_float(repository, rate):
    source = books(recurring=True)
    estimate = estimate_once(assembly_for((source[0], source[2]), indicative_rate=rate),
                             config(target_quantity=D("3")), scenario_for(source, rate=rate))
    decision = decide(estimate, thresholds())
    with localcontext() as context:
        context.prec = 2
        context.traps[Inexact] = True
        result = build_and_save_opportunity(repository, estimate, decision)
    assert result.status is S.SAVED
    loaded = repository.get_opportunity_by_id(result.saved_id)
    assert loaded.funding_rate.as_tuple() == D(rate).as_tuple()
    assert loaded.estimated_funding.as_tuple() == estimate.estimated_funding.as_tuple()
    assert loaded.net_profit.as_tuple() == estimate.net_profit.as_tuple()
    assert loaded.net_return.as_tuple() == estimate.net_return.as_tuple()
    assert loaded.matched_notional == D("305")


def test_invalid_quality_scenarios_can_be_recorded_with_null_profit(repository):
    source = scenarios()
    complete = source[0].estimate.assembly
    incomplete = assemble_snapshot(complete.symbol, replace(complete.sources, spot_depth=None,
                                                            open_interest=None), assembled_at=OPENED)
    evidence = evaluate_scenarios(incomplete, config(), tuple(item.definition for item in source))
    decision = decide_scenarios(evidence, limits())
    result = build_and_save_opportunity(repository, evidence[0].estimate, decision)
    assert result.status is S.SAVED
    loaded = repository.get_opportunity_by_id(result.saved_id)
    assert loaded.decision == "REJECT" and loaded.data_quality_status == "INVALID"
    assert loaded.data_quality_reason_codes == tuple(code.value for code in evidence[0].estimate.data_quality.reason_codes)
    assert len(loaded.data_quality_reason_codes) > 1 and len(loaded.decision_reason_codes) > 1
    assert loaded.net_profit is loaded.net_return is loaded.matched_quantity is None
    assert loaded.spot_bid is loaded.funding_rate is loaded.open_interest is None
    assert loaded.formula_version is loaded.config_version is None
    assert all(item.net_profit is None and item.net_return is None for item in loaded.scenarios)
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        assert connection.execute("SELECT net_profit, funding_rate FROM opportunities").fetchone() == (None, None)


@pytest.mark.parametrize("mutation,expected", [
    ("symbol", R.SYMBOL_MISMATCH),
    ("snapshot_time", R.SNAPSHOT_MISMATCH),
    ("source_time", R.SNAPSHOT_MISMATCH),
    ("quantity", R.QUANTITY_MISMATCH),
    ("definition", R.SCENARIO_MISMATCH),
])
@pytest.mark.parametrize("scenario_decision", [True, False])
def test_inconsistent_evidence_never_calls_repository(repository, monkeypatch, mutation, expected, scenario_decision):
    evidence = scenarios()
    primary = evidence[0].estimate
    other = evidence[1]
    if mutation == "symbol":
        other = scenarios("ETHUSDT")[1]
    elif mutation == "quantity":
        other = scenarios(target_quantity=D("1"))[1]
    elif mutation == "definition":
        other = replace(other, definition=replace(other.definition, exit_scenario=primary.scenario))
    else:
        assembly = other.estimate.assembly
        if mutation == "snapshot_time":
            assembly = replace(assembly, assembled_at=OPENED + timedelta(microseconds=1))
        else:
            depth = replace(assembly.sources.spot_depth, started_at=OPENED - timedelta(seconds=1))
            assembly = replace(assembly, sources=replace(assembly.sources, spot_depth=depth))
        other = replace(other, estimate=replace(other.estimate, assembly=assembly))
    mixed = (evidence[0], other, evidence[2])
    decision = (replace(decide_scenarios(evidence, limits()), scenarios=mixed) if scenario_decision
                else decide(primary, thresholds()))
    save = Mock(side_effect=AssertionError("inconsistent data must not reach SQLite"))
    monkeypatch.setattr(repository, "save_opportunity", save)
    result = build_and_save_opportunity(repository, primary, decision, scenarios=mixed)
    assert result.status is S.BUILD_FAILED and result.record is None and result.saved_id is None
    assert expected in result.failure.reason_codes
    save.assert_not_called()
    assert repository.list_recent_opportunities() == ()
    with pytest.raises(RecordBuildError):
        build_opportunity(primary, decision, scenarios=mixed)


@pytest.mark.parametrize("source_field", ["spot_depth", "perp_depth", "premium_index", "open_interest"])
def test_mismatched_source_request_symbol_is_rejected(repository, source_field):
    estimate = scenarios()[0].estimate
    sources = estimate.assembly.sources
    source = getattr(sources, source_field)
    params = tuple((name, "ETHUSDT" if name == "symbol" else value) for name, value in source.request.params)
    source = replace(source, request=replace(source.request, params=params))
    estimate = replace(estimate, assembly=replace(estimate.assembly, sources=replace(sources, **{source_field: source})))
    result = build_and_save_opportunity(repository, estimate, decide(estimate, thresholds()))
    assert result.status is S.BUILD_FAILED
    assert R.SYMBOL_MISMATCH in result.failure.reason_codes
    assert repository.list_recent_opportunities() == ()


def test_decision_provenance_mismatch_is_structured(repository):
    evidence = scenarios()
    other = scenarios("ETHUSDT")[0].estimate
    result = build_and_save_opportunity(repository, other, decide_scenarios(evidence, limits()))
    assert result.status is S.BUILD_FAILED
    assert result.failure.reason_codes == (R.DECISION_MISMATCH,)
    assert repository.list_recent_opportunities() == ()


def test_multiple_mismatches_and_template_symbol_are_reported(repository):
    evidence = scenarios()
    foreign = scenarios("ETHUSDT", target_quantity=D("1"))[1]
    mixed = (evidence[0], foreign, evidence[2])
    decision = replace(decide_scenarios(evidence, limits()), scenarios=mixed)
    result = build_and_save_opportunity(repository, evidence[0].estimate, decision)
    assert result.status is S.BUILD_FAILED
    assert {R.SYMBOL_MISMATCH, R.SNAPSHOT_MISMATCH, R.QUANTITY_MISMATCH} <= set(result.failure.reason_codes)
    definition = replace(evidence[1].definition, basis_assumption=foreign.definition.basis_assumption)
    mixed = (evidence[0], replace(evidence[1], definition=definition), evidence[2])
    decision = replace(decision, scenarios=mixed)
    result = build_and_save_opportunity(repository, evidence[0].estimate, decision)
    assert result.status is S.BUILD_FAILED
    assert R.SYMBOL_MISMATCH in result.failure.reason_codes
    assert repository.list_recent_opportunities() == ()


@pytest.mark.parametrize("metadata", [{"formula_version": ""}, {"created_at": OPENED.replace(tzinfo=None)}])
def test_invalid_metadata_is_a_build_failure(repository, metadata):
    evidence = scenarios()
    result = build_and_save_opportunity(repository, evidence[0].estimate,
                                        decide_scenarios(evidence, limits()), **metadata)
    assert result.status is S.BUILD_FAILED
    assert result.failure.reason_codes == (R.INVALID_INPUT,)
    assert repository.list_recent_opportunities() == ()


def test_equivalent_snapshot_values_are_accepted(repository):
    evidence = scenarios()
    other = replace(evidence[1], estimate=replace(evidence[1].estimate,
                                                 assembly=replace(evidence[1].estimate.assembly)))
    evidence = (evidence[0], other, evidence[2])
    result = build_and_save_opportunity(repository, evidence[0].estimate, decide_scenarios(evidence, limits()))
    assert result.status is S.SAVED


@pytest.mark.parametrize("failure", ["locked", "child_insert", "duplicate"])
def test_database_failure_is_not_a_research_rejection(repository, failure):
    evidence = scenarios()
    decision = decide_scenarios(evidence, limits())
    before = exact_json((evidence, decision))
    with closing(sqlite3.connect(repository.config.database_path)) as connection:
        if failure == "locked":
            connection.execute("BEGIN IMMEDIATE")
        elif failure == "child_insert":
            connection.execute("CREATE TRIGGER fail_child BEFORE INSERT ON opportunity_scenarios "
                               "BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END")
        else:
            first = build_and_save_opportunity(repository, evidence[0].estimate, decision, opportunity_id="same")
            assert first.status is S.SAVED
        result = build_and_save_opportunity(repository, evidence[0].estimate, decision, opportunity_id="same")
        connection.rollback()
    assert result.status is S.PERSISTENCE_FAILED and result.saved_id is None
    assert result.failure.operation is RepositoryOperation.SAVE
    assert result.failure.sqlite_errorcode is not None
    assert result.failure.reason_codes == ()
    assert result.record.decision == "PASS" and decision.status is Decision.PASS
    assert exact_json((evidence, decision)) == before
    assert len(repository.list_recent_opportunities()) == (1 if failure == "duplicate" else 0)


def test_integration_does_not_reinvoke_any_research_engine(repository, monkeypatch):
    evidence = scenarios()
    decision = decide_scenarios(evidence, limits())
    forbidden = Mock(side_effect=AssertionError("must reuse computed evidence"))
    for path in (
        "crypto_arbitrage_bot.arbitrage.estimate.estimate_once",
        "crypto_arbitrage_bot.arbitrage.scenario.evaluate_scenarios",
        "crypto_arbitrage_bot.arbitrage.scenario.evaluate_scenario",
        "crypto_arbitrage_bot.arbitrage.decision.decide",
        "crypto_arbitrage_bot.arbitrage.scenario_decision.decide_scenarios",
    ):
        monkeypatch.setattr(path, forbidden)
    result = build_and_save_opportunity(repository, evidence[0].estimate, decision)
    assert result.status is S.SAVED
    forbidden.assert_not_called()
