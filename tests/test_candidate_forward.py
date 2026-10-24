"""Candidate forward observations use original math/scheduler and no orders."""

import ast
import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_candidate_catalog import NOW
from test_candidate_economics import compare
from test_candidate_economics import settings as settings
from test_candidate_market import catalog as catalog
from test_candidate_market import collector, install
from test_candidate_market import criteria as criteria
from test_candidate_market import offline as offline
from test_forward_validation import shift_fixture

from crypto_arbitrage_bot import candidate_forward_research as cli
from crypto_arbitrage_bot.analytics.entry_evidence import decode_entry
from crypto_arbitrage_bot.analytics.forward_validation import validate_forward_outcome
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.arbitrage.profit import decomposed_net_profit, round_trip_from_fills
from crypto_arbitrage_bot.exchanges.funding_history import FundingFailure, FundingWindow
from crypto_arbitrage_bot.models.forward_validation import ValidationReason as R
from crypto_arbitrage_bot.models.forward_validation import ValidationStatus as S
from crypto_arbitrage_bot.models.market import SUPPORTED_SYMBOLS, OrderBookLevel
from crypto_arbitrage_bot.models.validation_task import TaskFailure, TaskStatus
from crypto_arbitrage_bot.storage.forward_validation_repository import (
    SQLiteForwardValidationRepository,
    ValidationPersistenceStatus,
    ValidationSaveResult,
)
from crypto_arbitrage_bot.storage.opportunity_records import build_opportunity, exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import SQLiteConfig, SQLiteOpportunityRepository
from crypto_arbitrage_bot.storage.validation_tasks import SQLiteValidationTaskRepository

D = Decimal


@pytest.fixture
def prepared(monkeypatch, catalog, criteria, settings, tmp_path):
    install(monkeypatch)
    db = SQLiteConfig(tmp_path / "candidate.sqlite")
    repository = SQLiteOpportunityRepository(db)
    repository.initialize()
    records, observations = {}, {}
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        observation = collector(catalog, criteria).collect(symbol)
        comparison = compare(catalog, criteria, observation, settings)[0]
        record = build_opportunity(comparison.scenarios[0].estimate, comparison.decision,
                                   created_at=NOW, opportunity_id=symbol)
        repository.save_opportunity(record)
        records[symbol] = repository.get_opportunity_by_id(symbol)
        observations[symbol] = observation
    research = settings.research.model_copy(update={"sqlite_path": db.database_path})
    configured = settings.model_copy(update={"research": research})
    return configured, repository, records, observations


def exit_books(observation, at, *, ask=None, shallow=None):
    assembly = shift_fixture(observation.assembly, at - NOW)
    if ask is not None:
        book = assembly.snapshot.perp_book
        book = replace(book, bids=(OrderBookLevel(D(ask) - D(2), D(5)),),
                       asks=(OrderBookLevel(D(ask), D(5)),))
        source = assembly.sources.perp_depth
        assembly = replace(assembly, snapshot=replace(assembly.snapshot, perp_book=book),
            sources=replace(assembly.sources, perp_depth=replace(source, data=replace(source.data, order_book=book))))
    if shallow is not None:
        market = "spot" if shallow == "spot" else "perp"
        book = getattr(assembly.snapshot, market + "_book")
        book = replace(book, bids=(OrderBookLevel(book.best_bid, D(".01")),),
                       asks=(OrderBookLevel(book.best_ask, D(".01")),))
        snapshot = replace(assembly.snapshot, **{market + "_book": book})
        sources = assembly.sources
        if market == "spot":
            sources = replace(sources, spot_depth=replace(sources.spot_depth, order_book=book))
        else:
            source = sources.perp_depth
            sources = replace(sources, perp_depth=replace(source, data=replace(source.data, order_book=book)))
        assembly = replace(assembly, snapshot=snapshot, sources=sources)
    return assembly


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
@pytest.mark.parametrize("rate", [None, ".01", "-.01", "0"])
def test_original_cash_flow_and_decomposition_apply_without_duplicate_costs(prepared, symbol, rate):
    _, repository, records, observations = prepared
    record = records[symbol]
    original = exact_json(record)
    at = NOW + timedelta(hours=8)
    events = () if rate is None else (FundingEvent("observed", at, D(100), D(rate), symbol),)
    with localcontext() as context:
        context.prec = 3
        result = validate_forward_outcome(record, exit_books(observations[symbol], at),
                                         validation_time=at, funding_events=events)
    assert result.status is S.VALIDATED
    assert result.price_pnl == -4
    assert result.funding_total == (D(0) if rate is None else D(100) * D(rate))
    assert result.validated_simulated_net_profit == D("-4.3") + result.funding_total
    assert result.fees.total == D(".3") and result.matched_quantity == 1
    assert result.entry_fills == (decode_entry(record).spot, decode_entry(record).perp)
    identity = observations[symbol].assembly.research_instrument
    assert round_trip_from_fills(result.entry_fills[0], result.exit_fills[0],
        result.entry_fills[1], result.exit_fills[1], result.fees, result.funding_total,
        research_instrument=identity) == result.calculation
    assert decomposed_net_profit(result.decomposition, result.funding_total, result.fees) == result.validated_simulated_net_profit
    validations = SQLiteForwardValidationRepository(repository.config)
    validations.initialize()
    saved = validations.save_forward_validation(result, created_at=at)
    assert saved.status is ValidationPersistenceStatus.SAVED
    reread = validations.get_forward_validation_by_id(saved.saved_id)
    assert reread.validated_simulated_net_profit == result.validated_simulated_net_profit
    assert reread.validated_simulated_net_return == result.validated_simulated_net_return
    assert exact_json(repository.get_opportunity_by_id(record.id)) == original


@pytest.mark.parametrize("ask,pnl,profit", [("90", "7", "6.7055"), ("97", "0", "-.298"), ("120", "-23", "-23.3095")])
def test_actual_basis_change_can_converge_stay_constant_or_worsen(prepared, ask, pnl, profit):
    _, _, records, observations = prepared
    at = NOW + timedelta(hours=1)
    result = validate_forward_outcome(records["SOLUSDT"], exit_books(observations["SOLUSDT"], at, ask=ask),
                                     validation_time=at, funding_events=())
    assert result.status is S.VALIDATED
    assert result.price_pnl == D(pnl)
    assert result.validated_simulated_net_profit == D(profit)
    assert result.spot_pnl + result.perp_pnl == result.price_pnl


@pytest.mark.parametrize("case", ["missing", "wrong_symbol", "scaled", "book_identity", "book_symbol"])
def test_modified_or_stripped_record_identity_cannot_authorize_alt_math(prepared, case):
    _, _, records, observations = prepared
    record = records["SOLUSDT"]
    details = json.loads(record.details_json)
    snapshot = details["snapshot"]
    if case == "missing":
        del details["snapshot"]
    elif case == "wrong_symbol":
        snapshot["research_instrument"]["symbol"] = "ETHUSDT"
    elif case == "scaled":
        snapshot["research_instrument"] = {"symbol": "1000PEPEUSDT", "base_asset": "1000PEPE"}
    elif case == "book_identity":
        snapshot["spot_book"]["research_instrument"] = None
    else:
        snapshot["perp_book"]["symbol"] = "ETHUSDT"
    record = replace(record, details_json=exact_json(details))
    at = NOW + timedelta(hours=1)
    result = validate_forward_outcome(record, exit_books(observations["SOLUSDT"], at),
                                     validation_time=at, funding_events=())
    assert result.status is S.INVALID and result.reason_codes == (R.INVALID_ENTRY_EVIDENCE,)
    assert result.validated_simulated_net_profit is None


def test_exit_identity_must_equal_recorded_identity(prepared):
    _, _, records, observations = prepared
    at = NOW + timedelta(hours=1)
    assembly = replace(exit_books(observations["SOLUSDT"], at), research_instrument=None)
    result = validate_forward_outcome(records["SOLUSDT"], assembly, validation_time=at, funding_events=())
    assert result.status is S.INVALID and R.RESEARCH_IDENTITY_MISMATCH in result.reason_codes
    assert result.validated_simulated_net_profit is None


@pytest.mark.parametrize("offset", [0, -1, 3600000001])
def test_candidate_funding_outside_window_never_counts(prepared, offset):
    _, _, records, observations = prepared
    at = NOW + timedelta(hours=1)
    event = FundingEvent("outside", NOW + timedelta(microseconds=offset), D(100), D(".01"), "SOLUSDT")
    result = validate_forward_outcome(records["SOLUSDT"], exit_books(observations["SOLUSDT"], at),
                                     validation_time=at, funding_events=(event,))
    assert result.status is S.INVALID
    assert result.reason_codes == (R.FUNDING_EVENT_OUTSIDE_HOLDING_WINDOW,)
    assert result.funding_total is result.validated_simulated_net_profit is None


@pytest.mark.parametrize("market", ["spot", "perp"])
def test_candidate_exit_depth_cannot_be_extrapolated(prepared, market):
    _, _, records, observations = prepared
    at = NOW + timedelta(hours=1)
    result = validate_forward_outcome(records["SOLUSDT"], exit_books(observations["SOLUSDT"], at, shallow=market),
                                     validation_time=at, funding_events=())
    assert result.status is S.INCOMPLETE and R.INSUFFICIENT_EXIT_DEPTH in result.reason_codes
    assert result.validated_simulated_net_profit is result.validated_simulated_net_return is None


def bridge(monkeypatch, configured, observations, at, *, horizon=3600, fault=None):
    settings = configured.research.validation_scheduler.model_copy(update={"validation_horizons_seconds": (horizon,)})
    configured = configured.model_copy(update={"research": configured.research.model_copy(update={"validation_scheduler": settings})})
    source = Mock()
    def collect(symbol):
        if fault == "api" and symbol == "BTCUSDT":
            raise TimeoutError("offline public timeout")
        observed = observations[symbol]
        delta = timedelta(minutes=1) if fault == "stale" and symbol == "BTCUSDT" else timedelta(0)
        return replace(observed, assembly=exit_books(observed, at - delta))
    source.collect.side_effect = collect
    def funding(symbol, opened, closed):
        events = (FundingEvent("actual-settlement", NOW + timedelta(hours=1), D(100), D(".01"), symbol),)
        if fault == "funding" and symbol == "BTCUSDT":
            return FundingWindow(symbol, opened, closed, (), failure=FundingFailure.TIMEOUT)
        if fault == "missing" and symbol == "BTCUSDT":
            events = ()
        return FundingWindow(symbol, opened, closed, events)
    source.fetch_funding_window.side_effect = funding
    monkeypatch.setattr(cli.CandidateValidationSources, "_current", lambda _: source)
    monkeypatch.setattr(cli, "utc_now", lambda: at)
    return configured, source


@pytest.mark.parametrize("horizon", [3600, 28800, 86400])
def test_candidate_due_horizons_use_original_scheduler_and_fixed_ids(prepared, monkeypatch, tmp_path, horizon):
    configured, repo, records, observations = prepared
    at = NOW + timedelta(seconds=horizon)
    configured, source = bridge(monkeypatch, configured, observations, at, horizon=horizon)
    original = exact_json(tuple(records.values()))
    result = cli.run_due_once(configured, tmp_path / "due.json")
    validations = SQLiteForwardValidationRepository(repo.config)
    tasks = SQLiteValidationTaskRepository(repo.config)
    for record in records.values():
        task, = tasks.list_for_opportunity(record.id)
        saved, = validations.list_validations_for_opportunity(record.id)
        assert task.status is TaskStatus.COMPLETED
        assert task.due_at == at and saved.validation_time == at
        assert saved.holding_duration == timedelta(seconds=horizon)
        assert saved.funding_total == 1 and saved.validated_simulated_net_profit == D("-3.3")
    evidence = json.loads((tmp_path / "due.json").read_text(encoding="utf-8"))
    assert len(evidence["captures"]) == 6
    assert all(c["observation"]["assembly"]["sources"]["spot_depth"]["raw_response"] is not None
               for c in evidence["captures"] if c["phase"] == "EXIT_MARKET")
    assert len(result["task_events"]) == 3
    cli.run_due_once(configured, tmp_path / "restart.json")
    assert source.collect.call_count == source.fetch_funding_window.call_count == 3
    assert exact_json(tuple(repo.get_opportunity_by_id(r.id) for r in records.values())) == original


@pytest.mark.parametrize("offset", [-1, 301])
def test_before_due_and_expired_candidates_never_fetch_or_fake_horizon(prepared, monkeypatch, tmp_path, offset):
    configured, repo, records, observations = prepared
    configured, source = bridge(monkeypatch, configured, observations, NOW + timedelta(seconds=3600 + offset))
    cli.run_due_once(configured, tmp_path / "not-due.json")
    source.collect.assert_not_called()
    source.fetch_funding_window.assert_not_called()
    tasks = SQLiteValidationTaskRepository(repo.config)
    for record in records.values():
        task, = tasks.list_for_opportunity(record.id)
        if offset < 0:
            assert task.status is TaskStatus.PENDING
        else:
            assert task.failure_code is TaskFailure.VALIDATION_WINDOW_MISSED


@pytest.mark.parametrize("fault,code", [("api", TaskFailure.COLLECTION_FAILED),
    ("stale", TaskFailure.INVALID_EXIT_MARKET_DATA), ("funding", TaskFailure.FUNDING_UNAVAILABLE),
    ("missing", TaskFailure.FUNDING_UNAVAILABLE)])
def test_failure_cannot_block_other_candidates_or_create_fake_validations(prepared, monkeypatch, tmp_path, fault, code):
    configured, repo, _, observations = prepared
    configured, _ = bridge(monkeypatch, configured, observations, NOW + timedelta(hours=1), fault=fault)
    cli.run_due_once(configured, tmp_path / "fail.json")
    tasks = SQLiteValidationTaskRepository(repo.config)
    validations = SQLiteForwardValidationRepository(repo.config)
    assert tasks.list_for_opportunity("BTCUSDT")[0].failure_code is code
    assert validations.list_validations_for_opportunity("BTCUSDT") == ()
    for symbol in ("ETHUSDT", "SOLUSDT"):
        assert tasks.list_for_opportunity(symbol)[0].status is TaskStatus.COMPLETED


def test_persistence_failure_preserves_original_entries(prepared, monkeypatch, tmp_path):
    configured, repo, records, observations = prepared
    configured, _ = bridge(monkeypatch, configured, observations, NOW + timedelta(hours=1))
    before = exact_json(tuple(records.values()))
    monkeypatch.setattr(cli.SQLiteForwardValidationRepository, "save_forward_validation",
                        Mock(return_value=ValidationSaveResult(ValidationPersistenceStatus.PERSISTENCE_FAILED)))
    cli.run_due_once(configured, tmp_path / "db-fail.json")
    tasks = SQLiteValidationTaskRepository(repo.config)
    assert all(tasks.list_for_opportunity(r.id)[0].failure_code is TaskFailure.PERSISTENCE_FAILED for r in records.values())
    assert exact_json(tuple(repo.get_opportunity_by_id(r.id) for r in records.values())) == before


def test_evidence_overwrite_is_rejected_before_scheduling(prepared, tmp_path, monkeypatch):
    configured, _, _, _ = prepared
    path = tmp_path / "retained.json"
    path.write_text("retained", encoding="utf-8")
    process = Mock(side_effect=AssertionError("must not process"))
    monkeypatch.setattr(cli.ValidationScheduler, "process_due", process)
    with pytest.raises(FileExistsError):
        cli.run_due_once(configured, path)
    assert path.read_text(encoding="utf-8") == "retained" and not process.called


def test_bridge_refreshes_metadata_without_falling_back_to_stale_catalog(settings, catalog, monkeypatch):
    bridge = cli.CandidateValidationSources(settings)
    now = [NOW]
    monkeypatch.setattr(cli, "utc_now", lambda: now[0])
    collect = Mock(return_value=catalog)
    factory = Mock()
    monkeypatch.setattr(cli, "collect_catalog", collect)
    monkeypatch.setattr(cli, "CandidateMarketCollector", factory)
    first = bridge._current()
    assert bridge._current() is first and collect.call_count == 1
    now[0] += timedelta(seconds=settings.research.candidate_catalog.max_metadata_age_seconds + 1)
    collect.side_effect = TimeoutError("metadata refresh failed")
    with pytest.raises(TimeoutError):
        bridge._current()
    assert bridge.catalog is None and bridge.collector is None


def test_candidate_forward_entrypoint_has_no_execution_or_network_bypass():
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert all(".execution" not in name for name in imports)
    assert SUPPORTED_SYMBOLS.keys() == {"BTCUSDT", "ETHUSDT"}
    assert not any(isinstance(node, (ast.While, ast.For)) for node in ast.walk(tree))


def test_catalog_assembly_time_cannot_reset_old_source_request_age(settings, catalog, monkeypatch):
    bridge = cli.CandidateValidationSources(settings)
    bridge.catalog = replace(catalog, evaluated_at=NOW + timedelta(seconds=100))
    bridge.collector = Mock()
    monkeypatch.setattr(cli, "utc_now", lambda: NOW + timedelta(seconds=181))
    collect = Mock(side_effect=TimeoutError("must refresh original source age"))
    monkeypatch.setattr(cli, "collect_catalog", collect)
    with pytest.raises(TimeoutError):
        bridge._current()
    assert collect.called and bridge.catalog is None and bridge.collector is None
