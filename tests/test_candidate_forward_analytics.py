"""Offline candidate summaries preserve the original analytics and trade boundary."""

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal, Inexact, localcontext
from unittest.mock import Mock

import pytest
from test_candidate_catalog import NOW
from test_candidate_forward import catalog as catalog
from test_candidate_forward import criteria as criteria
from test_candidate_forward import exit_books
from test_candidate_forward import offline as offline
from test_candidate_forward import prepared as prepared
from test_candidate_forward import settings as settings

from crypto_arbitrage_bot.analytics.forward_sqlite_summary import summarize_forward_database
from crypto_arbitrage_bot.analytics.forward_summary import (
    ForwardAnalyticsConfig,
    ValidationObservation,
    summarize_forward_observations,
)
from crypto_arbitrage_bot.analytics.forward_validation import validate_forward_outcome
from crypto_arbitrage_bot.analytics.sqlite_summary import AnalyticsReadError
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.models.decision import Decision
from crypto_arbitrage_bot.models.forward_validation import ValidationStatus as S
from crypto_arbitrage_bot.models.market import SUPPORTED_SYMBOLS, Side
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.storage.forward_validation_repository import (
    SQLiteForwardValidationRepository,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import SQLiteConfig, SQLiteOpportunityRepository

D = Decimal
SOL = ResearchInstrument("SOLUSDT", "SOL")
CONFIG = ForwardAnalyticsConfig(D(0), (timedelta(hours=1),timedelta(hours=8),timedelta(hours=24)), (SOL,))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("offline analytics")))


def summarize(prepared, config=CONFIG):
    return summarize_forward_database(prepared[1].config, config)


def save(prepared, symbol="SOLUSDT", *, status=S.VALIDATED, rate=None, hours=8):
    _, parents, records, observations = prepared
    at=NOW+timedelta(hours=hours)
    events=() if rate is None else (FundingEvent("actual-explicit",at-timedelta(minutes=1),D(100),D(rate),symbol),)
    snapshot=exit_books(observations[symbol],at,shallow="spot" if status is S.INCOMPLETE else None)
    result=validate_forward_outcome(records[symbol],snapshot,
        validation_time=at+timedelta(seconds=11) if status is S.INVALID else at,funding_events=events)
    assert result.status is status
    repository=SQLiteForwardValidationRepository(parents.config)
    repository.initialize()
    stored=repository.save_forward_validation(result)
    assert stored.saved_id is not None
    return result,stored.record


def test_candidate_parents_without_validation_are_counted(prepared):
    SQLiteForwardValidationRepository(prepared[1].config).initialize()
    result=summarize(prepared)
    assert result.total_opportunities==result.unvalidated_opportunities==3
    assert result.overall.validation_count==0
    assert tuple(g.key for g in result.by_symbol)==("BTCUSDT","ETHUSDT","SOLUSDT")
    assert result.overall.simulated_net_profit.mean is None


def test_candidates_need_explicit_identity_and_are_never_inferred_from_symbol(prepared):
    save(prepared)
    with pytest.raises(AnalyticsReadError,match="explicit research identity"):
        summarize(prepared,replace(CONFIG,research_instruments=()))


@pytest.mark.parametrize("status", list(S))
def test_all_statuses_preserve_saved_results_and_null_semantics(prepared,status):
    result,stored=save(prepared,status=status)
    before=prepared[1].config.database_path.read_bytes()
    summary=summarize(prepared)
    group=next(g.outcomes for g in summary.by_symbol if g.key=="SOLUSDT")
    assert group.validation_count==1
    assert next(s.count for s in group.statuses if s.status is status)==1
    assert group.simulated_net_profit.count==(1 if status is S.VALIDATED else 0)
    assert group.simulated_net_profit.mean==result.validated_simulated_net_profit
    assert summary.earliest_validation_time==stored.validation_time
    assert prepared[1].config.database_path.read_bytes()==before
    assert "not actual execution" in summary.interpretation


@pytest.mark.parametrize("rate,expected_group",[(".01","POSITIVE"),("0","ZERO"),("-.01","NEGATIVE")])
def test_candidate_actual_funding_signs_use_saved_values(prepared,rate,expected_group):
    result,_=save(prepared,rate=rate)
    summary=summarize(prepared)
    assert next(g.outcomes.validation_count for g in summary.by_funding if g.key==expected_group)==1
    assert summary.overall.simulated_net_profit.mean==result.validated_simulated_net_profit


def test_three_symbols_multiple_horizons_include_every_validation(prepared):
    expected=[]
    for symbol,hours in (("BTCUSDT",1),("ETHUSDT",8),("SOLUSDT",8),("SOLUSDT",24)):
        expected.append(save(prepared,symbol,hours=hours)[0].validated_simulated_net_profit)
    result=summarize(prepared)
    assert result.overall.validation_count==4 and result.opportunities_with_validation==3
    assert result.overall.simulated_net_profit.mean==sum(expected,D(0))/D(4)
    assert [g.outcomes.validation_count for g in result.by_symbol]==[1,1,2]
    assert [g.outcomes.validation_count for g in result.horizons]==[1,2,1,0]


@pytest.mark.parametrize("field",["missing_snapshot","identity","symbol","spot_book","perp_book"])
def test_wrong_or_missing_saved_identity_is_read_failure(prepared,field):
    save(prepared)
    with closing(sqlite3.connect(prepared[1].config.database_path)) as connection, connection:
        row=connection.execute("SELECT details_json FROM opportunities WHERE symbol='SOLUSDT'").fetchone()
        details=json.loads(row[0])
        # Production records may use the lossless scenario-reference envelope.
        body=details.get("details",details)
        snapshot=body["snapshot"]
        if field=="missing_snapshot":
            del body["snapshot"]
        elif field=="identity":
            snapshot["research_instrument"]=None
        elif field=="symbol":
            snapshot["symbol"]="ETHUSDT"
        else:
            snapshot[field]["research_instrument"]={"symbol":"ETHUSDT","base_asset":"ETH"}
        connection.execute("UPDATE opportunities SET details_json=? WHERE symbol='SOLUSDT'",(exact_json(details),))
    with pytest.raises(AnalyticsReadError):
        summarize(prepared)


def test_lossless_compact_scenario_records_are_supported(prepared):
    from crypto_arbitrage_bot.models.opportunity import ScenarioRecord
    from crypto_arbitrage_bot.storage.scenario_codec import FORMAT
    record=prepared[2]["SOLUSDT"]
    details=json.loads(record.details_json)
    config=json.loads(record.config_json)
    scenario=ScenarioRecord("baseline","BASELINE","SOLUSDT","COMPLETE",NOW,NOW+timedelta(hours=1),
        record.net_profit,record.net_return,record.capital_occupied,record.estimated_funding,
        exact_json({"entry_snapshot":details["snapshot"],"config":config["estimate"]}))
    record=replace(record,id="compact",scenarios=(scenario,))
    prepared[1].save_opportunity(record)
    with closing(sqlite3.connect(prepared[1].config.database_path)) as connection:
        wrapped=json.loads(connection.execute("SELECT details_json FROM opportunities WHERE id='compact'").fetchone()[0])
        assert wrapped["storage_format"]==FORMAT
    SQLiteForwardValidationRepository(prepared[1].config).initialize()
    assert summarize(prepared).unvalidated_opportunities==4


def test_explicit_builtin_identity_also_must_match_saved_books(prepared):
    save(prepared)
    identities=(SOL,ResearchInstrument("BTCUSDT","BTC"),ResearchInstrument("ETHUSDT","ETH"))
    result=summarize(prepared,replace(CONFIG,research_instruments=identities))
    assert [g.key for g in result.by_symbol]==["BTCUSDT","ETHUSDT","SOLUSDT"]
    with closing(sqlite3.connect(prepared[1].config.database_path)) as connection, connection:
        connection.execute("UPDATE opportunities SET details_json='{}' WHERE symbol='BTCUSDT'")
    with pytest.raises(AnalyticsReadError):
        summarize(prepared,replace(CONFIG,research_instruments=identities))


@pytest.mark.parametrize("identities,error",[([SOL],TypeError),(("SOLUSDT",),TypeError),((None,),TypeError),
    ((SOL,SOL),ValueError)])
def test_research_grouping_config_requires_unique_typed_identity(identities,error):
    with pytest.raises(error):
        replace(CONFIG,research_instruments=identities)


def test_decimal_precision_and_none_are_not_float_or_zero():
    values=(D("1.0000000000000000000000000000000000000000000000000001"),
            D("1.0000000000000000000000000000000000000000000000000003"),None)
    rows=tuple(ValidationObservation(str(i),"op","SOLUSDT",Decision.REJECT,S.VALIDATED,
        NOW,timedelta(hours=8),value,value,None,None,None,D(0)) for i,value in enumerate(values))
    with localcontext() as context:
        context.prec=2
        context.traps[Inexact]=True
        result=summarize_forward_observations(frozenset({"op"}),rows,CONFIG)
    assert result.overall.simulated_net_profit.mean==D("1.0000000000000000000000000000000000000000000000000002")
    assert result.overall.simulated_net_profit.null_count==1
    with pytest.raises(ValueError,match="explicit research identity"):
        summarize_forward_observations(frozenset({"op"}),rows,replace(CONFIG,research_instruments=()))


def test_empty_database_and_missing_database_keep_original_semantics(tmp_path):
    db=SQLiteConfig(tmp_path/"empty.sqlite")
    with pytest.raises(AnalyticsReadError):
        summarize_forward_database(db,CONFIG)
    assert not db.database_path.exists()
    SQLiteOpportunityRepository(db).initialize()
    SQLiteForwardValidationRepository(db).initialize()
    assert summarize_forward_database(db,CONFIG).total_opportunities==0


def test_research_analytics_cannot_expand_execution_whitelist():
    from crypto_arbitrage_bot.execution.types import ExecutionMarket, IocLimitOrder
    assert set(SUPPORTED_SYMBOLS)=={"BTCUSDT","ETHUSDT"}
    assert "SOLUSDT" in CONFIG.symbols
    with pytest.raises(ValueError):
        IocLimitOrder(ExecutionMarket.SPOT,"SOLUSDT",Side.BUY,D(1),D(100),"analytics")


def test_candidate_analytics_connection_is_read_only(prepared,monkeypatch):
    save(prepared)
    original=sqlite3.connect
    def connect(database,*args,**kwargs):
        assert database.endswith("?mode=ro")
        connection=original(database,*args,**kwargs)
        with pytest.raises(sqlite3.OperationalError,match="readonly"):
            connection.execute("DELETE FROM opportunities")
        connection.rollback()
        return connection
    monkeypatch.setattr(sqlite3,"connect",connect)
    assert summarize(prepared).overall.validation_count==1
