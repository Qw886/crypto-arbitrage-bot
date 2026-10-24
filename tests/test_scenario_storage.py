"""Storage-only compression preserves complete scenario evidence and cash flows."""

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from decimal import Decimal

import pytest
import test_opportunity_storage as storage_fixtures
from pydantic import TypeAdapter
from test_offline_estimate import OPENED, assembly_for, books, config
from test_scenario import baseline, event, target
from test_scenario_decision import limits, scenarios

from crypto_arbitrage_bot.arbitrage.scenario import evaluate_scenario, evaluate_scenarios
from crypto_arbitrage_bot.arbitrage.scenario_decision import decide_scenarios
from crypto_arbitrage_bot.models.market import MarketSnapshot, OrderBookLevel
from crypto_arbitrage_bot.models.scenario import ScenarioDefinition, ScenarioType
from crypto_arbitrage_bot.storage.opportunity_records import build_opportunity, exact_json
from crypto_arbitrage_bot.storage.scenario_codec import FORMAT, compact_record, expand_record
from crypto_arbitrage_bot.storage.sqlite_repository import (
    RepositoryError,
    SQLiteOpportunityRepository,
)

D = Decimal
record = storage_fixtures.record
repository = storage_fixtures.repository


def raw_payloads(repository, opportunity_id):
    with closing(sqlite3.connect(repository.config.database_path)) as db:
        parent = db.execute("SELECT details_json FROM opportunities WHERE id=?", (opportunity_id,)).fetchone()[0]
        children = tuple(row[0] for row in db.execute(
            "SELECT details_json FROM opportunity_scenarios WHERE opportunity_id=? ORDER BY position",
            (opportunity_id,),
        ))
        return parent, children


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_exact_roundtrip_and_replay_all_scenario_math(repository, symbol):
    evidence = scenarios(symbol)
    original = build_opportunity(evidence[0].estimate, decide_scenarios(evidence, limits()), created_at=OPENED)
    repository.save_opportunity(original)
    loaded = SQLiteOpportunityRepository(repository.config).get_opportunity_by_id(original.id)
    assert loaded == original  # Includes every serialized Decimal, time, fill and assumption.
    assert repository.list_recent_opportunities() == (original,)
    for stored, before, expected in zip(loaded.scenarios, evidence, ("9.0144", "25.0272", "-2.9952"), strict=True):
        data = json.loads(stored.details_json)
        definition = TypeAdapter(ScenarioDefinition).validate_json(exact_json(data["definition"]))
        snapshot = TypeAdapter(MarketSnapshot).validate_json(exact_json(data["entry_snapshot"]))
        assert definition == before.definition and snapshot == before.estimate.assembly.snapshot
        replay = evaluate_scenario(replace(before.estimate.assembly, snapshot=snapshot), before.estimate.config, definition)
        assert replay == before  # All PnL, fees, funding, spreads, slippage and return unchanged.
        assert replay.net_profit == stored.net_profit == D(expected)


def test_physical_payload_has_one_shared_snapshot_and_one_exit_per_scenario(repository, record):
    repository.save_opportunity(record)
    parent_text, children = raw_payloads(repository, record.id)
    parent = json.loads(parent_text)
    assert parent["storage_format"] == FORMAT and parent["version"] == 1
    assert parent["details"]["snapshot"] == json.loads(record.details_json)["snapshot"]
    assert parent["shared"] == {"entry_mid_basis": "10"}
    assert "scenario" not in parent["details"]["estimate"]
    for text in children:
        wrapped = json.loads(text)
        data = wrapped["details"]
        assert set(wrapped["references"]) == {"entry_snapshot", "config", "entry_mid_basis", "estimate.scenario"}
        assert not {"entry_snapshot", "config", "entry_mid_basis"}.intersection(data)
        assert "scenario" not in data["estimate"]
        assert "exit_scenario" in data["definition"]
    with closing(sqlite3.connect(repository.config.database_path)) as db:
        sources = tuple(row[0] for row in db.execute(
            "SELECT details_json FROM opportunity_sources WHERE opportunity_id=? ORDER BY position", (record.id,),
        ))
    assert sources == tuple(source.details_json for source in record.source_timings)


def test_legacy_and_compact_records_coexist_without_migration(repository, record):
    repository.save_opportunity(record)
    # Exact pre-change persisted payloads, as produced by the old writer.
    with closing(sqlite3.connect(repository.config.database_path)) as db, db:
        db.execute("UPDATE opportunities SET details_json=? WHERE id=?", (record.details_json, record.id))
        for position, scenario in enumerate(record.scenarios):
            db.execute("UPDATE opportunity_scenarios SET details_json=? WHERE opportunity_id=? AND position=?",
                       (scenario.details_json, record.id, position))
    legacy_payloads = raw_payloads(repository, record.id)
    newer = replace(record, id="new-compact")
    repository.save_opportunity(newer)
    repository.initialize()
    assert repository.get_opportunity_by_id(record.id) == record
    assert repository.get_opportunity_by_id(newer.id) == newer
    assert set(item.id for item in repository.list_recent_opportunities()) == {record.id, newer.id}
    assert raw_payloads(repository, record.id) == legacy_payloads


@pytest.mark.parametrize("field,value", [
    ("config", {"different": "0.00001"}), ("entry_snapshot", None),
    ("entry_mid_basis", "10.0000"),
])
def test_unequal_evidence_is_never_replaced_with_shared_value(repository, record, field, value):
    data = json.loads(record.scenarios[1].details_json)
    data[field] = value
    different = replace(record.scenarios[1], details_json=exact_json(data))
    original = replace(record, scenarios=(record.scenarios[0], different, record.scenarios[2]))
    repository.save_opportunity(original)
    assert repository.get_opportunity_by_id(original.id) == original
    persisted = json.loads(raw_payloads(repository, original.id)[1][1])
    assert field not in persisted["references"] and persisted["details"][field] == value


def test_unequal_exit_assumptions_are_not_silently_overwritten(repository, record):
    data = json.loads(record.scenarios[0].details_json)
    data["estimate"]["scenario"]["description"] = "Different explicit evidence"
    original = replace(record, scenarios=(replace(record.scenarios[0], details_json=exact_json(data)), *record.scenarios[1:]))
    repository.save_opportunity(original)
    assert repository.get_opportunity_by_id(original.id) == original
    persisted = json.loads(raw_payloads(repository, original.id)[1][0])
    assert "estimate.scenario" not in persisted["references"]


@pytest.mark.parametrize("location", ["parent", "scenario", "config"])
def test_noncanonical_legacy_json_is_preserved_verbatim(repository, record, location):
    if location == "parent":
        original = replace(record, details_json=json.dumps(json.loads(record.details_json), indent=2))
    elif location == "config":
        original = replace(record, config_json=json.dumps(json.loads(record.config_json), indent=2))
    else:
        first = replace(record.scenarios[0], details_json=json.dumps(json.loads(record.scenarios[0].details_json), indent=2))
        original = replace(record, scenarios=(first, *record.scenarios[1:]))
    repository.save_opportunity(original)
    assert repository.get_opportunity_by_id(original.id) == original


@pytest.mark.parametrize("corruption", ["version", "missing_shared", "unknown_reference", "bad_position", "missing_parent"])
def test_corrupt_or_future_format_is_explicit_read_failure(repository, record, corruption):
    repository.save_opportunity(record)
    parent_text, children = raw_payloads(repository, record.id)
    parent, child = json.loads(parent_text), json.loads(children[0])
    if corruption == "version":
        parent["version"] = 999
    elif corruption == "missing_shared":
        parent["shared"].clear()
    elif corruption == "unknown_reference":
        child["references"].append("unknown")
    elif corruption == "bad_position":
        parent["primary_exit_position"] = -1
    else:
        parent = json.loads(record.details_json)
    with closing(sqlite3.connect(repository.config.database_path)) as db, db:
        db.execute("UPDATE opportunities SET details_json=? WHERE id=?", (exact_json(parent), record.id))
        db.execute("UPDATE opportunity_scenarios SET details_json=? WHERE opportunity_id=? AND position=0", (exact_json(child), record.id))
    with pytest.raises(RepositoryError):
        repository.get_opportunity_by_id(record.id)


def large_record():
    source = books()
    def deepen(book):
        return replace(book,
            bids=book.bids + tuple(OrderBookLevel(book.bids[-1].price - D(i) / 100, D("3.1234567890123456789")) for i in range(1, 101)),
            asks=book.asks + tuple(OrderBookLevel(book.asks[-1].price + D(i) / 100, D("3.1234567890123456789")) for i in range(1, 101)),
        )
    spot, perp = deepen(source[0]), deepen(source[2])
    source = (spot, source[1], perp, source[3])
    definitions = (baseline(source, (event(".1"),)), target(source, events=(event(".1"),)),
                   target(source, ScenarioType.ADVERSE, ".16", (event(".1"),)))
    evidence = evaluate_scenarios(assembly_for((spot, perp)), config(), definitions)
    return build_opportunity(evidence[0].estimate, decide_scenarios(evidence, limits()), created_at=OPENED, opportunity_id="large-fixture")


def payload_bytes(record):
    # All stored JSON including unchanged raw sources/config; excludes SQLite page overhead.
    return sum(len(text.encode("utf-8")) for text in (
        record.config_json, record.details_json, *(item.details_json for item in record.scenarios),
        *(item.details_json for item in record.source_timings),
    ))


def test_large_depth_fixture_storage_size_regression(repository):
    original = large_record()
    compact = compact_record(original)
    before, after = payload_bytes(original), payload_bytes(compact)
    assert after * 100 < before * 60  # At least 40% less, counting preserved sources too.
    assert expand_record(compact) == original
    repository.save_opportunity(original)
    parent, children = raw_payloads(repository, original.id)
    assert parent == compact.details_json
    assert children == tuple(item.details_json for item in compact.scenarios)
    assert repository.get_opportunity_by_id(original.id) == original
    print(f"JSON_BYTES before={before} after={after} reduction_pct={(D(before-after)*100/D(before)):.2f}")
