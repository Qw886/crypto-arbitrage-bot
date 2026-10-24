"""Real temporary SQLite reads, immutable output and safe CLI failure handling."""

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_forward_validation import OPENED, research_record, validate

from crypto_arbitrage_bot import pilot_outcome_report as cli
from crypto_arbitrage_bot.analytics.pilot_outcome import PILOT_HORIZONS, PILOT_SYMBOLS
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.storage.forward_validation_repository import (
    SQLiteForwardValidationRepository,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.sqlite_repository import SQLiteConfig, SQLiteOpportunityRepository
from crypto_arbitrage_bot.storage.validation_tasks import SQLiteValidationTaskRepository


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("report tests cannot contact an exchange"))
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("http.client.HTTPSConnection", forbidden)


def rename(value, symbol):
    if isinstance(value, dict):
        return {k: rename(v, symbol) for k, v in value.items()}
    if isinstance(value, list):
        return [rename(v, symbol) for v in value]
    return symbol if value == "BTCUSDT" else value


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    config = SQLiteConfig(tmp_path / "pilot.sqlite")
    parents = SQLiteOpportunityRepository(config)
    parents.initialize()
    tasks = SQLiteValidationTaskRepository(config)
    tasks.initialize()
    SQLiteForwardValidationRepository(config).initialize()
    captures = []
    records = []
    for symbol in PILOT_SYMBOLS:
        original = research_record()
        details = rename(json.loads(original.details_json), symbol)
        if symbol not in ("BTCUSDT", "ETHUSDT"):
            identity = {"symbol": symbol, "base_asset": symbol[:-4]}
            for target in (
                details["snapshot"],
                details["snapshot"]["spot_book"],
                details["snapshot"]["perp_book"],
            ):
                target["research_instrument"] = identity
        record = replace(
            original,
            id=symbol,
            symbol=symbol,
            config_version="candidate-multisettlement-observation-v1:next_settlement",
            details_json=exact_json(details),
        )
        parents.save_opportunity(record)
        tasks.create_tasks(record.id, PILOT_HORIZONS, now=OPENED)
        records.append(record)
        captures.append(
            {
                "symbol": symbol,
                "cost_result": {
                    "captures": [
                        {
                            "status": "SAVED",
                            "symbol": symbol,
                            "windows": [
                                {
                                    "opportunity_id": symbol,
                                    "persistence_status": "SAVED",
                                    "profile": "next_settlement",
                                    "entry_time": OPENED,
                                    "decision": record.decision,
                                    "net_profit": record.net_profit,
                                    "net_return": record.net_return,
                                }
                            ],
                        }
                    ]
                },
            }
        )
    config_path = Path("docs/research-multisettlement.toml").resolve()
    manifest = {
        "captures": captures,
        "config": str(config_path),
        "selection_committed_before_entries": True,
        "started_at": OPENED - timedelta(seconds=1),
        "finished_at": OPENED + timedelta(seconds=1),
    }
    evidence = tmp_path / "entries.json"
    evidence.write_text(exact_json(manifest), encoding="utf-8")
    job = SimpleNamespace(
        database=config.database_path,
        started=OPENED + timedelta(seconds=2),
        deadline=OPENED + timedelta(days=8),
        settings=load_settings(config_path),
        binding={"config_path": str(config_path), "config_sha256": "a" * 64},
    )
    monkeypatch.setattr(cli, "load_legacy_job", lambda directory: job)
    monkeypatch.setattr(cli, "utc_now", lambda: OPENED + timedelta(days=2))
    return SimpleNamespace(
        config=config,
        parents=parents,
        tasks=tasks,
        records=records,
        job=job,
        directory=tmp_path,
        evidence=evidence,
        manifest=manifest,
    )


def read(pilot):
    return cli.read_pilot_outcome(
        pilot.directory, pilot.evidence, evaluated_at=OPENED + timedelta(days=2)
    )


def test_complete_original_snapshot_retains_ids_and_never_changes_database(pilot):
    before = pilot.config.database_path.read_bytes()
    result = read(pilot)
    assert pilot.config.database_path.read_bytes() == before
    assert result["task_status_counts"] == {"PENDING": 12}
    assert len(result["rows"]) == 12 and result["original_entry_count"] == 4
    assert set(result["original_record_sha256"]) == set(PILOT_SYMBOLS)
    assert (
        result["entry_evidence_sha256"] == hashlib.sha256(pilot.evidence.read_bytes()).hexdigest()
    )
    assert not result["orders_submitted"] and not result["credentials_read"]
    assert result["report_source_sha256"]["validation_evidence.py"] == hashlib.sha256(
        (Path(cli.__file__).parent / "analytics/validation_evidence.py").read_bytes()
    ).hexdigest()


def test_native_saved_validation_projects_all_amounts_without_recalculation(pilot):
    task = pilot.tasks.list_for_opportunity("BTCUSDT")[0]
    claim = pilot.tasks.mark_running(task.id, now=task.due_at, lease_seconds=300)
    from test_forward_validation import later_snapshot

    result = validate(pilot.records[0], later_snapshot(at=task.due_at), at=task.due_at)
    repository = SQLiteForwardValidationRepository(pilot.config)
    saved = repository.save_forward_validation(
        result, validation_id=task.validation_id, created_at=task.due_at
    )
    assert saved.saved_id and pilot.tasks.finish(claim, now=task.due_at)
    report = read(pilot)
    row = report["rows"][0]
    assert row["usable"] and report["usable_validations"] == 1
    assert row["validated_simulated_net_profit"] == result.validated_simulated_net_profit
    assert row["spread_costs"] == result.decomposition.spread_costs
    assert row["funding_event_count"] == 0


@pytest.mark.parametrize(
    "case", ["missing_parent", "extra_parent", "missing_table", "schema", "damaged_task"]
)
def test_missing_or_corrupt_sqlite_is_unavailable_instead_of_empty_success(pilot, case):
    with closing(sqlite3.connect(pilot.config.database_path)) as db, db:
        if case == "missing_parent":
            db.execute("DELETE FROM opportunities WHERE id='TSTUSDT'")
        elif case == "extra_parent":
            pilot.parents.save_opportunity(replace(pilot.records[0], id="unplanned"))
        elif case == "missing_table":
            db.execute("DROP TABLE forward_validations")
        elif case == "schema":
            db.execute("PRAGMA user_version=999")
        else:
            db.execute("UPDATE validation_tasks SET due_at='invalid'")
    with pytest.raises((ValueError, sqlite3.Error)):
        read(pilot)


@pytest.mark.parametrize(
    "case",
    [
        "missing_capture",
        "duplicate_symbol",
        "changed_id",
        "changed_profit",
        "changed_profile",
        "changed_time",
        "changed_config",
        "selection",
    ],
)
def test_archive_scope_cannot_select_a_replacement_or_best_entry(pilot, case):
    manifest = json.loads(pilot.evidence.read_text(encoding="utf-8"))
    first = manifest["captures"][0]
    window = first["cost_result"]["captures"][0]["windows"][0]
    if case == "missing_capture":
        manifest["captures"].pop()
    elif case == "duplicate_symbol":
        manifest["captures"][1] = first
    elif case == "changed_id":
        window["opportunity_id"] = "different"
    elif case == "changed_profit":
        window["net_profit"] = "999"
    elif case == "changed_profile":
        window["profile"] = "best_horizon"
    elif case == "changed_time":
        window["entry_time"] = (OPENED - timedelta(hours=1)).isoformat()
    elif case == "changed_config":
        manifest["config"] = "different.toml"
    else:
        manifest["selection_committed_before_entries"] = False
    pilot.evidence.write_text(exact_json(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        read(pilot)


def test_duplicate_json_fields_are_rejected(pilot):
    pilot.evidence.write_text('{"captures":[],"captures":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read(pilot)


def test_missing_database_is_not_created(pilot):
    pilot.job.database = pilot.directory / "missing.sqlite"
    with pytest.raises(sqlite3.Error):
        read(pilot)
    assert not pilot.job.database.exists()


def arguments(pilot, output):
    return [
        "--job-dir",
        str(pilot.directory),
        "--entry-evidence",
        str(pilot.evidence),
        "--output",
        str(output),
    ]


def test_cli_writes_once_and_rejects_overwrite(pilot, capsys):
    output = pilot.directory / "report.json"
    assert cli.main(arguments(pilot, output)) == 0
    before = output.read_bytes()
    assert cli.main(arguments(pilot, output)) == 2
    assert output.read_bytes() == before
    messages = capsys.readouterr()
    assert json.loads(messages.out)["conclusion"] == "PENDING_FINAL_OBSERVATIONS"
    assert json.loads(messages.err)["status"] == "UNAVAILABLE"


def test_cli_failure_has_no_exception_body_and_creates_no_report(pilot, monkeypatch, capsys):
    monkeypatch.setattr(cli, "read_pilot_outcome", Mock(side_effect=ValueError("PRIVATE_BODY")))
    output = pilot.directory / "failed.json"
    assert cli.main(arguments(pilot, output)) == 2
    assert not output.exists()
    assert "PRIVATE_BODY" not in capsys.readouterr().err


def test_unmodified_original_binding_is_required(tmp_path, monkeypatch):
    monkeypatch.undo()
    with pytest.raises(cli.ControlError):
        cli.read_pilot_outcome(tmp_path, tmp_path / "missing.json", evaluated_at=OPENED)


@pytest.fixture
def native_validation(pilot):
    from test_forward_validation import later_snapshot

    task = pilot.tasks.list_for_opportunity("BTCUSDT")[0]
    claim = pilot.tasks.mark_running(task.id, now=task.due_at, lease_seconds=300)
    result = validate(pilot.records[0], later_snapshot(at=task.due_at), at=task.due_at)
    saved = SQLiteForwardValidationRepository(pilot.config).save_forward_validation(
        result, validation_id=task.validation_id, created_at=task.due_at
    )
    assert saved.saved_id and pilot.tasks.finish(claim, now=task.due_at)
    return saved.record


@pytest.mark.parametrize(
    "field",
    [
        "capital_occupied",
        "original_entry_basis",
        "spot_entry_fee",
        "perp_entry_fee",
        "other_costs",
        "original_estimated_net_profit",
        "original_estimated_net_return",
    ],
)
def test_self_consistent_validation_cannot_replace_original_entry_amounts(
    pilot, native_validation, field
):
    from decimal import Decimal

    from crypto_arbitrage_bot.storage.forward_validation_repository import COLUMNS, _encode

    value = replace(native_validation, **{field: Decimal("999")})
    details = json.loads(value.details_json)
    if field.endswith("_fee"):
        details["fees"][field[:-4]] = "999"
    elif field == "other_costs":
        details["calculation"][field] = "999"
    else:
        details[field] = "999"
        if field == "original_entry_basis":
            details["calculation"]["entry_basis"] = "999"
    value = replace(value, details_json=exact_json(details))
    with closing(sqlite3.connect(pilot.config.database_path)) as db, db:
        db.execute(
            "UPDATE forward_validations SET "
            + ",".join(f"{name}=?" for name in COLUMNS)
            + " WHERE id=?",
            (*_encode(value), value.id),
        )
    row = read(pilot)["rows"][0]
    assert not row["usable"] and "ORIGINAL_ENTRY_MISMATCH" in row["reason_codes"]
    assert row["validated_simulated_net_profit"] is None


@pytest.mark.parametrize("case", ["invalid_quality", "missing_quality", "duplicate_json_field"])
def test_details_with_invalid_quality_or_ambiguous_json_are_unusable(
    pilot, native_validation, case
):
    details = json.loads(native_validation.details_json)
    if case == "invalid_quality":
        details["exit_data_quality"]["issues"] = [{"code": "STALE_PERP_DATA"}]
    elif case == "missing_quality":
        del details["exit_data_quality"]
    body = exact_json(details)
    if case == "duplicate_json_field":
        body = '{"status":"INVALID",' + body[1:]
    with closing(sqlite3.connect(pilot.config.database_path)) as db, db:
        db.execute(
            "UPDATE forward_validations SET details_json=? WHERE id=?", (body, native_validation.id)
        )
    assert "INVALID_STORED_DETAILS" in read(pilot)["rows"][0]["reason_codes"]


@pytest.mark.parametrize(
    "case", ["saved_before_exit", "task_finished_before_save", "task_created_after_update"]
)
def test_durable_timing_must_follow_exit_then_save_then_task_completion(
    pilot, native_validation, case
):
    at = native_validation.validation_time - timedelta(seconds=1)
    with closing(sqlite3.connect(pilot.config.database_path)) as db, db:
        if case == "saved_before_exit":
            db.execute(
                "UPDATE forward_validations SET created_at=? WHERE id=?",
                (at.isoformat(), native_validation.id),
            )
        elif case == "task_finished_before_save":
            db.execute(
                "UPDATE validation_tasks SET updated_at=? WHERE validation_id=?",
                (at.isoformat(), native_validation.id),
            )
        else:
            db.execute(
                "UPDATE validation_tasks SET created_at=? WHERE validation_id=?",
                (
                    (native_validation.validation_time + timedelta(seconds=1)).isoformat(),
                    native_validation.id,
                ),
            )
    row = read(pilot)["rows"][0]
    assert not row["usable"] and "INVALID_TIMING" in row["reason_codes"]
