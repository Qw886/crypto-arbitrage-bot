"""Public archive seam: latest failure must not erase earlier native accounting."""

import json
import sqlite3
import sys
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from test_trade_accounting import AT
from test_trade_accounting import evidence_bot as evidence_bot

from crypto_arbitrage_bot.execution.reconciliation_repository import (
    AccountingPersistenceCode,
    ReconciliationRepository,
)
from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as GatewayError
from crypto_arbitrage_bot.execution.trade_reconciliation import ReconciliationStatus as Status
from crypto_arbitrage_bot.execution.trade_reconciliation import reconcile_closed_attempt


def test_failed_requery_keeps_latest_failure_and_original_verified_report(evidence_bot, monkeypatch):
    bot, plan, *_ = evidence_bot
    original = reconcile_closed_attempt(bot, plan.attempt_id, capital=Decimal("50"))
    repository = ReconciliationRepository(bot.journal)
    assert repository.save(original).success
    def unavailable(*args, **kwargs):
        raise GatewayError(ExecutionErrorKind.HTTP_REJECTED, "legacy order unavailable", api_code=-2013)
    monkeypatch.setattr(bot.spot, "get_order", unavailable)
    bot.clock = lambda: AT + timedelta(seconds=500)
    fresh = reconcile_closed_attempt(bot, plan.attempt_id, capital=Decimal("50"))
    assert repository.save(fresh).success
    assert repository.get_report(plan.attempt_id)["calculation"] is None
    history = repository.list_report_history(plan.attempt_id)
    assert len(history) == 2
    assert history[0]["report"]["status"] == "RECONCILED"
    assert Decimal(history[0]["report"]["calculation"]["net_profit"]) == Decimal("0.03993")
    assert history[1]["report"]["status"] == "INCOMPLETE"
    assert history[1]["report"]["calculation"] is None


def test_legacy_schema_migration_preserves_original_report_and_is_idempotent(evidence_bot):
    bot, plan, *_ = evidence_bot
    repository = ReconciliationRepository(bot.journal)
    result = reconcile_closed_attempt(bot, plan.attempt_id, capital=Decimal("50"))
    assert repository.save(result).success
    before = repository.get_report(plan.attempt_id)
    # External version-1 SQLite fixture: history table did not exist.
    with closing(sqlite3.connect(bot.config.journal_path)) as db, db:
        db.execute("DROP TABLE testnet_trade_reconciliation_history")
    repository.initialize()
    repository.initialize()
    assert repository.get_report(plan.attempt_id) == before
    history = repository.list_report_history(plan.attempt_id)
    assert len(history) == 1 and history[0]["origin"] == "LEGACY_IMPORT"
    assert history[0]["report"] == before
    assert history[0]["checked_at"] == result.checked_at.isoformat()


def test_legacy_history_read_does_not_initialize_or_write_database(evidence_bot):
    bot, plan, *_ = evidence_bot
    repository = ReconciliationRepository(bot.journal)
    result = reconcile_closed_attempt(bot, plan.attempt_id)
    assert repository.save(result).success
    with closing(sqlite3.connect(bot.config.journal_path)) as db, db:
        db.execute("DROP TABLE testnet_trade_reconciliation_history")
    before = bot.config.journal_path.read_bytes()
    history = repository.list_report_history(plan.attempt_id)
    assert len(history) == 1 and history[0]["origin"] == "LEGACY_UNMIGRATED"
    assert history[0]["archived_at"] is None
    assert bot.config.journal_path.read_bytes() == before


def test_identical_observation_is_idempotent_and_preserves_decimal_and_utc(evidence_bot):
    bot, plan, *_ = evidence_bot
    repository = ReconciliationRepository(bot.journal)
    result = reconcile_closed_attempt(bot, plan.attempt_id, capital=Decimal("50"))
    assert repository.save(result).success and repository.save(result).success
    history = repository.list_report_history(plan.attempt_id)
    assert len(history) == 1 and history[0]["origin"] == "RECONCILIATION"
    assert history[0]["report"] == repository.get_report(plan.attempt_id)
    assert history[0]["report"]["funding_total"] == "0.01"
    assert Decimal(history[0]["report"]["net_return"]) == Decimal("0.0007986")
    assert history[0]["checked_at"] == result.checked_at.isoformat()


def test_history_write_failure_rolls_back_latest_report_and_revision(evidence_bot):
    bot, plan, *_ = evidence_bot
    repository = ReconciliationRepository(bot.journal)
    result = reconcile_closed_attempt(bot, plan.attempt_id)
    assert repository.save(result).success
    before_report = repository.get_report(plan.attempt_id)
    before_history = repository.list_report_history(plan.attempt_id)
    before_pair = bot.journal.get(plan.attempt_id)
    with closing(sqlite3.connect(bot.config.journal_path)) as db, db:
        db.execute("CREATE TRIGGER fault BEFORE INSERT ON testnet_trade_reconciliation_history "
            "WHEN NEW.origin='RECONCILIATION' BEGIN SELECT RAISE(ABORT,'offline injected failure'); END")
    next_result = replace(result, checked_at=result.checked_at + timedelta(seconds=1),
                          status=Status.INCOMPLETE, calculation=None, net_return=None)
    assert repository.save(next_result).error_code is AccountingPersistenceCode.PERSISTENCE_FAILED
    assert repository.get_report(plan.attempt_id) == before_report
    assert repository.list_report_history(plan.attempt_id) == before_history
    assert bot.journal.get(plan.attempt_id) == before_pair


def test_history_cli_reads_existing_archive_without_credentials_or_database_writes(evidence_bot, monkeypatch, capsys, tmp_path):
    from crypto_arbitrage_bot.config.settings import load_settings
    from crypto_arbitrage_bot.execution.main import main
    bot, plan, *_ = evidence_bot
    repository = ReconciliationRepository(bot.journal)
    assert repository.save(reconcile_closed_attempt(bot, plan.attempt_id)).success
    original = load_settings(Path("docs/testnet-funding.toml"))
    text = Path("docs/testnet-funding.toml").read_text(encoding="utf-8")
    text = text.replace(original.testnet.journal_path.as_posix(), bot.config.journal_path.as_posix())
    config = tmp_path / "history.toml"
    config.write_text(text, encoding="utf-8")
    def no_network(*args, **kwargs):
        raise AssertionError("history is strictly offline")
    monkeypatch.setattr("socket.create_connection", no_network)
    monkeypatch.setattr(sys, "argv", ["run_testnet.py", "--config", str(config), "--accounting-history", "--attempt-id", plan.attempt_id])
    before = bot.config.journal_path.read_bytes()
    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scope"] == "LOCAL_ACCOUNTING_REPORT_HISTORY"
    assert payload["orders_submitted"] is False and payload["credentials_read"] is False
    assert len(payload["records"]) == 1 and payload["records"][0]["status"] == "RECONCILED"
    assert bot.config.journal_path.read_bytes() == before


def test_history_cli_requires_attempt_id_before_reading_credentials(monkeypatch, capsys):
    from crypto_arbitrage_bot.execution.main import main

    monkeypatch.setattr(sys, "argv", ["run_testnet.py", "--accounting-history"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--accounting-history requires --attempt-id" in capsys.readouterr().err


def test_history_cli_missing_record_returns_not_found_without_writing(evidence_bot, monkeypatch, capsys, tmp_path):
    from crypto_arbitrage_bot.config.settings import load_settings
    from crypto_arbitrage_bot.execution.main import main

    bot, *_ = evidence_bot
    settings = load_settings(Path("docs/testnet-funding.toml"))
    text = Path("docs/testnet-funding.toml").read_text(encoding="utf-8")
    text = text.replace(settings.testnet.journal_path.as_posix(), bot.config.journal_path.as_posix())
    config = tmp_path / "history.toml"
    config.write_text(text, encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("local history must not construct clients or load credentials")

    monkeypatch.setattr("crypto_arbitrage_bot.execution.main.execution_environment", forbidden)
    monkeypatch.setattr("crypto_arbitrage_bot.execution.main.create_testnet_bot", forbidden)
    monkeypatch.setattr(sys, "argv", ["run_testnet.py", "--config", str(config),
                                    "--accounting-history", "--attempt-id", "unknown-attempt"])
    before = bot.config.journal_path.read_bytes()
    assert main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "NOT_FOUND" and payload["records"] == []
    assert payload["credentials_read"] is False and payload["orders_submitted"] is False
    assert bot.config.journal_path.read_bytes() == before
