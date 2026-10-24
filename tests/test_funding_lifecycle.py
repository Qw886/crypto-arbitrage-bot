"""Offline forecast, scheduled execution, restart and actual funding accounting."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest
from admission_fixture import assume_offline_admission

from crypto_arbitrage_bot.arbitrage.basis import mid_basis, mid_price
from crypto_arbitrage_bot.arbitrage.funding import accrue_short_funding
from crypto_arbitrage_bot.arbitrage.funding_projection import (
    project_next_funding,
    projected_scenario_definitions,
)
from crypto_arbitrage_bot.config.research import FundingProjectionSettings, ScenarioSettings
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.demo import DemoResponse, demo_bot
from crypto_arbitrage_bot.execution.funding_ledger import reconcile_funding_ledger
from crypto_arbitrage_bot.execution.main import print_cycle
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState as State
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot
from crypto_arbitrage_bot.execution.types import FundingPayment

D = Decimal


@pytest.fixture
def live_clock_bot(tmp_path, monkeypatch):
    # This fixture tests original funding/execution wiring, not qualification.
    assume_offline_admission(monkeypatch)
    bot, transport = demo_bot(load_settings(Path("docs/testnet-funding.toml")), tmp_path)
    now = datetime.now(UTC)
    state = {"now": now.replace(microsecond=now.microsecond // 1000 * 1000), "rate": "0.1"}
    state["settlement"] = state["now"] + timedelta(seconds=300)
    def clock():
        return state["now"]
    for client in (bot.spot, bot.perp):
        client._clock_ms = lambda: (state["now"] - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(milliseconds=1)
    transport.clock = bot.clock = bot.collector.clock = bot.research_runner.clock = clock
    original = transport.open
    def open_request(request, **kwargs):
        response = original(request, **kwargs)
        if "/premiumIndex?" in request.full_url:
            payload = json.loads(response.body)
            payload["lastFundingRate"] = str(D(state["rate"]))
            next_settlement = state["settlement"]
            while next_settlement <= state["now"]:
                next_settlement += timedelta(hours=8)
            payload["nextFundingTime"] = (next_settlement - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(milliseconds=1)
            return DemoResponse(payload)
        return response
    monkeypatch.setattr(transport, "open", open_request)
    return bot, transport, state


@pytest.mark.parametrize("rate,expected,stress", [(".1", ".05", "0"), ("0", "0", "0"), ("-.1", "-.1", "-.1")])
def test_single_explicit_projected_event_and_signs(live_clock_bot, rate, expected, stress):
    bot, _, state = live_clock_bot
    state["rate"] = rate
    snapshot = bot.collector.collect("ETHUSDT").snapshot
    window = project_next_funding(snapshot, state["now"], bot.research.scenarios["ETHUSDT"].funding_projection)
    assert len(window.baseline_events) == len(window.stress_events) == 1
    assert window.baseline_events[0].funding_rate == D(expected)
    assert window.stress_events[0].funding_rate == D(stress)
    assert window.baseline_events[0].event_id.startswith("FORECAST_ONLY:")
    assert window.closed_at == state["settlement"] + timedelta(seconds=60)
    assert "NOT accrued" in window.description
    assert accrue_short_funding(D(".01"), state["now"], state["settlement"] - timedelta(seconds=1),
                                window.baseline_events, "ETHUSDT") == 0
    assert snapshot.funding_rate == D(rate)  # No mutation/haircut of raw data.


@pytest.mark.parametrize("seconds", [-1, 0, 86401])
def test_stale_or_unbounded_settlement_is_not_projected(live_clock_bot, seconds):
    bot, _, state = live_clock_bot
    snapshot = replace(bot.collector.collect("ETHUSDT").snapshot,
                       next_funding_time=state["now"] + timedelta(seconds=seconds))
    with pytest.raises(ValueError, match="OUTSIDE_PROJECTION_WINDOW"):
        project_next_funding(snapshot, state["now"], bot.research.scenarios["ETHUSDT"].funding_projection)


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_funding_scene_is_persisted_with_actual_future_exit(live_clock_bot, symbol):
    bot, transport, state = live_clock_bot
    cycle = bot.research_runner._run_symbol(symbol)
    assert cycle.status.value == "SAVED" and cycle.decision == "PASS"
    record = bot.repository.get_opportunity_by_id(cycle.opportunity_id)
    assert record.estimated_funding > 0 and len(record.scenarios) == 3
    assert all(s.closed_at == state["settlement"] + timedelta(seconds=60) for s in record.scenarios)
    assert next(s for s in record.scenarios if s.scenario_type == "ADVERSE").funding_total == 0
    assert all("FORECAST_ONLY" in s.details_json for s in record.scenarios)
    assert record.next_funding_time == state["settlement"].replace(microsecond=state["settlement"].microsecond // 1000 * 1000)
    assert transport.orders == {}  # Public research cannot submit.


def test_stress_basis_really_widens_entry_basis(live_clock_bot):
    bot, _, state = live_clock_bot
    snapshot = bot.collector.collect("BTCUSDT").snapshot
    definitions = projected_scenario_definitions(snapshot, state["now"], bot.research.scenarios["BTCUSDT"])
    exit_books = definitions[2].exit_scenario
    spot_mid = mid_price(snapshot.spot_book.best_bid, snapshot.spot_book.best_ask)
    entry_basis = mid_basis(spot_mid, mid_price(snapshot.perp_book.best_bid, snapshot.perp_book.best_ask))
    exit_basis = mid_basis(mid_price(exit_books.spot_exit_book.best_bid, exit_books.spot_exit_book.best_ask),
                           mid_price(exit_books.perp_exit_book.best_bid, exit_books.perp_exit_book.best_ask))
    assert exit_basis == entry_basis + spot_mid * D(".01")
    assert exit_basis > entry_basis


def test_relative_stress_preserves_quote_basis_when_spot_exit_mid_changes(live_clock_bot):
    bot, _, state = live_clock_bot
    snapshot = bot.collector.collect("BTCUSDT").snapshot
    settings = bot.research.scenarios["BTCUSDT"].model_copy(update={
        "spot_exit_mid_mode": "explicit", "spot_exit_mid": D("51000"),
    })
    stress = projected_scenario_definitions(snapshot, state["now"], settings)[2]
    books = stress.exit_scenario
    # Entry midpoint 49999.5; entry basis 10; widening .01 * 49999.5 = 499.995.
    assert mid_price(books.spot_exit_book.best_bid, books.spot_exit_book.best_ask) == D("51000")
    assert mid_price(books.perp_exit_book.best_bid, books.perp_exit_book.best_ask) == D("51509.995")
    assert "USDT/base" in books.description


@pytest.mark.parametrize("rate", ["0", "-.1", ".0001"])
def test_projection_does_not_force_nonprofitable_entry(live_clock_bot, rate):
    bot, transport, state = live_clock_bot
    state["rate"] = rate
    cycles = bot.run_once()
    assert all(c.status == "NO_ENTRY" and c.reason == "REJECT" for c in cycles)
    assert transport.orders == {}
    assert all(c.estimated_funding is not None for c in cycles)


def test_automatic_pass_opens_and_restart_respects_funding_deadline(live_clock_bot):
    bot, transport, state = live_clock_bot
    opened = bot.run_once()
    assert [c.status for c in opened] == ["HEDGED", "HEDGED"] and len(transport.orders) == 4
    plans = bot.journal.plans()
    assert all(plan.scheduled_exit_at is not None and plan.opportunity_id and plan.projection_json for plan, _ in plans)
    assert all(plan.spot_entry.submit_before == plan.perp_entry.submit_before ==
               plan.funding_settlement_at - timedelta(seconds=60) for plan, _ in plans)
    assert all(plan.spot_exit.submit_before is None and plan.perp_exit.submit_before is None for plan, _ in plans)
    deadlines = [plan.scheduled_exit_at for plan, _ in plans]
    state["now"] += timedelta(seconds=65)
    bot._managed_exits()
    assert len(transport.orders) == 4  # Must not use legacy 60-second timeout.
    restarted = Bot(bot.settings, bot.spot, bot.perp, clock=lambda: state["now"])
    assert [p.scheduled_exit_at for p, _ in restarted.journal.plans()] == deadlines
    restarted._managed_exits()
    assert len(transport.orders) == 4
    state["now"] = max(deadlines) + timedelta(seconds=1)
    restarted.collector.clock = transport.clock
    restarted._managed_exits()
    assert all(restarted.journal.get(p.attempt_id).state is State.CLOSED for p, _ in plans)
    assert len(transport.orders) == 8
    restarted._managed_exits()
    assert len(transport.orders) == 8


def test_expiry_between_legs_flattens_spot_without_opening_late_perp(live_clock_bot, monkeypatch):
    bot, transport, state = live_clock_bot
    cycle = bot.research_runner._run_symbol("ETHUSDT")
    record = bot.repository.get_opportunity_by_id(cycle.opportunity_id)
    plan = bot._plan("ETHUSDT", record.matched_quantity, "expiry-between-legs", opportunity=record)
    original = transport.open
    def slow_spot(request, **kwargs):
        response = original(request, **kwargs)
        if request.get_method() == "POST" and "testnet.binance.vision" in request.full_url:
            state["now"] = plan.perp_entry.submit_before + timedelta(seconds=1)
        return response
    monkeypatch.setattr(transport, "open", slow_spot)
    result = bot.coordinator.open_pair(plan)
    assert result.state is State.FLATTENED
    assert transport.positions["ETHUSDT"] == transport.base["ETHUSDT"] == 0
    assert not any(method == "POST" and "demo-fapi" in url for method, url in transport.requests)
def test_fresh_rate_drop_blocks_old_pass_record(live_clock_bot):
    bot, transport, state = live_clock_bot
    cycle = bot.research_runner._run_symbol("ETHUSDT")
    record = bot.repository.get_opportunity_by_id(cycle.opportunity_id)
    assert record.decision == "PASS"
    state["rate"] = "0"
    with pytest.raises(ExecutionGuardError, match="FRESH_FUNDING_SCENARIOS_NOT_PASS"):
        bot._plan("ETHUSDT", record.matched_quantity, "fresh-rate-drop", opportunity=record)
    assert transport.orders == {}


@pytest.mark.parametrize("lead", [10, 901])
def test_positive_pass_outside_entry_window_cannot_place_order(live_clock_bot, lead):
    bot, transport, state = live_clock_bot
    state["settlement"] = state["now"] + timedelta(seconds=lead)
    cycle = bot.research_runner._run_symbol("ETHUSDT")
    record = bot.repository.get_opportunity_by_id(cycle.opportunity_id)
    assert record.decision == "PASS"
    with pytest.raises(ExecutionGuardError, match="OUTSIDE_FUNDING_ENTRY_WINDOW"):
        bot._plan("ETHUSDT", record.matched_quantity, "outside-window", opportunity=record)
    assert transport.orders == {}


def test_changed_next_settlement_blocks_old_opportunity(live_clock_bot):
    bot, transport, state = live_clock_bot
    cycle = bot.research_runner._run_symbol("ETHUSDT")
    record = bot.repository.get_opportunity_by_id(cycle.opportunity_id)
    state["settlement"] += timedelta(hours=8)
    with pytest.raises(ExecutionGuardError, match="FUNDING_SETTLEMENT_CHANGED"):
        bot._plan("ETHUSDT", record.matched_quantity, "changed-next", opportunity=record)
    assert transport.orders == {}


def test_status_display_explains_forecast_and_reject(live_clock_bot, capsys):
    bot, _, state = live_clock_bot
    state["rate"] = "0"
    bot.status_callback = print_cycle
    bot.run_once()
    output = capsys.readouterr().out
    assert "预计Funding=" in output and "预计净收益=" in output and "下次结算(UTC)=" in output
    assert "NO_ENTRY" in output and "NON_POSITIVE_NET_PROFIT" in output


def closed_funding_attempt(fixture):
    bot, transport, state = fixture
    opened = bot.run_once()
    plan = next(p for p, _ in bot.journal.plans() if p.symbol == "ETHUSDT")
    state["now"] = plan.scheduled_exit_at + timedelta(seconds=1)
    bot._managed_exits()
    # Simulate the wall-clock journal timestamp after a real holding window.
    with bot.journal._connect() as db:
        db.execute("UPDATE testnet_pair_attempts SET updated_at=? WHERE state='CLOSED'", (state["now"].isoformat(),))
    return bot, transport, state, plan, opened


@pytest.mark.parametrize("amount", [".12345678901234567890123456789", "0", "-.12345678901234567890123456789"])
def test_actual_ledger_round_trip_separate_from_forecast(live_clock_bot, monkeypatch, amount):
    bot, transport, state, plan, opened = closed_funding_attempt(live_clock_bot)
    record = bot.repository.get_opportunity_by_id(plan.opportunity_id)
    def payments(symbol, start, end):
        return (FundingPayment("account-" + symbol, symbol, plan.funding_settlement_at, D(amount)),)
    query = Mock(side_effect=payments)
    monkeypatch.setattr(bot.perp, "get_funding_payments", query)
    reconcile_funding_ledger(bot)
    with bot.journal._connect() as db:
        row = db.execute("SELECT * FROM testnet_funding_checks WHERE attempt_id=?", (plan.attempt_id,)).fetchone()
        assert row["status"] == "VERIFIED" and D(row["actual_funding"]) == D(amount)
        assert row["attempts"] == 1
    before = query.call_count
    reconcile_funding_ledger(bot)
    assert query.call_count == before
    assert bot.repository.get_opportunity_by_id(record.id) == record
    assert len(transport.orders) == 8


def test_no_payment_stays_null_retry_is_bounded_and_recovers_after_restart(live_clock_bot, monkeypatch):
    bot, _, state, plan, _ = closed_funding_attempt(live_clock_bot)
    query = Mock(return_value=())
    monkeypatch.setattr(bot.perp, "get_funding_payments", query)
    reconcile_funding_ledger(bot)
    first = query.call_count
    reconcile_funding_ledger(bot)
    assert query.call_count == first  # Not every 5s and no busy loop.
    restarted = Bot(bot.settings, bot.spot, bot.perp, clock=lambda: state["now"])
    for _ in range(7):
        state["now"] += timedelta(seconds=60)
        reconcile_funding_ledger(restarted)
    with bot.journal._connect() as db:
        row = db.execute("SELECT * FROM testnet_funding_checks WHERE attempt_id=?", (plan.attempt_id,)).fetchone()
        assert row["status"] == "UNAVAILABLE" and row["actual_funding"] is None
        assert row["attempts"] == 5
    assert query.call_count == 10  # Two independent symbols, five checks each.


def test_funding_query_failure_does_not_change_closed_execution(live_clock_bot, monkeypatch):
    bot, _, _, plan, _ = closed_funding_attempt(live_clock_bot)
    before = bot.journal.get(plan.attempt_id)
    monkeypatch.setattr(bot.perp, "get_funding_payments", Mock(side_effect=TimeoutError))
    reconcile_funding_ledger(bot)
    with bot.journal._connect() as db:
        row = db.execute("SELECT * FROM testnet_funding_checks WHERE attempt_id=?", (plan.attempt_id,)).fetchone()
        assert row["status"] == "PENDING" and row["actual_funding"] is None and row["failure_code"] == "TimeoutError"
    assert bot.journal.get(plan.attempt_id) == before


def test_one_ledger_failure_does_not_block_other_symbol(live_clock_bot, monkeypatch):
    bot, _, _, plan, _ = closed_funding_attempt(live_clock_bot)
    def payments(symbol, start, end):
        if symbol == "BTCUSDT":
            raise TimeoutError
        return (FundingPayment("eth-ledger-id", symbol, plan.funding_settlement_at, D(".01")),)
    monkeypatch.setattr(bot.perp, "get_funding_payments", payments)
    reconcile_funding_ledger(bot)
    with bot.journal._connect() as db:
        row = db.execute("SELECT status,actual_funding FROM testnet_funding_checks WHERE attempt_id=?", (plan.attempt_id,)).fetchone()
        assert row["status"] == "VERIFIED" and D(row["actual_funding"]) == D(".01")


def test_payment_id_cannot_be_attributed_to_two_attempts(live_clock_bot, monkeypatch):
    bot, _, _, plan, _ = closed_funding_attempt(live_clock_bot)
    monkeypatch.setattr(bot.perp, "get_funding_payments", lambda symbol, start, end: (
        FundingPayment("same-transaction", symbol, plan.funding_settlement_at, D(".01")),
    ))
    reconcile_funding_ledger(bot)
    with bot.journal._connect() as db:
        rows = db.execute("SELECT status,actual_funding FROM testnet_funding_checks").fetchall()
        assert sorted(r["status"] for r in rows) == ["PENDING", "VERIFIED"]
        assert sum(r["actual_funding"] is not None for r in rows) == 1
        assert db.execute("SELECT COUNT(*) FROM testnet_funding_payments").fetchone()[0] == 1


def test_config_preserves_fees_thresholds_and_old_journal():
    old, new = (load_settings(Path("docs") / filename) for filename in ("testnet-bot.toml", "testnet-funding.toml"))
    assert old.fees == new.fees and old.research.decision == new.research.decision
    assert old.testnet.journal_path == new.testnet.journal_path
    assert all(s.funding_projection is None for s in old.research.scenarios.values())


@pytest.mark.parametrize("field,value", [("positive_rate_multiplier", "NaN"), ("positive_rate_multiplier", "1.1"),
    ("positive_rate_multiplier", 0.5), ("entry_window_seconds", 60), ("max_wait_seconds", 86401)])
def test_invalid_projection_config(field, value):
    config = load_settings(Path("docs/testnet-funding.toml")).research.scenarios["ETHUSDT"].funding_projection.model_dump()
    config[field] = value
    with pytest.raises(ValueError):
        FundingProjectionSettings.model_validate(config)


def test_legacy_explicit_events_cannot_mix_with_projection(live_clock_bot):
    bot, _, state = live_clock_bot
    config = bot.research.scenarios["ETHUSDT"].model_dump()
    event = project_next_funding(bot.collector.collect("ETHUSDT").snapshot, state["now"],
                                bot.research.scenarios["ETHUSDT"].funding_projection).baseline_events[0]
    config["funding_events"] = (event,)
    with pytest.raises(ValueError, match="cannot be mixed"):
        ScenarioSettings.model_validate(config)
