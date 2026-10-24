"""Offline integration through real client parsing, research and durable execution."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from threading import Event
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest
from admission_fixture import assume_offline_admission

from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.demo import DemoResponse, demo_bot
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState as State
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError, process_lock
from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind as ErrorKind
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as GatewayError
from crypto_arbitrage_bot.execution.types import ExecutionMarket, OrderState
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.models.calculation import TradeLeg


@pytest.fixture
def bot(tmp_path):
    settings = load_settings(Path("docs/testnet-bot.toml"))
    instance, transport = demo_bot(settings, tmp_path)
    return instance, transport


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_real_parsers_research_save_and_mock_round_trip(bot, symbol):
    instance, transport = bot
    instance.settings = instance.settings.model_copy(update={"symbols": (symbol,)})
    instance.research_runner.settings = instance.settings
    cycles = instance.run_once()
    assert len(cycles) == 1 and cycles[0].status == "NO_ENTRY" and cycles[0].reason == "REJECT"
    assert transport.orders == {}  # Positive indicative funding is not an order signal.
    record = instance.repository.get_opportunity_by_id(cycles[0].opportunity_id)
    assert record.symbol == symbol and len(record.scenarios) == 3
    assert "actual_url" in record.source_timings[0].details_json
    assert "testnet.binance.vision" in record.source_timings[0].details_json
    quantity = instance.research.sizing[symbol].target_quantity
    result = instance.round_trip(symbol, quantity)
    assert result.state is State.CLOSED
    assert len(transport.orders) == 4
    assert transport.base[symbol] == transport.positions[symbol] == Decimal("0")
    assert instance.journal.get(result.attempt_id) == result
    assert all(url.startswith(("https://testnet.binance.vision/", "https://demo-fapi.binance.com/"))
               for _, url in transport.requests)


@pytest.mark.parametrize("market", [ExecutionMarket.SPOT, ExecutionMarket.USDTM])
def test_signed_account_state_uses_testnet_get_only(bot, market):
    instance, transport = bot
    client = instance.spot if market is ExecutionMarket.SPOT else instance.perp
    state = client.get_account_state("BTCUSDT")
    assert state.quote_available == Decimal("10000") and state.can_trade
    assert state.one_way and state.single_asset and state.leverage == Decimal("1")
    assert all(method == "GET" for method, _ in transport.requests)
    assert any("signature=" in url and "/account?" in url for _, url in transport.requests)


@pytest.mark.parametrize("field,value", [("can_trade", False), ("open_order_count", 1),
                                         ("one_way", False), ("single_asset", False),
                                         ("leverage", Decimal("2")), ("position_quantity", Decimal("1")),
                                         ("quote_available", Decimal("0"))])
def test_bad_futures_account_blocks_all_entry_orders(bot, monkeypatch, field, value):
    instance, transport = bot
    state = instance.perp.get_account_state("BTCUSDT")
    monkeypatch.setattr(instance.perp, "get_account_state", lambda _: replace(state, **{field: value}))
    if field == "leverage":
        monkeypatch.setattr(instance.perp, "ensure_one_x", Mock(side_effect=GatewayError(
            ErrorKind.ORDER_GUARD_REJECTED, "1x preparation not confirmed")))
    with pytest.raises(ExecutionGuardError):
        instance.round_trip("BTCUSDT", Decimal("0.001"))
    assert transport.orders == {}


def test_insufficient_spot_balance_blocks_all_orders(bot, monkeypatch):
    instance, transport = bot
    state = instance.spot.get_account_state("BTCUSDT")
    monkeypatch.setattr(instance.spot, "get_account_state", lambda _: replace(state, quote_available=Decimal("1")))
    with pytest.raises(ExecutionGuardError, match="BALANCE"):
        instance.round_trip("BTCUSDT", Decimal("0.001"))
    assert transport.orders == {}


@pytest.mark.parametrize("quantity", [Decimal("0.00001"), Decimal("1")])
def test_live_exchange_filters_and_notional_caps_precede_orders(bot, quantity):
    instance, transport = bot
    with pytest.raises(GatewayError) as error:
        instance.round_trip("BTCUSDT", quantity)
    assert error.value.kind is ErrorKind.ORDER_GUARD_REJECTED
    assert transport.orders == {}


@pytest.mark.parametrize("action", ["depth", "premium", "interest"])
def test_market_failure_creates_incomplete_snapshot_not_order(bot, monkeypatch, action):
    instance, transport = bot
    original = instance.perp.public_payload
    def failed(request_action, symbol, **kwargs):
        if request_action == action:
            raise GatewayError(ErrorKind.NETWORK_ERROR, "mock failure")
        return original(request_action, symbol, **kwargs)
    monkeypatch.setattr(instance.perp, "public_payload", failed)
    assembly = instance.collector.collect("BTCUSDT")
    assert not assembly.complete
    assert not validate_market_data(assembly, instance.research.data_quality, evaluated_at=instance.clock()).valid
    with pytest.raises(ExecutionGuardError):
        instance.round_trip("BTCUSDT", Decimal("0.001"))
    assert transport.orders == {}


def test_stale_during_preflight_blocks_entry(bot, monkeypatch):
    instance, transport = bot
    assembly = instance.collector.collect("BTCUSDT")
    now = instance.clock()
    monkeypatch.setattr(instance.collector, "collect", lambda _: assembly)
    clocks = iter((now, now + timedelta(seconds=60)))
    instance.clock = lambda: next(clocks)
    with pytest.raises(ExecutionGuardError, match="EXPIRED"):
        instance.round_trip("BTCUSDT", Decimal("0.001"))
    assert transport.orders == {}


def test_insufficient_depth_blocks_before_first_order(bot, monkeypatch):
    instance, transport = bot
    assembly = instance.collector.collect("BTCUSDT")
    snapshot = assembly.snapshot
    small = replace(snapshot.spot_book, asks=tuple(replace(level, quantity=Decimal("0.0001"))
                                                  for level in snapshot.spot_book.asks))
    bad = replace(assembly, snapshot=replace(snapshot, spot_book=small), sources=replace(
        assembly.sources, spot_depth=replace(assembly.sources.spot_depth, order_book=small)))
    monkeypatch.setattr(instance.collector, "collect", lambda _: bad)
    with pytest.raises(ExecutionGuardError, match="DEPTH"):
        instance.round_trip("BTCUSDT", Decimal("0.001"))
    assert transport.orders == {}


def test_exit_prices_are_refreshed_and_perp_exit_is_reduce_only(bot, monkeypatch):
    instance, transport = bot
    plan = instance._plan("BTCUSDT", Decimal("0.001"), "fresh-close")
    assert instance.coordinator.open_pair(plan).state is State.HEDGED
    original = transport.open
    seen = []
    def changed(request, **kwargs):
        if "/depth?" in request.full_url:
            response = original(request, **kwargs)
            import json
            payload = json.loads(response.body)
            for key in ("bids", "asks"):
                payload[key][0][0] = str(Decimal(payload[key][0][0]) + 100)
            return DemoResponse(payload)
        if request.get_method() == "POST":
            seen.append(request.data.decode())
        return original(request, **kwargs)
    monkeypatch.setattr(transport, "open", changed)
    result = instance._close(plan)
    assert result.state is State.CLOSED
    assert result.spot_exit.average_price == Decimal("50099")
    assert result.perp_exit.average_price == Decimal("50110")
    assert "reduceOnly=true" in seen[-1]


def test_external_position_change_blocks_managed_exit(bot, monkeypatch):
    instance, transport = bot
    plan = instance._plan("BTCUSDT", Decimal("0.001"), "external-change")
    assert instance.coordinator.open_pair(plan).state is State.HEDGED
    transport.positions["BTCUSDT"] = Decimal("-0.002")
    with pytest.raises(ExecutionGuardError, match="EXPOSURE"):
        instance._close(plan)
    assert len(transport.orders) == 2 and instance.journal.get(plan.attempt_id).state is State.HEDGED


def test_parallel_close_has_exactly_one_winner(bot):
    instance, transport = bot
    plan = instance._plan("BTCUSDT", Decimal("0.001"), "concurrent-close")
    assert instance.coordinator.open_pair(plan).state is State.HEDGED
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(instance.coordinator.close_pair, (plan, plan)))
    assert sorted(result.state.value for result in results) == ["CLOSED", "DUPLICATE_BLOCKED"]
    assert len(transport.orders) == 4


def test_changed_close_plan_cannot_mutate_or_submit(bot):
    instance, transport = bot
    plan = instance._plan("BTCUSDT", Decimal("0.001"), "unchanged-plan")
    instance.coordinator.open_pair(plan)
    changed = replace(plan, spot_exit=replace(plan.spot_exit, limit_price=Decimal("1")))
    with pytest.raises(ValueError, match="differs"):
        instance.coordinator.close_pair(changed)
    assert len(transport.orders) == 2
    assert instance.journal.get(plan.attempt_id).state is State.HEDGED


def test_active_symbol_blocks_new_attempt_even_different_id(bot):
    instance, transport = bot
    plan = instance._plan("BTCUSDT", Decimal("0.001"), "first-active")
    instance.coordinator.open_pair(plan)
    with pytest.raises(ExecutionGuardError, match="PAIR_LIMIT"):
        instance._plan("BTCUSDT", Decimal("0.001"), "second-active")
    assert len(transport.orders) == 2


def test_session_attempt_limit_is_durable(bot):
    instance, transport = bot
    instance.config = instance.config.model_copy(update={"max_total_attempts": 1})
    instance.round_trip("BTCUSDT", Decimal("0.001"))
    with pytest.raises(ExecutionGuardError, match="ATTEMPT_LIMIT"):
        instance.round_trip("ETHUSDT", Decimal("0.01"))
    assert len(transport.orders) == 4


def test_interrupted_both_entries_reconcile_using_get_without_resubmit(bot):
    instance, transport = bot
    plan = instance._plan("BTCUSDT", Decimal("0.001"), "restart-known")
    opened = instance.coordinator.open_pair(plan)
    assert opened.state is State.HEDGED
    instance.journal.update(plan.attempt_id, state=State.OPENING_PERP)
    transport.requests.clear()
    restored = instance.coordinator.reconcile(plan)
    assert restored.state is State.HEDGED
    assert all(method == "GET" for method, _ in transport.requests)
    assert len(transport.orders) == 2


def test_interrupted_first_leg_never_blindly_submits_second_on_restart(bot):
    instance, transport = bot
    plan = instance._plan("BTCUSDT", Decimal("0.001"), "restart-single")
    instance.journal.create(plan)
    instance.journal.update(plan.attempt_id, state=State.OPENING_SPOT)
    instance.journal.record_order(plan.spot_entry)
    instance.spot.place_ioc_limit_order(plan.spot_entry)
    transport.requests.clear()
    result = instance.coordinator.reconcile(plan)
    assert result.state is State.RECOVERY_REQUIRED
    assert all(method == "GET" for method, _ in transport.requests)
    assert len(transport.orders) == 1
    with pytest.raises(ExecutionGuardError, match="UNRESOLVED"):
        instance._plan("ETHUSDT", Decimal("0.01"), "blocked-eth")


def test_restart_fully_closed_does_not_query_or_submit_again(bot):
    instance, transport = bot
    result = instance.round_trip("BTCUSDT", Decimal("0.001"))
    plan = instance.journal.plans()[0][0]
    transport.requests.clear()
    assert instance.coordinator.reconcile(plan) == result
    assert transport.requests == []


def test_prepared_no_order_intent_is_failed_not_resent(bot):
    instance, transport = bot
    plan = instance._plan("BTCUSDT", Decimal("0.001"), "before-first-post")
    instance.journal.create(plan)
    transport.requests.clear()
    assert instance.coordinator.reconcile(plan).state is State.FAILED
    assert transport.requests == [] and transport.orders == {}


def test_timeout_after_post_queries_exact_order_and_does_not_repeat_post(bot, monkeypatch):
    instance, transport = bot
    original = transport.open
    injected = False
    posts = []
    def lost_response(request, **kwargs):
        nonlocal injected
        response = original(request, **kwargs)
        if request.get_method() == "POST":
            posts.append(request.data)
            if not injected:
                injected = True
                raise TimeoutError("mock: exchange filled but response was lost")
        return response
    monkeypatch.setattr(transport, "open", lost_response)
    result = instance.round_trip("BTCUSDT", Decimal("0.001"))
    assert result.state is State.CLOSED
    assert len(posts) == len(set(posts)) == 4
    assert any("origClientOrderId=" in url for _, url in transport.requests)


def test_wrong_receipt_symbol_is_ambiguous_and_never_hedged(bot, monkeypatch):
    instance, transport = bot
    original = instance.spot.place_ioc_limit_order
    monkeypatch.setattr(instance.spot, "place_ioc_limit_order", lambda order: replace(original(order), symbol="ETHUSDT"))
    result = instance.round_trip("BTCUSDT", Decimal("0.001"))
    assert result.state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 1



def test_unknown_spot_commission_stops_before_futures_entry(bot, monkeypatch):
    instance, transport = bot
    original = instance.spot.place_ioc_limit_order
    monkeypatch.setattr(instance.spot, "place_ioc_limit_order", lambda order: replace(original(order), base_commission=None))
    result = instance.round_trip("BTCUSDT", Decimal("0.001"))
    assert result.state is State.RECOVERY_REQUIRED and result.failure_code == "SPOT_COMMISSION_UNKNOWN"
    assert len(transport.orders) == 1


def test_base_commission_does_not_create_false_full_hedge(bot, monkeypatch):
    instance, transport = bot
    original = instance.spot.place_ioc_limit_order
    def fee(order):
        receipt = original(order)
        if order.side.value == "BUY":
            transport.base[order.symbol] -= Decimal("0.000001")
            return replace(receipt, base_commission=Decimal("0.000001"))
        return receipt
    monkeypatch.setattr(instance.spot, "place_ioc_limit_order", fee)
    result = instance.round_trip("BTCUSDT", Decimal("0.001"))
    # Net quantity is off the current exchange grid; do not sell gross or fake alignment.
    assert result.state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 1
    assert transport.positions["BTCUSDT"] == 0
    assert instance.journal.get(result.attempt_id).spot_entry.base_commission == Decimal("0.000001")


@pytest.mark.parametrize("api_code", [-1006, -1007])
def test_binance_unknown_execution_api_errors_are_not_definite_rejections(bot, monkeypatch, api_code):
    instance, transport = bot
    original = transport.open
    def unknown(request, **kwargs):
        if request.get_method() == "POST":
            raise HTTPError(request.full_url, 400, "unknown", {}, BytesIO(
                f'{{"code":{api_code},"msg":"unknown execution"}}'.encode()))
        return original(request, **kwargs)
    monkeypatch.setattr(transport, "open", unknown)
    plan = instance._plan("BTCUSDT", Decimal("0.001"), "unknown-api")
    with pytest.raises(GatewayError) as error:
        instance.spot.place_ioc_limit_order(plan.spot_entry)
    assert error.value.submission_may_have_reached_exchange
    assert error.value.kind is ErrorKind.SUBMISSION_UNKNOWN


def test_lock_blocks_second_runner_and_releases_on_exit(tmp_path):
    path = tmp_path / "bot.lock"
    with process_lock(path):
        with pytest.raises(ExecutionGuardError):
            with process_lock(path):
                pytest.fail("second process lock acquired")
    with process_lock(path):
        pass


def test_loop_waits_and_stops_without_blind_liquidation(bot, monkeypatch):
    instance, transport = bot
    stop = Mock(spec=Event)
    stop.is_set.return_value = False
    stop.wait.return_value = True
    instance.stop = stop
    cycle = Mock(side_effect=RuntimeError("offline cycle failed"))
    monkeypatch.setattr(instance, "run_once", cycle)
    instance.run_forever()
    cycle.assert_called_once()
    stop.wait.assert_called_once_with(instance.settings.poll_interval_seconds)
    assert transport.orders == {}


def test_keyboard_interrupt_sets_stop_without_order_retry(bot, monkeypatch):
    instance, transport = bot
    monkeypatch.setattr(instance, "run_once", Mock(side_effect=KeyboardInterrupt))
    instance.run_forever()
    assert instance.stop.is_set() and transport.orders == {}


def test_exit_limit_rounding_and_depth_use_existing_fill(bot):
    instance, _ = bot
    assembly = instance.collector.collect("BTCUSDT")
    rules = instance.spot.get_symbol_rules("BTCUSDT")
    assert instance._limit(assembly, TradeLeg.SPOT_ENTRY, Decimal("0.001"), rules) == Decimal("50000")
    assert instance._limit(assembly, TradeLeg.SPOT_EXIT, Decimal("0.001"), rules) == Decimal("49999")


def test_original_public_configuration_and_thresholds_unchanged(bot):
    instance, _ = bot
    original = load_settings(Path("docs/acceptance-smoke.toml"))
    assert original.testnet is None
    assert instance.research.decision == original.research.decision
    assert instance.settings.fees == original.fees
    assert instance.research.validation_scheduler is None


def test_positive_synthetic_scenario_can_pass_then_testnet_bot_manages_entry_exit(bot, monkeypatch):
    """An explicit fictional settlement tests wiring, not expected funding income."""
    assume_offline_admission(monkeypatch)
    instance, transport = bot
    now = datetime.now(UTC)
    transport.clock = lambda: now
    instance.clock = lambda: now
    instance.collector.clock = lambda: now
    instance.research_runner.clock = lambda: now
    scenarios = {symbol: config.model_copy(update={"funding_events": (FundingEvent(
        f"synthetic-{symbol}", now + timedelta(seconds=30),
        Decimal("50000") if symbol == "BTCUSDT" else Decimal("2000"), Decimal("0.1"), symbol),)})
        for symbol, config in instance.research.scenarios.items()}
    instance.research = instance.research.model_copy(update={"scenarios": scenarios})
    instance.settings = instance.settings.model_copy(update={"research": instance.research})
    instance.research_runner.settings = instance.settings
    opened = instance.run_once()
    assert [cycle.status for cycle in opened] == ["HEDGED", "HEDGED"]
    assert len(transport.orders) == 4
    now += timedelta(seconds=65)
    closed_cycle = instance.run_once()
    assert all(instance.journal.get(plan.attempt_id).state is State.CLOSED for plan, _ in instance.journal.plans())
    assert len(transport.orders) == 8  # Synthetic event now lies outside the new research window.
    assert all(cycle.status == "NO_ENTRY" for cycle in closed_cycle)


def test_btc_collection_failure_does_not_stop_eth_research(bot, monkeypatch):
    instance, transport = bot
    original = instance.collector.collect
    def collect(symbol):
        if symbol == "BTCUSDT":
            raise GatewayError(ErrorKind.NETWORK_ERROR, "offline BTC failure")
        return original(symbol)
    monkeypatch.setattr(instance.collector, "collect", collect)
    cycles = instance.run_once()
    assert [(cycle.symbol, cycle.status) for cycle in cycles] == [("BTCUSDT", "RESEARCH_FAILED"), ("ETHUSDT", "NO_ENTRY")]
    assert instance.repository.get_opportunity_by_id(cycles[1].opportunity_id).symbol == "ETHUSDT"
    assert transport.orders == {}


def test_base_commission_flatten_uses_net_inventory_and_preserves_fee(bot, monkeypatch):
    instance, transport = bot
    original = instance.spot.place_ioc_limit_order
    def fee(order):
        receipt = original(order)
        if order.side.value == "BUY":
            transport.base[order.symbol] -= Decimal("0.001")
            return replace(receipt, base_commission=Decimal("0.001"))
        return receipt
    monkeypatch.setattr(instance.spot, "place_ioc_limit_order", fee)
    result = instance.round_trip("BTCUSDT", Decimal("0.002"))
    assert result.state is State.FLATTENED
    assert result.spot_exit.original_quantity == Decimal("0.001")
    assert result.spot_entry.base_commission == Decimal("0.001")
    assert transport.base["BTCUSDT"] == transport.positions["BTCUSDT"] == 0
    assert len(transport.orders) == 2


def test_nonterminal_ioc_receipt_is_queried_before_any_hedge(bot, monkeypatch):
    instance, transport = bot
    original = instance.spot.place_ioc_limit_order
    monkeypatch.setattr(instance.spot, "place_ioc_limit_order", lambda order: replace(original(order), status=OrderState.NEW))
    monkeypatch.setattr(instance.spot, "get_order", Mock(side_effect=GatewayError(ErrorKind.NETWORK_ERROR, "query failed")))
    result = instance.round_trip("BTCUSDT", Decimal("0.001"))
    assert result.state is State.RECOVERY_REQUIRED
    assert len(transport.orders) == 1
    instance.spot.get_order.assert_called_once()


def test_journal_write_failure_after_spot_fill_never_submits_hedge(bot, monkeypatch):
    from crypto_arbitrage_bot.execution.pair_coordinator import ExecutionJournalError
    instance, transport = bot
    plan = instance._plan("BTCUSDT", Decimal("0.001"), "journal-after-fill")
    original = instance.journal.update
    def fail(attempt_id, **kwargs):
        if kwargs.get("spot_entry") is not None:
            raise ExecutionJournalError("injected write failure")
        return original(attempt_id, **kwargs)
    monkeypatch.setattr(instance.journal, "update", fail)
    with pytest.raises(ExecutionJournalError):
        instance.coordinator.open_pair(plan)
    assert len(transport.orders) == 1
    assert instance.journal.get(plan.attempt_id).state is State.OPENING_SPOT
    assert len(instance.journal.intents(plan.attempt_id)) == 1
    assert instance.coordinator.open_pair(plan).state is State.DUPLICATE_BLOCKED


def test_overdue_exit_blocks_any_new_symbol_entry(bot):
    instance, transport = bot
    plan = instance._plan("BTCUSDT", Decimal("0.001"), "overdue-entry")
    instance.coordinator.open_pair(plan)
    now = datetime.now(UTC) + timedelta(seconds=120)
    instance.clock = lambda: now
    with pytest.raises(ExecutionGuardError, match="OVERDUE_EXIT"):
        instance._plan("ETHUSDT", Decimal("0.01"), "blocked-overdue")
    assert len(transport.orders) == 2


def test_cli_disables_orders_before_constructing_clients(monkeypatch, capsys):
    from crypto_arbitrage_bot.execution import main as cli
    monkeypatch.setattr("sys.argv", ["testnet", "--once"])
    monkeypatch.delenv("CAB_TESTNET_ORDERS_ENABLED", raising=False)
    factory = Mock(side_effect=AssertionError("no client should be constructed"))
    monkeypatch.setattr(cli, "create_testnet_bot", factory)
    assert cli.main() == 2
    assert "TESTNET_ORDERS_DISABLED" in capsys.readouterr().out
    factory.assert_not_called()


def test_cli_check_is_read_only_with_no_orders_enabled(bot, monkeypatch, capsys, tmp_path):
    from crypto_arbitrage_bot.execution import main as cli
    instance, transport = bot
    monkeypatch.setattr("sys.argv", ["testnet", "--check", "--credentials-file",
                                    str(tmp_path / "absent-fixture-credentials.dpapi")])
    monkeypatch.delenv("CAB_TESTNET_ORDERS_ENABLED", raising=False)
    monkeypatch.setattr(cli, "create_testnet_bot", lambda _, **kwargs: instance)
    assert cli.main() == 0
    assert '"orders_submitted":false' in capsys.readouterr().out
    assert all(method == "GET" for method, _ in transport.requests)


def test_time_sync_refreshes_for_long_running_session(bot):
    instance, transport = bot
    client = instance.spot
    now = 10000
    client._clock_ms = lambda: now
    client.get_account_state("BTCUSDT")
    before = sum("/time" in url for _, url in transport.requests)
    now += 31000
    client.get_account_state("BTCUSDT")
    assert sum("/time" in url for _, url in transport.requests) == before + 1
