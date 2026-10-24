"""Only USDT-M Demo flat-account preparation can set a fixed multiplier of 1."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import parse_qs

import pytest
from test_testnet_execution import Opener, Response, client

from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution import main as cli
from crypto_arbitrage_bot.execution.demo import demo_bot
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as GatewayError
from crypto_arbitrage_bot.execution.types import ExecutionMarket


def account(leverage="20", symbol="BTCUSDT"):
    from crypto_arbitrage_bot.execution.types import AccountState
    return AccountState(ExecutionMarket.USDTM, symbol, Decimal("5000"), Decimal("0"),
                        Decimal("0"), True, True, True, Decimal(leverage), 0)


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_exact_one_x_signed_post_then_fresh_account_verify(monkeypatch, symbol):
    opener = Opener([Response({"serverTime": 10000}), Response({"symbol": symbol, "leverage": 1})])
    api = client(ExecutionMarket.USDTM, opener)
    states = Mock(side_effect=[account(symbol=symbol), account("1", symbol)])
    monkeypatch.setattr(api, "get_account_state", states)
    assert api.ensure_one_x(symbol).leverage == Decimal("1")
    assert states.call_count == 2
    request = opener.requests[-1][0]
    assert request.full_url == "https://demo-fapi.binance.com/fapi/v1/leverage"
    assert request.get_method() == "POST"
    params = parse_qs(request.data.decode())
    assert params["symbol"] == [symbol] and params["leverage"] == ["1"]
    assert "signature" in params
    assert all("/order" not in request.full_url for request, _ in opener.requests)


def test_already_one_x_does_not_submit_any_request(monkeypatch):
    opener = Opener([])
    api = client(ExecutionMarket.USDTM, opener)
    monkeypatch.setattr(api, "get_account_state", lambda _: account("1"))
    assert api.ensure_one_x("BTCUSDT").leverage == 1
    assert opener.requests == []


@pytest.mark.parametrize("field,value", [("position_quantity", Decimal("1")), ("position_quantity", Decimal("-1")),
                                         ("open_order_count", 1), ("can_trade", False),
                                         ("one_way", False), ("single_asset", False), ("leverage", Decimal("0"))])
def test_unmanaged_or_invalid_account_never_modified(monkeypatch, field, value):
    opener = Opener([])
    api = client(ExecutionMarket.USDTM, opener)
    monkeypatch.setattr(api, "get_account_state", lambda _: replace(account(), **{field: value}))
    with pytest.raises(GatewayError) as error:
        api.ensure_one_x("BTCUSDT")
    assert error.value.kind is ExecutionErrorKind.ORDER_GUARD_REJECTED
    assert opener.requests == []


@pytest.mark.parametrize("market,symbol", [(ExecutionMarket.SPOT, "BTCUSDT"), (ExecutionMarket.USDTM, "BNBUSDT")])
def test_no_spot_or_unsupported_symbol_preparation(market, symbol):
    opener = Opener([])
    with pytest.raises(ValueError):
        client(market, opener).ensure_one_x(symbol)
    assert opener.requests == []


@pytest.mark.parametrize("reply", [{}, {"symbol": "ETHUSDT", "leverage": 1},
                                  {"symbol": "BTCUSDT", "leverage": 2},
                                  {"symbol": "BTCUSDT", "leverage": "1"},
                                  {"symbol": "BTCUSDT", "leverage": True}])
def test_bad_response_cannot_authorize_orders(monkeypatch, reply):
    api = client(ExecutionMarket.USDTM, Opener([Response({"serverTime": 10000}), Response(reply)]))
    states = Mock(return_value=account())
    monkeypatch.setattr(api, "get_account_state", states)
    with pytest.raises(GatewayError) as error:
        api.ensure_one_x("BTCUSDT")
    assert error.value.kind is ExecutionErrorKind.INVALID_RESPONSE
    states.assert_called_once()


def test_server_acknowledgement_requires_account_recheck(monkeypatch):
    opener = Opener([Response({"serverTime": 10000}), Response({"symbol": "BTCUSDT", "leverage": 1})])
    api = client(ExecutionMarket.USDTM, opener)
    monkeypatch.setattr(api, "get_account_state", Mock(side_effect=[account(), account("20")]))
    with pytest.raises(GatewayError, match="recheck failed"):
        api.ensure_one_x("BTCUSDT")
    assert sum(req.get_method() == "POST" for req, _ in opener.requests) == 1


def test_timeout_is_not_retried_or_mistaken_for_success(monkeypatch):
    opener = Opener([Response({"serverTime": 10000}), TimeoutError("synthetic")])
    api = client(ExecutionMarket.USDTM, opener)
    monkeypatch.setattr(api, "get_account_state", lambda _: account())
    with pytest.raises(GatewayError) as error:
        api.ensure_one_x("BTCUSDT")
    assert error.value.kind is ExecutionErrorKind.SUBMISSION_UNKNOWN
    assert sum(req.get_method() == "POST" for req, _ in opener.requests) == 1


def test_runner_auto_prepares_before_any_entry_order(tmp_path, monkeypatch):
    instance, transport = demo_bot(load_settings(Path("docs/testnet-bot.toml")), tmp_path)
    original = instance.perp.get_account_state
    monkeypatch.setattr(instance.perp, "get_account_state", lambda symbol: replace(original(symbol), leverage=Decimal("20")))
    ready = Mock(return_value=account("1"))
    monkeypatch.setattr(instance.perp, "ensure_one_x", ready)
    plan = instance._plan("BTCUSDT", Decimal("0.001"), "prepare-before-entry")
    assert plan.symbol == "BTCUSDT" and transport.orders == {}
    ready.assert_called_once_with("BTCUSDT")


def test_runner_preparation_failure_blocks_all_orders(tmp_path, monkeypatch):
    instance, transport = demo_bot(load_settings(Path("docs/testnet-bot.toml")), tmp_path)
    original = instance.perp.get_account_state
    monkeypatch.setattr(instance.perp, "get_account_state", lambda symbol: replace(original(symbol), leverage=Decimal("20")))
    monkeypatch.setattr(instance.perp, "ensure_one_x", Mock(side_effect=GatewayError(ExecutionErrorKind.NETWORK_ERROR, "no confirmation")))
    with pytest.raises(ExecutionGuardError, match="1X_PREPARATION_FAILED"):
        instance.round_trip("BTCUSDT", Decimal("0.001"))
    assert transport.orders == {} and instance.journal.plans() == ()


def test_explicit_prepare_and_check_are_separate(tmp_path, monkeypatch):
    instance, transport = demo_bot(load_settings(Path("docs/testnet-bot.toml")), tmp_path)
    ready = Mock(return_value=account("1"))
    monkeypatch.setattr(instance.perp, "ensure_one_x", ready)
    instance.prepare("BTCUSDT")
    ready.assert_called_once_with("BTCUSDT")
    ready.reset_mock()
    instance.check("BTCUSDT")
    ready.assert_not_called()
    assert transport.orders == {}


def test_cli_prepare_is_explicit_and_never_round_trip(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["testnet", "--prepare"])
    monkeypatch.delenv("CAB_TESTNET_ORDERS_ENABLED", raising=False)
    bot = Mock()
    monkeypatch.setattr(cli, "execution_environment", lambda *args: {})
    monkeypatch.setattr(cli, "create_testnet_bot", lambda *args, **kwargs: bot)
    assert cli.main() == 0
    assert bot.prepare.call_count == 2
    bot.round_trip.assert_not_called()
    bot.run_once.assert_not_called()
    assert "PREPARED_1X" in capsys.readouterr().out


def test_slow_entry_account_preflight_precedes_fresh_snapshot(tmp_path, monkeypatch):
    instance, transport = demo_bot(load_settings(Path("docs/testnet-bot.toml")), tmp_path)
    now = instance.clock()
    instance.clock = lambda: now
    instance.collector.clock = lambda: now
    transport.clock = lambda: now
    original = instance.perp.get_account_state
    def slow(symbol):
        nonlocal now
        now += timedelta(seconds=20)
        return original(symbol)
    monkeypatch.setattr(instance.perp, "get_account_state", slow)
    plan = instance._plan("BTCUSDT", Decimal("0.001"), "slow-preflight")
    assert plan.symbol == "BTCUSDT" and transport.orders == {}


def test_exit_uses_new_snapshot_after_slow_account_preflight(tmp_path, monkeypatch):
    instance, transport = demo_bot(load_settings(Path("docs/testnet-bot.toml")), tmp_path)
    plan = instance._plan("BTCUSDT", Decimal("0.001"), "slow-exit")
    instance.coordinator.open_pair(plan)
    now = instance.clock()
    instance.clock = lambda: now
    instance.collector.clock = lambda: now
    transport.clock = lambda: now
    original = instance.perp.get_account_state
    def slow(symbol):
        nonlocal now
        now += timedelta(seconds=20)
        return original(symbol)
    monkeypatch.setattr(instance.perp, "get_account_state", slow)
    assert instance._close(plan).state.value == "CLOSED"
    assert len(transport.orders) == 4
