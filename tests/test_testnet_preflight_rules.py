"""One-exit fresh rule reuse never changes filters, routes or POST ambiguity."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock
from urllib.parse import urlsplit

import pytest
from test_funding_lifecycle import live_clock_bot as live_clock_bot

from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as GatewayError


def setup(fixture, market):
    bot, transport, _ = fixture
    plan = bot._plan("ETHUSDT", Decimal(".01"), "preflight-rules-test")
    client = bot.spot if market == "SPOT" else bot.perp
    order = plan.spot_entry if market == "SPOT" else plan.perp_entry
    return client, order, transport


def info_count(transport):
    return sum(urlsplit(url).path.endswith("exchangeInfo") for method, url in transport.requests)


@pytest.mark.parametrize("market", ["SPOT", "PERP"])
def test_fresh_scoped_rules_remove_duplicate_info_get_only(live_clock_bot, market):
    client, order, transport = setup(live_clock_bot, market)
    rules = client.get_symbol_rules(order.symbol)
    before = info_count(transport)
    with client.preflight_rules(rules, max_age=timedelta(seconds=10)):
        receipt = client.place_ioc_limit_order(order)
    assert receipt.executed_quantity == order.quantity
    assert info_count(transport) == before
    assert len(transport.orders) == 1
    assert client._preflight_rules is None


@pytest.mark.parametrize("market", ["SPOT", "PERP"])
def test_rules_expired_at_exact_boundary_require_fresh_get(live_clock_bot, market, monkeypatch):
    client, order, transport = setup(live_clock_bot, market)
    clock = {"ns": 100000000000}
    monkeypatch.setattr("crypto_arbitrage_bot.execution.testnet_client.time.monotonic_ns", lambda: clock["ns"])
    rules = client.get_symbol_rules(order.symbol)
    before = info_count(transport)
    with client.preflight_rules(rules, max_age=timedelta(seconds=10)):
        clock["ns"] += 10000000000
        client.place_ioc_limit_order(order)
    assert info_count(transport) == before + 1


@pytest.mark.parametrize("market", ["SPOT", "PERP"])
def test_outside_scope_always_fetches_fresh_rules(live_clock_bot, market):
    client, order, transport = setup(live_clock_bot, market)
    rules = client.get_symbol_rules(order.symbol)
    with client.preflight_rules(rules, max_age=timedelta(seconds=10)):
        pass
    before = info_count(transport)
    client.place_ioc_limit_order(order)
    assert info_count(transport) == before + 1


def test_fabricated_equal_rule_object_is_rejected(live_clock_bot):
    client, order, transport = setup(live_clock_bot, "SPOT")
    rules = client.get_symbol_rules(order.symbol)
    with pytest.raises(GatewayError, match="current preflight"):
        with client.preflight_rules(replace(rules), max_age=timedelta(seconds=10)):
            client.place_ioc_limit_order(order)
    assert transport.orders == {}


def test_each_order_is_still_filter_validated(live_clock_bot):
    client, order, transport = setup(live_clock_bot, "PERP")
    rules = client.get_symbol_rules(order.symbol)
    with pytest.raises(GatewayError):
        with client.preflight_rules(rules, max_age=timedelta(seconds=10)):
            client.place_ioc_limit_order(replace(order, quantity=Decimal(".01000001")))
    assert transport.orders == {} and client._preflight_rules is None


def test_unknown_post_not_retried_and_scope_is_released(live_clock_bot, monkeypatch):
    client, order, transport = setup(live_clock_bot, "PERP")
    rules = client.get_symbol_rules(order.symbol)
    original = transport.open
    posts = []
    def open_request(request, **kwargs):
        if request.get_method() == "POST":
            posts.append(request)
            raise TimeoutError("POST response unavailable")
        return original(request, **kwargs)
    monkeypatch.setattr(transport, "open", open_request)
    with pytest.raises(GatewayError) as result:
        with client.preflight_rules(rules, max_age=timedelta(seconds=10)):
            client.place_ioc_limit_order(order)
    assert result.value.submission_may_have_reached_exchange
    assert len(posts) == 1 and client._preflight_rules is None


def test_managed_close_reuses_its_own_fresh_filters(live_clock_bot):
    bot, transport, state = live_clock_bot
    plan = bot._plan("ETHUSDT", Decimal(".01"), "managed-fresh-rules")
    assert bot.coordinator.open_pair(plan).state.value == "HEDGED"
    before = info_count(transport)
    state["now"] += timedelta(seconds=65)
    assert bot._close(plan).state.value == "CLOSED"
    assert info_count(transport) == before + 2  # One Spot and one Perp preflight, no duplicate before POST.
    assert len(transport.orders) == 4


def test_expired_rules_fetch_failure_cannot_send_order(live_clock_bot, monkeypatch):
    client, order, transport = setup(live_clock_bot, "PERP")
    clock = {"ns": 100000000000}
    monkeypatch.setattr("crypto_arbitrage_bot.execution.testnet_client.time.monotonic_ns", lambda: clock["ns"])
    rules = client.get_symbol_rules(order.symbol)
    original = transport.open
    def open_request(request, **kwargs):
        if urlsplit(request.full_url).path.endswith("exchangeInfo"):
            raise TimeoutError("fresh filter fetch unavailable")
        return original(request, **kwargs)
    monkeypatch.setattr(transport, "open", open_request)
    with pytest.raises(GatewayError):
        with client.preflight_rules(rules, max_age=timedelta(seconds=10)):
            clock["ns"] += 11000000000
            client.place_ioc_limit_order(order)
    assert transport.orders == {} and client._preflight_rules is None


def test_quote_and_account_reads_are_not_cached_by_rule_scope(live_clock_bot, monkeypatch):
    client, order, _ = setup(live_clock_bot, "PERP")
    rules = client.get_symbol_rules(order.symbol)
    original = client._request
    requests = Mock(wraps=original)
    monkeypatch.setattr(client, "_request", requests)
    with client.preflight_rules(rules, max_age=timedelta(seconds=10)):
        client.public_payload("depth", "ETHUSDT", capture=lambda body: None)
        client.public_payload("depth", "ETHUSDT", capture=lambda body: None)
        client.get_account_state("ETHUSDT")
        client.get_account_state("ETHUSDT")
    assert sum(call.args[0] == "depth" for call in requests.call_args_list) == 2
    assert sum(call.args[0] == "account" for call in requests.call_args_list) == 2
