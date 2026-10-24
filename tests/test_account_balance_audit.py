"""Offline account wallet audit: currency units, raw evidence, no orders."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

import pytest
from test_testnet_execution import Opener, Response, client

from crypto_arbitrage_bot.execution.account_balance_audit import (
    AccountAdjustment,
    AssetBalance,
    BalanceSnapshot,
    audit_balance_changes,
    capture_balance_snapshot,
)
from crypto_arbitrage_bot.execution.account_balance_audit import (
    BalanceAuditCode as Code,
)
from crypto_arbitrage_bot.execution.account_balance_audit import (
    BalanceAuditStatus as Status,
)
from crypto_arbitrage_bot.execution.testnet_client import ExecutionErrorKind
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError as Error
from crypto_arbitrage_bot.execution.types import AccountTrade, FundingPayment
from crypto_arbitrage_bot.execution.types import ExecutionMarket as Market
from crypto_arbitrage_bot.models.market import Side

D = Decimal
AT = datetime(2026, 10, 5, tzinfo=UTC)


def snapshot(market, seconds, balances):
    at = AT + timedelta(seconds=seconds)
    return BalanceSnapshot(market, at, at, tuple(AssetBalance(a, D(v)) for a, v in balances.items()))


@pytest.fixture
def example():
    before = (snapshot(Market.SPOT, 0, {"USDT": "1000", "ETH": "1", "BNB": "1"}),
              snapshot(Market.USDTM, 0, {"USDT": "1000", "BNB": "1"}))
    after = (snapshot(Market.SPOT, 400, {"USDT": "999.96999", "ETH": "1", "BNB": "1"}),
             snapshot(Market.USDTM, 400, {"USDT": "1000.06994", "BNB": "1"}))
    trades = tuple(AccountTrade(market, "ETHUSDT", str(i + 1), str(i + 1), side,
                               D(price), D(".01"), D(price) * D(".01"), D(fee), "USDT",
                               AT + timedelta(seconds=seconds), None if pnl is None else D(pnl))
                   for i, (market, side, price, fee, seconds, pnl) in enumerate([
                       (Market.SPOT, Side.BUY, "2000", ".02", 1, None),
                       (Market.SPOT, Side.SELL, "2001", ".02001", 300, None),
                       (Market.USDTM, Side.SELL, "2010", ".01005", 2, "0"),
                       (Market.USDTM, Side.BUY, "2002", ".01001", 301, ".08"),
                   ]))
    funding = (FundingPayment("f1", "ETHUSDT", AT + timedelta(seconds=200), D(".01")),)
    return before, after, trades, funding


def run(example, **kwargs):
    return audit_balance_changes(*example, funding_evidence_complete=True, **kwargs)


def rows(result):
    return {(m.market, m.asset): m for m in result.movements}


def codes(result):
    return {i.code for i in result.issues}


def test_four_fill_cash_and_fees_funding_match_once(example):
    result = run(example)
    assert result.status is Status.MATCHED and result.issues == ()
    assert rows(result)[Market.SPOT, "USDT"].observed_delta == D("-.03001")
    assert rows(result)[Market.USDTM, "USDT"].known_event_delta == D(".06994")
    assert all(m.unexplained_delta == 0 for m in result.movements)
    assert sum(m.observed_delta for m in result.movements if m.asset == "USDT") == D(".03993")
    assert not hasattr(result, "net_profit")  # Not a second PnL/strategy verdict.


def test_missing_funding_proof_not_invented_zero(example):
    result = audit_balance_changes(*example)
    assert result.status is Status.INCOMPLETE
    assert Code.FUNDING_EVIDENCE_INCOMPLETE in codes(result)


@pytest.mark.parametrize("amount", [".01", "0", "-.02"])
def test_signed_actual_funding(example, amount):
    before, after, trades, funding = example
    after = (after[0], snapshot(Market.USDTM, 400, {"USDT": str(D("1000.05994") + D(amount)), "BNB": "1"}))
    result = run((before, after, trades, (replace(funding[0], amount=D(amount)),)))
    assert result.status is Status.MATCHED


def test_base_asset_fee_tracks_net_inventory_without_double_quote_charge(example):
    before, _, trades, funding = example
    entry = replace(trades[0], commission_asset="ETH", commission=D(".00001"))
    exit_trade = replace(trades[1], quantity=D(".00999"), quote_quantity=D("19.98999"), commission=D(".01998999"))
    after = (snapshot(Market.SPOT, 400, {"USDT": "999.97000001", "ETH": "1", "BNB": "1"}),
             example[1][1])
    result = run((before, after, (entry, exit_trade, *trades[2:]), funding))
    assert result.status is Status.MATCHED
    assert rows(result)[Market.SPOT, "ETH"].known_event_delta == 0
    assert rows(result)[Market.SPOT, "USDT"].known_event_delta == D("-.02999999")


def test_bnb_fee_stays_bnb_no_fake_usdt_conversion(example):
    before, after, trades, funding = example
    trade = replace(trades[0], commission_asset="BNB", commission=D(".00003"))
    after = (snapshot(Market.SPOT, 400, {"USDT": "999.98999", "ETH": "1", "BNB": ".99997"}), after[1])
    result = run((before, after, (trade, *trades[1:]), funding))
    assert result.status is Status.MATCHED
    assert rows(result)[Market.SPOT, "BNB"].known_event_delta == D("-.00003")


def test_rebate_is_actual_asset_credit(example):
    before, after, trades, funding = example
    after = (snapshot(Market.SPOT, 400, {"USDT": "999.99999", "ETH": "1", "BNB": "1"}), after[1])
    result = run((before, after, (replace(trades[0], commission=D("-.01")), *trades[1:]), funding))
    assert result.status is Status.MATCHED


def test_missing_asset_remains_null(example):
    before, after, trades, funding = example
    after = (snapshot(Market.SPOT, 400, {"USDT": "999.96999", "BNB": "1"}), after[1])
    result = run((before, after, trades, funding))
    assert result.status is Status.INCOMPLETE and Code.MISSING_BALANCE in codes(result)
    assert rows(result)[Market.SPOT, "ETH"].observed_delta is None


def test_unexplained_transfer_is_mismatch_until_explicit_event(example):
    before, after, trades, funding = example
    after = (snapshot(Market.SPOT, 400, {"USDT": "1000.96999", "ETH": "1", "BNB": "1"}), after[1])
    inputs = before, after, trades, funding
    assert run(inputs).status is Status.MISMATCH
    adjustment = AccountAdjustment("transfer-1", Market.SPOT, "USDT", D("1"), AT + timedelta(seconds=50), "explicit test transfer")
    assert run(inputs, adjustments=(adjustment,)).status is Status.MATCHED


@pytest.mark.parametrize("source", ["trade", "funding", "adjustment"])
def test_duplicate_source_never_double_counted(example, source):
    before, after, trades, funding = example
    adjustments = ()
    if source == "trade":
        trades += (trades[0],)
    elif source == "funding":
        funding += (funding[0],)
    else:
        a = AccountAdjustment("transfer-1", Market.SPOT, "USDT", D("0"), AT + timedelta(seconds=50), "explicit zero")
        adjustments = a, a
    result = run((before, after, trades, funding), adjustments=adjustments)
    assert result.status is Status.INVALID and Code.DUPLICATE_EVENT in codes(result)


@pytest.mark.parametrize("seconds", [0, 401])
def test_outside_event_window_invalid(example, seconds):
    before, after, trades, funding = example
    result = run((before, after, (replace(trades[0], traded_at=AT + timedelta(seconds=seconds)), *trades[1:]), funding))
    assert result.status is Status.INVALID and Code.EVENT_OUTSIDE_WINDOW in codes(result)


def test_missing_or_duplicate_account_frames(example):
    before, after, trades, funding = example
    assert run((before[:1], after, trades, funding)).status is Status.INCOMPLETE
    assert run((before + before[:1], after, trades, funding)).status is Status.INVALID


def test_frame_overlap_invalid(example):
    before, after, trades, funding = example
    result = run((before, (replace(after[0], requested_at=AT), after[1]), trades, funding))
    assert result.status is Status.INVALID and Code.INVALID_TIME_WINDOW in codes(result)


def test_low_precision_does_not_round_cash_or_mutate_sources(example):
    original = repr(example)
    with localcontext() as context:
        context.prec = 4
        result = run(example)
    assert result.status is Status.MATCHED and repr(example) == original
    assert rows(result)[Market.SPOT, "USDT"].known_event_delta == D("-.03001")


@pytest.mark.parametrize("bad", [D("-1"), D("NaN"), 0.01])
def test_tolerance_requires_nonnegative_finite_decimal(example, bad):
    with pytest.raises((TypeError, ValueError)):
        run(example, tolerances={"USDT": bad})


def test_explicit_rounding_tolerance_only_audits_not_changes_amount(example):
    before, after, trades, funding = example
    after = (snapshot(Market.SPOT, 400, {"USDT": "999.969990001", "ETH": "1", "BNB": "1"}), after[1])
    result = run((before, after, trades, funding), tolerances={"USDT": D(".00000001")})
    assert result.status is Status.MATCHED
    assert rows(result)[Market.SPOT, "USDT"].unexplained_delta == D(".000000001")


@pytest.mark.parametrize("market", list(Market))
def test_capture_reuses_only_fixed_signed_get_and_correct_balance_basis(market):
    payload = ({"balances": [{"asset": "USDT", "free": "100.1234567890123456789", "locked": "2"}], "updateTime": 0}
               if market is Market.SPOT else
               {"multiAssetsMargin": False, "updateTime": 999999999999,
                "availableBalance": "1", "assets": [{"asset": "USDT", "walletBalance": "102.1234567890123456789"}]})
    opener = Opener([Response({"serverTime": 10000}), Response(payload)])
    times = iter((AT, AT + timedelta(seconds=1)))
    result = capture_balance_snapshot(client(market, opener), clock=lambda: next(times))
    assert result.balances[0].amount == D("102.1234567890123456789")
    assert result.official_updated_at is None
    assert result.requested_at == AT and result.received_at == AT + timedelta(seconds=1)
    assert all(request.get_method() == "GET" for request, _ in opener.requests)
    assert result.balance_basis == ("SPOT_FREE_PLUS_LOCKED" if market is Market.SPOT else "USDTM_WALLET_BALANCE")
    assert "walletBalance" in result.raw_balances_json if market is Market.USDTM else "locked" in result.raw_balances_json


@pytest.mark.parametrize("payload", [
    {"balances": []}, {"balances": [{"asset": "USDT", "free": "1"}]},
    {"balances": [{"asset": "USDT", "free": "-1", "locked": "0"}]},
    {"balances": [{"asset": "USDT", "free": "1", "locked": "NaN"}]},
    {"balances": [{"asset": "BTC", "free": "1", "locked": "0"}]},
    {"balances": [{"asset": "USDT", "free": "1", "locked": "0"}] * 2},
])
def test_malformed_capture_safe_failure(payload):
    opener = Opener([Response({"serverTime": 10000}), Response(payload)])
    with pytest.raises(Error) as caught:
        capture_balance_snapshot(client(Market.SPOT, opener), clock=lambda: AT)
    assert caught.value.kind is ExecutionErrorKind.INVALID_RESPONSE


def test_capture_network_failure_never_makes_empty_balance_frame():
    opener = Opener([Response({"serverTime": 10000}), TimeoutError("mock network")])
    with pytest.raises(Error) as caught:
        capture_balance_snapshot(client(Market.USDTM, opener), clock=lambda: AT)
    assert caught.value.kind is ExecutionErrorKind.NETWORK_ERROR


def test_official_unicode_balance_identifiers_are_kept_not_dropped():
    payload = {"balances": [{"asset": "USDT", "free": "100", "locked": "0"},
                            {"asset": "币安人生", "free": "0.00000001", "locked": "0"}]}
    opener = Opener([Response({"serverTime": 10000}), Response(payload)])
    result = capture_balance_snapshot(client(Market.SPOT, opener), clock=lambda: AT)
    assert result.balances[1].asset == "币安人生"
    assert result.balances[1].amount == D(".00000001")
    assert all(request.get_method() == "GET" for request, _ in opener.requests)


@pytest.mark.parametrize("asset", ["", "usdt", "A B", "BTC/USDT", "USDT\n", "x" * 65])
def test_wallet_labels_reject_ambiguous_or_control_characters(asset):
    with pytest.raises(ValueError):
        AssetBalance(asset, D("0"))
