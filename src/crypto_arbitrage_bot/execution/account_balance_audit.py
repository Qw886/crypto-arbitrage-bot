"""Read-only currency ledger audit, not a second profit calculation.

Spot uses free+locked; USDT-M uses walletBalance, never available margin.
No currency conversion, invented funding, order or account mutation occurs.
MATCHED means balances match the supplied events, not a profitable strategy.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from crypto_arbitrage_bot.execution.testnet_client import (
    ExecutionErrorKind,
    TestnetExecutionClient,
    TestnetExecutionError,
    _decimal,
)
from crypto_arbitrage_bot.execution.types import AccountTrade, ExecutionMarket, FundingPayment
from crypto_arbitrage_bot.models.market import Side
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import (
    decimal_calculation,
    require_decimal,
    utc_timestamp,
)


def _asset(value: str) -> str:
    # Official account rows can include Chinese token identifiers. A wallet
    # label is not a supported trading symbol or permission to trade it.
    if (type(value) is not str or not 1 <= len(value) <= 64
            or not value.isalnum() or value != value.upper()):
        raise ValueError("invalid balance asset")
    return value


@dataclass(frozen=True)
class AssetBalance:
    asset: str
    amount: Decimal

    def __post_init__(self) -> None:
        _asset(self.asset)
        require_decimal(self.amount, "asset balance")


@dataclass(frozen=True)
class BalanceSnapshot:
    market: ExecutionMarket
    requested_at: datetime
    received_at: datetime
    balances: tuple[AssetBalance, ...]
    official_updated_at: datetime | None = None
    raw_balances_json: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.market, ExecutionMarket):
            raise TypeError("snapshot market must be an ExecutionMarket")
        requested = utc_timestamp(self.requested_at)
        received = utc_timestamp(self.received_at)
        if received < requested:
            raise ValueError("balance response predates request")
        if not isinstance(self.balances, tuple) or not self.balances:
            raise ValueError("nonempty immutable balances required")
        if any(not isinstance(b, AssetBalance) for b in self.balances):
            raise TypeError("balances must contain AssetBalance")
        if len({b.asset for b in self.balances}) != len(self.balances):
            raise ValueError("duplicate asset balances")
        if self.market is ExecutionMarket.SPOT:
            for balance in self.balances:
                require_decimal(balance.amount, "Spot total balance", nonnegative=True)
        object.__setattr__(self, "requested_at", requested)
        object.__setattr__(self, "received_at", received)
        if self.official_updated_at is not None:
            object.__setattr__(self, "official_updated_at", utc_timestamp(self.official_updated_at))

    @property
    def balance_basis(self) -> str:
        return "SPOT_FREE_PLUS_LOCKED" if self.market is ExecutionMarket.SPOT else "USDTM_WALLET_BALANCE"


@decimal_calculation
def capture_balance_snapshot(client: TestnetExecutionClient, *,
                             clock: Callable[[], datetime] = utc_now) -> BalanceSnapshot:
    """Reuse the existing fixed TESTNET signed account GET; no new endpoint."""
    requested = utc_timestamp(clock())
    payload = client._request("account", signed=True)
    received = utc_timestamp(clock())
    try:
        spot = client.market is ExecutionMarket.SPOT
        key = "balances" if spot else "assets"
        if not spot and payload["multiAssetsMargin"] is not False:
            raise ValueError("single asset Demo account required")
        rows = payload[key]
        if not isinstance(rows, list) or not rows:
            raise ValueError("missing asset balances")
        balances = []
        for row in rows:
            if spot:
                free = _decimal(row["free"], "free")
                locked = _decimal(row["locked"], "locked")
                require_decimal(free, "free", nonnegative=True)
                require_decimal(locked, "locked", nonnegative=True)
                amount = free + locked
            else:
                amount = _decimal(row["walletBalance"], "walletBalance")
            balances.append(AssetBalance(row["asset"], amount))
        if "USDT" not in {b.asset for b in balances}:
            raise ValueError("USDT balance missing")
        # Futures top-level updateTime is reserved, not a snapshot event time.
        update = payload.get("updateTime") if spot else None
        if update is not None and (type(update) is not int or update < 0):
            raise ValueError("invalid official balance update time")
        updated = None if not update else datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=update)
        return BalanceSnapshot(client.market, requested, received, tuple(balances), updated,
                               exact_json({key: rows, "updateTime": payload.get("updateTime")}))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise TestnetExecutionError(ExecutionErrorKind.INVALID_RESPONSE,
                                    "invalid testnet balance snapshot", market=client.market) from error


class BalanceAuditStatus(StrEnum):
    MATCHED = "MATCHED"
    MISMATCH = "MISMATCH"
    INCOMPLETE = "INCOMPLETE"
    INVALID = "INVALID"


class BalanceAuditCode(StrEnum):
    MISSING_SNAPSHOT = "MISSING_SNAPSHOT"
    MISSING_BALANCE = "MISSING_BALANCE"
    DUPLICATE_SNAPSHOT = "DUPLICATE_SNAPSHOT"
    INVALID_TIME_WINDOW = "INVALID_TIME_WINDOW"
    EVENT_OUTSIDE_WINDOW = "EVENT_OUTSIDE_WINDOW"
    DUPLICATE_EVENT = "DUPLICATE_EVENT"
    INVALID_TRADE_NOTIONAL = "INVALID_TRADE_NOTIONAL"
    FUNDING_EVIDENCE_INCOMPLETE = "FUNDING_EVIDENCE_INCOMPLETE"
    UNEXPLAINED_BALANCE_CHANGE = "UNEXPLAINED_BALANCE_CHANGE"


@dataclass(frozen=True)
class BalanceAuditIssue:
    code: BalanceAuditCode
    market: ExecutionMarket | None = None
    asset: str | None = None


@dataclass(frozen=True)
class AccountAdjustment:
    """Explicit external cash/asset event, e.g. transfer; never inferred."""
    event_id: str
    market: ExecutionMarket
    asset: str
    amount: Decimal
    posted_at: datetime
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.market, ExecutionMarket):
            raise TypeError("adjustment market must be typed")
        if (type(self.event_id) is not str or not self.event_id
                or type(self.description) is not str or not self.description):
            raise ValueError("adjustment needs identity and description")
        _asset(self.asset)
        require_decimal(self.amount, "explicit account adjustment")
        object.__setattr__(self, "posted_at", utc_timestamp(self.posted_at))


@dataclass(frozen=True)
class AssetMovement:
    market: ExecutionMarket
    asset: str
    known_event_delta: Decimal
    observed_delta: Decimal | None
    unexplained_delta: Decimal | None
    tolerance: Decimal


@dataclass(frozen=True)
class BalanceAudit:
    status: BalanceAuditStatus
    issues: tuple[BalanceAuditIssue, ...]
    movements: tuple[AssetMovement, ...]
    before: tuple[BalanceSnapshot, ...]
    after: tuple[BalanceSnapshot, ...]
    trades: tuple[AccountTrade, ...]
    funding: tuple[FundingPayment, ...]
    adjustments: tuple[AccountAdjustment, ...]
    funding_evidence_complete: bool
    scope: str = "TESTNET_CURRENCY_BALANCE_V1_NOT_PROFIT"


@decimal_calculation
def audit_balance_changes(before: tuple[BalanceSnapshot, ...], after: tuple[BalanceSnapshot, ...],
                          trades: tuple[AccountTrade, ...], funding: tuple[FundingPayment, ...], *,
                          funding_evidence_complete: bool = False,
                          adjustments: tuple[AccountAdjustment, ...] = (),
                          tolerances: Mapping[str, Decimal] | None = None) -> BalanceAudit:
    """Explain wallet/asset changes from actual supplied events, once each.

    Futures traded notional is not wallet cash. Use official fill PnL instead.
    Commissions retain their own assets (negative commissions are credits).
    Missing pre/post asset evidence is NULL, never an invented zero balance.
    """
    if type(funding_evidence_complete) is not bool:
        raise TypeError("funding evidence completeness must be explicit boolean")
    for values, item_type in ((before, BalanceSnapshot), (after, BalanceSnapshot),
                              (trades, AccountTrade), (funding, FundingPayment),
                              (adjustments, AccountAdjustment)):
        if not isinstance(values, tuple) or any(not isinstance(item, item_type) for item in values):
            raise TypeError("balance audit requires immutable, typed evidence tuples")
    thresholds = dict(tolerances or {})
    for asset, tolerance in thresholds.items():
        _asset(asset)
        require_decimal(tolerance, "balance comparison tolerance", nonnegative=True)
    issues: list[BalanceAuditIssue] = []
    start = {s.market: s for s in before}
    end = {s.market: s for s in after}
    if len(start) != len(before) or len(end) != len(after):
        issues.append(BalanceAuditIssue(BalanceAuditCode.DUPLICATE_SNAPSHOT))
    for market in ExecutionMarket:
        if market not in start or market not in end:
            issues.append(BalanceAuditIssue(BalanceAuditCode.MISSING_SNAPSHOT, market))
        elif start[market].received_at >= end[market].requested_at:
            issues.append(BalanceAuditIssue(BalanceAuditCode.INVALID_TIME_WINDOW, market))
    if not funding_evidence_complete:
        issues.append(BalanceAuditIssue(BalanceAuditCode.FUNDING_EVIDENCE_INCOMPLETE))
    totals: dict[tuple[ExecutionMarket, str], Decimal] = {}
    seen: set[tuple[str, ...]] = set()

    def event(identity: tuple[str, ...], market: ExecutionMarket, at: datetime) -> bool:
        if identity in seen:
            issues.append(BalanceAuditIssue(BalanceAuditCode.DUPLICATE_EVENT, market))
            return False
        seen.add(identity)
        if market not in start or market not in end:
            return False
        if not start[market].received_at < at <= end[market].requested_at:
            issues.append(BalanceAuditIssue(BalanceAuditCode.EVENT_OUTSIDE_WINDOW, market))
            return False
        return True

    def add(market: ExecutionMarket, asset: str, amount: Decimal) -> None:
        key = market, asset
        totals[key] = totals.get(key, Decimal("0")) + amount

    for trade in trades:
        if not event(("trade", trade.market.value, trade.symbol, trade.trade_id), trade.market, trade.traded_at):
            continue
        if trade.quote_quantity != trade.quantity * trade.price:
            issues.append(BalanceAuditIssue(BalanceAuditCode.INVALID_TRADE_NOTIONAL, trade.market))
            continue
        if trade.market is ExecutionMarket.SPOT:
            sign = Decimal("1") if trade.side is Side.BUY else Decimal("-1")
            add(trade.market, trade.symbol[:-4], sign * trade.quantity)
            add(trade.market, "USDT", -sign * trade.quote_quantity)
        else:
            assert trade.official_perp_pnl is not None
            add(trade.market, "USDT", trade.official_perp_pnl)
        if trade.commission != 0:
            add(trade.market, trade.commission_asset, -trade.commission)
    for payment in funding:
        if event(("funding", payment.transaction_id), ExecutionMarket.USDTM, payment.posted_at):
            add(ExecutionMarket.USDTM, payment.asset, payment.amount)
    for adjustment in adjustments:
        if event(("adjustment", adjustment.market.value, adjustment.event_id), adjustment.market, adjustment.posted_at):
            add(adjustment.market, adjustment.asset, adjustment.amount)
    initial = {(s.market, b.asset): b.amount for s in before for b in s.balances}
    final = {(s.market, b.asset): b.amount for s in after for b in s.balances}
    movements = []
    for market, asset in sorted(initial.keys() | final.keys() | totals.keys()):
        key = market, asset
        known = totals.get(key, Decimal("0"))
        tolerance = thresholds.get(asset, Decimal("0"))
        if key not in initial or key not in final:
            observed = residual = None
            issues.append(BalanceAuditIssue(BalanceAuditCode.MISSING_BALANCE, market, asset))
        else:
            observed = final[key] - initial[key]
            residual = observed - known
            if abs(residual) > tolerance:
                issues.append(BalanceAuditIssue(BalanceAuditCode.UNEXPLAINED_BALANCE_CHANGE, market, asset))
        movements.append(AssetMovement(market, asset, known, observed, residual, tolerance))
    codes = {i.code for i in issues}
    invalid = {BalanceAuditCode.DUPLICATE_SNAPSHOT, BalanceAuditCode.INVALID_TIME_WINDOW,
               BalanceAuditCode.EVENT_OUTSIDE_WINDOW, BalanceAuditCode.DUPLICATE_EVENT,
               BalanceAuditCode.INVALID_TRADE_NOTIONAL}
    incomplete = {BalanceAuditCode.MISSING_SNAPSHOT, BalanceAuditCode.MISSING_BALANCE,
                  BalanceAuditCode.FUNDING_EVIDENCE_INCOMPLETE}
    status = (BalanceAuditStatus.INVALID if codes & invalid else BalanceAuditStatus.INCOMPLETE
              if codes & incomplete else BalanceAuditStatus.MISMATCH if codes else BalanceAuditStatus.MATCHED)
    return BalanceAudit(status, tuple(dict.fromkeys(issues)), tuple(movements), before, after,
                        trades, funding, adjustments, funding_evidence_complete)
