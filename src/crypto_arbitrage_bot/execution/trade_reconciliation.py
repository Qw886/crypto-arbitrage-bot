"""Read-only test-account fill accounting, independent of forecasts/decisions.

RECONCILED covers the listed fills and funding, not the whole account balance.
Missing funding, unvalued fees, inventory differences or unknown orders leave
net profit NULL. No request in this module can change exposure.
"""

import json
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

from crypto_arbitrage_bot.arbitrage.fees import FourLegFees
from crypto_arbitrage_bot.arbitrage.profit import net_profit, net_return, round_trip_from_fills
from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState
from crypto_arbitrage_bot.execution.testnet_client import TestnetExecutionError
from crypto_arbitrage_bot.execution.types import (
    AccountTrade,
    ExecutionMarket,
    FundingPayment,
    IocLimitOrder,
    OrderState,
)
from crypto_arbitrage_bot.models.calculation import (
    CalculationResult,
    FillSlice,
    FillStatus,
    SimulatedFill,
    TradeLeg,
)
from crypto_arbitrage_bot.models.market import Side
from crypto_arbitrage_bot.utils.decimal_math import (
    decimal_calculation,
    require_decimal,
    utc_timestamp,
)

if TYPE_CHECKING:
    from crypto_arbitrage_bot.execution.runner import TestnetBot


class ReconciliationStatus(StrEnum):
    RECONCILED = "RECONCILED"
    INCOMPLETE = "INCOMPLETE"
    INVALID = "INVALID"


class ReconciliationCode(StrEnum):
    ATTEMPT_NOT_CLOSED = "ATTEMPT_NOT_CLOSED"
    ORDER_UNRESOLVED = "ORDER_UNRESOLVED"
    API_UNAVAILABLE = "API_UNAVAILABLE"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    DUPLICATE_TRADE = "DUPLICATE_TRADE"
    INCOMPLETE_FILLS = "INCOMPLETE_FILLS"
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"
    TRADE_NOTIONAL_MISMATCH = "TRADE_NOTIONAL_MISMATCH"
    TRADE_TIME_MISMATCH = "TRADE_TIME_MISMATCH"
    UNVALUED_COMMISSION = "UNVALUED_COMMISSION"
    REBATE_REQUIRES_ACCOUNTING = "REBATE_REQUIRES_ACCOUNTING"
    SPOT_INVENTORY_NOT_FLAT = "SPOT_INVENTORY_NOT_FLAT"
    OFFICIAL_PERP_PNL_MISMATCH = "OFFICIAL_PERP_PNL_MISMATCH"
    FUNDING_NOT_CONFIRMED = "FUNDING_NOT_CONFIRMED"
    FUNDING_ATTRIBUTION_CONFLICT = "FUNDING_ATTRIBUTION_CONFLICT"
    SETTLEMENT_NOT_CROSSED = "SETTLEMENT_NOT_CROSSED"


@dataclass(frozen=True)
class ReconciliationIssue:
    code: ReconciliationCode
    leg: TradeLeg | None = None
    error_kind: str | None = None
    api_code: int | None = None


@dataclass(frozen=True)
class LegTradeEvidence:
    leg: TradeLeg
    trades: tuple[AccountTrade, ...]
    quantity: Decimal
    quote_quantity: Decimal
    vwap: Decimal | None
    commissions: tuple[tuple[str, Decimal], ...]


@dataclass(frozen=True)
class TradeReconciliation:
    attempt_id: str
    symbol: str
    checked_at: datetime
    status: ReconciliationStatus
    issues: tuple[ReconciliationIssue, ...]
    legs: tuple[LegTradeEvidence, ...] = ()
    funding_payments: tuple[FundingPayment, ...] = ()
    funding_total: Decimal | None = None
    official_perp_pnl: Decimal | None = None
    spot_inventory_change: Decimal | None = None
    fees: FourLegFees | None = None
    price_pnl: Decimal | None = None
    spot_pnl: Decimal | None = None
    perp_pnl: Decimal | None = None
    calculation: CalculationResult | None = None
    capital: Decimal | None = None
    net_return: Decimal | None = None
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    accounting_scope: str = "TESTNET_ORDER_FILLS_AND_FUNDING_V1"


@decimal_calculation
def reconcile_closed_attempt(bot: "TestnetBot", attempt_id: str, *, capital: Decimal | None = None,
                             other_costs: Decimal = Decimal("0"),
                             pnl_tolerance: Decimal = Decimal("0.00000001")) -> TradeReconciliation:
    """Use actual four-leg cash flow. Capital, if supplied, is a stated budget.

    Quote-asset fees are fully supported. Other commission assets/rebates or
    unequal gross quantities retain evidence but require separate inventory
    accounting, rather than silently treating assets as USDT or neutral.
    """
    if capital is not None:
        require_decimal(capital, "explicit capital", positive=True)
    require_decimal(other_costs, "explicit other costs", nonnegative=True)
    require_decimal(pnl_tolerance, "official PnL comparison tolerance", nonnegative=True)
    plans = [plan for plan, _ in bot.journal.plans() if plan.attempt_id == attempt_id]
    if len(plans) != 1:
        raise ValueError("one existing execution attempt is required")
    plan = plans[0]
    current = bot.journal.get(attempt_id)
    result = TradeReconciliation(attempt_id, plan.symbol, utc_timestamp(bot.clock()),
                                 ReconciliationStatus.INCOMPLETE, (), capital=capital)
    if current is None or current.state is not PairExecutionState.CLOSED:
        return replace(result, issues=(ReconciliationIssue(ReconciliationCode.ATTEMPT_NOT_CLOSED),))
    issues: list[ReconciliationIssue] = []
    orders = {order.client_order_id: (leg, order) for leg, order in (
        (TradeLeg.SPOT_ENTRY, plan.spot_entry), (TradeLeg.SPOT_EXIT, plan.spot_exit),
        (TradeLeg.PERP_ENTRY, plan.perp_entry), (TradeLeg.PERP_EXIT, plan.perp_exit))}
    # Actual submitted quantity can differ from the planned gross exit after a
    # base-asset commission. Preserve the inventory discrepancy below.
    for intent in bot.journal.intents(attempt_id):
        if intent.client_order_id not in orders:
            issues.append(ReconciliationIssue(ReconciliationCode.IDENTITY_MISMATCH))
        else:
            leg, expected = orders[intent.client_order_id]
            if (intent.symbol, intent.market, intent.side) != (expected.symbol, expected.market, expected.side):
                issues.append(ReconciliationIssue(ReconciliationCode.IDENTITY_MISMATCH, leg))
            orders[intent.client_order_id] = (leg, intent)
    # Include ALL recovery exits, not just the last receipt stored on the pair.
    with bot.journal._connect() as db:
        for table, market, side, leg, reduce_only in (
            ('testnet_exit_recoveries', ExecutionMarket.USDTM, Side.BUY, TradeLeg.PERP_EXIT, True),
            ('testnet_spot_exit_recoveries', ExecutionMarket.SPOT, Side.SELL, TradeLeg.SPOT_EXIT, False),
        ):
            if not db.execute('SELECT 1 FROM sqlite_master WHERE name=?', (table,)).fetchone():
                continue
            # Fixed table names only. Add actual repair sources to the same
            # existing four grouped legs; all financial formulas below remain.
            rows = db.execute(f'SELECT order_json FROM {table} WHERE attempt_id=? ORDER BY ordinal',
                              (attempt_id,)).fetchall()
            for row in rows:
                try:
                    data = json.loads(row[0])
                    if (data['symbol'] != plan.symbol or data['market'] != market.value
                            or data['side'] != side.value or data['reduce_only'] is not reduce_only
                            or data['client_order_id'] in orders or type(data['quantity']) is not str
                            or type(data['limit_price']) is not str):
                        raise ValueError('recovery source identity mismatch')
                    recovery = IocLimitOrder(market, plan.symbol, side, Decimal(data['quantity']),
                        Decimal(data['limit_price']), data['client_order_id'], reduce_only)
                except (ValueError, KeyError, TypeError, ArithmeticError):
                    issues.append(ReconciliationIssue(ReconciliationCode.IDENTITY_MISMATCH, leg))
                    continue
                orders[recovery.client_order_id] = (leg, recovery)
    grouped: dict[TradeLeg, list[AccountTrade]] = {leg: [] for leg in TradeLeg}
    seen: set[tuple[ExecutionMarket, str]] = set()
    terminal = {OrderState.FILLED, OrderState.CANCELED, OrderState.EXPIRED,
                OrderState.EXPIRED_IN_MATCH, OrderState.REJECTED}
    for leg, order in orders.values():
        client = bot.spot if order.market is ExecutionMarket.SPOT else bot.perp
        try:
            receipt = client.get_order(plan.symbol, order.client_order_id, include_base_commission=False)
            if not receipt.matches_order(order):
                issues.append(ReconciliationIssue(ReconciliationCode.IDENTITY_MISMATCH, leg))
                continue
            if receipt.status not in terminal:
                issues.append(ReconciliationIssue(ReconciliationCode.ORDER_UNRESOLVED, leg))
                continue
            trades = client.get_order_trades(plan.symbol, receipt.order_id) if receipt.executed_quantity else ()
            if sum((trade.quantity for trade in trades), Decimal("0")) != receipt.executed_quantity:
                issues.append(ReconciliationIssue(ReconciliationCode.INCOMPLETE_FILLS, leg))
            for trade in trades:
                if (trade.market is not order.market or trade.symbol != plan.symbol
                        or trade.order_id != receipt.order_id or trade.side is not order.side):
                    issues.append(ReconciliationIssue(ReconciliationCode.IDENTITY_MISMATCH, leg))
                    continue
                key = (trade.market, trade.trade_id)
                if key in seen:
                    issues.append(ReconciliationIssue(ReconciliationCode.DUPLICATE_TRADE, leg))
                    continue
                seen.add(key)
                grouped[leg].append(trade)
                if trade.traded_at > result.checked_at:
                    issues.append(ReconciliationIssue(ReconciliationCode.TRADE_TIME_MISMATCH, leg))
                if trade.quote_quantity != trade.price * trade.quantity:
                    issues.append(ReconciliationIssue(ReconciliationCode.TRADE_NOTIONAL_MISMATCH, leg))
        except TestnetExecutionError as error:
            issues.append(ReconciliationIssue(ReconciliationCode.API_UNAVAILABLE, leg, error.kind.value, error.api_code))
    evidence: list[LegTradeEvidence] = []
    for leg, trade_list in grouped.items():
        qty = sum((trade.quantity for trade in trade_list), Decimal("0"))
        quote = sum((trade.quote_quantity for trade in trade_list), Decimal("0"))
        commissions: dict[str, Decimal] = {}
        for trade in trade_list:
            commissions[trade.commission_asset] = commissions.get(trade.commission_asset, Decimal("0")) + trade.commission
            if trade.commission < 0:
                issues.append(ReconciliationIssue(ReconciliationCode.REBATE_REQUIRES_ACCOUNTING, leg))
            if trade.commission != 0 and trade.commission_asset != "USDT":
                issues.append(ReconciliationIssue(ReconciliationCode.UNVALUED_COMMISSION, leg))
        evidence.append(LegTradeEvidence(leg, tuple(trade_list), qty, quote,
            None if qty == 0 else quote / qty, tuple(sorted(commissions.items()))))
    by_leg = {leg.leg: leg for leg in evidence}
    se, sx, pe, px = (by_leg[leg] for leg in (
        TradeLeg.SPOT_ENTRY, TradeLeg.SPOT_EXIT, TradeLeg.PERP_ENTRY, TradeLeg.PERP_EXIT))
    base_fee = sum((trade.commission for leg in (se, sx) for trade in leg.trades
                    if trade.commission_asset == plan.symbol[:-4]), Decimal("0"))
    inventory = se.quantity - sx.quantity - base_fee
    if inventory != 0:
        issues.append(ReconciliationIssue(ReconciliationCode.SPOT_INVENTORY_NOT_FLAT))
    if se.quantity <= 0 or len({leg.quantity for leg in evidence}) != 1 or se.quantity != plan.spot_entry.quantity:
        issues.append(ReconciliationIssue(ReconciliationCode.QUANTITY_MISMATCH))
    funding: tuple[FundingPayment, ...] = ()
    funding_total: Decimal | None = None
    entry_time = min((trade.traded_at for trade in pe.trades), default=None)
    exit_time = max((trade.traded_at for trade in px.trades), default=None)
    for entry, exit_leg in ((se, sx), (pe, px)):
        if entry.trades and exit_leg.trades and max(t.traded_at for t in entry.trades) > min(t.traded_at for t in exit_leg.trades):
            issues.append(ReconciliationIssue(ReconciliationCode.TRADE_TIME_MISMATCH, exit_leg.leg))
    if entry_time is not None and exit_time is not None and entry_time < exit_time:
        try:
            funding = bot.perp.get_funding_payments(plan.symbol, entry_time, exit_time)
            if funding:
                if (len({payment.transaction_id for payment in funding}) != len(funding)
                        or any(payment.symbol != plan.symbol or payment.asset != "USDT"
                               or not entry_time < payment.posted_at <= exit_time for payment in funding)):
                    issues.append(ReconciliationIssue(ReconciliationCode.FUNDING_ATTRIBUTION_CONFLICT))
                with bot.journal._connect() as db:
                    if db.execute("SELECT 1 FROM sqlite_master WHERE name='testnet_funding_payments'").fetchone():
                        for payment in funding:
                            owner = db.execute("SELECT attempt_id FROM testnet_funding_payments WHERE transaction_id=?",
                                               (payment.transaction_id,)).fetchone()
                            if owner is not None and owner[0] != attempt_id:
                                issues.append(ReconciliationIssue(ReconciliationCode.FUNDING_ATTRIBUTION_CONFLICT))
                funding_total = sum((payment.amount for payment in funding), Decimal("0"))
            else:
                issues.append(ReconciliationIssue(ReconciliationCode.FUNDING_NOT_CONFIRMED))
        except TestnetExecutionError as error:
            issues.append(ReconciliationIssue(ReconciliationCode.API_UNAVAILABLE, error_kind=error.kind.value, api_code=error.api_code))
        except ValueError:
            issues.append(ReconciliationIssue(ReconciliationCode.FUNDING_NOT_CONFIRMED))
    else:
        issues.append(ReconciliationIssue(ReconciliationCode.FUNDING_NOT_CONFIRMED))
    if plan.funding_settlement_at is not None and not (
        pe.trades and px.trades and max(t.traded_at for t in pe.trades) < plan.funding_settlement_at
        <= min(t.traded_at for t in px.trades)
    ):
        issues.append(ReconciliationIssue(ReconciliationCode.SETTLEMENT_NOT_CROSSED))
    official = sum((trade.official_perp_pnl for leg in (pe, px) for trade in leg.trades
                    if trade.official_perp_pnl is not None), Decimal("0")) if pe.trades and px.trades else None
    fees = None
    calculation = None
    price_component = spot_component = perp_component = None
    fill_blockers = [issue for issue in issues if issue.code is not ReconciliationCode.FUNDING_NOT_CONFIRMED
                    and not (issue.code is ReconciliationCode.API_UNAVAILABLE and issue.leg is None)]
    if not fill_blockers:
        # These are math adapters only; original exchange fills retain their
        # chronological timestamps above. Sorting satisfies the existing fill
        # contract without inventing a book or altering the executed prices.
        fills: dict[TradeLeg, SimulatedFill] = {}
        for leg_evidence in evidence:
            sorted_trades = sorted(leg_evidence.trades, key=lambda trade: trade.price, reverse=leg_evidence.leg.side is Side.SELL)
            fills[leg_evidence.leg] = SimulatedFill(plan.symbol, leg_evidence.leg, leg_evidence.quantity, leg_evidence.quantity,
                leg_evidence.quote_quantity, leg_evidence.vwap, FillStatus.FILLED,
                tuple(FillSlice(t.price, t.quantity) for t in sorted_trades), sorted_trades[0].price)
        fees = FourLegFees(*(dict(by_leg[leg].commissions).get("USDT", Decimal("0")) for leg in (
            TradeLeg.SPOT_ENTRY, TradeLeg.SPOT_EXIT, TradeLeg.PERP_ENTRY, TradeLeg.PERP_EXIT)))
        # Extract the known price component even while funding is unavailable.
        # This intermediate value is NOT published as a zero-funding net profit.
        components = round_trip_from_fills(fills[TradeLeg.SPOT_ENTRY], fills[TradeLeg.SPOT_EXIT],
            fills[TradeLeg.PERP_ENTRY], fills[TradeLeg.PERP_EXIT], fees, Decimal("0"), other_costs)
        expected_perp_pnl = pe.quote_quantity - px.quote_quantity
        assert official is not None
        if abs(official - expected_perp_pnl) > pnl_tolerance:
            issues.append(ReconciliationIssue(ReconciliationCode.OFFICIAL_PERP_PNL_MISMATCH))
        else:
            price_component = components.price_pnl
            spot_component = sx.quote_quantity - se.quote_quantity
            perp_component = expected_perp_pnl
            if not issues and funding_total is not None:
                calculation = replace(components, funding_total=funding_total,
                    net_profit=net_profit(components.price_pnl, funding_total, fees, other_costs))
    issues = list(dict.fromkeys(issues))
    invalid = {ReconciliationCode.IDENTITY_MISMATCH, ReconciliationCode.DUPLICATE_TRADE,
               ReconciliationCode.TRADE_NOTIONAL_MISMATCH, ReconciliationCode.FUNDING_ATTRIBUTION_CONFLICT,
               ReconciliationCode.OFFICIAL_PERP_PNL_MISMATCH, ReconciliationCode.TRADE_TIME_MISMATCH}
    status = ReconciliationStatus.RECONCILED if not issues else (
        ReconciliationStatus.INVALID if any(issue.code in invalid for issue in issues) else ReconciliationStatus.INCOMPLETE)
    return replace(result, status=status, issues=tuple(issues), legs=tuple(evidence), funding_payments=funding,
        funding_total=funding_total, official_perp_pnl=official, spot_inventory_change=inventory, fees=fees,
        price_pnl=price_component, spot_pnl=spot_component, perp_pnl=perp_component,
        calculation=calculation, entry_time=entry_time, exit_time=exit_time,
        net_return=None if calculation is None or capital is None else net_return(calculation.net_profit, capital))
