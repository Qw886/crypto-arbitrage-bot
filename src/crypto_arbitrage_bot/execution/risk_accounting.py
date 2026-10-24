"""Bind archived risk accounting to original fills, intents and cash-flow math.

No client, request, journal write, credential or alternative PnL formula.
"""

import json
import sqlite3
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from typing import Any

from crypto_arbitrage_bot.arbitrage.fees import FourLegFees
from crypto_arbitrage_bot.arbitrage.profit import net_return, round_trip_from_fills
from crypto_arbitrage_bot.execution.pair_coordinator import _receipt_from_json
from crypto_arbitrage_bot.execution.trade_reconciliation import LegTradeEvidence
from crypto_arbitrage_bot.execution.types import (
    AccountTrade,
    ExecutionMarket,
    FundingPayment,
    IocLimitOrder,
    OrderState,
)
from crypto_arbitrage_bot.models.calculation import FillSlice, FillStatus, SimulatedFill, TradeLeg
from crypto_arbitrage_bot.models.market import Side
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.decimal_math import (
    decimal_calculation,
    require_decimal,
    utc_timestamp,
)


def _decimal(value: Any) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("NATIVE_DECIMAL_TEXT_REQUIRED")
    return require_decimal(Decimal(value), "native accounting")


def _time(value: Any) -> datetime:
    return utc_timestamp(datetime.fromisoformat(value))


@decimal_calculation
def verify_native_accounting_projection(
    db: sqlite3.Connection,
    attempt: sqlite3.Row,
    report: dict[str, Any],
    trades: tuple[AccountTrade, ...],
    funding: tuple[FundingPayment, ...],
    *,
    finished_at: datetime,
) -> None:
    plan = json.loads(attempt["plan_json"])
    aid, symbol = attempt["attempt_id"], plan["spot_entry"]["symbol"]
    checked_at = _time(report["checked_at"])
    if report["symbol"] != symbol or not _time(attempt["updated_at"]) <= checked_at <= finished_at:
        raise ValueError("NATIVE_ACCOUNTING_IDENTITY_OR_TIME_DIFFERS")

    def summary(trade: AccountTrade) -> dict[str, Any]:
        value = asdict(trade)
        value.pop("raw_json")
        return value

    # ReconciliationRepository saves raw fills once in evidence_json and
    # intentionally omits them from report_json. Bind all numeric/identity
    # fields to that already verified source without demanding a second copy.
    native = {exact_json(summary(trade)): trade for trade in trades}
    if len(native) != len(trades):
        raise ValueError("DUPLICATE_NATIVE_ACCOUNTING_TRADE")
    legs: dict[TradeLeg, LegTradeEvidence] = {}
    used: list[str] = []
    for value in report["legs"]:
        leg = TradeLeg(value["leg"])
        if leg in legs:
            raise ValueError("DUPLICATE_NATIVE_ACCOUNTING_LEG")
        decoded_list = []
        for trade in value["trades"]:
            fields = dict(trade)
            raw = fields.pop("raw_json", None)
            key = exact_json(fields)
            if key not in native or (raw is not None and raw != native[key].raw_json):
                raise ValueError("NATIVE_ACCOUNTING_TRADE_DIFFERS")
            decoded_list.append(native[key])
        decoded = tuple(decoded_list)
        if any(
            t.market.value != leg.market.value
            or t.side is not leg.side
            or t.symbol != symbol
            or t.traded_at > checked_at
            for t in decoded
        ):
            raise ValueError("MISGROUPED_NATIVE_ACCOUNTING_TRADE")
        used.extend(exact_json(summary(t)) for t in decoded)
        quantity = sum((t.quantity for t in decoded), Decimal(0))
        quote = sum((t.quote_quantity for t in decoded), Decimal(0))
        commissions: dict[str, Decimal] = {}
        for trade in decoded:
            commissions[trade.commission_asset] = (
                commissions.get(trade.commission_asset, Decimal(0)) + trade.commission
            )
        projected = LegTradeEvidence(
            leg,
            decoded,
            quantity,
            quote,
            None if not quantity else quote / quantity,
            tuple(sorted(commissions.items())),
        )
        projection = asdict(projected)
        for expected_trade, archived_trade in zip(
            projection["trades"], value["trades"], strict=True
        ):
            if "raw_json" not in archived_trade:
                expected_trade.pop("raw_json")
        if exact_json(projection) != exact_json(value):
            raise ValueError("NATIVE_ACCOUNTING_LEG_PROJECTION_DIFFERS")
        legs[leg] = projected
    if set(legs) != set(TradeLeg) or Counter(used) != Counter(native.keys()):
        raise ValueError("NATIVE_ACCOUNTING_TRADE_COVERAGE_DIFFERS")

    # Every fill belongs to a terminal journal receipt and its durable intent,
    # including ALL repair orders rather than only the latest pair receipt.
    sources: dict[tuple[str, str], tuple[TradeLeg, str]] = {}
    repairs = {}
    for table, leg in (
        ("testnet_spot_exit_recoveries", TradeLeg.SPOT_EXIT),
        ("testnet_exit_recoveries", TradeLeg.PERP_EXIT),
    ):
        if db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
            for row in db.execute(
                f"SELECT order_json,receipt_json,created_at FROM {table} WHERE attempt_id=?", (aid,)
            ):
                order = json.loads(row["order_json"])
                cid = order["client_order_id"]
                if cid in repairs or cid in {
                    p["client_order_id"] for p in (plan[k.value.lower()] for k in TradeLeg)
                }:
                    raise ValueError("AMBIGUOUS_NATIVE_REPAIR_INTENT")
                repairs[cid] = (leg, order, row["receipt_json"], _time(row["created_at"]))
                if row["receipt_json"] is None:
                    raise ValueError("UNRESOLVED_NATIVE_REPAIR_INTENT")

    def source(leg: TradeLeg, payload: str | None, expected: dict[str, Any]) -> None:
        receipt = _receipt_from_json(payload)
        if receipt is None:
            if db.execute(
                "SELECT 1 FROM testnet_order_intents WHERE attempt_id=? AND client_order_id=?",
                (aid, expected["client_order_id"]),
            ).fetchone():
                raise ValueError("UNRESOLVED_NATIVE_ORDER_INTENT")
            return
        intent = db.execute(
            "SELECT order_json,created_at FROM testnet_order_intents WHERE attempt_id=? AND client_order_id=?",
            (aid, receipt.client_order_id),
        ).fetchall()
        # Repair intents are stored durably in their own tables.
        repair = repairs.get(receipt.client_order_id)
        order = (
            repair[1]
            if not intent and repair is not None and repair[0] is leg
            else (json.loads(intent[0]["order_json"]) if len(intent) == 1 else None)
        )
        if (
            order is None
            or receipt.symbol != symbol
            or receipt.market.value != leg.market.value
            or order["market"] != receipt.market.value
            or order["symbol"] != symbol
            or order["side"] != leg.side.value
            or order["client_order_id"] != receipt.client_order_id
            or order["reduce_only"] is not (leg is TradeLeg.PERP_EXIT)
            or (
                leg in (TradeLeg.SPOT_ENTRY, TradeLeg.PERP_ENTRY)
                and exact_json(order) != exact_json(expected)
            )
            or _decimal(order["quantity"]) != receipt.original_quantity
            or receipt.original_quantity > _decimal(plan[leg.value.lower()]["quantity"])
            or receipt.status
            not in {
                OrderState.FILLED,
                OrderState.EXPIRED,
                OrderState.CANCELED,
                OrderState.EXPIRED_IN_MATCH,
                OrderState.REJECTED,
            }
            or (
                receipt.status is OrderState.FILLED
                and receipt.executed_quantity != receipt.original_quantity
            )
        ):
            raise ValueError("NATIVE_RECEIPT_OR_INTENT_DIFFERS")
        actual = IocLimitOrder(
            ExecutionMarket(order["market"]),
            symbol,
            leg.side,
            _decimal(order["quantity"]),
            _decimal(order["limit_price"]),
            receipt.client_order_id,
            order["reduce_only"],
            submit_before=None
            if order.get("submit_before") is None
            else _time(order["submit_before"]),
        )
        matching = [
            t for t in trades if t.market is receipt.market and t.order_id == receipt.order_id
        ]
        if (
            any(t.side is not leg.side for t in matching)
            or any(
                t.price > actual.limit_price
                if leg.side is Side.BUY
                else t.price < actual.limit_price
                for t in matching
            )
            or sum((t.quantity for t in matching), Decimal(0)) != receipt.executed_quantity
            or (
                intent
                and matching
                and _time(intent[0]["created_at"]) > min(t.traded_at for t in matching)
            )
            or (repair is not None and matching and repair[3] > min(t.traded_at for t in matching))
            or (
                actual.submit_before is not None
                and intent
                and _time(intent[0]["created_at"]) >= actual.submit_before
            )
            or (
                receipt.cumulative_quote_quantity is not None
                and receipt.cumulative_quote_quantity
                != sum((t.quote_quantity for t in matching), Decimal(0))
            )
        ):
            raise ValueError("NATIVE_RECEIPT_FILL_DIFFERS")
        key = (receipt.market.value, receipt.order_id)
        projection = (leg, exact_json(receipt))
        if key in sources and sources[key] != projection:
            raise ValueError("AMBIGUOUS_NATIVE_RECEIPT")
        sources[key] = projection

    for leg in TradeLeg:
        source(leg, attempt[leg.value.lower()], plan[leg.value.lower()])
    for leg, order, payload, _ in repairs.values():
        source(leg, payload, order)
    if any((t.market.value, t.order_id) not in sources for t in trades):
        raise ValueError("NATIVE_FILL_WITHOUT_DURABLE_RECEIPT")
    for payment in funding:
        row = db.execute(
            "SELECT * FROM testnet_funding_payments WHERE transaction_id=?",
            (payment.transaction_id,),
        ).fetchone()
        if (
            row is None
            or row["attempt_id"] != aid
            or payment.symbol != symbol
            or payment.asset != "USDT"
            or row["symbol"] != symbol
            or _decimal(row["amount"]) != payment.amount
            or _time(row["posted_at"]) != payment.posted_at
        ):
            raise ValueError("NATIVE_FUNDING_OWNER_OR_AMOUNT_DIFFERS")

    # Aborted/partial entry currency evidence is not a four-leg profit.
    # Normal and fee/Funding cases require the full original cash-flow output.
    if attempt["state"] != "CLOSED":
        return
    if (
        report["status"] != "RECONCILED"
        or report["accounting_scope"] != "TESTNET_ORDER_FILLS_AND_FUNDING_V1"
        or report["issues"]
        or not funding
    ):
        raise ValueError("COMPLETE_NATIVE_CASH_FLOW_REQUIRED")
    se, sx, pe, px = (legs[leg] for leg in TradeLeg)
    if (
        se.quantity != _decimal(plan["spot_entry"]["quantity"])
        or not pe.trades
        or not px.trades
        or any(
            max(t.traded_at for t in entry.trades) > min(t.traded_at for t in exit_leg.trades)
            for entry, exit_leg in ((se, sx), (pe, px))
        )
        or any(
            not min(t.traded_at for t in pe.trades)
            < p.posted_at
            <= max(t.traded_at for t in px.trades)
            for p in funding
        )
        or (
            plan.get("funding_settlement_at") is not None
            and not max(t.traded_at for t in pe.trades)
            < _time(plan["funding_settlement_at"])
            <= min(t.traded_at for t in px.trades)
        )
    ):
        raise ValueError("NATIVE_CASH_FLOW_TIMING_OR_SIZE_DIFFERS")
    fills = []
    for leg in TradeLeg:
        evidence = legs[leg]
        if not evidence.trades or any(
            t.commission < 0 or (t.commission and t.commission_asset != "USDT")
            for t in evidence.trades
        ):
            raise ValueError("VALUED_NATIVE_FILLS_REQUIRED")
        ordered = sorted(evidence.trades, key=lambda t: t.price, reverse=leg.side is Side.SELL)
        fills.append(
            SimulatedFill(
                symbol,
                leg,
                evidence.quantity,
                evidence.quantity,
                evidence.quote_quantity,
                evidence.vwap,
                FillStatus.FILLED,
                tuple(FillSlice(t.price, t.quantity) for t in ordered),
                ordered[0].price,
            )
        )
    fees = FourLegFees(*(dict(legs[leg].commissions).get("USDT", Decimal(0)) for leg in TradeLeg))
    total = sum((p.amount for p in funding), Decimal(0))
    calculation = round_trip_from_fills(
        fills[0],
        fills[1],
        fills[2],
        fills[3],
        fees,
        total,
        _decimal(report["calculation"]["other_costs"]),
    )
    official = sum(
        (t.official_perp_pnl for t in (*pe.trades, *px.trades) if t.official_perp_pnl is not None),
        Decimal(0),
    )
    expected: dict[str, Any] = {
        "fees": fees,
        "funding_total": total,
        "official_perp_pnl": official,
        "spot_inventory_change": se.quantity - sx.quantity,
        "calculation": calculation,
        "price_pnl": calculation.price_pnl,
        "spot_pnl": sx.quote_quantity - se.quote_quantity,
        "perp_pnl": pe.quote_quantity - px.quote_quantity,
        "entry_time": min(t.traded_at for t in pe.trades),
        "exit_time": max(t.traded_at for t in px.trades),
        "net_return": None
        if report["capital"] is None
        else net_return(calculation.net_profit, _decimal(report["capital"])),
    }
    if abs(official - expected["perp_pnl"]) > Decimal("0.00000001") or any(
        exact_json(report[key]) != exact_json(value) for key, value in expected.items()
    ):
        raise ValueError("ORIGINAL_NATIVE_CASH_FLOW_PROJECTION_DIFFERS")
