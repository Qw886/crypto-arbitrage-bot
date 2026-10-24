"""One offline estimate composed exclusively from the existing math engine."""

from dataclasses import replace
from decimal import Decimal, Inexact, localcontext

from crypto_arbitrage_bot.arbitrage.basis import basis_pct, executable_entry_basis
from crypto_arbitrage_bot.arbitrage.execution import simulate_fill
from crypto_arbitrage_bot.arbitrage.fees import FourLegFees, calculate_fee
from crypto_arbitrage_bot.arbitrage.funding import accrue_short_funding
from crypto_arbitrage_bot.arbitrage.profit import (
    decompose_fills,
    net_return,
    occupied_capital,
    round_trip_from_fills,
)
from crypto_arbitrage_bot.arbitrage.sizing import size_position
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.models.calculation import TradeLeg
from crypto_arbitrage_bot.models.estimate import (
    EstimateConfig,
    EstimateStatus,
    ExitFundingScenario,
    OfflineEstimate,
)
from crypto_arbitrage_bot.models.market import BidAsk, OrderBook, Timestamp
from crypto_arbitrage_bot.models.snapshot import SnapshotAssembly
from crypto_arbitrage_bot.utils.decimal_math import decimal_calculation

ZERO = Decimal("0")


def _quote(book: OrderBook) -> BidAsk:
    if book.best_bid is None or book.best_ask is None:
        raise ValueError("both bid and ask are required for cost attribution")
    return BidAsk(book.best_bid, book.best_ask)


@decimal_calculation
def estimate_once(
    assembly: SnapshotAssembly, config: EstimateConfig, scenario: ExitFundingScenario,
    *, evaluated_at: Timestamp | None = None,
) -> OfflineEstimate:
    """Run one constant-quantity, four-leg, fully reserved research scenario.

    No network, filesystem, current clock, forecast, or implicit settlement.
    Data quality is always checked before math. The evaluation time defaults
    to the explicit scenario opening time for offline replay; callers can pass
    evaluated_at to evaluate freshness at another instant. There is no bypass.
    Existing sizing proposes a common lot at entry quotes; actual spot budget
    and total capital are checked after book execution. No automatic resizing.
    """
    if not isinstance(assembly, SnapshotAssembly):
        raise TypeError("assembly must be SnapshotAssembly")
    if not isinstance(config, EstimateConfig) or not isinstance(scenario, ExitFundingScenario):
        raise TypeError("config and scenario must be explicit estimate inputs")
    result = OfflineEstimate(EstimateStatus.INCOMPLETE_SNAPSHOT, assembly, config, scenario)
    quality = validate_market_data(
        assembly, config.data_quality,
        evaluated_at=scenario.opened_at if evaluated_at is None else evaluated_at,
    )
    result = replace(result, data_quality=quality)
    if not assembly.complete:
        return replace(result, issues=("COMPLETE_SNAPSHOT_REQUIRED",))
    if not quality.valid:
        return replace(
            result, status=EstimateStatus.INVALID_MARKET_DATA,
            issues=tuple(code.value for code in quality.reason_codes),
        )
    snapshot = assembly.snapshot
    assert snapshot is not None
    if assembly.symbol != snapshot.symbol or scenario.spot_exit_book.symbol != snapshot.symbol:
        raise ValueError("entry snapshot and exit scenario must have the same symbol")
    identity = assembly.research_instrument
    if any(book.research_instrument != identity for book in (
        snapshot.spot_book, snapshot.perp_book, scenario.spot_exit_book, scenario.perp_exit_book,
    )) or snapshot.research_instrument != identity:
        raise ValueError("entry and exit books must preserve the same research identity")
    spot_quote = _quote(snapshot.spot_book)
    perp_quote = _quote(snapshot.perp_book)
    reference_budget = config.spot_budget
    if reference_budget is None:
        assert config.target_quantity is not None
        try:
            with localcontext() as context:
                context.traps[Inexact] = True
                reference_budget = config.target_quantity * spot_quote.ask
        except Inexact as exc:
            raise ValueError("target reference notional exceeds supported Decimal precision") from exc
    sizing = size_position(
        reference_budget, spot_quote.ask, perp_quote.bid, config.spot_rules, config.perp_rules,
    )
    result = replace(result, status=EstimateStatus.SIZING_REJECTED, sizing=sizing)
    if not sizing.acceptable:
        return replace(result, issues=sizing.reasons)

    books = (
        snapshot.spot_book, scenario.spot_exit_book,
        snapshot.perp_book, scenario.perp_exit_book,
    )
    legs = (TradeLeg.SPOT_ENTRY, TradeLeg.SPOT_EXIT, TradeLeg.PERP_ENTRY, TradeLeg.PERP_EXIT)
    fills = tuple(
        simulate_fill(book, leg, sizing.perp_quantity)
        for book, leg in zip(books, legs, strict=True)
    )
    result = replace(result, status=EstimateStatus.INSUFFICIENT_DEPTH, fills=fills)
    insufficient = tuple(fill.leg.value for fill in fills if not fill.complete)
    if insufficient:
        return replace(result, issues=insufficient)
    spot_entry, spot_exit, perp_entry, perp_exit = fills
    actual_constraints = tuple(
        f"{fill.leg.value}_MIN_NOTIONAL_AT_FILL"
        for fill, rules in zip(
            fills, (config.spot_rules, config.spot_rules, config.perp_rules, config.perp_rules),
            strict=True,
        )
        if fill.quote_notional < rules.min_notional
    )
    if actual_constraints:
        return replace(result, status=EstimateStatus.SIZING_REJECTED, issues=actual_constraints)
    if config.spot_budget is not None and spot_entry.quote_notional > config.spot_budget:
        return replace(result, status=EstimateStatus.BUDGET_EXCEEDED, issues=("SPOT_BUDGET_EXCEEDED",))

    rates = config.fee_rates
    # Invoke the existing fee function per actual slice, never multiply a
    # recurring, rounded VWAP back by quantity to determine fee notional.
    fees = FourLegFees(*(
        sum((calculate_fee(part.price, part.quantity, rate) for part in fill.fills), ZERO)
        for fill, rate in zip(
            fills, (rates.spot_entry, rates.spot_exit, rates.perp_entry, rates.perp_exit), strict=True,
        )
    ))
    capital = occupied_capital(
        spot_entry.quote_notional, perp_entry.quote_notional,
        fees.total, config.stress_reserve + config.other_costs,
    )
    result = replace(result, fees=fees, capital_occupied=capital)
    if config.capital_limit is not None and capital > config.capital_limit:
        return replace(result, status=EstimateStatus.BUDGET_EXCEEDED, issues=("CAPITAL_LIMIT_EXCEEDED",))
    funding = accrue_short_funding(
        sizing.perp_quantity, scenario.opened_at, scenario.closed_at,
        scenario.funding_events, snapshot.symbol,
    )
    calculation = round_trip_from_fills(
        spot_entry, spot_exit, perp_entry, perp_exit, fees, funding, config.other_costs,
        research_instrument=identity,
    )
    decomposition = decompose_fills(
        spot_entry, spot_exit, perp_entry, perp_exit,
        spot_entry_quote=spot_quote, spot_exit_quote=_quote(scenario.spot_exit_book),
        perp_entry_quote=perp_quote, perp_exit_quote=_quote(scenario.perp_exit_book),
        research_instrument=identity,
    )
    best_basis = executable_entry_basis(spot_quote.ask, perp_quote.bid)
    assert spot_entry.vwap is not None
    return replace(
        result, status=EstimateStatus.ESTIMATED, calculation=calculation,
        decomposition=decomposition, executable_entry_basis=best_basis,
        basis_pct=basis_pct(best_basis, spot_quote.ask),
        entry_vwap_basis_pct=basis_pct(calculation.entry_basis, spot_entry.vwap),
        net_return=net_return(calculation.net_profit, capital),
    )
