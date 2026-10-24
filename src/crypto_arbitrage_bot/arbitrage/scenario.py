"""Offline exit hypotheses composed with estimate_once; no alternative PnL."""

from decimal import Decimal, Inexact, localcontext

from crypto_arbitrage_bot.arbitrage.basis import mid_basis, mid_price
from crypto_arbitrage_bot.arbitrage.estimate import estimate_once
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.models.estimate import EstimateConfig, ExitFundingScenario
from crypto_arbitrage_bot.models.market import OrderBook, OrderBookLevel, Timestamp
from crypto_arbitrage_bot.models.scenario import (
    ExitBasisAssumption,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioType,
)
from crypto_arbitrage_bot.models.snapshot import SnapshotAssembly
from crypto_arbitrage_bot.utils.decimal_math import decimal_calculation


def _mid(book: OrderBook) -> Decimal:
    if book.best_bid is None or book.best_ask is None:
        raise ValueError("both sides are required to define a mid-basis assumption")
    return mid_price(book.best_bid, book.best_ask)


def reverse_execution_scenario(
    *, name: str, spot_entry_book: OrderBook, perp_entry_book: OrderBook,
    opened_at: Timestamp, closed_at: Timestamp, funding_events: tuple[FundingEvent, ...],
    description: str,
) -> ScenarioDefinition:
    """Explicit unchanged-book baseline, including reverse-side execution costs.

    The very same books and original timestamps are retained as assumptions;
    they are not represented as newly observed data at the closing time.
    """
    return ScenarioDefinition(
        ScenarioType.NO_CONVERGENCE, name,
        ExitFundingScenario(
            spot_entry_book, perp_entry_book, opened_at, closed_at, funding_events, description,
        ),
    )


def _translate_book(book: OrderBook, target_mid: Decimal, closed_at: Timestamp) -> OrderBook:
    offset = target_mid - _mid(book)
    return OrderBook(
        symbol=book.symbol, market=book.market,
        bids=tuple(OrderBookLevel(level.price + offset, level.quantity) for level in book.bids),
        asks=tuple(OrderBookLevel(level.price + offset, level.quantity) for level in book.asks),
        observed_at=closed_at, event_time=None,
        research_instrument=book.research_instrument,
    )


@decimal_calculation
def target_basis_scenario(
    *, scenario_type: ScenarioType, name: str, assumption: ExitBasisAssumption,
    opened_at: Timestamp, closed_at: Timestamp, funding_events: tuple[FundingEvent, ...],
    description: str,
) -> ScenarioDefinition:
    """Translate explicit depth templates to an explicit exit midpoint/basis.

    Target perp mid = spot_exit_mid * (1 + exit_basis_pct). This only constructs
    hypothetical prices; execution, fees, funding and PnL remain in estimate_once.
    No target, holding time, liquidity or settlement event has an implicit default.
    Synthetic observed_at is the scenario closing time, not a market observation;
    event_time remains None. Original templates are retained in the definition.
    Invalid prices or unrepresentable translations fail without producing profit.
    """
    if not isinstance(assumption, ExitBasisAssumption):
        raise TypeError("assumption must be ExitBasisAssumption")
    try:
        with localcontext() as context:
            context.traps[Inexact] = True
            perp_mid = assumption.spot_exit_mid * (Decimal("1") + assumption.exit_basis_pct)
            spot_exit = _translate_book(
                assumption.spot_depth_template, assumption.spot_exit_mid, closed_at,
            )
            perp_exit = _translate_book(assumption.perp_depth_template, perp_mid, closed_at)
    except Inexact as exc:
        raise ValueError("exit price translation exceeds supported Decimal precision") from exc
    return ScenarioDefinition(
        scenario_type, name,
        ExitFundingScenario(
            spot_exit, perp_exit, opened_at, closed_at, funding_events, description,
        ),
        assumption,
    )


def evaluate_scenario(
    assembly: SnapshotAssembly, config: EstimateConfig, definition: ScenarioDefinition,
    *, evaluated_at: Timestamp | None = None,
) -> ScenarioResult:
    """Delegate once to the existing quality/estimate path and preserve its result."""
    if not isinstance(definition, ScenarioDefinition):
        raise TypeError("definition must be ScenarioDefinition")
    estimate = estimate_once(assembly, config, definition.exit_scenario, evaluated_at=evaluated_at)
    entry_mid_basis = None
    exit_mid_basis = None
    if estimate.complete:
        snapshot = assembly.snapshot
        assert snapshot is not None
        entry_mid_basis = mid_basis(_mid(snapshot.spot_book), _mid(snapshot.perp_book))
        exit_mid_basis = mid_basis(
            _mid(definition.exit_scenario.spot_exit_book),
            _mid(definition.exit_scenario.perp_exit_book),
        )
    return ScenarioResult(definition, estimate, entry_mid_basis, exit_mid_basis)


def evaluate_scenarios(
    assembly: SnapshotAssembly, config: EstimateConfig, definitions: tuple[ScenarioDefinition, ...],
    *, evaluated_at: Timestamp | None = None,
) -> tuple[ScenarioResult, ...]:
    """Evaluate a finite, explicit tuple in order; never scan, rank or select one."""
    if not isinstance(definitions, tuple) or any(
        not isinstance(item, ScenarioDefinition) for item in definitions
    ):
        raise TypeError("definitions must be a tuple of ScenarioDefinition")
    if len({item.name for item in definitions}) != len(definitions):
        raise ValueError("scenario names must be unique within a comparison")
    return tuple(
        evaluate_scenario(assembly, config, item, evaluated_at=evaluated_at)
        for item in definitions
    )
