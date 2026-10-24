"""Named exit/funding hypotheses and projections of the existing estimate."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from itertools import pairwise

from crypto_arbitrage_bot.models.estimate import ExitFundingScenario, OfflineEstimate
from crypto_arbitrage_bot.models.market import Market, OrderBook
from crypto_arbitrage_bot.utils.decimal_math import require_decimal


class ScenarioType(StrEnum):
    NO_CONVERGENCE = "NO_CONVERGENCE"
    CONVERGENCE = "CONVERGENCE"
    ADVERSE = "ADVERSE"


@dataclass(frozen=True)
class ExitBasisAssumption:
    """Explicit target mid prices with finite, caller-supplied depth templates.

    spot_exit_mid is USDT/base; exit_basis_pct is a decimal fraction relative
    to that spot midpoint. Every template level is translated by the same
    price offset within its market. Quantities, absolute spreads and level
    spacing are preserved; none of this assumes future executable liquidity.
    """

    spot_exit_mid: Decimal
    exit_basis_pct: Decimal
    spot_depth_template: OrderBook
    perp_depth_template: OrderBook

    def __post_init__(self) -> None:
        require_decimal(self.spot_exit_mid, "spot_exit_mid", positive=True)
        require_decimal(self.exit_basis_pct, "exit_basis_pct")
        if self.exit_basis_pct <= Decimal("-1"):
            raise ValueError("exit_basis_pct implies a non-positive perpetual midpoint")
        if not isinstance(self.spot_depth_template, OrderBook) or not isinstance(
            self.perp_depth_template, OrderBook,
        ):
            raise TypeError("depth templates must be OrderBook")
        if (
            self.spot_depth_template.market is not Market.SPOT
            or self.perp_depth_template.market is not Market.USDTM
            or self.spot_depth_template.symbol != self.perp_depth_template.symbol
        ):
            raise ValueError("templates must be Spot and USDT-M books of the same symbol")
        for book in (self.spot_depth_template, self.perp_depth_template):
            if book.best_bid is None or book.best_ask is None:
                raise ValueError("depth templates must have both bids and asks")
            if book.best_bid > book.best_ask:
                raise ValueError("depth templates must not be crossed")
            if any(left.price <= right.price for left, right in pairwise(book.bids)):
                raise ValueError("template bids must be strictly descending")
            if any(left.price >= right.price for left, right in pairwise(book.asks)):
                raise ValueError("template asks must be strictly ascending")


@dataclass(frozen=True)
class ScenarioDefinition:
    """A caller's named hypothesis, not a prediction or an enforced outcome.

    The type never selects prices or invents settlements. Direct construction
    supports any explicit ExitFundingScenario, including multiple depth levels.
    The optional target records how the target-basis helper built exit books.
    """

    scenario_type: ScenarioType
    name: str
    exit_scenario: ExitFundingScenario
    basis_assumption: ExitBasisAssumption | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_type, ScenarioType):
            raise TypeError("scenario_type must be ScenarioType")
        if not isinstance(self.name, str) or not self.name.strip() or self.name != self.name.strip():
            raise ValueError("name must be nonempty without outer whitespace")
        if not isinstance(self.exit_scenario, ExitFundingScenario):
            raise TypeError("exit_scenario must be ExitFundingScenario")
        if self.basis_assumption is not None and not isinstance(
            self.basis_assumption, ExitBasisAssumption,
        ):
            raise TypeError("basis_assumption must be ExitBasisAssumption")


@dataclass(frozen=True)
class ScenarioResult:
    """Keep the original estimate; all cash-flow fields are direct projections.

    entry_basis / assumed_exit_basis are execution VWAP differences in
    USDT/base, from the prior calculation. The separate mid-basis fields
    describe quote assumptions, never pretend midpoints were executable.
    Failed estimates retain their failure status and expose no profit/return.
    """

    definition: ScenarioDefinition
    estimate: OfflineEstimate
    entry_mid_basis: Decimal | None
    assumed_exit_mid_basis: Decimal | None

    @property
    def scenario_type(self) -> ScenarioType:
        return self.definition.scenario_type

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def assumptions(self) -> str:
        return self.definition.exit_scenario.description

    @property
    def complete(self) -> bool:
        return self.estimate.complete

    @property
    def entry_basis(self) -> Decimal | None:
        calculation = self.estimate.calculation
        return None if calculation is None else calculation.entry_basis

    @property
    def assumed_exit_basis(self) -> Decimal | None:
        calculation = self.estimate.calculation
        return None if calculation is None else calculation.exit_basis

    @property
    def matched_quantity(self) -> Decimal | None:
        return self.estimate.matched_quantity

    @property
    def price_pnl(self) -> Decimal | None:
        return self.estimate.price_pnl

    @property
    def funding_total(self) -> Decimal | None:
        return self.estimate.estimated_funding

    @property
    def spot_fees(self) -> Decimal | None:
        calculation = self.estimate.calculation
        return None if calculation is None else calculation.spot_fees

    @property
    def perp_fees(self) -> Decimal | None:
        calculation = self.estimate.calculation
        return None if calculation is None else calculation.perp_fees

    @property
    def other_costs(self) -> Decimal | None:
        calculation = self.estimate.calculation
        return None if calculation is None else calculation.other_costs

    @property
    def net_profit(self) -> Decimal | None:
        return self.estimate.net_profit

    @property
    def capital_occupied(self) -> Decimal | None:
        return self.estimate.capital_occupied

    @property
    def net_return(self) -> Decimal | None:
        return self.estimate.net_return
