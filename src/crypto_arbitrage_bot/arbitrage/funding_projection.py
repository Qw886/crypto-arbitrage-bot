"""Explicit single-settlement research hypothesis; not an accrued payment.

Only the exchange's next funding time is used. No interval extrapolation and
no hourly proration. Existing funding/PnL modules evaluate the supplied scenario.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, Inexact, localcontext

from crypto_arbitrage_bot.arbitrage.basis import mid_basis, mid_price
from crypto_arbitrage_bot.arbitrage.funding import FundingEvent
from crypto_arbitrage_bot.arbitrage.scenario import (
    _translate_book,
    reverse_execution_scenario,
    target_basis_scenario,
)
from crypto_arbitrage_bot.config.research import FundingProjectionSettings, ScenarioSettings
from crypto_arbitrage_bot.models.estimate import ExitFundingScenario
from crypto_arbitrage_bot.models.market import MarketSnapshot
from crypto_arbitrage_bot.models.scenario import (
    ExitBasisAssumption,
    ScenarioDefinition,
    ScenarioType,
)
from crypto_arbitrage_bot.utils.decimal_math import (
    decimal_calculation,
    require_decimal,
    utc_timestamp,
)


@dataclass(frozen=True)
class ProjectedFundingWindow:
    settlement_at: datetime
    closed_at: datetime
    baseline_events: tuple[FundingEvent, ...]
    stress_events: tuple[FundingEvent, ...]
    description: str


@decimal_calculation
def project_next_funding(snapshot: MarketSnapshot, opened_at: datetime,
                         config: FundingProjectionSettings) -> ProjectedFundingWindow:
    opened = utc_timestamp(opened_at)
    if snapshot.next_funding_time is None or snapshot.funding_rate is None or snapshot.mark_price is None:
        raise ValueError("FUNDING_PROJECTION_INPUT_MISSING")
    settlement = utc_timestamp(snapshot.next_funding_time)
    require_decimal(snapshot.funding_rate, "observed funding rate")
    require_decimal(snapshot.mark_price, "assumed settlement mark", positive=True)
    if not opened < settlement <= opened + timedelta(seconds=config.max_wait_seconds):
        raise ValueError("NEXT_SETTLEMENT_OUTSIDE_PROJECTION_WINDOW")
    # Discount income; never discount a negative funding cost toward zero.
    rate = snapshot.funding_rate
    if rate > 0:
        rate *= config.positive_rate_multiplier
    event_id = f"FORECAST_ONLY:{snapshot.symbol}:{settlement.isoformat()}"
    baseline = FundingEvent(event_id, settlement, snapshot.mark_price, rate, snapshot.symbol)
    stress = FundingEvent(event_id, settlement, snapshot.mark_price, min(rate, Decimal("0")), snapshot.symbol)
    description = (
        "FORECAST_ONLY: one hypothetical next settlement; NOT accrued or guaranteed income. "
        f"Observed rate={snapshot.funding_rate}; positive-rate multiplier={config.positive_rate_multiplier}; "
        f"assumed settlement mark={snapshot.mark_price}; official next funding time={settlement.isoformat()}. "
        "Stress removes positive funding income and retains negative funding cost."
    )
    return ProjectedFundingWindow(settlement, settlement + timedelta(seconds=config.exit_delay_seconds),
                                  (baseline,), (stress,), description)


@decimal_calculation
def projected_scenario_definitions(snapshot: MarketSnapshot, opened_at: datetime,
                                   settings: ScenarioSettings) -> tuple[ScenarioDefinition, ...]:
    """Compose existing scenario constructors with explicit forecast inputs."""
    if settings.funding_projection is None or settings.funding_events:
        raise ValueError("a projection policy without explicit settled events is required")
    window = project_next_funding(snapshot, opened_at, settings.funding_projection)
    if (snapshot.spot_book.best_bid is None or snapshot.spot_book.best_ask is None
            or snapshot.perp_book.best_bid is None or snapshot.perp_book.best_ask is None):
        raise ValueError("PROJECTED_SCENARIO_DEPTH_MISSING")
    spot_mid = mid_price(snapshot.spot_book.best_bid, snapshot.spot_book.best_ask)
    perp_mid = mid_price(snapshot.perp_book.best_bid, snapshot.perp_book.best_ask)
    exit_spot_mid = spot_mid if settings.spot_exit_mid_mode == "entry_mid" else settings.spot_exit_mid
    assert exit_spot_mid is not None
    baseline = reverse_execution_scenario(
        name="baseline", spot_entry_book=snapshot.spot_book, perp_entry_book=snapshot.perp_book,
        opened_at=opened_at, closed_at=window.closed_at, funding_events=window.baseline_events,
        description="Unchanged-depth baseline, no basis convergence assumed. " + window.description,
    )
    convergence = target_basis_scenario(
        scenario_type=ScenarioType.CONVERGENCE, name=ScenarioType.CONVERGENCE.value,
        assumption=ExitBasisAssumption(exit_spot_mid, settings.convergence_exit_basis_pct,
                                       snapshot.spot_book, snapshot.perp_book),
        opened_at=opened_at, closed_at=window.closed_at, funding_events=window.baseline_events,
        description="Hypothetical exit, not a price prediction. " + window.description,
    )
    if settings.adverse_exit_basis_mode == "entry_offset":
        # Quote-unit basis widening avoids dividing by the entry midpoint and
        # multiplying a rounded recurring ratio back into a target price.
        try:
            with localcontext() as context:
                context.traps[Inexact] = True
                widening = spot_mid * settings.adverse_exit_basis_pct
                target_perp_mid = exit_spot_mid + mid_basis(spot_mid, perp_mid) + widening
                adverse = ScenarioDefinition(
                    ScenarioType.ADVERSE, ScenarioType.ADVERSE.value,
                    ExitFundingScenario(
                        _translate_book(snapshot.spot_book, exit_spot_mid, window.closed_at),
                        _translate_book(snapshot.perp_book, target_perp_mid, window.closed_at),
                        opened_at, window.closed_at, window.stress_events,
                        f"Hypothetical basis widening of {widening} USDT/base, "
                        f"using {settings.adverse_exit_basis_pct} times entry Spot mid={spot_mid}; "
                        f"target Spot mid={exit_spot_mid}, target Perp mid={target_perp_mid}. "
                        "Not a price prediction. " + window.description,
                    ),
                )
        except Inexact as error:
            raise ValueError("stress price translation exceeds supported Decimal precision") from error
    else:
        adverse = target_basis_scenario(
            scenario_type=ScenarioType.ADVERSE, name=ScenarioType.ADVERSE.value,
            assumption=ExitBasisAssumption(exit_spot_mid, settings.adverse_exit_basis_pct,
                                           snapshot.spot_book, snapshot.perp_book),
            opened_at=opened_at, closed_at=window.closed_at,
            funding_events=window.stress_events,
            description="Hypothetical exit, not a price prediction. " + window.description,
        )
    return (baseline, convergence, adverse)
