"""Catalog-qualified cost comparisons through the existing estimate/scenario engine."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from crypto_arbitrage_bot.arbitrage.basis import mid_price
from crypto_arbitrage_bot.arbitrage.fees import FeeRates
from crypto_arbitrage_bot.arbitrage.funding_projection import projected_scenario_definitions
from crypto_arbitrage_bot.arbitrage.scenario import (
    evaluate_scenarios,
    reverse_execution_scenario,
    target_basis_scenario,
)
from crypto_arbitrage_bot.arbitrage.scenario_decision import decide_scenarios
from crypto_arbitrage_bot.config.research import (
    CandidateCatalogSettings,
    CandidateEconomicsSettings,
    ScenarioSettings,
)
from crypto_arbitrage_bot.market_data.candidate_catalog import (
    CandidateCatalog,
    CatalogStatus,
    ResearchCandidate,
    build_candidate_catalog,
)
from crypto_arbitrage_bot.market_data.candidate_market import CandidateObservation
from crypto_arbitrage_bot.market_data.quality import validate_market_data
from crypto_arbitrage_bot.models.data_quality import DataQualityConfig
from crypto_arbitrage_bot.models.estimate import EstimateConfig
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.models.scenario import (
    ExitBasisAssumption,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioType,
)
from crypto_arbitrage_bot.models.scenario_decision import (
    ScenarioDecisionConfig,
    ScenarioDecisionResult,
)
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


@dataclass(frozen=True)
class CandidateWindowResult:
    profile_name: str
    configured_holding_seconds: int
    scenarios: tuple[ScenarioResult, ...]
    decision: ScenarioDecisionResult

    @property
    def assumed_holding_duration(self) -> timedelta:
        scenario = self.scenarios[0].definition.exit_scenario
        return scenario.closed_at - scenario.opened_at


def _definitions(observation: CandidateObservation, profile: ScenarioSettings,
                 opened: datetime) -> tuple[ScenarioDefinition, ...]:
    snapshot = observation.assembly.snapshot
    assert snapshot is not None
    if profile.funding_projection is not None:
        # Preserve the existing one-settlement hypothesis and its actual closing
        # time. Configured holding_seconds is not presented as this duration.
        return projected_scenario_definitions(snapshot, opened, profile)
    closed = opened + timedelta(seconds=profile.holding_seconds)
    if any(event.symbol != snapshot.symbol or not opened < event.settled_at <= closed
           for event in profile.funding_events):
        raise ValueError("explicit candidate funding events must match the symbol and holding window")
    baseline = reverse_execution_scenario(
        name="baseline", spot_entry_book=snapshot.spot_book, perp_entry_book=snapshot.perp_book,
        opened_at=opened, closed_at=closed, funding_events=profile.funding_events,
        description="Hypothetical unchanged-depth reverse exit; no convergence. Explicit settlements only; not accrued income.",
    )
    spot_mid = profile.spot_exit_mid
    if profile.spot_exit_mid_mode == "entry_mid":
        assert snapshot.spot_book.best_bid is not None and snapshot.spot_book.best_ask is not None
        spot_mid = mid_price(snapshot.spot_book.best_bid, snapshot.spot_book.best_ask)
    assert spot_mid is not None
    return (baseline, *(target_basis_scenario(
        scenario_type=kind, name=kind.value,
        assumption=ExitBasisAssumption(spot_mid, basis, snapshot.spot_book, snapshot.perp_book),
        opened_at=opened, closed_at=closed, funding_events=profile.funding_events,
        description="Configured exit basis/depth hypothesis, not a prediction or later market observation.",
    ) for kind, basis in ((ScenarioType.CONVERGENCE, profile.convergence_exit_basis_pct),
                         (ScenarioType.ADVERSE, profile.adverse_exit_basis_pct))))


def candidate_estimate_config(candidate: ResearchCandidate, economics: CandidateEconomicsSettings,
                              fees: FeeRates, quality: DataQualityConfig) -> EstimateConfig:
    """Same candidate budget/rules configuration for calculation and preregistration."""
    return EstimateConfig(candidate.spot_rules, candidate.perp_rules, fees,
                          spot_budget=economics.spot_budget, capital_limit=economics.capital_limit,
                          stress_reserve=economics.stress_reserve, other_costs=economics.other_costs,
                          data_quality=quality)


def compare_candidate_windows(
    catalog: CandidateCatalog, criteria: CandidateCatalogSettings, observation: CandidateObservation,
    economics: CandidateEconomicsSettings, fees: FeeRates, quality: DataQualityConfig,
    decision: ScenarioDecisionConfig, *, evaluated_at: datetime,
) -> tuple[CandidateWindowResult, ...]:
    """Pure orchestration, no IO, no alternate cash flow and no execution authority.

    Catalog membership and exact quantity rules are re-derived from original
    metadata; an edited/stale candidate cannot bypass its original filters.
    Failed execution remains an incomplete estimate with no usable profit.
    All research PASS/OBSERVE/REJECT results come from the existing Decision.
    """
    opened = utc_timestamp(evaluated_at)
    if not isinstance(observation, CandidateObservation):
        raise TypeError("candidate observation is required")
    derived = build_candidate_catalog(*catalog.sources, criteria, now=opened)
    if (catalog.status is not CatalogStatus.COMPLETE or derived.status is not CatalogStatus.COMPLETE
            or catalog.candidates != derived.candidates or observation.candidate not in derived.candidates):
        raise ValueError("candidate metadata is stale, invalid or edited")
    identity = ResearchInstrument(observation.candidate.symbol, observation.candidate.base_asset)
    if observation.assembly.symbol != identity.symbol or observation.assembly.research_instrument != identity:
        raise ValueError("candidate and snapshot research identities differ")
    checked = validate_market_data(observation.assembly, quality, evaluated_at=opened)
    if not checked.valid or not observation.assembly.complete:
        raise ValueError("candidate market data is invalid or incomplete")
    candidate = observation.candidate
    config = candidate_estimate_config(candidate, economics, fees, quality)
    comparisons = []
    for name, profile in economics.windows.items():
        definitions = _definitions(observation, profile, opened)
        if any(item.exit_scenario.closed_at >= candidate.perp_delivery_at for item in definitions):
            raise ValueError("candidate exit hypothesis outlives the contract")
        scenarios = evaluate_scenarios(observation.assembly, config, definitions, evaluated_at=opened)
        comparisons.append(CandidateWindowResult(name, profile.holding_seconds, scenarios,
                                                decide_scenarios(scenarios, decision)))
    return tuple(comparisons)
