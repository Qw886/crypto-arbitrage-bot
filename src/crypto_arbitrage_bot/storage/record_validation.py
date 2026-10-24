"""Evidence consistency only; no market quality or decision rules are rerun."""

from enum import StrEnum

from crypto_arbitrage_bot.models.estimate import OfflineEstimate
from crypto_arbitrage_bot.models.scenario import ScenarioResult


class RecordBuildReason(StrEnum):
    SYMBOL_MISMATCH = "SYMBOL_MISMATCH"
    SNAPSHOT_MISMATCH = "SNAPSHOT_MISMATCH"
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"
    DECISION_MISMATCH = "DECISION_MISMATCH"
    SCENARIO_MISMATCH = "SCENARIO_MISMATCH"
    INVALID_INPUT = "INVALID_INPUT"


class RecordBuildError(ValueError):
    """Structured assembly failure, not a research REJECT decision."""

    def __init__(self, reason_codes: tuple[RecordBuildReason, ...], message: str) -> None:
        self.reason_codes = reason_codes
        super().__init__(message)


def validate_evidence(estimate: OfflineEstimate, scenarios: tuple[ScenarioResult, ...]) -> None:
    """Compare retained provenance by value, allowing equivalent deserialized inputs.

    Failed research with missing quantities can be recorded. Known scenario
    quantities must agree; missing values are never replaced with zero.
    """
    reasons: list[RecordBuildReason] = []

    def mismatch(reason: RecordBuildReason) -> None:
        if reason not in reasons:
            reasons.append(reason)

    quantities = []
    for current in (estimate, *(item.estimate for item in scenarios)):
        if current.assembly != estimate.assembly:
            mismatch(RecordBuildReason.SNAPSHOT_MISMATCH)
        if current.matched_quantity is not None:
            quantities.append(current.matched_quantity)
        if current.calculation is not None:
            if current.calculation.quantity != current.matched_quantity:
                mismatch(RecordBuildReason.QUANTITY_MISMATCH)
        snapshot = current.assembly.snapshot
        objects: list[object] = [current, current.calculation, *current.fills,
                   current.scenario.spot_exit_book, current.scenario.perp_exit_book,
                   *current.scenario.funding_events]
        if snapshot is not None:
            objects.extend((snapshot, snapshot.spot_book, snapshot.perp_book))
        for source_name in ("spot_depth", "perp_depth", "premium_index", "open_interest"):
            source = getattr(current.assembly.sources, source_name)
            if source is None:
                continue
            if dict(source.request.params).get("symbol") != estimate.symbol:
                mismatch(RecordBuildReason.SYMBOL_MISMATCH)
            data = getattr(source, "data", None)
            objects.extend((getattr(source, "order_book", None),
                            data, getattr(data, "order_book", None)))
        if any(getattr(item, "symbol", estimate.symbol) != estimate.symbol for item in objects):
            mismatch(RecordBuildReason.SYMBOL_MISMATCH)
    if quantities and any(quantity != quantities[0] for quantity in quantities):
        mismatch(RecordBuildReason.QUANTITY_MISMATCH)
    for item in scenarios:
        if item.definition.exit_scenario != item.estimate.scenario:
            mismatch(RecordBuildReason.SCENARIO_MISMATCH)
        assumption = item.definition.basis_assumption
        if assumption is not None and any(
            book.symbol != estimate.symbol for book in (
                assumption.spot_depth_template, assumption.perp_depth_template,
            )
        ):
            mismatch(RecordBuildReason.SYMBOL_MISMATCH)
    if reasons:
        raise RecordBuildError(tuple(reasons), "inconsistent research evidence: " +
                               ", ".join(reason.value for reason in reasons))
