"""Persistence records, independent of research mathematics and SQL."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import ClassVar


@dataclass(frozen=True)
class SourceTimingRecord:
    source: str
    request_started_at: datetime | None
    request_finished_at: datetime | None
    received_at: datetime | None
    observed_at: datetime | None
    event_time: datetime | None
    transaction_time: datetime | None
    details_json: str

    DATETIME_FIELDS: ClassVar[tuple[str, ...]] = (
        "request_started_at", "request_finished_at", "received_at", "observed_at",
        "event_time", "transaction_time",
    )


@dataclass(frozen=True)
class ScenarioRecord:
    name: str
    scenario_type: str
    symbol: str
    estimate_status: str
    opened_at: datetime
    closed_at: datetime
    net_profit: Decimal | None
    net_return: Decimal | None
    capital_occupied: Decimal | None
    funding_total: Decimal | None
    details_json: str

    DATETIME_FIELDS: ClassVar[tuple[str, ...]] = ("opened_at", "closed_at")
    DECIMAL_FIELDS: ClassVar[tuple[str, ...]] = (
        "net_profit", "net_return", "capital_occupied", "funding_total",
    )


@dataclass(frozen=True)
class OpportunityRecord:
    """A saved observation, including rejected/failed research attempts.

    Top-level metrics refer to the explicitly supplied primary estimate.
    All scenario variants remain in scenarios, rather than selecting an
    arbitrary baseline or synthesizing profit/return pairs from different cases.
    Metadata JSON uses decimal strings and UTC ISO timestamps, never floats.
    """

    id: str
    created_at: datetime
    symbol: str
    primary_scenario_name: str | None
    snapshot_status: str
    snapshot_assembled_at: datetime
    snapshot_observed_at: datetime | None
    spot_bid: Decimal | None
    spot_ask: Decimal | None
    perp_bid: Decimal | None
    perp_ask: Decimal | None
    mark_price: Decimal | None
    index_price: Decimal | None
    funding_rate: Decimal | None
    next_funding_time: datetime | None
    open_interest: Decimal | None
    matched_quantity: Decimal | None
    matched_notional: Decimal | None
    residual_delta: Decimal | None
    executable_entry_basis: Decimal | None
    basis_pct: Decimal | None
    estimated_funding: Decimal | None
    spot_fees: Decimal | None
    perp_fees: Decimal | None
    price_pnl: Decimal | None
    capital_occupied: Decimal | None
    net_profit: Decimal | None
    net_return: Decimal | None
    decision: str
    decision_reason_codes: tuple[str, ...]
    data_quality_status: str | None
    data_quality_reason_codes: tuple[str, ...]
    formula_version: str | None
    config_version: str | None
    config_fingerprint: str
    config_json: str
    details_json: str
    source_timings: tuple[SourceTimingRecord, ...]
    scenarios: tuple[ScenarioRecord, ...]

    DATETIME_FIELDS: ClassVar[tuple[str, ...]] = (
        "created_at", "snapshot_assembled_at", "snapshot_observed_at", "next_funding_time",
    )
    DECIMAL_FIELDS: ClassVar[tuple[str, ...]] = (
        "spot_bid", "spot_ask", "perp_bid", "perp_ask", "mark_price", "index_price",
        "funding_rate", "open_interest", "matched_quantity", "matched_notional",
        "residual_delta", "executable_entry_basis", "basis_pct", "estimated_funding",
        "spot_fees", "perp_fees", "price_pnl", "capital_occupied", "net_profit", "net_return",
    )
    TUPLE_FIELDS: ClassVar[tuple[str, ...]] = (
        "decision_reason_codes", "data_quality_reason_codes",
    )
