"""Project existing results into records without recomputing research outcomes."""

import base64
import hashlib
import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

from crypto_arbitrage_bot.models.decision import DecisionResult
from crypto_arbitrage_bot.models.estimate import OfflineEstimate
from crypto_arbitrage_bot.models.opportunity import (
    OpportunityRecord,
    ScenarioRecord,
    SourceTimingRecord,
)
from crypto_arbitrage_bot.models.scenario import ScenarioResult
from crypto_arbitrage_bot.models.scenario_decision import ScenarioDecisionResult
from crypto_arbitrage_bot.storage.record_validation import (
    RecordBuildError,
    RecordBuildReason,
    validate_evidence,
)
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp


def utc_text(value: datetime) -> str:
    """Fixed-width UTC ISO-8601, preserving all Python datetime microseconds."""
    return utc_timestamp(value).isoformat(timespec="microseconds")


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return utc_text(value)
    if isinstance(value, timedelta):
        return {"microseconds": (value.days * 86400 + value.seconds) * 1_000_000 + value.microseconds}
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(f"unsupported persistence value: {type(value).__name__}; floats are not allowed")


def exact_json(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _estimate_details(estimate: OfflineEstimate) -> dict[str, Any]:
    # The primary assembly and request provenance are saved separately.
    return {field.name: getattr(estimate, field.name) for field in fields(estimate)
            if field.name not in ("assembly", "config")}


def build_opportunity(
    estimate: OfflineEstimate, decision: DecisionResult | ScenarioDecisionResult,
    *, scenarios: tuple[ScenarioResult, ...] = (), created_at: datetime | None = None,
    opportunity_id: str | None = None, formula_version: str | None = None,
    config_version: str | None = None,
) -> OpportunityRecord:
    """Record supplied evidence, never call estimate/decision or invent missing data.

    For 7C, scenarios come from its decision and primary estimate must be one
    of them. For 7A, its estimate must match. Explicit versions are optional:
    the research models currently have no formula/config version fields.
    """
    if not isinstance(estimate, OfflineEstimate):
        raise TypeError("estimate must be OfflineEstimate")
    if not isinstance(scenarios, tuple) or any(not isinstance(item, ScenarioResult) for item in scenarios):
        raise TypeError("scenarios must be a tuple of ScenarioResult")
    primary_name = None
    if isinstance(decision, ScenarioDecisionResult):
        if scenarios and scenarios != decision.scenarios:
            raise RecordBuildError((RecordBuildReason.DECISION_MISMATCH,),
                                   "scenarios must match the supplied decision")
        scenarios = decision.scenarios
        matches = [item for item in scenarios if item.estimate == estimate]
        if not matches:
            raise RecordBuildError((RecordBuildReason.DECISION_MISMATCH,),
                                   "primary estimate must belong to the supplied scenario decision")
        primary_name = matches[0].name
    elif isinstance(decision, DecisionResult):
        if decision.estimate != estimate:
            raise RecordBuildError((RecordBuildReason.DECISION_MISMATCH,),
                                   "estimate must match the supplied decision")
    else:
        raise TypeError("decision must be an existing single or scenario decision")
    validate_evidence(estimate, scenarios)
    record_id = str(uuid4()) if opportunity_id is None else opportunity_id
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("opportunity_id must be nonempty")
    for name, version in (("formula_version", formula_version), ("config_version", config_version)):
        if version is not None and (not isinstance(version, str) or not version.strip()):
            raise ValueError(f"{name} must be nonempty when supplied")

    assembly, snapshot = estimate.assembly, estimate.assembly.snapshot
    quality, calculation = estimate.data_quality, estimate.calculation
    matched_notional = None
    if len(estimate.fills) == 4 and all(fill.complete for fill in estimate.fills):
        matched_notional = min(estimate.fills[0].quote_notional, estimate.fills[2].quote_notional)

    timing_records = []
    for source_name in ("spot_depth", "perp_depth", "premium_index", "open_interest"):
        source = getattr(assembly.sources, source_name)
        data = None if source is None else getattr(source, "data", None)
        observed = getattr(source, "order_book", None) if source_name == "spot_depth" else (
            getattr(data, "order_book", None) if source_name == "perp_depth" else data
        )
        timing_records.append(SourceTimingRecord(
            source_name, getattr(source, "started_at", None), getattr(source, "finished_at", None),
            getattr(source, "received_at", None), getattr(observed, "observed_at", None),
            getattr(observed, "event_time", None), getattr(data, "transaction_time", None),
            exact_json(source),
        ))

    scenario_records = tuple(ScenarioRecord(
        item.name, item.scenario_type.value, item.estimate.symbol, item.estimate.status.value,
        item.definition.exit_scenario.opened_at, item.definition.exit_scenario.closed_at,
        item.net_profit, item.net_return, item.capital_occupied, item.funding_total,
        exact_json({"definition": item.definition, "estimate": _estimate_details(item.estimate),
                    "config": item.estimate.config, "entry_snapshot": item.estimate.assembly.snapshot,
                    "entry_mid_basis": item.entry_mid_basis,
                    "assumed_exit_mid_basis": item.assumed_exit_mid_basis}),
    ) for item in scenarios)
    config_json = exact_json({"estimate": estimate.config, "decision": decision.config})
    return OpportunityRecord(
        id=record_id, created_at=utc_timestamp(utc_now() if created_at is None else created_at),
        symbol=estimate.symbol, primary_scenario_name=primary_name,
        snapshot_status=assembly.status.value, snapshot_assembled_at=assembly.assembled_at,
        snapshot_observed_at=None if snapshot is None else snapshot.observed_at,
        spot_bid=None if snapshot is None else snapshot.spot_book.best_bid,
        spot_ask=None if snapshot is None else snapshot.spot_book.best_ask,
        perp_bid=None if snapshot is None else snapshot.perp_book.best_bid,
        perp_ask=None if snapshot is None else snapshot.perp_book.best_ask,
        mark_price=None if snapshot is None else snapshot.mark_price,
        index_price=None if snapshot is None else snapshot.index_price,
        funding_rate=None if snapshot is None else snapshot.funding_rate,
        next_funding_time=None if snapshot is None else snapshot.next_funding_time,
        open_interest=None if snapshot is None else snapshot.open_interest,
        matched_quantity=estimate.matched_quantity, matched_notional=matched_notional,
        residual_delta=estimate.residual_delta, executable_entry_basis=estimate.executable_entry_basis,
        basis_pct=estimate.basis_pct, estimated_funding=estimate.estimated_funding,
        spot_fees=None if calculation is None else calculation.spot_fees,
        perp_fees=None if calculation is None else calculation.perp_fees,
        price_pnl=estimate.price_pnl, capital_occupied=estimate.capital_occupied,
        net_profit=estimate.net_profit, net_return=estimate.net_return,
        decision=decision.status.value,
        decision_reason_codes=tuple(reason.value for reason in decision.reason_codes),
        data_quality_status=None if quality is None else quality.status.value,
        data_quality_reason_codes=() if quality is None else tuple(code.value for code in quality.reason_codes),
        formula_version=formula_version, config_version=config_version,
        config_fingerprint=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        config_json=config_json,
        details_json=exact_json({"snapshot": snapshot, "snapshot_issues": assembly.issues,
                                 "spot_perp_sample_delta": assembly.spot_perp_sample_delta,
                                 "estimate": _estimate_details(estimate),
                                 "decision_issues": getattr(decision, "issues", None),
                                 "scenario_margin": getattr(decision, "scenario_margin", None)}),
        source_timings=tuple(timing_records), scenarios=scenario_records,
    )
