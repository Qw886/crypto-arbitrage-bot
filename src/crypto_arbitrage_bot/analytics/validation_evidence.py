"""Cross-check recorded exit projections, without recalculating any cash flow.

Both exploratory reporting and formal study review use the same saved-evidence
checks. Original entry binding and planned-task timing remain caller concerns.
Local consistency is not exchange authentication or protection against an
operator rewriting all evidence; no trading permission is produced here.
"""

import json
from datetime import datetime
from decimal import Decimal
from typing import Any

from crypto_arbitrage_bot.models.validation_record import ForwardValidationRecord
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.decimal_math import require_decimal, utc_timestamp

VALIDATED_METRICS = (
    "validated_simulated_net_profit", "validated_simulated_net_return", "price_pnl",
    "funding_total", "spot_pnl", "perp_pnl", "original_entry_basis", "validated_exit_basis",
    "basis_change", "capital_occupied", "matched_quantity", "spot_entry_fee", "perp_entry_fee",
    "spot_exit_fee", "perp_exit_fee", "other_costs",
)


def require_validation_amounts(value: ForwardValidationRecord) -> None:
    for name in VALIDATED_METRICS:
        require_decimal(getattr(value, name), name)
    for name in ("capital_occupied", "matched_quantity"):
        require_decimal(getattr(value, name), name, positive=True)
    for name in ("spot_entry_fee", "perp_entry_fee", "spot_exit_fee", "perp_exit_fee", "other_costs"):
        require_decimal(getattr(value, name), name, nonnegative=True)


def _unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(items)
    if len(result) != len(items):
        raise ValueError("duplicate validation evidence fields")
    return result


def read_validation_attribution(value: ForwardValidationRecord) -> dict[str, Any]:
    """Match persisted columns to saved validator evidence and observed events."""
    details = json.loads(value.details_json, object_pairs_hook=_unique)
    if details["exit_data_quality"]["issues"] != []:
        raise ValueError("saved exit quality is not valid")
    for name in ("symbol", "opportunity_id", "status", "reason_codes", "entry_time",
                 "validation_time", "holding_duration", "matched_quantity", "capital_occupied",
                 "spot_pnl", "perp_pnl", "validated_simulated_net_return", "original_entry_basis",
                 "basis_change"):
        if exact_json(details[name]) != exact_json(getattr(value, name)):
            raise ValueError("stored validation projection differs from details")
    calculation = details["calculation"]
    for name, stored in (("net_profit", value.validated_simulated_net_profit),
                         ("exit_basis", value.validated_exit_basis), ("price_pnl", value.price_pnl),
                         ("funding_total", value.funding_total), ("other_costs", value.other_costs),
                         ("quantity", value.matched_quantity), ("entry_basis", value.original_entry_basis)):
        if exact_json(calculation[name]) != exact_json(stored):
            raise ValueError("stored calculation projection differs from details")
    if calculation["symbol"] != value.symbol:
        raise ValueError("calculation identity mismatch")
    for name in ("spot_entry", "perp_entry", "spot_exit", "perp_exit"):
        if exact_json(details["fees"][name]) != exact_json(getattr(value, name + "_fee")):
            raise ValueError("fee projection differs from details")
    events = details["funding_events"]
    if not isinstance(events, list):
        raise ValueError("stored funding events must be a list")
    ids, times = set(), set()
    for event in events:
        identifier = event["event_id"]
        settled = utc_timestamp(datetime.fromisoformat(event["settled_at"]))
        if (not isinstance(identifier, str) or not identifier
                or identifier.startswith("FORECAST_ONLY:") or identifier in ids
                or settled in times or event["symbol"] != value.symbol
                or value.entry_time is None or value.validation_time is None
                or not value.entry_time < settled <= value.validation_time):
            raise ValueError("invalid observed funding identity or window")
        for name in ("funding_rate", "settlement_mark_price"):
            if not isinstance(event[name], str):
                raise ValueError("stored funding number must be Decimal text")
            require_decimal(Decimal(event[name]), name, positive=name == "settlement_mark_price")
        ids.add(identifier)
        times.add(settled)
    decomposition = details["decomposition"]
    attribution = {}
    for name in ("mid_basis_pnl", "spread_costs", "slippage_costs", "rounding_residual"):
        if not isinstance(decomposition[name], str):
            raise ValueError("stored attribution must be Decimal text")
        attribution[name] = require_decimal(Decimal(decomposition[name]), name,
                                            nonnegative=name in ("spread_costs", "slippage_costs"))
    if exact_json(decomposition["price_pnl"]) != exact_json(value.price_pnl):
        raise ValueError("attribution projection differs from details")
    return {"funding_event_count": len(events), **attribution}
