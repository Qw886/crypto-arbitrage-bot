"""Lossless, versioned storage-only references for repeated scenario evidence.

Repository callers always see the original expanded record. Only canonical
builder JSON is compacted; arbitrary legacy JSON is preserved verbatim.
No financial values are parsed, rounded, or recalculated here.
"""

import json
from dataclasses import replace
from typing import Any

from crypto_arbitrage_bot.models.opportunity import OpportunityRecord
from crypto_arbitrage_bot.storage.opportunity_records import exact_json

FORMAT = "opportunity-scenario-references"
VERSION = 1


def _canonical_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        if isinstance(value, dict) and exact_json(value) == text:
            return value
    except (ValueError, TypeError):
        pass
    return None


def _envelope(details: dict[str, Any], **metadata: Any) -> str:
    return exact_json({"storage_format": FORMAT, "version": VERSION,
                       "details": details, **metadata})


def _read_envelope(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None  # Legacy opaque JSON remains the repository's responsibility.
    if not isinstance(value, dict) or value.get("storage_format") != FORMAT:
        return None
    if type(value.get("version")) is not int or value["version"] != VERSION:
        raise ValueError("unsupported scenario storage format version")
    if not isinstance(value.get("details"), dict):
        raise ValueError("corrupt scenario storage envelope")
    return value


def compact_record(record: OpportunityRecord) -> OpportunityRecord:
    """Copy only storage JSON; never mutate source records or original evidence."""
    parent = _canonical_object(record.details_json)
    config = _canonical_object(record.config_json)
    if parent is None or config is None or not record.scenarios:
        return record
    shared: dict[str, Any] = {}
    children = []
    changed = False
    primary_exit_position = None
    parent_estimate = parent.get("estimate")
    for position, child in enumerate(record.scenarios):
        details = _canonical_object(child.details_json)
        if details is None:
            children.append(child)
            continue
        refs = []
        for field, source, key in (
            ("entry_snapshot", parent, "snapshot"),
            ("config", config, "estimate"),
        ):
            if field in details and key in source and details[field] == source[key]:
                del details[field]
                refs.append(field)
        if "entry_mid_basis" in details:
            if "entry_mid_basis" not in shared:
                shared["entry_mid_basis"] = details["entry_mid_basis"]
            if details["entry_mid_basis"] == shared["entry_mid_basis"]:
                del details["entry_mid_basis"]
                refs.append("entry_mid_basis")
        definition, estimate = details.get("definition"), details.get("estimate")
        if isinstance(definition, dict) and "exit_scenario" in definition:
            exit_scenario = definition["exit_scenario"]
            if isinstance(estimate, dict) and "scenario" in estimate and estimate["scenario"] == exit_scenario:
                del estimate["scenario"]
                refs.append("estimate.scenario")
            # The parent primary estimate carries the same exit evidence too.
            if (isinstance(parent_estimate, dict) and "scenario" in parent_estimate
                    and parent_estimate["scenario"] == exit_scenario):
                del parent_estimate["scenario"]
                primary_exit_position = position
        if refs or primary_exit_position == position:
            children.append(replace(child, details_json=_envelope(details, references=refs)))
            changed = True
        else:
            children.append(child)
    if not changed:
        return record
    return replace(record, scenarios=tuple(children), details_json=_envelope(
        parent, shared=shared, primary_exit_position=primary_exit_position,
    ))


def expand_record(record: OpportunityRecord) -> OpportunityRecord:
    """Restore the exact public record shape; malformed references fail closed."""
    envelope = _read_envelope(record.details_json)
    if envelope is None:
        if any(_read_envelope(child.details_json) is not None for child in record.scenarios):
            raise ValueError("scenario references require their opportunity envelope")
        return record
    try:
        parent = envelope["details"]
        shared = envelope["shared"]
        config = json.loads(record.config_json)
        children = []
        for child in record.scenarios:
            wrapped = _read_envelope(child.details_json)
            if wrapped is None:
                children.append(child)
                continue
            details = wrapped["details"]
            refs = wrapped["references"]
            if not isinstance(refs, list) or len(set(refs)) != len(refs):
                raise ValueError("invalid scenario references")
            for ref in refs:
                if ref == "estimate.scenario":
                    target, key = details["estimate"], "scenario"
                    value = details["definition"]["exit_scenario"]
                else:
                    target, key = details, ref
                    if ref == "entry_snapshot":
                        value = parent["snapshot"]
                    elif ref == "config":
                        value = config["estimate"]
                    elif ref == "entry_mid_basis":
                        value = shared["entry_mid_basis"]
                    else:
                        raise ValueError("unknown scenario reference")
                if key in target:
                    raise ValueError("ambiguous scenario reference")
                target[key] = value
            children.append(replace(child, details_json=exact_json(details)))
        position = envelope["primary_exit_position"]
        if position is not None:
            if type(position) is not int or not 0 <= position < len(children):
                raise ValueError("invalid primary scenario position")
            if "scenario" in parent["estimate"]:
                raise ValueError("ambiguous primary scenario reference")
            parent["estimate"]["scenario"] = json.loads(
                children[position].details_json,
            )["definition"]["exit_scenario"]
        return replace(record, details_json=exact_json(parent), scenarios=tuple(children))
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("corrupt scenario storage references") from exc
