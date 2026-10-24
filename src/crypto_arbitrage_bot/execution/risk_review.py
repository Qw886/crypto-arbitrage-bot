"""Pinned native TESTNET qualification evidence; no network or journal writes.

This verifies trusted local run witnesses against the native accounting archive.
It cannot cryptographically authenticate the operator or exchange responses.
No summary 'PASS' flag, pytest count or mock result authorizes an entry.
"""

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any

from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
from crypto_arbitrage_bot.config.settings import Settings
from crypto_arbitrage_bot.execution.account_balance_audit import (
    AssetBalance,
    BalanceAuditStatus,
    BalanceSnapshot,
    audit_balance_changes,
)
from crypto_arbitrage_bot.execution.risk_accounting import verify_native_accounting_projection
from crypto_arbitrage_bot.execution.types import AccountTrade, ExecutionMarket, FundingPayment
from crypto_arbitrage_bot.models.market import Side
from crypto_arbitrage_bot.models.strategy_admission import (
    AdmissionReason,
    ExecutionRiskCase,
    ExecutionRiskReview,
)
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.storage.study_registry import full_research_policy
from crypto_arbitrage_bot.utils.decimal_math import (
    decimal_calculation,
    require_decimal,
    utc_timestamp,
)

MAX_REVIEW_BYTES = 256_000


class RiskReviewError(RuntimeError):
    def __init__(self, reason: AdmissionReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def execution_policy_fingerprint(settings: Settings) -> str:
    if settings.testnet is None:
        raise ValueError("explicit TESTNET settings required")
    policy = settings.testnet.model_dump(exclude={"admission", "journal_path", "balance_evidence_directory"})
    return _sha(exact_json({"research": json.loads(full_research_policy(settings)), "testnet": policy}))


def implementation_fingerprint() -> str:
    """Bind every installed package source, not only the gate or a version label."""
    package = Path(__file__).resolve().parents[1]
    sources = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(package.rglob("*.py"))}
    return _sha(exact_json(sources))


def _decimal(value: object, *, positive: bool = False) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("native financial evidence must remain Decimal text")
    return require_decimal(Decimal(value), "risk evidence", positive=positive)


def _time(value: str | datetime) -> datetime:
    return utc_timestamp(value if isinstance(value, datetime) else datetime.fromisoformat(value))


def _pinned(path: Path, digest: str) -> dict[str, Any]:
    with path.open("rb") as source:
        body = source.read(MAX_REVIEW_BYTES+1)
    if len(body) > MAX_REVIEW_BYTES or hashlib.sha256(body).hexdigest() != digest:
        raise RiskReviewError(AdmissionReason.RISK_REVIEW_UNAVAILABLE)
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("object evidence required")
    # Do not accept duplicate fields or floating point financial values.
    if exact_json(data).encode("utf-8") != body:
        raise ValueError("canonical exact JSON required")
    return data


@decimal_calculation
def _native_accounting(db: sqlite3.Connection, attempt: sqlite3.Row, proof: dict[str, Any]) -> dict[str, Any]:
    audit = db.execute("SELECT * FROM testnet_balance_audits WHERE audit_id=? AND attempt_id=?",
                       (proof["audit_id"], attempt["attempt_id"])).fetchone()
    accounting = db.execute("SELECT * FROM testnet_trade_reconciliations WHERE attempt_id=?",
                            (attempt["attempt_id"],)).fetchone()
    if audit is None or accounting is None or audit["status"] != "MATCHED":
        raise ValueError("native balance/accounting evidence missing")
    if (_sha(audit["report_json"]) != proof["audit_sha256"]
            or _sha(accounting["report_json"]) != proof["accounting_sha256"]):
        raise ValueError("native evidence changed")
    balance, trades = json.loads(audit["report_json"]), json.loads(accounting["report_json"])
    if (accounting["status"] != trades["status"] or balance["status"] != "MATCHED" or balance["issues"] or balance["funding_evidence_complete"] is not True
            or balance["scope"] != "TESTNET_CURRENCY_BALANCE_V1_NOT_PROFIT"
            or trades["attempt_id"] != attempt["attempt_id"] or trades["issues"]):
        raise ValueError("native report is incomplete")
    frames = {}
    for phase in ("before", "after"):
        snapshots = balance[phase]
        if len(snapshots) != 2 or {s["market"] for s in snapshots} != {"SPOT", "USDTM"}:
            raise ValueError("both native wallets required")
        decoded = []
        for snapshot in snapshots:
            if _time(snapshot["requested_at"]) > _time(snapshot["received_at"]):
                raise ValueError("invalid native capture time")
            if _time(snapshot["received_at"]) > _time(proof["finished_at"]):
                raise ValueError("native wallet capture follows witness completion")
            native = json.loads(snapshot["raw_balances_json"])
            if not native.get("balances" if snapshot["market"] == "SPOT" else "assets"):
                raise ValueError("native wallet response missing")
            key = "balances" if snapshot["market"] == "SPOT" else "assets"
            amounts = {row["asset"]: (_decimal(row["free"])+_decimal(row["locked"]) if key == "balances"
                                      else _decimal(row["walletBalance"])) for row in native[key]}
            if (len(amounts) != len(native[key]) or amounts != {
                    item["asset"]: _decimal(item["amount"]) for item in snapshot["balances"]}):
                raise ValueError("native wallet fields differ from decoded snapshot")
            decoded.append(BalanceSnapshot(ExecutionMarket(snapshot["market"]), _time(snapshot["requested_at"]),
                _time(snapshot["received_at"]), tuple(AssetBalance(asset, amount) for asset, amount in amounts.items())))
        frames[phase] = tuple(decoded)
    if not balance["movements"] or any(_decimal(m["unexplained_delta"]) != 0 for m in balance["movements"]):
        raise ValueError("unexplained balance movement")
    if not balance["trades"]:
        raise ValueError("no actual fills")
    decoded_trades = []
    symbol = json.loads(attempt["plan_json"])["spot_entry"]["symbol"]
    for trade in balance["trades"]:
        row = db.execute("SELECT * FROM testnet_trade_evidence WHERE market=? AND symbol=? AND trade_id=?",
                         (trade["market"], trade["symbol"], trade["trade_id"])).fetchone()
        if row is None or row["attempt_id"] != attempt["attempt_id"] or row["evidence_json"] != exact_json(trade):
            raise ValueError("unlinked native trade")
        native = json.loads(trade["raw_json"])
        buyer_key = "isBuyer" if trade["market"] == "SPOT" else "buyer"
        if (str(native["id"]) != trade["trade_id"] or str(native["orderId"]) != trade["order_id"]
                or trade["symbol"] != symbol or native["symbol"] != symbol
                or type(native[buyer_key]) is not bool or native[buyer_key] != (trade["side"] == "BUY")
                or type(native["time"]) is not int or native["time"] <= 0
                or datetime(1970, 1, 1, tzinfo=UTC)+timedelta(milliseconds=native["time"]) != _time(trade["traded_at"])
                or _decimal(native["qty"], positive=True) != _decimal(trade["quantity"], positive=True)
                or _decimal(native["price"], positive=True) != _decimal(trade["price"], positive=True)
                or _decimal(native["quoteQty"], positive=True) != _decimal(trade["quote_quantity"], positive=True)
                or _decimal(native["commission"]) != _decimal(trade["commission"])
                or native["commissionAsset"] != trade["commission_asset"]):
            raise ValueError("native trade fields differ")
        if trade["market"] == "USDTM" and (native["side"] != trade["side"]
                or _decimal(native["realizedPnl"]) != _decimal(trade["official_perp_pnl"])):
            raise ValueError("native perpetual trade fields differ")
        decoded_trades.append(AccountTrade(ExecutionMarket(trade["market"]), trade["symbol"], trade["trade_id"],
            trade["order_id"], Side(trade["side"]), _decimal(trade["price"], positive=True),
            _decimal(trade["quantity"], positive=True), _decimal(trade["quote_quantity"], positive=True),
            _decimal(trade["commission"]), trade["commission_asset"], _time(trade["traded_at"]),
            None if trade["official_perp_pnl"] is None else _decimal(trade["official_perp_pnl"]), trade["raw_json"]))
    if balance["adjustments"]:
        raise ValueError("qualification frames cannot use unexplained external adjustments")
    funding = tuple(FundingPayment(payment["transaction_id"], payment["symbol"], _time(payment["posted_at"]),
                                   _decimal(payment["amount"]), payment["asset"]) for payment in balance["funding"])
    if balance["funding"] != trades["funding_payments"]:
        raise ValueError("native funding evidence differs between archives")
    # Reuse the original account audit, never create a second PnL formula or
    # accept a manually asserted MATCHED summary with contradictory sources.
    checked = audit_balance_changes(frames["before"], frames["after"], tuple(decoded_trades), funding,
                                    funding_evidence_complete=True)
    if checked.status is not BalanceAuditStatus.MATCHED or any(m.unexplained_delta != 0 for m in checked.movements):
        raise ValueError("native source accounting does not match")
    for leg in ("spot_entry", "perp_entry"):
        receipt = None if attempt[leg] is None else json.loads(attempt[leg])
        market = "SPOT" if leg == "spot_entry" else "USDTM"
        leg_trades = [trade for trade in decoded_trades if trade.market.value == market
                      and trade.side.value == ("BUY" if market == "SPOT" else "SELL")]
        if receipt is None and leg_trades:
            raise ValueError("unexpected native entry")
        if receipt is not None and sum((trade.quantity for trade in leg_trades if trade.order_id == receipt["order_id"]),
                                      Decimal(0)) != _decimal(receipt["executed_quantity"]):
            raise ValueError("native entry fill differs from journal")
    verify_native_accounting_projection(db, attempt, trades, tuple(decoded_trades), funding,
                                       finished_at=_time(proof["finished_at"]))
    return dict(trades)


@decimal_calculation
def _case_matches(kind: ExecutionRiskCase, attempt: sqlite3.Row, proof: dict[str, Any],
                  accounting: dict[str, Any], db: sqlite3.Connection) -> bool:
    plan = json.loads(attempt["plan_json"])
    receipts = {leg: None if attempt[leg] is None else json.loads(attempt[leg])
                for leg in ("spot_entry", "perp_entry", "spot_exit", "perp_exit")}
    events = proof["events"]
    if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
        raise ValueError("structured run witness events required")
    terminal = attempt["state"] in ("CLOSED", "FLATTENED")
    if not terminal:
        return False
    if kind in (ExecutionRiskCase.NORMAL_ENTRY_EXIT, ExecutionRiskCase.FEES_FUNDING_BALANCE):
        complete = (attempt["state"] == "CLOSED" and accounting["status"] == "RECONCILED"
                    and accounting["accounting_scope"] == "TESTNET_ORDER_FILLS_AND_FUNDING_V1"
                    and accounting["calculation"] is not None and accounting["fees"] is not None
                    and {leg["leg"] for leg in accounting["legs"]} == {"SPOT_ENTRY", "SPOT_EXIT", "PERP_ENTRY", "PERP_EXIT"}
                    and all(value is not None and _decimal(value["executed_quantity"], positive=True)
                            == _decimal(plan[leg]["quantity"], positive=True) for leg, value in receipts.items()))
        if kind is ExecutionRiskCase.FEES_FUNDING_BALANCE:
            complete = complete and bool(accounting["funding_payments"])
            for payment in accounting["funding_payments"]:
                source = db.execute("SELECT * FROM testnet_funding_payments WHERE attempt_id=? AND transaction_id=?",
                                    (attempt["attempt_id"], payment["transaction_id"])).fetchone()
                if source is None:
                    return False
                if (source["symbol"] != payment["symbol"] or _decimal(source["amount"]) != _decimal(payment["amount"])
                        or _time(source["posted_at"]) != _time(payment["posted_at"])):
                    return False
        return complete
    if kind is ExecutionRiskCase.SPOT_ACK_RECOVERY:
        spot, perp = receipts["spot_entry"], receipts["perp_entry"]
        return (attempt["state"] == "FLATTENED" and spot is not None and perp is None
                and _decimal(spot["executed_quantity"], positive=True) > 0
                and any(e.get("kind") == "LOCAL_ACK_DELIVERY_INTERRUPT" and e.get("market") == "SPOT"
                        and e.get("client_order_id") == spot["client_order_id"] for e in events))
    if kind is ExecutionRiskCase.PERP_UNKNOWN_RECOVERY:
        perp = receipts["perp_entry"]
        return (perp is not None and _decimal(perp["executed_quantity"], positive=True) > 0
                and any(e.get("kind") == "POST_RESULT_UNKNOWN" and e.get("market") == "USDTM"
                        and e.get("error_kind") in ("TIMEOUT", "NETWORK_ERROR", "HTTP_ERROR")
                        and e.get("client_order_id") == perp["client_order_id"] for e in events)
                and any(e.get("kind") == "FIXED_ID_QUERY_RESOLVED" and e.get("market") == "USDTM"
                        and e.get("order_id") == perp["order_id"]
                        and e.get("client_order_id") == perp["client_order_id"] for e in events))
    if kind is ExecutionRiskCase.PARTIAL_ENTRY_RECOVERY:
        return any(value is not None and 0 < _decimal(value["executed_quantity"])
                   < _decimal(plan[leg]["quantity"], positive=True)
                   for leg, value in receipts.items() if leg in ("spot_entry", "perp_entry"))
    if kind is ExecutionRiskCase.PARTIAL_EXIT_RECOVERY:
        # Native trade groups prove the original exit IOC filled partially;
        # replacement receipts alone cannot be relabeled as a partial exit.
        quantities: dict[tuple[str, str], Decimal] = {}
        for leg in accounting["legs"]:
            if leg["leg"] not in ("SPOT_EXIT", "PERP_EXIT"):
                continue
            for trade in leg["trades"]:
                key = (trade["market"], trade["order_id"])
                quantities[key] = quantities.get(key, Decimal(0)) + _decimal(trade["quantity"], positive=True)
        for event in events:
            if event.get("kind") != "PARTIAL_EXIT_TERMINAL":
                continue
            native = event["native_order"]
            intent = db.execute("SELECT order_json FROM testnet_order_intents WHERE attempt_id=? AND client_order_id=?",
                                (attempt["attempt_id"], native["clientOrderId"])).fetchone()
            if intent is None:
                continue
            order = json.loads(intent[0])
            size = _decimal(native["origQty"], positive=True)
            filled = quantities.get((event.get("market"), str(native["orderId"])), Decimal(0))
            if (order["market"] == event.get("market") and order["side"] == ("SELL" if order["market"] == "SPOT" else "BUY")
                    and order["symbol"] == plan["spot_entry"]["symbol"] and _decimal(order["quantity"]) == size
                    and native["status"] in ("EXPIRED", "CANCELED", "EXPIRED_IN_MATCH")
                    and 0 < filled < size and _decimal(native["executedQty"]) == filled):
                return True
        return False
    if kind is ExecutionRiskCase.RESTART_IDEMPOTENCY:
        intents = [dict(r) for r in db.execute("SELECT * FROM testnet_order_intents WHERE attempt_id=? ORDER BY client_order_id",
                                              (attempt["attempt_id"],))]
        digest = _sha(exact_json(intents))
        return any(e.get("kind") == "TERMINAL_RESTART_NO_NEW_INTENTS"
                   and e.get("before_intents_sha256") == e.get("after_intents_sha256") == digest
                   and type(e.get("entry_post_count")) is int and e["entry_post_count"] == 0 for e in events)
    return False


def read_execution_risk_review(reference: StudyRegistrationSettings, settings: Settings,
                               *, evaluated_at: datetime) -> ExecutionRiskReview:
    """Require native proof for every configured symbol/case, current code and policy.

    Old evidence without the original code/policy-bound run envelope is useful
    historical engineering evidence, but cannot qualify a changed implementation.
    """
    try:
        at = utc_timestamp(evaluated_at)
        data = _pinned(reference.path, reference.sha256)
        if type(data["schema_version"]) is not int or data["schema_version"] != 1 or data["environment"] != "TESTNET":
            raise ValueError("native TESTNET risk schema required")
        registered, reviewed, expires = (_time(data[n]) for n in ("registered_at", "reviewed_at", "expires_at"))
        if not registered <= reviewed <= at < expires:
            raise RiskReviewError(AdmissionReason.RISK_REVIEW_EXPIRED)
        if data["policy_sha256"] != execution_policy_fingerprint(settings):
            raise RiskReviewError(AdmissionReason.RISK_POLICY_MISMATCH)
        if data["implementation_sha256"] != implementation_fingerprint():
            raise RiskReviewError(AdmissionReason.IMPLEMENTATION_MISMATCH)
        limits = tuple((s, _decimal(q, positive=True)) for s, q in data["qualified_quantities"])
        if len(dict(limits)) != len(limits) or set(dict(limits)) != set(settings.symbols):
            raise ValueError("exact qualified symbol scope required")
        journal = Path(data["journal_path"])
        if not journal.is_absolute():
            raise ValueError("explicit absolute qualification journal required")
        found = []
        identities = set()
        with closing(sqlite3.connect(journal.as_uri()+"?mode=ro", uri=True, timeout=0)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            for case in data["cases"]:
                kind, symbol = ExecutionRiskCase(case["case"]), case["symbol"]
                identity = (symbol, kind)
                if identity in identities or symbol not in settings.symbols:
                    raise ValueError("duplicate/out-of-scope risk case")
                identities.add(identity)
                proof_path = Path(case["proof_path"])
                if not proof_path.is_absolute():
                    raise ValueError("explicit proof path required")
                proof = _pinned(proof_path, case["proof_sha256"])
                attempt = db.execute("SELECT * FROM testnet_pair_attempts WHERE attempt_id=?", (case["attempt_id"],)).fetchone()
                if attempt is None:
                    raise ValueError("qualification attempt missing")
                plan = json.loads(attempt["plan_json"])
                if (proof["environment"] != "TESTNET" or proof["evidence_origin"] != "NATIVE_TESTNET"
                        or proof["attempt_id"] != case["attempt_id"] or proof["case"] != kind.value
                        or proof["policy_sha256"] != data["policy_sha256"]
                        or proof["implementation_sha256"] != data["implementation_sha256"]
                        or _sha(attempt["plan_json"]) != attempt["plan_sha256"]
                        or proof["attempt_sha256"] != _sha(exact_json(dict(attempt)))
                        or plan["spot_entry"]["symbol"] != symbol or plan["perp_entry"]["symbol"] != symbol
                        or not registered <= _time(proof["started_at"]) <= _time(attempt["created_at"])
                        <= _time(attempt["updated_at"]) <= _time(proof["finished_at"]) <= reviewed
                        or dict(limits)[symbol] > _decimal(plan["spot_entry"]["quantity"], positive=True)):
                    raise ValueError("native case identity/build/timing/sizing mismatch")
                accounting = _native_accounting(db, attempt, proof)
                if not _case_matches(kind, attempt, proof, accounting, db):
                    raise ValueError("native branch was not demonstrated")
                found.append((symbol, kind, case["attempt_id"]))
        if identities != {(s, kind) for s in settings.symbols for kind in ExecutionRiskCase}:
            raise RiskReviewError(AdmissionReason.RISK_REVIEW_INCOMPLETE)
        return ExecutionRiskReview(reference.sha256, data["policy_sha256"], data["implementation_sha256"],
                                   reviewed, expires, limits, tuple(found))
    except RiskReviewError:
        raise
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, AttributeError,
            OverflowError, DecimalException) as exc:
        raise RiskReviewError(AdmissionReason.RISK_REVIEW_UNAVAILABLE) from exc
