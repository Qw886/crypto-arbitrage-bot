"""Demo account FUNDING_FEE ledger, independent of scenario forecasts."""

from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from enum import StrEnum
from typing import TYPE_CHECKING

from crypto_arbitrage_bot.execution.pair_coordinator import PairExecutionState
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

if TYPE_CHECKING:
    from crypto_arbitrage_bot.execution.runner import TestnetBot


class FundingCheckStatus(StrEnum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    UNAVAILABLE = "UNAVAILABLE"


def reconcile_funding_ledger(bot: "TestnetBot") -> None:
    """Read after close. Missing payments remain NULL with bounded retry.

    A ledger payment has the account's signed amount; no quantity × forecast
    multiplication is used here. Forecasted PnL/Decision remain immutable.
    """
    now = utc_timestamp(bot.clock())
    with bot.journal._connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS testnet_funding_checks (
            attempt_id TEXT PRIMARY KEY, status TEXT NOT NULL,
            attempts INTEGER NOT NULL, next_check_at TEXT NOT NULL,
            actual_funding TEXT, payments_json TEXT, failure_code TEXT,
            checked_at TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS testnet_funding_payments (
            transaction_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL,
            symbol TEXT NOT NULL, posted_at TEXT NOT NULL, amount TEXT NOT NULL)""")
    for plan, opened_at in bot.journal.plans():
        if plan.funding_settlement_at is None or plan.funding_settlement_at > now:
            continue  # Legacy acceptance round trips have no forecast/settlement.
        with bot.journal._connect() as db:
            parent = db.execute("SELECT state, updated_at FROM testnet_pair_attempts WHERE attempt_id=?",
                                (plan.attempt_id,)).fetchone()
            prior = db.execute("SELECT * FROM testnet_funding_checks WHERE attempt_id=?", (plan.attempt_id,)).fetchone()
        if parent is None or parent["state"] != PairExecutionState.CLOSED.value:
            continue  # Closing exposure takes priority over accounting.
        if prior is not None and (prior["status"] in ("VERIFIED", "UNAVAILABLE")
                                 or datetime.fromisoformat(prior["next_check_at"]) > now):
            continue
        attempts = 1 if prior is None else prior["attempts"] + 1
        closed_at = datetime.fromisoformat(parent["updated_at"])
        status = FundingCheckStatus.PENDING
        failure: str | None = "FUNDING_PAYMENT_NOT_FOUND"
        total: str | None = None
        encoded: str | None = None
        try:
            payments = bot.perp.get_funding_payments(plan.symbol, opened_at, closed_at)
            if payments:
                with localcontext() as ctx:
                    ctx.prec = 50
                    total = str(sum((payment.amount for payment in payments), Decimal("0")))
                encoded = exact_json(payments)
                with bot.journal._connect() as db:
                    for payment in payments:
                        old = db.execute("SELECT attempt_id,symbol,posted_at,amount FROM testnet_funding_payments WHERE transaction_id=?",
                                         (payment.transaction_id,)).fetchone()
                        if old is not None and (old[0] != plan.attempt_id or old[1] != payment.symbol
                                or datetime.fromisoformat(old[2]) != payment.posted_at
                                or Decimal(old[3]) != payment.amount):
                            raise ValueError("funding payment attribution or payload conflict")
                    for payment in payments:
                        db.execute("INSERT OR IGNORE INTO testnet_funding_payments VALUES(?,?,?,?,?)",
                                   (payment.transaction_id, plan.attempt_id, payment.symbol,
                                    payment.posted_at.isoformat(), str(payment.amount)))
                status, failure = FundingCheckStatus.VERIFIED, None
        except Exception as error:
            status, total, encoded = FundingCheckStatus.PENDING, None, None
            failure = getattr(getattr(error, "kind", None), "value", type(error).__name__)
        if status is FundingCheckStatus.PENDING and attempts >= bot.config.funding_check_max_attempts:
            status = FundingCheckStatus.UNAVAILABLE
        with bot.journal._connect() as db:
            db.execute("INSERT INTO testnet_funding_checks VALUES(?,?,?,?,?,?,?,?) "
                       "ON CONFLICT(attempt_id) DO UPDATE SET status=excluded.status,attempts=excluded.attempts,"
                       "next_check_at=excluded.next_check_at,actual_funding=excluded.actual_funding,"
                       "payments_json=excluded.payments_json,failure_code=excluded.failure_code,checked_at=excluded.checked_at",
                       (plan.attempt_id, status, attempts,
                        (now + timedelta(seconds=bot.config.funding_check_interval_seconds)).isoformat(),
                        total, encoded, failure, now.isoformat()))
        bot.logger.info(exact_json({"stage": "TESTNET_FUNDING_LEDGER", "attempt_id": plan.attempt_id,
            "status": status, "actual_funding": total, "failure_code": failure,
            "meaning": "Demo account funding only; not complete trade net PnL"}))
