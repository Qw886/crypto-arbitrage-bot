"""Explicit OFFLINE wiring assumption; never an actual strategy/risk proof.

Used only by old order/fee/funding/quote tests to isolate those existing guards
from the new independent admission guard. Production code has no bypass flag.
"""

from crypto_arbitrage_bot.models.strategy_admission import (
    AdmissionReason,
    AdmissionStatus,
    StrategyAdmissionResult,
)


def assume_offline_admission(monkeypatch):
    def eligible(settings, record, *, evaluated_at):
        return StrategyAdmissionResult(AdmissionStatus.ELIGIBLE_FOR_TESTNET_PREFLIGHT,
            record.symbol, record.id, record.matched_quantity, evaluated_at,
            (AdmissionReason.ALL_EVIDENCE_RULES_MET,))
    monkeypatch.setattr("crypto_arbitrage_bot.execution.runner.review_automatic_entry", eligible)
    monkeypatch.setattr("crypto_arbitrage_bot.execution.runner.refresh_entry_admission", lambda result, *args, **kwargs: result)
