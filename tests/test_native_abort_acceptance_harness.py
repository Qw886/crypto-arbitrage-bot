"""Controlled local interruption cannot invent ACKs or replay entries on resume."""

import importlib.util
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_funding_lifecycle import live_clock_bot as live_clock_bot

from crypto_arbitrage_bot.execution.balance_check_repository import BalanceCheck, BalanceCheckStatus
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError

spec=importlib.util.spec_from_file_location("native_abort_acceptance",Path("work/testnet_abort_native_acceptance.py"))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("socket.create_connection",Mock(side_effect=AssertionError("offline")))


def test_fault_injection_preserves_real_callback_receipt_unmodified_and_submits_once(tmp_path):
    receipt={"example":"actual fake-venue ACK; never synthesized by harness"}
    underlying=Mock(return_value=receipt)
    bot=SimpleNamespace(spot=SimpleNamespace(place_ioc_limit_order=underlying),clock=lambda:"timestamp")
    order=object()
    original,submit,calls=module.interrupted_gateway(bot,SimpleNamespace(spot_entry=order),tmp_path/"ack.json")
    assert original is underlying
    with pytest.raises(module.LocalAckInterruption):
        submit(order)
    saved=json.loads((tmp_path/"ack.json").read_text(encoding="utf-8"))
    assert saved["receipt"]==receipt and calls==[1]
    with pytest.raises(ExecutionGuardError,match="REPLAY"):
        submit(order)
    assert underlying.call_count==1


def test_wrong_order_is_rejected_before_any_submit(tmp_path):
    underlying=Mock()
    bot=SimpleNamespace(spot=SimpleNamespace(place_ioc_limit_order=underlying),clock=Mock())
    _,submit,calls=module.interrupted_gateway(bot,SimpleNamespace(spot_entry=object()),tmp_path/"ack.json")
    with pytest.raises(ExecutionGuardError):
        submit(object())
    assert not underlying.called and calls==[0] and not (tmp_path/"ack.json").exists()


def test_upstream_error_is_not_called_an_injected_ack(tmp_path):
    underlying=Mock(side_effect=TimeoutError("offline original submit"))
    order=object()
    bot=SimpleNamespace(spot=SimpleNamespace(place_ioc_limit_order=underlying),clock=Mock())
    _,submit,calls=module.interrupted_gateway(bot,SimpleNamespace(spot_entry=order),tmp_path/"ack.json")
    with pytest.raises(TimeoutError):
        submit(order)
    assert calls==[1] and not (tmp_path/"ack.json").exists()


def test_existing_report_without_plan_never_restarts_preflight_or_entry(tmp_path,monkeypatch):
    report=tmp_path/"report.json"
    module.save(report,{"scope":"TESTNET_CONTROLLED_ACK_INTERRUPTION_NATIVE_ABORT",
                       "symbol":"ETHUSDT","attempt_id":"durable","phase":"PREFLIGHT"})
    bot=SimpleNamespace(journal=SimpleNamespace(plans=lambda:[]))
    entry=Mock(side_effect=AssertionError("must not replay entry"))
    monkeypatch.setattr(module,"start_once",entry)
    with pytest.raises(ExecutionGuardError,match="RECOVERY_ONLY"):
        module.run(bot,report)
    assert not entry.called


def test_resume_manages_same_plan_without_submitting_an_entry(tmp_path,monkeypatch):
    report=tmp_path/"report.json"
    plan=SimpleNamespace(attempt_id="durable")
    module.save(report,{"scope":"TESTNET_CONTROLLED_ACK_INTERRUPTION_NATIVE_ABORT",
                       "symbol":"ETHUSDT","attempt_id":"durable","phase":"RECOVERING"})
    bot=SimpleNamespace(journal=SimpleNamespace(plans=lambda:[(plan,None)]),stop=Event(),
                        _managed_exits=Mock(),clock=lambda:"timestamp")
    entry=Mock(side_effect=AssertionError("must not replay entry"))
    ledger=Mock()
    monkeypatch.setattr(module,"start_once",entry)
    monkeypatch.setattr(module,"reconcile_balance_ledger",ledger)
    monkeypatch.setattr(module,"outcome",lambda current,p:{"status":"PASS","balance_check":None})
    result=module.run(bot,report,wait=Mock(),monotonic=lambda:0)
    assert result["status"]=="PASS" and not entry.called
    assert bot._managed_exits.call_count==ledger.call_count==1


def test_failed_balance_budget_stops_without_reset_or_extra_entry(tmp_path,monkeypatch):
    report=tmp_path/"report.json"
    plan=SimpleNamespace(attempt_id="durable")
    module.save(report,{"scope":"TESTNET_CONTROLLED_ACK_INTERRUPTION_NATIVE_ABORT",
                       "symbol":"ETHUSDT","attempt_id":"durable","phase":"RECOVERING"})
    bot=SimpleNamespace(journal=SimpleNamespace(plans=lambda:[(plan,None)]),stop=Event(),
                        _managed_exits=Mock(),clock=lambda:"timestamp")
    check=BalanceCheck("durable",BalanceCheckStatus.FAILED,3,datetime(2026,10,6,tzinfo=UTC),
                       None,None,None,None,"BUDGET_EXHAUSTED")
    monkeypatch.setattr(module,"reconcile_balance_ledger",Mock())
    monkeypatch.setattr(module,"outcome",lambda *args:{"status":"NEEDS_ATTENTION","balance_check":check})
    wait=Mock()
    result=module.run(bot,report,wait=wait,monotonic=lambda:0)
    assert result["status"]=="NEEDS_ATTENTION" and not wait.called
    assert bot._managed_exits.call_count==1


@pytest.mark.parametrize("duration",[0,301,True,1.5])
def test_acceptance_is_bounded_before_any_account_or_order_access(tmp_path,duration):
    with pytest.raises(ValueError):
        module.run(None,tmp_path/"report.json",duration_seconds=duration)


@pytest.mark.parametrize("mode,case,status",[
    ("passive","NO_FILL","PASS"),
    ("bounded_marketable","NO_FILL","CASE_NOT_OBSERVED"),
    ("bounded_marketable","PARTIAL_FILL","PASS"),
    ("bounded_marketable","FULL_FILL","PASS"),
])
def test_acceptance_classifies_observed_fill_without_inventing_success(mode,case,status):
    quantities={"NO_FILL":"0","PARTIAL_FILL":".005","FULL_FILL":".01"}
    receipt=SimpleNamespace(executed_quantity=Decimal(quantities[case]),original_quantity=Decimal(".01"))
    source={"status":"PASS","pair":SimpleNamespace(spot_entry=receipt)}
    result=module.classify_result(source,mode)
    assert result["status"]==status and result["observed_entry_case"]==case
    assert source["status"]=="PASS" and "observed_entry_case" not in source
    assert result["pair"] is source["pair"]


@pytest.mark.parametrize("quantity",[None,0.0,Decimal("NaN"),Decimal("-.01"),Decimal(".02")])
def test_missing_or_invalid_quantity_cannot_pass_filled_acceptance(quantity):
    source={"status":"PASS","pair":SimpleNamespace(spot_entry=SimpleNamespace(
        executed_quantity=quantity,original_quantity=Decimal(".01")))}
    result=module.classify_result(source,"bounded_marketable")
    assert result["status"]=="NEEDS_ATTENTION" and result["observed_entry_case"]=="UNKNOWN"


def test_filled_receipt_without_accounting_proof_is_not_pass():
    source={"status":"NEEDS_ATTENTION","pair":SimpleNamespace(spot_entry=SimpleNamespace(
        executed_quantity=Decimal(".01"),original_quantity=Decimal(".01")))}
    result=module.classify_result(source,"bounded_marketable")
    assert result["status"]=="NEEDS_ATTENTION" and result["observed_entry_case"]=="FULL_FILL"


@pytest.mark.parametrize("mode",["market",True,1,"",[]])
def test_invalid_entry_mode_rejected_before_preflight(tmp_path,mode):
    with pytest.raises(ValueError,match="entry mode"):
        module.run(None,tmp_path/"report.json",entry_mode=mode)


@pytest.mark.parametrize("mode",["passive","bounded_marketable"])
def test_entry_modes_use_existing_bounded_plan_and_only_one_spot_submission(live_clock_bot,tmp_path,mode):
    bot,transport,_=live_clock_bot
    bot.config=bot.config.model_copy(update={"max_price_deviation":Decimal(".001")})
    original=bot.spot.place_ioc_limit_order
    report=tmp_path/"report.json"
    state={"phase":"PREFLIGHT","attempt_id":"offline-bounded-abort","entry_mode":mode}
    plan=module.start_once(bot,report,state)
    saved=json.loads(report.with_suffix(".ack.json").read_text(encoding="utf-8"))
    assert saved["receipt"]["executed_quantity"]=="0.01"
    assert state["entry_mode"]==mode and state["local_gateway_entry_calls"]==1
    assert state["initial_pair"].failure_code=="SPOT_SUBMISSION_UNKNOWN"
    assert bot.spot.place_ioc_limit_order==original
    assert len(transport.orders)==1
    assert not any(method=="POST" and "demo-fapi" in url for method,url in transport.requests)
    assert not any(intent.market.value=="USDTM" for intent in bot.journal.intents(plan.attempt_id))
    if mode=="bounded_marketable":
        assert plan.spot_entry.limit_price>Decimal("2000")
    else:
        assert plan.spot_entry.limit_price<Decimal("2000")


def test_report_mode_cannot_be_changed_on_resume(tmp_path):
    report=tmp_path/"report.json"
    module.save(report,{"scope":"TESTNET_CONTROLLED_ACK_INTERRUPTION_NATIVE_ABORT",
                       "symbol":"ETHUSDT","attempt_id":"durable","phase":"RECOVERING",
                       "entry_mode":"passive"})
    with pytest.raises(ExecutionGuardError,match="MODE_MISMATCH"):
        module.run(None,report,entry_mode="bounded_marketable")


def test_zero_fill_bounded_case_stops_without_retrying_entry(tmp_path,monkeypatch):
    report=tmp_path/"report.json"
    plan=SimpleNamespace(attempt_id="durable")
    module.save(report,{"scope":"TESTNET_CONTROLLED_ACK_INTERRUPTION_NATIVE_ABORT",
                       "symbol":"ETHUSDT","attempt_id":"durable","phase":"RECOVERING",
                       "entry_mode":"bounded_marketable"})
    bot=SimpleNamespace(journal=SimpleNamespace(plans=lambda:[(plan,None)]),stop=Event(),
                        _managed_exits=Mock(),clock=lambda:"timestamp")
    entry=Mock(side_effect=AssertionError("must not replay entry"))
    monkeypatch.setattr(module,"start_once",entry)
    monkeypatch.setattr(module,"reconcile_balance_ledger",Mock())
    monkeypatch.setattr(module,"outcome",lambda *args:{"status":"PASS","balance_check":None,
        "pair":SimpleNamespace(spot_entry=SimpleNamespace(executed_quantity=Decimal(0),original_quantity=Decimal(".01")))})
    # The minimal mock pair isn't JSON-serializable; serialization is separately
    # tested with actual domain receipts above. This verifies loop termination.
    monkeypatch.setattr(module,"save",Mock())
    wait=Mock()
    result=module.run(bot,report,wait=wait,monotonic=lambda:0)
    assert result["status"]=="CASE_NOT_OBSERVED" and not wait.called and not entry.called
    assert bot._managed_exits.call_count==1
