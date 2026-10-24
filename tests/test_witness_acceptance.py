"""CLI/session integration uses virtual offline fixtures, never native qualification."""

import json
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest
from test_funding_lifecycle import live_clock_bot as funding_clock_fixture

from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
from crypto_arbitrage_bot.execution import main
from crypto_arbitrage_bot.execution.funding_acceptance import FundingAcceptance
from crypto_arbitrage_bot.execution.risk_witness import (
    RiskWitness,
    WitnessError,
    WitnessFailure,
    load_witness,
    read_witness_events,
)
from crypto_arbitrage_bot.execution.runner import ExecutionGuardError
from crypto_arbitrage_bot.execution.types import FundingPayment
from crypto_arbitrage_bot.execution.witness_acceptance import (
    observe_acceptance,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(
        "socket.create_connection", Mock(side_effect=AssertionError("offline witness acceptance"))
    )


@pytest.fixture(name="live_clock_bot")
def local_clock_fixture(tmp_path, monkeypatch):
    return funding_clock_fixture.__wrapped__(tmp_path, monkeypatch)


def run_fixture(fixture, tmp_path, monkeypatch, *, stop_after_entry=False):
    bot, transport, state = fixture
    state["rate"] = "0"
    acceptance = FundingAcceptance(bot, tmp_path / "report.json")

    def wait(interval):
        assert interval == bot.settings.poll_interval_seconds > 0
        if acceptance.state["status"] == "WAITING":
            state["now"] = state["settlement"] - timedelta(seconds=180)
        elif stop_after_entry:
            bot.stop.set()
            return True
        else:
            state["now"] = state["settlement"] + timedelta(seconds=61)
        return False

    monkeypatch.setattr(bot.stop, "wait", wait)
    monkeypatch.setattr(
        bot.perp,
        "get_funding_payments",
        Mock(
            return_value=(
                FundingPayment(
                    "explicit-fixture-payment", "ETHUSDT", state["settlement"], Decimal("0.01")
                ),
            )
        ),
    )
    return bot, transport, state, acceptance


def registration(acceptance):
    ref = acceptance.state["witness_registration"]
    return load_witness(StudyRegistrationSettings(**ref))


def test_complete_original_acceptance_auto_registers_records_and_exports(
    live_clock_bot, tmp_path, monkeypatch
):
    bot, transport, state, acceptance = run_fixture(live_clock_bot, tmp_path, monkeypatch)
    assert acceptance.run(witness_directory=tmp_path / "witness") == "VERIFIED"
    assert acceptance.state["witness_status"] == "EXPORTED"
    saved = registration(acceptance)
    events = read_witness_events(saved)
    assert saved.origin == acceptance.state["witness_origin"] == "OFFLINE_FIXTURE"
    assert [e["kind"] for e in events].count("SESSION_BEGIN") == 1
    assert [e["kind"] for e in events].count("SESSION_END") == 1
    assert [e["kind"] for e in events].count("POST_RESULT") == 4
    assert len(transport.orders) == 4
    assert (
        bot.repository.get_opportunity_by_id(acceptance.state["opportunity_id"]).decision
        == "REJECT"
    )
    trace = json.loads(saved.reference.path.parent.joinpath("trace.json").read_text())
    assert (
        trace["origin"] == "OFFLINE_FIXTURE"
        and trace["registration"]["sha256"] == saved.reference.sha256
    )
    assert all(
        isinstance(e["receipt"]["executed_quantity"], str)
        for e in events
        if e["kind"] == "POST_RESULT"
    )


def test_graceful_stop_resume_keeps_attempt_expiry_and_no_duplicate_entry(
    live_clock_bot, tmp_path, monkeypatch
):
    bot, transport, state, acceptance = run_fixture(
        live_clock_bot, tmp_path, monkeypatch, stop_after_entry=True
    )
    assert acceptance.run(witness_directory=tmp_path / "witness") == "HOLDING"
    assert acceptance.state["witness_status"] == "STOPPED" and len(transport.orders) == 2
    original = registration(acceptance)
    bot.stop.clear()
    state["now"] = state["settlement"] + timedelta(seconds=61)
    resumed = FundingAcceptance(bot, acceptance.report_path)
    assert (
        resumed.run() == "VERIFIED"
    )  # Existing reference reenables capture without the CLI option.
    assert registration(resumed) == original and len(transport.orders) == 4
    assert resumed.state["witness_status"] == "EXPORTED"
    assert [e["kind"] for e in read_witness_events(original)].count("SESSION_BEGIN") == 2
    before = original.reference.path.parent.joinpath("trace.json").read_bytes()
    assert resumed.run(witness_directory=tmp_path / "witness") == "VERIFIED"
    assert (
        original.reference.path.parent.joinpath("trace.json").read_bytes() == before
        and len(transport.orders) == 4
    )


def test_idle_polling_does_not_consume_witness_limit(live_clock_bot, tmp_path, monkeypatch):
    bot, transport, state = live_clock_bot
    acceptance = FundingAcceptance(bot, tmp_path / "report.json")
    waits = []

    def wait(interval):
        waits.append(interval)
        if len(waits) == 250:
            bot.stop.set()
            return True
        return False

    monkeypatch.setattr(bot.stop, "wait", wait)
    assert acceptance.run(witness_directory=tmp_path / "witness") == "WAITING"
    assert len(read_witness_events(registration(acceptance))) == 2
    assert len(waits) == 250 and transport.orders == {}


def test_registration_before_gateway_and_journal_claim(live_clock_bot, tmp_path):
    bot, transport, state = live_clock_bot
    acceptance = FundingAcceptance(bot, tmp_path / "report.json")
    with observe_acceptance(acceptance, tmp_path / "witness"):
        saved = registration(acceptance)
        assert saved.created_at <= state["now"] and bot.journal.get(saved.attempt_id) is None
        assert transport.orders == {} and read_witness_events(saved)[0]["kind"] == "SESSION_BEGIN"
    assert acceptance.state["witness_status"] == "STOPPED"


@pytest.mark.parametrize("failure", [WitnessFailure.IO_ERROR, WitnessFailure.CORRUPT_TRACE])
def test_capture_setup_failure_blocks_entry(live_clock_bot, tmp_path, monkeypatch, failure):
    bot, transport, state, acceptance = run_fixture(live_clock_bot, tmp_path, monkeypatch)
    monkeypatch.setattr(RiskWitness, "register", Mock(side_effect=WitnessError(failure)))
    assert acceptance.run(witness_directory=tmp_path / "witness") == "ABORTED"
    assert (
        acceptance.state["witness_status"] == "FAILED"
        and acceptance.state["witness_failure"] == failure.value
    )
    assert not transport.orders and bot.journal.plans() == ()


@pytest.mark.parametrize("failure", ["deleted_registration", "expired", "broken", "policy_changed"])
def test_failed_capture_on_resume_does_not_block_original_exit(
    live_clock_bot, tmp_path, monkeypatch, failure
):
    bot, transport, state, acceptance = run_fixture(
        live_clock_bot, tmp_path, monkeypatch, stop_after_entry=True
    )
    acceptance.run(witness_directory=tmp_path / "witness")
    saved = registration(acceptance)
    bot.stop.clear()
    state["now"] = state["settlement"] + timedelta(seconds=61)
    if failure == "deleted_registration":
        saved.reference.path.unlink()
    elif failure == "expired":
        state["now"] = saved.expires_at + timedelta(seconds=1)
    elif failure == "broken":
        saved.events_directory.joinpath("00001.json").unlink()
    else:
        monkeypatch.setattr(
            "crypto_arbitrage_bot.execution.risk_witness.implementation_fingerprint",
            lambda: "0" * 64,
        )
    resumed = FundingAcceptance(bot, acceptance.report_path)
    # The original actual funding result is distinct from failed capture.
    assert resumed.run() == "VERIFIED"
    assert resumed.state["witness_status"] == "FAILED"
    assert bot.journal.get(str(resumed.state["attempt_id"])).state.value == "CLOSED"
    assert (
        len(transport.orders) == 4 and bot.perp.get_account_state("ETHUSDT").position_quantity == 0
    )


def test_cannot_retroactively_register_existing_entry(live_clock_bot, tmp_path, monkeypatch):
    bot, transport, state, acceptance = run_fixture(
        live_clock_bot, tmp_path, monkeypatch, stop_after_entry=True
    )
    assert acceptance.run() == "HOLDING"
    bot.stop.clear()
    state["now"] = state["settlement"] + timedelta(seconds=61)
    resumed = FundingAcceptance(bot, acceptance.report_path)
    assert resumed.run(witness_directory=tmp_path / "witness") == "VERIFIED"
    assert resumed.state["witness_status"] == "FAILED"
    assert resumed.state["witness_failure"] == WitnessFailure.ATTEMPT_ALREADY_EXISTS.value
    assert not (tmp_path / "witness").exists() and len(transport.orders) == 4


def test_exhausted_original_budget_prevents_even_witness_registration(
    live_clock_bot, tmp_path, monkeypatch
):
    bot, transport, state = live_clock_bot
    monkeypatch.setattr(
        bot,
        "_assert_open_allowed",
        Mock(side_effect=ExecutionGuardError("EXECUTION_BUDGET_EXHAUSTED")),
    )
    register = Mock(wraps=RiskWitness.register)
    monkeypatch.setattr(RiskWitness, "register", register)
    acceptance = FundingAcceptance(bot, tmp_path / "report.json")
    assert acceptance.run(witness_directory=tmp_path / "witness") == "ABORTED"
    register.assert_not_called()
    assert not transport.orders


def test_export_write_failure_keeps_actual_terminal_result(live_clock_bot, tmp_path, monkeypatch):
    bot, transport, state, acceptance = run_fixture(live_clock_bot, tmp_path, monkeypatch)
    monkeypatch.setattr(
        RiskWitness, "export_trace", Mock(side_effect=OSError("PRIVATE_RESPONSE_SECRET"))
    )
    assert acceptance.run(witness_directory=tmp_path / "witness") == "VERIFIED"
    assert (
        acceptance.state["witness_status"] == "FAILED"
        and acceptance.state["actual_funding"] is not None
    )
    assert (
        len(transport.orders) == 4
        and "PRIVATE_RESPONSE_SECRET" not in acceptance.report_path.read_text()
    )


@pytest.mark.parametrize("mode", ["--once", "--forever", "--round-trip", "--check", "--demo"])
def test_witness_flag_cannot_enable_other_modes(monkeypatch, mode):
    monkeypatch.setattr("sys.argv", ["bot", mode, "--witness-directory", "forbidden"])
    create = Mock(side_effect=AssertionError("must not create client"))
    monkeypatch.setattr(main, "create_testnet_bot", create)
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code == 2
    create.assert_not_called()


def test_cli_exposes_capture_reference_and_no_qualification(
    live_clock_bot, tmp_path, monkeypatch, capsys
):
    bot, transport, state, acceptance = run_fixture(live_clock_bot, tmp_path, monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "bot",
            "--config",
            "docs/testnet-funding.toml",
            "--funding-acceptance",
            "--acceptance-report",
            str(acceptance.report_path),
            "--witness-directory",
            str(tmp_path / "witness"),
        ],
    )
    monkeypatch.setenv("CAB_TESTNET_ORDERS_ENABLED", "true")
    monkeypatch.setattr(main, "execution_environment", lambda *a: {})
    monkeypatch.setattr(main, "create_testnet_bot", lambda *a, **k: bot)
    monkeypatch.setattr(main, "configure_logging", Mock())
    monkeypatch.setattr(
        "crypto_arbitrage_bot.execution.funding_acceptance.FundingAcceptance",
        lambda *a, **k: acceptance,
    )
    assert main.main() == 0
    output = capsys.readouterr().out
    assert "OFFLINE_FIXTURE" in output and '"risk_qualification_issued":false' in output


def test_stopped_before_run_makes_no_registration_or_network(live_clock_bot, tmp_path):
    bot, transport, state = live_clock_bot
    bot.stop.set()
    calls = len(transport.requests)
    acceptance = FundingAcceptance(bot, tmp_path / "report.json")
    assert acceptance.run(witness_directory=tmp_path / "witness") == "CANCELLED"
    assert len(transport.requests) == calls and not (tmp_path / "witness").exists()


def test_unexpected_interrupt_remains_broken_on_restart(live_clock_bot, tmp_path):
    bot, transport, state = live_clock_bot
    acceptance = FundingAcceptance(bot, tmp_path / "report.json")
    with pytest.raises(KeyboardInterrupt), observe_acceptance(acceptance, tmp_path / "witness"):
        raise KeyboardInterrupt()
    saved = registration(acceptance)
    assert read_witness_events(saved)[-1]["capture_complete"] is False
    resumed = FundingAcceptance(bot, acceptance.report_path)
    assert resumed.run() == "ABORTED"
    assert resumed.state["witness_status"] == "FAILED" and transport.orders == {}


def test_capture_failure_after_spot_fill_preserves_safety_flatten(
    live_clock_bot, tmp_path, monkeypatch
):
    bot, transport, state, acceptance = run_fixture(live_clock_bot, tmp_path, monkeypatch)
    original = RiskWitness._event

    def event(self, kind, **fields):
        if kind == "POST_RESULT" and fields["receipt"]["market"] == "SPOT":
            raise WitnessError(WitnessFailure.IO_ERROR)
        return original(self, kind, **fields)

    monkeypatch.setattr(RiskWitness, "_event", event)
    assert acceptance.run(witness_directory=tmp_path / "witness") == "ABORTED"
    assert acceptance.state["witness_status"] == "FAILED"
    assert transport.positions["ETHUSDT"] == transport.base["ETHUSDT"] == 0
    assert len(transport.orders) == 2  # Spot entry + original safe sell-back; no Perp fill.
    assert not registration(acceptance).reference.path.parent.joinpath("trace.json").exists()
