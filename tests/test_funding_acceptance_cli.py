"""The controlled execution and its read-only accounting run sequentially."""

from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot.execution import acceptance_accounting, funding_acceptance, main
from crypto_arbitrage_bot.execution.acceptance_accounting import AccountingFollowupResult
from crypto_arbitrage_bot.execution.acceptance_accounting import AccountingFollowupStatus as Status


@pytest.mark.parametrize("status,accounting_status,expected", [(Status.SAVED, "RECONCILED", 0),
    (Status.SAVED, "INCOMPLETE", 1), (Status.FAILED, None, 1), (Status.SKIPPED, None, 1)])
def test_cli_finishes_acceptance_before_accounting(monkeypatch, tmp_path, status, accounting_status, expected):
    monkeypatch.setattr("sys.argv", ["bot", "--config", "docs/testnet-funding-acceptance.toml",
        "--funding-acceptance", "--account-after-acceptance", "--acceptance-report", str(tmp_path / "report.json"),
        "--acceptance-max-wait-seconds", "21600", "--acceptance-preflight-attempts", "3"])
    monkeypatch.setenv("CAB_TESTNET_ORDERS_ENABLED", "true")
    monkeypatch.setattr(main, "configure_logging", Mock())
    monkeypatch.setattr(main, "execution_environment", Mock(return_value={}))
    bot = Mock()
    monkeypatch.setattr(main, "create_testnet_bot", Mock(return_value=bot))
    call_order = []
    def run():
        call_order.append("acceptance_finished")
        return "VERIFIED"
    acceptance = Mock()
    acceptance.run.side_effect = run
    constructor = Mock(return_value=acceptance)
    monkeypatch.setattr(funding_acceptance, "FundingAcceptance", constructor)
    def account(*args, **kwargs):
        assert call_order == ["acceptance_finished"]
        assert args[0] is bot and kwargs["max_wait_seconds"] == 1
        call_order.append("accounted")
        return AccountingFollowupResult(status, "existing-attempt", accounting_status)
    monkeypatch.setattr(acceptance_accounting, "reconcile_after_acceptance", account)
    assert main.main() == expected
    assert constructor.call_args.kwargs["max_wait_seconds"] == 21600
    assert constructor.call_args.kwargs["max_preflight_attempts"] == 3
    assert call_order == ["acceptance_finished", "accounted"]
    bot.spot.place_ioc_limit_order.assert_not_called()
    bot.perp.place_ioc_limit_order.assert_not_called()


def test_accounting_option_cannot_enable_another_execution_mode(monkeypatch):
    monkeypatch.setattr("sys.argv", ["bot", "--forever", "--account-after-acceptance"])
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code == 2
