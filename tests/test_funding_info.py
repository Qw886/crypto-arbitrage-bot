"""Public funding-configuration evidence, not hypothetical settlement income."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot.exchanges import binance_usdtm
from crypto_arbitrage_bot.exchanges.binance_usdtm import BinanceUsdtmClient
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.models.research_instrument import ResearchInstrument
from crypto_arbitrage_bot.models.usdtm import UsdtmErrorKind

NOW = datetime(2026, 10, 8, 12, tzinfo=UTC)
ROW = {"symbol": "THEUSDT", "adjustedFundingRateCap": "0.02500000",
       "adjustedFundingRateFloor": "-0.02500000", "fundingIntervalHours": 4,
       "disclaimer": False}


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "")
    monkeypatch.setenv("https_proxy", "")
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setattr(binance_usdtm, "utc_now", lambda: NOW)
    factory = Mock(side_effect=AssertionError("unconfigured HTTP must not reach network"))
    monkeypatch.setattr(binance_usdtm, "HTTPSConnection", factory)
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("offline only")))
    return factory


def response(factory, payload, *, status=200):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    reply = Mock(status=status)
    reply.getheaders.return_value = [("Content-Length", str(len(body)))]
    reply.read.return_value = body
    connection = Mock()
    connection.getresponse.return_value = reply
    factory.side_effect = None
    factory.return_value = connection
    return connection, body


def test_public_adjustment_catalog_keeps_decimal_raw_bytes_and_observation_times(transport):
    connection, body = response(transport, [ROW])
    result = BinanceUsdtmClient(3).fetch_funding_info()
    assert result.ok and result.error is None
    info = result.data[0]
    assert info.symbol == "THEUSDT" and info.funding_interval_hours == 4
    assert info.adjusted_funding_rate_cap == Decimal("0.02500000")
    assert info.adjusted_funding_rate_floor == Decimal("-0.02500000")
    assert isinstance(info.adjusted_funding_rate_cap, Decimal)
    assert info.observed_at == NOW and info.official_time is None
    assert result.raw_response == body
    assert result.started_at == result.received_at == result.finished_at == NOW
    assert result.request.endpoint is PublicEndpoint.PERP_FUNDING_INFO
    method, route = connection.request.call_args.args
    assert method == "GET" and route == "/fapi/v1/fundingInfo?"
    assert not any(name.lower() == "x-mbx-apikey" for name in connection.request.call_args.kwargs["headers"])


def test_empty_adjustment_list_is_valid_without_inventing_default_intervals(transport):
    response(transport, [])
    result = BinanceUsdtmClient(3).fetch_funding_info()
    assert result.ok and result.data == () and result.raw_response == b"[]"


def test_bulk_metadata_does_not_confuse_unicode_or_usdc_rows_with_execution_eligibility(transport):
    response(transport, [ROW, {**ROW, "symbol": "币安人生USDT"}, {**ROW, "symbol": "BTCUSDC"}])
    result = BinanceUsdtmClient(3).fetch_funding_info()
    assert result.ok and tuple(row.symbol for row in result.data) == ("THEUSDT", "币安人生USDT", "BTCUSDC")


@pytest.mark.parametrize("interval", [1, 4, 8])
def test_interval_is_preserved_not_normalized_to_eight_hours(transport, interval):
    response(transport, [{**ROW, "fundingIntervalHours": interval}])
    assert BinanceUsdtmClient(3).fetch_funding_info().data[0].funding_interval_hours == interval


@pytest.mark.parametrize("payload", [
    b"", b"null", b"{}", b"not-json", [None], [{}],
    [{**ROW, "fundingIntervalHours": 0}], [{**ROW, "fundingIntervalHours": -1}],
    [{**ROW, "fundingIntervalHours": True}], [{**ROW, "fundingIntervalHours": 4.0}],
    [{**ROW, "fundingIntervalHours": "4"}], [{**ROW, "adjustedFundingRateCap": 0.025}],
    [{**ROW, "adjustedFundingRateFloor": None}], [{**ROW, "adjustedFundingRateCap": "NaN"}],
    [{**ROW, "adjustedFundingRateFloor": "0.03"}], [{**ROW, "symbol": "THE/USDT"}],
    [ROW, ROW], [ROW, {"symbol": "BTCUSDT"}],
    b'[{"symbol":"THEUSDT","symbol":"BTCUSDT"}]',
    b'[{"symbol":"THEUSDT","fundingIntervalHours":NaN}]',
])
def test_malformed_or_partial_catalog_exposes_no_adjustments(transport, payload):
    response(transport, payload)
    result = BinanceUsdtmClient(3).fetch_funding_info()
    assert not result.ok and result.data is None
    assert result.error.kind is UsdtmErrorKind.INVALID_RESPONSE


@pytest.mark.parametrize("status,kind", [
    (400, UsdtmErrorKind.HTTP_ERROR), (429, UsdtmErrorKind.RATE_LIMITED),
    (500, UsdtmErrorKind.HTTP_ERROR), (301, UsdtmErrorKind.HTTP_ERROR),
])
def test_http_failures_keep_structured_semantics_without_following_redirects(transport, status, kind):
    connection, body = response(transport, [ROW], status=status)
    result = BinanceUsdtmClient(3).fetch_funding_info()
    assert result.data is None and result.error.kind is kind and result.raw_response == body
    assert connection.request.call_count == 1


@pytest.mark.parametrize("error,kind", [
    (TimeoutError(), UsdtmErrorKind.TIMEOUT),
    (ConnectionRefusedError(), UsdtmErrorKind.NETWORK_ERROR),
])
def test_network_failure_does_not_reuse_a_previous_catalog(transport, error, kind):
    connection, _ = response(transport, [ROW])
    client = BinanceUsdtmClient(3)
    assert client.fetch_funding_info().ok
    connection.request.side_effect = error
    failed = client.fetch_funding_info()
    assert failed.data is None and failed.error.kind is kind and failed.raw_response is None


def test_api_error_is_not_an_empty_successful_catalog(transport):
    response(transport, {"code": -1003, "msg": "synthetic API failure"})
    result = BinanceUsdtmClient(3).fetch_funding_info()
    assert result.data is None and result.error.kind is UsdtmErrorKind.API_ERROR
    assert result.error.api_code == -1003


def test_profile_entry_point_saves_only_requested_symbols_without_guessing_absent_values(transport, tmp_path):
    from pathlib import Path

    from crypto_arbitrage_bot.config.settings import load_settings
    from crypto_arbitrage_bot.funding_info_research import read_funding_profile

    _, body = response(transport, [ROW])
    output = tmp_path / "profile.json"
    summary = read_funding_profile(load_settings(Path("docs/research-multisettlement.toml")),
                                  ("BTCUSDT", "THEUSDT"), output, evaluated_at=lambda: NOW)
    report = json.loads(output.read_text(encoding="utf-8"))
    btc, the = summary["profiles"]
    assert btc["symbol"] == "BTCUSDT" and btc["adjustment_status"] == "NOT_REPORTED"
    assert btc["funding_interval_hours"] is None and btc["adjusted_funding_rate_cap"] is None
    assert the["symbol"] == "THEUSDT" and the["funding_interval_hours"] == 4
    assert the["adjusted_funding_rate_cap"] == Decimal("0.02500000")
    assert report["scope"] == "PUBLIC_FUNDING_CONFIGURATION_NOT_SETTLEMENT_OR_PNL"
    assert report["orders_submitted"] is False and report["research_config_version"] == "candidate-multisettlement-observation-v1"
    assert report["source"]["request"]["endpoint"] == PublicEndpoint.PERP_FUNDING_INFO.value
    assert summary["source_sha256"] == __import__("hashlib").sha256(body).hexdigest()


@pytest.mark.parametrize("status", [429, 500])
def test_profile_failure_is_saved_as_unavailable_never_default_zero_rates(transport, tmp_path, status):
    from pathlib import Path

    from crypto_arbitrage_bot.config.settings import load_settings
    from crypto_arbitrage_bot.funding_info_research import read_funding_profile

    response(transport, [ROW], status=status)
    summary = read_funding_profile(load_settings(Path("docs/research-multisettlement.toml")),
                                  ("THEUSDT",), tmp_path / "failed.json", evaluated_at=lambda: NOW)
    assert summary["status"] == "UNAVAILABLE" and "SOURCE_FAILED" in summary["reason_codes"]
    row = summary["profiles"][0]
    assert row["adjustment_status"] == "SOURCE_UNAVAILABLE"
    assert row["funding_interval_hours"] is None and row["adjusted_funding_rate_cap"] is None
    assert row["observed_at"] is None and row["official_time"] is None
    assert summary["source_error"] is not None


def test_profile_stale_response_does_not_claim_current_intervals(transport, tmp_path):
    from pathlib import Path

    from crypto_arbitrage_bot.config.settings import load_settings
    from crypto_arbitrage_bot.funding_info_research import read_funding_profile

    response(transport, [ROW])
    summary = read_funding_profile(load_settings(Path("docs/research-multisettlement.toml")),
        ("THEUSDT",), tmp_path / "stale.json", evaluated_at=lambda: NOW + timedelta(seconds=11))
    assert summary["status"] == "UNAVAILABLE" and "STALE_SOURCE" in summary["reason_codes"]
    assert summary["profiles"][0]["adjusted_funding_rate_cap"] is None


def test_profile_will_not_overwrite_old_evidence_or_make_an_extra_get(transport, tmp_path):
    from pathlib import Path

    from crypto_arbitrage_bot.config.settings import load_settings
    from crypto_arbitrage_bot.funding_info_research import read_funding_profile

    output = tmp_path / "existing.json"
    output.write_bytes(b"original evidence")
    with pytest.raises(FileExistsError):
        read_funding_profile(load_settings(Path("docs/research-multisettlement.toml")), ("THEUSDT",), output)
    assert output.read_bytes() == b"original evidence"
    transport.assert_not_called()


@pytest.mark.parametrize("symbols", [(), ("THEUSDT", "THEUSDT"), ("THEUSDC",),
                                    ("1000SHIBUSDT",), ("theusdt",)])
def test_invalid_or_scaled_research_identity_is_rejected_before_io(transport, tmp_path, symbols):
    from pathlib import Path

    from crypto_arbitrage_bot.config.settings import load_settings
    from crypto_arbitrage_bot.funding_info_research import read_funding_profile

    with pytest.raises(ValueError):
        read_funding_profile(load_settings(Path("docs/research-multisettlement.toml")), symbols, tmp_path / "no.json")
    transport.assert_not_called()
    assert not (tmp_path / "no.json").exists()


def test_single_instrument_client_cannot_request_bulk_adjustments(transport):
    with pytest.raises(ValueError):
        BinanceUsdtmClient(3, research_instrument=ResearchInstrument("THEUSDT", "THE")).fetch_funding_info()
    transport.assert_not_called()


def test_profile_cli_uses_public_get_without_sqlite_or_credentials(transport, monkeypatch, tmp_path, capsys):
    from crypto_arbitrage_bot.funding_info_research import main

    response(transport, [ROW])
    # Native clock is external; use it here for the CLI's observation-age check.
    monkeypatch.setattr(binance_usdtm, "utc_now", lambda: datetime.now(UTC))
    monkeypatch.setenv("CAB_TESTNET_SPOT_API_KEY", "offline-test-sentinel")
    monkeypatch.setattr("sqlite3.connect", Mock(side_effect=AssertionError("profile must not touch SQLite")))
    code = main(["--config", "docs/research-multisettlement.toml", "--symbols", "THEUSDT", "BTCUSDT",
                 "--output", str(tmp_path / "cli.json")])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "COMPLETE" and summary["orders_submitted"] is False
    assert summary["profiles"][0]["adjusted_funding_rate_cap"] == "0.02500000"
    assert summary["profiles"][1]["funding_interval_hours"] is None
    assert "offline-test-sentinel" not in json.dumps(summary)
