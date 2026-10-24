"""Offline acceptance for the separate long-running research configuration."""

import hashlib
import os
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from crypto_arbitrage_bot.config.settings import Settings, load_settings
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.exchanges.http_client import prepare_get, validate_public_request
from crypto_arbitrage_bot.storage.sqlite_repository import SQLiteConfig, SQLiteOpportunityRepository

ROOT = Path(__file__).resolve().parents[1]
SOAK = ROOT / "docs/research-soak.toml"
SMOKE = ROOT / "docs/acceptance-smoke.toml"


@pytest.fixture(autouse=True)
def isolate_settings_environment(monkeypatch):
    for name in os.environ:
        if name.upper().startswith("CAB_"):
            monkeypatch.delenv(name)


def test_research_soak_loads_expected_paths_interval_and_horizons():
    settings = load_settings(SOAK)
    assert settings.symbols == ("BTCUSDT", "ETHUSDT")
    assert settings.poll_interval_seconds == 30
    assert settings.log_dir == Path("logs/research")
    assert settings.research.sqlite_path == Path("data/research/soak.sqlite")
    assert settings.research.config_version == "research-soak-v1"
    assert settings.research.validation_scheduler.validation_horizons_seconds == (3600, 28800, 86400)


def test_only_documented_operational_settings_differ_from_smoke():
    original = tomllib.loads(SMOKE.read_text(encoding="utf-8"))
    expected = tomllib.loads(SMOKE.read_text(encoding="utf-8"))
    expected["poll_interval_seconds"] = 30
    expected["log_dir"] = "logs/research"
    expected["research"]["sqlite_path"] = "data/research/soak.sqlite"
    expected["research"]["config_version"] = "research-soak-v1"
    expected["research"]["validation_scheduler"]["validation_horizons_seconds"] = [3600, 28800, 86400]
    expected["research"]["validation_scheduler"]["max_lateness_seconds"] = 300
    expected["research"]["validation_scheduler"]["funding_time_tolerance_ms"] = 1000
    assert tomllib.loads(SOAK.read_text(encoding="utf-8")) == expected
    assert original["poll_interval_seconds"] == 5
    assert original["research"]["sqlite_path"] == "data/acceptance/live.sqlite"


def test_smoke_configuration_unchanged():
    # Normalize line endings for checkouts on different operating systems.
    contents = SMOKE.read_text(encoding="utf-8").encode("utf-8")
    assert hashlib.sha256(contents).hexdigest() == "6f075a0fec187406be9d8ddb72eac6633e2f4788114dcebb554bbd418a84f6a0"


def test_research_database_parent_is_created_safely(tmp_path, monkeypatch):
    settings = load_settings(SOAK)
    monkeypatch.chdir(tmp_path)
    assert not Path("data/research").exists()
    repository = SQLiteOpportunityRepository(SQLiteConfig(settings.research.sqlite_path))
    repository.initialize()
    repository.initialize()
    assert repository.config.database_path == tmp_path / "data/research/soak.sqlite"
    assert repository.config.database_path.is_file()
    assert repository.list_recent_opportunities() == ()
    assert not Path("data/acceptance/live.sqlite").exists()


def test_research_config_keeps_public_get_boundary():
    settings = load_settings(SOAK)
    for symbol in settings.symbols:
        for endpoint in (PublicEndpoint.SPOT_DEPTH, PublicEndpoint.PERP_DEPTH):
            assert prepare_get(endpoint, {"symbol": symbol}).method == "GET"
    for forbidden in ("https://api.binance.com/api/v3/order", "https://fapi.binance.com/fapi/v2/account"):
        with pytest.raises(ValueError):
            validate_public_request("GET", forbidden)
    # Settings remain closed to extra credential, trading and proxy overrides.
    for name in ("api_key", "api_secret", "live_trading", "proxy"):
        with pytest.raises(ValidationError):
            Settings(**{**settings.model_dump(), name: "unsupported"})


@pytest.mark.parametrize("interval,valid", [(4, False), (5, True), (10, True), (30, True), (31, False)])
def test_polling_bounds_support_soak_and_preserve_finite_wait(interval, valid):
    values = {**load_settings(SOAK).model_dump(), "poll_interval_seconds": interval}
    if valid:
        assert Settings(**values).poll_interval_seconds == interval
    else:
        with pytest.raises(ValidationError, match="5 to 30"):
            Settings(**values)
