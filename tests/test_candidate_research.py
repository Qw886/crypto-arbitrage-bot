"""One-shot public catalog entry point: reproducible evidence, no trading wiring."""

import base64
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock

import pytest
from test_candidate_catalog import NOW, instrument, source

from crypto_arbitrage_bot import candidate_research as module
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint

CONFIG = Path("docs/research-candidate-catalog.toml")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(module, "utc_now", lambda: NOW)
    client = Mock()
    client.fetch.side_effect = [source([instrument()]), source([instrument(perp=True)], True)]
    monkeypatch.setattr(module, "BinanceMetadataClient", lambda **kw: client)
    return client


def test_existing_settings_system_preserves_all_financial_rules():
    original = load_settings(Path("docs/research-soak.toml"))
    added = load_settings(CONFIG)
    assert added.fees == original.fees and added.symbols == original.symbols
    assert added.research.sizing == original.research.sizing
    assert added.research.scenarios == original.research.scenarios
    assert added.research.decision == original.research.decision
    assert added.research.data_quality == original.research.data_quality
    assert added.research.validation_scheduler == original.research.validation_scheduler
    assert original.research.candidate_catalog is None
    assert added.research.candidate_catalog.min_perp_age_seconds == 604800
    assert added.testnet is None


def test_catalog_collects_only_two_official_metadata_gets(client):
    result = module.collect_catalog(load_settings(CONFIG), client=client)
    assert result.candidates[0].symbol == "SOLUSDT"
    assert [call.args[0] for call in client.fetch.call_args_list] == [
        PublicEndpoint.SPOT_EXCHANGE_INFO, PublicEndpoint.PERP_EXCHANGE_INFO]


def test_cli_preserves_evidence_once_and_decimal_strings(client, tmp_path, capsys):
    output = tmp_path / "nested/evidence.json"
    assert module.main(["--config", str(CONFIG), "--output", str(output)]) == 0
    data = json.loads(output.read_text())
    assert len(data["catalog"]["sources"]) == 2
    raw = data["catalog"]["sources"][0]["raw_response"]
    assert json.loads(base64.b64decode(raw["base64"]))["symbols"][0]["baseAsset"] == "SOL"
    assert data["catalog"]["candidates"][0]["spot_rules"]["min_notional"] == "5.000000000000000001"
    assert "raw_response" not in capsys.readouterr().out
    assert not list(tmp_path.rglob("*.sqlite"))


def test_existing_evidence_file_is_not_overwritten_or_recollected(client, tmp_path):
    output = tmp_path / "existing.json"
    output.write_bytes(b"archived")
    assert module.main(["--config", str(CONFIG), "--output", str(output)]) == 2
    assert output.read_bytes() == b"archived" and not client.fetch.called


def test_missing_config_section_cannot_trigger_requests(client):
    with pytest.raises(ValueError, match="candidate_catalog"):
        module.collect_catalog(load_settings(Path("docs/research-soak.toml")), client=client)
    assert not client.fetch.called


def test_invalid_catalog_is_saved_as_failure_not_opportunity(client, tmp_path):
    client.fetch.side_effect = [source([instrument()], status_code=429), source([instrument(perp=True)], True)]
    output = tmp_path / "failed.json"
    assert module.main(["--config", str(CONFIG), "--output", str(output)]) == 1
    data = json.loads(output.read_text())
    assert data["catalog"]["status"] == "INVALID" and data["catalog"]["candidates"] == []
    assert "SOURCE_FAILED" in data["catalog"]["reason_codes"]
    assert not list(tmp_path.rglob("*.sqlite"))


def test_catalog_settings_are_injected_through_existing_environment(monkeypatch):
    monkeypatch.setenv("CAB_RESEARCH__CANDIDATE_CATALOG__MIN_PERP_AGE_SECONDS", "1209600")
    settings = load_settings(CONFIG)
    assert settings.research.candidate_catalog.min_perp_age_seconds == 1209600


@pytest.mark.parametrize("slow_market", [PublicEndpoint.SPOT_EXCHANGE_INFO, PublicEndpoint.PERP_EXCHANGE_INFO])
def test_metadata_reads_overlap_without_relaxing_age_or_sampling_limits(monkeypatch, slow_market):
    settings = load_settings(CONFIG)
    barrier = Barrier(2)
    client = Mock()
    def fetch(endpoint):
        barrier.wait(timeout=2)
        perp = endpoint is PublicEndpoint.PERP_EXCHANGE_INFO
        original = source([instrument(perp=perp)], perp)
        received = NOW + timedelta(seconds=119 if endpoint == slow_market else 3)
        return replace(original, requested_at=NOW, received_at=received, finished_at=received)
    client.fetch.side_effect = fetch
    monkeypatch.setattr(module, "utc_now", lambda: NOW + timedelta(seconds=120))
    before = settings.research.candidate_catalog.model_dump()
    result = module.collect_catalog(settings, client=client)
    assert result.status.value == "COMPLETE" and result.candidates[0].symbol == "SOLUSDT"
    assert all(item.requested_at == NOW for item in result.sources)
    assert result.sources[0].endpoint is PublicEndpoint.SPOT_EXCHANGE_INFO
    assert result.sources[1].endpoint is PublicEndpoint.PERP_EXCHANGE_INFO
    assert settings.research.candidate_catalog.model_dump() == before


def test_concurrent_catalog_keeps_source_failure_and_no_candidate_fallback(client):
    values = {PublicEndpoint.SPOT_EXCHANGE_INFO: source([instrument()], status_code=429),
              PublicEndpoint.PERP_EXCHANGE_INFO: source([instrument(perp=True)], True)}
    client.fetch.side_effect = lambda endpoint: values[endpoint]
    result = module.collect_catalog(load_settings(CONFIG), client=client)
    assert result.status.value == "INVALID" and result.candidates == ()
    assert result.sources[0].status_code == 429
    assert "SOURCE_FAILED" in result.reason_codes
