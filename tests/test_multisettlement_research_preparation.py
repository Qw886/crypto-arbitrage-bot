"""Long observation changes horizons/runtime only, not estimates or trade gates."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot.config.settings import load_settings

spec=importlib.util.spec_from_file_location("extended_candidate_acceptance_watch",Path("work/candidate-forward/watch.py"))
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_long_observation_config_preserves_every_financial_and_quality_parameter():
    original=load_settings(Path("docs/research-funding-screen.toml"))
    new=load_settings(Path("docs/research-multisettlement.toml"))
    old_data=original.model_dump()
    new_data=new.model_dump()
    assert new.research.sqlite_path==Path("data/research/multisettlement.sqlite")
    assert new.research.validation_scheduler.validation_horizons_seconds==(86400,259200,604800)
    for field in ("sqlite_path","config_version"):
        new_data["research"][field]=old_data["research"][field]
    new_data["research"]["validation_scheduler"]["validation_horizons_seconds"]=old_data["research"]["validation_scheduler"]["validation_horizons_seconds"]
    assert new_data==old_data # No fee, Scenario, quality, sizing, Decision or network policy change.


def test_loading_pilot_cannot_modify_smoke_soak_or_funding_configuration():
    paths=tuple(Path("docs",name) for name in ("acceptance-smoke.toml","research-soak.toml","research-funding-screen.toml"))
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    load_settings(Path("docs/research-multisettlement.toml"))
    assert {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}==before


def test_default_worker_still_rejects_long_or_excessive_jobs(tmp_path):
    with pytest.raises(ValueError):
        worker.watch(None,tmp_path,duration_seconds=604800,poll_seconds=30,max_batches=25000)


def test_explicit_extended_worker_is_finite_and_delegates_existing_due_batch(tmp_path):
    cycle=Mock(return_value={"task_events":[]})
    result=worker.watch(SimpleNamespace(research=SimpleNamespace(sqlite_path=tmp_path/"db")),tmp_path,
        duration_seconds=8*86400,poll_seconds=30,max_batches=1,extended_duration=True,
        cycle=cycle,monotonic=lambda:0,wait=Mock(),stop=lambda:False)
    assert result["state"]=="FINISHED" and result["extended_duration"] is True
    assert cycle.call_count==1 and result["completed_batches"]==1


@pytest.mark.parametrize("duration,poll,batches,extended",[(8*86400+1,30,1,True),(60,0,1,True),
    (60,30,30001,True),(60,30,1,"yes"),(True,30,1,True),(60.5,30,1,True)])
def test_extended_worker_is_not_unbounded_or_ambiguously_typed(tmp_path,duration,poll,batches,extended):
    with pytest.raises(ValueError):
        worker.watch(None,tmp_path,duration_seconds=duration,poll_seconds=poll,max_batches=batches,
                     extended_duration=extended)


def test_extended_failure_waits_and_stops_without_crash_loop(tmp_path):
    clock=[0]
    waits=[]
    def wait(seconds):
        waits.append(seconds)
        clock[0]+=seconds
    cycle=Mock(side_effect=TimeoutError("offline transport"))
    result=worker.watch(SimpleNamespace(research=SimpleNamespace(sqlite_path=tmp_path/"db")),tmp_path,
        duration_seconds=604800,poll_seconds=30,max_batches=2,extended_duration=True,
        cycle=cycle,monotonic=lambda:clock[0],wait=wait,stop=lambda:False)
    assert result["failed_batches"]==2 and waits==[30] and result["state"]=="FINISHED"


def test_extended_stop_file_prevents_network(tmp_path):
    (tmp_path/"stop").touch()
    cycle=Mock()
    result=worker.watch(SimpleNamespace(research=SimpleNamespace(sqlite_path=tmp_path/"db")),tmp_path,
        duration_seconds=604800,poll_seconds=30,max_batches=25000,extended_duration=True,cycle=cycle)
    assert result["state"]=="STOPPED" and not cycle.called
