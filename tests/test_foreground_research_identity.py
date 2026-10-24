"""Exact existing foreground resume identity, offline native-observation seam."""
import json
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_offline_estimate import OPENED
from test_research_control import job as managed_fixture
from test_runtime_status import legacy as legacy_fixture

from crypto_arbitrage_bot import runtime_status as module
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.process_probe import ProcessObservation, ProcessState


@pytest.mark.parametrize('kind',['legacy','managed'])
@pytest.mark.parametrize('change',['resume','worker','wrong_exe','pythonw','wrong_directory',
                                  'extra','background','start','wrong_module','wrong_flag'])
def test_exact_foreground_owner_keeps_all_other_command_identity_guards(tmp_path,monkeypatch,kind,change):
    monkeypatch.setattr('socket.create_connection',Mock(side_effect=AssertionError('offline only')))
    if kind == 'legacy':
        settings,directory,saved=legacy_fixture.__wrapped__(tmp_path)
        at=OPENED+timedelta(hours=1,seconds=5)
        worker='_legacy_worker'
    else:
        config,directory,*_=managed_fixture.__wrapped__(tmp_path,monkeypatch)
        settings=load_settings(config)
        saved=json.loads((directory/'state.json').read_text(encoding='utf-8'))
        saved.update(state='RUNNING',worker_pid=4242,cycles_started=1,updated_at=OPENED+timedelta(seconds=10))
        (directory/'state.json').write_text(exact_json(saved),encoding='utf-8')
        at=OPENED+timedelta(seconds=15)
        worker='_worker'
        (directory/'worker.lock').write_bytes(b'0')
    args=[sys.executable,'-m','crypto_arbitrage_bot.research_control','resume','--job-dir',str(directory)]
    if change=='worker':
        args[3]=worker
    elif change=='wrong_exe':
        args[0]=str(tmp_path/'foreign/python.exe')
    elif change=='pythonw':
        args[0]=str(Path(sys.executable).with_name('pythonw.exe'))
    elif change=='wrong_directory':
        args[-1]=str(tmp_path/'other')
    elif change in ('extra','background'):
        args.append('--bad' if change=='extra' else '--background')
    elif change=='start':
        args[3]='start'
    elif change=='wrong_module':
        args[2]='crypto_arbitrage_bot.execution.main'
    elif change=='wrong_flag':
        args[4]='--config'
    seen=[]
    def probe(pid,*,matches,not_before,not_after):
        seen.append(pid)
        assert not_before <= OPENED+timedelta(seconds=1) <= not_after
        state=ProcessState.MATCHED if matches(tuple(args)) else ProcessState.DIFFERENT_IDENTITY
        return ProcessObservation(state,pid,OPENED+timedelta(seconds=1))
    result=module.research_status(settings,directory,at=at,max_heartbeat_age=timedelta(seconds=60),
                                  probe=probe,lock_probe=lambda p:True)
    assert seen==[4242] and result['budget_consumed']==(20 if kind=='legacy' else 1)
    assert result['worker_lock_held'] is True
    expected='RUNNING_CONFIRMED' if change in ('resume','worker') else 'OWNER_UNVERIFIED'
    assert result['status']==expected
    expected_deadline=(OPENED+timedelta(days=2)) if kind=='legacy' else (OPENED+timedelta(seconds=65))
    assert result['original_deadline']==expected_deadline
