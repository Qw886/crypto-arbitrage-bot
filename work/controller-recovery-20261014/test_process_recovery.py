"""Actual disposable Windows processes preserve a single offline case after crash."""
import json
import os
import subprocess
import sys
from pathlib import Path

import native_case_control as control

from crypto_arbitrage_bot.config.research import StudyRegistrationSettings as Reference
from crypto_arbitrage_bot.utils.native_identity import (
    NativeIdentity,
    NativeState,
    observe_native_identity,
)

WORKER=Path(__file__).with_name('fixture_worker.py').resolve()

def run(mode,directory):
    result=subprocess.run([sys.executable,str(WORKER),mode,str(directory)],cwd=Path.cwd(),
        env=dict(os.environ,PYTHONPATH=str(Path.cwd()/'src')+os.pathsep+str(Path.cwd()/'work')),
        capture_output=True,encoding='utf-8',errors='replace',timeout=45,
        creationflags=subprocess.CREATE_NO_WINDOW)
    return result


def test_new_process_resumes_original_case_and_reduces_after_stop_without_new_entry(tmp_path):
    setup=run('setup',tmp_path)
    assert setup.returncode==0,setup.stderr
    reference=Reference(**json.loads((tmp_path/'reference.json').read_text(encoding='utf-8')))
    crashed=run('crash',tmp_path)
    assert crashed.returncode==23,crashed.stderr
    before=control.status(reference)
    assert before['claims_consumed']==1
    old=json.loads((reference.path.parent/'cases/07/state.json').read_text(encoding='utf-8'))
    identity=NativeIdentity.parse(old['owner'])
    assert observe_native_identity(identity) in (NativeState.MISSING,NativeState.EXITED,NativeState.REPLACED)
    claim=(reference.path.parent/'claims/07.json').read_bytes()
    exchange=json.loads((tmp_path/'exchange.json').read_text(encoding='utf-8'))
    assert len(exchange['orders'])==6 and exchange['positions']['ETHUSDT']=='-0.01'
    resumed=run('resume',tmp_path)
    assert resumed.returncode==0,resumed.stderr
    result=json.loads((tmp_path/'resume-result.json').read_text(encoding='utf-8'))
    assert result['state']['status']=='NOT_PROVEN' and result['state']['capture_failed'] is True
    assert result['mock_order_count']==8 and result['base']==result['perp']=='0.00'
    assert NativeIdentity.parse(result['state']['owner'])!=identity
    assert result['state']['attempt_id']==old['attempt_id']
    assert (reference.path.parent/'claims/07.json').read_bytes()==claim
    assert control.status(reference)['claims_consumed']==1
    assert result['actual_network_requests']==0
    assert not (reference.path.parent/'cases/07/proof.json').exists()


def test_controller_terminal_restart_uses_independent_process_with_zero_requests(tmp_path):
    import hashlib

    assert run('setup',tmp_path).returncode==0
    first=run('restart-first',tmp_path)
    assert first.returncode==0,first.stderr
    reference=Reference(**json.loads((tmp_path/'reference.json').read_text(encoding='utf-8')))
    original=json.loads((tmp_path/'restart-first-result.json').read_text(encoding='utf-8'))['state']
    assert original['status']=='RESTART_PENDING_NEW_PROCESS'
    _,settings=control.load(reference)
    before=hashlib.sha256(settings.testnet.journal_path.read_bytes()).hexdigest()
    replay=run('restart-second',tmp_path)
    assert replay.returncode==0,replay.stderr
    result=json.loads((tmp_path/'restart-second-result.json').read_text(encoding='utf-8'))
    assert result['state']['status']=='NOT_PROVEN'  # Real process replay, explicit offline source.
    assert NativeIdentity.parse(original['owner'])!=NativeIdentity.parse(result['state']['owner'])
    assert result['actual_network_requests']==0 and control.status(reference)['claims_consumed']==1
    assert hashlib.sha256(settings.testnet.journal_path.read_bytes()).hexdigest()==before
    trace=json.loads(Path(result['state']['trace']['path']).read_text(encoding='utf-8'))
    ended=[event for event in trace['events'] if event['kind']=='TERMINAL_REPLAY_END']
    assert len(ended)==1 and ended[0]['request_call_count']==0
    began=[event for event in trace['events'] if event['kind']=='TERMINAL_REPLAY_BEGIN']
    assert len(began)==1 and began[0]['before']==ended[0]['after']
