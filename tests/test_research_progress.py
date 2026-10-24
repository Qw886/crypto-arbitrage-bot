"""Long public cycles pulse their existing claim; no budget reissuance."""

from datetime import timedelta

from test_research_control import job as job
from test_research_control import run, state


def test_progress_preserves_claim_and_counters_during_long_cycle(job):
    _,_,plan,elapsed,now,_,_,factory=job
    snapshots=[]
    def cycle(runner):
        pulse=runner.set_progress.call_args.args[0]
        for _ in range(4):
            elapsed[0]+=6
            pulse()
            current=state(job)
            assert current['active_cycle']==0 and current['cycles_started']==1
            assert current['cycles_completed']==0 and current['job_id']==plan['job_id']
            assert current['updated_at']==now().isoformat(timespec='microseconds')
            snapshots.append(current['updated_at'])
        # Explicit cooperation stop ends the single long cycle.
        from crypto_arbitrage_bot.research_control import request_stop
        request_stop(job[1])
        return {'failed':False}
    final=run(job,cycle)
    assert len(snapshots)==4 and final['state']=='STOPPED'
    assert final['cycles_started']==final['cycles_completed']==1


def test_progress_rate_limit_does_not_write_for_every_small_tick(job):
    elapsed=job[3]
    def cycle(runner):
        pulse=runner.set_progress.call_args.args[0]
        before=state(job)['updated_at']
        elapsed[0]+=1
        pulse()
        assert state(job)['updated_at']==before
        elapsed[0]+=4
        pulse()
        assert state(job)['updated_at']!=before
        from crypto_arbitrage_bot.research_control import request_stop
        request_stop(job[1])
        return {'failed':False}
    assert run(job,cycle)['cycles_completed']==1


def test_progress_at_original_deadline_stops_feeding_without_extending_job(job):
    elapsed=job[3]
    def cycle(runner):
        pulse=runner.set_progress.call_args.args[0]
        elapsed[0]+=65
        pulse()
        assert runner.stop_signal.is_set()
        return {'failed':False}
    final=run(job,cycle)
    assert final['state']=='FINISHED' and final['cycles_started']==1
    assert job[5]==[]
    assert job[2]['ends_at']==job[4]()-timedelta(seconds=0)


def test_runner_without_progress_capability_keeps_original_controller_behavior(job):
    from types import SimpleNamespace
    runner=SimpleNamespace(scheduler=None)
    job[7].return_value=runner
    final=run(job)
    assert final['state']=='FINISHED' and final['cycles_completed']==3


def test_progress_storage_failure_surfaces_without_replaying_cycle(job,monkeypatch):
    from crypto_arbitrage_bot import research_control as module
    elapsed=job[3]
    original_save=module._save_state
    def save(directory,value):
        if elapsed[0]==6 and value['active_cycle']==0:
            raise OSError('synthetic heartbeat write failure')
        return original_save(directory,value)
    monkeypatch.setattr(module,'_save_state',save)
    calls=[]
    def cycle(runner):
        calls.append(state(job)['active_cycle'])
        if len(calls)==1:
            elapsed[0]+=6
            runner.set_progress.call_args.args[0]()
        return {'failed':False}
    final=run(job,cycle)
    assert calls==[0,1] and final['failed_cycles']==1 and final['cycles_completed']==1
    assert final['cycles_started']==2
