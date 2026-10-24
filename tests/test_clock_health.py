"""Operational clock uncertainty, bounded PUBLIC transport and silent recovery."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot import clock_health as module
from crypto_arbitrage_bot import windows_public_backend as backend
from crypto_arbitrage_bot.exchanges import binance_usdtm as transport_module
from crypto_arbitrage_bot.exchanges.endpoints import PublicEndpoint
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.native_identity import (
    NativeIdentity,
    NativeState,
    observe_native_identity,
)

AT = datetime(2026, 10, 12, tzinfo=UTC)


def sample(endpoint=PublicEndpoint.SPOT_TIME, offset_us=0, duration_us=200_000):
    return module.Sample(endpoint, AT, AT+timedelta(microseconds=duration_us), duration_us*1000,
                         AT+timedelta(microseconds=duration_us//2+offset_us), 200)


def assess(spot_offset=0, perp_offset=0, duration_us=200_000):
    return module.assess((sample(offset_us=spot_offset, duration_us=duration_us),
        sample(PublicEndpoint.PERP_TIME, perp_offset, duration_us)), evaluated_at=AT+timedelta(seconds=1))


@pytest.mark.parametrize('offset,state', [(0, module.Health.WITHIN_BOUND),
    (120_000, module.Health.WITHIN_BOUND), (6_000_000, module.Health.BEHIND),
    (-6_000_000, module.Health.AHEAD), (200_000, module.Health.UNKNOWN)])
def test_clock_direction_retains_round_trip_uncertainty(offset, state):
    report = assess(offset, offset)
    assert report['status'] is state
    assert report['attention_required'] is (state is not module.Health.WITHIN_BOUND)
    assert report['server_minus_local_intervals'][0]['lower_us'] == offset-101_000
    assert report['server_minus_local_intervals'][0]['upper_us'] == offset+101_000
    assert report['clock_modified'] is False and report['market_quality_verified'] is False
    assert not report['qualification_issued'] and not report['orders_submitted']


@pytest.mark.parametrize('offset,expected', [(149_000, module.Health.WITHIN_BOUND),
    (149_001, module.Health.UNKNOWN), (-149_000, module.Health.WITHIN_BOUND),
    (-149_001, module.Health.UNKNOWN), (351_000, module.Health.UNKNOWN),
    (351_001, module.Health.BEHIND), (-351_001, module.Health.AHEAD)])
def test_inclusive_bound_and_millisecond_quantization(offset, expected):
    assert assess(offset, offset)['status'] is expected


def test_disagreeing_official_sources_never_claim_healthy_or_a_direction():
    report = assess(-200_000, 200_000)
    assert report['status'] is module.Health.UNKNOWN
    assert report['reason_codes'] == ['PUBLIC_CLOCK_INTERVALS_DISAGREE']


@pytest.mark.parametrize('change,reason', [('stale', 'CLOCK_SAMPLE_NOT_FRESH'),
    ('future', 'CLOCK_SAMPLE_NOT_FRESH'), ('step', 'LOCAL_CLOCK_CHANGED_DURING_PROBE'),
    ('reverse', 'LOCAL_CLOCK_CHANGED_DURING_PROBE'), ('slow', 'CLOCK_ROUND_TRIP_TOO_WIDE')])
def test_old_data_clock_steps_and_high_latency_are_unverified(change, reason):
    spot = sample()
    at = AT+timedelta(seconds=1)
    if change == 'stale':
        at = AT+timedelta(seconds=16)
    elif change == 'future':
        at = AT
    elif change == 'step':
        spot = module.Sample(spot.endpoint, AT, AT+timedelta(seconds=1), spot.elapsed_ns, AT, 200)
    elif change == 'reverse':
        spot = module.Sample(spot.endpoint, AT, AT-timedelta(seconds=1), spot.elapsed_ns, AT, 200)
    else:
        spot = sample(duration_us=2_000_001)
        at = AT+timedelta(seconds=3)
    report = module.assess((spot, sample(PublicEndpoint.PERP_TIME)), evaluated_at=at)
    assert report['status'] is module.Health.UNKNOWN and reason in report['reason_codes']


@pytest.mark.parametrize('failure', list(module.Failure))
def test_one_failed_source_cannot_certify_the_other(failure):
    failed = module.Sample(PublicEndpoint.PERP_TIME, AT, AT, 0, None, None, failure)
    report = module.assess((sample(offset_us=6_000_000), failed), evaluated_at=AT+timedelta(seconds=1))
    assert report['status'] is module.Health.UNKNOWN
    assert report['reason_codes'] == [failure.value]
    assert len(report['server_minus_local_intervals']) == 1


def test_both_distinct_sources_and_aware_utc_are_required():
    with pytest.raises(ValueError):
        module.assess((sample(), sample()), evaluated_at=AT)
    with pytest.raises(ValueError):
        module.Sample(PublicEndpoint.SPOT_TIME, AT.replace(tzinfo=None), AT, 1, AT, 200)
    with pytest.raises(ValueError):
        module.Sample(PublicEndpoint.SPOT_DEPTH, AT, AT, 1, AT, 200)
    with pytest.raises(ValueError):
        module.Sample(PublicEndpoint.SPOT_TIME, AT, AT, True, AT, 200)


class Response:
    def __init__(self, body=b'{"serverTime":1791763200100}', status=200, length=None):
        self.body, self.status, self.length = body, status, length
        self.read_calls = []

    def getheader(self, name):
        assert name == 'Content-Length'
        return self.length

    def read(self, size):
        self.read_calls.append(size)
        return self.body[:size]


@pytest.fixture
def transport(monkeypatch):
    response = Response()
    connection = Mock()
    connection.getresponse.return_value = response
    opened = Mock(return_value=connection)
    monkeypatch.setattr(transport_module, 'open_public_connection', opened)
    times = iter([AT, AT+timedelta(microseconds=200_000)])
    ticks = iter([100, 200_000_100])
    monkeypatch.setattr(transport_module, 'utc_now', lambda: next(times))
    monkeypatch.setattr(transport_module, 'monotonic_ns', lambda: next(ticks))
    return response, connection, opened


@pytest.mark.parametrize('endpoint', [PublicEndpoint.SPOT_TIME, PublicEndpoint.PERP_TIME])
def test_one_public_get_uses_original_whitelist_and_closes_response(transport, endpoint):
    response, connection, opened = transport
    result = module.public_sample(endpoint, 5)
    assert result.failure is None and result.server_time == AT+timedelta(milliseconds=100)
    opened.assert_called_once_with(endpoint.value, timeout=5)
    args, kwargs = connection.request.call_args
    assert args == ('GET', '/api/v3/time' if endpoint is PublicEndpoint.SPOT_TIME else '/fapi/v1/time')
    assert set(kwargs['headers']) == {'Accept', 'Cache-Control'}
    assert response.read_calls == [4097]
    connection.close.assert_called_once()


@pytest.mark.parametrize('body', [b'{"serverTime":true}', b'{"serverTime":1.5}',
    b'{"serverTime":-1}', b'{"serverTime":253402300800000}', b'{"serverTime":NaN}',
    b'{"serverTime":1,"serverTime":2}', b'[]', b'{}', b'raw secret error', b'\xff', b'x'*4097])
def test_bad_public_body_is_bounded_and_only_structured_failure_escapes(transport, body):
    response, connection, _ = transport
    response.body = body
    result = module.public_sample(PublicEndpoint.SPOT_TIME, 5)
    assert result.failure is module.Failure.RESPONSE and result.server_time is None
    assert b'raw secret' not in exact_json(result).encode()
    connection.close.assert_called_once()


@pytest.mark.parametrize('length', ['4097', '-1', '1.5', '\u0661'])
def test_large_or_invalid_length_refused_before_body_read(transport, length):
    response, _, _ = transport
    response.length = length
    assert module.public_sample(PublicEndpoint.SPOT_TIME, 5).failure is module.Failure.RESPONSE
    assert not response.read_calls


@pytest.mark.parametrize('code', [301, 302, 429, 500])
def test_no_redirect_or_retry_and_no_http_error_body(transport, code):
    response, connection, opened = transport
    response.status = code
    response.body = b'private-looking error contents'
    result = module.public_sample(PublicEndpoint.SPOT_TIME, 5)
    assert result.failure is module.Failure.HTTP and result.http_status == code
    assert not response.read_calls and opened.call_count == 1
    assert connection.request.call_count == 1


@pytest.mark.parametrize('error,failure', [(TimeoutError('private error'), module.Failure.TIMEOUT),
    (OSError('private error'), module.Failure.IO)])
def test_network_failures_are_sanitized_and_not_retried(transport, error, failure):
    _, connection, _ = transport
    connection.request.side_effect = error
    result = module.public_sample(PublicEndpoint.SPOT_TIME, 5)
    assert result.failure is failure and 'private error' not in exact_json(result)
    connection.close.assert_called_once()


def test_bad_endpoint_or_timeout_has_no_network_io(transport):
    _, _, opened = transport
    for value in [0, 11, True, 1.5]:
        with pytest.raises(ValueError):
            module.public_sample(PublicEndpoint.SPOT_TIME, value)
    with pytest.raises(ValueError):
        module.public_sample(PublicEndpoint.PERP_DEPTH, 5)
    opened.assert_not_called()


@pytest.mark.parametrize('endpoint,timeout', [(PublicEndpoint.PERP_DEPTH, 5),
    (PublicEndpoint.SPOT_DEPTH, 5), (PublicEndpoint.SPOT_TIME, 0), (PublicEndpoint.PERP_TIME, True)])
def test_transport_boundary_rejects_non_time_or_unbounded_request(transport, endpoint, timeout):
    _, _, opened = transport
    params = {'symbol': 'BTCUSDT'} if endpoint in (PublicEndpoint.PERP_DEPTH, PublicEndpoint.SPOT_DEPTH) else None
    request = module.prepare_get(endpoint, params)
    with pytest.raises(ValueError):
        transport_module.fetch_public_time(request, timeout)
    opened.assert_not_called()


@pytest.mark.parametrize('change', ['wrong_source', 'large', 'exit', 'malformed', 'extra', 'naive'])
def test_child_scope_malformed_or_unbounded_output_cannot_supply_clock_time(monkeypatch, change):
    body = json.loads(exact_json(sample()))
    if change == 'wrong_source':
        body['endpoint'] = PublicEndpoint.PERP_TIME.value
    elif change == 'extra':
        body['raw_error'] = 'sensitive child contents'
    elif change == 'naive':
        body['started_at'] = '2026-10-12T00:00:00'
    raw = json.dumps(body).encode()
    if change == 'large':
        raw = b'x'*8193
    elif change == 'malformed':
        raw = b'invalid sensitive child output'
    monkeypatch.setattr(module, 'utc_now', lambda: AT)
    monkeypatch.setattr(module.subprocess, 'run', Mock(return_value=SimpleNamespace(
        stdout=raw, returncode=1 if change == 'exit' else 0)))
    result = module.run_probe(PublicEndpoint.SPOT_TIME, 5)
    assert result.failure is module.Failure.CHILD and result.server_time is None
    assert 'sensitive' not in exact_json(result)


@pytest.mark.skipif(sys.platform != 'win32', reason='native Windows child identity proof')
def test_actual_blocked_child_is_killed_and_waited_without_touching_any_owner(monkeypatch):
    real_run = subprocess.run
    captured = {}
    def blocking_child(args, **kwargs):
        code = ('import json,time; from crypto_arbitrage_bot.utils.native_identity import current_native_identity; '
                'print(json.dumps(current_native_identity().payload()),flush=True); time.sleep(30)')
        try:
            return real_run([args[0], '-c', code], **kwargs)
        except subprocess.TimeoutExpired as error:
            captured['identity'] = NativeIdentity.parse(json.loads(error.output.decode().splitlines()[0]))
            captured['timeout'] = error.timeout
            raise
    monkeypatch.setattr(module.subprocess, 'run', blocking_child)
    result = module.run_probe(PublicEndpoint.SPOT_TIME, 1)
    assert result.failure is module.Failure.TIMEOUT and captured['timeout'] == 2
    assert observe_native_identity(captured['identity']) in (NativeState.MISSING, NativeState.EXITED)


def test_successful_child_preserves_exact_source_and_process_limits(monkeypatch):
    clocks = iter([AT, AT+timedelta(seconds=1)])
    monkeypatch.setattr(module, 'utc_now', lambda: next(clocks))
    run = Mock(return_value=SimpleNamespace(stdout=exact_json(sample()).encode(), returncode=0))
    monkeypatch.setattr(module.subprocess, 'run', run)
    assert module.run_probe(PublicEndpoint.SPOT_TIME, 5) == sample()
    args, kwargs = run.call_args
    assert args[0][-3:] == ['spot', '--timeout-seconds', '5']
    assert kwargs['timeout'] == 6 and kwargs['stdin'] == subprocess.DEVNULL
    assert kwargs['creationflags'] == (subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


def test_collector_checks_both_sources_once_without_research_writes(monkeypatch):
    run = Mock(side_effect=lambda endpoint, timeout: sample(endpoint))
    monkeypatch.setattr(module, 'run_probe', run)
    monkeypatch.setattr(module, 'utc_now', lambda: AT+timedelta(seconds=1))
    report = module.check(5)
    assert report['status'] is module.Health.WITHIN_BOUND and run.call_count == 2
    assert {call.args[0] for call in run.call_args_list} == set(module.ENDPOINTS.values())


@pytest.mark.parametrize('offset,exit_code', [(0, 0), (6_000_000, 1), (200_000, 2)])
def test_cli_reports_attention_and_exclusive_current_archive(tmp_path, monkeypatch, capsys, offset, exit_code):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'work').mkdir()
    check = Mock(return_value=assess(offset, offset))
    monkeypatch.setattr(module, 'check', check)
    assert module.main(['check', '--output', 'work/new.json']) == exit_code
    result = json.loads(capsys.readouterr().out)
    assert json.loads((tmp_path/'work/new.json').read_text(encoding='utf-8')) == result
    assert result['diagnostic_bound_us'] == 250000
    original = (tmp_path/'work/new.json').read_bytes()
    assert module.main(['check', '--output', 'work/new.json']) == 2
    assert check.call_count == 1 and (tmp_path/'work/new.json').read_bytes() == original


@pytest.mark.parametrize('path', ['data/original.json', '../outside.json', 'work/report.txt', 'work/missing/report.json'])
def test_refused_report_paths_have_no_public_calls(tmp_path, monkeypatch, path):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'work').mkdir()
    check = Mock(side_effect=AssertionError('no public IO'))
    monkeypatch.setattr(module, 'check', check)
    assert module.main(['check', '--output', path]) == 2
    check.assert_not_called()


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    (tmp_path/'plan.json').write_text('{}', encoding='utf-8')
    value = {'root': str(tmp_path), 'target': str(tmp_path/'job'),
             'ends_at': (AT+timedelta(hours=1)).isoformat(),
             'arguments': ['-m', 'crypto_arbitrage_bot.research_control', 'resume', '--job-dir', str(tmp_path/'job')]}
    monkeypatch.setattr(backend, 'load_plan', lambda directory: value)
    monkeypatch.setattr(backend, 'utc_now', lambda: AT)
    return tmp_path, value, backend._sha(tmp_path/'plan.json')


def test_silent_backend_keeps_original_foreground_wait_and_no_shell(recovery, monkeypatch):
    directory, value, sha = recovery
    monkeypatch.setattr(backend, 'gate', lambda path: True)
    run = Mock(return_value=SimpleNamespace(returncode=7))
    monkeypatch.setattr(backend.subprocess, 'run', run)
    assert backend.run_backend(directory, sha) == 7
    args, kwargs = run.call_args
    assert args[0] == [str(directory/'.venv/Scripts/python.exe'), *value['arguments']]
    assert '--background' not in args[0] and 'shell' not in kwargs
    assert kwargs['stdin'] == kwargs['stdout'] == kwargs['stderr'] == subprocess.DEVNULL
    assert kwargs['env']['PYTHONPATH'] == str(directory/'src')
    assert kwargs['creationflags'] == (subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


@pytest.mark.parametrize('condition', ['live', 'backend_stop', 'original_stop', 'expired', 'plan_changed'])
def test_silent_backend_cannot_resume_after_gate_or_boundary_changes(recovery, monkeypatch, condition):
    directory, value, sha = recovery
    def gate(path):
        if condition == 'backend_stop':
            (directory/'stop').touch()
        elif condition == 'original_stop':
            (directory/'job').mkdir()
            (directory/'job/stop').touch()
        elif condition == 'expired':
            value['ends_at'] = AT.isoformat()
        elif condition == 'plan_changed':
            (directory/'plan.json').write_text('{"changed":1}', encoding='utf-8')
        return condition != 'live'
    monkeypatch.setattr(backend, 'gate', gate)
    run = Mock(side_effect=AssertionError('no resume'))
    monkeypatch.setattr(backend.subprocess, 'run', run)
    if condition == 'plan_changed':
        with pytest.raises(ValueError):
            backend.run_backend(directory, sha)
    else:
        assert backend.run_backend(directory, sha) == 0
    run.assert_not_called()


def test_new_native_task_action_uses_standalone_pythonw_without_cli_environment(recovery, monkeypatch):
    directory, value, sha = recovery
    backend.write_scripts(directory)
    install = (directory/'install.ps1').read_text(encoding='utf-8')
    assert "$backendAction.Path=Join-Path $backendData.root '.venv\\Scripts\\pythonw.exe'" in install
    assert "Join-Path $backendDir 'launch.py'" in install
    assert 'powershell.exe' not in install and 'action.ps1' not in install
    launch = (directory/'launch.py').read_text(encoding='utf-8')
    assert sha in launch and 'sys.path.insert(0' in launch and 'PYTHONPATH' in launch
    assert 'run_backend(directory' in launch
    with pytest.raises(ValueError):
        backend.run_backend(directory, '0'*64)
    assert backend.main(['run', '--directory', str(directory)]) == 2


def test_actual_standalone_launcher_ignores_missing_pythonpath_and_refuses_plan_tamper(recovery, monkeypatch):
    directory, value, _ = recovery
    package = directory/'src/crypto_arbitrage_bot'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('', encoding='utf-8')
    (package/'windows_public_backend.py').write_text(
        'def run_backend(directory, sha):\n    (directory/"bootstrap-proof").write_text(sha)\n    return 0\n',
        encoding='utf-8')
    (directory/'plan.json').write_text(exact_json(value), encoding='utf-8')
    backend.write_scripts(directory)
    env = {key: item for key, item in os.environ.items() if key != 'PYTHONPATH'}
    result = subprocess.run([sys.executable, str(directory/'launch.py')], env=env, capture_output=True, timeout=5)
    assert result.returncode == 0 and (directory/'bootstrap-proof').exists()
    proof = (directory/'bootstrap-proof').read_bytes()
    (directory/'plan.json').write_text('{}', encoding='utf-8')
    result = subprocess.run([sys.executable, str(directory/'launch.py')], env=env, capture_output=True, timeout=5)
    assert result.returncode == 2 and (directory/'bootstrap-proof').read_bytes() == proof
