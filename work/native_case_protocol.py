"""Fixed BTC/ETH native-case protocol. Preparation is never trading authorization."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Any

import crypto_arbitrage_bot
from crypto_arbitrage_bot.config.research import StudyRegistrationSettings
from crypto_arbitrage_bot.config.settings import Settings, load_settings
from crypto_arbitrage_bot.execution.risk_review import (
    execution_policy_fingerprint,
    implementation_fingerprint,
)
from crypto_arbitrage_bot.models.strategy_admission import ExecutionRiskCase
from crypto_arbitrage_bot.snapshot_backup import EXECUTION
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

MAX_DOCUMENT = 256000
MAX_ROWS = 50000
MAX_RECORD_BYTES = 64*1024*1024
MAX_HOURS = 168
RESERVED_NORMAL_SLOTS = 2
ENV_PACKAGES = ('pydantic','pydantic-settings','pytest','mypy','ruff','setuptools')
TABLES = EXECUTION | {'testnet_trade_reconciliation_history'}


class ProtocolError(RuntimeError):
    pass


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as source:
        for body in iter(lambda:source.read(1024*1024),b''):
            result.update(body)
    return result.hexdigest()


def exclusive(path: Path,value: object) -> None:
    body = exact_json(value).encode('utf-8')
    if len(body)>MAX_DOCUMENT:
        raise ProtocolError('PROTOCOL_DOCUMENT_LIMIT')
    with path.open('xb') as output:
        output.write(body)
        output.flush()
        os.fsync(output.fileno())


def document(path: Path,sha256: str) -> dict[str,Any]:
    if any(p.is_symlink() or p.is_junction() for p in (path,*path.parents)):
        raise ProtocolError('REGULAR_PINNED_DOCUMENT_REQUIRED')
    with path.open('rb') as source:
        body = source.read(MAX_DOCUMENT+1)
    if len(body)>MAX_DOCUMENT or hashlib.sha256(body).hexdigest()!=sha256:
        raise ProtocolError('PROTOCOL_DOCUMENT_HASH_DIFFERS')
    data = json.loads(body)
    if not isinstance(data,dict) or exact_json(data).encode()!=body:
        raise ProtocolError('CANONICAL_PROTOCOL_DOCUMENT_REQUIRED')
    return data


def original_records(journal: Path,identities: list[str] | None = None) -> tuple[list[str],str]:
    deadline = monotonic()+10
    with closing(sqlite3.connect(journal.resolve().as_uri()+'?mode=ro',uri=True,timeout=0)) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        db.set_progress_handler(lambda:int(monotonic()>=deadline),1000)
        parents = db.execute('SELECT attempt_id,state FROM testnet_pair_attempts ORDER BY attempt_id LIMIT 10001').fetchall()
        if len(parents)>10000:
            raise ProtocolError('PROTOCOL_JOURNAL_LIMIT')
        current = [row['attempt_id'] for row in parents]
        if identities is None:
            if not parents or any(row['state'] not in ('CLOSED','FLATTENED','FAILED') for row in parents):
                raise ProtocolError('ORIGINAL_TERMINAL_JOURNAL_REQUIRED')
            identities = current
        if not set(identities)<=set(current):
            raise ProtocolError('ORIGINAL_ATTEMPTS_MISSING')
        marks = ','.join('?' for _ in identities)
        available = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        values: dict[str,list[dict[str,Any]]] = {}
        count = used = 0
        for table in sorted(TABLES):
            rows = [] if table not in available else [dict(row) for row in db.execute(
                f'SELECT * FROM {table} WHERE attempt_id IN ({marks}) LIMIT ?',[*identities,MAX_ROWS-count+1])]
            count += len(rows)
            rows.sort(key=exact_json)
            used += len(exact_json(rows).encode())
            if count>MAX_ROWS or used>MAX_RECORD_BYTES or monotonic()>=deadline:
                raise ProtocolError('ORIGINAL_RECORD_LIMIT')
            values[table] = rows
    return current,hashlib.sha256(exact_json(values).encode()).hexdigest()


def environment() -> dict[str,Any]:
    return {'python':platform.python_version(),'platform':os.name,
            'packages':{name:importlib.metadata.version(name) for name in ENV_PACKAGES}}


def prepare(settings: Settings,directory: Path,*,workspace: Path | None = None,
            hours: int = MAX_HOURS,clock: Callable[[], datetime] = utc_now) -> StudyRegistrationSettings:
    workspace = (workspace or Path.cwd()).resolve()
    directory = directory.resolve()
    if (directory.exists() or not directory.is_relative_to(workspace)
            or type(hours) is not int or not 1<=hours<=MAX_HOURS
            or settings.testnet is None or settings.research is None
            or set(settings.symbols)!={'BTCUSDT','ETHUSDT'}):
        raise ProtocolError('NEW_FIXED_PROTOCOL_DIRECTORY_REQUIRED')
    journal = settings.testnet.journal_path.resolve()
    if not journal.is_relative_to(workspace):
        raise ProtocolError('ORIGINAL_WORKSPACE_JOURNAL_REQUIRED')
    ids,records_sha = original_records(journal)
    if len(ids)!=settings.testnet.max_total_attempts:
        raise ProtocolError('EXHAUSTED_ORIGINAL_BUDGET_REQUIRED')
    cases = [{'slot':number,'symbol':symbol,'case':kind.value} for number,(symbol,kind) in enumerate(
        (symbol,kind) for symbol in settings.symbols for kind in ExecutionRiskCase)]
    proposed = settings.model_copy(update={
        'testnet':settings.testnet.model_copy(update={'journal_path':journal,
            'max_total_attempts':len(ids)+len(cases)+RESERVED_NORMAL_SLOTS,'balance_evidence_enabled':True,
            'balance_evidence_directory':(settings.testnet.balance_evidence_directory or journal.parent/'balance-evidence').resolve()}),
        'research':settings.research.model_copy(update={'sqlite_path':settings.research.sqlite_path.resolve()}),
        'log_dir':settings.log_dir.resolve()})
    assert proposed.testnet is not None
    at = utc_timestamp(clock())
    source_sha = implementation_fingerprint()
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'claims').mkdir()
    (directory/'cases').mkdir()
    config_path = directory/'settings.json'
    exclusive(config_path,proposed.model_dump(mode='json'))
    package = Path(crypto_arbitrage_bot.__file__).resolve().parent
    sources = sorted([*package.rglob('*.py'),*package.rglob('*.toml')])
    hashes = {}
    frozen = directory/'runtime-src'/'crypto_arbitrage_bot'
    for source in sources:
        if source.is_symlink():
            raise ProtocolError('SOURCE_SYMLINK_REFUSED')
        target = frozen/source.relative_to(package)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as output:
            output.write(source.read_bytes())
        hashes[source.relative_to(package).as_posix()]=digest(target)
    if implementation_fingerprint()!=source_sha or any(digest(package/name)!=value for name,value in hashes.items()):
        raise ProtocolError('SOURCE_CHANGED_DURING_FREEZE')
    exclusive(directory/'runtime.json',hashes)
    plan = {'schema_version':'native-case-protocol-v1','environment':'TESTNET',
        'created_at':at,'ends_at':at+timedelta(hours=hours),'directory':str(directory),'workspace':str(workspace),
        'journal_path':str(journal),'original_attempt_ids':ids,'original_records_sha256':records_sha,
        'original_limit':len(ids),'native_case_slots':len(cases),'future_normal_slots_unapproved':RESERVED_NORMAL_SLOTS,
        'proposed_global_limit':proposed.testnet.max_total_attempts,'cases':cases,
        'config_sha256':digest(config_path),'runtime_manifest_sha256':digest(directory/'runtime.json'),
        'source_sha256':source_sha,'policy_sha256':execution_policy_fingerprint(proposed),
        'worker_sha256':digest(Path(__file__)),'runtime_environment':environment(),
        'normal_trading_authorized':False,'private_calls_authorized':False,'acceptance_max_wait_seconds':28800}
    exclusive(directory/'plan.json',plan)
    return StudyRegistrationSettings(path=directory/'plan.json',sha256=digest(directory/'plan.json'))


def load(reference: StudyRegistrationSettings) -> tuple[dict[str,Any],Settings]:
    plan = document(reference.path,reference.sha256)
    directory = reference.path.parent.resolve()
    if (plan['schema_version']!='native-case-protocol-v1' or plan['environment']!='TESTNET'
            or plan['directory']!=str(directory) or plan['normal_trading_authorized'] is not False
            or plan['private_calls_authorized'] is not False or plan['worker_sha256']!=digest(Path(__file__))):
        raise ProtocolError('FIXED_PROTOCOL_IDENTITY_DIFFERS')
    start,end = (datetime.fromisoformat(plan[field]) for field in ('created_at','ends_at'))
    ids = plan['original_attempt_ids']
    if (start.utcoffset()!=timedelta(0) or end.utcoffset()!=timedelta(0)
            or not timedelta(0)<end-start<=timedelta(hours=MAX_HOURS)
            or type(plan['original_limit']) is not int or not 1<=plan['original_limit']<=10000
            or not isinstance(ids,list) or len(ids)!=plan['original_limit']
            or any(not isinstance(value,str) or not value for value in ids) or len(set(ids))!=len(ids)
            or type(plan['native_case_slots']) is not int or plan['native_case_slots']!=14
            or type(plan['proposed_global_limit']) is not int
            or plan['future_normal_slots_unapproved']!=RESERVED_NORMAL_SLOTS
            or plan['acceptance_max_wait_seconds']!=28800
            or not directory.is_relative_to(Path(plan['workspace']))
            or not Path(plan['journal_path']).is_relative_to(Path(plan['workspace']))):
        raise ProtocolError('FIXED_PROTOCOL_BOUNDARIES_DIFFER')
    raw_config = document(directory/'settings.json',plan['config_sha256'])
    settings = Settings(_env_file=None,**raw_config)
    if exact_json(settings.model_dump(mode='json'))!=exact_json(raw_config):
        raise ProtocolError('ENVIRONMENT_CHANGED_PINNED_CONFIG')
    if settings.testnet is None or settings.research is None:
        raise ProtocolError('PINNED_TESTNET_SETTINGS_REQUIRED')
    expected = [{'slot':number,'symbol':symbol,'case':kind.value} for number,(symbol,kind) in enumerate(
        (symbol,kind) for symbol in settings.symbols for kind in ExecutionRiskCase)]
    if (set(settings.symbols)!={'BTCUSDT','ETHUSDT'} or plan['cases']!=expected or plan['native_case_slots']!=14
            or plan['proposed_global_limit']!=plan['original_limit']+14+RESERVED_NORMAL_SLOTS
            or settings.testnet.max_total_attempts!=plan['proposed_global_limit']
            or settings.testnet.journal_path.resolve()!=Path(plan['journal_path'])
            or execution_policy_fingerprint(settings)!=plan['policy_sha256'] or environment()!=plan['runtime_environment']):
        raise ProtocolError('FIXED_POLICY_OR_ENVIRONMENT_DIFFERS')
    hashes = document(directory/'runtime.json',plan['runtime_manifest_sha256'])
    frozen = directory/'runtime-src/crypto_arbitrage_bot'
    files = [*frozen.rglob('*.py'),*frozen.rglob('*.toml')]
    if {path.relative_to(frozen).as_posix() for path in files}!=set(hashes) or any(
        not path.resolve().is_relative_to(frozen.resolve())
        or any(p.is_symlink() or p.is_junction() for p in (path,*path.parents))
        or digest(path)!=hashes[path.relative_to(frozen).as_posix()] for path in files):
        raise ProtocolError('FROZEN_RUNTIME_CHANGED')
    python_hashes = {name:value for name,value in hashes.items() if name.endswith('.py')}
    if hashlib.sha256(exact_json(python_hashes).encode()).hexdigest()!=plan['source_sha256']:
        raise ProtocolError('FROZEN_IMPLEMENTATION_DIFFERS')
    current,original_sha = original_records(settings.testnet.journal_path,plan['original_attempt_ids'])
    if original_sha!=plan['original_records_sha256'] or len(current)>plan['proposed_global_limit']:
        raise ProtocolError('ORIGINAL_RECORDS_OR_GLOBAL_BUDGET_DIFFERS')
    return plan,settings


def inspect(reference: StudyRegistrationSettings) -> dict[str,Any]:
    plan,_ = load(reference)
    claims = list((reference.path.parent/'claims').iterdir())
    allowed = {f"{item['slot']:02d}.json" for item in plan['cases']}
    if any(path.name not in allowed or path.is_symlink() or not path.is_file() for path in claims):
        raise ProtocolError('UNKNOWN_CASE_CLAIM')
    return {'required_case_count':len(plan['cases']),'cases':plan['cases'],'original_attempts':plan['original_limit'],
        'native_case_slots':plan['native_case_slots'],'proposed_global_limit':plan['proposed_global_limit'],
        'future_normal_slots_unapproved':plan['future_normal_slots_unapproved'],'authorization':'NOT_AUTHORIZED',
        'case_claims_consumed':len(claims),'ends_at':plan['ends_at'],'inspection_private_requests':0,'orders_submitted':False,
        'execution_entry_implemented':False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','inspect'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    try:
        if args.command=='prepare':
            if args.config is None or args.directory is None:
                raise ProtocolError('PREPARATION_ARGUMENTS_REQUIRED')
            reference = prepare(load_settings(args.config),args.directory)
            print(exact_json({'status':'PROPOSED_NOT_AUTHORIZED','plan':reference.model_dump(mode='json')}))
        else:
            print(exact_json(inspect(StudyRegistrationSettings(path=args.plan,sha256=args.plan_sha256))))
        return 0
    except Exception as error:
        print(exact_json({'status':'PROTOCOL_REJECTED','code':str(error) if isinstance(error,ProtocolError)
            else type(error).__name__,'credentials_read':False,'orders_submitted':False}))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
