"""Bounded local evidence presentation, no network, finance or qualification."""

import argparse
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from crypto_arbitrage_bot.models.data_quality import DataQualityReason, DataQualityStatus
from crypto_arbitrage_bot.models.snapshot import SnapshotStatus
from crypto_arbitrage_bot.storage.opportunity_records import exact_json
from crypto_arbitrage_bot.utils.atomic_publish import publish_state
from crypto_arbitrage_bot.utils.clock import utc_now
from crypto_arbitrage_bot.utils.decimal_math import utc_timestamp

MAX_RECENT_ROWS = 8
MAX_REPORT_BYTES = 256000
_COLUMNS = {'symbol','created_at','snapshot_status','data_quality_status','data_quality_reason_codes','decision'}
_STATES = {'RUNNING_CONFIRMED','RUNNING_STALE_HEARTBEAT','STOPPED_CONFIRMED','FINISHED_NO_OWNER',
    'OWNER_UNVERIFIED','UNAVAILABLE','NOT_CREATED','READ','VALID','INVALID','EMPTY','NOT_AVAILABLE'}


def read_recent_quality(db: sqlite3.Connection) -> dict[str, Any]:
    """Use caller's read-only transaction; insertion order is explicitly limited.

    No numeric profit aggregation or historical quality reclassification. Length
    checks happen in SQLite before fetching fields, so large JSON is not loaded.
    Missing metadata remains unknown, not an empty set of successful samples.
    """
    base: dict[str, Any] = {'scope':'LAST_SAVED_ROWS_NOT_ALL_ATTEMPTS','limit':MAX_RECENT_ROWS}
    try:
        if not _COLUMNS <= {row[1] for row in db.execute('PRAGMA table_info(opportunities)')}:
            return {**base,'status':'NOT_AVAILABLE','rows':None,'reason':'METADATA_COLUMNS_MISSING'}
        lengths = db.execute('SELECT length(symbol),length(created_at),length(snapshot_status),'
            'length(data_quality_status),length(data_quality_reason_codes),length(decision) '
            'FROM opportunities ORDER BY rowid DESC LIMIT 8').fetchall()
        limits=(24,64,24,24,4096,24)
        if any(size is not None and size>bound for row in lengths for size,bound in zip(row,limits,strict=True)):
            raise ValueError('bounded metadata required')
        rows=[]
        for symbol,created,snapshot,quality,codes,decision in db.execute('SELECT symbol,created_at,snapshot_status,'
                'data_quality_status,data_quality_reason_codes,decision FROM opportunities ORDER BY rowid DESC LIMIT 8'):
            if not isinstance(symbol,str) or re.fullmatch(r'[A-Z0-9]{2,20}USDT',symbol) is None:
                raise ValueError('structured symbol required')
            when=utc_timestamp(datetime.fromisoformat(created))
            state=SnapshotStatus(snapshot).value
            valid=None if quality is None else DataQualityStatus(quality).value
            reasons=None if codes is None else json.loads(codes)
            if reasons is not None:
                if not isinstance(reasons,list) or len(reasons)>len(DataQualityReason):
                    raise ValueError('structured reasons required')
                reasons=[DataQualityReason(reason).value for reason in reasons]
                if len(set(reasons))!=len(reasons):
                    raise ValueError('unique reasons required')
            if decision not in ('PASS','OBSERVE','REJECT') or (valid=='VALID' and
                    (reasons is None or reasons or state!='COMPLETE')):
                raise ValueError('consistent recorded metadata required')
            rows.append({'symbol':symbol,'saved_at':when.isoformat(),'snapshot_status':state,
                'quality_status':valid,'quality_reasons':reasons,'decision':decision})
        return {**base,'status':'READ' if rows else 'EMPTY','rows':rows}
    except (sqlite3.Error,ValueError,TypeError,OverflowError,RecursionError):
        return {**base,'status':'UNAVAILABLE','rows':None,'reason':'METADATA_INVALID_OR_UNREADABLE'}


def quality_signal(value: object) -> dict[str, Any]:
    """Stable alert categories; no sample IDs, timestamps or arbitrary strings."""
    if not isinstance(value,dict) or value.get('status')!='READ' or not isinstance(value.get('rows'),list):
        status=value.get('status') if isinstance(value,dict) else None
        return {'status':status if status in ('EMPTY','NOT_AVAILABLE','UNAVAILABLE') else 'NOT_AVAILABLE',
                'valid':None,'invalid':None,'unknown':None,'reasons':None}
    rows=value['rows']
    if not rows or len(rows)>MAX_RECENT_ROWS or any(not isinstance(row,dict) for row in rows):
        return quality_signal(None)
    reasons=set()
    counts={'valid':0,'invalid':0,'unknown':0}
    for row in rows:
        quality=row.get('quality_status')
        codes=row.get('quality_reasons')
        category='valid' if quality=='VALID' and codes==[] and row.get('snapshot_status')=='COMPLETE' else (
            'invalid' if quality=='INVALID' and isinstance(codes,list) and codes else 'unknown')
        counts[category]+=1
        if codes is not None:
            if not isinstance(codes,list) or len(codes)>len(DataQualityReason) or any(not isinstance(code,str) or code not in
                    {r.value for r in DataQualityReason} for code in codes):
                return quality_signal(None)
            reasons.update(codes)
    return {'status':'READ',**counts,'reasons':sorted(reasons)}


def _count(value: object) -> str:
    return str(value) if type(value) is int and 0<=value<=2**63-1 else 'NULL'


def _state(value: object) -> str:
    return str(value) if isinstance(value,str) and value in _STATES else '未核查'


def _at(value: object) -> str:
    try:
        if isinstance(value,str) and len(value)<=64:
            value=datetime.fromisoformat(value)
        if isinstance(value,datetime):
            return utc_timestamp(value).isoformat()
    except (ValueError,TypeError,OverflowError):
        pass
    return 'NULL'


def _task_count(database: dict[str, Any], name: str) -> str:
    tasks=database.get('tasks')
    known={'COMPLETED','FAILED','PENDING','RUNNING'}
    if database.get('status')!='READ' or not isinstance(tasks,dict) or set(tasks)-known or any(
            type(value) is not int or not 0<=value<=2**63-1 for value in tasks.values()):
        return 'NULL'
    # GROUP BY covers the whole verified task table: absent categories are zero.
    # Missing/unreadable tables or unverified mappings remain unknown instead.
    return _count(tasks.get(name,0))


def render_report(report: dict[str, Any]) -> str:
    """Present bounded whitelisted facts; no raw errors, payloads or credentials."""
    lines=['# 自动套利系统本地验收快照','',f"检查时间（UTC）：{_at(report.get('evaluated_at'))}",'',
        '本页只说明这个检查时点。进程存活、数据合格、工程资格和独立净优势分别验收。',
        '交易所账户状态、当前交易所时钟、工程风险资格、独立净优势：本报告未核查。',
        '本报告不发放交易许可，不把历史 CLOSED 或研究 PASS 当作当前账户安全或盈利证明。','']
    research=report.get('research')
    if isinstance(research,dict):
        lines += ['## 公共研究','',f"实际 owner：{_state(research.get('status'))}；PID：{_count(research.get('worker_pid'))}",
            f"原研究次数预算：{_count(research.get('budget_consumed'))} / {_count(research.get('budget_limit'))}",
            f"原截止（UTC）：{_at(research.get('original_deadline'))}",'']
        counters=research.get('collection_counters')
        if isinstance(counters,dict):
            lines += [f"已结束尝试（包含失败）：{_count(counters.get('finished'))}；失败尝试：{_count(counters.get('failed'))}；中断尝试：{_count(counters.get('interrupted'))}",
                '原 legacy 消耗批次不冒充成功结束次数；缺少计数保持 NULL。','']
        database=research.get('database')
        if isinstance(database,dict):
            lines += [f"数据库读取：{_state(database.get('status'))}",'',
                      '| 任务 | 数量 |','|---|---:|']
            for name in ('COMPLETED','FAILED','PENDING','RUNNING'):
                lines.append(f"| {name} | {_task_count(database,name)} |")
            lines += ['', '已完整读取的任务分类中缺省类别为0；没有读取依据的数量为NULL。FAILED保留，不重做过去窗口。',
                f"下一未完成任务（UTC）：{_at(database.get('earliest_unfinished_due_at'))}",'']
            recent=database.get('recent_saved_quality')
            signal=quality_signal(recent)
            lines += ['### 最近保存样本的质量','',
                f"读取：{_state(signal['status'])}；VALID：{_count(signal['valid'])}；INVALID：{_count(signal['invalid'])}；未核查：{_count(signal['unknown'])}",
                '仅最后保存的最多8条，按 SQLite 插入顺序；不是全部尝试、当前盘口或完整候选覆盖。',
                '保存时的质量状态保持原记录；本报告不重新判定历史数据质量。','']
            if signal['reasons']:
                lines.append('记录的质量问题：'+', '.join(signal['reasons']))
            if isinstance(recent,dict) and isinstance(recent.get('rows'),list):
                lines += ['','| 币种 | 保存时间 UTC | 采集时质量 | Decision |','|---|---|---|---|']
                for row in recent['rows'][:MAX_RECENT_ROWS]:
                    if not isinstance(row,dict):
                        continue
                    symbol=row.get('symbol')
                    symbol=symbol if isinstance(symbol,str) and re.fullmatch(r'[A-Z0-9]{2,20}USDT',symbol) else '未核查'
                    decision=row.get('decision')
                    decision=decision if decision in ('PASS','OBSERVE','REJECT') else '未核查'
                    lines.append(f"| {symbol} | {_at(row.get('saved_at'))} | {_state(row.get('quality_status'))} | {decision} |")
    testnet=report.get('testnet')
    if isinstance(testnet,dict):
        lines += ['','## 测试网本地账本','',f"读取：{_state(testnet.get('status'))}",
            f"原入场预算：{_count(testnet.get('budget_consumed'))} / {_count(testnet.get('budget_limit'))}",
            '该预算来自本地账本；耗尽不重置。该读取不是交易所当前账户核查。','']
    publication=report.get('operational_report')
    if isinstance(publication,dict) and publication.get('status')=='UNAVAILABLE':
        lines += ['','报告发布异常：上一轮文件不能当作本轮成功；以当前 latest.json 为准。']
    digest=hashlib.sha256(exact_json(report).encode()).hexdigest()
    lines += ['','## 对应证据','',f'latest.json SHA256：`{digest}`',
        '文件与 latest.json 独立原子发布；核对摘要才能确认同一轮，不能把文件存在当成新鲜证据。',
        '本地事件文件保存状态和质量分类变化；本工具没有向外部用户发送通知。','']
    body='\n'.join(lines)
    if len(body.encode())>MAX_REPORT_BYTES:
        raise ValueError('bounded operational report required')
    return body


def publish_report(directory: Path, report: dict[str, Any]) -> None:
    """Caller owns monitor lock; publish same fsynced bytes without replay."""
    body=render_report(report).encode()
    temporary=directory/'report.md.tmp'
    if temporary.is_symlink() or (directory/'report.md').is_symlink():
        raise ValueError('owned regular report files required')
    with temporary.open('wb') as output:
        output.write(body)
        output.flush()
        os.fsync(output.fileno())
    publish_state(temporary,directory/'report.md')


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description='Render an existing bounded local monitor snapshot; no requests')
    parser.add_argument('--monitor-dir',type=Path,required=True)
    parser.add_argument('--original-plan-sha256',required=True)
    args=parser.parse_args(argv)
    try:
        from crypto_arbitrage_bot import monitor_archive
        directory=args.monitor_dir.resolve()
        monitor_archive.load_plan(directory,args.original_plan_sha256)
        report=monitor_archive.read(directory/'latest.json')
        at=_at(report.get('evaluated_at'))
        if at=='NULL' or not 0 <= (utc_now()-datetime.fromisoformat(at)).total_seconds() <= 90:
            print('注意：保存的检查时间过期或不可核实；以下是历史快照。')
        print(render_report(report))
        return 0
    except (OSError,ValueError,TypeError,RecursionError):
        print('LOCAL_REPORT_UNAVAILABLE')
        return 2


if __name__=='__main__':
    raise SystemExit(main())
