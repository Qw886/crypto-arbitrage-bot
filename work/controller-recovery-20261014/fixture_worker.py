"""Disposable subprocess exchange fixture. No sockets, credentials or native orders."""
import argparse
import json
import os
import socket
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import native_case_control as control

from crypto_arbitrage_bot.config.research import StudyRegistrationSettings as Reference
from crypto_arbitrage_bot.config.settings import load_settings
from crypto_arbitrage_bot.execution.demo import DemoResponse, demo_bot
from crypto_arbitrage_bot.execution.runner import TestnetBot as Bot
from crypto_arbitrage_bot.utils.clock import utc_now


def forbidden(*args,**kwargs):
    raise AssertionError('OFFLINE PROCESS FIXTURE HAS NO NETWORK')
socket.create_connection=forbidden

parser=argparse.ArgumentParser()
parser.add_argument('mode',choices=['setup','crash','resume','restart-first','restart-second'])
parser.add_argument('directory',type=Path)
args=parser.parse_args()
directory=args.directory.resolve()
case_slot=12 if args.mode.startswith('restart-') else 7
settings=load_settings(Path('docs/testnet-funding-acceptance.toml'))
settings=settings.model_copy(update={'testnet':settings.testnet.model_copy(update={'max_total_attempts':1})})
old,transport=demo_bot(settings,directory/'original')
state={'now':utc_now(),'settlement':utc_now()+timedelta(seconds=300)}
opener=transport.open

def response(request,**kwargs):
    row=opener(request,**kwargs)
    payload=json.loads(row.body)
    if '/premiumIndex?' in request.full_url:
        payload['lastFundingRate']='0'
        payload['nextFundingTime']=int(state['settlement'].timestamp()*1000)
        payload['time']=int(state['now'].timestamp()*1000)
    if '/account?' in request.full_url:
        if 'balances' in payload:
            for asset in payload['balances']:
                asset['locked']='0'
        else:
            payload['assets']=[{'asset':'USDT','walletBalance':'10000'}]
    return DemoResponse(payload)
transport.open=response
transport.clock=lambda:state['now']
for client in (old.spot,old.perp):
    client._clock_ms=lambda:int(state['now'].timestamp()*1000)
old.clock=old.collector.clock=old.research_runner.clock=lambda:state['now']

def save_exchange():
    (directory/'exchange.json').write_text(json.dumps({'orders':transport.orders,
        'base':{key:str(value) for key,value in transport.base.items()},
        'positions':{key:str(value) for key,value in transport.positions.items()}}),encoding='utf-8')

if args.mode=='setup':
    assert old.round_trip('ETHUSDT',Decimal('.01')).state.value=='CLOSED'
    reference=control.prepare(old.settings,directory/'controller',workspace=directory)
    control.protocol.exclusive(directory/'reference.json',reference.model_dump(mode='json'))
    approval={'schema':'native-case-current-authorization-v1','controller_sha256':reference.sha256,
        'scope':'TESTNET_NATIVE_CASES_ONLY','native_case_slots':14,'normal_trading_allowed':False,
        'current_user_confirmed':True,'authorization_record':'OFFLINE_FIXTURE','approved_at':utc_now()}
    control.protocol.exclusive(directory/'approval.json',approval)
    save_exchange()
else:
    reference=Reference(**json.loads((directory/'reference.json').read_text(encoding='utf-8')))
    approved=Reference(path=directory/'approval.json',sha256=control.protocol.digest(directory/'approval.json'))
    if args.mode=='restart-second':
        result=control.replay_slot(reference,case_slot,clock=utc_now)
        control.protocol.exclusive(directory/'restart-second-result.json',{'state':result,'actual_network_requests':0})
        raise SystemExit(0)
    exchange=json.loads((directory/'exchange.json').read_text(encoding='utf-8'))
    transport.orders=exchange['orders']
    transport.base={key:Decimal(value) for key,value in exchange['base'].items()}
    transport.positions={key:Decimal(value) for key,value in exchange['positions'].items()}
    def factory(config):
        bot=Bot(config,old.spot,old.perp,clock=lambda:state['now'])
        bot.collector.clock=bot.research_runner.clock=lambda:state['now']
        return bot
    def wait(seconds):
        if len(transport.orders)==4:
            state['now']=state['settlement']-timedelta(seconds=180)
        elif args.mode=='crash' and len(transport.orders)==6:
            save_exchange()
            os._exit(23)  # Actual process termination without stack/context cleanup.
        else:
            state['now']=state['settlement']+timedelta(seconds=61)
    if args.mode=='resume':
        report=json.loads((reference.path.parent/'cases/07/acceptance.json').read_text(encoding='utf-8'))
        from datetime import datetime
        state['settlement']=datetime.fromisoformat(report['settlement_at'])
        state['now']=state['settlement']+timedelta(seconds=61)
        control.stop(reference)
    result=control.run_offline_slot(reference,case_slot,approval=approved,bot_factory=factory,clock=utc_now,wait=wait)
    save_exchange()
    control.protocol.exclusive(directory/(args.mode+'-result.json'),{
        'state':result,'mock_order_count':len(transport.orders),'base':str(transport.base['ETHUSDT']),
        'perp':str(transport.positions['ETHUSDT']),'actual_network_requests':0})
