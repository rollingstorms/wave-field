"""Inspect or control the per-user Wave Field launchd service."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from service_common import command_of, read_json

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('action',choices=('status','start','stop','restart'))
args=parser.parse_args()
base=Path.home()/'Library/Application Support/WaveField'
label='com.wavefield.training'
domain=f'gui/{os.getuid()}'
plist=Path.home()/f'Library/LaunchAgents/{label}.plist'
if args.action=='status':
    print(json.dumps(read_json(base/'service-status.json',{}),indent=2))
else:
    if args.action in ('stop','restart'):
        previous=read_json(base/'service-status.json',{})
        result=subprocess.run(['launchctl','bootout',f'{domain}/{label}'],capture_output=True,text=True)
        if result.returncode and 'No such process' not in result.stderr:
            print(result.stderr.strip())
        deadline=time.time()+25
        pid=int(previous.get('manager_pid') or 0)
        while time.time()<deadline and command_of(pid):time.sleep(.25)
    if args.action in ('start','restart'):
        subprocess.run(['launchctl','bootstrap',domain,str(plist)],check=True)
    print(args.action,'requested')
