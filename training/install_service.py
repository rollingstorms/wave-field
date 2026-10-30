"""Install/update the per-user macOS Wave Field training service."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import plistlib
import shlex
import shutil
import signal
import subprocess
import sys
import time

from service_common import atomic_json, command_of, descendants, read_json, replace_option, terminate_tree


def move_with_link(source, destination):
    if source.is_symlink():
        if source.resolve() != destination.resolve():
            raise RuntimeError(f'Unexpected link: {source}')
        return
    if destination.exists():
        raise RuntimeError(f'Destination already exists: {destination}')
    source.rename(destination)
    source.symlink_to(destination, target_is_directory=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',default='selfplay-replay-discrete-20261028')
    parser.add_argument('--skip-build',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    base=Path.home()/'Library/Application Support/WaveField'
    runtime=base/'runtime';runtime.mkdir(parents=True,exist_ok=True)
    (base/'logs').mkdir(exist_ok=True)
    if not (base/'training-env/bin/python').exists():
        uv=shutil.which('uv')
        if uv is None:
            raise RuntimeError('Install uv before setting up the service runtime')
        subprocess.run([uv,'venv',str(base/'training-env'),'--python','python3.12'],check=True)
        subprocess.run([uv,'pip','install','--python',str(base/'training-env/bin/python'),
                        'torch==2.7.0','numpy==2.0.2'],check=True)
    oldrun=root/'training/runs'/args.run_name
    oldtracking=root/'training/tracking'
    config_path=base/'service-config.json'
    previous=read_json(config_path,{})
    label='com.wavefield.training';domain=f'gui/{os.getuid()}'
    # An update stops the managed job first; its recovery snapshot is retained.
    if previous:
        subprocess.run(['launchctl','bootout',f'{domain}/{label}'],capture_output=True)
        prior=read_json(base/'service-status.json',{})
        deadline=time.time()+25
        while time.time()<deadline and command_of(int(prior.get('manager_pid') or 0)):
            time.sleep(.25)
    for name in ['service.py','service_common.py','service_control.py','serve_selfplay.py','checkpoint_info.py','track_runs.py']:
        dest=runtime/'training'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/'training'/name,dest)
    shutil.copytree(root/'training/wavefield',runtime/'training/wavefield',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    binary=runtime/'engine/target/release/wave-field-engine';binary.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/'engine/target/release/wave-field-engine',binary)
    fixture=runtime/'engine/tests/product-initial-state.json';fixture.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/'engine/tests/product-initial-state.json',fixture)
    if not args.skip_build:
        node=shutil.which('node')
        subprocess.run([node,str(root/'node_modules/vite/bin/vite.js'),'build','--outDir',str(base/'web')],cwd=root,check=True)
    original_status=read_json(oldrun/'status.json',{})
    legacy_pid=int(original_status.get('training_pid') or 0)
    legacy_alive='wavefield.experiment' in command_of(legacy_pid) and str(oldrun) in command_of(legacy_pid)
    source_command=shlex.split((oldrun/'command.txt').read_text())
    # Freeze the existing process tree for the atomic directory moves, so no
    # writer sees the short gap between rename and installing its old-path link.
    paused=[]
    try:
        if legacy_alive:
            os.kill(legacy_pid,signal.SIGSTOP);paused.append(legacy_pid)
            for pid in sorted(descendants(legacy_pid)-{legacy_pid}):
                try:os.kill(pid,signal.SIGSTOP);paused.append(pid)
                except ProcessLookupError:pass
        watcher=read_json(oldtracking/'sync-state.json',{})
        for pidfile,expected in [('watcher.pid','track_runs.py'),('server.pid','mlflow'),('viewer.pid','vite')]:
            path=oldtracking/pidfile
            if path.exists():
                terminate_tree(int(path.read_text()),expected)
        # Preserve MLflow run identities while changing their canonical paths.
        mapping={key:str(base/'runs'/Path(key).name) for key in watcher}
        if mapping:
            code="""import json,sys
from mlflow import MlflowClient
c=MlflowClient(tracking_uri='sqlite:///'+sys.argv[1])
state=json.loads(sys.argv[2]);mapping=json.loads(sys.argv[3])
for key,value in state.items():c.set_tag(value['run_id'],'wavefield_run_dir',mapping[key])
"""
            subprocess.run([str(root/'training/.venv-tracking/bin/python'),'-c',code,
                            str(oldtracking/'mlflow.db'),json.dumps(watcher),json.dumps(mapping)],check=True,
                           env={**os.environ,'MLFLOW_DISABLE_AGENT_HINT':'1'})
        move_with_link(root/'training/runs',base/'runs')
        move_with_link(oldtracking,base/'tracking')
        move_with_link(root/'training/.venv-tracking',base/'tracking-env')
        state=read_json(base/'tracking/sync-state.json',{})
        atomic_json(base/'tracking/sync-state.json',{mapping.get(key,key):value for key,value in state.items()})
    finally:
        for pid in reversed(paused):
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
    link=runtime/'training/tracking'
    if not link.exists():link.symlink_to(base/'tracking',target_is_directory=True)
    run=base/'runs'/args.run_name
    rows=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines() if line.strip()]
    completed=max((row['iteration'] for row in rows if row.get('event')=='learning_update'),default=0)
    checkpoint_iteration=max(10,math.ceil(completed/10)*10)
    training=[str(base/'training-env/bin/python'),'-m','wavefield.experiment',*source_command[3:]]
    for key,value in {
        '--run-dir':str(run),'--mcts-workers':'8','--batch-size':'128','--effective-batch-size':'4096',
        '--epochs':'3','--mcts-value-discount':'0.997','--mcts-inference-device':'local',
        '--mcts-repetition-limit':'3',
        '--validation-games-per-batch':'1','--validation-capacity':'2048',
        '--milestone-every-updates':'1000','--keep-milestones':'3','--save-every':'1',
    }.items():replace_option(training,key,value)
    if '--recovery-checkpoints' not in training:training.append('--recovery-checkpoints')
    hostname=subprocess.check_output(['scutil','--get','LocalHostName'],text=True).strip().lower()+'.local'
    config={**previous,'base':str(base),'run_dir':str(run),'training_python':str(base/'training-env/bin/python'),
            'tracking_python':str(base/'tracking-env/bin/python'),'training_command':training,
            'tracking_runs':[str(base/'runs'/Path(key).name) for key in watcher],
            'target_large_batch_updates':previous.get('target_large_batch_updates',100000),'local_hostname':hostname,
            'large_batch_start_step':previous.get('large_batch_start_step'),
            'legacy':dict(training_pid=legacy_pid,launcher_pid=int((run/'launcher.pid').read_text()),
                          original_run_dir=str(oldrun),checkpoint_iteration=checkpoint_iteration) if legacy_alive else None}
    atomic_json(config_path,config)
    atomic_json(base/'tracking/spectator-config.json',{'directory':str(base/'tracking/selfplay-live'),'seed_offset':150210})
    overrides=read_json(base/'tracking/mcts-runtime.json',{})
    overrides.update(run_dir=str(run),final_iteration=1_000_000_000)
    atomic_json(base/'tracking/mcts-runtime.json',overrides)
    plist=Path.home()/f'Library/LaunchAgents/{label}.plist'
    plist.write_bytes(plistlib.dumps(dict(Label=label,ProgramArguments=[str(base/'training-env/bin/python'),str(runtime/'training/service.py'),'--config',str(config_path)],
        WorkingDirectory=str(runtime),RunAtLoad=True,KeepAlive=True,ThrottleInterval=15,ExitTimeOut=25,
        EnvironmentVariables={'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONPATH':str(runtime/'training'),'PYTHONUNBUFFERED':'1'},
        StandardOutPath=str(base/'logs/launchd.log'),StandardErrorPath=str(base/'logs/launchd-error.log'))))
    subprocess.run(['launchctl','bootstrap',domain,str(plist)],check=True)
    print(json.dumps(dict(service=label,base=str(base),waiting_for_checkpoint=checkpoint_iteration if legacy_alive else None),indent=2))


if __name__=='__main__':main()
