"""Compare Rust and batched tensor rules / full MCTS on the same fixed checkpoint."""
from __future__ import annotations

import argparse
import atexit
import signal
import subprocess
import sys
import gc
import json
import os
from pathlib import Path
import statistics
import time

import torch

from wavefield.encoding import legal_action_mask
from wavefield.engine import RustEngine,load_initial_state
from wavefield.eval import load_model
from wavefield.mcts import MctsConfig,parallel_mcts_selfplay_records
from wavefield.tensor_rules import TensorRules
from wavefield.tensor_selfplay import tensor_selfplay_records


def synchronize(device):
    if device=='mps':torch.mps.synchronize()
    elif device.startswith('cuda'):torch.cuda.synchronize()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('rules','selfplay','all'),default='all')
    parser.add_argument('--devices',default='cpu,mps')
    parser.add_argument('--batch-sizes',default='1,8,32,128')
    parser.add_argument('--games',type=int,default=8)
    parser.add_argument('--leaf-batch-size',type=int,default=1)
    parser.add_argument('--cpu-threads',type=int,default=1)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--max-plies',type=int,default=8)
    parser.add_argument('--simulations',type=int,default=128)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--rule-repeats',type=int,default=5)
    parser.add_argument('--pause-training',action='store_true',help='Temporarily SIGSTOP the verified local trainer group; resume on exit.')
    parser.add_argument('--positions',type=Path,help='Optional JSON list of varied rule states')
    args=parser.parse_args()
    if args.pause_training:
        status=json.loads((Path.home()/'Library/Application Support/WaveField/service-status.json').read_text())
        pid=int(status['training_pid'])
        command=subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True)
        if '-m wavefield.experiment' not in command or os.getpgid(pid)!=int(status['training_pgid']):
            raise RuntimeError('Refusing to pause an unverified training process group')
        group=os.getpgid(pid)
        def resume():
            try:os.killpg(group,signal.SIGCONT)
            except ProcessLookupError:pass
            print('Resumed production trainer',pid,flush=True)
        atexit.register(resume)
        signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
        os.killpg(group,signal.SIGSTOP)
        print('Temporarily paused production trainer',pid,flush=True)
    os.environ['WAVEFIELD_SPECTATOR_DISABLE']='1'
    torch.set_num_threads(args.cpu_threads)
    report=dict(checkpoint=str(args.checkpoint),rules=[],selfplay=[],config=vars(args).copy())
    report['config']={k:str(v) if isinstance(v,Path) else v for k,v in report['config'].items()}
    devices=args.devices.split(',')
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output)
    states=json.loads(args.positions.read_text()) if args.positions else [load_initial_state()]
    engine=RustEngine(retain_history=False)
    if args.mode in ('rules','all'):
        try:
            for size in map(int,args.batch_sizes.split(',')):
                positions=[states[i%len(states)] for i in range(size)]
                started=time.perf_counter()
                expected=[legal_action_mask(engine.legal_actions(state)) for state in positions]
                rust=[]
                for repeat in range(args.rule_repeats):
                    started=time.perf_counter()
                    for state in positions:engine.legal_actions(state)
                    rust.append(time.perf_counter()-started)
                row=dict(backend='rust_json_serial',batch_size=size,median_seconds=statistics.median(rust))
                report['rules'].append(row);print(json.dumps(row),flush=True);save()
                for device in devices:
                    rules=TensorRules(device);packed=rules.pack(positions)
                    with torch.inference_mode():
                        got=rules.legal_moves(packed).cpu().numpy()
                        import numpy as np
                        np.testing.assert_array_equal(got,expected)
                        times=[]
                        for repeat in range(args.rule_repeats):
                            synchronize(device);started=time.perf_counter()
                            rules.legal_moves(packed)
                            synchronize(device);times.append(time.perf_counter()-started)
                    row=dict(backend='tensor_'+device,batch_size=size,median_seconds=statistics.median(times))
                    if device=='mps':row['device_allocated_mib']=torch.mps.current_allocated_memory()/2**20
                    report['rules'].append(row);print(json.dumps(row),flush=True);save()
                    del packed,rules;gc.collect()
        finally:engine.close()
    if args.mode in ('selfplay','all'):
        model,view=load_model(args.checkpoint,256,torch.device('cpu'))
        config=MctsConfig(simulations=args.simulations,max_depth=32,max_tuning_actions=20,
                          value_discount=.997,repetition_limit=3,dirichlet_alpha=.3,
                          exploration_fraction=.25,sample_first_plies=20)
        backends=['rust']+['tensor_'+device for device in devices]
        for repeat in range(args.repeats):
            for backend in (backends if repeat%2==0 else list(reversed(backends))):
                print('START',backend,'repeat',repeat,flush=True)
                started=time.perf_counter();metrics={}
                if backend=='rust':
                    records=parallel_mcts_selfplay_records(model,args.games,args.workers,args.max_plies,50100,
                                                          config,input_view=view,cap_value='position')
                else:
                    device=backend.removeprefix('tensor_')
                    records=tensor_selfplay_records(model,args.games,args.max_plies,50100,config,device=device,
                                                     input_view=view,cap_value='position',inference_stats=metrics,leaf_batch_size=args.leaf_batch_size)
                    synchronize(device)
                elapsed=time.perf_counter()-started
                decisions=sum(len(record.samples) for record in records)
                plies=sum(record.stats.plies for record in records)
                row=dict(backend=backend,repeat=repeat,seconds=elapsed,games=args.games,plies=plies,
                         decisions=decisions,decisions_per_second=decisions/elapsed,
                         plies_per_second=plies/elapsed,wins={p:sum(r.stats.winner==p for r in records) for p in ('red','blue')},
                         search=metrics)
                report['selfplay'].append(row);print(json.dumps(row),flush=True);save()
                del records;gc.collect()
                if torch.backends.mps.is_available():torch.mps.empty_cache()
    save()

if __name__=='__main__':main()
