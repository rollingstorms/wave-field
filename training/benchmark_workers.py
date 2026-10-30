"""Benchmark equal self-play game batches at different worker counts/backends."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import threading
import time

import torch
from benchmark_shared_inference import load_model
from wavefield.mcts import MctsConfig, parallel_mcts_selfplay_records


class Usage(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [(x, ctypes.c_uint64) for x in (
        'user_time','system_time','pkg_idle_wkups','interrupt_wkups','pageins',
        'wired_size','resident_size','phys_footprint','proc_start_abstime','proc_exit_abstime')]


def system_memory():
    result = {}
    swap = subprocess.check_output(['sysctl','-n','vm.swapusage'], text=True)
    result['swap_used_mib'] = float(re.search(r'used = ([\d.]+)M', swap).group(1))
    result['pressure'] = int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'], text=True))
    vm = subprocess.check_output(['vm_stat'], text=True)
    result['swapins_pages'] = int(re.search(r'Swapins:\s+(\d+)', vm).group(1))
    result['swapouts_pages'] = int(re.search(r'Swapouts:\s+(\d+)', vm).group(1))
    result['page_bytes'] = int(re.search(r'page size of (\d+) bytes', vm).group(1))
    return result


class Monitor:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.proc = ctypes.CDLL('/usr/lib/libproc.dylib') if platform.system() == 'Darwin' else None
        if self.proc:
            self.proc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.thread = threading.Thread(target=self.run, daemon=True)
    def sample(self):
        if not self.proc: return
        rows = subprocess.check_output(['ps','-axo','pid=,ppid=,pcpu='], text=True)
        processes = [tuple(map(float, x.split())) for x in rows.splitlines()]
        ids = {os.getpid()}
        while True:
            more = {int(pid) for pid,parent,cpu in processes if int(parent) in ids}
            if more <= ids: break
            ids |= more
        total = 0
        for pid in ids:
            info = Usage()
            if self.proc.proc_pid_rusage(pid, 0, ctypes.byref(info)) == 0:
                total += info.phys_footprint
        sample = system_memory()
        sample.update(footprint_mib=total / 2**20,
                      cpu_percent=sum(cpu for pid,parent,cpu in processes if int(pid) in ids))
        self.samples.append(sample)
    def run(self):
        while not self.stop.is_set():
            try: self.sample()
            except (subprocess.SubprocessError, ValueError, AttributeError): pass
            self.stop.wait(1)
    def __enter__(self):
        self.thread.start(); return self
    def __exit__(self, *args):
        self.stop.set(); self.thread.join(timeout=3)
    def summary(self):
        return dict(peak_tree_footprint_mib=max((x['footprint_mib'] for x in self.samples),default=0),
                    peak_cpu_percent=max((x['cpu_percent'] for x in self.samples),default=0),
                    peak_pressure=max((x['pressure'] for x in self.samples),default=0),
                    samples=self.samples)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--games',type=int,default=16)
    parser.add_argument('--max-plies',type=int,default=16)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    os.environ['WAVEFIELD_SPECTATOR_DISABLE']='1'
    torch.set_num_threads(1)
    model=load_model(args.checkpoint)
    config=MctsConfig(simulations=128,max_depth=32,max_tuning_actions=20,top_k=0,
                      value_discount=.997,dirichlet_alpha=.3,exploration_fraction=.25,sample_first_plies=20)
    report=dict(checkpoint=str(args.checkpoint.resolve()),games=args.games,max_plies=args.max_plies,
                config=vars(config),runs=[],cpu='Apple M1 Pro',memory_note='Summed proc_pid_rusage physical footprint for benchmark process tree; excludes paused production trainer. System swap counters include other applications.')
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    settings=[('local',8),('mps',8),('local',16),('mps',16)]
    for repeat in range(args.repeats):
        order=settings if repeat%2==0 else list(reversed(settings))
        for backend,workers in order:
            print('START',backend,workers,'repeat',repeat,flush=True)
            before=system_memory(); stats={}
            with Monitor() as monitor:
                started=time.perf_counter()
                records=parallel_mcts_selfplay_records(model,games=args.games,workers=workers,
                    max_plies=args.max_plies,seed=50100,config=config,input_view='piece_identity',
                    cap_value='position',inference_device=None if backend=='local' else backend,
                    inference_max_batch=workers,inference_wait_ms=2.,inference_stats=stats)
                elapsed=time.perf_counter()-started
            after=system_memory()
            signatures=[hashlib.sha256(json.dumps(record.final_state,sort_keys=True).encode()).hexdigest() for record in records]
            decisions=sum(len(record.samples) for record in records)
            result=dict(backend=backend,workers=workers,repeat=repeat,seconds=elapsed,
                games_per_hour=len(records)*3600/elapsed,decisions=decisions,
                decisions_per_second=decisions/elapsed,plies=sum(r.stats.plies for r in records),
                memory=monitor.summary(),swap_used_change_mib=after['swap_used_mib']-before['swap_used_mib'],
                swapin_mib=(after['swapins_pages']-before['swapins_pages'])*after['page_bytes']/2**20,
                swapout_mib=(after['swapouts_pages']-before['swapouts_pages'])*after['page_bytes']/2**20,
                signatures=signatures,inference_stats=stats)
            report['runs'].append(result);save()
            print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ('memory','signatures')} | {k:v for k,v in result['memory'].items() if k!='samples'}),flush=True)
    report['summary']={}
    for backend,workers in settings:
        runs=[r for r in report['runs'] if r['backend']==backend and r['workers']==workers]
        report['summary'][f'{backend}-{workers}']=dict(median_seconds=statistics.median(r['seconds'] for r in runs),
            median_decisions_per_second=statistics.median(r['decisions_per_second'] for r in runs),
            peak_footprint_mib=max(r['memory']['peak_tree_footprint_mib'] for r in runs),
            max_swapout_mib=max(r['swapout_mib'] for r in runs))
    baseline=report['runs'][0]['signatures']
    report['same_final_states']=[dict(backend=r['backend'],workers=r['workers'],repeat=r['repeat'],matching=sum(a==b for a,b in zip(baseline,r['signatures']))) for r in report['runs']]
    save();print('SUMMARY',json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
