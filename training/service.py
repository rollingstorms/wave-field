"""Supervise recoverable self-play training, MLflow, and the LAN viewer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import threading
import time

from service_common import atomic_json, command_of, descendants, read_json, replace_option, terminate_tree


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(('192.0.2.1', 9))
            return connection.getsockname()[0]
    except OSError:
        return '127.0.0.1'


class ProgressTail:
    def __init__(self, path):
        self.path, self.offset = path, 0
        self.result = dict(optimizer_steps=0, iteration=0, phase='starting', checkpoint=None)
    def poll(self):
        if not self.path.exists():
            return dict(self.result)
        if self.path.stat().st_size < self.offset:
            self.offset = 0
        with self.path.open('rb') as handle:
            handle.seek(self.offset)
            while True:
                line = handle.readline()
                if not line or not line.endswith(b'\n'):
                    break
                self.offset = handle.tell()
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                event = row.get('event')
                if event == 'config':
                    self.result['optimizer_steps'] = row['config'].get('optimizer_steps_start', self.result['optimizer_steps'])
                elif event == 'learning_update':
                    self.result.update(optimizer_steps=row['optimizer_steps_total'], iteration=row['iteration'], phase='self_play')
                elif event == 'selfplay_game':
                    self.result.update(active_iteration=row['iteration'], phase='self_play')
                elif event == 'train_epoch':
                    self.result['phase'] = 'neural_training'
                    self.result['optimizer_steps'] += row.get('optimizer_steps', 0)
                elif event == 'saved':
                    self.result.update(checkpoint=row['checkpoint'], checkpoint_iteration=row['iteration'])
        return dict(self.result)


def latest_progress(run):
    return ProgressTail(run / 'events.jsonl').poll()


class Supervisor:
    def __init__(self, path):
        self.config_path = path
        self.config = read_json(path)
        self.base = Path(self.config['base'])
        self.runtime = self.base / 'runtime'
        self.run = Path(self.config['run_dir'])
        self.logs = self.base / 'logs'
        self.logs.mkdir(exist_ok=True)
        self.stop = threading.Event()
        self.children = {}
        self.retry_at = {}
        self.trainer = None
        self.paused_ids = set()
        self.status_path = self.base / 'service-status.json'
        previous = read_json(self.status_path, {})
        self.boot_id = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'kern.boottime'], text=True).strip()
        rebooted = bool(previous.get('boot_id') and previous['boot_id'] != self.boot_id)
        self.status = dict(boot_id=self.boot_id, reboot_count=previous.get('reboot_count', 0) + int(rebooted),
                           start_reason='computer reboot' if rebooted else ('service restart' if previous else 'initial start'),
                           start_count=previous.get('start_count', 0) + 1,
                           training_restarts=previous.get('training_restarts', 0),
                           trainer_starts=previous.get('trainer_starts', 0),
                           inactivity_count=previous.get('inactivity_count', 0),
                           last_start_at=timestamp(), previous_heartbeat=previous.get('heartbeat'),
                           manager_pid=os.getpid(), run_dir=str(self.run),
                           target_large_batch_updates=self.config['target_large_batch_updates'],
                           workers=8, effective_batch_size=4096, microbatch_size=128,
                           inference_device='local CPU', state='starting', last_resume_at=previous.get('last_resume_at'),
                           last_gap_seconds=previous.get('last_gap_seconds'))
        command = self.config['training_command']
        def option(name, default):
            return command[command.index(name) + 1] if name in command else default
        self.search_backend = option('--mcts-backend', 'rust')
        self.rules_device = option('--mcts-rules-device', 'cpu')
        self.desired_workers = 1 if self.search_backend == 'tensor' else int(option('--mcts-workers', 8))
        self.status.update(search_backend=self.search_backend, rules_device=self.rules_device,
                           selfplay_games=int(option('--mcts-games-per-iteration', 8)))
        if self.search_backend == 'tensor':
            self.status['inference_device'] = f'batched tensor {self.rules_device.upper()}'
        self.desired_effective = int(option('--effective-batch-size', 4096))
        self.desired_micro = int(option('--batch-size', 128))
        self.updates_per_iteration = int(option('--epochs', 3)) * math.ceil(int(option('--replay-sample-size', 4096)) / self.desired_effective)
        self.status.update(workers=self.desired_workers, effective_batch_size=self.desired_effective, microbatch_size=self.desired_micro,
                           repetition_limit=int(option('--mcts-repetition-limit', 0)))
        self.progress = ProgressTail(self.run / 'events.jsonl')
        self.network_address = lan_ip()
        self.last_tick = time.time()
        self.log = self.logger('service')
        self.env = {**os.environ, 'PYTHONPATH': str(self.runtime / 'training'),
                    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MLFLOW_DISABLE_AGENT_HINT': '1',
                    'PYTHONUNBUFFERED': '1', **self.config.get('environment', {})}
        self.events = self.base / 'service-events.jsonl'
        if previous.get('boot_id') == self.boot_id and time.time() - previous.get('heartbeat', 0) < 300:
            owned = [previous.get('training_pgid')] + [component.get('pgid') for component in previous.get('components', {}).values()]
            for pgid in owned:
                if pgid and pgid != os.getpgrp():
                    try:
                        os.killpg(pgid, signal.SIGTERM)
                        os.killpg(pgid, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
        # Recover ownership after an unclean supervisor exit without duplicating
        # trainers or listeners that may have survived it.
        if not self.config.get('legacy'):
            prior_pid = int(previous.get('training_pid') or 0)
            if str(self.run) in command_of(prior_pid):
                terminate_tree(prior_pid, str(self.run))
        for component in previous.get('components', {}).values():
            pid = int(component.get('pid') or 0)
            if str(self.base) in command_of(pid):
                terminate_tree(pid, str(self.base))
        legacy = self.config.get('legacy')
        if legacy and legacy.get('paused_by_service') and legacy['original_run_dir'] in command_of(legacy['training_pid']):
            for pid in descendants(legacy['training_pid']):
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
            legacy['paused_by_service'] = False
            atomic_json(self.config_path, self.config)
            self.event('legacy_training_resumed')
        self.event('service_start', starts=self.status['start_count'], reason=self.status['start_reason'])

    def logger(self, name):
        logger = logging.getLogger(f'wavefield.{name}')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = RotatingFileHandler(self.logs / f'{name}.log', maxBytes=5_000_000, backupCount=3)
            handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
            logger.addHandler(handler)
        return logger

    def event(self, kind, **details):
        row = dict(event=kind, at=timestamp(), **details)
        with self.events.open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        self.log.info(json.dumps(row))

    def launch(self, name, command):
        logger = self.logger(name)
        process = subprocess.Popen(command, cwd=self.runtime, env=self.env,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        def consume():
            assert process.stdout is not None
            for line in process.stdout:
                logger.info(line.rstrip())
        threading.Thread(target=consume, daemon=True).start()
        self.event('component_start', component=name, pid=process.pid)
        return process

    def stop_process(self, process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            os.killpg(process.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    def commands(self):
        python = self.config['tracking_python']
        tracking = self.base / 'tracking'
        hostname = self.config['local_hostname'].lower()
        mlflow_port = str(self.config.get('mlflow_port', 5050))
        viewer_port = str(self.config.get('viewer_port', 5174))
        addresses = ['127.0.0.1', 'localhost', lan_ip(), hostname]
        origins = ','.join(f'http://{host}:{mlflow_port}' for host in addresses)
        hosts = ','.join(f'{host}:*' for host in addresses)
        return {
            'mlflow': [python, '-m', 'mlflow', 'server', '--host', '0.0.0.0', '--port', mlflow_port, '--workers', '1',
                       '--backend-store-uri', 'sqlite:///' + str(tracking / 'mlflow.db'),
                       '--artifacts-destination', str(tracking / 'artifacts'),
                       '--allowed-hosts', hosts, '--cors-allowed-origins', origins],
            'watcher': [python, str(self.runtime / 'training/track_runs.py'),
                        *self.config['tracking_runs'], '--tracking-uri', f'http://127.0.0.1:{mlflow_port}', '--watch', '--state', str(tracking / 'sync-state.json')],
            'viewer': [self.config['training_python'], str(self.runtime / 'training/serve_selfplay.py'),
                       '--web-root', str(self.base / 'web'), '--feed-dir', str(tracking / 'selfplay-live'),
                       '--status-file', str(self.status_path), '--port', viewer_port],
        }

    def ensure_components(self):
        address = lan_ip()
        if address != self.network_address:
            process = self.children.pop('mlflow', None)
            if process is not None:
                self.stop_process(process)
            self.event('network_address_changed', old=self.network_address, new=address)
            self.network_address = address
        for name, command in self.commands().items():
            process = self.children.get(name)
            if process is not None and process.poll() is not None:
                self.stop_process(process)
                self.event('component_exit', component=name, exit_code=process.returncode)
                self.children.pop(name)
                self.retry_at[name] = time.time() + 10
            if name not in self.children and time.time() >= self.retry_at.get(name, 0):
                self.children[name] = self.launch(name, command)

    def select_checkpoint(self):
        candidates = [self.run / 'checkpoint-latest.pt', self.run / 'checkpoint-previous.pt']
        candidates += sorted(self.run.glob('checkpoint-iter-*.pt'),
                             key=lambda p: int(p.stem.split('-')[-1]), reverse=True)
        if (self.run / 'checkpoint.pt').exists():
            candidates.append(self.run / 'checkpoint.pt')
        for path in candidates:
            if not path.exists():
                continue
            result = subprocess.run([self.config['training_python'],
                str(self.runtime / 'training/checkpoint_info.py'), str(path)], env=self.env,
                cwd=self.runtime, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return path, json.loads(result.stdout.strip().splitlines()[-1])
            self.event('invalid_checkpoint', checkpoint=str(path), error=result.stderr[-1000:])
        raise RuntimeError('No valid checkpoint is available; refusing to restart from random weights')

    def legacy_active(self):
        legacy = self.config.get('legacy')
        if not legacy:
            return False
        command = command_of(legacy['training_pid'])
        alive = 'wavefield.experiment' in command and legacy['original_run_dir'] in command
        checkpoint = self.run / f"checkpoint-iter-{legacy['checkpoint_iteration']}.pt"
        if alive and not checkpoint.exists():
            self.status.update(state='waiting_for_checkpoint', training_pid=legacy['training_pid'],
                               pending_checkpoint=legacy['checkpoint_iteration'], active_workers=6,
                               active_effective_batch_size=128)
            return True
        if alive:
            self.event('checkpoint_handoff', checkpoint=str(checkpoint))
            terminate_tree(legacy['training_pid'], legacy['original_run_dir'])
            deadline = time.time() + 10
            while time.time() < deadline and command_of(legacy['launcher_pid']):
                time.sleep(.1)
        else:
            self.event('legacy_recovery', reason='Previous trainer is no longer running')
        self.config['legacy'] = None
        atomic_json(self.config_path, self.config)
        return False

    def ensure_training(self):
        if self.status['state'] == 'target_complete':
            return
        if self.legacy_active():
            return
        if self.trainer is not None:
            code = self.trainer.poll()
            if code is None:
                self.status.update(state='training', training_pid=self.trainer.pid,
                                   active_workers=self.desired_workers, active_effective_batch_size=self.desired_effective)
                return
            self.stop_process(self.trainer)
            self.event('training_exit', exit_code=code)
            self.trainer = None
            self.retry_at['training'] = time.time() + 15
        if time.time() < self.retry_at.get('training', 0):
            self.status['state'] = 'recovering'
            return
        checkpoint, info = self.select_checkpoint()
        start = self.config.get('large_batch_start_step')
        if start is None:
            start = info['optimizer_steps']
            self.config['large_batch_start_step'] = start
            atomic_json(self.config_path, self.config)
            self.event('large_batch_phase_start', start_step=start, checkpoint=str(checkpoint))
        target = start + self.config['target_large_batch_updates']
        self.status.update(large_batch_start_step=start, target_optimizer_steps=target,
                           resumed_checkpoint=str(checkpoint), resumed_iteration=info['iteration'])
        if info['optimizer_steps'] >= target:
            self.status['state'] = 'target_complete'
            return
        if shutil.disk_usage(self.base).free < 1_000_000_000:
            self.status['state'] = 'paused_low_disk'
            return
        command = list(self.config['training_command'])
        remaining = target - info['optimizer_steps']
        replace_option(command, '--resume-checkpoint', str(checkpoint))
        replace_option(command, '--iterations', str(math.ceil(remaining / self.updates_per_iteration)))
        replace_option(command, '--target-optimizer-steps', str(target))
        replace_option(command, '--large-batch-start-step', str(start))
        previous_progress = self.progress.poll()
        self.progress.result.update(optimizer_steps=info["optimizer_steps"], iteration=info["iteration"],
                                    checkpoint=str(checkpoint), checkpoint_iteration=info["iteration"], phase="starting_self_play")
        self.status['trainer_starts'] += 1
        self.status['training_restarts'] = max(0, self.status['trainer_starts'] - 1)
        self.trainer = self.launch('training', command)
        (self.run / 'pid').write_text(str(self.trainer.pid) + '\n')
        self.status.update(state='training', training_pid=self.trainer.pid, training_pgid=self.trainer.pid, active_workers=self.desired_workers,
                           active_effective_batch_size=self.desired_effective, recovered_steps=info['optimizer_steps'],
                           last_recovery_at=timestamp(),
                           unsaved_updates_repeated=max(0, previous_progress['optimizer_steps'] - info['optimizer_steps']))
        self.event('training_resume', checkpoint=str(checkpoint), optimizer_steps=info['optimizer_steps'],
                   target_optimizer_steps=target, pid=self.trainer.pid)

    def check_disk(self):
        free = shutil.disk_usage(self.base).free
        self.status['free_disk_gib'] = round(free / 2**30, 2)
        if self.trainer is None or self.trainer.poll() is not None:
            return
        if free < 1_000_000_000 and not self.paused_ids:
            ids = descendants(self.trainer.pid)
            for pid in ids:
                try:
                    os.kill(pid, signal.SIGSTOP)
                except ProcessLookupError:
                    pass
            self.paused_ids = ids
            self.event('low_disk_pause', free_bytes=free)
        elif free > 1_500_000_000 and self.paused_ids:
            for pid in self.paused_ids:
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
            self.paused_ids.clear()
            self.event('disk_space_resume', free_bytes=free)
        if self.paused_ids:
            self.status['state'] = 'paused_low_disk'

    def write_status(self):
        progress = self.progress.poll()
        phase_start = self.config.get("large_batch_start_step")
        self.status.update(progress, heartbeat=time.time(), lan_ip=lan_ip(),
                           local_hostname=self.config['local_hostname'],
                           large_batch_updates=max(0, progress['optimizer_steps'] - (phase_start if phase_start is not None else progress['optimizer_steps'])),
                           components={name: dict(pid=process.pid, pgid=process.pid, running=process.poll() is None)
                                       for name, process in self.children.items()})
        atomic_json(self.status_path, self.status)
        atomic_json(self.run / 'status.json', dict(state='complete' if self.status['state']=='target_complete' else 'running',
                    training_pid=self.status.get('training_pid'), workers=self.status.get('active_workers', 8),
                    effective_batch_size=self.status.get('active_effective_batch_size', 4096),
                    target_optimizer_steps=self.status.get('target_optimizer_steps'), service_state=self.status['state']))

    def run_loop(self):
        for event in (signal.SIGTERM, signal.SIGINT):
            signal.signal(event, lambda *args: self.stop.set())
        try:
            while not self.stop.is_set():
                now = time.time()
                if now - self.last_tick > 35:
                    gap = now - self.last_tick
                    self.status['inactivity_count'] += 1
                    self.status.update(last_resume_at=timestamp(), last_gap_seconds=round(gap))
                    self.event('resume_after_inactivity', seconds=round(gap))
                self.last_tick = now
                self.ensure_components()
                try:
                    self.ensure_training()
                    self.check_disk()
                    self.status.pop('error', None)
                except Exception as error:
                    self.status.update(state='recovering', error=str(error))
                    self.log.exception('Training supervision error')
                    self.retry_at['training'] = time.time() + 30
                self.write_status()
                self.stop.wait(5)
        finally:
            self.event('service_stop')
            legacy = self.config.get('legacy')
            if self.trainer is None and legacy and legacy['original_run_dir'] in command_of(legacy['training_pid']):
                os.kill(legacy['training_pid'], signal.SIGSTOP)
                for pid in descendants(legacy['training_pid']):
                    try:
                        os.kill(pid, signal.SIGSTOP)
                    except ProcessLookupError:
                        pass
                legacy['paused_by_service'] = True
                atomic_json(self.config_path, self.config)
                self.event('legacy_training_suspended')
            if self.trainer is not None:
                self.stop_process(self.trainer)
            for name, process in self.children.items():
                if process.poll() is None:
                    self.stop_process(process)
            self.status.update(state='stopped', heartbeat=time.time())
            atomic_json(self.status_path, self.status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    base = Path(read_json(args.config)['base'])
    with (base / 'service.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Supervisor(args.config).run_loop()


if __name__ == '__main__':
    main()
