"""One-shot worker-count handoff after an atomic training checkpoint is saved."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


def replace_option(command: list[str], name: str, value: str) -> None:
    if name in command:
        command[command.index(name) + 1] = value
    else:
        command.extend([name, value])


def process_command(pid: int) -> str:
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    return result.stdout.strip()


def descendant_pids(pid: int) -> set[int]:
    rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid='], text=True)
    pairs = [tuple(map(int, line.split())) for line in rows.splitlines()]
    found = {pid}
    while True:
        children = {child for child, parent in pairs if parent in found}
        if children <= found:
            return found
        found |= children


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--workers', required=True, type=int)
    parser.add_argument('--checkpoint-iteration', required=True, type=int)
    parser.add_argument('--training-pid', required=True, type=int)
    parser.add_argument('--launcher-pid', required=True, type=int)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    run = args.run_dir.resolve()
    checkpoint = run / f'checkpoint-iter-{args.checkpoint_iteration}.pt'
    state_path = run / 'worker-change.json'
    def record(state: str, **details) -> None:
        row = dict(state=state, at=datetime.now(timezone.utc).isoformat(), workers=args.workers,
                   checkpoint_iteration=args.checkpoint_iteration, **details)
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(row, indent=2) + '\n')
        temp.replace(state_path)
        print(json.dumps(row), flush=True)
    record('waiting_for_checkpoint', training_pid=args.training_pid)
    try:
        while not checkpoint.exists():
            command = process_command(args.training_pid)
            if 'wavefield.experiment' not in command or str(run) not in command:
                raise RuntimeError('Original trainer exited before the handoff checkpoint.')
            time.sleep(5)
        # save_checkpoint replaces this file atomically, so existence means complete.
        command = shlex.split((run / 'command.txt').read_text())
        status = json.loads((run / 'status.json').read_text())
        final_iteration = int(status['final_iteration'])
        if args.checkpoint_iteration >= final_iteration:
            raise ValueError('Handoff checkpoint must precede the final iteration.')
        replace_option(command, '--mcts-workers', str(args.workers))
        replace_option(command, '--resume-checkpoint', str(checkpoint))
        replace_option(command, '--iterations', str(final_iteration - args.checkpoint_iteration))
        (run / 'worker-change-command.txt').write_text(shlex.join(command) + '\n')
        original = process_command(args.training_pid)
        if 'wavefield.experiment' not in original or str(run) not in original:
            raise RuntimeError('Trainer identity changed before handoff; left processes untouched.')
        record('restarting', checkpoint=str(checkpoint))
        targets = descendant_pids(args.training_pid)
        for pid in sorted(targets, reverse=True):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        # Let the old launcher record its exit before a new launcher writes status.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if not process_command(args.training_pid) and not process_command(args.launcher_pid):
                break
            time.sleep(.25)
        else:
            raise RuntimeError('Old trainer/launcher did not exit; new trainer was not started.')
        (run / 'command-before-worker-change.txt').write_text((run / 'command.txt').read_text())
        (run / 'command.txt').write_text(shlex.join(command) + '\n')
        root = Path(__file__).resolve().parents[1]
        with (run / 'launcher.log').open('ab') as log:
            launcher = subprocess.Popen([command[0], str(run / 'launcher.py')], cwd=root,
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                        start_new_session=True)
        (run / 'launcher.pid').write_text(str(launcher.pid) + '\n')
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = json.loads((run / 'status.json').read_text())
            pid = int(status.get('training_pid', 0))
            if status.get('state') == 'running' and pid != args.training_pid:
                actual = process_command(pid)
                if f'--mcts-workers {args.workers}' in actual and str(checkpoint) in actual:
                    record('running', training_pid=pid, launcher_pid=launcher.pid,
                           checkpoint=str(checkpoint))
                    return
            if launcher.poll() is not None:
                raise RuntimeError('Replacement launcher exited; inspect run.log.')
            time.sleep(.5)
        raise RuntimeError('Replacement trainer startup was not confirmed.')
    except Exception as error:
        record('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
