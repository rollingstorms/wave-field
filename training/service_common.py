"""Small, standard-library helpers for the local training service."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def command_of(pid: int) -> str:
    if pid <= 0:
        return ''
    return subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout.strip()


def descendants(pid: int) -> set[int]:
    pairs = [tuple(map(int, line.split())) for line in
             subprocess.check_output(['ps', '-axo', 'pid=,ppid='], text=True).splitlines()]
    found = {pid}
    while True:
        more = {child for child, parent in pairs if parent in found}
        if more <= found:
            return found
        found |= more


def terminate_tree(pid: int, expected: str, timeout: float = 8) -> None:
    if expected not in command_of(pid):
        return
    targets = descendants(pid)
    for target in sorted(targets, reverse=True):
        try:
            os.kill(target, signal.SIGTERM)
            os.kill(target, signal.SIGCONT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and command_of(pid):
        time.sleep(.1)
    for target in targets:
        if command_of(target):
            try:
                os.kill(target, signal.SIGKILL)
            except ProcessLookupError:
                pass


def replace_option(command: list[str], name: str, value: str) -> None:
    if name in command:
        command[command.index(name) + 1] = value
    else:
        command.extend([name, value])
