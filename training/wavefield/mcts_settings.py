"""Optional batch-scoped updates for already-running spawned MCTS workers."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

RUNTIME_CONFIG = Path(__file__).resolve().parents[1] / 'tracking' / 'mcts-runtime.json'


def worker_config(config: Any, seed: int) -> Any:
    try:
        settings = json.loads(RUNTIME_CONFIG.read_text())
    except FileNotFoundError:
        return config
    iteration = seed - int(settings['seed_offset'])
    if not int(settings['start_iteration']) <= iteration <= int(settings['final_iteration']):
        return config
    discount = float(settings['value_discount'])
    if not math.isfinite(discount) or not 0 < discount <= 1:
        raise ValueError('MCTS value discount must be in (0, 1].')
    config = replace(config, value_discount=discount)
    # This receipt is written only once workers actually adopt the setting.
    active = dict(value_discount=discount, effective_iteration=iteration,
                  at=datetime.now(timezone.utc).isoformat(),
                  discount_unit='piece_move', repetition_limit=config.repetition_limit,
                  repetition_rule='threefold' if config.repetition_limit == 3 else 'disabled')
    path = Path(settings['run_dir']) / 'selfplay-settings.json'
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(active) + '\n')
    temporary.replace(path)
    return config
