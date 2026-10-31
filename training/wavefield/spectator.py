"""Optional local feed of committed self-play actions (never search branches)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

CONFIG = Path(__file__).resolve().parents[1] / 'tracking' / 'spectator-config.json'


class SpectatorFeed:
    def __init__(self, game: int, seed: int, simulations: int, max_plies: int = 600, max_tuning_actions: int = 20, value_discount: float = 1.0, *, iteration=None, game_id=None, model_version=None):
        self.handle = None
        self.sequence = 0
        self.game = game
        self.seed = seed
        self.simulations = simulations
        self.value_discount = value_discount
        self.max_plies = max_plies
        self.max_tuning_actions = max_tuning_actions
        self.iteration = iteration
        self.game_id = game_id
        self.model_version = model_version
        if os.environ.get("WAVEFIELD_SPECTATOR_DISABLE") == "1":
            return
        try:
            config = json.loads(CONFIG.read_text())
            if iteration is None and isinstance(config.get('seed_offset'), int):
                if seed < config['seed_offset']:
                    return
                self.iteration = seed - config['seed_offset']
            self.directory = Path(config['directory'])
            self.directory.mkdir(parents=True, exist_ok=True)
            self.handle = (self.directory / f'game-{game}.jsonl').open('w')
        except (OSError, ValueError, KeyError):
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.handle:
            self.handle.close()

    def publish(self, state: dict[str, Any], ply: int, action=None, complete=False):
        if self.handle is None:
            return
        try:
            metadata = dict(game=self.game, seed=self.seed, simulations=self.simulations,
                            iteration=self.iteration, game_id=self.game_id, model_version=self.model_version, value_discount=self.value_discount, max_plies=self.max_plies,
                            max_tuning_actions=self.max_tuning_actions,
                            sequence=self.sequence, ply=ply, updated=time.time(),
                            complete=complete, status=state['status'])
            metadata['termination_reason'] = state.get('terminationReason')
            self.handle.write(json.dumps(dict(**metadata, action=action, state={**state, 'history': []})) + '\n')
            self.handle.flush()
            path = self.directory / f'game-{self.game}.meta.json'
            temporary = path.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(metadata))
            temporary.replace(path)
            self.sequence += 1
        except OSError:
            # An optional viewer must never interrupt training.
            self.handle.close()
            self.handle = None
