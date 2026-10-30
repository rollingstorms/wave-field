from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from wavefield.engine import RustEngine
from wavefield.mcts import MctsConfig, mcts_selfplay_records
from wavefield.model import PolicyValueNet
from wavefield.spectator import SpectatorFeed


class SpectatorTest(unittest.TestCase):
    def test_feed_preserves_selfplay_and_records_committed_actions(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        engine = RustEngine(retain_history=False)
        model = PolicyValueNet(hidden_size=32)
        args = dict(games=1, max_plies=2, seed=19,
                    config=MctsConfig(value_discount=.997, simulations=2, top_k=4, max_depth=2,
                                      max_tuning_actions=1, force_first_tune_prob=1.0))
        try:
            expected = mcts_selfplay_records(engine, model, **args)[0]
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                config = directory / 'config.json'
                config.write_text(json.dumps({'directory': tmp}))
                with patch('wavefield.spectator.CONFIG', config):
                    with SpectatorFeed(0, 19, 2) as feed:
                        actual = mcts_selfplay_records(engine, model, on_position=feed.publish, **args)[0]
                frames = [json.loads(line) for line in (directory / 'game-0.jsonl').read_text().splitlines()]
                self.assertEqual(expected.final_state, actual.final_state)
                self.assertEqual(len(frames), len(actual.samples) + 2)
                self.assertEqual([f['sequence'] for f in frames], list(range(len(frames))))
                self.assertEqual(frames[0]['ply'], 0)
                self.assertTrue(frames[-1]['complete'])
                self.assertEqual(frames[-1]['state'], actual.final_state)
                self.assertTrue(any(f['action'] and f['action'].get('type') == 'tune' for f in frames))
                self.assertTrue(all(not f['state']['history'] for f in frames))
                meta = json.loads((directory / 'game-0.meta.json').read_text())
                self.assertEqual(meta['sequence'], frames[-1]['sequence'])
                # A new batch replaces each game slot, keeping disk usage bounded.
                with patch('wavefield.spectator.CONFIG', config):
                    with SpectatorFeed(0, 20, 2) as feed:
                        feed.publish(actual.final_state, 0)
                self.assertEqual(len((directory / 'game-0.jsonl').read_text().splitlines()), 1)
        finally:
            engine.close()


if __name__ == '__main__':
    unittest.main()
