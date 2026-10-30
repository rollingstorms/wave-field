from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from wavefield.mcts import (MctsConfig, SearchChild, SearchNode,
                            _child_value_for_parent, _discount_mcts_values, _run_simulation)
from wavefield.mcts_settings import worker_config


class ValueDiscountTest(unittest.TestCase):
    def test_search_backup_discounts_piece_moves_only_and_preserves_sign(self):
        gamma = 0.997
        tune = {'type': 'tune'}
        move = {'pieceId': 'blue-pawn-1'}
        for status, outcome in [('blue-won', 1), ('red-won', -1)]:
            root = SearchNode({'status': 'playing'}, 'blue', 0)
            tuned = SearchNode({'status': 'playing'}, 'blue', 1)
            moved = SearchNode({'status': 'playing'}, 'red', 0)
            terminal = SearchNode({'status': status}, 'blue', 0)
            root.children['tune'] = SearchChild(tune, 1, tuned)
            tuned.children['move'] = SearchChild(move, 1, moved)
            moved.children['move'] = SearchChild(move, 1, terminal)
            _run_simulation(root, None, None, MctsConfig(value_discount=gamma), 'cpu', 'base', None, 1)
            self.assertAlmostEqual(terminal.value, outcome)
            self.assertAlmostEqual(moved.value, -outcome * gamma)
            self.assertAlmostEqual(tuned.value, outcome * gamma ** 2)
            self.assertAlmostEqual(root.value, outcome * gamma ** 2)
            self.assertAlmostEqual(_child_value_for_parent(tuned, moved, gamma), root.value)
            self.assertAlmostEqual(_child_value_for_parent(root, tuned, 1), root.value)

    def test_targets_count_remaining_moves_without_charging_tuning(self):
        gamma = 0.997
        samples = [SimpleNamespace(action_kind=kind, value=value, metadata={})
                   for kind, value in [(1, 1.), (0, 1.), (1, -1.), (0, -1.)]]
        _discount_mcts_values(samples, gamma)
        self.assertEqual([s.metadata['remaining_plies'] for s in samples], [2, 2, 1, 1])
        for sample, expected in zip(samples, [gamma ** 2, gamma ** 2, -gamma, -gamma]):
            self.assertAlmostEqual(sample.value, expected)
        cutoff = [SimpleNamespace(action_kind=0, value=.4, metadata={})]
        _discount_mcts_values(cutoff, gamma)
        self.assertAlmostEqual(cutoff[0].value, .4 * gamma)
        terminal_tune = [SimpleNamespace(action_kind=1, value=1., metadata={})]
        _discount_mcts_values(terminal_tune, gamma)
        self.assertEqual(terminal_tune[0].value, 1.)

    def test_runtime_change_is_scoped_and_receipt_confirms_adoption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'runtime.json'
            path.write_text(json.dumps(dict(seed_offset=100, start_iteration=5,
                                          final_iteration=10, value_discount=.997, run_dir=tmp)))
            config = MctsConfig(value_discount=1.)
            with patch('wavefield.mcts_settings.RUNTIME_CONFIG', path):
                self.assertIs(worker_config(config, 104), config)
                self.assertFalse((Path(tmp) / 'selfplay-settings.json').exists())
                actual = worker_config(config, 105)
                self.assertEqual(actual.value_discount, .997)
                self.assertEqual(config.value_discount, 1.)
                receipt = json.loads((Path(tmp) / 'selfplay-settings.json').read_text())
                self.assertEqual(receipt['effective_iteration'], 5)
                self.assertEqual(receipt['value_discount'], .997)
                self.assertIs(worker_config(config, 111), config)

    def test_invalid_discount_is_rejected(self):
        for value in [0, -1, 1.01, float('nan')]:
            with self.assertRaises(ValueError):
                _discount_mcts_values([], value)


if __name__ == '__main__':
    unittest.main()
