from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from wavefield.engine import load_initial_state
from wavefield.mcts import MctsConfig, mcts_selfplay_records, repetition_key


class RepetitionTest(unittest.TestCase):
    def test_key_preserves_rule_state_and_ignores_ui_and_counters(self):
        state = load_initial_state()
        changed = deepcopy(state)
        changed.update(turnNumber=100, message='changed', selectedPieceId='anything', history=[state])
        changed['pieces'].reverse()
        self.assertEqual(repetition_key(state), repetition_key(changed))
        for key in ['currentPlayer', 'components', 'activationOrders', 'waveScales', 'homeEnergy']:
            changed = deepcopy(state)
            changed[key] = 'different'
            self.assertNotEqual(repetition_key(state), repetition_key(changed))
        changed = deepcopy(state)
        changed['pieces'][0]['position']['x'] += 1
        self.assertNotEqual(repetition_key(state), repetition_key(changed))

    def play(self, limit, terminal=False):
        state = load_initial_state()
        engine = Mock()
        engine.legal_actions.return_value = [{'pieceId': 'test', 'destination': {'x': 0, 'y': 0}}]
        def move(current, _action, _engine):
            following = deepcopy(current)
            following['currentPlayer'] = 'red' if current['currentPlayer'] == 'blue' else 'blue'
            following['turnNumber'] += 1
            if terminal and following['turnNumber'] == state['turnNumber'] + 4:
                following['status'] = 'red-won'
            return following
        def sample(current, *_args, **_kwargs):
            return SimpleNamespace(player=current['currentPlayer'], action_kind=0, metadata={}, value=999)
        with patch('wavefield.mcts.select_mcts_action', return_value=SimpleNamespace(action=engine.legal_actions.return_value[0])), \
             patch('wavefield.mcts.sample_from_mcts_result', side_effect=sample), \
             patch('wavefield.mcts._apply_search_action', side_effect=move):
            return mcts_selfplay_records(engine, Mock(), games=1, max_plies=12, seed=1,
                config=MctsConfig(repetition_limit=limit, max_tuning_actions=0, value_discount=.997),
                cap_value='position', initial_states=[state])[0]

    def test_third_occurrence_draw_overrides_position_value(self):
        record = self.play(3)
        self.assertEqual(record.stats.plies, 4)
        self.assertEqual(record.stats.termination_reason, 'threefold_repetition')
        self.assertEqual(record.stats.plies_avoided_by_repetition, 8)
        self.assertIsNone(record.stats.winner)
        self.assertTrue(all(sample.value == 0 for sample in record.samples))

    def test_disabled_rule_and_terminal_win(self):
        self.assertEqual(self.play(0).stats.plies, 12)
        record = self.play(3, terminal=True)
        self.assertEqual(record.stats.winner, 'red')
        self.assertIsNone(record.stats.termination_reason)


if __name__ == '__main__':
    unittest.main()
