from copy import deepcopy
import unittest

from wavefield.mcts import record_turn_tactics
from wavefield.selfplay import GameStats


class TacticsTest(unittest.TestCase):
    def state(self):
        return dict(currentPlayer='red', pieces=[
            dict(id='r1', owner='red', type='rook', unstable=True),
            dict(id='r2', owner='red', type='pawn', unstable=True),
            dict(id='b1', owner='blue', type='rook', unstable=True)])

    def test_rescue_requires_every_endangered_piece_safe_and_alive(self):
        before = self.state()
        for kind in ('all_safe', 'one_unsafe', 'one_lost'):
            stats = GameStats()
            after = deepcopy(before)
            for piece in after['pieces']:
                if piece['owner'] == 'red': piece['unstable'] = False
            if kind == 'one_unsafe': after['pieces'][1]['unstable'] = True
            if kind == 'one_lost': after['pieces'].pop(1)
            record_turn_tactics(stats, before, after)
            self.assertEqual(stats.rescue_opportunities, 1)
            self.assertEqual(stats.rescues, int(kind == 'all_safe'))
            self.assertEqual(stats.rescue_opportunities_by_player, dict(red=1, blue=0))
            self.assertEqual(stats.losses_by_player['red'], int(kind == 'one_lost'))

    def test_losses_on_either_side_and_no_opportunity(self):
        before = self.state()
        for piece in before['pieces']: piece['unstable'] = False
        after = deepcopy(before)
        after['pieces'] = [after['pieces'][0]]
        stats = GameStats()
        record_turn_tactics(stats, before, after)
        self.assertEqual(stats.rescue_opportunities, 0)
        self.assertEqual(stats.rescues, 0)
        self.assertEqual(stats.losses_by_player, dict(red=1, blue=1))
        self.assertEqual(stats.losses_by_piece_type, dict(pawn=1, rook=1))
