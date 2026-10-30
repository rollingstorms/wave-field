import unittest
import copy
from collections import Counter
import numpy as np
import torch

from wavefield.augmentation import swap_sample, paired_samples, spatial_sample, PIECE_PERMUTATION
from wavefield.color_audit import identity_map, swap_state, swap_action
from wavefield.encoding import encode_state, action_index, PIECE_IDS, legal_tuning_actions, legal_tuning_mask
from wavefield.engine import RustEngine, load_initial_state
from wavefield.selfplay import Sample
from wavefield.experiment import train_samples
from wavefield.model import PolicyValueNet
from wavefield.mcts import repetition_key


class ColorAugmentationTest(unittest.TestCase):
    def test_encoded_mapping_matches_engine_and_is_involution(self):
        engine = RustEngine()
        mapping = identity_map()
        self.assertEqual([PIECE_IDS[i] for i in PIECE_PERMUTATION], [mapping[p] for p in PIECE_IDS])
        state = load_initial_state()
        try:
            for step in range(10):
                tunes = legal_tuning_actions(state)
                if tunes:
                    state = engine.apply_tuning(state, tunes[step % len(tunes)])
                actions = engine.legal_actions(state)
                if not actions:
                    break
                action = actions[step % len(actions)]
                for view in ('base', 'piece_identity'):
                    board, side, mask = encode_state(state, engine, actions, input_view=view)
                    policy = mask / mask.sum()
                    source = Sample(board, side, mask, action_index(action), state['currentPlayer'],
                                    value=.37, move_policy=policy,
                                    legal_tuning_mask=legal_tuning_mask(legal_tuning_actions(state)),
                                    kind_policy=np.array([.7, .3], np.float32),
                                    tuning_policy=np.arange(14, dtype=np.float32)/91,
                                    history_board=np.stack([board, board]), history_side=np.stack([side, side]))
                    swapped = swap_sample(source)
                    expected = encode_state(swap_state(state, mapping), engine, input_view=view)
                    for actual, wanted in zip((swapped.board, swapped.side, swapped.legal_mask), expected):
                        np.testing.assert_allclose(actual, wanted, atol=1e-6)
                    self.assertEqual(swapped.action_index, action_index(swap_action(action, mapping)))
                    self.assertEqual(swapped.value, source.value)
                    restored = swap_sample(swapped)
                    for field in ('board','side','legal_mask','move_policy','history_board','history_side',
                                  'kind_policy','tuning_policy','legal_tuning_mask'):
                        np.testing.assert_array_equal(getattr(restored,field),getattr(source,field))
                    self.assertEqual(restored.action_index, source.action_index)
                    self.assertEqual(restored.player, source.player)
                state = engine.apply_action(state, action)
        finally:
            engine.close()

    def test_all_sixteen_variants_preserve_engine_rules(self):
        def position(point, symmetry):
            x, y = point['x'], point['y']
            if symmetry >= 4:
                x = 6-x
            for _ in range(symmetry % 4):
                x, y = y, 6-x
            return dict(x=x,y=y)
        def state_transform(state, symmetry):
            result=copy.deepcopy(state)
            for piece in result['pieces']:
                piece['position']=position(piece['position'],symmetry)
            if 'ampSquares' in result:
                result['ampSquares']=[position(point,symmetry) for point in result['ampSquares']]
            return result
        engine=RustEngine(retain_history=False)
        state=load_initial_state()
        mapping=identity_map()
        try:
            for step in range(16):
                tunes=legal_tuning_actions(state)
                if tunes:
                    state=engine.apply_tuning(state,tunes[step % len(tunes)])
                actions=engine.legal_actions(state)
                if not actions:
                    break
                chosen=actions[step % len(actions)]
                for view in ('base','piece_identity'):
                    board,side,mask=encode_state(state,engine,input_view=view)
                    policy=mask * (np.arange(588,dtype=np.float32)+1)
                    policy/=policy.sum()
                    source=Sample(board,side,mask,action_index(chosen),state['currentPlayer'],value=-.4,
                                  move_policy=policy,history_board=np.stack([board,board]),
                                  history_side=np.stack([side,side]))
                    for color in (False,True):
                        base=swap_state(state,mapping) if color else state
                        source_variant=swap_sample(source) if color else source
                        base_action=swap_action(chosen,mapping) if color else chosen
                        for symmetry in range(8):
                            target=state_transform(base,symmetry)
                            actual=spatial_sample(source_variant,symmetry)
                            expected=encode_state(target,engine,input_view=view)
                            for got,wanted in zip((actual.board,actual.side,actual.legal_mask),expected):
                                np.testing.assert_allclose(got,wanted,atol=1e-6)
                            target_action={**base_action,'destination':position(base_action['destination'],symmetry)}
                            self.assertEqual(actual.action_index,action_index(target_action))
                            self.assertAlmostEqual(actual.move_policy[actual.action_index],policy[source.action_index])
                            self.assertEqual(actual.value,source.value)
                            np.testing.assert_array_equal(actual.history_board[0],actual.board)
                            if view == 'base':
                                after=engine.apply_action(base,base_action)
                                mapped_after=state_transform(after,symmetry)
                                got_after=engine.apply_action(target,target_action)
                                self.assertEqual(repetition_key(mapped_after),repetition_key(got_after))
                                self.assertEqual(mapped_after['status'],got_after['status'])
                                legal_tunes=legal_tuning_actions(base)
                                if legal_tunes:
                                    tune=legal_tunes[0]
                                    left=state_transform(engine.apply_tuning(base,tune),symmetry)
                                    right=engine.apply_tuning(target,tune)
                                    self.assertEqual(repetition_key(left),repetition_key(right))
                state=engine.apply_action(state,chosen)
        finally:
            engine.close()

    def test_spatial_pairs_balance_orientations_and_keep_independent_positions(self):
        row=Sample(np.zeros((13,7,7),np.float32),np.zeros(15,np.float32),
                   np.ones(588,np.float32),-100,'blue',action_kind=1,tuning_action_index=3)
        batch=paired_samples([row]*4096,seed=9,spatial=True)
        self.assertEqual(len(batch),4096)
        self.assertEqual(Counter(item.player for item in batch),{'red':2048,'blue':2048})
        self.assertEqual(Counter(item.metadata['spatial_symmetry'] for item in batch),dict.fromkeys(range(8),512))
        self.assertTrue(all(item.action_index == -100 and item.tuning_action_index == 3 for item in batch))
        self.assertEqual([r.metadata for r in batch], [r.metadata for r in paired_samples([row]*4096,9,True)])

    def test_paired_training_keeps_budget_and_validation_separate(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(1)
        rows=[Sample(rng.normal(size=(13,7,7)).astype(np.float32), np.zeros(15,np.float32),
                     np.ones(588,np.float32), i, 'blue', value=.2) for i in range(8)]
        paired=paired_samples(rows, 1)
        self.assertEqual(len(paired),8)
        self.assertEqual(sum(row is source for row in paired for source in rows),4)
        heldout=[rows.pop()]
        before=heldout[0].board.copy()
        class Logger:
            def __init__(self):self.events=[]
            def write(self,row):self.events.append(row)
        class Progress:
            def epoch(self,*args):pass
            def clear(self):pass
        model=PolicyValueNet(hidden_size=16)
        logger=Logger()
        steps=train_samples(model,torch.optim.SGD(model.parameters(),lr=.01),rows,
                            torch.device('cpu'),2,1,'test',1,logger,Progress(),
                            effective_batch_size=8,heldout_samples=heldout,color_augmentation=True,spatial_augmentation=True)
        self.assertEqual(steps,1)
        self.assertEqual(logger.events[0]['samples'],6)
        self.assertEqual(logger.events[0]['validation_samples'],1)
        self.assertTrue(logger.events[0]['color_augmentation'])
        np.testing.assert_array_equal(heldout[0].board,before)

if __name__ == '__main__':unittest.main()
