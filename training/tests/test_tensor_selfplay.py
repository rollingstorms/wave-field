import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from wavefield.engine import RustEngine,load_initial_state
from wavefield.mcts import MctsConfig,MctsResult,mcts_selfplay_records,repetition_key
from wavefield.model import PolicyValueNet
from wavefield.tensor_selfplay import tensor_selfplay_records
from wavefield.tensor_mcts import BatchedMcts


class TensorSelfplayTest(unittest.TestCase):
    def test_same_games_targets_discount_stats_and_committed_feed(self):
        torch.set_num_threads(1);torch.manual_seed(87)
        model=PolicyValueNet(hidden_size=16)
        config=MctsConfig(simulations=8,max_depth=4,max_tuning_actions=1,value_discount=.997,
                          force_first_tune_prob=1.,sample_first_plies=3,repetition_limit=3)
        engine=RustEngine(retain_history=False)
        try:
            expected=mcts_selfplay_records(engine,model,2,6,123,config,input_view='base',cap_value='position')
        finally:engine.close()
        with tempfile.TemporaryDirectory() as tmp:
            cfg=Path(tmp)/'config.json';cfg.write_text(json.dumps(dict(directory=tmp)))
            with patch('wavefield.spectator.CONFIG',cfg):
                actual=tensor_selfplay_records(model,2,6,123,config,input_view='base',cap_value='position')
            for index,(a,b) in enumerate(zip(expected,actual)):
                self.assertEqual(repetition_key(a.final_state),repetition_key(b.final_state))
                self.assertEqual(a.stats.to_dict(),b.stats.to_dict())
                self.assertEqual(len(a.samples),len(b.samples))
                for x,y in zip(a.samples,b.samples):
                    self.assertEqual(x.action_index,y.action_index)
                    self.assertEqual(x.tuning_action_index,y.tuning_action_index)
                    self.assertAlmostEqual(x.value,y.value,places=6)
                    for name in ('board','side','legal_mask','legal_tuning_mask','kind_policy','move_policy','tuning_policy'):
                        aa,bb=getattr(x,name),getattr(y,name)
                        if aa is None:self.assertIsNone(bb)
                        else:np.testing.assert_allclose(aa,bb,atol=1e-6)
                frames=[json.loads(row) for row in (Path(tmp)/f'game-{index}.jsonl').read_text().splitlines()]
                self.assertEqual(len(frames),len(b.samples)+2)
                self.assertTrue(frames[-1]['complete'])
                self.assertEqual(frames[-1]['state'],b.final_state)
                self.assertEqual([f['sequence'] for f in frames],list(range(len(frames))))

    def test_threefold_stays_neutral_with_real_reversible_moves(self):
        torch.set_num_threads(1)
        state=load_initial_state()
        state['pieces']=[piece for piece in state['pieces'] if piece['type']=='spy']
        state['pieces'][0]['position']=dict(x=1,y=1)
        state['pieces'][1]['position']=dict(x=5,y=5)
        engine=RustEngine(retain_history=False)
        try:state=engine.begin_turn(state,analyze_checkmate=False)
        finally:engine.close()
        original=BatchedMcts.search
        def cycle(search,states,counts,rngs):
            results,roots=original(search,states,counts,rngs)
            for row,root in enumerate(roots):
                blue=root.player=='blue'
                slot=4 if blue else 7
                square=int(root.state.positions[0,slot])
                x=2 if blue and square%7==1 else 1 if blue else 4 if square%7==5 else 5
                action=dict(pieceId='blue-spy-1' if blue else 'red-spy-1',destination=dict(x=x,y=1 if blue else 5))
                results[row]=MctsResult(action,1,0.,0.,[dict(action=action,visits=1,prior=1.,value=0.)])
            return results,roots
        with patch.object(BatchedMcts,'search',cycle):
            record=tensor_selfplay_records(PolicyValueNet(hidden_size=16),1,20,7,
                MctsConfig(simulations=1,max_tuning_actions=0,repetition_limit=3,value_discount=.997),
                input_view='base',initial_states=[state])[0]
        self.assertEqual(record.stats.plies,8)
        self.assertEqual(record.stats.termination_reason,'threefold_repetition')
        self.assertTrue(all(row.value==0 for row in record.samples))

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_gpu_selfplay_publishes_rust_valid_committed_transitions(self):
        torch.set_num_threads(1);torch.manual_seed(87)
        model=PolicyValueNet(hidden_size=16)
        with tempfile.TemporaryDirectory() as tmp:
            cfg=Path(tmp)/'config.json';cfg.write_text(json.dumps(dict(directory=tmp)))
            with patch('wavefield.spectator.CONFIG',cfg):
                records=tensor_selfplay_records(model,2,3,42,MctsConfig(simulations=4,max_depth=3,max_tuning_actions=1),
                                                device='mps',input_view='base')
            engine=RustEngine(retain_history=False)
            try:
                for game in range(2):
                    frames=[json.loads(line) for line in (Path(tmp)/f'game-{game}.jsonl').read_text().splitlines()]
                    for before,after in zip(frames,frames[1:]):
                        if after['complete']:continue
                        action=after['action']
                        expected=engine.apply_tuning(before['state'],action) if action.get('type')=='tune' else engine.apply_action(before['state'],action)
                        self.assertEqual(repetition_key(expected),repetition_key(after['state']))
            finally:engine.close()

if __name__=='__main__':unittest.main()
