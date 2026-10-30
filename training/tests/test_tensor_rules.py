import copy
import random
import unittest

import numpy as np
import torch

from wavefield.encoding import (ACTION_SIZE, PIECE_IDS, TUNING_SLOTS, action_index, decode_action,
                                decode_tuning_action, encode_state, legal_action_mask,
                                legal_tuning_actions, legal_tuning_mask, tuning_action_index)
from wavefield.engine import RustEngine, load_initial_state
from wavefield.mcts import repetition_key
from wavefield.tensor_rules import TensorRules


def rule_positions(count=64):
    """Independent signs plus reachable game positions, including saved deadlines."""
    rng=random.Random(573)
    engine=RustEngine(retain_history=False)
    states=[]
    current=load_initial_state()
    try:
        for index in range(count):
            if index%3==0:
                current=load_initial_state()
                for player in ('red','blue'):
                    for values in current['components'][player].values():
                        for slot in range(len(values)):values[slot]=rng.choice((-1,1))
                current=engine.begin_turn(current,analyze_checkmate=False)
            states.append(copy.deepcopy(current))
            moves=engine.legal_actions(current)
            if moves:
                current=engine.apply_action(current,rng.choice(moves))
            for _ in range(rng.randrange(3)):
                tunes=legal_tuning_actions(current)
                if tunes:current=engine.apply_tuning(current,rng.choice(tunes))
    finally:engine.close()
    return states


class TensorRulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.states=rule_positions()
        cls.rules=TensorRules()
        cls.engine=RustEngine(retain_history=False)

    @classmethod
    def tearDownClass(cls):cls.engine.close()

    def test_fields_masks_encodings_match_rust(self):
        for start in range(0,len(self.states),16):
            states=self.states[start:start+16]
            packed=self.rules.pack(states)
            fields=self.rules.field(packed).numpy()
            masks=self.rules.legal_moves(packed).numpy()
            tunes=self.rules.tuning_mask(packed).numpy()
            for rich in (False,True):
                boards,sides=(x.numpy() for x in self.rules.encode(packed,rich))
                for row,state in enumerate(states):
                    actions=self.engine.legal_actions(state)
                    board,side,mask=encode_state(state,self.engine,actions,input_view='piece_identity' if rich else 'base')
                    np.testing.assert_array_equal(fields[row].reshape(7,7),self.engine.evaluate_field(state))
                    np.testing.assert_array_equal(masks[row],mask)
                    np.testing.assert_array_equal(tunes[row],legal_tuning_mask(legal_tuning_actions(state)))
                    np.testing.assert_array_equal(boards[row],board)
                    np.testing.assert_array_equal(sides[row],side)

    def test_transitions_match_rust_including_deadlines_and_order(self):
        parents=[];actions=[];expected=[]
        for state in self.states:
            moves=self.engine.legal_actions(state)
            for move in moves[::max(1,len(moves)//3)]:
                parents.append(state);actions.append(action_index(move))
                expected.append(self.engine.apply_action(state,move))
            for tune in legal_tuning_actions(state):
                parents.append(state);actions.append(ACTION_SIZE+tuning_action_index(tune))
                expected.append(self.engine.apply_tuning(state,tune))
        losses=0
        for start in range(0,len(parents),32):
            inputs=parents[start:start+32]
            result=self.rules.step(self.rules.pack(inputs),self.rules.tensor(actions[start:start+32],torch.long))
            for parent,got,wanted in zip(inputs,self.rules.unpack(result,inputs),expected[start:start+32]):
                self.assertEqual(repetition_key(got),repetition_key(wanted))
                self.assertEqual(got['status'],wanted['status'])
                self.assertEqual(got['turnNumber'],wanted['turnNumber'])
                losses+=len(parent['pieces'])-len(got['pieces'])
        self.assertGreater(losses,0,'Transition corpus must exercise actual rescue-deadline losses')

    def test_rejects_unsupported_rules(self):
        state=load_initial_state();state['variant']='classic'
        with self.assertRaises(ValueError):self.rules.pack([state])
        state=load_initial_state();state['components']['blue']['rook'][0]=0
        with self.assertRaises(ValueError):self.rules.pack([state])

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_mps_matches_cpu_without_fallback(self):
        gpu=TensorRules('mps',fused=False);states=self.states[:8]
        cpu_state=self.rules.pack(states);gpu_state=gpu.pack(states)
        for a,b in ((self.rules.field(cpu_state),gpu.field(gpu_state)),
                    (self.rules.legal_moves(cpu_state),gpu.legal_moves(gpu_state)),
                    (self.rules.tuning_mask(cpu_state),gpu.tuning_mask(gpu_state))):
            torch.testing.assert_close(a,b.cpu(),rtol=0,atol=0)
        masks=self.rules.legal_moves(cpu_state)
        actions=[]
        for row,state in enumerate(states):
            move=masks[row].nonzero().flatten()
            actions.append(int(move[0]) if len(move) else ACTION_SIZE+int(self.rules.tuning_mask(cpu_state)[row].nonzero()[0]))
        cpu_next=self.rules.step(cpu_state,torch.tensor(actions))
        gpu_next=gpu.step(gpu_state,torch.tensor(actions,device='mps'))
        for got,wanted in zip(gpu.unpack(gpu_next,states),self.rules.unpack(cpu_next,states)):
            self.assertEqual(repetition_key(got),repetition_key(wanted))
        for a,b in zip(self.rules.encode(cpu_next),gpu.encode(gpu_next)):
            torch.testing.assert_close(a,b.cpu(),rtol=0,atol=0)

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_fused_metal_fields_masks_and_all_transition_kinds(self):
        gpu=TensorRules('mps')
        self.assertIsNotNone(gpu.metal)
        states=self.states
        cpu_state=self.rules.pack(states);gpu_state=gpu.pack(states)
        for a,b in ((self.rules.field(cpu_state),gpu.field(gpu_state)),
                    (self.rules.legal_moves(cpu_state),gpu.legal_masks(gpu_state)[0]),
                    (self.rules.tuning_mask(cpu_state),gpu.legal_masks(gpu_state)[1])):
            torch.testing.assert_close(a,b.cpu(),rtol=0,atol=0)
        parents=[];actions=[]
        masks=self.rules.legal_moves(cpu_state)
        tunes=self.rules.tuning_mask(cpu_state)
        for row,state in enumerate(states):
            for move in masks[row].nonzero().flatten().tolist()[::7]:
                parents.append(state);actions.append(move)
            for tune in tunes[row].nonzero().flatten().tolist():
                parents.append(state);actions.append(ACTION_SIZE+tune)
        for start in range(0,len(parents),32):
            inputs=parents[start:start+32];chosen=actions[start:start+32]
            cpu_next=self.rules.step(self.rules.pack(inputs),torch.tensor(chosen))
            gpu_next=gpu.step(gpu.pack(inputs),torch.tensor(chosen,device='mps'))
            for name in cpu_next.__dataclass_fields__:
                torch.testing.assert_close(getattr(cpu_next,name),getattr(gpu_next,name).cpu(),rtol=0,atol=0)
            for rich in (True,False):
                for a,b in zip(self.rules.encode(cpu_next,rich),gpu.encode(gpu_next,rich)):
                    torch.testing.assert_close(a,b.cpu(),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
