import unittest

import numpy as np
import torch

from wavefield.engine import RustEngine,load_initial_state
from wavefield.mcts import MctsConfig,select_mcts_action
from wavefield.model import PolicyValueNet
from wavefield.tensor_rules import TensorRules
from wavefield.tensor_mcts import BatchedMcts,search_action_index


class TensorMctsTest(unittest.TestCase):
    def test_batched_search_matches_reference_visits_values_and_targets(self):
        torch.set_num_threads(1);torch.manual_seed(324)
        model=PolicyValueNet(hidden_size=16)
        config=MctsConfig(simulations=16,max_depth=5,max_tuning_actions=2,value_discount=.997)
        engine=RustEngine(retain_history=False)
        try:
            first=load_initial_state()
            second=engine.apply_action(first,engine.legal_actions(first)[5])
            states=[first,second]
            reference=[select_mcts_action(model,state,engine,config,input_view='base',rng=np.random.default_rng(22+i)) for i,state in enumerate(states)]
            rules=TensorRules()
            result,roots=BatchedMcts(rules,model,config,input_view='base').search(rules.pack(states),[0,0],[np.random.default_rng(22+i) for i in range(2)])
            for a,b in zip(reference,result):
                self.assertEqual(a.action,b.action)
                self.assertEqual(a.visits,b.visits)
                self.assertAlmostEqual(a.root_value,b.root_value,places=6)
                self.assertAlmostEqual(a.value,b.value,places=6)
                ap={search_action_index(row['action']):row for row in a.policy}
                bp={search_action_index(row['action']):row for row in b.policy}
                self.assertEqual(set(ap),set(bp))
                for key in ap:
                    self.assertEqual(ap[key]['visits'],bp[key]['visits'])
                    self.assertAlmostEqual(ap[key]['prior'],bp[key]['prior'],places=6)
                    self.assertAlmostEqual(ap[key]['value'],bp[key]['value'],places=6)
        finally:engine.close()

    def test_parallel_leaves_keep_exact_budget_and_clear_virtual_visits(self):
        torch.set_num_threads(1);torch.manual_seed(324)
        model=PolicyValueNet(hidden_size=16)
        config=MctsConfig(simulations=32,max_depth=5,max_tuning_actions=2,value_discount=.997)
        rules=TensorRules();states=rules.pack([load_initial_state()]*2)
        search=BatchedMcts(rules,model,config,input_view='base',leaves_per_root=8)
        results,roots=search.search(states,[0,0],[np.random.default_rng(4),np.random.default_rng(5)])
        for result,root in zip(results,roots):
            self.assertEqual(root.visits,32)
            self.assertEqual(sum(row['visits'] for row in result.policy),32)
            self.assertTrue(root.move_mask[search_action_index(result.action)] if result.action.get('type')!='tune' else root.tune_mask[search_action_index(result.action)-588])
            stack=[root]
            while stack:
                node=stack.pop()
                self.assertIsNotNone(node.state)
                self.assertGreaterEqual(node.visits,0)
                self.assertLessEqual(abs(node.value_sum),node.visits+1e-6)
                stack.extend(child.node for child in node.children.values() if child.node is not None)
        self.assertLess(search.metrics['expansion_batches'],33)

    def test_terminal_root_returns_no_action(self):
        torch.set_num_threads(1)
        state=load_initial_state();state['status']='red-won'
        rules=TensorRules()
        results,roots=BatchedMcts(rules,PolicyValueNet(hidden_size=16),MctsConfig(simulations=2),input_view='base').search(rules.pack([state]),[0],[np.random.default_rng(3)])
        self.assertEqual(results,[None]);self.assertEqual(roots[0].outcome,1)

if __name__=='__main__':unittest.main()
