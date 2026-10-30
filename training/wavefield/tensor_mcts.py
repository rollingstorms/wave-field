"""Batch one MCTS leaf per game; tensor rules and NN share the selected device.

Tree selection/backup stay on CPU. Board transitions, exhaustive move safety,
encoding, and policy/value inference are batched on CPU/MPS/CUDA. No rollout
teacher or reduced search budget is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .encoding import ACTION_SIZE, action_index, decode_action, decode_tuning_action, tuning_action_index
from .mcts import (MctsConfig, MctsResult, SearchChild, _action_key, _is_tune,
                   _select_child, _child_value_for_parent)
from .model import masked_policy_logits
from .tensor_rules import TensorRules, TensorState


@dataclass
class TensorNode:
    state: TensorState | None
    player: str
    tune_count: int
    after_move: bool = False
    visits: int = 0
    value_sum: float = 0.
    evaluation: float | None = None
    children: dict = field(default_factory=dict)
    outcome: int = 0
    move_mask: np.ndarray | None = None
    tune_mask: np.ndarray | None = None
    observation_tune_mask: np.ndarray | None = None
    board: np.ndarray | None = None
    side: np.ndarray | None = None

    @property
    def value(self):
        return self.value_sum/self.visits if self.visits else 0.


def search_action_index(action):
    return ACTION_SIZE+tuning_action_index(action) if _is_tune(action) else action_index(action)


def decode_search_action(index):
    return decode_tuning_action(index-ACTION_SIZE) if index>=ACTION_SIZE else decode_action(index)


class BatchedMcts:
    def __init__(self, rules: TensorRules, model, config: MctsConfig, input_view='piece_identity', leaves_per_root=1):
        if not 0<config.value_discount<=1:
            raise ValueError('MCTS value discount must be in (0, 1]')
        if config.simulations<1 or config.max_depth<1:
            raise ValueError('Tensor MCTS requires positive simulations and depth')
        if getattr(model,'history_plies',1)!=1:
            raise ValueError('Tensor MCTS currently supports history_plies=1')
        if leaves_per_root<1:raise ValueError("Leaf batch size must be positive")
        self.leaves_per_root=leaves_per_root
        self.rules=rules;self.model=model;self.config=config;self.input_view=input_view
        self.model.eval()
        self.metrics=dict(expansion_batches=0,expanded_positions=0,transition_batches=0,transition_positions=0)

    @torch.inference_mode()
    def _expand_many(self, nodes, root_rngs=None):
        if not nodes:return
        rules=self.rules;config=self.config
        state=TensorState.cat([n.state for n in nodes])
        moves,tunes=rules.legal_masks(state)
        raw_tunes=tunes.clone() if root_rngs is not None else None
        allowed=rules.tensor([config.max_tuning_actions<0 or n.tune_count<config.max_tuning_actions for n in nodes],torch.bool)
        tunes &= allowed[:,None]
        after_move=rules.tensor([n.after_move for n in nodes],torch.bool)
        loss=((~moves.any(1))&after_move)|(~(moves.any(1)|tunes.any(1)))
        state.status=torch.where(loss,2-state.player,state.status)
        active=state.status==0
        moves &= active[:,None];tunes &= active[:,None]
        board,side=rules.encode(state,rich=self.input_view=='piece_identity')
        kind,move_logits,tune_logits,values=self.model.search_outputs(board,side)
        temperature=max(config.policy_temperature,1e-6)
        kind_mask=torch.stack((moves.any(1),tunes.any(1)),1)
        kind_probs=torch.softmax(masked_policy_logits(kind,kind_mask)/temperature,1)
        move_probs=torch.softmax(masked_policy_logits(move_logits,moves)/temperature,1)*kind_probs[:,0,None]
        tune_probs=torch.softmax(masked_policy_logits(tune_logits,tunes)/temperature,1)*kind_probs[:,1,None]
        # One group of host transfers per expansion batch, never per candidate.
        moves_np=moves.cpu().numpy();tunes_np=tunes.cpu().numpy()
        probs=torch.cat((move_probs,tune_probs),1).cpu().numpy()
        values_np=values.cpu().numpy();statuses=state.status.cpu().numpy()
        # Only roots need observations as training targets. Internal boards are
        # left on device / regenerated, avoiding a large per-tree host cache.
        raw_tunes_np=raw_tunes.cpu().numpy() if raw_tunes is not None else None
        boards=board.cpu().numpy() if root_rngs is not None else None
        sides=side.cpu().numpy() if root_rngs is not None else None
        self.metrics['expansion_batches']+=1;self.metrics['expanded_positions']+=len(nodes)
        for row,node in enumerate(nodes):
            node.state=state.take(slice(row,row+1))
            node.outcome=int(statuses[row])
            node.move_mask=moves_np[row];node.tune_mask=tunes_np[row]
            if boards is not None:
                node.board=boards[row];node.side=sides[row]
                node.observation_tune_mask=raw_tunes_np[row]
            if node.outcome:
                winner='red' if node.outcome==1 else 'blue'
                node.evaluation=1. if node.player==winner else -1.
                continue
            node.evaluation=float(values_np[row])
            tune_scored=[(float(probs[row,ACTION_SIZE+i]),decode_tuning_action(int(i))) for i in np.flatnonzero(tunes_np[row])]
            move_scored=[(float(probs[row,i]),decode_action(int(i))) for i in np.flatnonzero(moves_np[row])]
            tune_scored.sort(key=lambda item:item[0],reverse=True)
            move_scored.sort(key=lambda item:item[0],reverse=True)
            selected={}
            for prior,action in tune_scored[:max(0,config.min_tune_candidates)]+move_scored[:max(0,config.min_move_candidates)]:
                selected[_action_key(action)]=(prior,action)
            for prior,action in sorted(tune_scored+move_scored,key=lambda item:item[0],reverse=True):
                if config.top_k>0 and len(selected)>=config.top_k:break
                selected.setdefault(_action_key(action),(prior,action))
            scored=sorted(selected.values(),key=lambda item:item[0],reverse=True)
            total=sum(max(1e-8,prior) for prior,action in scored)
            priors=np.array([max(1e-8,prior)/total for prior,action in scored])
            if root_rngs is not None and config.dirichlet_alpha>0 and config.exploration_fraction>0 and len(scored)>1:
                noise=root_rngs[row].dirichlet([config.dirichlet_alpha]*len(scored))
                priors=priors*(1-config.exploration_fraction)+noise*config.exploration_fraction
            for prior,(_,action) in zip(priors,scored):
                node.children[_action_key(action)]=SearchChild(action=action,prior=float(prior))

    def _parallel_simulations(self,roots):
        """Reserve independent leaves with virtual loss, then remove every reservation.

        The real simulation budget is unchanged. Batch size 1 uses the reference
        sequential path instead; larger sizes change the usual parallel search
        order without changing outcomes/rewards or training targets.
        """
        remaining=[self.config.simulations]*len(roots)
        while any(remaining):
            pending=[];paths=[];reservations=[]
            for root_index,root in enumerate(roots):
                for _ in range(min(self.leaves_per_root,remaining[root_index])):
                    node=root;path=[root];edges=[];blocked=False
                    while node.children and len(edges)<self.config.max_depth:
                        child=_select_child(node,self.config)
                        edges.append(1. if _is_tune(child.action) else self.config.value_discount)
                        if child.node is None:
                            tune=_is_tune(child.action)
                            placeholder=TensorNode(None,node.player if tune else ('red' if node.player=='blue' else 'blue'),
                                                   node.tune_count+1 if tune else 0,after_move=not tune)
                            child.node=placeholder
                            pending.append((node,child,placeholder))
                            path.append(placeholder)
                            break
                        node=child.node
                        if node.state is None:
                            blocked=True;break
                        path.append(node)
                    if blocked:break
                    for index,current in enumerate(path):
                        # Penalize this edge from its parent's perspective while
                        # the GPU evaluates it. Tuning preserves the player.
                        delta=0. if index==0 else (1. if path[index-1].player!=current.player else -1.)
                        current.visits+=1;current.value_sum+=delta
                        reservations.append((current,delta))
                    paths.append((path,edges));remaining[root_index]-=1
            if not paths:raise RuntimeError('Parallel MCTS made no progress')
            if pending:
                parents=TensorState.cat([parent.state for parent,child,node in pending])
                actions=self.rules.tensor([search_action_index(child.action) for parent,child,node in pending],torch.long)
                next_states=self.rules.step(parents,actions)
                self.metrics['transition_batches']+=1;self.metrics['transition_positions']+=len(pending)
                new=[]
                for row,(parent,child,node) in enumerate(pending):
                    node.state=next_states.take(slice(row,row+1));new.append(node)
                self._expand_many(new)
            for current,delta in reversed(reservations):
                current.visits-=1;current.value_sum-=delta
            for path,edges in paths:
                value=path[-1].evaluation
                assert value is not None
                for index in range(len(path)-1,-1,-1):
                    current=path[index];current.visits+=1;current.value_sum+=value
                    if index:
                        value*=edges[index-1]
                        if path[index-1].player!=current.player:value=-value

    @torch.inference_mode()
    def search(self, states: TensorState, tune_counts: list[int], rngs: list[np.random.Generator]):
        players=states.player.cpu().tolist()
        roots=[TensorNode(states.take(slice(i,i+1)),('red','blue')[players[i]],tune_counts[i]) for i in range(len(states))]
        self._expand_many(roots,rngs)
        if self.leaves_per_root>1:self._parallel_simulations(roots)
        for simulation in range(self.config.simulations if self.leaves_per_root==1 else 0):
            paths=[];discounts=[];pending=[]
            for root in roots:
                node=root;path=[node];edges=[]
                while node.children and len(edges)<self.config.max_depth:
                    child=_select_child(node,self.config)
                    edge=1. if _is_tune(child.action) else self.config.value_discount
                    edges.append(edge)
                    if child.node is None:
                        pending.append((node,child,path))
                        break
                    node=child.node;path.append(node)
                paths.append(path);discounts.append(edges)
            if pending:
                parents=TensorState.cat([node.state for node,child,path in pending])
                actions=self.rules.tensor([search_action_index(child.action) for node,child,path in pending],torch.long)
                next_states=self.rules.step(parents,actions)
                self.metrics['transition_batches']+=1;self.metrics['transition_positions']+=len(pending)
                new=[]
                for row,(parent,child,path) in enumerate(pending):
                    tune=_is_tune(child.action)
                    node=TensorNode(next_states.take(slice(row,row+1)),parent.player if tune else ('red' if parent.player=='blue' else 'blue'),
                                    parent.tune_count+1 if tune else 0,after_move=not tune)
                    child.node=node;path.append(node);new.append(node)
                self._expand_many(new)
            for path,edges in zip(paths,discounts):
                value=path[-1].evaluation
                assert value is not None
                for index in range(len(path)-1,-1,-1):
                    current=path[index];current.visits+=1;current.value_sum+=value
                    if index:
                        value*=edges[index-1]
                        if path[index-1].player!=current.player:value=-value
        results=[]
        for root in roots:
            children=sorted(root.children.values(),key=lambda c:c.node.visits if c.node is not None else 0,reverse=True)
            if not children:
                results.append(None);continue
            best=children[0]
            policy=[dict(action=child.action,prior=child.prior,
                         visits=child.node.visits if child.node else 0,
                         value=_child_value_for_parent(root,child.node,1. if _is_tune(child.action) else self.config.value_discount) if child.node else 0.)
                    for child in children]
            results.append(MctsResult(best.action,best.node.visits,policy[0]['value'],root.evaluation,policy))
        return results,roots
