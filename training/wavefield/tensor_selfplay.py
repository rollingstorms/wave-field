"""Self-play with batched tensor MCTS and the existing committed-action feed."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import time

import numpy as np
import torch

from .encoding import ACTION_SIZE, action_index, tuning_action_index
from .engine import load_initial_state
from .mcts import (_discount_mcts_values, _force_tune_result, _is_tune,
                   _sample_visit_result, record_turn_tactics, repetition_key)
from .selfplay import Sample, GameStats, GameRecord, _assign_values, _piece_counts
from .spectator import SpectatorFeed
from .tensor_rules import TensorRules
from .tensor_mcts import BatchedMcts, search_action_index


def sample_from_root(root, result):
    move_policy=np.zeros(ACTION_SIZE,np.float32);tuning_policy=np.zeros(14,np.float32)
    kind_policy=np.zeros(2,np.float32)
    total=sum(max(0,row['visits']) for row in result.policy)
    weights=[max(0,row['visits']) for row in result.policy] if total else [1]*len(result.policy)
    denominator=total or len(result.policy)
    for row,weight in zip(result.policy,weights):
        action=row['action'];probability=weight/denominator
        if _is_tune(action):
            kind_policy[1]+=probability;tuning_policy[tuning_action_index(action)]+=probability
        else:
            kind_policy[0]+=probability;move_policy[action_index(action)]+=probability
    chosen=result.action;tune=_is_tune(chosen)
    return Sample(root.board.copy(),root.side.copy(),root.move_mask.astype(np.float32),
                  -100 if tune else action_index(chosen),root.player,action_kind=int(tune),
                  legal_tuning_mask=root.observation_tune_mask.astype(np.float32),
                  tuning_action_index=tuning_action_index(chosen) if tune else -100,
                  kind_policy=kind_policy,move_policy=move_policy if move_policy.sum() else None,
                  tuning_policy=tuning_policy if tuning_policy.sum() else None,
                  metadata=dict(source='mcts_model',legal_count=int(root.move_mask.sum()),
                                legal_tuning_count=int(root.observation_tune_mask.sum()),mcts_visits=denominator,
                                mcts_root_value=result.root_value,mcts_selected_value=result.value,
                                rules_backend='tensor'))


@torch.inference_mode()
def tensor_selfplay_records(model, games, max_plies, seed, config, device='cpu',
                            input_view='piece_identity',history_plies=1,cap_value='position',
                            initial_states=None,on_game_complete=None,inference_stats=None,leaf_batch_size=1):
    if games<0 or max_plies<1:raise ValueError('Invalid game/ply count')
    if games==0:return []
    if history_plies!=1:raise ValueError('Tensor self-play currently requires history_plies=1')
    from .mcts_settings import worker_config
    config=worker_config(config,seed)
    # Never migrate the live learner or its optimizer to another device.
    search_model=copy.deepcopy(model).to(device).eval()
    rules=TensorRules(device)
    search=BatchedMcts(rules,search_model,config,input_view,leaves_per_root=leaf_batch_size)
    originals=[copy.deepcopy(initial_states[i]) if initial_states is not None and i<len(initial_states) else load_initial_state() for i in range(games)]
    tensor=rules.pack(originals)
    states=originals[:];before=originals[:]
    records=[None]*games;samples=[[] for _ in range(games)];stats=[GameStats() for _ in range(games)]
    plies=[0]*games;tune_counts=[0]*games
    rngs=[np.random.default_rng(seed+i) for i in range(games)]
    repetitions=[{repetition_key(state):1} if config.repetition_limit>0 else {} for state in states]
    active=list(range(games));started=time.perf_counter()
    with ExitStack() as stack:
        feeds=[stack.enter_context(SpectatorFeed(i,seed,config.simulations,max_plies,config.max_tuning_actions,config.value_discount)) for i in range(games)]
        for i in active:feeds[i].publish(states[i],0)
        while active:
            batch=tensor.take(rules.tensor(active,torch.long))
            results,roots=search.search(batch,[tune_counts[i] for i in active],[rngs[i] for i in active])
            selected=[];selected_ids=[];ended=[]
            for row,i in enumerate(active):
                result=results[row]
                if result is None:
                    state=states[i]
                    winner='blue' if state['currentPlayer']=='red' else 'red'
                    if state['status']=='playing':
                        states[i]={**state,'status':f'{winner}-won'}
                        # Match the reference loop's completed-turn accounting
                        # when a turn exhausts tuning without a playable move.
                        player=before[i]['currentPlayer']
                        stats[i].ai_turns_by_player[player]+=1
                        record_turn_tactics(stats[i],before[i],states[i])
                        stats[i].tune_actions_by_player[player]+=tune_counts[i]
                        stats[i].tune_turns_by_player[player]+=int(tune_counts[i]>0)
                        stats[i].effective_tune_changes_by_player[player]+=sum(
                            old!=new for kind in before[i]['components'][player]
                            for old,new in zip(before[i]['components'][player][kind],states[i]['components'][player][kind]))
                        plies[i]+=1;stats[i].plies=plies[i]
                    ended.append(i)
                    continue
                if roots[row].tune_mask.any():
                    force=False
                    if tune_counts[i]==0 and config.force_first_tune_prob>0:
                        force=rngs[i].random()<config.force_first_tune_prob
                    if not force and config.force_tune_prob>0:
                        force=rngs[i].random()<config.force_tune_prob
                    if force:result=_force_tune_result(result)
                if plies[i]<config.sample_first_plies:result=_sample_visit_result(result,rngs[i])
                samples[i].append(sample_from_root(roots[row],result))
                selected.append(result);selected_ids.append(i)
            if selected:
                ids=rules.tensor(selected_ids,torch.long)
                next_tensor=rules.step(tensor.take(ids),rules.tensor([search_action_index(r.action) for r in selected],torch.long))
                # Preserve the existing search's post-move no-playable-move loss.
                moved=rules.tensor([not _is_tune(r.action) for r in selected],torch.bool)
                no_moves=~rules.legal_moves(next_tensor).any(1)
                next_tensor.status=torch.where(moved&no_moves,2-next_tensor.player,next_tensor.status)
                json_states=rules.unpack(next_tensor,[states[i] for i in selected_ids])
                for name in tensor.__dataclass_fields__:
                    getattr(tensor,name)[ids]=getattr(next_tensor,name)
                for row,(i,result) in enumerate(zip(selected_ids,selected)):
                    states[i]=json_states[row]
                    if _is_tune(result.action):
                        tune_counts[i]+=1
                        feeds[i].publish(states[i],plies[i],result.action)
                    else:
                        player=before[i]['currentPlayer']
                        stats[i].ai_turns_by_player[player]+=1
                        record_turn_tactics(stats[i],before[i],states[i])
                        stats[i].tune_actions_by_player[player]+=tune_counts[i]
                        stats[i].tune_turns_by_player[player]+=int(tune_counts[i]>0)
                        stats[i].effective_tune_changes_by_player[player]+=sum(
                            old!=new for kind in before[i]['components'][player]
                            for old,new in zip(before[i]['components'][player][kind],states[i]['components'][player][kind]))
                        plies[i]+=1;stats[i].plies=plies[i];tune_counts[i]=0;before[i]=states[i]
                        if config.repetition_limit>0 and states[i]['status']=='playing':
                            key=repetition_key(states[i]);repetitions[i][key]=repetitions[i].get(key,0)+1
                            if repetitions[i][key]>=config.repetition_limit:
                                states[i]={**states[i],'terminationReason':'threefold_repetition','message':'Draw by threefold repetition'}
                                stats[i].termination_reason='threefold_repetition'
                                stats[i].plies_avoided_by_repetition=max_plies-plies[i]
                        feeds[i].publish(states[i],plies[i],result.action)
                    if states[i]['status']!='playing' or stats[i].termination_reason or plies[i]>=max_plies:
                        ended.append(i)
            for i in ended:
                final=states[i]
                feeds[i].publish(final,plies[i],complete=True)
                _assign_values(samples[i],final,cap_value);_discount_mcts_values(samples[i],config.value_discount)
                stats[i].status=final['status']
                stats[i].winner=final['status'].split('-')[0] if final['status']!='playing' else None
                stats[i].final_piece_counts=_piece_counts(final)
                records[i]=GameRecord(samples[i],stats[i],final)
                if on_game_complete is not None:on_game_complete(i,records[i])
            finished=set(ended);active=[i for i in active if i not in finished]
    if inference_stats is not None:
        inference_stats.update(search.metrics,backend='tensor',device=str(rules.device),games=games,
                               rules_implementation='metal' if rules.metal is not None else 'torch',leaf_batch_size=leaf_batch_size,
                               seconds=time.perf_counter()-started)
    return records
