"""Read-only color symmetry and side-balanced frozen-checkpoint experiments."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .encoding import action_index, encode_state, legal_tuning_actions, tuning_limit_allows
from .engine import RustEngine, load_initial_state, without_undo_history
from .eval import load_model
from .mcts import (MctsConfig, _apply_search_action, _no_move_loss, _sample_visit_result,
                   repetition_key, select_mcts_action)
from .model import masked_policy_logits
from .selfplay import select_model_full_turn


def opposite(player):
    return 'red' if player == 'blue' else 'blue'


def identity_map():
    pieces = load_initial_state()['pieces']
    return {piece['id']: next(other['id'] for other in pieces
            if other['owner'] != piece['owner'] and other['type'] == piece['type']
            and other['position'] == rotate(piece['position'])) for piece in pieces}


def rotate(position):
    return dict(x=6-position['x'], y=6-position['y'])


def swap_action(action, mapping):
    if action.get('type') == 'tune':
        return copy.deepcopy(action)
    return dict(pieceId=mapping[action['pieceId']], destination=rotate(action['destination']))


def swap_state(state, mapping):
    result = copy.deepcopy(state)
    for piece in result['pieces']:
        piece['owner'] = opposite(piece['owner'])
        piece['id'] = mapping[piece['id']]
        piece['position'] = rotate(piece['position'])
    for key in ('components', 'activationOrders'):
        result[key] = {player: copy.deepcopy(state[key][opposite(player)]) for player in ('red', 'blue')}
    result['currentPlayer'] = opposite(state['currentPlayer'])
    if state['status'] in ('red-won', 'blue-won'):
        result['status'] = opposite(state['status'].split('-')[0]) + '-won'
    result['history'] = []
    result['selectedPieceId'] = None
    result['message'] = ''
    if 'ampSquares' in state:
        result['ampSquares'] = [rotate(position) for position in state['ampSquares']]
    return result


def distribution(model, state, engine, view):
    actions = engine.legal_actions(state)
    board, side, mask = encode_state(state, engine, actions, input_view=view)
    with torch.no_grad():
        kind, moves, tunes, value = model.search_outputs(torch.from_numpy(board[None]), torch.from_numpy(side[None]))
        move_probs = torch.softmax(masked_policy_logits(moves, torch.from_numpy(mask[None])), dim=1)[0].numpy()
        kind_probs = torch.softmax(kind, dim=1)[0].numpy()
    return actions, move_probs, kind_probs, float(value.item())


def js_divergence(left, right):
    middle = (left+right)/2
    return float(sum(np.sum(p * np.log(np.maximum(p, 1e-30)/np.maximum(middle, 1e-30))) for p in (left, right))/2)


def symmetry_audit(model, view, positions, mapping, engine):
    rows = []
    for state in positions:
        swapped = swap_state(state, mapping)
        actions, probs, kinds, value = distribution(model, state, engine, view)
        alternatives, other_probs, other_kinds, other_value = distribution(model, swapped, engine, view)
        expected = {json.dumps(swap_action(action, mapping), sort_keys=True) for action in actions}
        actual = {json.dumps(action, sort_keys=True) for action in alternatives}
        field_error = float(np.max(np.abs(np.asarray(engine.evaluate_field(swapped))
                                          + np.asarray(engine.evaluate_field(state))[::-1, ::-1])))
        transition_errors = 0
        for action in actions[:2] + legal_tuning_actions(state)[:1]:
            left = swap_state(_apply_search_action(state, action, engine), mapping)
            right = _apply_search_action(swapped, swap_action(action, mapping), engine)
            transition_errors += int(repetition_key(left) != repetition_key(right) or left['status'] != right['status'])
        mapped = np.array([other_probs[action_index(swap_action(action, mapping))] for action in actions])
        original = np.array([probs[action_index(action)] for action in actions])
        rows.append(dict(player=state['currentPlayer'], legal_match=expected == actual,
            field_error=field_error, transition_errors=transition_errors,
            value=value, swapped_value=other_value, value_difference=abs(value-other_value),
            move_total_variation=float(np.abs(original-mapped).sum()/2),
            move_js=js_divergence(original, mapped), kind_total_variation=float(np.abs(kinds-other_kinds).sum()/2),
            same_best_move=bool(np.argmax(original) == np.argmax(mapped))))
    return dict(positions=len(rows), legal_mismatches=sum(not row['legal_match'] for row in rows),
        max_field_error=max(row['field_error'] for row in rows),
        transition_errors=sum(row['transition_errors'] for row in rows),
        mean_value_difference=float(np.mean([row['value_difference'] for row in rows])),
        mean_move_total_variation=float(np.mean([row['move_total_variation'] for row in rows])),
        mean_move_js=float(np.mean([row['move_js'] for row in rows])),
        mean_kind_total_variation=float(np.mean([row['kind_total_variation'] for row in rows])),
        best_move_agreement=float(np.mean([row['same_best_move'] for row in rows])), rows=rows)


_models = {}
_engine = None
_view = None


def init_worker(paths):
    global _engine, _view
    os.nice(10)
    torch.set_num_threads(1)
    for key, path in paths.items():
        _models[key], _view = load_model(Path(path), 256, torch.device('cpu'))
    _engine = RustEngine(retain_history=False)


def match(task):
    start = time.perf_counter()
    state = copy.deepcopy(task['initial_state'])
    rng = np.random.default_rng(task['seed'])
    torch.manual_seed(task['seed'])
    config = MctsConfig(simulations=128, top_k=0, max_depth=32,
        max_tuning_actions=20, value_discount=.997, sample_first_plies=20)
    repetitions = {repetition_key(state): 1}
    reason = None
    decisions = 0
    plies = 0
    for ply in range(600):
        if state['status'] != 'playing': break
        player = state['currentPlayer']
        model = _models['latest' if player == task['latest_side'] else task['anchor']]
        if task['mode'] == 'policy':
            state, _, tunes = select_model_full_turn(model, state, _engine, temperature=.3,
                device='cpu', record_samples=False, input_view=_view, max_tuning_actions=20)
            decisions += tunes + 1
            if state['status'] == 'playing' and not _engine.legal_actions(state):
                state = _no_move_loss(state)
        else:
            tune_count = 0
            while state['status'] == 'playing' and state['currentPlayer'] == player:
                if not _engine.legal_actions(state) and not tuning_limit_allows(tune_count, 20):
                    state = _no_move_loss(state)
                    break
                result = select_mcts_action(model, state, _engine, config, device='cpu',
                    input_view=_view, tune_count=tune_count, rng=rng)
                if ply < 20: result = _sample_visit_result(result, rng)
                decisions += 1
                state = _apply_search_action(state, result.action, _engine)
                if result.action.get('type') == 'tune': tune_count += 1
                else: break
        plies = ply + 1
        if state['status'] == 'playing':
            key = repetition_key(state)
            repetitions[key] = repetitions.get(key, 0) + 1
            if repetitions[key] >= 3:
                reason = 'threefold_repetition'
                break
    winner = state['status'].split('-')[0] if state['status'] in ('blue-won', 'red-won') else None
    return {key: value for key, value in task.items() if key != 'initial_state'} | dict(
        winner=winner, result='draw' if winner is None else 'win' if winner == task['latest_side'] else 'loss',
        reason=reason or (state['status'] if winner else 'ply_cap'), plies=plies,
        decisions=decisions, seconds=time.perf_counter()-start)


def summarize(rows):
    def group(items):
        n = len(items)
        w, l, d = (sum(row['result'] == outcome for row in items) for outcome in ('win','loss','draw'))
        score = (w + d/2)/n if n else None
        return dict(games=n, wins=w, losses=l, draws=d, score=score,
                    mean_plies=sum(row['plies'] for row in items)/n if n else None)
    result = {}
    for anchor in sorted({row['anchor'] for row in rows}):
        result[anchor] = {}
        for mode in ('policy', 'mcts'):
            chosen = [row for row in rows if row['anchor'] == anchor and row['mode'] == mode]
            result[anchor][mode] = dict(overall=group(chosen),
                by_color={side: group([row for row in chosen if row['latest_side']==side]) for side in ('blue','red')},
                by_turn={str(first): group([row for row in chosen if row['latest_first']==first]) for first in (True,False)},
                by_starting_color={side: group([row for row in chosen if row['starting_color']==side]) for side in ('blue','red')})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--policy-seeds',type=int,default=8)
    parser.add_argument('--mcts-seeds',type=int,default=4)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    inputs={'latest':args.run_dir/'checkpoint-latest.pt', 'anchor_mid':args.run_dir/'checkpoint-update-000001002.pt',
            'anchor_early':args.run_dir/'checkpoint-iter-40.pt'}
    paths={};versions={}
    for key, source in inputs.items():
        checkpoint=torch.load(source,map_location='cpu',weights_only=False)
        versions[key]=dict(iteration=checkpoint['iteration'], optimizer_steps=checkpoint.get('optimizer_steps'),source=str(source))
        path=args.out/f'{key}.pt'
        metadata={name:checkpoint[name] for name in ('model','hidden_size','model_arch','input_view','history_plies','board_channels','side_size')}
        torch.save(metadata,path);paths[key]=str(path)
        del checkpoint,metadata
    (args.out/'checkpoints.json').write_text(json.dumps(versions,indent=2))
    mapping=identity_map()
    candidates={}
    for path in (args.run_dir.parents[1]/'tracking/selfplay-live').glob('game-*.jsonl'):
        for line in path.read_text().splitlines():
            try: state=json.loads(line)['state']
            except (ValueError,KeyError): continue
            if state['status']=='playing': candidates[repetition_key(state)]=state
    positions=[]
    for player in ('blue','red'):
        states=[state for state in candidates.values() if state['currentPlayer']==player]
        indexes=np.linspace(0,len(states)-1,min(48,len(states)),dtype=int)
        positions.extend(states[index] for index in indexes)
    if not positions: raise RuntimeError('No live positions available for color audit')
    (args.out/'positions.json').write_text(json.dumps(positions))
    engine=RustEngine(retain_history=False)
    model,view=load_model(Path(paths['latest']),256,torch.device('cpu'))
    symmetry=symmetry_audit(model,view,positions,mapping,engine)
    (args.out/'symmetry.json').write_text(json.dumps(symmetry,indent=2))
    print('SYMMETRY',json.dumps({k:v for k,v in symmetry.items() if k!='rows'}),flush=True)
    if symmetry['legal_mismatches'] or symmetry['transition_errors'] or symmetry['max_field_error']>1e-6:
        raise RuntimeError('Transformation is not engine-equivalent; refusing to interpret paired matches')
    engine.close();del model
    initial=load_initial_state();swapped=swap_state(initial,mapping)
    tasks=[]
    for anchor in ('anchor_early','anchor_mid'):
        for mode,count in [('policy',args.policy_seeds),('mcts',args.mcts_seeds)]:
            for seed in range(count):
                for state in (initial,swapped):
                    for latest_side in ('blue','red'):
                        tasks.append(dict(anchor=anchor,mode=mode,seed=910000+seed,
                            starting_color=state['currentPlayer'],latest_side=latest_side,
                            latest_first=latest_side==state['currentPlayer'],initial_state=state))
    results=[]
    import multiprocessing
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=init_worker,initargs=(paths,)) as pool:
        futures=[pool.submit(match,task) for task in tasks]
        for future in as_completed(futures):
            row=future.result();results.append(row)
            with (args.out/'matches.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
            (args.out/'results.json').write_text(json.dumps(dict(checkpoints=versions,finished=len(results),
                total=len(tasks),matches=summarize(results)),indent=2))
            print(f"MATCH {len(results)}/{len(tasks)} {row['anchor']} {row['mode']} {row['latest_side']} first={row['latest_first']} {row['result']} {row['plies']} plies",flush=True)
    print('COMPLETE',str(args.out),flush=True)


if __name__=='__main__':main()
