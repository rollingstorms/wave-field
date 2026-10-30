"""Compare fixed-position MCTS with local CPU and batched shared inference."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from wavefield.encoding import encode_state
from wavefield.engine import RustEngine, load_initial_state
from wavefield.mcts import MctsConfig, select_mcts_action
from wavefield.model import PolicyValueNet
from wavefield.shared_inference import SharedInferenceClient, SharedInferencePool

_model = None
_engine = None
_barrier = None


class CountedModel:
    def __init__(self, model):
        self.model, self.calls = model, 0
    def eval(self):
        self.model.eval(); return self
    def search_outputs(self, *args, **kwargs):
        self.calls += 1
        return self.model.search_outputs(*args, **kwargs)


def init_worker(model, spec, barrier):
    global _model, _engine, _barrier
    torch.set_num_threads(1)
    _model = CountedModel(SharedInferenceClient(spec) if spec is not None else model)
    _model.eval()
    _engine = RustEngine(retain_history=False)
    _barrier = barrier


def search_task(task):
    index, positions, config, count = task
    select_mcts_action(_model, positions[index % len(positions)], _engine,
                       MctsConfig(simulations=2, max_depth=2), input_view='piece_identity')
    _model.calls = 0
    _barrier.wait(timeout=120)
    started = time.perf_counter()
    results = []
    for j in range(count):
        result = select_mcts_action(_model, positions[(index + j) % len(positions)], _engine,
                                   config, input_view='piece_identity')
        results.append(dict(action=result.action, value=result.value,
                            visits=[int(row['visits']) for row in result.policy],
                            root_value=result.root_value))
    return dict(index=index, started=started, finished=time.perf_counter(),
                evaluations=_model.calls, results=results)


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = PolicyValueNet(hidden_size=checkpoint['hidden_size'],
        board_channels=checkpoint['board_channels'], side_size=checkpoint['side_size'],
        architecture=checkpoint['model_arch'], history_plies=checkpoint.get('history_plies', 1))
    model.load_state_dict(checkpoint['model'])
    return model.eval()


def collect_positions(directory):
    positions = [load_initial_state()]
    for path in sorted(directory.glob('game-*.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        valid = [r['state'] for r in rows if r['state']['status'] == 'playing'
                 and r.get('action') and r['action'].get('type') != 'tune']
        if valid:
            positions.append(valid[len(valid) // 2])
    return positions[:8]


def validate_outputs(model, positions, device):
    engine = RustEngine(retain_history=False)
    differences = []
    try:
        with SharedInferencePool(model, 1, device, max_batch=8, wait_ms=0) as service:
            client = SharedInferenceClient(service.spec)
            for position in positions:
                board, side, _ = encode_state(position, engine, engine.legal_actions(position), input_view='piece_identity')
                b, s = torch.from_numpy(board).unsqueeze(0), torch.from_numpy(side).unsqueeze(0)
                with torch.no_grad():
                    expected = model.search_outputs(b, s)
                actual = client.search_outputs(b, s)
                errors = [float((x - y).abs().max()) for x, y in zip(expected, actual)]
                for x, y in zip(expected, actual):
                    torch.testing.assert_close(x, y, atol=2e-4, rtol=2e-4)
                differences.append(errors)
    finally:
        engine.close()
    return dict(device=device, positions=len(positions), max_absolute_error=max(max(x) for x in differences))


def benchmark(model, positions, backend, workers, searches, config, max_batch, wait_ms):
    context = mp.get_context('spawn')
    barrier = context.Barrier(workers)
    service_context = SharedInferencePool(model, workers, backend, max_batch, wait_ms) if backend != 'local' else nullcontext(None)
    cold_start = time.perf_counter()
    with service_context as service, ProcessPoolExecutor(max_workers=workers,
            mp_context=context, initializer=init_worker,
            initargs=(None if service else model, service.spec if service else None, barrier)) as pool:
        futures = [pool.submit(search_task, (i, positions, config, searches)) for i in range(workers)]
        rows = [f.result(timeout=300) for f in futures]
        wall = max(r['finished'] for r in rows) - min(r['started'] for r in rows)
    total = time.perf_counter() - cold_start
    return dict(backend=backend, workers=workers, searches=workers * searches,
                wall_seconds=wall, searches_per_second=workers * searches / wall,
                evaluations=sum(r['evaluations'] for r in rows),
                evaluations_per_second=sum(r['evaluations'] for r in rows) / wall,
                including_startup_seconds=total,
                inference_stats=service.stats if service else {}, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--positions-dir', type=Path, default=Path('training/tracking/selfplay-live'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--searches-per-worker', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--simulations', type=int, default=128)
    parser.add_argument('--max-batch', type=int, default=8)
    parser.add_argument('--wait-ms', type=float, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['WAVEFIELD_SPECTATOR_DISABLE'] = '1'
    torch.set_num_threads(1)
    model = load_model(args.checkpoint)
    positions = collect_positions(args.positions_dir)
    config = MctsConfig(simulations=args.simulations, max_depth=32, max_tuning_actions=20,
                        value_discount=.997, top_k=0, dirichlet_alpha=0., exploration_fraction=0.)
    report = dict(checkpoint=str(args.checkpoint.resolve()), torch_version=torch.__version__,
                  config=vars(config), positions=positions, runs=[], validation=[],
                  note='Same fixed positions and weights. Root noise off. Training paused externally during timing.')
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    for backend in ['cpu', 'mps']:
        result = validate_outputs(model, positions, backend)
        report['validation'].append(result); print('VALIDATION', json.dumps(result), flush=True); save()
    # Alternate ordering to reduce thermal/order effects.
    for repeat in range(args.repeats):
        order = ['local', 'mps', 'cpu'] if repeat % 2 == 0 else ['cpu', 'mps', 'local']
        for backend in order:
            result = benchmark(model, positions, backend, args.workers, args.searches_per_worker,
                               config, args.max_batch, args.wait_ms)
            result['repeat'] = repeat
            report['runs'].append(result); save()
            print('BENCHMARK', json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)
    report['summary'] = {}
    for backend in ['local','cpu','mps']:
        runs = [r for r in report['runs'] if r['backend'] == backend]
        report['summary'][backend] = dict(median_wall_seconds=statistics.median(r['wall_seconds'] for r in runs),
            median_searches_per_second=statistics.median(r['searches_per_second'] for r in runs),
            median_including_startup_seconds=statistics.median(r['including_startup_seconds'] for r in runs))
    baseline = report['summary']['local']['median_searches_per_second']
    for row in report['summary'].values(): row['speedup_vs_local_cpu'] = row['median_searches_per_second'] / baseline
    save(); print('SUMMARY', json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
