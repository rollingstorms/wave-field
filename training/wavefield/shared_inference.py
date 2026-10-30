"""One batched inference process shared by independent CPU MCTS workers."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import multiprocessing as mp
from queue import Empty
import time
import traceback
from typing import Any

import numpy as np
import torch


@dataclass
class InferenceSpec:
    requests: Any
    responses: list[Any]
    next_client: Any
    timeout_seconds: float = 120.0


class SharedInferenceClient:
    def __init__(self, spec: InferenceSpec):
        self.spec = spec
        with spec.next_client.get_lock():
            self.index = spec.next_client.value
            spec.next_client.value += 1
        if self.index >= len(spec.responses):
            raise RuntimeError('Too many clients for shared inference service')
        self.calls = 0

    def eval(self):
        return self

    def search_outputs(self, board, side, history_board=None, history_side=None):
        if board.shape[0] != 1:
            raise ValueError("Each MCTS inference client must send exactly one position")
        arrays = tuple(x.detach().cpu().numpy() if x is not None else None
                       for x in (board, side, history_board, history_side))
        self.spec.requests.put((self.index, arrays))
        try:
            ok, result = self.spec.responses[self.index].get(timeout=self.spec.timeout_seconds)
        except Empty as error:
            raise RuntimeError('Shared inference service timed out') from error
        if not ok:
            raise RuntimeError(f'Shared inference failed: {result}')
        self.calls += 1
        return tuple(torch.from_numpy(array) for array in result)


def _serve(model, device, spec, ready, stats_queue, max_batch, wait_seconds):
    stats = dict(device=device, batches=0, positions=0, forward_seconds=0., batch_sizes={})
    try:
        torch.set_num_threads(1)
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS GPU is unavailable')
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU is unavailable')
        model = model.to(device).eval()
        def forward(arrays):
            tensors = [torch.from_numpy(x).to(device, non_blocking=True) if x is not None else None for x in arrays]
            with torch.inference_mode():
                outputs = model.search_outputs(tensors[0], tensors[1], history_board=tensors[2], history_side=tensors[3])
            # One download/synchronization for all heads, rather than four tiny
            # GPU-to-CPU copies per inference batch.
            widths = [x.shape[1] if x.ndim == 2 else 1 for x in outputs]
            packed = torch.cat([x if x.ndim == 2 else x.unsqueeze(1) for x in outputs], dim=1).cpu().numpy()
            result = []
            offset = 0
            for output, width in zip(outputs, widths):
                array = packed[:, offset:offset + width]
                result.append(array if output.ndim == 2 else array[:, 0])
                offset += width
            return tuple(result)
        for size in sorted({1, max_batch}):
            forward((np.zeros((size, model.board_channels, 7, 7), np.float32),
                     np.zeros((size, model.side_size), np.float32), None, None))
        ready.put((True, None))
        stop = False
        while not stop:
            first = spec.requests.get()
            if first is None:
                break
            batch = [first]
            deadline = time.perf_counter() + wait_seconds
            while len(batch) < max_batch:
                try:
                    remaining = deadline - time.perf_counter()
                    request = spec.requests.get(timeout=max(0, remaining))
                except Empty:
                    break
                if request is None:
                    stop = True
                    break
                batch.append(request)
            inputs = tuple(np.concatenate([item[1][i] for item in batch], axis=0)
                           if batch[0][1][i] is not None else None for i in range(4))
            started = time.perf_counter()
            outputs = forward(inputs)
            stats['forward_seconds'] += time.perf_counter() - started
            stats['batches'] += 1
            stats['positions'] += len(batch)
            sizes = stats['batch_sizes']; sizes[str(len(batch))] = sizes.get(str(len(batch)), 0) + 1
            for row, (client, _) in enumerate(batch):
                # Each client sends one position and waits for its own response.
                spec.responses[client].put((True, tuple(x[row:row + 1].copy() for x in outputs)))
    except BaseException:
        error = traceback.format_exc()
        ready.put((False, error))
        for response in spec.responses:
            response.put((False, error))
        stats['error'] = error
    finally:
        stats_queue.put(stats)


class SharedInferencePool:
    def __init__(self, model, workers, device='mps', max_batch=32, wait_ms=2.):
        if workers < 1 or max_batch < 1 or wait_ms < 0:
            raise ValueError('Workers/batch must be positive and wait time nonnegative')
        self.context = mp.get_context('spawn')
        self.spec = InferenceSpec(self.context.Queue(), [self.context.Queue() for _ in range(workers)],
                                  self.context.Value('i', 0))
        self.ready = self.context.Queue()
        self.stats_queue = self.context.Queue()
        self.model = copy.deepcopy(model).cpu().eval()
        self.device, self.max_batch, self.wait_ms = device, min(max_batch, workers), wait_ms
        self.process = None
        self.stats = {}

    def __enter__(self):
        self.process = self.context.Process(target=_serve,
            args=(self.model, self.device, self.spec, self.ready, self.stats_queue,
                  self.max_batch, self.wait_ms / 1000), name='shared-mcts-inference')
        self.process.start()
        try:
            ok, error = self.ready.get(timeout=120)
            if not ok:
                raise RuntimeError(error)
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.spec.requests.put(None)
            self.process.join(timeout=15)
            if self.process.is_alive():
                self.process.terminate(); self.process.join(timeout=5)
            try:
                self.stats = self.stats_queue.get(timeout=2)
            except Empty:
                self.stats = {'error': 'Inference service exited without statistics'}
            self.process = None
        for queue in [self.spec.requests, *self.spec.responses, self.ready, self.stats_queue]:
            queue.cancel_join_thread()
            queue.close()

    def __exit__(self, *args):
        self.close()
