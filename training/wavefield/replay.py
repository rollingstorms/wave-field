from __future__ import annotations

from collections import deque
from typing import Any, Iterable

import numpy as np

from .selfplay import Sample


class ReplayBuffer:
    """FIFO experience window, sampled without replacement for each update."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.samples: deque[Sample] = deque(maxlen=capacity)
        self.total_seen = 0

    def __len__(self) -> int:
        return len(self.samples)

    def extend(self, samples: Iterable[Sample]) -> None:
        for sample in samples:
            self.samples.append(sample)
            self.total_seen += 1

    def draw(self, count: int, seed: int) -> list[Sample]:
        if count < 1:
            raise ValueError("Replay draw size must be positive")
        items = list(self.samples)
        indexes = np.random.default_rng(seed).choice(len(items), size=min(count, len(items)), replace=False)
        return [items[int(index)] for index in indexes]

    def state_dict(self) -> dict[str, Any]:
        return {"samples": list(self.samples), "total_seen": self.total_seen}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.samples.clear()
        self.samples.extend(state["samples"])
        self.total_seen = int(state["total_seen"])
