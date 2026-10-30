"""Validate a recovery checkpoint and emit its small metadata summary."""
import json
from pathlib import Path
import sys
import torch

checkpoint = torch.load(Path(sys.argv[1]), map_location='cpu', weights_only=False)
if not all(key in checkpoint for key in ('model', 'optimizer', 'iteration', 'optimizer_steps')):
    raise ValueError('Checkpoint is missing required training state')
print(json.dumps({key: checkpoint[key] for key in ('iteration', 'optimizer_steps')}))
