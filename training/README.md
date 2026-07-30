# Wave Field Training

This is the first Python training scaffold for Wave Field. It uses the native
Rust engine CLI as the rule source and keeps model code separate from engine
code so we can swap input views later.

## Setup

```bash
python3 -m pip install -r training/requirements.txt
cargo build --manifest-path engine/Cargo.toml --release
```

## Smoke Checks

```bash
PYTHONPATH=training python3 -m unittest discover training/tests
```

## First Training Run

```bash
PYTHONPATH=training python3 -m wavefield.train --games 8 --epochs 3 --batch-size 64
```

Random-policy training uses the Rust batch generator by default. Add
`--python-selfplay` when you need to debug the slower Python rollout path.

The MVP model uses a hybrid view:

- board planes: occupancy, instability, field sign/magnitude, current player
- side vector: tuning coefficients and turn number
- action mask: fixed piece slot by destination square

The trainer can generate either random self-play samples or samples from the
current model:

```bash
PYTHONPATH=training python3 -m wavefield.train --policy model --resume --games 16 --iterations 4
```

This is still a pipeline MVP, not a strong bot. Capped games use a material
target by default so non-decisive runs do not all collapse to value `0.0`.
Use `--cap-value zero` to disable that.

Model-policy training uses batched rollouts by default: multiple active game
states are encoded together and passed through one PyTorch forward call per ply.
Tune that with `--rollout-batch-size`.

## Evaluation

```bash
PYTHONPATH=training python3 -m wavefield.eval --policy random --games 25 --max-plies 80
PYTHONPATH=training python3 -m wavefield.eval --policy model --checkpoint training/checkpoints/policy_value.pt --games 25
PYTHONPATH=training python3 -m wavefield.eval --policy heuristic --games 25
```

The evaluator reports winner counts, mean plies, capped games, first-loss
outcomes, piece-loss frequency, rescue rate, pressure, underdog wins, and wins
by final piece count.

Model eval can use the Rust session rollout path for larger runs:

```bash
PYTHONPATH=training python3 -m wavefield.eval \
  --policy model --session --games 200 --max-plies 120 \
  --checkpoint training/checkpoints/policy_value.pt --json
```

Session eval collects losses, rescues, final pieces, and exact pressure by
default. Add `--no-pressure` when you need a cheaper bulk check and can accept
zeroed pressure fields.

## Logged Experiments

Use `wavefield.experiment` when you want a repeatable training run with JSONL
events, checkpointing, sample metadata summaries, and periodic session eval:

```bash
PYTHONPATH=training python3 -m wavefield.experiment \
  --run-dir training/runs/dev-smoke \
  --pretrain-random-games 100 \
  --model-games 100 \
  --iterations 3 \
  --epochs 3 \
  --eval-every 1 \
  --eval-games 50
```

Each run writes `events.jsonl` and `checkpoint.pt` under `--run-dir`. Generated
samples include metadata such as source, phase, ply, and legal action count
when that information is available.

The experiment runner can also mix fresh Rust random games into every
model-self-play iteration and replay-weight parts of the batch. This is the
current curriculum/augmentation MVP: it does not transform boards, but it can
increase exposure to specific data sources or phases while preserving legal
positions exactly as produced by the engine.

```bash
PYTHONPATH=training python3 -m wavefield.experiment \
  --run-dir training/runs/curriculum-smoke \
  --pretrain-random-games 100 \
  --random-games-per-iteration 50 \
  --model-games 100 \
  --scenario-games-per-iteration 20 \
  --iterations 5 \
  --phase-weights opening=1,midgame=1,endgame=2 \
  --source-weights rust_random=1,rust_session_model=2
```

Session model samples include material metadata when available: current-player
material balance, total pieces, low-material tags, and piece-type counts.
Experiment logs report both raw sample counts and replay-weighted training
sample counts so we can see what the model actually trained on.

Scenario games start from named non-initial states and join the same training
batch. The first scenario set is `opening,midgame,low_material,rescue`.
Use `--scenario-eval-games` to run a matching targeted eval suite after normal
eval checkpoints.

## Head-to-Head Matches

```bash
PYTHONPATH=training python3 -m wavefield.match --red model --blue heuristic --games 25 --max-plies 150
PYTHONPATH=training python3 -m wavefield.match --red heuristic --blue model --games 25 --max-plies 150
```

Use side-swapped matches when comparing policies so first-player and color bias
do not get mistaken for model strength.

## Benchmark

```bash
PYTHONPATH=training python3 -m wavefield.bench --games 25 --max-plies 80
```

The benchmark compares the pure Rust random batch path against the Python loop
that encodes states for training. `rust_random_training_batch` is the current
fast path for random training data: Rust rolls out games, encodes board tensors,
emits legal action indexes, and returns the batch to Python for PyTorch updates.
Model-driven self-play still runs through Python because PyTorch selects moves
at each ply, but inference is batched across active games.
