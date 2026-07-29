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

## Evaluation

```bash
PYTHONPATH=training python3 -m wavefield.eval --policy random --games 25 --max-plies 80
PYTHONPATH=training python3 -m wavefield.eval --policy model --checkpoint training/checkpoints/policy_value.pt --games 25
PYTHONPATH=training python3 -m wavefield.eval --policy heuristic --games 25
```

The evaluator reports winner counts, mean plies, capped games, first-loss
outcomes, piece-loss frequency, rescue rate, pressure, underdog wins, and wins
by final piece count.

## Benchmark

```bash
PYTHONPATH=training python3 -m wavefield.bench --games 25 --max-plies 80
```

The benchmark compares the pure Rust random batch path against the Python loop
that encodes states for training. `rust_random_training_batch` is the current
fast path for random training data: Rust rolls out games, encodes board tensors,
emits legal action indexes, and returns the batch to Python for PyTorch updates.
Model-driven self-play still runs through Python because PyTorch selects moves
at each ply.
