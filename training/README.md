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
PYTHONPATH=training python3 -m wavefield.train --games 8 --epochs 3
```

The MVP model uses a hybrid view:

- board planes: occupancy, instability, field sign/magnitude, current player
- side vector: tuning coefficients and turn number
- action mask: fixed piece slot by destination square

The training loop currently learns from random legal self-play samples. It is
intended as a pipeline smoke test, not a strong bot.
