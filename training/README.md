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
by final piece count. It also reports win rates, decisive/capped rates, ply
distribution, and average final material balance.

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

To continue an experiment from a checkpoint, use `--resume-checkpoint`. The
runner restores both model and optimizer state and continues checkpoint
iteration numbers from the saved checkpoint:

```bash
PYTHONPATH=training python3 -m wavefield.experiment \
  --run-dir training/runs/transformer-128-full-policy-next80 \
  --resume-checkpoint training/runs/transformer-128-full-policy-long/checkpoint.pt \
  --hidden-size 128 \
  --model-arch transformer \
  --input-view piece_identity \
  --model-games 100 \
  --scenario-games-per-iteration 25 \
  --iterations 80 \
  --epochs 5 \
  --batch-size 128 \
  --rollout-batch-size 128 \
  --temperature 0.9 \
  --full-policy \
  --max-tuning-actions 3 \
  --eval-every 10 \
  --eval-games 25 \
  --eval-max-plies 150 \
  --baseline-eval-games 10 \
  --baseline-eval-max-plies 300 \
  --baseline-opponents heuristic \
  --progress
```

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

For input/model experiments, the default remains `--input-view base`
`--model-arch conv`. The first richer view is `--input-view piece_identity`,
which appends one board plane per piece slot. Pair it with
`--model-arch residual` for the small residual CNN baseline, or
`--model-arch transformer` for the compact board-token transformer. Rust random
training batches currently emit only the base view, so rich-view runs should use
model/session and scenario-session data until the Rust encoder grows that mode.

For sequence modeling, use `--model-arch sequence_transformer` with a history
window. This encodes each position with the board-token state encoder, then runs
temporal attention over the last N state embeddings before predicting the next
tune/move action. The first supported path is Python full-policy rollout:

```bash
PYTHONPATH=training python3 -m wavefield.experiment \
  --run-dir training/runs/sequence-transformer-16 \
  --hidden-size 256 \
  --model-arch sequence_transformer \
  --history-plies 16 \
  --input-view piece_identity \
  --model-games 100 \
  --scenario-games-per-iteration 25 \
  --iterations 300 \
  --epochs 10 \
  --batch-size 128 \
  --rollout-batch-size 128 \
  --temperature 0.9 \
  --full-policy \
  --max-tuning-actions 3 \
  --eval-every 10 \
  --eval-games 25 \
  --eval-max-plies 150 \
  --baseline-eval-games 10 \
  --baseline-eval-max-plies 300 \
  --baseline-opponents heuristic \
  --progress
```

The Rust session rollout path is still state-only; sequence experiments
intentionally require `--full-policy` so the generated samples carry
`history_board` and `history_side` windows. The local model server uses browser
history snapshots when serving sequence checkpoints.

## Head-to-Head Matches

```bash
PYTHONPATH=training python3 -m wavefield.match --red model --blue heuristic --games 25 --max-plies 150
PYTHONPATH=training python3 -m wavefield.match --red heuristic --blue model --games 25 --max-plies 150
```

Use side-swapped matches when comparing policies so first-player and color bias
do not get mistaken for model strength.

For a longer checkpoint check, run both colors with a larger cap and JSON output:

```bash
PYTHONPATH=training python3 -m wavefield.match \
  --checkpoint training/runs/transformer-rich-20m/checkpoint-iter-52.pt \
  --model-arch transformer --input-view piece_identity \
  --red model --blue heuristic --games 100 --max-plies 320 --temperature 0 --json

PYTHONPATH=training python3 -m wavefield.match \
  --checkpoint training/runs/transformer-rich-20m/checkpoint-iter-52.pt \
  --model-arch transformer --input-view piece_identity \
  --red heuristic --blue model --games 100 --max-plies 320 --temperature 0 --json
```

Logged experiments can evaluate on a longer horizon than training and add
side-swapped baseline matches at each eval checkpoint. Add these flags to the
training command for a new run:

```bash
--eval-games 100 --eval-max-plies 320 \
  --baseline-eval-games 50 --baseline-eval-max-plies 320 \
  --baseline-opponents heuristic,random
```

## Local Neural Arena

The browser arena can call a local Python model server in dev. This is only for
local experiments; it is not part of the GitHub Pages build because `.pt`
checkpoints and PyTorch run outside the browser.

Terminal 1:

```bash
PYTHONPATH=training python3 -m wavefield.serve_model \
  --checkpoint training/runs/residual-rich-20m/checkpoint.pt \
  --port 8765
```

Optional transformer server:

```bash
PYTHONPATH=training python3 -m wavefield.serve_model \
  --checkpoint training/runs/transformer-rich-20m/checkpoint.pt \
  --port 8766
```

Terminal 2:

```bash
npm run arena
```

Open `/local-arena`, then choose `Human`, `Heuristic`, `Neural residual`, or
`Neural transformer` separately for Blue and Red. The public `/arena` route
stays static-safe for GitHub Pages and only exposes browser-hosted policies.
Human-vs-AI advances automatically when the AI side is to move. AI-vs-AI uses
the arena Run/Pause/Step controls.

The local server exposes a full-turn policy path: a model kind head chooses
whether to tune or move, a tuning head chooses component edits, and the existing
move head chooses the final move. Older movement-only checkpoints can still
load, but their tuning heads start untrained; train fresh full-policy
checkpoints before reading tuning behavior as learned strategy.

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

## Tactical Eval

Use tactical evals to measure whether a checkpoint ranks heuristic tactical
targets highly without paying for long full-game evals:

```bash
PYTHONPATH=training python3 -m wavefield.tactical_eval \
  --checkpoint training/runs/transformer-256-tactical-long/checkpoint.pt \
  --model-arch transformer \
  --input-view piece_identity \
  --games 64 \
  --scenarios opening,midgame,low_material,rescue
```

Use policy inspection when you want concrete examples of what the model is
considering:

```bash
PYTHONPATH=training python3 -m wavefield.policy_inspect \
  --checkpoint training/runs/transformer-256-tactical-long/checkpoint.pt \
  --model-arch transformer \
  --input-view piece_identity \
  --positions 8 \
  --top-k 5
```

Long pretrain template:

```bash
PYTHONPATH=training python3 -m wavefield.experiment \
  --run-dir training/runs/transformer-256-tactical-long \
  --hidden-size 256 \
  --model-arch transformer \
  --input-view piece_identity \
  --heuristic-bootstrap-games 200 \
  --heuristic-bootstrap-per-iteration 32 \
  --model-games 160 \
  --scenario-games-per-iteration 48 \
  --scenario-bootstrap-per-iteration 24 \
  --iterations 240 \
  --epochs 6 \
  --batch-size 128 \
  --rollout-batch-size 128 \
  --temperature 0.9 \
  --kind-temperature 1.1 \
  --tuning-temperature 1.4 \
  --force-first-tune-prob 0.25 \
  --full-policy \
  --max-tuning-actions 3 \
  --eval-every 20 \
  --eval-games 25 \
  --eval-max-plies 180 \
  --scenario-eval-games 24 \
  --tactical-eval-games 64 \
  --baseline-eval-games 10 \
  --baseline-eval-max-plies 240 \
  --baseline-opponents heuristic \
  --source-weights heuristic_bootstrap=2,python_model_full_policy=1 \
  --phase-weights endgame=2,midgame=1,opening=1 \
  --progress
```

## Model Analysis

Use `wavefield.analyze_model` to inspect a checkpoint without running an
expensive eval. It samples reachable states from the Rust rules engine, captures
activations, reports centered effective rank/isotropy metrics, summarizes legal
policy concentration, and trains cheap linear probes against rule and strategy
labels.

```bash
PYTHONPATH=training python3 -m wavefield.analyze_model \
  --checkpoint training/runs/transformer-128-full-policy-long/checkpoint.pt \
  --samples 256 \
  --max-plies 150 \
  --probe-epochs 120 \
  --json-out training/runs/transformer-128-full-policy-long/analysis.json
```

Useful probe labels include current/opponent king pressure, legal move count,
material balance, low-material phase, and opening/midgame/endgame phase. Read
probe scores relative to their majority or variance baseline: strong probe
scores mean the representation contains the signal, not that the policy is
using it well.
