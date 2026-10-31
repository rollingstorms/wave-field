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

## MCTS self-play

### Persistent macOS training service


The spectator dashboard includes a scoreboard backed by the read-only
`/__selfplay/stats` endpoint. Results summarize completed MCTS generation batches
under the current repetition setting, with a separate last-20-batches window.
Red/Blue wins, repetition/capped draws, average plies, generation pace and tuning
frequency come from the same `generate` event summaries used by analysis.
Latest training/validation losses and frozen-checkpoint evaluation are shown
separately from that results window. Recovered generation iterations replace
earlier entries instead of being counted twice. Event reading is incremental and
statistics are cached for five seconds; the viewer refreshes them every ten.

The compact progress banner also shows elapsed time for the current service
session (through its latest heartbeat, minus recorded inactivity gaps) and an
estimated remaining awake compute time. ETA uses up to twenty completed batches
under the current repetition rule and effective batch size, requiring at least
five observations. It includes recorded generation, neural training/validation,
and evaluation durations; checkpoint disk overhead is not included. The estimate
adapts as game lengths change and is not a calendar completion date.

The tactical section uses newly collected MCTS turn measurements; it displays
its game coverage and starting iteration instead of assigning zeros to older
games. A rescue opportunity is a turn beginning with at least one unstable own
piece. It succeeds only when every such piece survives and is stable after that
turn, matching the existing analysis definition. Aggregate rescue rates divide
total rescues by total opportunities, with per-player counts shown alongside.
Other metrics are losses per game by owner (including removals on either side's
turn), turns with any tuning, net changed components divided by tuning actions,
mean legal moves per decision position, and mean plies in decisive games.
Measurement adds no engine searches and does not change rewards or rules.

Validation reserves one complete MCTS game per eight-game batch before inserting
the other seven games into training replay. The selected game slot rotates each
iteration. A separate rolling buffer retains up to 2,048 validation positions and
is saved/restored with recovery checkpoints. These game samples never enter the
training replay. Identical opening positions can naturally recur across different
games; this measures generalization to held-out game trajectories, not unseen
board configurations. The validation window changes as self-play evolves.

After each epoch MLflow logs `validation_loss/total`, `action_kind`, `move_policy`,
`tuning_policy`, and `value`, plus `validation/positions` and held-out buffer
counts. Evaluation uses no gradients and preserves the model's prior mode.
Configure with `--validation-games-per-batch 1 --validation-capacity 2048`.
The older `--validation-fraction` splits sampled rows and is unsuitable for
holding out replay games across iterations; leave it at zero for this service.

The active service uses `--mcts-repetition-limit 3`: the third identical
turn-boundary position ends an MCTS self-play game with a neutral value target
(0 on the -1 to +1 scale). Pieces, player to move, tuning, activation orders and
other rule settings must match; UI state and turn counters are ignored. Tuning
actions within a turn do not count as occurrences. This is a self-play cutoff;
search simulations do not track repetition history. The 600-ply cap still applies.
MLflow records `self_play/repetition_draw_rate` and
`self_play/plies_avoided_by_repetition`; the latter measures remaining plies to
the cap, rather than a measured counterfactual game length. The live viewer
labels completed repetition draws explicitly. Disable with
`--mcts-repetition-limit 0`.

The installed user LaunchAgent is `com.wavefield.training`. It supervises training,
MLflow, the log importer, and a built (non-development) self-play dashboard.
It starts when this user logs in after a reboot, and restarts failed components.
It does not run before login or bypass FileVault. It does not prevent Mac sleep:
processes suspend during sleep and continue on wake; the dashboards are unreachable
while the Mac is asleep. Locking the screen alone does not stop training.

The service runtime, environments, data, and logs live in
`~/Library/Application Support/WaveField/`, since launchd jobs cannot read this
Mac's protected Documents folder. The repository's `training/runs`,
`training/tracking`, and tracking environment remain accessible through local
links. The Rust engine, initial state, Python sources, and built viewer are copied
into the service runtime; repository edits require redeployment.

```bash
python3 training/service_control.py status
python3 training/service_control.py restart
python3 training/service_control.py stop
python3 training/service_control.py start

# Redeploy sources and rebuild the viewer, preserving data and the update target:
python3 training/install_service.py
```

`stop` stops the managed services until restarted or the next login. During the
one-time legacy handoff, it suspends the old trainer in memory rather than losing
its unsaved updates; starting the service resumes it. Once the managed trainer is
active, restarts recover from its latest complete checkpoint and may repeat the
unfinished self-play batch or unsaved optimizer updates.

Current configuration:

- Eight independent CPU inference workers, eight self-play games per batch.
- 128 MCTS simulations, all legal candidates, depth 32, 600 plies, 20 tunes/turn.
- Outcome discount 0.997 per piece move, with no added repetition rule.
- Effective optimizer batch **4,096**, accumulated through **128-position**
  microbatches. Per-head gradient weights match a real full batch even when the
  proportions of move/tuning examples differ across microbatches.
- Three epochs over 4,096 sampled replay positions: **three optimizer updates**
  per generation cycle. This does not increase the number of generated games.
- Target: **100,000 new large-batch updates**, with the prior small-batch history
  retained separately. The supervisor persists the phase's starting lifetime step.
  At current generation rates this is a months-long target, not the earlier
  small-batch estimate of days.
- Atomic, flushed recovery checkpoint after every completed generation/training
  cycle. Keep `checkpoint-latest.pt` and `checkpoint-previous.pt`; preserve model,
  optimizer, replay, RNG state, and frozen evaluation opponent.
- Milestone snapshots every 1,000 large-batch updates, retaining the latest three.
  Existing historical checkpoints are preserved. Milestones use hard links to the
  current snapshot initially. Logs rotate; low disk space pauses training below
  1 GB free and resumes above 1.5 GB.

The original six-worker run is adopted until checkpoint 40, then the service
switches settings without losing completed training. If that old process dies
beforehand, the service recovers its most recent saved checkpoint and starts the
new phase. Until handoff, the old checkpoint schedule still applies.

The live viewer shows active versus pending settings, large-batch and lifetime
updates, durable checkpoint iteration, service starts/reboot counts, and inactivity
resumes. Network requests time out and retry while the Mac is asleep. Heartbeat
gaps are recorded as inactivity, without assuming every gap was sleep. MLflow logs
`training/large_batch_optimizer_steps`, `settings/effective_batch_size`, and
`service/*` metrics, plus start/resume tags. The dashboards remain available after
the target is reached. Increase `target_large_batch_updates` in
`service-config.json` and restart the service to extend the target.

Use a stable Bonjour hostname when possible:

- `http://michaels-macbook-pro-2.local:5174/?watch=1` — self-play and service status.
- `http://michaels-macbook-pro-2.local:5050` — MLflow.

The current LAN IP also works. The supervisor refreshes MLflow's allowed origins
if its LAN address changes. No cloud resources are provisioned.

Recovery verification includes a launchd-managed isolated trainer killed after
saving a checkpoint, recovery to a replacement PID with the same saved update
count, a supervisor restart preserving the legacy trainer, and a simulated
inactivity gap. A real reboot or actual system sleep is not induced by setup.

### Color and checkpoint audit

`PYTHONPATH=training WAVEFIELD_SPECTATOR_DISABLE=1 python -m wavefield.color_audit
--run-dir training/runs/selfplay-replay-discrete-20261028 --out /tmp/color-audit`
freezes model-only copies of the latest checkpoint and the retained iteration-40
and update-1002 anchors. It saves sampled positions, engine transformation checks,
network symmetry measurements, and individual match results. The default matrix
is 32 policy-only and 16 MCTS games per anchor, balancing model color and starting
color independently with shared seed blocks. Policy-only temperature is 0.3;
MCTS uses 128 simulations, all legal actions, depth 32, no root noise, visit
sampling for the first 20 plies, up to 20 tunes per turn, threefold draws and a
600-ply cap. Two lower-priority CPU workers run separately from training. The
rotated color swap must preserve engine legality, field values, and sampled
transitions before matches proceed. These small paired tests are exploratory;
draw-heavy scores do not establish playing strength.

### Shared GPU inference for MCTS

`wavefield.experiment` supports one inference process shared by the CPU search
workers. The Rust rules engine and MCTS trees stay on CPU; neural requests are
batched onto `mps` (Apple GPU), `cuda` (NVIDIA), or a shared `cpu` control:

```bash
# Add to an experiment command:
--mcts-workers 8 --mcts-inference-device mps \
  --mcts-inference-max-batch 8 --mcts-inference-wait-ms 2
```

The default `--mcts-inference-device local` preserves independent CPU inference.
The training model stays on CPU for this parallel path. At each batch, a snapshot
of its weights is copied into the inference process, which returns raw head
outputs to workers. Each worker has one outstanding request. The server flushes
partial batches after the wait interval so stragglers and single active games
continue making progress. All heads are downloaded in one GPU transfer.
Failures are reported; the backend does not silently fall back to CPU. Generated
`generate` events include inference batch sizes, position counts, and forward
time. Inference batching is independent of optimizer minibatch size.

Benchmark fixed positions from the spectator traces using the same checkpoint,
128 MCTS simulations, all legal candidates, a 32-decision depth limit, and no
root noise:

```bash
PYTHONPATH=training python training/benchmark_shared_inference.py \
  --checkpoint training/runs/RUN/checkpoint-iter-30.pt \
  --workers 8 --searches-per-worker 4 --repeats 3 \
  --output training/tracking/shared-inference-benchmark.json
```

The benchmark checks CPU/GPU head outputs, alternates backend order, excludes
worker startup from steady search timing, and reports startup-inclusive time
separately. Pause competing training for a controlled run. The saved report
contains the exact positions, configuration, actions, timings, and batch sizes.
Apple MPS is not guaranteed to beat eight CPU workers for this small network;
measure before enabling it for long training runs. Current training remains on
local CPU inference unless explicitly changed.

### Outcome discount for MCTS

`--mcts-value-discount 0.997` discounts an outcome by `0.997 ** remaining_plies`.
The default is `1.0` (undiscounted). Each piece move incurs one discount factor;
tuning incurs none. Both decisive results and position-based cap values use
the same discount. Losses remain negative, so delaying defeat is preferable
to losing immediately. No repetition rule or repetition penalty is added.

MCTS applies the factor on move edges during backup and child selection.
Generated samples retain the undiscounted outcome, remaining piece moves, and
discount in metadata. Existing replay samples retain their original targets
and age out through the FIFO buffer.

For an already-running parallel run, the optional local file
`training/tracking/mcts-runtime.json` can override the discount at game start:

```json
{
  "run_dir": "/absolute/path/to/the/training/run",
  "seed_offset": 150210,
  "start_iteration": 33,
  "final_iteration": 100,
  "value_discount": 0.997
}
```

`seed_offset` is the training seed plus 60000. Set `start_iteration` beyond
all games already launched so one complete batch uses the same discount.
The override takes precedence over the CLI within that seed range. Workers
write `selfplay-settings.json` in the run directory when they adopt it.
MLflow reads this receipt into `settings/mcts_value_discount` and a current
setting tag; the spectator feed reports it per game. Spawned workers pick up
code/settings at the next batch without restarting the training parent.
Retain the runtime configuration when resuming this active run, or explicitly
supply the same CLI setting after removing the override.

### Watch actual self-play games

Run `npm run watch:selfplay` from the repository root and open
`http://<LAN-IP>:5174/?watch=1`. The viewer shows committed training actions,
including tuning, with a game selector, replay slider, playback speeds, and
**Follow live**. Polling continues while replay is paused.

Enable the optional worker feed by creating `training/tracking/spectator-config.json`:

```json
{"directory": "/absolute/path/to/wave-field/training/tracking/selfplay-live"}
```

Parallel MCTS workers read this configuration when each game starts. A running
training process picks up the recorder when its next batch spawns workers;
there is no training restart. The viewer waits until those workers publish.
Each game slot keeps its latest game, replacing it at the next batch, so trace
storage stays bounded. Recorded states omit undo history and are written only
for chosen actions, never MCTS search branches. Completed capped games are
labelled as capped. Playback steps include tunes; plies count piece moves.
Game limits come from worker metadata (currently 600 plies / 20 tunes).
Optionally set `seed_offset` to the training seed plus 60000 in the feed
configuration to label games with their training iteration.

Remove the configuration to stop recording at the next game, or set
`WAVEFIELD_SPECTATOR_DISABLE=1` before launching training/tests. The feed and
Vite server stay separate from MLflow and are available only while running
locally. The feed endpoint is provided by the development server; it is not
included in the published static site. Optional feed I/O errors disable the
feed for that game without stopping training.

### Live charts with MLflow

The tracker imports `events.jsonl` from existing runs and watches for new events.
It runs separately from training, so enabling charts needs no training restart.
Install it in its own environment:

```bash
uv venv training/.venv-tracking --python python3.12
uv pip install --python training/.venv-tracking/bin/python -r training/requirements-tracking.txt
mkdir -p training/tracking
```

Start the local server from the repository root:

```bash
training/.venv-tracking/bin/mlflow server \
  --host 127.0.0.1 --port 5050 --workers 1 \
  --backend-store-uri sqlite:///training/tracking/mlflow.db \
  --artifacts-destination training/tracking/artifacts
```

In another terminal, start the watcher (multiple run directories may be supplied):

```bash
training/.venv-tracking/bin/python training/track_runs.py \
  training/runs/selfplay-replay-discrete-20261028 --watch
```

Open <http://127.0.0.1:5050>, choose **Wave Field Self-play**, open a run, and
select **Model metrics**. The watcher checks for new log entries every five
seconds; MLflow's **Auto-refresh** updates the charts. Useful metric groups:

- `loss/`: total, value, move policy, tuning policy, and action-kind losses.
- `self_play/`: generation time, positions per second, game length, capped rate,
  and tuning frequency.
- `replay/` and `training/`: retained positions, sampled positions, and optimizer updates.
- `evaluation/`: results against earlier self-play checkpoints (or historical baselines).
- `settings/`: changes to game, tuning, and search limits across resumes.

Use the **Step** chart axis: steps count optimizer updates, not MCTS simulations
or game plies. Historical timestamps are reconstructed from logged durations;
new entries use observed arrival time. `initial.*` parameters describe the first
configuration; current settings also appear in tags, settings charts, and
versioned configuration artifacts. Large model/replay checkpoint files remain
in the original run directory. The watcher persists its log cursors in
`training/tracking/sync-state.json`, so it can be restarted without reimporting
previous events.

Loss measures how well the network fits its targets. Use checkpoint match results
and capped-game rates alongside loss when judging playing strength.

### Training configuration

Training and evaluation start from `engine/tests/product-initial-state.json`,
which explicitly selects the current discrete (`sparse`) rules. New experiments
log the rules variant and starting-position path, and checkpoints record the
variant. Classic-rule checkpoints were trained on a different game; start a
fresh experiment when training the discrete rules.

MCTS self-play is opt-in with `--mcts-games-per-iteration`. Its default search
budget is 128 simulations per action, with root exploration noise during
training. The first 20 plies sample actions from MCTS visit counts. Use
`--mcts-simulations`, `--mcts-dirichlet-alpha`,
`--mcts-exploration-fraction`, and `--mcts-sample-first-plies` to tune the
search. `--mcts-workers N` generates up to N independent games concurrently
on CPU and combines their samples before each model update. Each worker owns a
model copy and Rust engine; one worker preserves serial generation. Head-to-head `model_search`
matches also default to 128 simulations, without root noise.

Search discards UI undo history before crossing the Python/Rust boundary; encoded
history for sequence models is maintained separately. By default, every legal
move and tuning action is a search candidate (`--mcts-top-k 0`). A positive
candidate limit enables policy pruning. Cached leaf evaluations are reused when
search reaches its depth limit.

For a from-scratch run with no teacher games or pretrained encoder, use MCTS
self-play as the only training source. `--cap-value position` gives an unfinished
game a fractional result based on the final piece count and proximity to the
opposing king. Wins and losses still use the actual result:

```bash
PYTHONPATH=training python3 -m wavefield.experiment \
  --run-dir training/runs/pure-mcts-replay \
  --device cpu --model-arch transformer --hidden-size 256 --input-view piece_identity \
  --full-policy --model-games 0 --mcts-games-per-iteration 8 --mcts-workers 6 \
  --mcts-simulations 128 --mcts-top-k 0 --mcts-depth 32 \
  --max-plies 600 --max-tuning-actions 20 --cap-value position \
  --iterations 100 --epochs 3 --batch-size 128 --lr 0.0003 \
  --replay-capacity 20000 --replay-sample-size 4096 \
  --eval-every 10 --skip-self-eval --checkpoint-eval-games 8 \
  --baseline-eval-games 0 --baseline-opponents "" --save-every 10
```

This starts from random network weights. The search and final game outcomes
provide the policy and value targets. At the ply limit, the signed value is
`piece_count_difference / 6 + 0.25 * proximity_difference`, clipped to [-1, 1].
Proximity is the mean normalized closeness of surviving non-king pieces to the
opposing king. The corresponding game points are `0.5 + signed_value / 2` for
one player and the complement for the other. This is a declared training score
for unfinished positions, not an observed win probability or a rules-defined
draw. Use `--cap-value zero` for neutral outcomes at the length limit.
`--max-tuning-actions 20` permits up to 20 tuning actions before a move, in both
MCTS and network evaluation. A ply counts a player turn ending with a move;
tuning actions within that turn do not consume plies. The tuning counter resets
each turn. `-1` removes this limit, allowing tuning cycles to keep a turn open
indefinitely.

With replay enabled, each eight-game batch enters a FIFO window of up to 20,000
positions. Each iteration samples up to 4,096 retained positions without
replacement, then performs three shuffled passes in minibatches of 128. That
means up to 96 optimizer steps per iteration; the initial iterations use fewer
steps while the buffer fills. Logs distinguish individual `selfplay_game`
completions, search simulations, replay size, and optimizer steps. Training
computes policy and value losses with one shared network forward pass.

Checkpoint evaluation uses the current network against a frozen earlier version
of itself, with both colors and a small sampling temperature for varied games.
It evaluates the networks without MCTS and logs `checkpoint_eval`; the first
opponent is the initial network, and subsequent opponents are the previous
evaluation snapshot. These matches do not feed the training buffer. The command
above uses no random or heuristic opponents, teacher data, or bootstrap phase.

Saved checkpoints include replay contents, optimizer state, and the optimizer
step count. Resume with `--resume-checkpoint` and matching replay options; rule
variant mismatches are rejected. Checkpoints are written to a temporary file
before replacing the saved file.

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

### Paired color augmentation

`--color-augmentation` trains on original positions and 180° rotated, color-swapped
copies. Piece identities, board channels, component ownership, move masks, hard
actions, MCTS move distributions, and history frames are mapped together. Values
remain player-to-move relative; tuning and action-kind targets are unchanged.

The 4,096-row replay draw becomes 2,048 originals plus 2,048 mirrors, preserving
the effective batch and optimizer update budget. This trades independent positions
per update for exact color balance. Replay stores original experiences only.
Held-out games are separated before augmentation and validation remains unaugmented.
The feature works with existing checkpoints; it encourages symmetry without
forcing identical predictions. Epoch logs and MLflow record whether it is active.

#### Full board symmetries

Add `--spatial-augmentation` alongside `--color-augmentation` for the sparse rules.
Each of the 2,048 selected source positions is assigned one of eight D4 orientations
(four rotations, with/without reflection), balanced across the batch. Its paired
color swap gets the same spatial transform. Thus all 16 variants are represented
across each full batch without reducing independent source positions to 256 or
increasing the 4,096-row budget. Orientation assignments vary each iteration.
Piece IDs stay with their pieces during spatial transforms; the color counterpart
mapping remains as above. Move distributions, masks, actions, and history board
frames follow the spatial transform. Side features, values, tuning targets, and
kind targets do not change under spatial transforms. This mode rejects non-sparse
rules because other wave presets may depend on orientation.

MLflow `training/augmentation_variants` records 16 when enabled; training and
validation losses retain their existing definitions.

### Tensor rules and batched MCTS

`--mcts-backend tensor --mcts-rules-device cpu|mps|cuda` runs batched self-play
with the same model, 128 simulations, legal action set, value discount, tuning cap,
ply cap, game holdouts, replay training, and repetition rule. The default remains
`--mcts-backend rust`: benchmark full self-play before changing the service config.

- `tensor_rules.py`: fixed-size board/state tensors, discrete influence lookup,
  sliding ray movement, complete candidate king-safety checks, simultaneous rescue
  deadline cascades, tuning, and NN encoding. CPU and CUDA use portable PyTorch.
- `metal_rules.py`: fused Apple GPU kernels for fields, all action masks, state
  transitions, and encoding. Uses the installed PyTorch `mps.compile_shader` API;
  no PyTorch upgrade, compiler extension, or separate GPU runtime is needed.
  `TensorRules('mps', fused=False)` retains the portable MPS implementation.
- `tensor_mcts.py`: one leaf from each independent game per expansion batch. Rules
  and NN share a device; tree selection and value backup remain Python on CPU.
  The model is copied for inference, preserving the learner and optimizer device.
- `tensor_selfplay.py`: records ordinary training samples, outcomes, tuning and
  tactical metrics, and committed-action spectator frames. Viewing never slows
  generation. Buffered playback stays on its loaded game across batch replacement;
  click Live to rejoin the current producer.

Supported states are the current sparse product rules with named product pieces,
full-strength ±1 controls, the one-component triangle, and history_plies=1. Other
rule variants / model histories are rejected explicitly. CUDA code is available
but cannot be hardware-tested on this Mac. CPU and MPS are checked against Rust
for fields, masks, encodings, rescue losses, activation ordering, and transitions.

Example (separate run directory; existing service continues with its configured backend):

```sh
PYTHONPATH=training python -m wavefield.experiment \
  --run-dir training/runs/tensor-trial --resume-checkpoint /path/to/checkpoint.pt \
  --model-arch transformer --hidden-size 256 --input-view piece_identity \
  --full-policy --model-games 0 --mcts-games-per-iteration 8 \
  --mcts-backend tensor --mcts-rules-device mps --mcts-simulations 128 \
  --mcts-depth 32 --mcts-top-k 0 --max-tuning-actions 20 --max-plies 600 \
  --mcts-repetition-limit 3 --mcts-value-discount .997 --iterations 1
```

Reproduce local timing comparisons with `training/benchmark_tensor_rules.py`:

```sh
PYTHONPATH=training python training/benchmark_tensor_rules.py \
  --checkpoint /path/to/frozen-model.pt --output /tmp/tensor-benchmark.json \
  --games 8 --max-plies 4 --simulations 128 --repeats 2 --pause-training
```

`--pause-training` verifies and pauses the local trainer process group during
measurement, then resumes it on normal exit, errors, or SIGTERM. The benchmark
never writes the production replay/checkpoint or spectator feed. Rules timings
compare serialized Rust requests with tensor positions already resident on device;
full self-play timings also include model copying and Rust worker startup.

`--mcts-leaf-batch-size 8` optionally reserves up to eight leaves per game using
virtual loss before one batched evaluation. Every reservation is removed before
backup, and the real simulation budget remains exactly 128 per decision. This is
parallel MCTS: search order and therefore visit targets / selected actions can
change. The default of 1 matches the existing sequential search in parity tests.
The benchmark accepts the equivalent `--leaf-batch-size 8` option. Do not interpret
short timing differences under parallel search as identical game trajectories.

### Continuous asynchronous CPU self-play

Add `--async-selfplay` to local Rust CPU MCTS training. Persistent actors start
new games independently; every `--mcts-games-per-iteration` completed games feeds
the learner while actors continue playing. Each game freezes its model at the
start and checks the published weights before its next game. Eight workers,
eight games per learner cohort, and `--max-plies 300 --eval-max-plies 300` are the
service settings. The NN effective batch remains 4096 with microbatches of 128.

The queue is bounded to three times the worker count by default (24 games,
including games in progress and the cohort being trained). Actors wait when
this fills. Set `--async-queue-capacity` to override it; capacity must accommodate
all workers plus one learner cohort. This bounds both memory/disk use and policy
lag. Generation logs include game IDs, source model update counts, summed actor
time, learner wait time (`seconds`), and cohort wall time (`wall_seconds`).

Completed games are atomically saved under `async-selfplay/inbox/`. They are
removed only after a learner checkpoint records their consumption, so restarting
reuses completed games without counting them twice. Games still in progress
restart. Asynchronous mode requires CPU/local Rust inference and checkpoints
after every cohort (`--recovery-checkpoints` or `--save-every 1`). The existing
launchd service, MLflow logging, validation holdouts, and live viewer still apply.
