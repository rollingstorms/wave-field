# Tensor rules and self-play benchmark — 2026-10-30

Machine: Apple M1 Pro, 16 GB unified memory, PyTorch 2.7.0. Frozen checkpoint
912, 6,345 lifetime optimizer updates, 5,101,405 parameters. Production was paused
for each measurement group and automatically resumed afterward. Benchmarks never
fed samples into training or changed the production checkpoint.

## Full MCTS self-play

All cases use 128 simulations, depth 32, all legal moves, 20 tunes per turn,
value discount .997, root noise, visit sampling, and seed 50100. Each game stops
at four piece plies for timing only; this is an opening throughput benchmark,
not a completed-game / training-duration or playing-strength result. Production
retains its 600-ply cap and threefold rule.

| Concurrent game batch | Backend | Elapsed | Decisions |
|---|---|---:|---:|
| 8 | Rust, eight workers | 7.12 s median | 48 |
| 8 | Tensor CPU, one thread | 7.61 s median | 48 |
| 8 | Tensor MPS, fused Metal rules | 11.62 s median | 48 |
| 128 | Rust, eight workers | 139.80 s | 832 |
| 128 | Tensor MPS, fused Metal rules | 84.40 s | 832 |

Eight-game cases have two repetitions with reversed backend order. The 128-game
case has one repetition: GPU throughput was 1.66× Rust, with
both producing 512 piece plies and 832 decisions. A tensor CPU four-thread probe
was also tested (about 7.34 s at eight games); it did not establish an advantage
over the Rust workers. Cross-run machine load and startup costs cause variation.

Unfused portable MPS rules took 55.84 s cold and 29.07 s warm for the same eight
games. Kernel fusion substantially reduces dispatch overhead. Search selection
and backup still run on CPU; transitions, fields, full move safety, encoding,
and inference run on the selected tensor device.

## Rules only

Median legal-mask times over five repetitions, using varied audited states.
Tensor inputs and lookup tables are already resident on device. The Rust path
uses the existing serial JSON interface, including serialization/transport; this
is not a comparison with an in-process native Rust function or eight workers.

| Positions | Rust JSON serial | Tensor CPU | Fused Metal |
|---:|---:|---:|---:|
| 1 | 0.560 ms | 0.288 ms | 0.333 ms |
| 8 | 3.059 ms | 0.694 ms | 0.390 ms |
| 32 | 15.078 ms | 1.897 ms | 0.303 ms |
| 128 | 54.760 ms | 6.785 ms | 0.537 ms |

## Parallel leaves (optional)

`--mcts-leaf-batch-size 8` uses virtual loss to reserve several leaves per game.
It preserves exactly 128 real simulations per decision but changes search order.
At eight games, MPS took 7.54 / 7.02 s versus Rust's 8.10 / 9.40 s in that
measurement group. MPS made 45 decisions versus Rust's 48, so this is not an
identical-trajectory speedup. The result does not yet justify silently changing
the production search policy.

## Verification and deployment

- 80 Python tests passed, including CPU/Rust parity, portable MPS parity, fused
  Metal transitions/masks/encoding, real repetition draws, equal sequential MCTS
  visits/targets, virtual-reservation cleanup and budgets, and committed feeds.
- 23 Rust tests passed for the baseline; TypeScript check and viewer build passed.
- A real CLI run completed GPU self-play, game holdout, symmetry augmentation,
  CPU NN optimization, and checkpoint saving in an isolated temporary run.
- CUDA is supported by the portable tensor code but was not hardware-tested here.
- Backend code is deployed; production remains eight Rust workers because the
  controlled eight-game benchmark favors them. Trainer resumed after benchmarks.
- The deployed viewer supports slow buffered replay across producer batch changes;
  playback does not throttle training. Existing Play / speed / Live controls apply.

Raw JSON and the frozen model live in the local ignored directory
`training/tracking/tensor-benchmark-20261030/`. Reproduce using
`training/benchmark_tensor_rules.py`; use `--mode selfplay --games 128` for the
larger batch, and `--leaf-batch-size 8` for the optional parallel search.
