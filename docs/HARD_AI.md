# Hard AI under the discrete rules

The browser gives Hard the same 2,500 ms search budget as before. The Rust/WASM
search considers all 64 complete tuning profiles at the root. Deeper nodes rank
complete profiles and use a narrower selection to fit the budget. Patterns for
piece types no longer on the board are not searched redundantly.

The earlier search considered only the current settings and one-control flips.
This could falsely resign when a legal rescue needed several simultaneous
changes. `engine/tests/sparse-coordinated-rescue.json` records a position where
the nearest rescue requires three changes. Hard now finds it.

The discrete evaluator scores material, support, mobility, and threats
symmetrically for both players. A temporary Big Hat check receives a positional
bonus; terminal wins retain their separate winning score. Search caches label
pruned values as bounds, and incomplete iterations do not replace the last
completed result. Root scores are reused without searching the iteration twice.
The discrete field is recomputed directly because it costs less than building
and looking up its old string cache key. The search reserves part of its budget
for checking immediate losing replies.
Iterative search can continue up to 12 plies when the time budget permits.
Browser cache keys use the WASM file contents so each engine update loads the
matching binary.

## Audit

```sh
npm run engine:audit-hard
AI_BUDGET_MS=250 BASELINE_ENGINE=/path/to/previous/wave-field-engine npm run engine:audit-hard
```

This reports root profile coverage, completed depth, candidate counts, elapsed
search time, and the result of the coordinated rescue. It uses the native
release build; browser throughput depends on the device.

A development check against the previous engine used four seeded openings,
each played from both sides, with 250 ms per turn and a 64-ply cap. The updated
engine won one game, lost none, and seven games reached the cap. This small
sample is a regression check, not a playing-strength rating.
