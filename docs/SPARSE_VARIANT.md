# Sparse variant

The `/sparse` route is an experimental 7×7 version of Wave Field. It keeps the
current piece patterns, movement, shared tuning controls, instability, and Big
Hat rescue rules. The default route retains the original field rules.

## Field rule

For each piece and square, first calculate that piece's ordinary signed wave
contribution using its current tuning, pattern, strength, and scales. Then:

- On the piece's own square, its contribution is `+1` for its owner.
- Elsewhere, values below `1`, including all negative lobes, become `0`.
  Values from `1` to below `2` become `1`, and values `2` or greater become `2`.
- Add contributions from Red pieces to get `R`, and from Blue pieces to get
  `B`. Both totals are nonnegative. The field is `R - B`: the higher count owns
  the square, and a tie is Neutral.

The board shows `R` and `B` on each square by default. Moving a piece
recomputes the field from current positions, so its old contribution is erased
and its pattern is applied at its destination. This also happens in the
original game; it is not a new sparse rule.

The cutoff, maximum stamp, and home-square strength are constants in
`src/game/constants.ts`. They can be tuned for this variant without changing
the default game.

## Shelved idea: per-piece tuning on move

A later experiment could replace free, shared tuning with a choice attached to
the moved piece, such as rotating a Tower's pattern as part of its move. That
would make the resulting field change local to one piece and help players
predict it. It is intentionally not active in `/sparse`: this first experiment
tests the effect of sparse, integer contributions while keeping the current
turn and tuning rules. Implementing per-piece tuning would need a piece-level
profile in game state, move UI, undo history, AI actions, and Rust parity.
