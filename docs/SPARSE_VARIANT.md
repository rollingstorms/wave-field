# Discrete field rules

These are the default 7×7 Wave Field rules at `/`. The `/sparse` route is an
alias. Movement, shared tuning, instability, and Big Hat rescue use the same
turn rules. The original wave field is available at `/classic`.

## Discrete support patterns

Every piece gives support only to its own team. Each tuning profile selects a
fixed footprint centered on that piece. A covered square receives 1, 2, or 4
support; all other squares receive 0. The pattern reaches at most two squares
away. Tuning trades broad 1-point coverage for fewer stronger squares.

| Piece | `+` or `++` | `-` or `+-` | `-+` | `--` |
| --- | --- | --- | --- | --- |
| Round Hat | Cross, two steps, 1 each | Four adjacent diagonals, 2 each | — | — |
| Triangle Hat | Diamond, two steps, 1 each | Four adjacent orthogonal squares, 2 each | — | — |
| Tower | Cross and diagonal rays, two steps, 1 each | Cross, two steps, 2 each | Diagonal rays, two steps, 2 each | Four adjacent orthogonal squares, 4 each |
| Big Hat | Full square, two steps, 1 each | Diamond, two steps, 2 each | Diagonal rays, two steps, 2 each | Four adjacent orthogonal squares, 4 each |

The origin square receives 0 support from its own piece, except the Triangle
Hat, which gives 1. This keeps the original game's special Triangle Hat origin
in a discrete form. For both teams, the controls show pattern signs directly.
The small preview and its caption show the chosen footprint and support per
covered square.

Add the Red pieces to get `R` and Blue pieces to get `B`. Both totals are
nonnegative. The field is `R - B`: the higher total owns the square; a tie is
Neutral. The board shows `R` and `B` on every square. Moving a piece removes
its old pattern and applies the same pattern at its destination.

The TypeScript footprints are defined in `src/field/sparsePattern.ts` and
ported to `engine/src/field.rs` for the Rust/WASM engine. Both engines add local
integer patterns. The classic route uses its wave definitions and scales.

## Shelved idea: per-piece tuning on move

A later experiment could replace free, shared tuning with a choice attached to
the moved piece, such as rotating a Tower's pattern as part of its move. That
would make the resulting field change local to one piece and help players
predict it. It is intentionally not active in the default game, which keeps
the current turn and tuning rules. Implementing per-piece tuning would need a
piece-level profile in game state, move UI, undo history, AI actions, and Rust
parity.
