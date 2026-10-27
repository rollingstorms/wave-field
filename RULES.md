# Wave Field Rules

## Objective

The objective is to trap the opposing Big Hat.

A Big Hat is trapped when it sits on hostile territory. In that state the Big Hat is
unstable and must be rescued. The game does not remove Big Hats from the board:
the trapped player remains in control until they rescue the Big Hat, resign, or
undo.

## How You Play

Every piece contributes a small, exact pattern of support to its own team.
Tuning trades wider 1-point coverage for focused 2-point coverage. Towers and
Big Hats can concentrate 4 support on the four adjacent cross squares.

On your turn you may tune your pattern controls, then move one piece. Click a
piece to see legal destinations, or drag it to move. Tuning is shared by type:
both of your Towers use the same profile, and the opponent controls theirs.

## Field

Add Red support and Blue support separately on each square. The higher total
owns it; equal totals are Neutral. A piece cannot contribute to the opponent.
The board shows both team totals, and the signed field is `Red − Blue`.

Each pattern reaches at most two squares from its piece. Moving removes the
piece's old footprint and adds it at the destination. Its own square receives
0 from that piece, except the Triangle Hat, which gives itself 1 support.
Other pieces can still support that square.

The `+` and `−` controls select the same profiles for both teams. Round Hats
switch between a wide cross and focused diagonals. Triangle Hats switch
between a wide diamond and a focused small diamond. Towers and Big Hats have
wide, mixed, and tight modes; `−−` is the tight cross at 4 per adjacent square.
See the [complete pattern table](docs/SPARSE_VARIANT.md).

The original wave rules are available at `/classic` and documented in
[Classic Rules](docs/CLASSIC_RULES.md). `/sparse` remains an alias for the default
rules.

## Setup

Each player begins with two Round Hats, two Towers, one Triangle Hat, and one Big Hat. Blue moves
first.

```text
        Blue
    . . T B T . .
    . . R A R . .
    . . . . . . .
    . . . . . . .
    . . . . . . .
    . . R A R . .
    . . T B T . .
         Red
```

## Turn

On a turn, a player may tune their pattern controls any number of times, then move
one piece. Moving ends the turn. Tuning by itself does not.

A tuning change may temporarily make the player's own Big Hat unstable. The
turn-ending move must leave that Big Hat on friendly or Neutral territory.

## Movement

Pieces move any distance in one straight horizontal, vertical, or diagonal ray.
Pieces do not use chess movement despite their names.

Round Hats, Towers, and Big Hats may cross only friendly or Neutral empty squares. Triangle Hats
ignore territory while moving. All pieces are blocked by occupied squares, and
there are no captures by collision.

## Unstable Pieces

A non-Big Hat piece on hostile territory is unstable. On that player's next turn,
it can be rescued by moving it to safety or by moving another piece so the field
around it becomes friendly or Neutral. If it is still unstable after the move
resolves, it is removed.

Triangle Hats can move through hostile territory, but they can still become unstable if
they end up standing on hostile territory.

## Trapped Big Hats

A Big Hat on hostile territory is an unstable trapped Big Hat. When a move traps the
opposing Big Hat, the opponent gets a rescue turn.

The rescue can come from tuning, from moving the Big Hat, or from moving another
piece whose relocated pattern makes the Big Hat's square friendly or Neutral. The
Hint control searches for a nearby rescue profile and move. If no rescue is
found, the trapped player still controls the turn and may keep searching,
resign, or undo.

Big Hats are never captured or removed.
