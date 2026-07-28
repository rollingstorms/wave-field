# Wave Field Rules

## Objective

The objective is to trap the opposing king.

A king is trapped when it sits on hostile territory. In that state the king is
unstable and must be rescued. The game does not remove kings from the board:
the trapped player remains in control until they rescue the king, resign, or
undo.

## How You Play

Every piece emits an invisible wave of energy across the board. Red and Blue
use opposite sign orientations, and each square is the sum of all waves from
all pieces. A single pattern can contribute both positive and negative energy
on different squares.

On your turn you may tune your wave controls, then move one piece. You can play
without tuning, but tuning is often how you rescue endangered pieces, protect
your king, and reshape the board before committing a move. Click a piece to see
legal destinations, or drag it to move.

The `+` and `-` controls activate a component with that field orientation, or
reverse the polarity of an active component. Pawns have one checkerboard-like
pattern. Rooks combine two overlapping patterns. Spies choose one of three
patterns. Kings keep two of three patterns active.

## Field

Every piece emits a controllable wave pattern. A pattern is not simply "good"
or "bad" energy: each component can contain both positive and negative energy
across different squares. Tuning changes which components are active and which
orientation they use, so the same piece type can reshape the field in several
ways.

The board adds all active piece patterns together. The final total on each
square decides territory:

| Field total | Territory |
|---|---|
| Greater than zero | Red |
| Zero | Neutral |
| Less than zero | Blue |

Moving a piece changes the origin of its pattern, so one move can change
territory far away from the moved piece.

## Setup

Each player begins with two pawns, two rooks, one spy, and one king. Blue moves
first.

```text
        Blue
    . . R K R . .
    . . P S P . .
    . . . . . . .
    . . . . . . .
    . . . . . . .
    . . P S P . .
    . . R K R . .
         Red
```

## Turn

On a turn, a player may tune their wave controls any number of times, then move
one piece. Moving ends the turn. Tuning by itself does not.

A tuning change may temporarily make the player's own king unstable. The
turn-ending move must leave that king on friendly or Neutral territory.

## Movement

Pieces move any distance in one straight horizontal, vertical, or diagonal ray.
Pieces do not use chess movement despite their names.

Pawns, rooks, and kings may cross only friendly or Neutral empty squares. Spies
ignore territory while moving. All pieces are blocked by occupied squares, and
there are no captures by collision.

## Unstable Pieces

A non-king piece on hostile territory is unstable. On that player's next turn,
it can be rescued by moving it to safety or by moving another piece so the field
around it becomes friendly or Neutral. If it is still unstable after the move
resolves, it is removed.

Spies can move through hostile territory, but they can still become unstable if
they end up standing on hostile territory.

## Trapped Kings

A king on hostile territory is an unstable trapped king. When a move traps the
opposing king, the opponent gets a rescue turn.

The rescue can come from tuning, from moving the king, or from moving another
piece whose relocated wave makes the king's square friendly or Neutral. The
Hint control searches for a nearby rescue profile and move. If no rescue is
found, the trapped player still controls the turn and may keep searching,
resign, or undo.

Kings are never captured or removed.
