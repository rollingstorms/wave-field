# Wave Field Strategy Notes

Wave Field is a deterministic strategy game about interference. Like Conway's
Game of Life, the core appeal is not hidden randomness or narrative state. A
small fixed rule set creates complicated behavior because each move changes a
field, and the field changes what future moves mean.

The game is also a study of power dynamics. Pieces do not capture by occupying
the same square. They pressure, cancel, reinforce, strand, rescue, and trap. A
winning position often looks less like taking material and more like making a
region impossible for the opposing Big Hat to inhabit safely.

## Core Thesis

The strategic depth comes from several deterministic systems being readable at
different speeds:

- the local board, where pieces block rays and occupy specific squares;
- the territory field, where every piece contributes pressure across the board;
- the tuning profile, where each piece type can change how its pressure lands;
- the instability clock, where unsafe pieces can be risked, rescued, or lost;
- the trap race, where the Big Hat's safety matters more than material.

A player who only reads legal moves usually loses quickly. A stronger player
reads field consequences: which moves thicken friendly territory, which moves
stretch the position too thin, and which moves make the opponent's next rescue
impossible.

## Determinism and Emergence

The game should be documented as a deterministic system that produces emergent
patterns. That framing matters because new players often experience the rules as
arbitrary complexity. The better explanation is that nothing arbitrary is
happening: every surprising position is the visible result of fixed waves,
movement rays, and rescue deadlines.

Useful comparison points:

- **Game of Life:** simple local rules produce life-like structures, oscillators,
  and long-lived patterns.
- **Chess and checkers:** simple movement rules produce tactics, openings, and
  endgames.
- **Wave Field:** simple field addition and ray movement produce interference,
  siege, rescue, sacrifice, and collapse patterns.

The long-term design opportunity is to name Wave Field patterns the way other
deterministic games name openings, forks, pins, ladders, gliders, or still lifes.

## Current Strategic Concepts

### Safety Margin

A safe square is friendly or Neutral. A strong square is not merely safe; it has
enough margin that one opposing move or tuning change is unlikely to flip it.
Good play keeps the Big Hat and key pieces inside a connected region with
margin.

### Stretching the Field

Moving pieces apart increases reach but can tear holes in the position. A useful
mental model is stretching dough: a broad shape is powerful only while it stays
connected. Stretch too far and the opponent can create a hostile pocket.

### Interference

The most important moves are often not direct advances. A move can reinforce a
friendly region, cancel an enemy region, or move a wave origin so that a distant
square changes sign. Strong players look for moves whose field effect is larger
than the moved piece's local displacement.

### Rescue Rate

Unsafe non-Big Hat pieces are not automatically bad. A player can intentionally
risk a piece if the move creates enough pressure, wins tempo, or sets up a trap.
Still, early learners need a high rescue rate. They should first learn to keep
pieces stable, then learn when sacrifice is worth it.

### Siege

A siege occurs when both players stabilize into a standoff and the win requires
slowly improving field shape rather than finding an immediate trap. Siege play
tests whether a player can thicken territory, rotate pressure, and avoid
overextension while waiting for one decisive collapse.

### Sinkholes

A sinkhole is a hostile pocket created under or near an opposing piece. Unlike a
capture in chess, the tactical action is not occupying that square. The tactic is
changing the field so the opponent must spend a turn rescuing, moving, or losing
material.

## Teaching Implications

New players can move pieces immediately, but they cannot yet see why they are
losing. The tutorial should reduce the branching factor before introducing the
full game.

Recommended teaching sequence:

1. Move without tuning. Teach rays, blockers, territory, and Big Hat safety.
2. Show unsafe moves clearly. Explain that a yellow risky marker means the move
   can be played, but one of your pieces may be lost after the turn resolves.
3. Teach rescue. Start with a position where one move obviously saves an
   unstable piece.
4. Teach field movement. Show that moving a different piece can rescue or trap
   without touching the endangered square.
5. Add tuning. Limit the first tuning lesson to one piece type and one visible
   before/after effect.
6. Play against a generous opponent. The first AI should leave traps and rescues
   available so the player can experience intentional success.
7. Reintroduce the heuristic opponent only after the player understands unsafe
   markers, rescue, and Big Hat traps.

The game may be hard to explain verbally because its core skill is spatial and
dynamic. A better tutorial should let the player feel one concept at a time.

## AI and Research Hooks

The current heuristic model is useful because it gives the game tactical
pressure. A human win rate around 80% against that model is evidence that the
game has learnable strategy: repeated play produces skills that the heuristic
does not fully capture.

Research questions to track:

- Which board patterns most often precede successful Big Hat traps?
- Which unstable-piece risks produce wins rather than material collapse?
- Which tuning profiles create reliable openings, sieges, or rescues?
- How often do strong moves change distant territory compared with local
  mobility?
- Can self-play discover named structures that humans also recognize?
- Does a generous-opponent curriculum teach faster than direct play against the
  heuristic model?

Useful metrics:

- human and model win rate by side;
- mean plies to win;
- rescue rate after unsafe moves;
- first-loss outcome;
- pressure around both Big Hats;
- number of field sign changes caused by each move;
- material sacrificed before a successful trap;
- frequency of repeated or stalled states.

## Pattern Vocabulary to Develop

These are working names, not final rules terms:

- **Thickening:** increasing safety margin around an important piece or region.
- **Tearing:** overextending pieces until friendly territory splits or exposes a
  hole.
- **Sinkhole:** creating hostile territory under an opposing piece.
- **Rescue swing:** moving one piece to stabilize another piece at a distance.
- **Trap net:** a position where the opposing Big Hat has legal-looking movement
  but no safe final square after tuning and movement are considered.
- **Siege line:** a stable front where both players have pressure but neither can
  immediately trap.
- **Frequency switch:** a tuning change that reorients the same material into a
  different strategic pathway.

## Generative Art and Music

Wave Field is not only a board game. The board state can be decomposed into
visual and musical signals, which makes play feel like a generative instrument.
Because the position changes deterministically and continuously, the soundtrack
can reflect pressure, territory, instability, rescue, and collapse without
needing a fixed composition.

That matters for product design. The game can teach through sound and motion as
much as through text. If the player hears tension rising around an unsafe piece
or a Big Hat trap, the board becomes easier to read.

## Product Direction

The next strategy-facing work should focus on:

- a beginner AI that deliberately offers clear wins and rescues;
- a tutorial suite that limits complexity one layer at a time;
- better explanations for unsafe moves and risky sacrifices;
- saved example positions for rescue, sinkhole, siege, and trap-net patterns;
- post-game summaries that name the decisive pattern;
- analysis tooling that mines self-play and human games for repeated structures.

The goal is not to simplify the game into something shallow. The goal is to
stage the complexity so players can build intuition before facing the full
system.
