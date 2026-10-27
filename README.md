# Wave Field

**Wave Field** is a deterministic two-player strategy game on a 7×7 board.
Pieces generate the territory themselves by contributing exact, local support
patterns. Count Red and Blue separately: the higher total owns a square, and
a tie is Neutral.

**[Play Wave Field](https://rollingstorms.github.io/wave-field/)**

New games use the [discrete field rules](RULES.md). `/sparse` is an alias for
this default game. The original wave game remains available at
[`/classic`](https://rollingstorms.github.io/wave-field/classic/) with its
[original rules](docs/CLASSIC_RULES.md). Existing wave strategy and research
notes in [docs/STRATEGY.md](docs/STRATEGY.md) describe that classic game.

## Objective

Win by **trapping the opposing Big Hat**. A Big Hat is trapped when its square is
hostile territory, making the Big Hat unstable. After a move traps a Big Hat, the
trapped player gets a visible rescue turn. If the rules cannot find a rescue,
the status line says so instead of ending the game immediately.

## How You Play

Each turn ends when you move one piece. Before moving, you may tune your pattern
controls as much as you like. You can play by moving alone, but tuning is
usually the way to escape danger, protect your Big Hat, or turn a distant part of
the board.

Click a piece to see its legal destinations, or drag it across the board. Moving
one piece ends the turn, removes its old support footprint, and applies it at
the destination.

The `+` and `−` controls select a piece type's support pattern. Tuning is
shared by type, so both of your Towers use the same Tower profile.

## The Field

A piece supports only its own team. Each covered square receives 1, 2, or 4
support. Patterns extend at most two squares; all other contributions are 0.
The field is `Red support − Blue support`, so its sign decides territory.

Tuning concentrates support: broad patterns give 1 per square, focused patterns
give 2 on fewer squares, and Tower/Big Hat `−−` gives 4 on the four adjacent
cross squares. The control previews show the exact footprints. See the
[complete pattern table](docs/SPARSE_VARIANT.md).

Every piece contributes 0 on its own square except the Triangle Hat, which
contributes 1. Other pieces may support that square. Field totals are integers.

## Pieces

Each player begins with two Round Hats, two Towers, one Triangle Hat, and one
Big Hat. Blue moves first.

| Piece | Controls | Own-square support | Movement |
| --- | ---: | ---: | --- |
| Round Hat | 1 | 0 | Any distance along a clear horizontal, vertical, or diagonal ray |
| Tower | 2 | 0 | Same |
| Triangle Hat | 1 | 1 | Same, ignoring hostile territory |
| Big Hat | 2 | 0 | Same; must end the turn on friendly or Neutral territory |

## Turn

During a turn, a player may:

1. Tune their pattern controls any number of times.
2. Move one piece.

Moving a piece ends the turn. Tuning does not.

The dice control randomizes every piece type's tuning while keeping each type
at full active strength. It counts as tuning and does not end the turn.
The adjacent reset control restores the current player to the configured
defaults as one undoable tuning action.

A tuning change may temporarily leave the player's Big Hat unprotected because the
player can keep tuning during the turn. The turn-ending move must leave their
Big Hat safe.

## Tuning

Both teams see the same pattern signs. `+` or `++` selects wide coverage;
changing signs focuses the footprint. The dice randomizes a valid profile,
and reset restores the configured defaults. Tuning is shared by piece type.

Developer mode can change default profiles and inspect support, mobility, and
pressure. Classic wave definitions and scale editors are available in the
classic game.

## Movement and Territory

Choose one of the eight directions and move any distance along that ray. For
non-Triangle Hat pieces, every crossed square, including the destination, must be
unoccupied and either friendly territory or Neutral. The first occupied or
hostile square stops the ray, so pieces cannot jump or turn during a move.

- Red pieces may move through Red or Neutral territory.
- Blue pieces may move through Blue or Neutral territory.
- Triangle Hats may move through territory of either sign or Neutral.
- Pieces cannot capture by collision; friendly and opposing pieces both block.

The board distinguishes movement from turn safety:

- A white ring marks a move that can be completed.
- A yellow `B` marks a reachable square that is blocked because ending there
  would leave your Big Hat unprotected.
- A yellow diamond with `!` marks a playable move that will cause one of your
  non-Big Hat pieces to be lost when the turn resolves.

## Instability

A non-Big Hat piece standing on hostile territory is **unstable**. The player may
rescue it by moving that piece to a safe square or by moving a different piece
whose relocated pattern makes the unstable piece's square friendly or Neutral.
A newly unstable non-Big Hat piece remains on the board through its owner's next
turn, including all tuning changes and the turn-ending move.

One move may rescue multiple unstable pieces if the resulting field stabilizes
all of their squares. Any piece that was unstable at the start of the turn and
remains on hostile territory after the move is removed.

The field is recalculated after each removal because the lost piece's pattern
disappears. This can trigger further removals among pieces whose rescue deadline
has arrived.

Triangle Hats still ignore hostile territory while moving, but a Triangle Hat that stands on
hostile territory after a turn resolves is unstable like any other non-Big Hat
piece.

## Traps and Victory

A Big Hat must be protected by friendly or Neutral territory at the end of every
move. A player cannot complete a move that leaves their own Big Hat on hostile
territory.

When a completed move places the opposing Big Hat on hostile territory, the
opponent begins their turn with an unstable, trapped Big Hat. The game looks for a
rescue across every valid combined tuning profile, followed by one legal move.
If a rescue exists, the status line gives a concrete hint such as "tune, then
move Triangle Hat to 4,5". If no rescue is found, the status line says so and leaves the
trapped player in control.

While trapped, a player may resign from the unprotected-Big Hat warning instead
of continuing to search for an alternate tuning and legal escape. The adjacent
Hint control searches every valid combined tuning profile, applies the profile
with the fewest changed component slots, and selects a piece with a legal
rescuing move.

Big Hats are never captured or removed. A trapped player remains in control until
they find a rescue, resign, or undo.

## Starting Position

The opening formation is rotationally symmetric:

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

`R` = Round Hat, `T` = Tower, `A` = Triangle Hat, `B` = Big Hat.

## The Live Rules Demo

The demo includes an optional heuristic Red opponent, selected-piece influence
borders colored by contribution, drag-to-preview field updates, live
wave-pattern thumbnails, a per-square Red and Blue support overlay,
an interactive CMYK energy view with isolatable Round Hat, Tower, Triangle Hat, and Big Hat
channels and per-square composition ratios,
legal-move indicators, blocked-Big Hat move markers, instability warnings, loss
burst animations, undo and restart, continuous field-magnitude shading, a compact
mobile tuning panel, and a developer mode for inspecting the raw field, piece
contributions, mobility, and default pattern controls.
The selected control profile is applied to both players on restart. Its
scrollable History Roll reconstructs moves, tuning
changes, dice and hint profiles, piece losses, and turn transitions from the
active undo timeline. The Pressure panel summarizes legal moves, unstable
pieces, Big Hat safety, and a recent pressure trend.

The optional Red AI uses a bounded heuristic search. It evaluates many tuning
and movement candidates quickly, then runs exact safety validation on a small
shortlist so mobile browsers remain responsive.

The implementation is deterministic: the same position and component settings
always produce the same field and legal moves.

## Run Locally

```bash
npm install
npm run engine:build
npm run dev
```

The development app and AI arena use the Rust rule engine through WebAssembly.
Install its one-time toolchain prerequisites with:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.126 --locked
```

`npm run dev` and `npm run arena` rebuild the WebAssembly module before starting
Vite. The public `/arena` route is static-safe for GitHub Pages; local neural
model controls live under `/local-arena` in development and call Python model
servers on localhost. The Rust crate lives in `engine/`; its native JSON-lines
binary is also a convenient boundary for scripts and future Python bindings.
The default production game uses Rust/WASM, with TypeScript as its fallback.
Local neural models and wave-parameter experiments continue to use classic rules.

Use `npm test` to build Rust, run the existing TypeScript rules suite, and run
cross-language parity checks for field evaluation, movement, turn resolution,
tuning, instability, and check behavior. Use `cargo test --manifest-path
engine/Cargo.toml` for Rust-only tests and `npm run build` for the production
build.

## Engine

The TypeScript and Rust engines implement the same discrete footprints. Rust
serves the browser through WASM and native clients through the existing API.
New games use the discrete rules; old serialized states without a variant stay
classic. Undo and restart preserve the selected ruleset.

To compare field evaluation in a release build:

```sh
cargo run --release --manifest-path engine/Cargo.toml --example bench_fields
```

The benchmark compares eight opening/spread positions and tuning profiles. It
measures field evaluation, not whole AI turns.

## Rust Engine Benchmarks

The native Rust CLI supports developer simulation modes for arena analysis and
training experiments:

- `simulateRandomGames`: random legal self-play with rich per-ply metrics.
- `simulateRandomLeanGames`: random legal self-play with terminal stats only.
- `profileRandomGames`: lean random self-play with timing for candidate
  generation and move application.
- `simulateAiGames`: heuristic AI self-play baseline.

Run the layered benchmark with:

```bash
npm run engine:bench
```

Useful overrides:

```bash
npm run engine:bench -- --games 5000 --heuristic-games 50 --time-budget-ms 10
```

Lean mode uses a training-oriented move path that skips browser-only history,
status-message work, and full no-rescue trap search while preserving ordinary
move consequences and cheap terminal checks.

## License

The code in this repository is licensed under the MIT License. See
[LICENSE](./LICENSE).
