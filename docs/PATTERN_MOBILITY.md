# Pattern mobility audit

Run `npm run patterns:audit` after changing a pattern or wave scale. The report
enumerates every full-strength tuning phase for Round Hats, Towers, Triangle
Hats, and Big Hats. It measures two different things:

- **Isolated moves and dead origins:** Place one piece at each of the 49 board
  squares. This catches patterns that leave no outward path anywhere.
- **Reachable, playable, and safe opening moves:** Apply each phase to both
  players in the complete opening. A safe move preserves every allied piece
  through the move's immediate rescue and loss rules.

The opening is only one position. For deeper evaluations, collect complete
`GameState` snapshots from self-play or saved games and pass them to
`positionProfileMobilityDiagnostics(states)`. Sample openings, crowded middle
games, rescue turns, and positions with one or more pieces missing. Track the
minimum safe moves per piece and phase, the number of pieces with no safe move,
and the distribution across states. Occupancy can legitimately block the Big
Hat in the initial position, so a zero opening count alone is not a pattern
failure.

Select candidates by their **worst phase and position**, then compare total
mobility and field strength. A high average can hide a dead phase. Check the
opening territory layout, both players, and AI play after each change. Keep
rendering tests separate: field shading should sample the same field used by
the game's movement rules.

The current Tower redesign replaces its second ring with directional lanes.
The Big Hat's first pattern balances nearby support against its own movement.
Together they remove the Tower's zero-move isolated phase and give every Tower
phase safe opening moves without changing the opening's red, neutral, and blue
rows. Compare all Big Hat phases in played positions before changing the first
ring: `[-1, 1, 1, 1, -1]` preserved Tower safety but sharply reduced Big Hat
safe moves in the `--` phase. The current `[-1, 1, -1, 0, -1]` ring improved
the weakest phase in a paired sample of 108 positions.
