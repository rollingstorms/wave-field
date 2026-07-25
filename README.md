# Wave Field

Wave Field is a local hot-seat spectral strategy game MVP. Two players move and
tune pieces on a 7x7 board whose red, neutral, and blue territories are computed
from deterministic wave components.

Play the live demo at [rollingstorms.github.io/wave-field](https://rollingstorms.github.io/wave-field/).

## Commands

- `npm run dev` starts the local Vite app.
- `npm test` runs the Vitest unit suite.
- `npm run build` type-checks and builds the production bundle.

## Current MVP

- Flat three-state board projection from continuous field values.
- Geometric SVG pieces for pawns, spies, rooks, and kings.
- Legal adjacent movement with territory restrictions and spy immunity.
- Unlimited in-turn tuning with each piece type's nonzero components capped by strength;
  moving a piece ends the turn.
- Rescue-window instability, forced removals, king trap detection, undo, and restart.
- Developer mode with raw field metrics, mobility, contribution views, kernel previews,
  editable component definitions, and JSON import/export.
