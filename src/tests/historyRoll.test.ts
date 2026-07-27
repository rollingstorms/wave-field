import { describe, expect, it } from "vitest";
import { evaluateField } from "../field/evaluateField";
import { buildHistoryRoll } from "../game/historyRoll";
import { createInitialState } from "../game/initialState";
import { getLegalMoves } from "../game/movement";
import { gameReducer } from "../game/reducer";

describe("debug history roll", () => {
  it("describes player-relative tuning changes", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "pawn", componentIndex: 0, value: 0 });
    const entries = buildHistoryRoll(tuned);

    expect(entries).toHaveLength(1);
    expect(entries[0].summary).toBe("Blue tuned 1 control");
    expect(entries[0].details).toContain("Blue pawn C1 -→0");
  });

  it("describes moves with board coordinates", () => {
    const state = createInitialState();
    const destination = getLegalMoves("blue-pawn-1", state, evaluateField(state))[0];
    const moved = gameReducer(state, { type: "move", pieceId: "blue-pawn-1", destination });
    const entries = buildHistoryRoll(moved);

    expect(entries.at(-1)?.summary).toBe("Blue moved pawn");
    expect(entries.at(-1)?.details.some((detail) => detail.includes("Blue pawn 3,6→"))).toBe(true);
  });
});
