import { describe, expect, it } from "vitest";
import { evaluateField } from "../field/evaluateField";
import { buildHistoryRoll } from "../game/historyRoll";
import { createInitialState } from "../game/initialState";
import { getLegalMoves } from "../game/movement";
import { gameReducer } from "../game/reducer";

describe("debug history roll", () => {
  it("describes player-relative tuning changes", () => {
    const state = createInitialState();
    const tuned = gameReducer(state, { type: "tune", pieceType: "pawn", componentIndex: 0, value: -1 });
    const entries = buildHistoryRoll(tuned);

    expect(entries).toHaveLength(1);
    expect(entries[0].summary).toBe("Blue tuned 1 control");
    expect(entries[0].details).toContain("Blue round hat C1 -→+");
  });

  it("describes moves with board coordinates", () => {
    const state = createInitialState();
    const destination = getLegalMoves("blue-rook-1", state, evaluateField(state))[0];
    const moved = gameReducer(state, { type: "move", pieceId: "blue-rook-1", destination });
    const entries = buildHistoryRoll(moved);

    expect(entries.at(-1)?.summary).toBe("Blue moved tower");
    expect(entries.at(-1)?.details.some((detail) => detail.includes("Blue tower C7→"))).toBe(true);
  });

  it("describes home energy changes", () => {
    const state = createInitialState();
    const changed = gameReducer(state, { type: "update-home-energy", pieceType: "spy", value: 0.75 });
    const entries = buildHistoryRoll(changed);

    expect(entries.at(-1)?.details).toContain("triangle hat home 0.50→0.75");
  });
});
