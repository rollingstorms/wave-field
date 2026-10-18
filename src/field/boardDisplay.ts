import { BOARD_SIZE } from "../game/constants";

export interface FieldDisplaySample {
  value: number;
  activity: number;
}

function weights(fraction: number): number[] {
  const t = Math.max(0, Math.min(1, fraction));
  const inverse = 1 - t;
  return [
    inverse ** 3 / 6,
    (3 * t ** 3 - 6 * t ** 2 + 4) / 6,
    (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6,
    t ** 3 / 6,
  ];
}

export function sampleBoardField(field: number[][], x: number, y: number): FieldDisplaySample {
  const baseX = Math.floor(x);
  const baseY = Math.floor(y);
  const xWeights = weights(x - baseX);
  const yWeights = weights(y - baseY);
  let value = 0;
  let activity = 0;
  for (let row = 0; row < 4; row += 1) {
    const sampleY = Math.max(0, Math.min(BOARD_SIZE - 1, baseY + row - 1));
    for (let column = 0; column < 4; column += 1) {
      const sampleX = Math.max(0, Math.min(BOARD_SIZE - 1, baseX + column - 1));
      const fieldValue = field[sampleY]?.[sampleX] ?? 0;
      const weight = xWeights[column] * yWeights[row];
      value += fieldValue * weight;
      activity += Math.abs(fieldValue) * weight;
    }
  }
  return { value, activity };
}

export function sampleBoardDisplay(field: number[][], samplesPerSquare = 9): FieldDisplaySample[][] {
  const resolution = BOARD_SIZE * samplesPerSquare;
  return Array.from({ length: resolution }, (_, row) =>
    Array.from({ length: resolution }, (_, column) => sampleBoardField(
      field,
      (column + 0.5) / samplesPerSquare - 0.5,
      BOARD_SIZE - 0.5 - (row + 0.5) / samplesPerSquare,
    )),
  );
}
