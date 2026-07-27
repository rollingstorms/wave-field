import { FIELD_EPSILON } from "../game/constants";

const NEUTRAL = [184, 181, 173] as const;
const RED = [200, 75, 64] as const;
const BLUE = [55, 102, 167] as const;

function mixChannel(start: number, end: number, amount: number) {
  return Math.round(start + (end - start) * amount);
}

export function continuousFieldColor(value: number, maximumMagnitude: number) {
  if (Math.abs(value) <= FIELD_EPSILON || maximumMagnitude <= FIELD_EPSILON) {
    return `rgb(${NEUTRAL.join(", ")})`;
  }

  const target = value > 0 ? RED : BLUE;
  const normalized = Math.min(Math.abs(value) / maximumMagnitude, 1);
  const amount = Math.pow(normalized, 0.65);
  const channels = NEUTRAL.map((channel, index) => mixChannel(channel, target[index], amount));
  return `rgb(${channels.join(", ")})`;
}
