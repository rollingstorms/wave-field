import { FIELD_EPSILON } from "../game/constants";

const NEUTRAL = [184, 181, 173] as const;
const RED = [200, 75, 64] as const;
const BLUE = [55, 102, 167] as const;
const IOS_FIELD_MAGNITUDE = 2.4;

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

// Match the iOS patterned board's fixed scale and activity-aware neutral pigment.
export function boardDisplayColor(value: number, activity: number): string {
  const normalized = Math.min(Math.abs(value) / IOS_FIELD_MAGNITUDE, 1);
  const strength = Math.pow(normalized, 0.72);
  const target = value >= 0 ? [226, 74, 59] : [43, 92, 196];
  const surrounding = Math.min(Math.max(activity / 0.4, 0), 1);
  const pigment = surrounding * surrounding * (3 - 2 * surrounding) * 0.85;
  const canvas = [19, 20, 22];
  const neutralPigment = [135, 83, 128];
  const channels = target.map((channel, index) => {
    const backdrop = neutralPigment[index] * pigment + canvas[index] * (1 - pigment);
    // The iOS palette blends from a transparent neutral, then composites the
    // resulting alpha over the activity backdrop.
    return Math.round(channel * strength * strength + backdrop * (1 - strength));
  });
  return `rgb(${channels.join(", ")})`;
}
