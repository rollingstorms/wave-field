#!/usr/bin/env node

const PIECE_TYPES = ["pawn", "rook", "spy", "king"];
const DISPLAY = { pawn: "Round Hat", rook: "Tower", spy: "Triangle Hat", king: "Big Hat" };
const SHORT = { pawn: "R", rook: "T", spy: "A", king: "B" };
const PIECE_STRENGTH = { pawn: 1, rook: 2, spy: 2, king: 2 };
const ACTIVE_COMPONENTS = { pawn: [1], rook: [1, 1], spy: [0, 1], king: [1, 1] };
const DEFAULT_WAVE_SCALES = {
  pawn: { friendly: 4, hostile: 1 },
  rook: { friendly: 3, hostile: 1 },
  spy: { friendly: 3, hostile: 0 },
  king: { friendly: 4, hostile: 2 },
};
const DEFAULT_HOME_ENERGY = { pawn: 0, rook: 0, spy: 0.5, king: 0 };
const towerComponentOneGrid = [
  [1, 1, 1, -1, 1, 1, 1],
  [1, 1, -1, 0, -1, 1, 1],
  [1, -1, 0, -1, 0, -1, 1],
  [-1, 0, -1, 0, -1, 0, -1],
  [1, -1, 0, -1, 0, -1, 0],
  [1, 1, -1, 0, -1, 1, 1],
  [1, 1, 1, -1, 1, 1, 1],
];
const DEFINITIONS = {
  pawn: [{ kind: "preset", preset: "checkerboard", decayBase: 2, originScale: 1 }],
  rook: [
    { kind: "grid", gridValues: towerComponentOneGrid, decayBase: 2, originScale: 1 },
    { kind: "ring", ringValues: [0, 1, 0, -1], repeat: true, decayBase: 2, originScale: 1 },
  ],
  spy: [
    { kind: "preset", preset: "checkerboard", decayBase: 2, originScale: 1 },
    { kind: "preset", preset: "diamond-core", decayBase: 2, originScale: 1 },
  ],
  king: [
    { kind: "ring", ringValues: [-1, -1, 1, 1, -1], repeat: true, decayBase: 2, originScale: 1 },
    { kind: "preset", preset: "horizontal-versus-vertical", decayBase: 2, originScale: 1 },
  ],
};

function ring(delta) {
  return Math.max(Math.abs(delta.x), Math.abs(delta.y));
}

function presetSign(preset, delta, r) {
  const absX = Math.abs(delta.x);
  const absY = Math.abs(delta.y);
  if (preset === "checkerboard") return (absX + absY) % 2 === 0 ? 1 : -1;
  if (preset === "horizontal-versus-vertical") return absX >= absY ? 1 : -1;
  if (preset === "diamond-core") return absX + absY <= 2 ? 1 : -1;
  throw new Error(`Unsupported preset: ${preset}`);
}

function evaluateBasis(definition, delta) {
  const r = ring(delta);
  const multiplier = Math.pow(definition.decayBase, -r) * (r === 0 ? definition.originScale : 1);
  if (definition.kind === "preset") return presetSign(definition.preset, delta, r) * multiplier;
  if (definition.kind === "grid") {
    const center = Math.floor(definition.gridValues.length / 2);
    return (definition.gridValues[delta.y + center]?.[delta.x + center] ?? 0) * multiplier;
  }
  const index = definition.repeat ? r % definition.ringValues.length : r;
  return (definition.ringValues[index] ?? 0) * multiplier;
}

function pieceContribution(type, delta) {
  if (delta.x === 0 && delta.y === 0) return DEFAULT_HOME_ENERGY[type];
  let positive = 0;
  let negative = 0;
  const coefficients = ACTIVE_COMPONENTS[type];
  const bases = DEFINITIONS[type];
  for (let index = 0; index < coefficients.length; index += 1) {
    const value = coefficients[index] * evaluateBasis(bases[index], delta);
    if (value > 0) positive += value;
    else if (value < 0) negative += value;
  }
  const scale = DEFAULT_WAVE_SCALES[type];
  return PIECE_STRENGTH[type] * (positive * scale.friendly + negative * scale.hostile);
}

function supportExpandedContribution(type, delta) {
  return pieceContribution(type, { x: Math.round(delta.x / 2), y: Math.round(delta.y / 2) });
}

function sourceContribution(atom, sample) {
  const delta = { x: sample.x - atom.x, y: sample.y - atom.y };
  return atom.level === 2 ? supportExpandedContribution(atom.type, delta) : pieceContribution(atom.type, delta);
}

function offsetsWithin(radius, includeCenter = false) {
  const offsets = [];
  for (let y = -radius; y <= radius; y += 1) {
    for (let x = -radius; x <= radius; x += 1) {
      if (!includeCenter && x === 0 && y === 0) continue;
      offsets.push({ x, y });
    }
  }
  return offsets;
}

function targetVector(type, offsets) {
  return offsets.map((delta) => supportExpandedContribution(type, delta));
}

function fieldVector(config, offsets) {
  return offsets.map((sample) => config.reduce((sum, atom) => sum + sourceContribution(atom, sample), 0));
}

function norm(vector) {
  return Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
}

function dot(left, right) {
  return left.reduce((sum, value, index) => sum + value * right[index], 0);
}

function metrics(field, target) {
  const fieldNorm = norm(field);
  const targetNorm = norm(target);
  const d = dot(field, target);
  const cosine = fieldNorm && targetNorm ? d / (fieldNorm * targetNorm) : 0;
  const alpha = fieldNorm ? d / (fieldNorm * fieldNorm) : 0;
  const rawError = Math.sqrt(field.reduce((sum, value, index) => sum + (value - target[index]) ** 2, 0)) / (targetNorm || 1);
  const scaledError = Math.sqrt(field.reduce((sum, value, index) => sum + (alpha * value - target[index]) ** 2, 0)) / (targetNorm || 1);
  let signs = 0;
  let matches = 0;
  for (let index = 0; index < target.length; index += 1) {
    if (Math.abs(target[index]) < 1e-9) continue;
    signs += 1;
    if (Math.sign(field[index]) === Math.sign(target[index])) matches += 1;
  }
  return { cosine, rawError, scaledError, signAgreement: matches / signs, magnitudeRatio: fieldNorm / targetNorm };
}

function selectivity(config, offsets) {
  const field = fieldVector(config, offsets);
  const ranked = PIECE_TYPES.map((type) => ({
    type,
    target: DISPLAY[type],
    metrics: metrics(field, targetVector(type, offsets)),
  })).sort((a, b) => b.metrics.cosine - a.metrics.cosine);
  const tower = ranked.find((entry) => entry.type === "rook");
  const next = ranked.find((entry) => entry.type !== "rook");
  return {
    rank: ranked.findIndex((entry) => entry.type === "rook") + 1,
    margin: tower.metrics.cosine - next.metrics.cosine,
    ranked: ranked.map((entry) => ({ target: entry.target, cosine: round(entry.metrics.cosine) })),
  };
}

function atomVector(atom, offsets) {
  return offsets.map((sample) => sourceContribution(atom, sample));
}

function add(left, right) {
  return left.map((value, index) => value + right[index]);
}

function render(config) {
  const radius = Math.max(4, ...config.map((atom) => Math.max(Math.abs(atom.x), Math.abs(atom.y))));
  const cells = new Map(config.map((atom) => [`${atom.x},${atom.y}`, `${SHORT[atom.type]}${atom.level}`.padStart(2, " ")]));
  const rows = [];
  for (let y = -radius; y <= radius; y += 1) {
    const row = [];
    for (let x = -radius; x <= radius; x += 1) row.push(x === 0 && y === 0 ? " *" : cells.get(`${x},${y}`) ?? " .");
    rows.push(row.join(" "));
  }
  return rows.join("\n");
}

function score(candidate) {
  const m = candidate.metrics;
  const s = candidate.selectivity;
  return m.cosine + 0.25 * m.signAgreement + 0.35 * Math.max(-0.5, s.margin) - 0.12 * Math.abs(Math.log(Math.max(m.magnitudeRatio, 1e-9)));
}

function configKey(config) {
  return config.map((atom) => `${atom.type}${atom.level}:${atom.x},${atom.y}`).sort().join("|");
}

function dedupe(entries) {
  const seen = new Set();
  const result = [];
  for (const entry of entries) {
    const key = configKey(entry.config);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(entry);
  }
  return result;
}

function compositionLabel(config) {
  const counts = new Map();
  for (const atom of config) {
    const key = `${SHORT[atom.type]}${atom.level}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return [...counts.entries()].sort().map(([key, value]) => `${value} ${key}`).join(" + ");
}

function round(value, digits = 4) {
  return Number(value.toFixed(digits));
}

const args = new Map(process.argv.slice(2).map((arg) => {
  const match = arg.match(/^--([^=]+)=(.*)$/);
  return match ? [match[1], match[2]] : [arg.replace(/^--/, ""), "true"];
}));
const searchRadius = Number(args.get("radius") ?? 3);
const beamWidth = Number(args.get("beam") ?? 350);
const maxPieces = Number(args.get("max-pieces") ?? 6);

const offsets = offsetsWithin(searchRadius + 1, false);
const target = targetVector("rook", offsets);
const positions = offsetsWithin(searchRadius, false);
const atomSpecs = [
  { type: "pawn", level: 2 },
  { type: "rook", level: 1 },
  { type: "pawn", level: 1 },
  { type: "spy", level: 1 },
  { type: "king", level: 1 },
];
const atoms = atomSpecs.flatMap((spec) => positions.map((pos) => ({ ...spec, ...pos })))
  .map((atom) => ({ ...atom, vector: atomVector(atom, offsets) }));

let beam = [{ config: [], occupied: new Set(), field: target.map(() => 0) }];
const allBest = [];
for (let count = 1; count <= maxPieces; count += 1) {
  let next = [];
  for (const entry of beam) {
    for (const atom of atoms) {
      const key = `${atom.x},${atom.y}`;
      if (entry.occupied.has(key)) continue;
      const config = [...entry.config, atom];
      const r2 = config.filter((item) => item.type === "pawn" && item.level === 2).length;
      const t1 = config.filter((item) => item.type === "rook" && item.level === 1).length;
      if (r2 > 2 || t1 > 2) continue;
      const occupied = new Set(entry.occupied);
      occupied.add(key);
      const field = add(entry.field, atom.vector);
      const m = metrics(field, target);
      const s = selectivity(config, offsets);
      const candidate = { config, occupied, field, metrics: m, selectivity: s };
      next.push({ ...candidate, score: score(candidate) });
    }
  }
  next.sort((a, b) => b.score - a.score);
  beam = dedupe(next).slice(0, beamWidth);
  allBest.push(...beam.slice(0, 120));
}

const mixed = dedupe(allBest)
  .filter((entry) => entry.config.some((atom) => atom.type === "pawn" && atom.level === 2))
  .filter((entry) => entry.config.some((atom) => atom.type === "rook" && atom.level === 1))
  .sort((a, b) => b.score - a.score);

function summarize(entry) {
  return {
    label: compositionLabel(entry.config),
    metrics: {
      cosine: round(entry.metrics.cosine),
      signAgreement: round(entry.metrics.signAgreement),
      magnitudeRatio: round(entry.metrics.magnitudeRatio),
      rawError: round(entry.metrics.rawError),
      scaledError: round(entry.metrics.scaledError),
    },
    selectivity: {
      rank: entry.selectivity.rank,
      margin: round(entry.selectivity.margin),
      ranked: entry.selectivity.ranked,
    },
    config: entry.config.map((atom) => ({ piece: DISPLAY[atom.type], type: atom.type, level: atom.level, x: atom.x, y: atom.y })),
    diagram: render(entry.config),
  };
}

const filtered = mixed.filter((entry) => entry.selectivity.rank === 1).slice(0, 30).map(summarize);
const nearMisses = mixed.slice(0, 30).map(summarize);

console.log(JSON.stringify({
  note: "Focused Tower II mixed-scale search. Triangle is diamond-only; Tower II target uses support-expanded scaling. Sources include Round II plus Level-I pieces, with at most two Tower I.",
  target: "Tower II",
  scaling: "support-expanded",
  candidates: filtered,
  nearMisses,
}, null, 2));
