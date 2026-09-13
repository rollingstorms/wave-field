#!/usr/bin/env node

const PIECE_TYPES = ["pawn", "rook", "spy", "king"];
const DISPLAY = {
  pawn: "Round Hat",
  rook: "Tower",
  spy: "Triangle Hat",
  king: "Big Hat",
};

const PIECE_STRENGTH = { pawn: 1, rook: 2, spy: 2, king: 2 };
const DEFAULT_WAVE_SCALES = {
  pawn: { friendly: 4, hostile: 1 },
  rook: { friendly: 3, hostile: 1 },
  spy: { friendly: 3, hostile: 0 },
  king: { friendly: 4, hostile: 2 },
};
const DEFAULT_COMPONENTS = {
  pawn: [1],
  rook: [1, 1],
  spy: [1, 0],
  king: [1, 1],
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
  pawn: [{ kind: "preset", name: "Checkerboard", preset: "checkerboard", decayBase: 2, originScale: 1 }],
  rook: [
    { kind: "grid", name: "Tower grid", gridValues: towerComponentOneGrid, decayBase: 2, originScale: 1 },
    { kind: "ring", name: "Pull gap push", geometry: "chebyshev", ringValues: [0, 1, 0, -1], repeat: true, decayBase: 2, originScale: 1 },
  ],
  spy: [
    { kind: "preset", name: "Round Hat mask", preset: "checkerboard", decayBase: 2, originScale: 1 },
    { kind: "preset", name: "Diamond core", preset: "diamond-core", decayBase: 2, originScale: 1 },
  ],
  king: [
    { kind: "ring", name: "Big Hat c1 rings", geometry: "chebyshev", ringValues: [-1, -1, 1, 1, -1], repeat: true, decayBase: 2, originScale: 1 },
    { kind: "preset", name: "Horizontal mode", preset: "horizontal-versus-vertical", decayBase: 2, originScale: 1 },
  ],
};

function parseArgs() {
  const args = new Map();
  for (const arg of process.argv.slice(2)) {
    const match = arg.match(/^--([^=]+)=(.*)$/);
    if (match) args.set(match[1], match[2]);
    else if (arg.startsWith("--")) args.set(arg.slice(2), "true");
  }
  return {
    placementRadius: Number(args.get("placement-radius") ?? 4),
    signatureRadius: Number(args.get("signature-radius") ?? 3),
    maxPieces: Number(args.get("max-pieces") ?? 6),
    beamWidth: Number(args.get("beam") ?? 2500),
    randomSamples: Number(args.get("random") ?? 20000),
    seed: Number(args.get("seed") ?? 20260913),
    signed: args.get("signed") !== "false",
    includeSame: args.get("include-same") === "true",
    scaled: args.get("scaled") !== "false",
  };
}

function ring(delta) {
  return Math.max(Math.abs(delta.x), Math.abs(delta.y));
}

function decay(r, decayBase, originScale) {
  return Math.pow(decayBase, -r) * (r === 0 ? originScale : 1);
}

function presetSign(preset, delta, r) {
  const absX = Math.abs(delta.x);
  const absY = Math.abs(delta.y);
  switch (preset) {
    case "checkerboard":
      return (absX + absY) % 2 === 0 ? 1 : -1;
    case "horizontal-versus-vertical":
      return absX >= absY ? 1 : -1;
    case "diamond-core":
      return absX + absY <= 2 ? 1 : -1;
    default:
      throw new Error(`Preset ${preset} is not used by the default piece set in this search.`);
  }
}

function evaluateBasis(definition, delta) {
  const r = ring(delta);
  const multiplier = decay(r, definition.decayBase, definition.originScale);
  if (definition.kind === "preset") return presetSign(definition.preset, delta, r) * multiplier;
  if (definition.kind === "grid") {
    const center = Math.floor(definition.gridValues.length / 2);
    return (definition.gridValues[delta.y + center]?.[delta.x + center] ?? 0) * multiplier;
  }
  const index = definition.repeat ? r % definition.ringValues.length : r;
  return (definition.ringValues[index] ?? 0) * multiplier;
}

function pieceContribution(pieceType, delta, ownerSign = 1) {
  if (delta.x === 0 && delta.y === 0) return ownerSign * DEFAULT_HOME_ENERGY[pieceType];
  const coefficients = DEFAULT_COMPONENTS[pieceType];
  const bases = DEFINITIONS[pieceType];
  let positive = 0;
  let negative = 0;
  for (let index = 0; index < coefficients.length; index += 1) {
    const value = coefficients[index] * evaluateBasis(bases[index], delta);
    if (value > 0) positive += value;
    else if (value < 0) negative += value;
  }
  const scale = DEFAULT_WAVE_SCALES[pieceType];
  return ownerSign * PIECE_STRENGTH[pieceType] * (positive * scale.friendly + negative * scale.hostile);
}

function scaledPieceContribution(pieceType, delta, scale) {
  if (delta.x === 0 && delta.y === 0) return DEFAULT_HOME_ENERGY[pieceType] * scale;
  const coarse = {
    x: Math.trunc(delta.x / scale),
    y: Math.trunc(delta.y / scale),
  };
  return pieceContribution(pieceType, coarse, 1) / scale;
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

function vectorForTarget(pieceType, signatureOffsets, scale = 1) {
  return signatureOffsets.map((delta) => (
    scale === 1
      ? pieceContribution(pieceType, delta, 1)
      : scaledPieceContribution(pieceType, delta, scale)
  ));
}

function atomVector(atom, signatureOffsets) {
  return signatureOffsets.map((sample) => {
    const deltaFromAtom = { x: sample.x - atom.x, y: sample.y - atom.y };
    return pieceContribution(atom.type, deltaFromAtom, atom.sign);
  });
}

function add(left, right) {
  return left.map((value, index) => value + right[index]);
}

function norm(vector) {
  return Math.sqrt(vector.reduce((total, value) => total + value * value, 0));
}

function dot(left, right) {
  return left.reduce((total, value, index) => total + value * right[index], 0);
}

function metrics(field, target) {
  const fieldNorm = norm(field);
  const targetNorm = norm(target);
  const d = dot(field, target);
  const cosine = fieldNorm > 0 && targetNorm > 0 ? d / (fieldNorm * targetNorm) : 0;
  const alpha = fieldNorm > 0 ? d / (fieldNorm * fieldNorm) : 0;
  const scaledError = Math.sqrt(field.reduce((total, value, index) => {
    const error = alpha * value - target[index];
    return total + error * error;
  }, 0)) / (targetNorm || 1);
  const rawError = Math.sqrt(field.reduce((total, value, index) => {
    const error = value - target[index];
    return total + error * error;
  }, 0)) / (targetNorm || 1);
  let signMatches = 0;
  let signCount = 0;
  for (let index = 0; index < target.length; index += 1) {
    if (Math.abs(target[index]) < 1e-9) continue;
    signCount += 1;
    if (Math.sign(field[index]) === Math.sign(target[index])) signMatches += 1;
  }
  const magnitudeRatio = targetNorm > 0 ? fieldNorm / targetNorm : 0;
  const score = cosine - 0.12 * Math.abs(Math.log(Math.max(magnitudeRatio, 1e-9)));
  return { cosine, alpha, scaledError, rawError, signAgreement: signMatches / signCount, magnitudeRatio, score };
}

function atomKey(atom) {
  return `${atom.x},${atom.y}`;
}

function configKey(config) {
  return config.map((atom) => `${atom.type}:${atom.sign}:${atom.x},${atom.y}`).sort().join("|");
}

function makeAtoms({ placementRadius, signed, types }) {
  const atoms = [];
  const signs = signed ? [1, -1] : [1];
  for (const pos of offsetsWithin(placementRadius, false)) {
    for (const type of types) {
      for (const sign of signs) atoms.push({ type, sign, x: pos.x, y: pos.y });
    }
  }
  return atoms;
}

function searchTarget(targetType, options, targetScale = 1) {
  const signatureRadius = options.signatureRadius * targetScale;
  const signatureOffsets = offsetsWithin(signatureRadius, false);
  const target = vectorForTarget(targetType, signatureOffsets, targetScale);
  const types = options.includeSame ? PIECE_TYPES : PIECE_TYPES.filter((type) => type !== targetType);
  const atoms = makeAtoms({ placementRadius: options.placementRadius * targetScale, signed: options.signed, types })
    .map((atom) => ({ ...atom, vector: atomVector(atom, signatureOffsets) }));
  let beam = [{ config: [], occupied: new Set(), field: target.map(() => 0), metrics: metrics(target.map(() => 0), target) }];
  const bestByCount = [];
  for (let count = 1; count <= options.maxPieces; count += 1) {
    let next = [];
    let cutoff = -Infinity;
    for (const entry of beam) {
      for (const atom of atoms) {
        const key = atomKey(atom);
        if (entry.occupied.has(key)) continue;
        const config = [...entry.config, atom];
        const field = add(entry.field, atom.vector);
        const occupied = new Set(entry.occupied);
        occupied.add(key);
        const candidate = { config, occupied, field, metrics: metrics(field, target) };
        if (candidate.metrics.score < cutoff) continue;
        next.push(candidate);
        if (next.length > options.beamWidth * 8) {
          next.sort((a, b) => b.metrics.score - a.metrics.score);
          next = dedupeConfigs(next).slice(0, options.beamWidth);
          cutoff = next.at(-1)?.metrics.score ?? -Infinity;
        }
      }
    }
    next.sort((a, b) => b.metrics.score - a.metrics.score);
    beam = dedupeConfigs(next).slice(0, options.beamWidth);
    bestByCount.push(beam[0]);
  }
  return { targetType, targetScale, signatureOffsets, target, bestByCount };
}

function dedupeConfigs(entries) {
  const seen = new Set();
  const deduped = [];
  for (const entry of entries) {
    const key = configKey(entry.config);
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(entry);
  }
  return deduped;
}

function mulberry32(seed) {
  return function rand() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function randomBaseline(targetType, options, targetScale = 1) {
  const rand = mulberry32(options.seed + targetScale * 101 + PIECE_TYPES.indexOf(targetType) * 997);
  const signatureOffsets = offsetsWithin(options.signatureRadius * targetScale, false);
  const target = vectorForTarget(targetType, signatureOffsets, targetScale);
  const types = options.includeSame ? PIECE_TYPES : PIECE_TYPES.filter((type) => type !== targetType);
  const atoms = makeAtoms({ placementRadius: options.placementRadius * targetScale, signed: options.signed, types })
    .map((atom) => ({ ...atom, vector: atomVector(atom, signatureOffsets) }));
  const byCount = [];
  for (let count = 2; count <= options.maxPieces; count += 1) {
    const scores = [];
    let best = null;
    for (let sample = 0; sample < options.randomSamples; sample += 1) {
      const occupied = new Set();
      let field = target.map(() => 0);
      const config = [];
      while (config.length < count) {
        const atom = atoms[Math.floor(rand() * atoms.length)];
        const key = atomKey(atom);
        if (occupied.has(key)) continue;
        occupied.add(key);
        config.push(atom);
        field = add(field, atom.vector);
      }
      const m = metrics(field, target);
      scores.push(m.cosine);
      if (!best || m.score > best.metrics.score) best = { config, field, metrics: m };
    }
    scores.sort((a, b) => a - b);
    byCount.push({
      count,
      p50: quantile(scores, 0.5),
      p90: quantile(scores, 0.9),
      p99: quantile(scores, 0.99),
      over90: scores.filter((score) => score >= 0.9).length / scores.length,
      over95: scores.filter((score) => score >= 0.95).length / scores.length,
      best,
    });
  }
  return byCount;
}

function quantile(sorted, q) {
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}

function compactAtom(atom) {
  return { piece: DISPLAY[atom.type], type: atom.type, ownerSign: atom.sign, x: atom.x, y: atom.y };
}

function renderConfig(config, radius) {
  const cells = new Map();
  for (const atom of config) {
    const label = `${atom.sign > 0 ? "" : "-"}${DISPLAY[atom.type][0]}`;
    cells.set(`${atom.x},${atom.y}`, label.padStart(2, " "));
  }
  const rows = [];
  for (let y = -radius; y <= radius; y += 1) {
    const row = [];
    for (let x = -radius; x <= radius; x += 1) {
      if (x === 0 && y === 0) row.push(" *");
      else row.push(cells.get(`${x},${y}`) ?? " .");
    }
    rows.push(row.join(" "));
  }
  return rows.join("\n");
}

function summarizeResult(result, random) {
  return {
    targetPiece: DISPLAY[result.targetType],
    targetType: result.targetType,
    targetScale: result.targetScale,
    bestByPieceCount: result.bestByCount.map((entry, index) => ({
      pieceCount: index + 1,
      metrics: roundMetrics(entry.metrics),
      config: entry.config.map(compactAtom),
      diagram: renderConfig(entry.config, Math.max(...entry.config.map((atom) => Math.max(Math.abs(atom.x), Math.abs(atom.y))), 1)),
    })),
    randomBaseline: random.map((entry) => ({
      pieceCount: entry.count,
      cosineP50: round(entry.p50),
      cosineP90: round(entry.p90),
      cosineP99: round(entry.p99),
      accidentalRateCosineAtLeast90: round(entry.over90, 5),
      accidentalRateCosineAtLeast95: round(entry.over95, 5),
      bestRandom: {
        metrics: roundMetrics(entry.best.metrics),
        config: entry.best.config.map(compactAtom),
      },
    })),
  };
}

function round(value, digits = 4) {
  return Number(value.toFixed(digits));
}

function roundMetrics(m) {
  return {
    cosine: round(m.cosine),
    scaledError: round(m.scaledError),
    rawError: round(m.rawError),
    signAgreement: round(m.signAgreement),
    magnitudeRatio: round(m.magnitudeRatio),
    optimalScaleAlpha: round(m.alpha),
    score: round(m.score),
  };
}

const options = parseArgs();
const scaleOne = PIECE_TYPES.map((targetType) => {
  const result = searchTarget(targetType, options, 1);
  const random = randomBaseline(targetType, options, 1);
  return summarizeResult(result, random);
});
const scaleTwo = options.scaled
  ? PIECE_TYPES.map((targetType) => {
    const result = searchTarget(targetType, { ...options, maxPieces: Math.min(options.maxPieces, 6) }, 2);
    const random = randomBaseline(targetType, { ...options, randomSamples: Math.max(2000, Math.floor(options.randomSamples / 4)) }, 2);
    return summarizeResult(result, random);
  })
  : [];

const output = {
  generatedAt: new Date().toISOString(),
  note: "Experimental Wavefielder analysis. This script mirrors the current production default kernels and superposition formula; it does not alter game rules.",
  options,
  signatureDefinition: {
    scaleI: "Vector of signed production contribution values from a single red target piece over a Chebyshev-local window around the target, excluding the origin.",
    comparison: "Cosine similarity for shape, normalized RMSE after optimal scalar fit for shape error, raw normalized RMSE for actual magnitude, sign agreement, and field-norm magnitude ratio.",
    scaleIIExtension: "Reasonable extension only: spatial dilation by integer scale using coarse coordinates trunc(delta / scale), with amplitude divided by scale. This is not implemented in normal Wave Field.",
  },
  scaleI: scaleOne,
  scaleII: scaleTwo,
};

console.log(JSON.stringify(output, null, 2));
