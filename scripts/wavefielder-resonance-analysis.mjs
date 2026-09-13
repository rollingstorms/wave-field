#!/usr/bin/env node

const PIECE_TYPES = ["pawn", "rook", "spy", "king"];
const DISPLAY = { pawn: "Round Hat", rook: "Tower", spy: "Triangle Hat", king: "Big Hat" };
const SHORT = { pawn: "R", rook: "T", spy: "A", king: "B" };
const PIECE_STRENGTH = { pawn: 1, rook: 2, spy: 2, king: 2 };
const DEFAULT_WAVE_SCALES = {
  pawn: { friendly: 4, hostile: 1 },
  rook: { friendly: 3, hostile: 1 },
  spy: { friendly: 3, hostile: 0 },
  king: { friendly: 4, hostile: 2 },
};
let ACTIVE_COMPONENTS = { pawn: [1], rook: [1, 1], spy: [1, 0], king: [1, 1] };
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
    placementRadius: Number(args.get("placement-radius") ?? 3),
    signatureRadius: Number(args.get("signature-radius") ?? 3),
    maxPieces: Number(args.get("max-pieces") ?? 6),
    beamWidth: Number(args.get("beam") ?? 160),
    randomSamples: Number(args.get("random") ?? 600),
    topPerTarget: Number(args.get("top") ?? 12),
    seed: Number(args.get("seed") ?? 20260914),
    triangleMode: args.get("triangle-mode") ?? "current",
    includeBase: args.get("base") !== "false",
    includeScale: args.get("scale") !== "false",
    includeDiagnostic: args.get("diagnostic") !== "false",
  };
}

function applyCounterfactuals(options) {
  if (options.triangleMode === "current") return;
  if (options.triangleMode === "diamond-only") {
    ACTIVE_COMPONENTS = { ...ACTIVE_COMPONENTS, spy: [0, 1] };
    return;
  }
  if (options.triangleMode === "none") {
    ACTIVE_COMPONENTS = { ...ACTIVE_COMPONENTS, spy: [0, 0] };
    return;
  }
  throw new Error(`Unknown triangle-mode: ${options.triangleMode}`);
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
    case "checkerboard": return (absX + absY) % 2 === 0 ? 1 : -1;
    case "horizontal-versus-vertical": return absX >= absY ? 1 : -1;
    case "diamond-core": return absX + absY <= 2 ? 1 : -1;
    default: throw new Error(`Unsupported preset in default search: ${preset}`);
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

function pieceContribution(pieceType, delta) {
  if (delta.x === 0 && delta.y === 0) return DEFAULT_HOME_ENERGY[pieceType];
  const coefficients = ACTIVE_COMPONENTS[pieceType];
  const bases = DEFINITIONS[pieceType];
  let positive = 0;
  let negative = 0;
  for (let index = 0; index < coefficients.length; index += 1) {
    const value = coefficients[index] * evaluateBasis(bases[index], delta);
    if (value > 0) positive += value;
    else if (value < 0) negative += value;
  }
  const scale = DEFAULT_WAVE_SCALES[pieceType];
  return PIECE_STRENGTH[pieceType] * (positive * scale.friendly + negative * scale.hostile);
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

function signatureOffsets(kind, radius) {
  const all = offsetsWithin(radius, false);
  if (kind === "full") return all;
  if (kind === "ring1") return all.filter((p) => ring(p) === 1);
  if (kind === "rings12") return all.filter((p) => ring(p) <= 2);
  if (kind === "axes") return all.filter((p) => p.x === 0 || p.y === 0);
  if (kind === "diagonals") return all.filter((p) => Math.abs(p.x) === Math.abs(p.y));
  throw new Error(`Unknown signature kind: ${kind}`);
}

function targetVector(pieceType, offsets, scaleMode = "base") {
  return offsets.map((delta) => scaledContribution(pieceType, delta, scaleMode));
}

function scaledContribution(pieceType, delta, scaleMode) {
  if (scaleMode === "base") return pieceContribution(pieceType, delta);
  if (scaleMode === "dilate-fixed") {
    return pieceContribution(pieceType, { x: Math.trunc(delta.x / 2), y: Math.trunc(delta.y / 2) });
  }
  if (scaleMode === "dilate-half") {
    return 0.5 * pieceContribution(pieceType, { x: Math.trunc(delta.x / 2), y: Math.trunc(delta.y / 2) });
  }
  if (scaleMode === "support-expanded") {
    return pieceContribution(pieceType, { x: Math.round(delta.x / 2), y: Math.round(delta.y / 2) });
  }
  if (scaleMode === "bilinear-half") {
    return 0.5 * bilinearPieceContribution(pieceType, delta.x / 2, delta.y / 2);
  }
  if (scaleMode === "footprint-2x2") {
    const anchors = [{ x: -0.5, y: -0.5 }, { x: 0.5, y: -0.5 }, { x: -0.5, y: 0.5 }, { x: 0.5, y: 0.5 }];
    return anchors.reduce((sum, anchor) => sum + bilinearPieceContribution(pieceType, delta.x - anchor.x, delta.y - anchor.y), 0) / 4;
  }
  throw new Error(`Unknown scale mode: ${scaleMode}`);
}

function bilinearPieceContribution(pieceType, x, y) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = x - x0;
  const ty = y - y0;
  const v00 = pieceContribution(pieceType, { x: x0, y: y0 });
  const v10 = pieceContribution(pieceType, { x: x0 + 1, y: y0 });
  const v01 = pieceContribution(pieceType, { x: x0, y: y0 + 1 });
  const v11 = pieceContribution(pieceType, { x: x0 + 1, y: y0 + 1 });
  return (1 - ty) * ((1 - tx) * v00 + tx * v10) + ty * ((1 - tx) * v01 + tx * v11);
}

function atomVector(atom, offsets) {
  return offsets.map((sample) => pieceContribution(atom.type, { x: sample.x - atom.x, y: sample.y - atom.y }));
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
  const score = cosine + 0.18 * (signMatches / signCount) - 0.12 * Math.abs(Math.log(Math.max(magnitudeRatio, 1e-9)));
  return { cosine, scaledError, rawError, signAgreement: signMatches / signCount, magnitudeRatio, optimalScaleAlpha: alpha, score };
}

function compositions(maxPieces) {
  const output = [];
  function rec(index, remaining, counts) {
    if (index === PIECE_TYPES.length - 1) {
      counts[PIECE_TYPES[index]] = remaining;
      const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
      if (total >= 2) output.push({ ...counts });
      return;
    }
    for (let count = 0; count <= remaining; count += 1) {
      counts[PIECE_TYPES[index]] = count;
      rec(index + 1, remaining - count, counts);
    }
  }
  for (let total = 2; total <= maxPieces; total += 1) rec(0, total, {});
  return output;
}

function expandComposition(composition) {
  return PIECE_TYPES.flatMap((type) => Array.from({ length: composition[type] ?? 0 }, () => type));
}

function compositionLabel(composition) {
  return PIECE_TYPES
    .filter((type) => composition[type] > 0)
    .map((type) => `${composition[type]} ${SHORT[type]}`)
    .join(" + ");
}

function searchComposition(composition, targetType, options, scaleMode = "base", sigKind = "full") {
  const pieceList = expandComposition(composition);
  const offsets = signatureOffsets(sigKind, scaleMode === "base" ? options.signatureRadius : options.signatureRadius + 1);
  const target = targetVector(targetType, offsets, scaleMode);
  const positions = offsetsWithin(scaleMode === "base" ? options.placementRadius : options.placementRadius + 1, false);
  const vectorsByType = new Map();
  for (const type of PIECE_TYPES) {
    vectorsByType.set(type, positions.map((pos) => ({ ...pos, type, vector: atomVector({ ...pos, type }, offsets) })));
  }
  let beam = [{ config: [], occupied: new Set(), field: target.map(() => 0), metrics: metrics(target.map(() => 0), target) }];
  for (const type of pieceList) {
    let next = [];
    for (const entry of beam) {
      for (const atom of vectorsByType.get(type)) {
        const key = `${atom.x},${atom.y}`;
        if (entry.occupied.has(key)) continue;
        const occupied = new Set(entry.occupied);
        occupied.add(key);
        const field = add(entry.field, atom.vector);
        const config = [...entry.config, { type, x: atom.x, y: atom.y }];
        next.push({ config, occupied, field, metrics: metrics(field, target) });
      }
    }
    next.sort((a, b) => b.metrics.score - a.metrics.score);
    beam = dedupe(next).slice(0, options.beamWidth);
  }
  const best = beam[0];
  const random = randomBaseline(composition, targetType, options, target, offsets, scaleMode);
  return annotateCandidate({ composition, targetType, scaleMode, sigKind, ...best }, target, offsets, options, random);
}

function dedupe(entries) {
  const seen = new Set();
  const result = [];
  for (const entry of entries) {
    const key = entry.config.map((atom) => `${atom.type}:${atom.x},${atom.y}`).sort().join("|");
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(entry);
  }
  return result;
}

function randomBaseline(composition, targetType, options, target, offsets, scaleMode) {
  const rand = mulberry32(options.seed + hashString(`${compositionLabel(composition)}:${targetType}:${scaleMode}`));
  const pieceList = expandComposition(composition);
  const positions = offsetsWithin(scaleMode === "base" ? options.placementRadius : options.placementRadius + 1, false);
  const scores = [];
  for (let sample = 0; sample < options.randomSamples; sample += 1) {
    const occupied = new Set();
    let field = target.map(() => 0);
    for (const type of pieceList) {
      let pos;
      do {
        pos = positions[Math.floor(rand() * positions.length)];
      } while (occupied.has(`${pos.x},${pos.y}`));
      occupied.add(`${pos.x},${pos.y}`);
      field = add(field, atomVector({ type, ...pos }, offsets));
    }
    scores.push(metrics(field, target).cosine);
  }
  scores.sort((a, b) => a - b);
  return {
    p50: quantile(scores, 0.5),
    p90: quantile(scores, 0.9),
    p95: quantile(scores, 0.95),
    p99: quantile(scores, 0.99),
  };
}

function annotateCandidate(candidate, target, offsets, options, random) {
  const cosineScores = [random.p50, random.p90, random.p95, random.p99];
  const percentile = candidate.metrics.cosine >= cosineScores[3] ? 0.99
    : candidate.metrics.cosine >= cosineScores[2] ? 0.95
      : candidate.metrics.cosine >= cosineScores[1] ? 0.90
        : candidate.metrics.cosine >= cosineScores[0] ? 0.50
          : 0;
  const centerValue = fieldAt(candidate.config, { x: 0, y: 0 });
  const neighbors = offsetsWithin(1, false).map((pos) => fieldAt(candidate.config, pos));
  const localAbsMax = Math.abs(centerValue) >= Math.max(...neighbors.map(Math.abs));
  const localMax = centerValue >= Math.max(...neighbors);
  const localMin = centerValue <= Math.min(...neighbors);
  const robustness = perturbationRobustness(candidate.config, target, offsets, options);
  const selectivity = targetSelectivity(candidate.config, offsets, candidate.scaleMode, candidate.targetType);
  return {
    ...candidate,
    random,
    percentile,
    localExtremum: { centerValue, localAbsMax, localMax, localMin },
    robustness,
    selectivity,
    symmetry: symmetryScore(candidate.config),
  };
}

function targetSelectivity(config, offsets, scaleMode, intendedTargetType) {
  const field = fieldVector(config, offsets);
  const ranked = PIECE_TYPES.map((targetType) => ({
    target: DISPLAY[targetType],
    type: targetType,
    cosine: metrics(field, targetVector(targetType, offsets, scaleMode)).cosine,
  })).sort((a, b) => b.cosine - a.cosine);
  const intendedRank = ranked.findIndex((entry) => entry.type === intendedTargetType) + 1;
  const intended = ranked.find((entry) => entry.type === intendedTargetType);
  const bestOther = ranked.find((entry) => entry.type !== intendedTargetType);
  return {
    intendedRank,
    intendedCosine: intended?.cosine ?? 0,
    bestOtherTarget: bestOther?.target ?? null,
    bestOtherCosine: bestOther?.cosine ?? 0,
    margin: (intended?.cosine ?? 0) - (bestOther?.cosine ?? 0),
    ranked: ranked.map((entry) => ({ target: entry.target, cosine: round(entry.cosine) })),
  };
}

function fieldAt(config, point) {
  return config.reduce((sum, atom) => sum + pieceContribution(atom.type, { x: point.x - atom.x, y: point.y - atom.y }), 0);
}

function perturbationRobustness(config, target, offsets, options) {
  const baseField = fieldVector(config, offsets);
  const base = metrics(baseField, target).cosine;
  const values = [];
  const occupiedBase = new Set(config.map((atom) => `${atom.x},${atom.y}`));
  const steps = [{ x: 1, y: 0 }, { x: -1, y: 0 }, { x: 0, y: 1 }, { x: 0, y: -1 }];
  for (let index = 0; index < config.length; index += 1) {
    for (const step of steps) {
      const moved = { ...config[index], x: config[index].x + step.x, y: config[index].y + step.y };
      if (moved.x === 0 && moved.y === 0) continue;
      if (Math.max(Math.abs(moved.x), Math.abs(moved.y)) > options.placementRadius + 1) continue;
      const oldKey = `${config[index].x},${config[index].y}`;
      const newKey = `${moved.x},${moved.y}`;
      if (occupiedBase.has(newKey) && newKey !== oldKey) continue;
      const next = config.slice();
      next[index] = moved;
      values.push(metrics(fieldVector(next, offsets), target).cosine);
    }
  }
  values.sort((a, b) => a - b);
  return {
    baseCosine: base,
    minPerturbedCosine: values[0] ?? base,
    meanPerturbedCosine: values.reduce((sum, value) => sum + value, 0) / (values.length || 1),
    samples: values.length,
  };
}

function fieldVector(config, offsets) {
  return offsets.map((sample) => fieldAt(config, sample));
}

function symmetryScore(config) {
  const keys = new Set(config.map((atom) => `${atom.type}:${atom.x},${atom.y}`));
  const transforms = [
    (a) => `${a.type}:${-a.x},${a.y}`,
    (a) => `${a.type}:${a.x},${-a.y}`,
    (a) => `${a.type}:${-a.x},${-a.y}`,
    (a) => `${a.type}:${a.y},${a.x}`,
  ];
  const scores = transforms.map((transform) => config.filter((atom) => keys.has(transform(atom))).length / config.length);
  return Math.max(...scores);
}

function allResonances(options) {
  const comps = compositions(options.maxPieces);
  const all = [];
  for (const composition of comps) {
    for (const targetType of PIECE_TYPES) {
      all.push(searchComposition(composition, targetType, options));
    }
  }
  return all;
}

function pareto(candidates) {
  return candidates.filter((candidate) => !candidates.some((other) => (
    other !== candidate
    && other.config.length <= candidate.config.length
    && other.metrics.cosine >= candidate.metrics.cosine
    && other.metrics.signAgreement >= candidate.metrics.signAgreement
    && other.robustness.minPerturbedCosine >= candidate.robustness.minPerturbedCosine
    && (
      other.config.length < candidate.config.length
      || other.metrics.cosine > candidate.metrics.cosine
      || other.metrics.signAgreement > candidate.metrics.signAgreement
      || other.robustness.minPerturbedCosine > candidate.robustness.minPerturbedCosine
    )
  )));
}

function diagnosticAnalysis(options) {
  const kinds = ["ring1", "rings12", "axes", "diagonals", "full"];
  return kinds.map((kind) => {
    const offsets = signatureOffsets(kind, options.signatureRadius);
    const vectors = Object.fromEntries(PIECE_TYPES.map((type) => [type, targetVector(type, offsets)]));
    const pairwise = [];
    for (const a of PIECE_TYPES) {
      for (const b of PIECE_TYPES) {
        if (a >= b) continue;
        pairwise.push({ a: DISPLAY[a], b: DISPLAY[b], cosine: round(metrics(vectors[a], vectors[b]).cosine) });
      }
    }
    const maxConfusion = Math.max(...pairwise.map((entry) => Math.abs(entry.cosine)));
    return { kind, sampleCount: offsets.length, maxAbsPairwiseCosine: round(maxConfusion), pairwise };
  });
}

function scaleAnalysis(options) {
  const modes = ["dilate-fixed", "dilate-half", "support-expanded", "bilinear-half", "footprint-2x2"];
  const smallOptions = { ...options, maxPieces: Math.min(5, options.maxPieces), beamWidth: Math.min(120, options.beamWidth), randomSamples: Math.min(250, options.randomSamples) };
  const comps = compositions(smallOptions.maxPieces);
  return modes.map((mode) => {
    const byTarget = PIECE_TYPES.map((targetType) => {
      const candidates = comps.map((composition) => searchComposition(composition, targetType, smallOptions, mode));
      candidates.sort((a, b) => b.metrics.score - a.metrics.score);
      return summarizeCandidate(candidates[0]);
    });
    return { mode, byTarget };
  });
}

function resonanceGraph(candidates) {
  const edges = [];
  for (const source of PIECE_TYPES) {
    for (const target of PIECE_TYPES) {
      const relevant = candidates.filter((candidate) => candidate.composition[source] > 0 && candidate.targetType === target);
      relevant.sort((a, b) => b.metrics.score - a.metrics.score);
      const best = relevant[0];
      if (best) edges.push({
        source: DISPLAY[source],
        target: DISPLAY[target],
        bestCosine: round(best.metrics.cosine),
        signAgreement: round(best.metrics.signAgreement),
        pieces: best.config.length,
        composition: compositionLabel(best.composition),
      });
    }
  }
  return edges;
}

function bootstrapSets(candidates) {
  const strongEdges = candidates.filter((candidate) => (
    candidate.metrics.cosine >= 0.75
    && candidate.metrics.signAgreement >= 0.85
    && candidate.metrics.magnitudeRatio >= 0.5
    && candidate.metrics.magnitudeRatio <= 1.8
  ));
  const seedSets = compositions(4).filter((composition) => Object.values(composition).reduce((sum, value) => sum + value, 0) <= 4);
  return seedSets.map((seed) => {
    const reachable = new Set(PIECE_TYPES.filter((type) => seed[type] > 0));
    let changed = true;
    while (changed) {
      changed = false;
      for (const edge of strongEdges) {
        const available = PIECE_TYPES.every((type) => (edge.composition[type] ?? 0) === 0 || reachable.has(type));
        if (available && !reachable.has(edge.targetType)) {
          reachable.add(edge.targetType);
          changed = true;
        }
      }
    }
    return {
      seed: compositionLabel(seed),
      reachable: [...reachable].map((type) => DISPLAY[type]),
      unreachable: PIECE_TYPES.filter((type) => !reachable.has(type)).map((type) => DISPLAY[type]),
    };
  }).filter((entry) => entry.unreachable.length < 4);
}

function summarizeCandidate(candidate) {
  return {
    sources: compositionLabel(candidate.composition),
    target: DISPLAY[candidate.targetType],
    scaleMode: candidate.scaleMode,
    pieces: candidate.config.length,
    metrics: roundMetrics(candidate.metrics),
    randomPercentileAtLeast: candidate.percentile,
    random: roundRandom(candidate.random),
    localExtremum: {
      centerValue: round(candidate.localExtremum.centerValue),
      localAbsMax: candidate.localExtremum.localAbsMax,
      localMax: candidate.localExtremum.localMax,
      localMin: candidate.localExtremum.localMin,
    },
    robustness: {
      minPerturbedCosine: round(candidate.robustness.minPerturbedCosine),
      meanPerturbedCosine: round(candidate.robustness.meanPerturbedCosine),
      samples: candidate.robustness.samples,
    },
    selectivity: {
      intendedRank: candidate.selectivity.intendedRank,
      bestOtherTarget: candidate.selectivity.bestOtherTarget,
      bestOtherCosine: round(candidate.selectivity.bestOtherCosine),
      margin: round(candidate.selectivity.margin),
      ranked: candidate.selectivity.ranked,
    },
    symmetryScore: round(candidate.symmetry),
    config: candidate.config.map((atom) => ({ piece: DISPLAY[atom.type], type: atom.type, x: atom.x, y: atom.y })),
    diagram: renderConfig(candidate.config),
  };
}

function renderConfig(config) {
  const radius = Math.max(2, ...config.map((atom) => Math.max(Math.abs(atom.x), Math.abs(atom.y))));
  const cells = new Map(config.map((atom) => [`${atom.x},${atom.y}`, SHORT[atom.type].padStart(2, " ")]));
  const rows = [];
  for (let y = -radius; y <= radius; y += 1) {
    const row = [];
    for (let x = -radius; x <= radius; x += 1) row.push(x === 0 && y === 0 ? " *" : cells.get(`${x},${y}`) ?? " .");
    rows.push(row.join(" "));
  }
  return rows.join("\n");
}

function mulberry32(seed) {
  return function rand() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function hashString(value) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) hash = Math.imul(31, hash) + value.charCodeAt(index) | 0;
  return hash;
}

function quantile(sorted, q) {
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
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
    optimalScaleAlpha: round(m.optimalScaleAlpha),
    score: round(m.score),
  };
}

function roundRandom(random) {
  return { p50: round(random.p50), p90: round(random.p90), p95: round(random.p95), p99: round(random.p99) };
}

const options = parseArgs();
applyCounterfactuals(options);
const candidates = options.includeBase ? allResonances(options) : [];
const byTarget = Object.fromEntries(PIECE_TYPES.map((targetType) => {
  const list = candidates.filter((candidate) => candidate.targetType === targetType)
    .sort((a, b) => b.metrics.score - a.metrics.score)
    .slice(0, options.topPerTarget)
    .map(summarizeCandidate);
  return [DISPLAY[targetType], list];
}));
const byComposition = candidates
  .filter((candidate) => candidate.metrics.cosine >= 0.5 || candidate.percentile >= 0.95)
  .sort((a, b) => b.metrics.score - a.metrics.score)
  .slice(0, 80)
  .map(summarizeCandidate);
const paretoCandidates = pareto(candidates)
  .sort((a, b) => a.config.length - b.config.length || b.metrics.score - a.metrics.score)
  .slice(0, 40)
  .map(summarizeCandidate);

console.log(JSON.stringify({
  generatedAt: new Date().toISOString(),
  note: "Experimental same-polarity Wavefielder resonance analysis. It mirrors current production kernels and does not alter game rules.",
  options,
  topByTarget: byTarget,
  resonanceMatrix: byComposition,
  paretoCandidates,
  graph: resonanceGraph(candidates),
  bootstrapSets: bootstrapSets(candidates),
  diagnosticSignatures: options.includeDiagnostic ? diagnosticAnalysis(options) : [],
  scaleIIHypotheses: options.includeScale ? scaleAnalysis(options) : [],
}, null, 2));
