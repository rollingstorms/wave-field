import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import type { PieceType, Position } from "../game/types";

const BOARD_SIZE = 50;
const CENTER = { x: 25, y: 25 };

type Scale = 1 | 2;

interface PatternPiece {
  type: PieceType;
  level: Scale;
  x: number;
  y: number;
}

interface Pattern {
  id: string;
  name: string;
  target: PieceType;
  targetLevel: Scale;
  pieces: PatternPiece[];
  metrics: {
    cosine: number;
    sign: number;
    magnitude: number;
    selectivity: number;
  };
  note: string;
}

const labels: Record<PieceType, string> = {
  pawn: "Round Hat",
  rook: "Tower",
  spy: "Triangle Hat",
  king: "Big Hat",
};

const initials: Record<PieceType, string> = {
  pawn: "R",
  rook: "T",
  spy: "A",
  king: "B",
};

const strengths: Record<PieceType, number> = {
  pawn: 1,
  rook: 2,
  spy: 2,
  king: 2,
};

const waveScales: Record<PieceType, { friendly: number; hostile: number }> = {
  pawn: { friendly: 4, hostile: 1 },
  rook: { friendly: 3, hostile: 1 },
  spy: { friendly: 3, hostile: 0 },
  king: { friendly: 4, hostile: 2 },
};

const homeEnergy: Record<PieceType, number> = {
  pawn: 0,
  rook: 0,
  spy: 0.5,
  king: 0,
};

const activeComponents: Record<PieceType, number[]> = {
  pawn: [1],
  rook: [1, 1],
  spy: [0, 1],
  king: [1, 1],
};

const towerGrid = [
  [1, 1, 1, -1, 1, 1, 1],
  [1, 1, -1, 0, -1, 1, 1],
  [1, -1, 0, -1, 0, -1, 1],
  [-1, 0, -1, 0, -1, 0, -1],
  [1, -1, 0, -1, 0, -1, 0],
  [1, 1, -1, 0, -1, 1, 1],
  [1, 1, 1, -1, 1, 1, 1],
];

type Basis =
  | { kind: "preset"; preset: "checkerboard" | "diamond-core" | "horizontal-versus-vertical"; decayBase: number }
  | { kind: "ring"; ringValues: number[]; repeat: boolean; decayBase: number }
  | { kind: "grid"; gridValues: number[][]; decayBase: number };

const definitions: Record<PieceType, Basis[]> = {
  pawn: [{ kind: "preset", preset: "checkerboard", decayBase: 2 }],
  rook: [
    { kind: "grid", gridValues: towerGrid, decayBase: 2 },
    { kind: "ring", ringValues: [0, 1, 0, -1], repeat: true, decayBase: 2 },
  ],
  spy: [
    { kind: "preset", preset: "checkerboard", decayBase: 2 },
    { kind: "preset", preset: "diamond-core", decayBase: 2 },
  ],
  king: [
    { kind: "ring", ringValues: [-1, -1, 1, 1, -1], repeat: true, decayBase: 2 },
    { kind: "preset", preset: "horizontal-versus-vertical", decayBase: 2 },
  ],
};

const patterns: Pattern[] = [
  {
    id: "round-1",
    name: "Round I",
    target: "pawn",
    targetLevel: 1,
    pieces: [
      { type: "pawn", level: 1, x: 0, y: -2 },
      { type: "pawn", level: 1, x: 0, y: 2 },
    ],
    metrics: { cosine: 0.8991, sign: 1, magnitude: 1.5312, selectivity: 0.2895 },
    note: "Cheap self-growth. This is the runaway-risk primitive.",
  },
  {
    id: "triangle-1",
    name: "Triangle I",
    target: "spy",
    targetLevel: 1,
    pieces: [
      { type: "pawn", level: 1, x: 0, y: -2 },
      { type: "spy", level: 1, x: 0, y: -1 },
      { type: "spy", level: 1, x: 0, y: 1 },
    ],
    metrics: { cosine: 0.8097, sign: 1, magnitude: 1.6468, selectivity: 0.1733 },
    note: "Balanced Triangle key after removing Triangle's Round-like component.",
  },
  {
    id: "tower-1",
    name: "Tower I",
    target: "rook",
    targetLevel: 1,
    pieces: [
      { type: "pawn", level: 1, x: -2, y: 0 },
      { type: "pawn", level: 1, x: 0, y: 2 },
      { type: "pawn", level: 1, x: -1, y: 2 },
      { type: "rook", level: 1, x: 3, y: -3 },
      { type: "rook", level: 1, x: -3, y: -3 },
    ],
    metrics: { cosine: 0.7475, sign: 0.8636, magnitude: 1.4232, selectivity: 0.0535 },
    note: "Weakest Level-I key, but Tower ranks first under the current diagnostic.",
  },
  {
    id: "big-1",
    name: "Big I",
    target: "king",
    targetLevel: 1,
    pieces: [
      { type: "pawn", level: 1, x: -1, y: 1 },
      { type: "pawn", level: 1, x: 0, y: 1 },
      { type: "pawn", level: 1, x: 1, y: -1 },
      { type: "rook", level: 1, x: -3, y: -2 },
      { type: "spy", level: 1, x: 3, y: 1 },
      { type: "spy", level: 1, x: -3, y: 1 },
    ],
    metrics: { cosine: 0.8476, sign: 0.9583, magnitude: 1.036, selectivity: 0.3429 },
    note: "Good high-order Level-I pattern with near-perfect magnitude.",
  },
  {
    id: "round-2",
    name: "Round II",
    target: "pawn",
    targetLevel: 2,
    pieces: [
      { type: "rook", level: 1, x: -1, y: 4 },
      { type: "rook", level: 1, x: 4, y: 4 },
      { type: "spy", level: 1, x: -4, y: 2 },
      { type: "king", level: 1, x: -1, y: -4 },
    ],
    metrics: { cosine: 0.5833, sign: 0.6104, magnitude: 1.7962, selectivity: 0.0852 },
    note: "Weak but unique-ish support-expanded Level-II seed.",
  },
  {
    id: "triangle-2",
    name: "Triangle II",
    target: "spy",
    targetLevel: 2,
    pieces: [
      { type: "pawn", level: 1, x: -1, y: -1 },
      { type: "pawn", level: 1, x: -1, y: 0 },
      { type: "king", level: 1, x: 0, y: -1 },
      { type: "spy", level: 1, x: 0, y: 2 },
    ],
    metrics: { cosine: 0.8527, sign: 0.9574, magnitude: 1.2994, selectivity: 0.0847 },
    note: "Best balanced Triangle II key.",
  },
  {
    id: "tower-2",
    name: "Tower II",
    target: "rook",
    targetLevel: 2,
    pieces: [
      { type: "pawn", level: 2, x: -3, y: 0 },
      { type: "pawn", level: 2, x: 2, y: 2 },
      { type: "pawn", level: 1, x: -1, y: -1 },
      { type: "king", level: 1, x: -1, y: 0 },
      { type: "rook", level: 1, x: 1, y: 1 },
    ],
    metrics: { cosine: 0.8015, sign: 0.7385, magnitude: 1.7513, selectivity: -0.0314 },
    note: "Provisional. Needs the Tower-specific constraint: Round II scaffold, a Tower contributor, no Triangle contributor.",
  },
  {
    id: "big-2",
    name: "Big II",
    target: "king",
    targetLevel: 2,
    pieces: [
      { type: "king", level: 1, x: -1, y: -4 },
      { type: "king", level: 1, x: -1, y: 1 },
      { type: "spy", level: 1, x: 4, y: -3 },
      { type: "spy", level: 1, x: 4, y: 2 },
    ],
    metrics: { cosine: 0.6729, sign: 0.9221, magnitude: 0.9345, selectivity: 0.1107 },
    note: "Modest cosine, but balanced magnitude and decent uniqueness.",
  },
];

function ring(delta: Position) {
  return Math.max(Math.abs(delta.x), Math.abs(delta.y));
}

function presetSign(preset: "checkerboard" | "diamond-core" | "horizontal-versus-vertical", delta: Position, r: number) {
  const absX = Math.abs(delta.x);
  const absY = Math.abs(delta.y);
  if (preset === "checkerboard") return (absX + absY) % 2 === 0 ? 1 : -1;
  if (preset === "diamond-core") return absX + absY <= 2 ? 1 : -1;
  return absX >= absY ? 1 : -1;
}

function evaluateBasis(definition: Basis, delta: Position) {
  const r = ring(delta);
  const multiplier = Math.pow(definition.decayBase, -r);
  if (definition.kind === "preset") return presetSign(definition.preset, delta, r) * multiplier;
  if (definition.kind === "grid") {
    const center = Math.floor(definition.gridValues.length / 2);
    return (definition.gridValues[delta.y + center]?.[delta.x + center] ?? 0) * multiplier;
  }
  const index = definition.repeat ? r % definition.ringValues.length : r;
  return (definition.ringValues[index] ?? 0) * multiplier;
}

function pieceContribution(piece: PatternPiece, sample: Position) {
  const delta = { x: sample.x - piece.x, y: sample.y - piece.y };
  if (piece.level === 2) {
    return supportExpandedContribution(piece.type, delta);
  }
  return levelOneContribution(piece.type, delta);
}

function supportExpandedContribution(type: PieceType, delta: Position) {
  return levelOneContribution(type, { x: Math.round(delta.x / 2), y: Math.round(delta.y / 2) });
}

function levelOneContribution(type: PieceType, delta: Position) {
  if (delta.x === 0 && delta.y === 0) return homeEnergy[type];
  let positive = 0;
  let negative = 0;
  const coefficients = activeComponents[type];
  const bases = definitions[type];
  for (let index = 0; index < coefficients.length; index += 1) {
    const value = coefficients[index] * evaluateBasis(bases[index], delta);
    if (value > 0) positive += value;
    else if (value < 0) negative += value;
  }
  const scale = waveScales[type];
  return strengths[type] * (positive * scale.friendly + negative * scale.hostile);
}

function fieldColor(value: number, maximum: number) {
  if (maximum <= 0) return "#b8b5ad";
  const amount = Math.min(1, Math.abs(value) / maximum);
  const alpha = 0.18 + amount * 0.72;
  if (value > 0) return `rgba(200, 75, 64, ${alpha})`;
  if (value < 0) return `rgba(55, 102, 167, ${alpha})`;
  return "#b8b5ad";
}

function absolutePieces(pattern: Pattern): PatternPiece[] {
  return pattern.pieces.map((piece) => ({
    ...piece,
    x: CENTER.x + piece.x,
    y: CENTER.y + piece.y,
  }));
}

function footprintCells(pattern: Pattern) {
  if (pattern.targetLevel === 1) return [`${CENTER.x}:${CENTER.y}`];
  return [
    `${CENTER.x}:${CENTER.y}`,
    `${CENTER.x + 1}:${CENTER.y}`,
    `${CENTER.x}:${CENTER.y + 1}`,
    `${CENTER.x + 1}:${CENTER.y + 1}`,
  ];
}

function formatMetric(value: number) {
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

export function WavefielderDemo() {
  const [selectedId, setSelectedId] = useState(patterns[0].id);
  const selected = patterns.find((pattern) => pattern.id === selectedId) ?? patterns[0];
  const pieces = useMemo(() => absolutePieces(selected), [selected]);
  const pieceMap = useMemo(() => new Map(pieces.map((piece) => [`${piece.x}:${piece.y}`, piece])), [pieces]);
  const footprint = useMemo(() => new Set(footprintCells(selected)), [selected]);
  const startedAt = globalThis.performance?.now?.() ?? Date.now();
  const field = useMemo(() => {
    return Array.from({ length: BOARD_SIZE }, (_, y) =>
      Array.from({ length: BOARD_SIZE }, (_, x) =>
        pieces.reduce((sum, piece) => sum + pieceContribution(piece, { x, y }), 0),
      ),
    );
  }, [pieces]);
  const elapsedMs = ((globalThis.performance?.now?.() ?? Date.now()) - startedAt);
  const maximum = Math.max(...field.flat().map((value) => Math.abs(value)), 0);
  const cells = Array.from({ length: BOARD_SIZE * BOARD_SIZE }, (_, index) => {
    const x = index % BOARD_SIZE;
    const y = Math.floor(index / BOARD_SIZE);
    const visualY = BOARD_SIZE - 1 - y;
    const piece = pieceMap.get(`${x}:${y}`);
    const isAnchor = x === CENTER.x && y === CENTER.y;
    const isFootprint = footprint.has(`${x}:${y}`);
    return { x, y, visualY, piece, isAnchor, isFootprint, value: field[y][x] };
  });

  return (
    <main className="wavefielder-page">
      <section className="wavefielder-header">
        <div>
          <p className="eyebrow">Wavefielder / RTS experiment</p>
          <h1>50x50 Field Nucleation Demo</h1>
        </div>
        <p>
          Sparse 50x50 terrain, current Wave Field kernels, Triangle diamond-only counterfactual, and
          support-expanded Level-II signatures. This is a demo surface, not a change to normal Wave Field rules.
        </p>
      </section>

      <section className="wavefielder-layout">
        <div className="wavefielder-board-panel">
          <div className="wavefielder-board-meta">
            <strong>{selected.name}</strong>
            <span>{labels[selected.target]} {selected.targetLevel === 2 ? "II" : "I"} anchor at ({CENTER.x}, {CENTER.y})</span>
          </div>
          <div className="wavefielder-board" style={{ "--wavefielder-board-size": BOARD_SIZE } as CSSProperties}>
            {cells.map((cell) => (
              <span
                key={`${cell.x}:${cell.y}`}
                className={`wavefielder-cell ${cell.isFootprint ? "footprint" : ""} ${cell.isAnchor ? "anchor" : ""} ${cell.piece ? "occupied" : ""}`}
                style={{
                  gridColumn: cell.x + 1,
                  gridRow: cell.visualY + 1,
                  backgroundColor: fieldColor(cell.value, maximum),
                }}
                title={`${cell.x},${cell.y} field ${cell.value.toFixed(3)}`}
              >
                {cell.piece && <i className={`wavefielder-piece ${cell.piece.type}`}>{initials[cell.piece.type]}{cell.piece.level}</i>}
                {cell.isAnchor && <b>*</b>}
              </span>
            ))}
          </div>
        </div>

        <aside className="wavefielder-panel">
          <div className="wavefielder-pattern-grid">
            {patterns.map((pattern) => (
              <button
                type="button"
                key={pattern.id}
                className={pattern.id === selected.id ? "active" : ""}
                onClick={() => setSelectedId(pattern.id)}
              >
                <strong>{pattern.name}</strong>
                <span>{pattern.pieces.map((piece) => `${initials[piece.type]}${piece.level}`).join(" ")}</span>
              </button>
            ))}
          </div>

          <dl className="wavefielder-metrics">
            <div><dt>Cosine</dt><dd>{formatMetric(selected.metrics.cosine)}</dd></div>
            <div><dt>Sign</dt><dd>{formatMetric(selected.metrics.sign)}</dd></div>
            <div><dt>Magnitude</dt><dd>{formatMetric(selected.metrics.magnitude)}</dd></div>
            <div><dt>Selectivity</dt><dd>{formatMetric(selected.metrics.selectivity)}</dd></div>
          </dl>

          <p>{selected.note}</p>

          <div className="wavefielder-assessment">
            <strong>Is this ridiculous in the current engine?</strong>
            <p>
              A sparse 50x50 field is fine for this demo. The normal board component would render 2,500
              interactive squares and continuous mode would explode to more than 200k samples, so Wavefielder
              wants a specialized renderer like this one.
            </p>
            <small>Last field recompute: {elapsedMs.toFixed(2)} ms for {pieces.length} pieces.</small>
          </div>
        </aside>
      </section>
    </main>
  );
}
