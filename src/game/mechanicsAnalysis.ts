import { BOARD_SIZE, PIECE_STRENGTH, TUNING_STRENGTH } from "./constants";
import { evaluateField } from "../field/evaluateField";
import { createInitialState } from "./initialState";
import { getLegalMoves } from "./movement";
import { applyMove, opponent } from "./rules";
import type { BasisDefinition, Coefficient, ComponentDefinitions, GameState, HomeEnergy, Piece, PieceType, Player, PlayerComponents, Position, WaveScales } from "./types";
import { evaluateComponentBasis } from "../field/kernels";
import { evaluatePieceContribution } from "../field/evaluateField";
import { getUnstablePieces } from "./victory";

const pieceTypes: PieceType[] = ["pawn", "rook", "spy", "king"];
const coefficients: Exclude<Coefficient, 0>[] = [-1, 1];

export interface ProfilePowerMetrics {
  pieceType: PieceType;
  profile: Coefficient[];
  l1: number;
  positiveMass: number;
  negativeMass: number;
  netMass: number;
  maxAbs: number;
  averageAbs: number;
  polarityL1Ratio: number;
  polarityNetDelta: number;
  signBalance: number;
}

export interface ComboOutlier {
  first: ComboPiece;
  second: ComboPiece;
  metrics: {
    l1: number;
    netMass: number;
    positiveCells: number;
    negativeCells: number;
    neutralCells: number;
    edgeCount: number;
    sameSignReinforcement: number;
    cancellation: number;
    maxAbs: number;
  };
}

export interface ComboPiece {
  pieceType: PieceType;
  profile: Coefficient[];
  position: Position;
}

export interface WaveScaleSearchResult {
  scales: WaveScales;
  score: number;
  maxPolarityRatio: number;
  maxPolarityProfile: { pieceType: PieceType; profile: Coefficient[]; ratio: number };
  deadProfiles: Array<{ pieceType: PieceType; profile: Coefficient[]; mobility: number }>;
  minMobility: number;
  maxMobility: number;
  mobilitySpread: number;
  averageMobility: number;
}

export type WaveScaleOptions = Record<PieceType, Array<{ friendly: number; hostile: number }>>;

export interface DefaultComponentSearchResult {
  components: PlayerComponents;
  score: number;
  openingMoves: { red: number; blue: number };
  selectedProfileMaxPolarityRatio: number;
  selectedProfileMinMobility: number;
}

export interface ComponentPatternMetrics {
  pieceType: PieceType;
  componentIndex: number;
  name: string;
  l1: number;
  positiveMass: number;
  negativeMass: number;
  netMass: number;
  signBalance: number;
  positiveCells: number;
  negativeCells: number;
  zeroCells: number;
  adjacentPositive: number;
  adjacentNegative: number;
  adjacentZero: number;
}

export interface ProfileMobilityDiagnostics {
  pieceType: PieceType;
  profile: Coefficient[];
  averageMoves: number;
  averageFirstSteps: number;
  deadOrigins: number;
  minFirstSteps: number;
  maxFirstSteps: number;
}

export interface HomeEnergySearchResult {
  homeEnergy: HomeEnergy;
  score: number;
  unstablePieces: number;
  minPieceMargin: number;
  averagePieceMargin: number;
  bigHatMargins: { red: number; blue: number };
  marginsByType: Record<PieceType, { min: number; average: number }>;
}

export type HomeEnergyOptions = Record<PieceType, number[]>;

export interface DefinitionVariantSearchResult {
  pieceType: PieceType;
  componentIndex: number;
  definition: BasisDefinition;
  score: number;
  maxPolarityRatio: number;
  deadProfiles: Array<{ profile: Coefficient[]; averageMoves: number; deadOrigins: number }>;
  minAverageMoves: number;
  minAverageFirstSteps: number;
  componentNetMass: number;
  componentSignBalance: number;
  adjacentCounts: { positive: number; negative: number; zero: number };
}

export interface DefinitionSetVariantSearchResult {
  replacements: Array<{ pieceType: PieceType; componentIndex: number; definition: BasisDefinition }>;
  score: number;
  affectedPieceTypes: PieceType[];
  maxPolarityRatio: number;
  deadProfiles: Array<{ pieceType: PieceType; profile: Coefficient[]; averageMoves: number; deadOrigins: number }>;
  minAverageMoves: number;
  minAverageFirstSteps: number;
  adjacentSummary: Array<{ pieceType: PieceType; componentIndex: number; positive: number; negative: number; zero: number; netMass: number }>;
}

export interface CombinedParameterCandidate {
  name: string;
  replacements?: Array<{ pieceType: PieceType; componentIndex: number; definition: BasisDefinition }>;
  scales?: WaveScales;
  components?: PlayerComponents;
  homeEnergy?: HomeEnergy;
}

export interface CombinedParameterSearchResult {
  name: string;
  score: number;
  wave: WaveScaleSearchResult;
  defaults: DefaultComponentSearchResult;
  home: HomeEnergySearchResult;
  replacements: Array<{ pieceType: PieceType; componentIndex: number; definition: BasisDefinition }>;
}

export interface PieceMobilityMetric {
  pieceId: string;
  owner: Player;
  pieceType: PieceType;
  position: Position;
  legalMoves: number;
  safetyMargin: number;
  unstable: boolean;
}

export interface MobilitySummary {
  total: number;
  byPlayer: Record<Player, number>;
  byType: Record<PieceType, number>;
  pieces: PieceMobilityMetric[];
}

export interface LossMobilityImpact {
  removedPieceId: string;
  owner: Player;
  pieceType: PieceType;
  position: Position;
  totalMobilityDelta: number;
  ownerMobilityDelta: number;
  opponentMobilityDelta: number;
  signChanges: number;
  unstableDelta: number;
  ownKingMarginDelta: number;
  enemyKingMarginDelta: number;
}

export interface MoveConsequenceMetric {
  pieceId: string;
  pieceType: PieceType;
  owner: Player;
  destination: Position;
  fieldSignChanges: number;
  fieldL1Delta: number;
  actingMobilityDelta: number;
  enemyMobilityDelta: number;
  unstableDelta: number;
  actingKingMarginDelta: number;
  enemyKingMarginDelta: number;
  apparentSafetyScore: number;
  trapScore: number;
  lureTrapScore: number;
  status: GameState["status"];
}

export interface FragmentationMetrics {
  redCells: number;
  blueCells: number;
  neutralCells: number;
  signEdges: number;
  redRegions: number;
  blueRegions: number;
  neutralRegions: number;
  largestRegion: number;
}

export interface ComplexitySnapshot {
  mobility: MobilitySummary;
  fragmentation: FragmentationMetrics;
  averageSafetyMargin: number;
  minSafetyMargin: number;
  nearZeroPieceCount: number;
  unstablePieces: number;
  moveConsequences: {
    count: number;
    averageSignChanges: number;
    maxSignChanges: number;
    averageFieldL1Delta: number;
    maxFieldL1Delta: number;
    averageMobilitySwing: number;
    maxMobilitySwing: number;
    topVolatileMoves: MoveConsequenceMetric[];
    topTrapMoves: MoveConsequenceMetric[];
    topLureTrapMoves: MoveConsequenceMetric[];
  };
}

function positions(): Position[] {
  return Array.from({ length: BOARD_SIZE * BOARD_SIZE }, (_, index) => ({
    x: index % BOARD_SIZE,
    y: Math.floor(index / BOARD_SIZE),
  }));
}

function samePosition(left: Position, right: Position): boolean {
  return left.x === right.x && left.y === right.y;
}

function rayStep(origin: Position, destination: Position): boolean {
  const dx = destination.x - origin.x;
  const dy = destination.y - origin.y;
  return Math.max(Math.abs(dx), Math.abs(dy)) === 1;
}

function ownerMargin(owner: Player, value: number): number {
  return owner === "red" ? value : -value;
}

function fieldSign(value: number): -1 | 0 | 1 {
  if (value > 0) return 1;
  if (value < 0) return -1;
  return 0;
}

function kingMargin(player: Player, state: GameState, field = evaluateField(state)): number {
  const king = state.pieces.find((piece) => piece.owner === player && piece.type === "king");
  return king ? ownerMargin(player, field[king.position.y][king.position.x]) : 0;
}

function unstableCount(state: GameState, field = evaluateField(state)): number {
  return getUnstablePieces("blue", state, field).length + getUnstablePieces("red", state, field).length;
}

function fieldDelta(before: number[][], after: number[][]): { signChanges: number; l1Delta: number } {
  let signChanges = 0;
  let l1Delta = 0;
  for (let y = 0; y < BOARD_SIZE; y += 1) {
    for (let x = 0; x < BOARD_SIZE; x += 1) {
      if (fieldSign(before[y][x]) !== fieldSign(after[y][x])) signChanges += 1;
      l1Delta += Math.abs(after[y][x] - before[y][x]);
    }
  }
  return { signChanges, l1Delta };
}

export function mobilitySummary(state: GameState = createInitialState(), field = evaluateField(state)): MobilitySummary {
  const pieces = state.pieces.map((piece) => ({
    pieceId: piece.id,
    owner: piece.owner,
    pieceType: piece.type,
    position: piece.position,
    legalMoves: getLegalMoves(piece.id, state, field).length,
    safetyMargin: ownerMargin(piece.owner, field[piece.position.y][piece.position.x]),
    unstable: ownerMargin(piece.owner, field[piece.position.y][piece.position.x]) < 0,
  }));
  const byPlayer = { blue: 0, red: 0 };
  const byType = { pawn: 0, rook: 0, spy: 0, king: 0 };
  for (const piece of pieces) {
    byPlayer[piece.owner] += piece.legalMoves;
    byType[piece.pieceType] += piece.legalMoves;
  }
  return {
    total: pieces.reduce((total, piece) => total + piece.legalMoves, 0),
    byPlayer,
    byType,
    pieces,
  };
}

export function fragmentationMetrics(state: GameState = createInitialState(), field = evaluateField(state)): FragmentationMetrics {
  const signs = field.map((row) => row.map(fieldSign));
  let redCells = 0;
  let blueCells = 0;
  let neutralCells = 0;
  let signEdges = 0;
  for (let y = 0; y < BOARD_SIZE; y += 1) {
    for (let x = 0; x < BOARD_SIZE; x += 1) {
      if (signs[y][x] > 0) redCells += 1;
      else if (signs[y][x] < 0) blueCells += 1;
      else neutralCells += 1;
      if (x + 1 < BOARD_SIZE && signs[y][x] !== signs[y][x + 1]) signEdges += 1;
      if (y + 1 < BOARD_SIZE && signs[y][x] !== signs[y + 1][x]) signEdges += 1;
    }
  }

  const seen = Array.from({ length: BOARD_SIZE }, () => Array.from({ length: BOARD_SIZE }, () => false));
  const regions = { "-1": 0, "0": 0, "1": 0 };
  let largestRegion = 0;
  for (let y = 0; y < BOARD_SIZE; y += 1) {
    for (let x = 0; x < BOARD_SIZE; x += 1) {
      if (seen[y][x]) continue;
      const sign = signs[y][x];
      regions[String(sign) as "-1" | "0" | "1"] += 1;
      let size = 0;
      const stack = [{ x, y }];
      seen[y][x] = true;
      while (stack.length > 0) {
        const current = stack.pop()!;
        size += 1;
        for (const next of [
          { x: current.x + 1, y: current.y },
          { x: current.x - 1, y: current.y },
          { x: current.x, y: current.y + 1 },
          { x: current.x, y: current.y - 1 },
        ]) {
          if (!inBoundsPosition(next) || seen[next.y][next.x] || signs[next.y][next.x] !== sign) continue;
          seen[next.y][next.x] = true;
          stack.push(next);
        }
      }
      largestRegion = Math.max(largestRegion, size);
    }
  }

  return {
    redCells,
    blueCells,
    neutralCells,
    signEdges,
    redRegions: regions["1"],
    blueRegions: regions["-1"],
    neutralRegions: regions["0"],
    largestRegion,
  };
}

function inBoundsPosition(position: Position): boolean {
  return position.x >= 0 && position.x < BOARD_SIZE && position.y >= 0 && position.y < BOARD_SIZE;
}

export function lossMobilityImpacts(state: GameState = createInitialState()): LossMobilityImpact[] {
  const beforeField = evaluateField(state);
  const beforeMobility = mobilitySummary(state, beforeField);
  const beforeUnstable = unstableCount(state, beforeField);
  return state.pieces
    .filter((piece) => piece.type !== "king")
    .map((piece) => {
      const next = { ...state, pieces: state.pieces.filter((candidate) => candidate.id !== piece.id) };
      const afterField = evaluateField(next);
      const afterMobility = mobilitySummary(next, afterField);
      const delta = fieldDelta(beforeField, afterField);
      const enemy = opponent(piece.owner);
      return {
        removedPieceId: piece.id,
        owner: piece.owner,
        pieceType: piece.type,
        position: piece.position,
        totalMobilityDelta: afterMobility.total - beforeMobility.total,
        ownerMobilityDelta: afterMobility.byPlayer[piece.owner] - beforeMobility.byPlayer[piece.owner],
        opponentMobilityDelta: afterMobility.byPlayer[enemy] - beforeMobility.byPlayer[enemy],
        signChanges: delta.signChanges,
        unstableDelta: unstableCount(next, afterField) - beforeUnstable,
        ownKingMarginDelta: kingMargin(piece.owner, next, afterField) - kingMargin(piece.owner, state, beforeField),
        enemyKingMarginDelta: kingMargin(enemy, next, afterField) - kingMargin(enemy, state, beforeField),
      };
    });
}

export function moveConsequenceMetrics(state: GameState = createInitialState()): MoveConsequenceMetric[] {
  if (state.status !== "playing") return [];
  const player = state.currentPlayer;
  const enemy = opponent(player);
  const beforeField = evaluateField(state);
  const beforeMobility = mobilitySummary(state, beforeField);
  const beforeUnstable = unstableCount(state, beforeField);
  const beforePlayerKing = kingMargin(player, state, beforeField);
  const beforeEnemyKing = kingMargin(enemy, state, beforeField);
  return state.pieces
    .filter((piece) => piece.owner === player)
    .flatMap((piece) =>
      getLegalMoves(piece.id, state, beforeField).flatMap((destination) => {
        const result = applyMove(piece.id, destination, state, { analyzeCheckmate: false });
        if (!result.ok) return [];
        const after = result.state;
        const afterField = evaluateField(after);
        const afterMobility = mobilitySummary(after, afterField);
        const delta = fieldDelta(beforeField, afterField);
        const actingMobilityDelta = afterMobility.byPlayer[player] - beforeMobility.byPlayer[player];
        const enemyMobilityDelta = afterMobility.byPlayer[enemy] - beforeMobility.byPlayer[enemy];
        const unstableDelta = unstableCount(after, afterField) - beforeUnstable;
        const actingKingMarginDelta = kingMargin(player, after, afterField) - beforePlayerKing;
        const enemyKingMarginDelta = kingMargin(enemy, after, afterField) - beforeEnemyKing;
        const apparentSafetyScore = enemyMobilityDelta + Math.max(0, actingMobilityDelta) * 0.5;
        const trapScore =
          Math.max(0, enemyMobilityDelta) * 1.5
          + Math.max(0, -enemyKingMarginDelta) * 8
          + Math.max(0, unstableDelta) * 12
          + Math.max(0, -actingMobilityDelta) * 0.75
          + delta.signChanges * 0.25;
        const lureTrapScore =
          Math.max(0, enemyMobilityDelta) * 4
          + Math.max(0, -enemyKingMarginDelta) * 8
          + Math.max(0, unstableDelta) * 12
          + Math.max(0, apparentSafetyScore) * 2;
        return [{
          pieceId: piece.id,
          pieceType: piece.type,
          owner: piece.owner,
          destination,
          fieldSignChanges: delta.signChanges,
          fieldL1Delta: delta.l1Delta,
          actingMobilityDelta,
          enemyMobilityDelta,
          unstableDelta,
          actingKingMarginDelta,
          enemyKingMarginDelta,
          apparentSafetyScore,
          trapScore,
          lureTrapScore,
          status: after.status,
        }];
      }),
    );
}

export function complexitySnapshot(state: GameState = createInitialState(), limit = 8): ComplexitySnapshot {
  const field = evaluateField(state);
  const mobility = mobilitySummary(state, field);
  const margins = mobility.pieces.map((piece) => piece.safetyMargin);
  const moves = moveConsequenceMetrics(state);
  const mobilitySwings = moves.map((move) => Math.abs(move.actingMobilityDelta) + Math.abs(move.enemyMobilityDelta));
  return {
    mobility,
    fragmentation: fragmentationMetrics(state, field),
    averageSafetyMargin: margins.reduce((total, value) => total + value, 0) / Math.max(1, margins.length),
    minSafetyMargin: Math.min(...margins),
    nearZeroPieceCount: margins.filter((value) => Math.abs(value) < 0.25).length,
    unstablePieces: mobility.pieces.filter((piece) => piece.unstable).length,
    moveConsequences: {
      count: moves.length,
      averageSignChanges: moves.reduce((total, move) => total + move.fieldSignChanges, 0) / Math.max(1, moves.length),
      maxSignChanges: Math.max(0, ...moves.map((move) => move.fieldSignChanges)),
      averageFieldL1Delta: moves.reduce((total, move) => total + move.fieldL1Delta, 0) / Math.max(1, moves.length),
      maxFieldL1Delta: Math.max(0, ...moves.map((move) => move.fieldL1Delta)),
      averageMobilitySwing: mobilitySwings.reduce((total, value) => total + value, 0) / Math.max(1, mobilitySwings.length),
      maxMobilitySwing: Math.max(0, ...mobilitySwings),
      topVolatileMoves: [...moves].sort((left, right) =>
        right.fieldSignChanges - left.fieldSignChanges || right.fieldL1Delta - left.fieldL1Delta,
      ).slice(0, limit),
      topTrapMoves: [...moves].sort((left, right) => right.trapScore - left.trapScore).slice(0, limit),
      topLureTrapMoves: [...moves].sort((left, right) => right.lureTrapScore - left.lureTrapScore).slice(0, limit),
    },
  };
}

export function enumerateProfiles(pieceType: PieceType): Coefficient[][] {
  const count = createInitialState().components.red[pieceType].length;
  const activeLimit = TUNING_STRENGTH[pieceType];
  const profiles: Coefficient[][] = [];

  function build(index: number, active: number, values: Coefficient[]) {
    if (index === count) {
      if (active === activeLimit) profiles.push(values);
      return;
    }
    if (count - index > activeLimit - active) build(index + 1, active, [...values, 0]);
    if (active < activeLimit) {
      for (const coefficient of coefficients) build(index + 1, active + 1, [...values, coefficient]);
    }
  }

  build(0, 0, []);
  return profiles;
}

function profileContribution(pieceType: PieceType, profile: readonly Coefficient[], delta: Position, state: GameState): number {
  if (delta.x === 0 && delta.y === 0) return state.homeEnergy[pieceType];
  return PIECE_STRENGTH[pieceType] * profile.reduce<number>((total, coefficient, index) => {
    const raw = coefficient * evaluateComponentBasis(pieceType, state.definitions[pieceType][index], delta);
    const scale = raw >= 0 ? state.waveScales[pieceType].friendly : state.waveScales[pieceType].hostile;
    return total + raw * scale;
  }, 0);
}

function gridForPiece(piece: ComboPiece, state: GameState): number[][] {
  return Array.from({ length: BOARD_SIZE }, (_, y) =>
    Array.from({ length: BOARD_SIZE }, (_, x) =>
      profileContribution(piece.pieceType, piece.profile, { x: x - piece.position.x, y: y - piece.position.y }, state),
    ),
  );
}

function summarizeGrid(grid: number[][]) {
  let l1 = 0;
  let positiveMass = 0;
  let negativeMass = 0;
  let maxAbs = 0;
  for (const row of grid) {
    for (const value of row) {
      const abs = Math.abs(value);
      l1 += abs;
      maxAbs = Math.max(maxAbs, abs);
      if (value > 0) positiveMass += value;
      if (value < 0) negativeMass += -value;
    }
  }
  return {
    l1,
    positiveMass,
    negativeMass,
    netMass: positiveMass - negativeMass,
    maxAbs,
    averageAbs: l1 / (BOARD_SIZE * BOARD_SIZE),
  };
}

export function profilePowerMetrics(state: GameState = createInitialState()): ProfilePowerMetrics[] {
  const boardPositions = positions();
  return pieceTypes.flatMap((pieceType) =>
    enumerateProfiles(pieceType).map((profile) => {
      const grid = Array.from({ length: BOARD_SIZE }, () => Array.from({ length: BOARD_SIZE }, () => 0));
      for (const origin of boardPositions) {
        for (const square of boardPositions) {
          grid[square.y][square.x] += profileContribution(
            pieceType,
            profile,
            { x: square.x - origin.x, y: square.y - origin.y },
            state,
          );
        }
      }
      const summary = summarizeGrid(grid);
      const negated = profile.map((value) => (value === 0 ? 0 : -value)) as Coefficient[];
      const negatedGrid = Array.from({ length: BOARD_SIZE }, () => Array.from({ length: BOARD_SIZE }, () => 0));
      for (const origin of boardPositions) {
        for (const square of boardPositions) {
          negatedGrid[square.y][square.x] += profileContribution(
            pieceType,
            negated,
            { x: square.x - origin.x, y: square.y - origin.y },
            state,
          );
        }
      }
      const opposite = summarizeGrid(negatedGrid);
      return {
        pieceType,
        profile,
        ...summary,
        polarityL1Ratio: summary.l1 / Math.max(opposite.l1, Number.EPSILON),
        polarityNetDelta: summary.netMass - opposite.netMass,
        signBalance: Math.min(summary.positiveMass, summary.negativeMass) / Math.max(summary.positiveMass, summary.negativeMass, Number.EPSILON),
      };
    }),
  );
}

function singletonState(pieceType: PieceType, profile: Coefficient[], position: Position, state: GameState): GameState {
  const components = structuredClone(state.components);
  for (const player of ["red", "blue"] as const) {
    components[player] = {
      pawn: state.components[player].pawn.map(() => 0) as PlayerComponents["pawn"],
      rook: state.components[player].rook.map(() => 0) as PlayerComponents["rook"],
      spy: state.components[player].spy.map(() => 0) as PlayerComponents["spy"],
      king: state.components[player].king.map(() => 0) as PlayerComponents["king"],
    };
  }
  components.red[pieceType] = profile as never;
  const piece: Piece = { id: "probe", owner: "red", type: pieceType, position, unstable: false };
  return { ...state, pieces: [piece], components };
}

function comboMetrics(first: ComboPiece, second: ComboPiece, state: GameState): ComboOutlier["metrics"] {
  const firstGrid = gridForPiece(first, state);
  const secondGrid = gridForPiece(second, state);
  let l1 = 0;
  let netMass = 0;
  let maxAbs = 0;
  let positiveCells = 0;
  let negativeCells = 0;
  let neutralCells = 0;
  let sameSignReinforcement = 0;
  let cancellation = 0;
  const signGrid = Array.from({ length: BOARD_SIZE }, () => Array.from({ length: BOARD_SIZE }, () => 0));

  for (let y = 0; y < BOARD_SIZE; y += 1) {
    for (let x = 0; x < BOARD_SIZE; x += 1) {
      const a = firstGrid[y][x];
      const b = secondGrid[y][x];
      const value = a + b;
      const abs = Math.abs(value);
      l1 += abs;
      netMass += value;
      maxAbs = Math.max(maxAbs, abs);
      if (value > 0) {
        positiveCells += 1;
        signGrid[y][x] = 1;
      } else if (value < 0) {
        negativeCells += 1;
        signGrid[y][x] = -1;
      } else {
        neutralCells += 1;
      }
      if (Math.sign(a) === Math.sign(b) && Math.sign(a) !== 0) sameSignReinforcement += Math.min(Math.abs(a), Math.abs(b));
      cancellation += Math.abs(a) + Math.abs(b) - Math.abs(value);
    }
  }

  let edgeCount = 0;
  for (let y = 0; y < BOARD_SIZE; y += 1) {
    for (let x = 0; x < BOARD_SIZE; x += 1) {
      if (x + 1 < BOARD_SIZE && signGrid[y][x] !== signGrid[y][x + 1]) edgeCount += 1;
      if (y + 1 < BOARD_SIZE && signGrid[y][x] !== signGrid[y + 1][x]) edgeCount += 1;
    }
  }

  return { l1, netMass, positiveCells, negativeCells, neutralCells, edgeCount, sameSignReinforcement, cancellation, maxAbs };
}

export function findComboOutliers(
  state: GameState = createInitialState(),
  limit = 12,
  options: { positions?: Position[]; pieceTypes?: PieceType[] } = {},
): Record<"reinforcement" | "cancellation" | "fragmentation" | "bias", ComboOutlier[]> {
  const boardPositions = options.positions ?? positions();
  const searchPieceTypes = options.pieceTypes ?? pieceTypes;
  const pieces = searchPieceTypes.flatMap((pieceType) =>
    enumerateProfiles(pieceType).flatMap((profile) =>
      boardPositions.map((position) => ({ pieceType, profile, position })),
    ),
  );
  const outliers = {
    reinforcement: [] as ComboOutlier[],
    cancellation: [] as ComboOutlier[],
    fragmentation: [] as ComboOutlier[],
    bias: [] as ComboOutlier[],
  };

  function remember(bucket: keyof typeof outliers, outlier: ComboOutlier, score: (candidate: ComboOutlier) => number) {
    outliers[bucket].push(outlier);
    outliers[bucket].sort((left, right) => score(right) - score(left));
    outliers[bucket].length = Math.min(outliers[bucket].length, limit);
  }

  for (let firstIndex = 0; firstIndex < pieces.length; firstIndex += 1) {
    for (let secondIndex = firstIndex + 1; secondIndex < pieces.length; secondIndex += 1) {
      const first = pieces[firstIndex];
      const second = pieces[secondIndex];
      if (samePosition(first.position, second.position)) continue;
      const outlier = { first, second, metrics: comboMetrics(first, second, state) };
      remember("reinforcement", outlier, (candidate) => candidate.metrics.sameSignReinforcement);
      remember("cancellation", outlier, (candidate) => candidate.metrics.cancellation);
      remember("fragmentation", outlier, (candidate) => candidate.metrics.edgeCount);
      remember("bias", outlier, (candidate) => Math.abs(candidate.metrics.netMass));
    }
  }

  return outliers;
}

export function legalMoveMobilityForProfile(pieceType: PieceType, profile: Coefficient[], state: GameState = createInitialState()): number {
  return positions().reduce((total, position) => {
    const singleton = singletonState(pieceType, profile, position, state);
    const field = Array.from({ length: BOARD_SIZE }, (_, y) =>
      Array.from({ length: BOARD_SIZE }, (_, x) =>
        evaluatePieceContribution(singleton.pieces[0], { x, y }, singleton),
      ),
    );
    return total + getLegalMoves("probe", singleton, field).length;
  }, 0);
}

export function profileMobilityDiagnostics(state: GameState = createInitialState()): ProfileMobilityDiagnostics[] {
  const boardPositions = positions();
  return pieceTypes.flatMap((pieceType) =>
    enumerateProfiles(pieceType).map((profile) => {
      let moveCount = 0;
      let firstStepCount = 0;
      let deadOrigins = 0;
      let minFirstSteps = Number.POSITIVE_INFINITY;
      let maxFirstSteps = 0;

      for (const position of boardPositions) {
        const singleton = singletonState(pieceType, profile, position, state);
        const field = Array.from({ length: BOARD_SIZE }, (_, y) =>
          Array.from({ length: BOARD_SIZE }, (_, x) =>
            evaluatePieceContribution(singleton.pieces[0], { x, y }, singleton),
          ),
        );
        const moves = getLegalMoves("probe", singleton, field);
        const firstSteps = moves.filter((move) => rayStep(position, move)).length;
        moveCount += moves.length;
        firstStepCount += firstSteps;
        if (moves.length === 0) deadOrigins += 1;
        minFirstSteps = Math.min(minFirstSteps, firstSteps);
        maxFirstSteps = Math.max(maxFirstSteps, firstSteps);
      }

      return {
        pieceType,
        profile,
        averageMoves: moveCount / boardPositions.length,
        averageFirstSteps: firstStepCount / boardPositions.length,
        deadOrigins,
        minFirstSteps,
        maxFirstSteps,
      };
    }),
  );
}

export function componentPatternMetrics(state: GameState = createInitialState()): ComponentPatternMetrics[] {
  const boardPositions = positions();
  return pieceTypes.flatMap((pieceType) =>
    state.definitions[pieceType].map((definition, componentIndex) => {
      let positiveMass = 0;
      let negativeMass = 0;
      let l1 = 0;
      let positiveCells = 0;
      let negativeCells = 0;
      let zeroCells = 0;
      let adjacentPositive = 0;
      let adjacentNegative = 0;
      let adjacentZero = 0;

      for (const origin of boardPositions) {
        for (const square of boardPositions) {
          const delta = { x: square.x - origin.x, y: square.y - origin.y };
          const value = evaluateComponentBasis(pieceType, definition, delta);
          const abs = Math.abs(value);
          l1 += abs;
          if (value > 0) {
            positiveMass += value;
            positiveCells += 1;
          } else if (value < 0) {
            negativeMass += -value;
            negativeCells += 1;
          } else {
            zeroCells += 1;
          }
          if (rayStep(origin, square)) {
            if (value > 0) adjacentPositive += 1;
            else if (value < 0) adjacentNegative += 1;
            else adjacentZero += 1;
          }
        }
      }

      return {
        pieceType,
        componentIndex,
        name: definition.name,
        l1,
        positiveMass,
        negativeMass,
        netMass: positiveMass - negativeMass,
        signBalance: Math.min(positiveMass, negativeMass) / Math.max(positiveMass, negativeMass, Number.EPSILON),
        positiveCells,
        negativeCells,
        zeroCells,
        adjacentPositive,
        adjacentNegative,
        adjacentZero,
      };
    }),
  );
}

function stateWithDefinition(
  pieceType: PieceType,
  componentIndex: number,
  definition: BasisDefinition,
  baseState: GameState,
): GameState {
  const definitions = structuredClone(baseState.definitions) as ComponentDefinitions;
  definitions[pieceType][componentIndex] = structuredClone(definition);
  return { ...baseState, definitions };
}

function stateWithDefinitions(
  replacements: Array<{ pieceType: PieceType; componentIndex: number; definition: BasisDefinition }>,
  baseState: GameState,
): GameState {
  const definitions = structuredClone(baseState.definitions) as ComponentDefinitions;
  for (const replacement of replacements) {
    definitions[replacement.pieceType][replacement.componentIndex] = structuredClone(replacement.definition);
  }
  return { ...baseState, definitions };
}

export function evaluateDefinitionVariant(
  pieceType: PieceType,
  componentIndex: number,
  definition: BasisDefinition,
  baseState: GameState = createInitialState(),
): DefinitionVariantSearchResult {
  const state = stateWithDefinition(pieceType, componentIndex, definition, baseState);
  const powerMetrics = profilePowerMetrics(state).filter((metric) => metric.pieceType === pieceType);
  const mobilityMetrics = profileMobilityDiagnostics(state).filter((metric) => metric.pieceType === pieceType);
  const pattern = componentPatternMetrics(state).find((metric) =>
    metric.pieceType === pieceType && metric.componentIndex === componentIndex);
  const maxPolarityRatio = Math.max(...powerMetrics.map((metric) => normalizedRatio(metric.polarityL1Ratio)));
  const minAverageMoves = Math.min(...mobilityMetrics.map((metric) => metric.averageMoves));
  const minAverageFirstSteps = Math.min(...mobilityMetrics.map((metric) => metric.averageFirstSteps));
  const deadProfiles = mobilityMetrics
    .filter((metric) => metric.averageMoves < 4 || metric.deadOrigins > 0)
    .map((metric) => ({ profile: metric.profile, averageMoves: metric.averageMoves, deadOrigins: metric.deadOrigins }));
  const adjacentImbalance = pattern
    ? Math.abs(pattern.adjacentPositive - pattern.adjacentNegative) / Math.max(pattern.adjacentPositive + pattern.adjacentNegative, 1)
    : 1;
  const score =
    deadProfiles.length * 100
    + maxPolarityRatio * 12
    + Math.max(0, 4 - minAverageMoves) * 20
    + Math.max(0, 2 - minAverageFirstSteps) * 25
    + adjacentImbalance * 8
    + Math.abs(pattern?.netMass ?? 0) * 0.01;

  return {
    pieceType,
    componentIndex,
    definition,
    score,
    maxPolarityRatio,
    deadProfiles,
    minAverageMoves,
    minAverageFirstSteps,
    componentNetMass: pattern?.netMass ?? 0,
    componentSignBalance: pattern?.signBalance ?? 0,
    adjacentCounts: {
      positive: pattern?.adjacentPositive ?? 0,
      negative: pattern?.adjacentNegative ?? 0,
      zero: pattern?.adjacentZero ?? 0,
    },
  };
}

export function candidateDefinitionVariants(pieceType: PieceType, componentIndex: number, baseState: GameState = createInitialState()): BasisDefinition[] {
  const current = baseState.definitions[pieceType][componentIndex];
  const variants: BasisDefinition[] = [structuredClone(current)];
  const seen = new Set([JSON.stringify(current)]);

  function add(definition: BasisDefinition) {
    const key = JSON.stringify(definition);
    if (!seen.has(key)) {
      seen.add(key);
      variants.push(definition);
    }
  }

  if (current.kind === "ring") {
    const length = current.ringValues.length;
    const templates: Coefficient[][] = [
      [0, -1, 1, -1],
      [0, 1, -1, -1],
      [0, 1, -1, 1],
      [0, -1, 1, 1],
      [0, 0, 1, -1],
      [0, 0, -1, 1],
      [0, 1, 0, -1],
      [0, -1, 0, 1],
      [0, 1, 1, -1],
      [0, -1, -1, 1],
      [1, -1, 1, -1, 1, -1],
      [1, -1, -1, 1, 1, -1],
      [-1, 1, 1, -1, -1, 1],
    ];
    for (const template of templates) {
      add({
        ...current,
        name: `${current.name} variant ${template.join("/")}`,
        ringValues: Array.from({ length }, (_, index) => template[index % template.length]) as Coefficient[],
      });
    }
  } else {
    for (const preset of [
      "checkerboard",
      "diagonal-stripes",
      "horizontal-versus-vertical",
      "axis-favor",
      "diagonal-favor",
      "block-checker",
      "diamond-core",
      "astigmatism",
      "local-flip",
      "adjacent-opinion",
    ] as const) {
      add({ ...current, name: `${current.name} variant ${preset}`, preset });
    }
  }

  return variants;
}

export function searchDefinitionVariants(
  targets: Array<{ pieceType: PieceType; componentIndex: number }>,
  limit = 12,
  baseState: GameState = createInitialState(),
): DefinitionVariantSearchResult[] {
  return targets
    .flatMap((target) =>
      candidateDefinitionVariants(target.pieceType, target.componentIndex, baseState)
        .map((definition) => evaluateDefinitionVariant(target.pieceType, target.componentIndex, definition, baseState)))
    .sort((left, right) => left.score - right.score)
    .slice(0, limit);
}

export function evaluateDefinitionSetVariant(
  replacements: Array<{ pieceType: PieceType; componentIndex: number; definition: BasisDefinition }>,
  baseState: GameState = createInitialState(),
): DefinitionSetVariantSearchResult {
  const state = stateWithDefinitions(replacements, baseState);
  const affectedPieceTypes = [...new Set(replacements.map((replacement) => replacement.pieceType))];
  const powerMetrics = profilePowerMetrics(state).filter((metric) => affectedPieceTypes.includes(metric.pieceType));
  const mobilityMetrics = profileMobilityDiagnostics(state).filter((metric) => affectedPieceTypes.includes(metric.pieceType));
  const patternMetrics = componentPatternMetrics(state);
  const maxPolarityRatio = Math.max(...powerMetrics.map((metric) => normalizedRatio(metric.polarityL1Ratio)));
  const minAverageMoves = Math.min(...mobilityMetrics.map((metric) => metric.averageMoves));
  const minAverageFirstSteps = Math.min(...mobilityMetrics.map((metric) => metric.averageFirstSteps));
  const deadProfiles = mobilityMetrics
    .filter((metric) => metric.averageMoves < 4 || metric.deadOrigins > 0)
    .map((metric) => ({
      pieceType: metric.pieceType,
      profile: metric.profile,
      averageMoves: metric.averageMoves,
      deadOrigins: metric.deadOrigins,
    }));
  const adjacentSummary = replacements.map((replacement) => {
    const metric = patternMetrics.find((candidate) =>
      candidate.pieceType === replacement.pieceType && candidate.componentIndex === replacement.componentIndex);
    return {
      pieceType: replacement.pieceType,
      componentIndex: replacement.componentIndex,
      positive: metric?.adjacentPositive ?? 0,
      negative: metric?.adjacentNegative ?? 0,
      zero: metric?.adjacentZero ?? 0,
      netMass: metric?.netMass ?? 0,
    };
  });
  const adjacentPenalty = adjacentSummary.reduce((total, row) => {
    const signed = row.positive + row.negative;
    return total + Math.abs(row.positive - row.negative) / Math.max(signed, 1);
  }, 0);
  const score =
    deadProfiles.length * 120
    + maxPolarityRatio * 12
    + Math.max(0, 4 - minAverageMoves) * 20
    + Math.max(0, 2 - minAverageFirstSteps) * 25
    + adjacentPenalty * 4
    + adjacentSummary.reduce((total, row) => total + Math.abs(row.netMass) * 0.005, 0);

  return {
    replacements,
    score,
    affectedPieceTypes,
    maxPolarityRatio,
    deadProfiles,
    minAverageMoves,
    minAverageFirstSteps,
    adjacentSummary,
  };
}

export function searchDefinitionSetVariants(
  targetSets: Array<Array<{ pieceType: PieceType; componentIndex: number }>>,
  limit = 12,
  baseState: GameState = createInitialState(),
): DefinitionSetVariantSearchResult[] {
  const results: DefinitionSetVariantSearchResult[] = [];

  function build(
    targets: Array<{ pieceType: PieceType; componentIndex: number }>,
    index: number,
    replacements: Array<{ pieceType: PieceType; componentIndex: number; definition: BasisDefinition }>,
  ) {
    if (index === targets.length) {
      results.push(evaluateDefinitionSetVariant(replacements, baseState));
      return;
    }
    const target = targets[index];
    for (const definition of candidateDefinitionVariants(target.pieceType, target.componentIndex, baseState)) {
      build(targets, index + 1, [...replacements, { ...target, definition }]);
    }
  }

  for (const targets of targetSets) build(targets, 0, []);
  return results
    .sort((left, right) => left.score - right.score)
    .slice(0, limit);
}

function normalizedRatio(ratio: number): number {
  return Math.max(ratio, 1 / Math.max(ratio, Number.EPSILON));
}

function profileLabelKey(pieceType: PieceType, profile: readonly Coefficient[], state: GameState): string {
  const scale = state.waveScales[pieceType];
  return `${pieceType}:${profile.join(",")}:${scale.friendly}:${scale.hostile}`;
}

export function evaluateWaveScaleSet(
  scales: WaveScales,
  baseState: GameState = createInitialState(),
  mobilityCache = new Map<string, number>(),
): WaveScaleSearchResult {
  const state = { ...baseState, waveScales: structuredClone(scales) };
  const profileMetrics = profilePowerMetrics(state);
  const polarityRows = profileMetrics.map((metric) => ({
    pieceType: metric.pieceType,
    profile: metric.profile,
    ratio: normalizedRatio(metric.polarityL1Ratio),
  }));
  polarityRows.sort((left, right) => right.ratio - left.ratio);

  const mobilityRows = pieceTypes.flatMap((pieceType) =>
    enumerateProfiles(pieceType).map((profile) => {
      const key = profileLabelKey(pieceType, profile, state);
      const cached = mobilityCache.get(key);
      const mobility = cached ?? legalMoveMobilityForProfile(pieceType, profile, state) / (BOARD_SIZE * BOARD_SIZE);
      if (cached === undefined) mobilityCache.set(key, mobility);
      return { pieceType, profile, mobility };
    }),
  );
  const nonSpyMobility = mobilityRows.filter((row) => row.pieceType !== "spy").map((row) => row.mobility);
  const minMobility = Math.min(...nonSpyMobility);
  const maxMobility = Math.max(...nonSpyMobility);
  const averageMobility = nonSpyMobility.reduce((total, value) => total + value, 0) / nonSpyMobility.length;
  const deadProfiles = mobilityRows.filter((row) => row.pieceType !== "spy" && row.mobility < 4);
  const maxPolarityRatio = polarityRows[0].ratio;
  const mobilitySpread = maxMobility - minMobility;
  const score =
    maxPolarityRatio * 16
    + deadProfiles.length * 80
    + Math.max(0, 4 - minMobility) * 24
    + mobilitySpread * 1.5
    + Math.abs(averageMobility - 10) * 0.5;

  return {
    scales,
    score,
    maxPolarityRatio,
    maxPolarityProfile: polarityRows[0],
    deadProfiles,
    minMobility,
    maxMobility,
    mobilitySpread,
    averageMobility,
  };
}

export function searchWaveScaleOptions(
  options: WaveScaleOptions,
  limit = 12,
  baseState: GameState = createInitialState(),
): WaveScaleSearchResult[] {
  const results: WaveScaleSearchResult[] = [];
  const mobilityCache = new Map<string, number>();

  for (const pawn of options.pawn) {
    for (const rook of options.rook) {
      for (const spy of options.spy) {
        for (const king of options.king) {
          results.push(evaluateWaveScaleSet({ pawn, rook, spy, king }, baseState, mobilityCache));
        }
      }
    }
  }

  return results
    .sort((left, right) => left.score - right.score)
    .slice(0, limit);
}

export function enumerateDefaultComponentSets(): PlayerComponents[] {
  const sets: PlayerComponents[] = [];
  for (const pawn of enumerateProfiles("pawn")) {
    for (const rook of enumerateProfiles("rook")) {
      for (const spy of enumerateProfiles("spy")) {
        for (const king of enumerateProfiles("king")) {
          sets.push({
            pawn: pawn as PlayerComponents["pawn"],
            rook: rook as PlayerComponents["rook"],
            spy: spy as PlayerComponents["spy"],
            king: king as PlayerComponents["king"],
          });
        }
      }
    }
  }
  return sets;
}

export function evaluateDefaultComponentSet(
  components: PlayerComponents,
  baseState: GameState = createInitialState(),
  mobilityCache = new Map<string, number>(),
): DefaultComponentSearchResult {
  const state = createInitialState(
    components,
    structuredClone(baseState.definitions),
    structuredClone(baseState.waveScales),
    structuredClone(baseState.homeEnergy),
  );
  const field = evaluateField(state);
  const openingMoves = { red: 0, blue: 0 };
  for (const piece of state.pieces) {
    openingMoves[piece.owner] += getLegalMoves(piece.id, state, field).length;
  }

  const profileMetrics = profilePowerMetrics(state);
  const selectedProfiles = pieceTypes.map((pieceType) => {
    const profile = components[pieceType];
    const metric = profileMetrics.find((candidate) =>
      candidate.pieceType === pieceType && candidate.profile.every((value, index) => value === profile[index]));
    const mobilityKey = profileLabelKey(pieceType, profile, state);
    const cached = mobilityCache.get(mobilityKey);
    const mobility = cached ?? legalMoveMobilityForProfile(pieceType, [...profile], state) / (BOARD_SIZE * BOARD_SIZE);
    if (cached === undefined) mobilityCache.set(mobilityKey, mobility);
    return {
      ratio: metric ? normalizedRatio(metric.polarityL1Ratio) : Number.POSITIVE_INFINITY,
      mobility,
    };
  });
  const selectedProfileMaxPolarityRatio = Math.max(...selectedProfiles.map((row) => row.ratio));
  const selectedProfileMinMobility = Math.min(...selectedProfiles.filter((_, index) => pieceTypes[index] !== "spy").map((row) => row.mobility));
  const totalOpeningMoves = openingMoves.red + openingMoves.blue;
  const score =
    selectedProfileMaxPolarityRatio * 10
    + Math.max(0, 4 - selectedProfileMinMobility) * 30
    + Math.max(0, 24 - totalOpeningMoves) * 2
    + Math.abs(openingMoves.red - openingMoves.blue) * 5;

  return {
    components,
    score,
    openingMoves,
    selectedProfileMaxPolarityRatio,
    selectedProfileMinMobility,
  };
}

export function searchDefaultComponentSets(
  limit = 12,
  baseState: GameState = createInitialState(),
  candidates: PlayerComponents[] = enumerateDefaultComponentSets(),
): DefaultComponentSearchResult[] {
  const mobilityCache = new Map<string, number>();
  return candidates
    .map((components) => evaluateDefaultComponentSet(components, baseState, mobilityCache))
    .sort((left, right) => left.score - right.score)
    .slice(0, limit);
}

export function evaluateHomeEnergySet(
  homeEnergy: HomeEnergy,
  baseState: GameState = createInitialState(),
): HomeEnergySearchResult {
  const state = createInitialState(
    structuredClone(baseState.defaultComponents),
    structuredClone(baseState.definitions),
    structuredClone(baseState.waveScales),
    structuredClone(homeEnergy),
  );
  const field = evaluateField(state);
  const margins = state.pieces.map((piece) => ({
    pieceType: piece.type,
    owner: piece.owner,
    margin: ownerMargin(piece.owner, field[piece.position.y][piece.position.x]),
  }));
  const minPieceMargin = Math.min(...margins.map((row) => row.margin));
  const averagePieceMargin = margins.reduce((total, row) => total + row.margin, 0) / margins.length;
  const unstablePieces = margins.filter((row) => row.margin < 0).length;
  const redBigHat = margins.find((row) => row.owner === "red" && row.pieceType === "king")?.margin ?? 0;
  const blueBigHat = margins.find((row) => row.owner === "blue" && row.pieceType === "king")?.margin ?? 0;
  const marginsByType = Object.fromEntries(pieceTypes.map((pieceType) => {
    const rows = margins.filter((row) => row.pieceType === pieceType);
    return [pieceType, {
      min: Math.min(...rows.map((row) => row.margin)),
      average: rows.reduce((total, row) => total + row.margin, 0) / rows.length,
    }];
  })) as HomeEnergySearchResult["marginsByType"];
  const score =
    unstablePieces * 100
    + Math.max(0, 0.25 - minPieceMargin) * 20
    + Math.abs(redBigHat - blueBigHat) * 10
    + Math.abs(averagePieceMargin - 1) * 0.25;

  return {
    homeEnergy,
    score,
    unstablePieces,
    minPieceMargin,
    averagePieceMargin,
    bigHatMargins: { red: redBigHat, blue: blueBigHat },
    marginsByType,
  };
}

export function searchHomeEnergyOptions(
  options: HomeEnergyOptions,
  limit = 12,
  baseState: GameState = createInitialState(),
): HomeEnergySearchResult[] {
  const results: HomeEnergySearchResult[] = [];
  for (const pawn of options.pawn) {
    for (const rook of options.rook) {
      for (const spy of options.spy) {
        for (const king of options.king) {
          results.push(evaluateHomeEnergySet({ pawn, rook, spy, king }, baseState));
        }
      }
    }
  }
  return results
    .sort((left, right) => left.score - right.score)
    .slice(0, limit);
}

export function evaluateCombinedParameterCandidate(
  candidate: CombinedParameterCandidate,
  baseState: GameState = createInitialState(),
): CombinedParameterSearchResult {
  const definitions = candidate.replacements
    ? stateWithDefinitions(candidate.replacements, baseState).definitions
    : structuredClone(baseState.definitions);
  const components = structuredClone(candidate.components ?? baseState.defaultComponents);
  const scales = structuredClone(candidate.scales ?? baseState.waveScales);
  const homeEnergy = structuredClone(candidate.homeEnergy ?? baseState.homeEnergy);
  const state = createInitialState(components, definitions, scales, homeEnergy);
  const wave = evaluateWaveScaleSet(scales, state);
  const defaults = evaluateDefaultComponentSet(components, state);
  const home = evaluateHomeEnergySet(homeEnergy, state);
  const score =
    wave.deadProfiles.length * 100
    + wave.maxPolarityRatio * 14
    + Math.max(0, 4 - wave.minMobility) * 30
    + wave.mobilitySpread * 1.2
    + defaults.selectedProfileMaxPolarityRatio * 8
    + Math.max(0, 24 - defaults.openingMoves.blue - defaults.openingMoves.red) * 2
    + Math.abs(defaults.openingMoves.blue - defaults.openingMoves.red) * 5
    + home.unstablePieces * 100
    + Math.max(0, 0.25 - home.minPieceMargin) * 20;

  return {
    name: candidate.name,
    score,
    wave,
    defaults,
    home,
    replacements: candidate.replacements ?? [],
  };
}

export function searchCombinedParameterCandidates(
  candidates: CombinedParameterCandidate[],
  limit = 12,
  baseState: GameState = createInitialState(),
): CombinedParameterSearchResult[] {
  return candidates
    .map((candidate) => evaluateCombinedParameterCandidate(candidate, baseState))
    .sort((left, right) => left.score - right.score)
    .slice(0, limit);
}
