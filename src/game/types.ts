export type Player = "red" | "blue";
export type PieceType = "pawn" | "rook" | "spy" | "king";
export type Territory = "red" | "neutral" | "blue";
export type Coefficient = -1 | 0 | 1;
export type GameStatus = "playing" | "red-won" | "blue-won";
export type FormulaPreset =
  | "checkerboard"
  | "diagonal-stripes"
  | "horizontal-versus-vertical"
  | "quadrants"
  | "constant-basin"
  | "skipped-rings"
  | "compass-rose"
  | "axis-favor"
  | "diagonal-favor"
  | "wide-bullseye"
  | "pulse-gap"
  | "block-checker"
  | "diamond-core"
  | "astigmatism"
  | "local-flip"
  | "adjacent-opinion"
  | "sink"
  | "deep-sink"
  | "far-crown"
  | "slow-governance"
  | "dipole-x"
  | "dipole-y";

export interface Position {
  x: number;
  y: number;
}

export interface Piece {
  id: string;
  owner: Player;
  type: PieceType;
  position: Position;
  unstable: boolean;
}

export interface PlayerComponents {
  pawn: Coefficient[];
  rook: Coefficient[];
  spy: Coefficient[];
  king: Coefficient[];
}

export type PlayerActivationOrder = Record<PieceType, number[]>;
export type ActivationOrders = Record<Player, PlayerActivationOrder>;

export interface WaveScale {
  friendly: number;
  hostile: number;
}

export type WaveScales = Record<PieceType, WaveScale>;

export type HomeEnergy = Record<PieceType, number>;

export interface RingBasisDefinition {
  kind: "ring";
  name: string;
  geometry: "chebyshev";
  ringValues: Coefficient[];
  repeat: boolean;
  decayBase: number;
  originScale: number;
}

export interface PresetBasisDefinition {
  kind: "preset";
  name: string;
  preset: FormulaPreset;
  decayBase: number;
  originScale: number;
}

export interface GridBasisDefinition {
  kind: "grid";
  name: string;
  gridValues: number[][];
  decayBase: number;
  originScale: number;
}

export interface ComboBasisTerm {
  weight: number;
  definition: RingBasisDefinition | PresetBasisDefinition | GridBasisDefinition;
}

export interface ComboBasisDefinition {
  kind: "combo";
  name: string;
  components: ComboBasisTerm[];
  decayBase: number;
  originScale: number;
}

export type BasisDefinition = RingBasisDefinition | PresetBasisDefinition | GridBasisDefinition | ComboBasisDefinition;

export type ComponentDefinitions = {
  [K in PieceType]: BasisDefinition[];
};

export interface GameSnapshot {
  pieces: Piece[];
  currentPlayer: Player;
  components: Record<Player, PlayerComponents>;
  activationOrders: ActivationOrders;
  status: GameStatus;
  selectedPieceId: string | null;
  turnNumber: number;
  definitions: ComponentDefinitions;
  waveScales: WaveScales;
  homeEnergy: HomeEnergy;
}

export interface GameState extends GameSnapshot {
  defaultComponents: PlayerComponents;
  history: GameSnapshot[];
  message: string;
}

export interface MoveResult {
  ok: boolean;
  state: GameState;
  reason?: string;
}

export interface InfluenceContributor {
  pieceID: string;
  owner: Player;
  kind: PieceType;
  position: Position;
  value: number;
  magnitude: number;
  shareOfTotalMagnitude: number;
}

export interface SquareInfluenceContributors {
  position: Position;
  total: number;
  contributors: InfluenceContributor[];
  highestNegativeContributor?: InfluenceContributor;
}

export interface InstabilityInfluenceLink {
  target: Position;
  targetPieceID: string;
  targetOwner: Player;
  targetKind: PieceType;
  contributor: InfluenceContributor;
}
