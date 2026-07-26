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
  | "skipped-rings";

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
  pawn: [Coefficient];
  rook: [Coefficient, Coefficient];
  spy: [Coefficient, Coefficient, Coefficient];
  king: [Coefficient, Coefficient, Coefficient];
}

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

export type BasisDefinition = RingBasisDefinition | PresetBasisDefinition;

export type ComponentDefinitions = {
  [K in PieceType]: BasisDefinition[];
};

export interface GameSnapshot {
  pieces: Piece[];
  currentPlayer: Player;
  components: Record<Player, PlayerComponents>;
  status: GameStatus;
  selectedPieceId: string | null;
  turnNumber: number;
  definitions: ComponentDefinitions;
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
