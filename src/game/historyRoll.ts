import { BOARD_SIZE, DEFAULT_HOME_ENERGY } from "./constants";
import type { Coefficient, GameSnapshot, GameState, HomeEnergy, PieceType, Player } from "./types";

const players: Player[] = ["blue", "red"];
const pieceTypes: PieceType[] = ["pawn", "rook", "spy", "king"];

export interface HistoryRollEntry {
  number: number;
  turnNumber: number;
  player: Player;
  summary: string;
  details: string[];
}

function playerName(player: Player) {
  return player === "blue" ? "Blue" : "Red";
}

function signLabel(player: Player, coefficient: Coefficient) {
  const fieldSign = player === "blue" ? -coefficient : coefficient;
  return fieldSign === 1 ? "+" : fieldSign === -1 ? "-" : "0";
}

function coordinate(x: number, y: number) {
  return `${String.fromCharCode(65 + x)}${BOARD_SIZE - y}`;
}

function homeEnergy(snapshot: GameSnapshot): HomeEnergy {
  return snapshot.homeEnergy ?? DEFAULT_HOME_ENERGY;
}

function describeTransition(before: GameSnapshot, after: GameSnapshot, number: number): HistoryRollEntry {
  const details: string[] = [];
  const moved = before.pieces.filter((piece) => {
    const next = after.pieces.find((candidate) => candidate.id === piece.id);
    return next && (next.position.x !== piece.position.x || next.position.y !== piece.position.y);
  });
  const removed = before.pieces.filter((piece) => !after.pieces.some((candidate) => candidate.id === piece.id));
  let tuningChanges = 0;

  for (const player of players) {
    for (const pieceType of pieceTypes) {
      before.components[player][pieceType].forEach((coefficient, index) => {
        const next = after.components[player][pieceType][index];
        if (coefficient === next) return;
        tuningChanges += 1;
        details.push(`${playerName(player)} ${pieceType} C${index + 1} ${signLabel(player, coefficient)}→${signLabel(player, next)}`);
      });
    }
  }

  const beforeHomeEnergy = homeEnergy(before);
  const afterHomeEnergy = homeEnergy(after);
  for (const pieceType of pieceTypes) {
    const beforeScale = before.waveScales[pieceType];
    const afterScale = after.waveScales[pieceType];
    if (beforeScale.friendly !== afterScale.friendly) {
      details.push(`${pieceType} friendly ${beforeScale.friendly.toFixed(2)}→${afterScale.friendly.toFixed(2)}`);
    }
    if (beforeScale.hostile !== afterScale.hostile) {
      details.push(`${pieceType} hostile ${beforeScale.hostile.toFixed(2)}→${afterScale.hostile.toFixed(2)}`);
    }
    if (beforeHomeEnergy[pieceType] !== afterHomeEnergy[pieceType]) {
      details.push(`${pieceType} home ${beforeHomeEnergy[pieceType].toFixed(2)}→${afterHomeEnergy[pieceType].toFixed(2)}`);
    }
  }

  for (const piece of moved) {
    const next = after.pieces.find((candidate) => candidate.id === piece.id)!;
    details.push(`${playerName(piece.owner)} ${piece.type} ${coordinate(piece.position.x, piece.position.y)}→${coordinate(next.position.x, next.position.y)}`);
  }
  for (const piece of removed) {
    details.push(`${playerName(piece.owner)} ${piece.type} lost at ${coordinate(piece.position.x, piece.position.y)}`);
  }
  if (before.status !== after.status) details.push(`Status ${before.status}→${after.status}`);

  let summary = "State checkpoint";
  if (moved.length > 0) summary = `${playerName(moved[0].owner)} moved ${moved[0].type}`;
  else if (tuningChanges > 0) summary = `${playerName(before.currentPlayer)} tuned ${tuningChanges} control${tuningChanges === 1 ? "" : "s"}`;
  else if (removed.length > 0) summary = `${removed.length} piece${removed.length === 1 ? "" : "s"} lost`;
  else if (before.currentPlayer !== after.currentPlayer) summary = `${playerName(after.currentPlayer)} turn began`;

  return { number, turnNumber: before.turnNumber, player: before.currentPlayer, summary, details };
}

export function buildHistoryRoll(state: GameState): HistoryRollEntry[] {
  const timeline: GameSnapshot[] = [...state.history, state];
  return timeline.slice(0, -1).map((before, index) =>
    describeTransition(before, timeline[index + 1], index + 1));
}
