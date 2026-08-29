import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const input = resolve(process.env.IN ?? process.argv[2] ?? "analysis/selfplay-position-miner.jsonl");
const limit = Number(process.env.LIMIT ?? 12);

const text = await readFile(input, "utf8");
const rows = text
  .split(/\n+/)
  .filter(Boolean)
  .map((line, index) => {
    try {
      return JSON.parse(line);
    } catch (error) {
      throw new Error("Invalid JSONL at line " + (index + 1) + ": " + error.message);
    }
  });

function inc(map, key, amount = 1) {
  map.set(key, (map.get(key) ?? 0) + amount);
}

function top(map, n = limit) {
  return [...map.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, n);
}

function pct(value, total) {
  return total === 0 ? "0.0%" : (value / total * 100).toFixed(1) + "%";
}

function pieceAt(row, owner, type = "king") {
  return row.boardState.find((piece) => piece.owner === owner && piece.type === type);
}

function cornerDistance(piece) {
  if (!piece) return 99;
  return Math.min(
    piece.x + piece.y,
    piece.x + Math.abs(6 - piece.y),
    Math.abs(6 - piece.x) + piece.y,
    Math.abs(6 - piece.x) + Math.abs(6 - piece.y),
  );
}

function pressureAgainst(row, owner) {
  return row.pressureLinksAboveThreshold.filter((link) => link.targetOwner === owner).length;
}

function shapeName(row, side) {
  const king = pieceAt(row, side);
  const escapes = side === "blue" ? row.blueEscapes : row.redEscapes;
  const safeMoves = side === "blue" ? row.blueSafeMoves : row.redSafeMoves;
  const support = row.connectednessSupportScore[side];
  const distance = row.alliedAverageDistance[side];
  const enemyPressure = pressureAgainst(row, side);
  if (king && cornerDistance(king) <= 1 && escapes <= 1 && enemyPressure > 0) return "Corner Pocket";
  if (escapes <= 1 && safeMoves <= 2) return "Net Collapse";
  if (escapes > 0 && safeMoves <= 1) return "False Escape";
  if (distance >= 4 && support <= 1) return "Split Field";
  if (support >= 4 && escapes >= 2) return "Anchor Chain";
  if (enemyPressure >= 2 && escapes <= 2) return "Long Pin";
  return "Unclassified";
}

function gameRows() {
  const games = new Map();
  for (const row of rows) {
    if (!games.has(row.game)) games.set(row.game, []);
    games.get(row.game).push(row);
  }
  return [...games.values()].map((game) => game.sort((a, b) => a.halfTurn - b.halfTurn));
}

const games = gameRows();
const winnerCounts = new Map();
for (const game of games) inc(winnerCounts, game.at(-1)?.winner ?? "limit");

const openingWins = new Map();
for (const game of games) {
  const winner = game.at(-1)?.winner;
  if (!winner) continue;
  const opening = game.slice(0, 4).map((row) => row.player[0] + ":" + row.move.notation).join(" ");
  inc(openingWins, winner + " | " + opening);
}

const finalShapes = new Map();
for (const game of games) {
  const winner = game.at(-1)?.winner;
  if (!winner) continue;
  const loser = winner === "blue" ? "red" : "blue";
  for (const row of game.slice(-5)) inc(finalShapes, shapeName(row, loser));
}

const blunders = rows
  .filter((row) => row.winner && row.player !== row.winner && row.turnsUntilWin !== null && row.turnsUntilWin <= 6)
  .map((row) => ({
    key: "g" + row.game + " t" + row.turn + " " + row.player + " " + row.bot + " " + row.move.notation,
    score: -row.bigHatEscapeDelta.mineDelta + row.bigHatEscapeDelta.enemyDelta + row.piecesLost[row.player].reduce((total, piece) => total + piece.value, 0),
    reason: row.moveAnalysis.reason,
  }))
  .sort((left, right) => right.score - left.score)
  .slice(0, limit);

const rescueByBot = new Map();
const rescueWinsByBot = new Map();
for (const row of rows.filter((candidate) => candidate.hintSearchFoundRescue)) {
  inc(rescueByBot, row.bot);
  if (row.winner === row.player) inc(rescueWinsByBot, row.bot);
}

const hardOnlyFingerprints = new Map();
for (const row of rows) {
  const fingerprint = [
    row.player,
    row.boardState.map((piece) => piece.id + "@" + piece.x + "," + piece.y).join("|"),
    JSON.stringify(row.components),
  ].join(" ");
  const entry = hardOnlyFingerprints.get(fingerprint) ?? { bots: new Set(), hardRescue: null, weakMiss: false };
  entry.bots.add(row.bot);
  if (row.bot === "hard" && row.hintSearchFoundRescue) entry.hardRescue = row;
  if ((row.bot === "random_ish" || row.bot === "medium") && !row.hintSearchFoundRescue) entry.weakMiss = true;
  hardOnlyFingerprints.set(fingerprint, entry);
}

const hardRescues = [...hardOnlyFingerprints.values()]
  .filter((entry) => entry.hardRescue && entry.weakMiss)
  .map((entry) => entry.hardRescue)
  .slice(0, limit);

const materialLosingWins = rows
  .filter((row) => row.winner === row.player && row.piecesLost[row.player].length > 0)
  .map((row) => ({
    key: "g" + row.game + " t" + row.turn + " " + row.player + " " + row.bot + " " + row.move.notation,
    lost: row.piecesLost[row.player].map((piece) => piece.type).join(","),
    turnsUntilWin: row.turnsUntilWin,
    delta: row.bigHatEscapeDelta,
    reason: row.moveAnalysis.reason,
  }))
  .sort((left, right) => left.turnsUntilWin - right.turnsUntilWin)
  .slice(0, limit);

const botOutcomes = new Map();
for (const row of rows) {
  const key = row.bot + " as " + row.player;
  const entry = botOutcomes.get(key) ?? { turns: 0, wins: 0, escapeDelta: 0, materialLost: 0 };
  entry.turns += 1;
  if (row.winner === row.player) entry.wins += 1;
  entry.escapeDelta += row.bigHatEscapeDelta.mineDelta - row.bigHatEscapeDelta.enemyDelta;
  entry.materialLost += row.piecesLost[row.player].reduce((total, piece) => total + piece.value, 0);
  botOutcomes.set(key, entry);
}

console.log("Self-play analysis: " + input);
console.log("turns=" + rows.length + " games=" + games.length);
console.log("");
console.log("Winner balance");
for (const [winner, count] of top(winnerCounts, 4)) console.log("  " + winner + ": " + count + " (" + pct(count, games.length) + ")");

console.log("");
console.log("Top winning openings");
for (const [opening, count] of top(openingWins)) console.log("  " + count + "x " + opening);

console.log("");
console.log("Most common final trap shapes");
for (const [shape, count] of top(finalShapes)) console.log("  " + count + "x " + shape);

console.log("");
console.log("Most common losing blunders");
for (const row of blunders) console.log("  score=" + row.score.toFixed(2) + " " + row.key + " | " + row.reason);

console.log("");
console.log("Rescues found by hint/search");
for (const [bot, count] of top(rescueByBot)) {
  console.log("  " + bot + ": " + count + " found, " + (rescueWinsByBot.get(bot) ?? 0) + " eventual wins");
}

console.log("");
console.log("Positions where hard found rescue and random/medium did not");
if (hardRescues.length === 0) console.log("  none in this sample");
for (const row of hardRescues) console.log("  g" + row.game + " t" + row.turn + " " + row.player + " " + row.move.notation + " | " + row.moveAnalysis.reason);

console.log("");
console.log("Material-losing moves that led to wins");
if (materialLosingWins.length === 0) console.log("  none in this sample");
for (const row of materialLosingWins) {
  console.log("  " + row.key + " lost=" + row.lost + " winIn=" + row.turnsUntilWin + " enemyEscapeDelta=" + row.delta.enemyDelta + " | " + row.reason);
}

console.log("");
console.log("Bot/personality summary");
for (const [key, entry] of [...botOutcomes.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
  console.log("  " + key + ": turns=" + entry.turns + " winShare=" + pct(entry.wins, entry.turns) + " avgEscapeDelta=" + (entry.escapeDelta / entry.turns).toFixed(3) + " materialLost/turn=" + (entry.materialLost / entry.turns).toFixed(3));
}
