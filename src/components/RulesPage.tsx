import { useLayoutEffect, useRef } from "react";
import { ArrowLeft, Bot, CircleDot, CircleHelp, Crown, Grid2X2, MoveUpRight, Palette, RotateCcw, Sparkles, Undo2, Waves, Wrench } from "lucide-react";
import { sampleBoardDisplay } from "../field/boardDisplay";
import { boardDisplayColor } from "../field/continuousColor";
import { evaluateField } from "../field/evaluateField";
import { projectFieldValue } from "../field/projection";
import { createInitialState } from "../game/initialState";
import { getLegalMoves } from "../game/movement";
import { PIECE_DISPLAY_NAMES, PIECE_INITIALS, PIECE_TYPES, pieceName, pieceNamePlural } from "../game/pieceLabels";
import { applyMove, getPlayableMoves } from "../game/rules";
import type { Piece as PieceModel, PieceType, Player } from "../game/types";
import { Piece } from "./Piece";

interface RulesPageProps {
  onBack: () => void;
  sparse?: boolean;
}

const playerLabel: Record<Player, string> = {
  red: "Red",
  blue: "Blue",
};
const rulesBoardSize = 7;

const pieceQuickNotes: Record<PieceType, string> = {
  pawn: "Checkerboard energy.",
  rook: "Two energies that overlap.",
  spy: "Switch between two opposite Diamond core phases, and move through hostile territory.",
  king: "Must always end in stable territory.",
};

const setupPieces: PieceModel[] = [
  { id: "blue-rook-1", owner: "blue", type: "rook", position: { x: 2, y: 0 }, unstable: false },
  { id: "blue-king-1", owner: "blue", type: "king", position: { x: 3, y: 0 }, unstable: false },
  { id: "blue-rook-2", owner: "blue", type: "rook", position: { x: 4, y: 0 }, unstable: false },
  { id: "blue-pawn-1", owner: "blue", type: "pawn", position: { x: 2, y: 1 }, unstable: false },
  { id: "blue-spy-1", owner: "blue", type: "spy", position: { x: 3, y: 1 }, unstable: false },
  { id: "blue-pawn-2", owner: "blue", type: "pawn", position: { x: 4, y: 1 }, unstable: false },
  { id: "red-pawn-1", owner: "red", type: "pawn", position: { x: 2, y: 5 }, unstable: false },
  { id: "red-spy-1", owner: "red", type: "spy", position: { x: 3, y: 5 }, unstable: false },
  { id: "red-pawn-2", owner: "red", type: "pawn", position: { x: 4, y: 5 }, unstable: false },
  { id: "red-rook-1", owner: "red", type: "rook", position: { x: 2, y: 6 }, unstable: false },
  { id: "red-king-1", owner: "red", type: "king", position: { x: 3, y: 6 }, unstable: false },
  { id: "red-rook-2", owner: "red", type: "rook", position: { x: 4, y: 6 }, unstable: false },
];
const selectedMovementPieceId = "rules-blue-spy";

function createMovementDemoState() {
  const state = createInitialState();
  state.variant = "classic";
  state.currentPlayer = "blue";
  state.selectedPieceId = selectedMovementPieceId;
  state.pieces = [
    { id: "rules-blue-king", owner: "blue", type: "king", position: { x: 0, y: 0 }, unstable: false },
    { id: selectedMovementPieceId, owner: "blue", type: "spy", position: { x: 1, y: 1 }, unstable: false },
    { id: "rules-blue-pawn", owner: "blue", type: "pawn", position: { x: 3, y: 3 }, unstable: true },
    { id: "rules-red-pawn", owner: "red", type: "pawn", position: { x: 0, y: 1 }, unstable: false },
    { id: "rules-red-rook", owner: "red", type: "rook", position: { x: 3, y: 4 }, unstable: false },
    { id: "rules-red-king", owner: "red", type: "king", position: { x: 6, y: 6 }, unstable: false },
  ];
  state.components.blue.king = [0, 0];
  state.components.blue.spy = [1];
  state.components.blue.pawn = [0];
  state.components.red.pawn = [0];
  state.components.red.rook = [0, 0];
  state.components.red.king = [1, 1];
  return state;
}

function pieceAt(x: number, y: number) {
  return setupPieces.find((piece) => piece.position.x === x && piece.position.y === y);
}

function MiniPiece({ owner, type, id = `${owner}-${type}-rules` }: { owner: Player; type: PieceType; id?: string }) {
  const piece: PieceModel = { id, owner, type, position: { x: 0, y: 0 }, unstable: false };
  return <Piece piece={piece} selected={false} dragging={false} />;
}

function SetupBoard() {
  return (
    <div className="rules-board" aria-label="Starting position diagram">
      {Array.from({ length: rulesBoardSize * rulesBoardSize }, (_, index) => {
        const x = index % rulesBoardSize;
        const y = Math.floor(index / rulesBoardSize);
        const piece = pieceAt(x, y);
        return (
          <div className={`rules-board-cell ${piece?.owner ?? ""}`} key={`${x}-${y}`}>
            {piece && <MiniPiece owner={piece.owner} type={piece.type} id={piece.id} />}
          </div>
        );
      })}
    </div>
  );
}

function MovementBoard() {
  const state = createMovementDemoState();
  const field = evaluateField(state);
  const playable = new Set(getPlayableMoves(selectedMovementPieceId, state, field).map((move) => `${move.x},${move.y}`));
  const reachable = getLegalMoves(selectedMovementPieceId, state, field);
  const ownPieceIds = new Set(state.pieces.filter((piece) => piece.owner === state.currentPlayer).map((piece) => piece.id));
  const risky = new Set([...playable].filter((key) => {
    const [x, y] = key.split(",").map(Number);
    const result = applyMove(selectedMovementPieceId, { x, y }, state);
    if (!result.ok) return false;
    const remainingIds = new Set(result.state.pieces.map((piece) => piece.id));
    return [...ownPieceIds].some((id) => !remainingIds.has(id));
  }));
  const kingBlocked = new Set(reachable.flatMap((move) => {
    const key = `${move.x},${move.y}`;
    if (playable.has(key)) return [];
    const result = applyMove(selectedMovementPieceId, move, state);
    return result.reason?.toLowerCase().includes("big hat unprotected")
      || result.reason?.toLowerCase().includes("king unprotected")
      ? [key]
      : [];
  }));

  return (
    <div className="rules-board movement-demo" aria-label="Example movement markers">
      {Array.from({ length: rulesBoardSize * rulesBoardSize }, (_, index) => {
        const x = index % rulesBoardSize;
        const y = Math.floor(index / rulesBoardSize);
        const key = `${x},${y}`;
        const piece = state.pieces.find((candidate) => candidate.position.x === x && candidate.position.y === y);
        const territory = projectFieldValue(field[y][x]);
        return (
          <div className={`rules-board-cell ${territory}`} key={key}>
            {piece && <Piece piece={piece} selected={piece.id === selectedMovementPieceId} dragging={false} />}
            {piece?.unstable && <span className="unstable">!</span>}
            {playable.has(key) && <span className="legal-dot" />}
            {risky.has(key) && (
              <span className="legal-dot risky-dot" />
            )}
            {kingBlocked.has(key) && <span className="king-block-marker">{PIECE_INITIALS.king}</span>}
          </div>
        );
      })}
    </div>
  );
}

function ToolbarKey() {
  return (
    <div className="toolbar-key" aria-label="Bottom toolbar key">
      <span><CircleHelp size={18} /> How to play</span>
      <span><Undo2 size={18} /> Undo</span>
      <span><RotateCcw size={18} /> Restart</span>
      <span><Wrench size={18} /> Developer tools</span>
      <span><Grid2X2 size={18} /> Piece-type sums</span>
      <span><Palette size={18} /> CMYK energy view</span>
      <span><Bot size={18} /> AI mode</span>
      <span><i className="gradient-swatch" aria-hidden="true" /> Continuous field shading</span>
    </div>
  );
}

function WaveExampleBoard({ focused }: { focused: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const state = createInitialState();
  state.variant = "sparse";
  state.pieces = [
    { id: "example-blue-tower", owner: "blue", type: "rook", position: { x: 1, y: 3 }, unstable: false },
    { id: "example-red-tower", owner: "red", type: "rook", position: { x: 5, y: 3 }, unstable: false },
  ];
  state.components.blue.rook = focused ? [1, -1] : [1, 1];
  state.components.red.rook = [1, 1];
  const field = evaluateField(state);

  useLayoutEffect(() => {
    const context = canvasRef.current?.getContext("2d");
    if (!context) return;
    const samples = sampleBoardDisplay(field, 9);
    samples.forEach((row, y) => row.forEach((sample, x) => {
      context.fillStyle = boardDisplayColor(sample.value, sample.activity);
      context.fillRect(x, y, 1, 1);
    }));
  }, [focused]);

  return <div className="rules-wave-board" role="img" aria-label={focused
    ? "Wave field after Blue focuses its Tower: center square is Blue, 2 Blue support to 1 Red"
    : "Wave field before tuning: center square is Neutral, 1 Blue support to 1 Red"}>
    <canvas ref={canvasRef} width={63} height={63} aria-hidden="true" />
    <div className="rules-wave-grid" aria-hidden="true">
      {Array.from({ length: 49 }, (_, index) => {
        const x = index % 7;
        const y = 6 - Math.floor(index / 7);
        return <div className={`rules-wave-cell ${x === 3 && y === 3 ? "target" : ""}`} key={index}>
          {x === 1 && y === 3 && <MiniPiece owner="blue" type="rook" />}
          {x === 5 && y === 3 && <MiniPiece owner="red" type="rook" />}
          {x === 3 && y === 3 && <strong>{focused ? "2:1" : "1:1"}</strong>}
        </div>;
      })}
    </div>
  </div>;
}

function SparseRulesPage({ onBack }: { onBack: () => void }) {
  return <main className="app rules-page sparse-rules-page">
    <header className="rules-header">
      <div><p className="side-label blue">HOW TO PLAY</p><h1>Wave Field</h1></div>
      <button className="secondary back-button" onClick={onBack}><ArrowLeft size={18} /> Play</button>
    </header>

    <section className="rules-hero sparse-rules-hero">
      <div className="rules-copy">
        <h2>Trap the opposing Big Hat.</h2>
        <p>Pieces shape the field around them. More Blue support makes a square Blue; more Red makes it Red. Equal support is Neutral.</p>
        <div className="rules-territory-key"><span className="blue">Blue</span><span className="neutral">Neutral</span><span className="red">Red</span></div>
      </div>
      <figure className="rules-figure"><SetupBoard /><figcaption>Opening board · Blue moves first</figcaption></figure>
    </section>

    <section className="rules-big-steps" aria-label="Four steps to play">
      <article><span className="rules-step-number">01</span><Waves size={28} /><h2>Read the field</h2><p>Color shows territory. Each square can show its Red and Blue support totals.</p></article>
      <article><span className="rules-step-number">02</span><Sparkles size={28} /><h2>Tune</h2><p>Use + and − to spread or focus support. Both of your pieces of a type share one pattern.</p></article>
      <article><span className="rules-step-number">03</span><MoveUpRight size={28} /><h2>Move one piece</h2><p>Travel along a clear straight line. White rings mark playable squares. Moving ends your turn.</p></article>
      <article><span className="rules-step-number">04</span><Crown size={28} /><h2>Trap or rescue</h2><p>Make the opposing Big Hat's square hostile. They get a turn to make it safe again.</p></article>
    </section>

    <section className="rules-wave-example">
      <div className="rules-example-heading"><div><p className="side-label blue">THE FIELD IN ACTION</p><h2>One tune changes the board</h2></div><p>Two Towers sit two squares from the center. Blue changes only its Tower profile.</p></div>
      <div className="rules-wave-comparison">
        <figure><div className="rules-wave-caption"><strong>Before</strong><span>Blue ++ · Red ++</span></div><WaveExampleBoard focused={false} /><figcaption><strong>1 Blue : 1 Red</strong><span>Center is Neutral</span></figcaption></figure>
        <div className="rules-wave-arrow" aria-hidden="true">→</div>
        <figure><div className="rules-wave-caption"><strong>After tuning</strong><span>Blue +− · Red ++</span></div><WaveExampleBoard focused /><figcaption><strong>2 Blue : 1 Red</strong><span>Center is Blue</span></figcaption></figure>
      </div>
      <p className="rules-example-footnote">Tuning changes territory now. Blue still moves one piece to finish the turn.</p>
    </section>

    <section className="rules-short-reference">
      <h2>Keep in mind</h2>
      <div>
        <p><strong>Movement</strong> Most pieces cross only friendly or Neutral empty squares. Triangle Hats can cross hostile territory. Pieces block paths.</p>
        <p><strong>Unstable pieces</strong> Rescue a non-Big Hat piece on hostile territory by the end of your next turn, or it is removed.</p>
        <p><strong>Big Hat safety</strong> Your Big Hat must be safe after your move. Hint can find a rescue when it is trapped.</p>
      </div>
    </section>
  </main>;
}

export function RulesPage({ onBack, sparse = false }: RulesPageProps) {
  if (sparse) return <SparseRulesPage onBack={onBack} />;
  return (
    <main className="app rules-page">
      <header className="rules-header">
        <div>
          <p className="side-label blue">HOW TO PLAY</p>
          <h1>Wave Field<small className="variant-label">CLASSIC</small></h1>
        </div>
        <button className="secondary back-button" onClick={onBack}>
          <ArrowLeft size={18} />
          Play
        </button>
      </header>

      <section className="rules-hero">
        <div className="rules-copy">
          <h2>Trap the opposing Big Hat.</h2>
          <p>Every piece emits an invisible wave of energy across the board. Red and Blue use opposite sign orientations, and each square is the sum of every active wave reaching it. A single wave can contribute both signs in different places.</p>
        </div>
        <SetupBoard />
      </section>

      <section className="rules-panel how-to-panel">
        <div>
          <h2>How to Play</h2>
          <ol>
            <li>Move one piece along a clear horizontal, vertical, or diagonal line.</li>
            <li>Find a safe square: friendly or Neutral territory keeps your piece stable.</li>
            <li>Trap the opposing Big Hat by making its square hostile.</li>
            <li>Rescue unstable pieces before their deadline marker removes them.</li>
            <li>Adjust the entire field by tuning each piece type's wave controls before you move.</li>
          </ol>
        </div>
        <div>
          <strong>Bottom toolbar</strong>
          <ToolbarKey />
        </div>
      </section>

      <section className="rules-grid">
        <article className="rules-panel rules-piece-panel">
          <Sparkles size={22} />
          <h2>Pieces</h2>
          <p>A piece's energy is a mixture of positive and negative values across its wave pattern.</p>
          <div className="piece-rules">
            {PIECE_TYPES.map((type) => (
              <div className="piece-rule" key={type}>
                <span className="mini-piece-slot"><MiniPiece owner="blue" type={type} /></span>
                <strong>{pieceName(type)}</strong>
                <small>{pieceQuickNotes[type]}</small>
              </div>
            ))}
          </div>
        </article>

        <article className="rules-panel">
          <CircleDot size={22} />
          <h2>Your Turn</h2>
          <ol>
            <li>Tune any of your piece-type wave controls, or leave them as they are.</li>
            <li>Move one piece to commit the field and end the turn.</li>
          </ol>
          <p>Tuning is shared by type, so both of your Towers use the same Tower settings. The + and - controls activate a component in that orientation or flip an active component. At the active limit, choosing another component replaces the least recently pressed one. The dice randomizes a valid profile.</p>
        </article>

        <article className="rules-panel">
          <Waves size={22} />
          <h2>Territory</h2>
          <p>Each component pattern can contain both positive and negative cells. Add all piece waves together; the final sign decides who controls the square. Moving a piece relocates its wave origin, so one move can reshape distant territory.</p>
          <div className="territory-rule">
            <span className="swatch red" /> <strong>Red</strong> <small>total above 0</small>
            <span className="swatch neutral" /> <strong>Neutral</strong> <small>total equals 0</small>
            <span className="swatch blue" /> <strong>Blue</strong> <small>total below 0</small>
          </div>
        </article>

        <article className="rules-panel">
          <Crown size={22} />
          <h2>Trapped Big Hats</h2>
          <p>Your Big Hat must end your move on friendly or Neutral territory. If it is trapped, tune and move to rescue it. Hint finds a rescuing profile and selects a piece.</p>
        </article>
      </section>

      <section className="rules-split">
        <div>
          <h2>Movement</h2>
          <p>
            Choose any horizontal, vertical, or diagonal ray and move any distance. {pieceNamePlural("pawn")},
            {` ${pieceNamePlural("rook")}, and ${pieceNamePlural("king")} can cross only friendly or Neutral empty squares. ${pieceNamePlural("spy")} ignore`}
            territory, but all pieces are blocked by occupied squares.
          </p>
          <ul>
            <li>White rings mark playable destinations.</li>
            <li>A yellow {PIECE_INITIALS.king} marks a move blocked by Big Hat safety.</li>
            <li>A yellow dot warns that one of your pieces may be lost.</li>
          </ul>
        </div>
        <MovementBoard />
      </section>

      <section className="rules-panel">
        <Waves size={22} />
        <h2>Continuous Variant</h2>
        <p>
          The /continuous route treats each pattern as low-resolution samples of a continuous
          wave. The board still uses the sign of the summed field for stability, but pieces
          move between high-resolution points instead of square centers.
        </p>
        <ol>
          <li>Continuous field shading paints the sampled wave surface; turning it off keeps point movement active.</li>
          <li>Playable markers are precise destinations, not square centers.</li>
          <li>Continuous moves must travel at least one square of distance.</li>
          <li>Non-Spy pieces must stay in friendly or Neutral field along the sampled path.</li>
          <li>Pieces have body radius, so overlapping destinations and paths through another piece are blocked.</li>
        </ol>
      </section>

      <section className="rules-split">
        <div>
          <h2>Instability</h2>
          <p>
            A non-Big Hat piece on hostile territory is unstable. Rescue it on your next turn by
            moving it to safety or by moving another piece so the field protects it. If it is
            still unstable after your move, it is removed. An unstable Big Hat is different:
            it is trapped and must be rescued, resigned, or undone.
          </p>
        </div>
        <div className="marker-demo" aria-label="Move marker examples">
          <span><i className="legal-dot" /> Safe move</span>
          <span><i className="king-block-marker">{PIECE_INITIALS.king}</i> Big Hat unsafe</span>
          <span><i className="legal-dot risky-dot" /> Loss warning</span>
        </div>
      </section>

      <section className="rules-panel quick-reference">
        <h2>Quick Reference</h2>
        <dl>
          <div><dt>Blue moves first</dt><dd>{playerLabel.blue} opens the game.</dd></div>
          <div><dt>No captures by collision</dt><dd>Friendly and opposing pieces both block movement.</dd></div>
          <div><dt>{PIECE_DISPLAY_NAMES.spy}s are special</dt><dd>They ignore hostile territory while moving, but can still become unstable.</dd></div>
          <div><dt>Move ends turn</dt><dd>Tune freely first, then commit one safe move.</dd></div>
        </dl>
      </section>
    </main>
  );
}
