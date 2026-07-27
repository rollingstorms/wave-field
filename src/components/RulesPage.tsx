import { Fragment } from "react";
import { ArrowLeft, CircleDot, Crown, Sparkles, Waves } from "lucide-react";
import { createInitialPieces, createInitialState } from "../game/initialState";
import type { Piece as PieceModel, PieceType, Player } from "../game/types";
import { Piece } from "./Piece";
import { WaveThumbnail } from "./WaveThumbnail";

interface RulesPageProps {
  onBack: () => void;
}

const pieceNames: Record<PieceType, string> = {
  pawn: "Pawn",
  rook: "Rook",
  spy: "Spy",
  king: "King",
};

const playerLabel: Record<Player, string> = {
  red: "Red",
  blue: "Blue",
};

const pieceDetails: Record<PieceType, { components: string; energy: string; active: string; home: string; default: string; movement: string; note: string }> = {
  pawn: {
    components: "1",
    energy: "1",
    active: "1",
    home: "0",
    default: "+",
    movement: "Any distance in one direction",
    note: "Simple pressure piece. It contributes no energy to its own square.",
  },
  rook: {
    components: "2",
    energy: "2",
    active: "2",
    home: "0",
    default: "+ +",
    movement: "Any distance in one direction",
    note: "Broad field shaper. It contributes no energy to its own square.",
  },
  spy: {
    components: "3",
    energy: "2",
    active: "1",
    home: "0",
    default: "+ 0 0",
    movement: "Any distance in one direction, ignoring territory",
    note: "The spy can cross hostile territory, but it is unstable there after a turn resolves.",
  },
  king: {
    components: "3",
    energy: "2",
    active: "2",
    home: "0",
    default: "0 + +",
    movement: "Any distance in one direction",
    note: "The king contributes no energy to its own square. An unstable king is trapped and must be rescued.",
  },
};

const setupPieces = createInitialPieces();
const previewState = createInitialState();

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
      {Array.from({ length: 49 }, (_, index) => {
        const x = index % 7;
        const y = Math.floor(index / 7);
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
  const redCells = new Set(["0,0", "1,0", "2,0", "0,1", "1,1", "4,1", "5,1", "6,1", "0,2", "2,2", "3,2", "5,2", "6,2", "1,3", "5,3", "6,3", "0,4", "1,4", "4,4", "6,4", "0,5", "1,5", "2,5", "6,5", "0,6", "1,6", "5,6", "6,6"]);
  const blueCells = new Set(["3,0", "4,0", "5,0", "2,1", "3,1", "2,3", "3,3", "4,3", "2,4", "3,4", "2,6", "3,6", "4,6"]);
  const playable = new Set(["2,2", "3,2", "4,2", "5,2", "3,3", "4,4"]);
  const risky = new Set(["2,4"]);
  const kingBlocked = new Set(["5,4"]);
  const pieces: Array<PieceModel> = [
    { id: "rules-blue-king", owner: "blue", type: "king", position: { x: 1, y: 2 }, unstable: true },
    { id: "rules-blue-rook", owner: "blue", type: "rook", position: { x: 2, y: 5 }, unstable: false },
    { id: "rules-blue-spy", owner: "blue", type: "spy", position: { x: 4, y: 3 }, unstable: false },
    { id: "rules-red-pawn", owner: "red", type: "pawn", position: { x: 5, y: 5 }, unstable: false },
    { id: "rules-red-rook", owner: "red", type: "rook", position: { x: 6, y: 2 }, unstable: false },
  ];

  return (
    <div className="rules-board movement-demo" aria-label="Example movement markers">
      {Array.from({ length: 49 }, (_, index) => {
        const x = index % 7;
        const y = Math.floor(index / 7);
        const key = `${x},${y}`;
        const piece = pieces.find((candidate) => candidate.position.x === x && candidate.position.y === y);
        const territory = redCells.has(key) ? "red" : blueCells.has(key) ? "blue" : "neutral";
        return (
          <div className={`rules-board-cell ${territory}`} key={key}>
            {piece && <Piece piece={piece} selected={piece.id === "rules-blue-rook"} dragging={false} />}
            {piece?.unstable && <span className="unstable">!</span>}
            {playable.has(key) && <span className="legal-dot" />}
            {risky.has(key) && (
              <>
                <span className="legal-dot risky-dot" />
                <span className="risk-marker">!</span>
              </>
            )}
            {kingBlocked.has(key) && <span className="king-block-marker">K</span>}
          </div>
        );
      })}
    </div>
  );
}

export function RulesPage({ onBack }: RulesPageProps) {
  return (
    <main className="app rules-page">
      <header className="rules-header">
        <div>
          <p className="side-label blue">HOW TO PLAY</p>
          <h1>Wave Field</h1>
        </div>
        <button className="secondary back-button" onClick={onBack}>
          <ArrowLeft size={18} />
          Play
        </button>
      </header>

      <section className="rules-hero">
        <div className="rules-copy">
          <h2>Win by trapping the opposing king.</h2>
          <p>
            Every piece emits a controllable wave pattern. Add the active patterns together
            and each square becomes Red, Neutral, or Blue territory. Moving one piece can
            reshape the whole board.
          </p>
        </div>
        <SetupBoard />
      </section>

      <section className="rules-grid">
        <article className="rules-panel">
          <Waves size={22} />
          <h2>Territory</h2>
          <p>Each wave pattern can contain both positive and negative energy. The total on a square decides who controls it. The gradient view shades each square by its normalized field magnitude.</p>
          <div className="territory-rule">
            <span className="swatch red" /> <strong>Red</strong> <small>total above 0</small>
            <span className="swatch neutral" /> <strong>Neutral</strong> <small>total equals 0</small>
            <span className="swatch blue" /> <strong>Blue</strong> <small>total below 0</small>
          </div>
        </article>

        <article className="rules-panel">
          <CircleDot size={22} />
          <h2>Your Turn</h2>
          <ol>
            <li>Tune any of your piece-type wave controls.</li>
            <li>Move exactly one piece to end the turn.</li>
          </ol>
          <p>Tuning is shared by type, so both of your rooks use the same rook settings. Each type always keeps its full active count; choosing another component at the limit replaces the least recently pressed one. The dice randomizes a valid profile.</p>
        </article>

        <article className="rules-panel rules-piece-panel">
          <Sparkles size={22} />
          <h2>Pieces</h2>
          <div className="piece-rules">
            {(["pawn", "rook", "spy", "king"] as PieceType[]).map((type) => (
              <div className="piece-rule" key={type}>
                <span className="mini-piece-slot"><MiniPiece owner="blue" type={type} /></span>
                <strong>{pieceNames[type]}</strong>
                <small>
                  {type === "pawn" && "1 component, energy 1, active 1"}
                  {type === "rook" && "2 components, energy 2, active 2"}
                  {type === "spy" && "3 components, energy 2, active 1"}
                  {type === "king" && "3 components, energy 2, active 2"}
                  {`, home ${pieceDetails[type].home}`}
                </small>
              </div>
            ))}
          </div>
        </article>

        <article className="rules-panel">
          <Crown size={22} />
          <h2>Trapped Kings</h2>
          <p>Your king must end every move on friendly or Neutral territory. A king on hostile territory is unstable and trapped. Hint applies the nearest rescuing tuning profile. If no rescue is found, the trapped player still gets the turn.</p>
        </article>
      </section>

      <section className="rules-panel piece-reference">
        <h2>Piece Reference</h2>
        <div className="piece-reference-grid">
          <strong>Piece</strong>
          <strong>Wave components</strong>
          <strong>Energy</strong>
          <strong>Active</strong>
          <strong>Home</strong>
          <strong>Default</strong>
          <strong>Movement</strong>
          {(["pawn", "rook", "spy", "king"] as PieceType[]).map((type) => (
            <Fragment key={type}>
              <span className="piece-reference-name">
                <span className="mini-piece-slot"><MiniPiece owner="blue" type={type} /></span>
                {pieceNames[type]}
              </span>
              <span>{pieceDetails[type].components}</span>
              <span>{pieceDetails[type].energy}</span>
              <span>{pieceDetails[type].active}</span>
              <span>{pieceDetails[type].home}</span>
              <span><code>{pieceDetails[type].default}</code></span>
              <span>{pieceDetails[type].movement}</span>
            </Fragment>
          ))}
        </div>
        <div className="piece-notes">
          {(["pawn", "rook", "spy", "king"] as PieceType[]).map((type) => (
            <p key={`${type}-note`}><strong>{pieceNames[type]}:</strong> {pieceDetails[type].note}</p>
          ))}
        </div>
      </section>

      <section className="rules-panel wave-patterns">
        <div>
          <h2>Wave Patterns</h2>
          <p>
            These thumbnails show the default Blue contribution shape for one piece placed
            in the center. Each pattern may mix positive and negative energy; the outlined
            center is the piece's origin.
          </p>
        </div>
        <div className="wave-pattern-grid">
          {(["pawn", "rook", "spy", "king"] as PieceType[]).map((type) => (
            <div className="wave-pattern-card" key={type}>
              <WaveThumbnail state={previewState} player="blue" pieceType={type} />
              <strong>{pieceNames[type]}</strong>
              <small>
                {type === "pawn" && "Checkerboard pressure"}
                {type === "rook" && "C1 + + 0, C2 - 0 +"}
                {type === "spy" && "Single active mask by default"}
                {type === "king" && "Two active protective modes"}
              </small>
            </div>
          ))}
        </div>
      </section>

      <section className="rules-split">
        <div>
          <h2>Movement</h2>
          <p>
            Choose any horizontal, vertical, or diagonal ray and move any distance. Pawns,
            rooks, and kings can cross only friendly or Neutral empty squares. Spies ignore
            territory, but all pieces are blocked by occupied squares.
          </p>
          <ul>
            <li>White rings mark playable destinations.</li>
            <li>A yellow K marks a move blocked by king safety.</li>
            <li>A yellow diamond with ! warns that one of your pieces may be lost.</li>
          </ul>
        </div>
        <MovementBoard />
      </section>

      <section className="rules-split">
        <div>
          <h2>Instability</h2>
          <p>
            A non-king piece on hostile territory is unstable. Rescue it on your next turn by
            moving it to safety or by moving another piece so the field protects it. If it is
            still unstable after your move, it is removed. An unstable king is different:
            it is trapped and must be rescued, resigned, or undone.
          </p>
        </div>
        <div className="marker-demo" aria-label="Move marker examples">
          <span><i className="legal-dot" /> Legal move</span>
          <span><i className="king-block-marker">K</i> King unsafe</span>
          <span><i className="legal-dot risky-dot" /><b>!</b> Loss warning</span>
        </div>
      </section>

      <section className="rules-panel quick-reference">
        <h2>Quick Reference</h2>
        <dl>
          <div><dt>Blue moves first</dt><dd>{playerLabel.blue} opens the game.</dd></div>
          <div><dt>No captures by collision</dt><dd>Friendly and opposing pieces both block movement.</dd></div>
          <div><dt>Spies are special</dt><dd>They ignore hostile territory while moving, but can still become unstable.</dd></div>
          <div><dt>Move ends turn</dt><dd>Tune freely first, then commit one legal move.</dd></div>
        </dl>
      </section>
    </main>
  );
}
