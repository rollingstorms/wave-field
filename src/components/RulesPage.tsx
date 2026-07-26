import { ArrowLeft, CircleDot, Crown, Sparkles, Waves } from "lucide-react";
import { createInitialPieces } from "../game/initialState";
import type { Piece as PieceModel, PieceType, Player } from "../game/types";
import { Piece } from "./Piece";

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

const setupPieces = createInitialPieces();

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
  return (
    <div className="rules-board movement-demo" aria-label="Example movement ray">
      {Array.from({ length: 49 }, (_, index) => {
        const x = index % 7;
        const y = Math.floor(index / 7);
        const ray = y === 3 && x > 1 && x < 6;
        const blocked = x === 6 && y === 3;
        return (
          <div className={`rules-board-cell ${ray ? "ray" : ""} ${blocked ? "blocked" : ""}`} key={`${x}-${y}`}>
            {x === 1 && y === 3 && <MiniPiece owner="blue" type="rook" />}
            {ray && x !== 1 && <span className="legal-dot" />}
            {blocked && <MiniPiece owner="red" type="pawn" />}
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
          <h2>Win by checkmating the opposing king.</h2>
          <p>
            Every piece emits waves. Add the waves together and each square becomes Red,
            Neutral, or Blue territory. Moving one piece can reshape the whole board.
          </p>
        </div>
        <SetupBoard />
      </section>

      <section className="rules-grid">
        <article className="rules-panel">
          <Waves size={22} />
          <h2>Territory</h2>
          <p>Red waves count positive. Blue waves count negative. The total on a square decides who controls it.</p>
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
          <p>Tuning is shared by type, so both of your rooks use the same rook settings. Red controls show positive friendly waves; Blue controls show negative friendly waves.</p>
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
                  {type === "pawn" && "1 component, strength 1"}
                  {type === "rook" && "2 components, strength 2"}
                  {type === "spy" && "3 components, strength 1"}
                  {type === "king" && "3 components, strength 2"}
                </small>
              </div>
            ))}
          </div>
        </article>

        <article className="rules-panel">
          <Crown size={22} />
          <h2>Check</h2>
          <p>Your king must end every move on friendly or Neutral territory. If the enemy king starts its turn in hostile territory and has no rescue, it is checkmate.</p>
        </article>
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
            A non-spy piece on hostile territory is unstable. Rescue it on your next turn by
            moving it to safety or by moving another piece so the field protects it. If it is
            still unstable after your move, it is removed.
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
          <div><dt>Spies are special</dt><dd>They ignore hostile territory and are never removed for instability.</dd></div>
          <div><dt>Move ends turn</dt><dd>Tune freely first, then commit one legal move.</dd></div>
        </dl>
      </section>
    </main>
  );
}
