import { buildHistoryRoll } from "../game/historyRoll";
import type { GameState } from "../game/types";

interface HistoryRollProps {
  state: GameState;
}

export function HistoryRoll({ state }: HistoryRollProps) {
  const entries = buildHistoryRoll(state).reverse();
  return (
    <section className="history-roll" aria-labelledby="history-roll-title">
      <header>
        <h2 id="history-roll-title">Move Log</h2>
        <span>{entries.length} event{entries.length === 1 ? "" : "s"}</span>
      </header>
      <ol aria-label="Game history, newest first">
        <li className="current">
          <b>NOW</b>
          <span>Turn {state.turnNumber} · {state.currentPlayer.toUpperCase()}</span>
          <small>{state.message}</small>
        </li>
        {entries.map((entry) => (
          <li key={entry.number}>
            <b>#{entry.number}</b>
            <span>Turn {entry.turnNumber} · {entry.summary}</span>
            <small>{entry.details.length > 0 ? entry.details.join(" · ") : "No board or control change"}</small>
          </li>
        ))}
        <li className="start">
          <b>START</b>
          <span>Turn 1 · Blue</span>
          <small>Opening position</small>
        </li>
      </ol>
    </section>
  );
}
