import { useEffect, useMemo, useRef, useState } from 'react';
import { Board } from '../components/Board';
import { evaluateField, evaluateTypeFields } from '../field/evaluateField';
import { ALL_ENERGY_CHANNELS } from '../field/cmykEnergy';
import type { GameState } from '../game/types';
import './selfplay-viewer.css';

type Game = {value_discount?: number; iteration?: number; max_plies?: number; max_tuning_actions?: number; game: number; seed: number; simulations: number; sequence: number; ply: number; updated: number; complete: boolean; status: string};
type ServiceStatus = {repetition_limit?: number; start_reason?: string; reboot_count?: number; unsaved_updates_repeated?: number; resumed_iteration?: number; state?: string; heartbeat?: number; start_count?: number; active_workers?: number; active_effective_batch_size?: number; microbatch_size?: number; large_batch_updates?: number; target_large_batch_updates?: number; optimizer_steps?: number; checkpoint_iteration?: number; pending_checkpoint?: number; last_start_at?: string; last_resume_at?: string; free_disk_gib?: number; error?: string};
type Frame = Game & {termination_reason?: string; state: GameState; action: {type?: string; pieceType?: string; componentIndex?: number; value?: number; pieceId?: string; destination?: {x: number; y: number}} | null};
type Tactics = {games: number; first_iteration: number | null; rescues: number; rescue_opportunities: number; rescue_rate: number | null; rescues_by_player: Record<'red' | 'blue', number>; opportunities_by_player: Record<'red' | 'blue', number>; pieces_lost_per_game: Record<'red' | 'blue', number | null>; tuning_turn_rate: number | null; tuning_change_rate: number | null; decisive_mean_plies: number | null; mean_legal_choices: number | null};
type ScoreWindow = {tactics?: Tactics; games: number; batches: number; red_wins: number; blue_wins: number; draws: number; repetition_draws: number; capped: number; decisive_rate: number | null; mean_plies: number | null; batch_seconds: number | null; games_per_hour: number | null; tunes_per_move: number | null; first_iteration: number | null; last_iteration: number | null};
type Scoreboard = {timing?: {remaining_seconds: number | null; running_seconds: number | null; batches: number}; available: boolean; updated?: number; current?: ScoreWindow; recent?: ScoreWindow; losses?: {iteration: number; loss: number; value: number; validation_loss?: number; validation_value?: number; validation_samples?: number}; evaluation?: {iteration: number; opponent_iteration: number; games: number; wins: number; losses: number; capped: number; score: number}};
const number = (value: number | null | undefined, digits = 0) => value == null ? '—' : value.toLocaleString(undefined, {maximumFractionDigits: digits, minimumFractionDigits: digits});
const duration = (seconds: number | null | undefined) => {
  if (seconds == null) return '—';
  const minutes = Math.floor(seconds / 60), hours = Math.floor(minutes / 60), days = Math.floor(hours / 24);
  return days ? `${days}d ${hours % 24}h` : hours ? `${hours}h ${minutes % 60}m` : minutes ? `${minutes}m` : '<1m';
};
const noAction = () => {};

export function SelfPlayViewer() {
  const [games, setGames] = useState<Game[]>([]);
  const [game, setGame] = useState(0);
  const [frames, setFrames] = useState<Frame[]>([]);
  const [cursor, setCursor] = useState(0);
  const [live, setLive] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [error, setError] = useState('');
  const [service, setService] = useState<ServiceStatus>({});
  const [now, setNow] = useState(Date.now());
  const [scoreboard, setScoreboard] = useState<Scoreboard>();
  const [scoreScope, setScoreScope] = useState<'current' | 'recent'>('current');
  const [statsError, setStatsError] = useState(false);
  const feed = useRef<{game: number; seed?: number; after: number}>({game: -1, after: -1});
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    let controller: AbortController;
    async function pollStats() {
      controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      try {
        const response = await fetch('/__selfplay/stats', {signal: controller.signal});
        if (!response.ok) throw new Error('Statistics unavailable');
        const data = await response.json() as Scoreboard;
        if (!stopped) {setScoreboard(data); setStatsError(false);}
      } catch { if (!stopped) setStatsError(true); }
      finally {clearTimeout(timeout);}
      if (!stopped) timer = setTimeout(pollStats, 10000);
    }
    void pollStats();
    return () => {stopped = true; clearTimeout(timer); controller?.abort();};
  }, []);
  useEffect(() => {
    let stopped = false;
    let activeController: AbortController | undefined;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      setNow(Date.now());
      const controller = new AbortController();
      activeController = controller;
      const timeout = setTimeout(() => controller.abort(), 5000);
      try {
        const response = await fetch('/__selfplay/games', {signal: controller.signal});
        if (!response.ok) throw new Error('Service unreachable. The Mac may be asleep or restarting; reconnecting…');
        const data = await response.json() as {games: Game[]; service?: ServiceStatus};
        if (stopped) return;
        setGames(data.games); setService(data.service ?? {}); setNow(Date.now());
        if (data.games.length && !data.games.some(x => x.game === game)) {
          setGame(data.games[0].game);
          return;
        }
        if (feed.current.game !== game) {
          feed.current = {game, after: -1}; setFrames([]); setCursor(0);
        }
        const selectedGame = data.games.find(x => x.game === game);
        const previous = feed.current;
        // A replay stays pinned when the producer reuses its game slot. The
        // trainer can finish many batches while this browser plays one slowly.
        const replacedReplay = !live && previous.seed !== undefined && selectedGame?.seed !== previous.seed;
        if (selectedGame && !replacedReplay) {
          const result = await fetch(`/__selfplay/game/${game}?seed=${previous.seed ?? ''}&after=${previous.after}`, {signal: controller.signal});
          if (!result.ok) throw new Error('Waiting for the game feed. Retrying…');
          const update = await result.json() as {seed: number; reset: boolean; rows: Frame[]};
          if (stopped) return;
          if (update.rows.length && (!update.reset || live || previous.seed === undefined)) {
            feed.current = {game, seed: update.seed, after: update.rows[update.rows.length - 1].sequence};
            setFrames(old => update.reset ? update.rows : [...old, ...update.rows]);
          }
        }
        setError('');
      } catch (e) {
        if (!stopped) setError(['AbortError', 'TypeError'].includes((e as Error).name) ? 'Service unreachable. The Mac may be asleep or restarting; reconnecting…' : String((e as Error).message));
      } finally { clearTimeout(timeout); }
      if (!stopped) timer = setTimeout(poll, 1500);
    }
    void poll();
    return () => { stopped = true; clearTimeout(timer); activeController?.abort(); };
  }, [game, live]);
  useEffect(() => {
    if (!live || !games.find(x => x.game === game)?.complete) return;
    const active = games.filter(x => !x.complete).sort((a, b) => (b.iteration ?? 0) - (a.iteration ?? 0) || b.updated - a.updated)[0];
    if (active) setGame(active.game);
  }, [games, game, live]);
  useEffect(() => { if (live) setCursor(Math.max(0, frames.length - 1)); }, [frames, live]);
  useEffect(() => {
    if (!playing || live) return;
    const timer = setInterval(() => setCursor(x => Math.min(x + 1, frames.length - 1)), speed);
    return () => clearInterval(timer);
  }, [playing, live, speed, frames.length]);
  const frame = frames[cursor];
  const state = frame?.state;
  const field = useMemo(() => state ? evaluateField(state) : [], [state]);
  const typeFields = useMemo(() => state ? evaluateTypeFields(state) : null, [state]);
  const latest = games.find(x => x.game === game);
  const age = latest ? Math.max(0, Math.floor((now - latest.updated * 1000) / 1000)) : 0;
  const action = frame?.action;
  const description = action?.type === 'tune'
    ? `Tune ${action.pieceType} · component ${(action.componentIndex ?? 0) + 1} → ${action.value}`
    : action?.destination ? `${action.pieceId} → ${String.fromCharCode(65 + action.destination.x)}${action.destination.y + 1}`
    : frame?.termination_reason === 'threefold_repetition' ? 'Draw by threefold repetition'
    : frame?.complete ? (frame.status === 'playing' ? `${frame.max_plies ?? 600}-ply cap reached` : frame.status.replace('-won', ' wins')) : 'Initial position';
  const seek = (index: number) => { setLive(false); setPlaying(false); setCursor(Math.max(0, Math.min(frames.length - 1, index))); };
  const scores = scoreboard?.[scoreScope];
  const tactics = scores?.tactics?.games ? scores.tactics : undefined;
  const losses = scoreboard?.losses;
  const evaluation = scoreboard?.evaluation;
  return <main className="spectator">
    <header className="spectator-heading"><h1>Self-play, live</h1><a href={`http://${window.location.hostname}:5050/#/experiments/1/runs/e4c214049f904af0bad02044b9bfedd3/model-metrics`}>Training charts ↗</a></header>
    <section className="spectator-progress" aria-label="Training progress"><span>Steps</span><div className="spectator-progress-numbers"><strong>{number(service.large_batch_updates)} <small>/ {number(service.target_large_batch_updates ?? 100000)}</small></strong><p><span title="Current service session, excluding detected sleep/inactivity gaps">Running {duration(scoreboard?.timing?.running_seconds)}</span><span title={`Estimated awake compute time from ${scoreboard?.timing?.batches ?? 0} recent completed batches, including self-play, neural training, validation and evaluation. Checkpoint disk overhead is not included.`}>{scoreboard?.timing?.remaining_seconds === 0 ? 'Complete' : scoreboard?.timing?.remaining_seconds == null ? 'Estimating remaining time…' : `~${duration(scoreboard.timing.remaining_seconds)} remaining (awake)`}</span></p></div></section>
    <section className="spectator-layout">
      <div className="spectator-board">{state && typeFields ? <Board state={state} field={field} typeFields={typeFields} continuousField showTypeSums={false} energyView={false} energyChannels={ALL_ENERGY_CHANNELS} locked onSelect={noAction} onMove={noAction} onResign={noAction} onHint={noAction} onToggleEnergyChannel={noAction} /> : <div className="spectator-empty"><h2>Waiting for self-play</h2><p>The next batch of training workers will start broadcasting here automatically.</p></div>}</div>
      <aside className="spectator-panel">
        <div className="spectator-status"><span className={latest && !latest.complete && age < 30 ? 'spectator-dot active' : 'spectator-dot'} />{!live && frame ? 'Watching buffered game' : !latest ? 'Waiting for next batch' : latest.complete ? 'Game complete' : age > 30 ? 'Waiting for next action' : 'Generating self-play'}</div>
        <label>Training game<select value={game} onChange={e => {setGame(Number(e.target.value)); setLive(true); setPlaying(false);}}>{(games.length ? games : [{game: 0}]).map(x => <option key={x.game} value={x.game}>Game {x.game + 1}{'iteration' in x && x.iteration ? ` · iteration ${x.game === game && !live && frame?.iteration ? frame.iteration : x.iteration}` : ''}</option>)}</select></label>
        <div className="spectator-stats"><div><span>Ply</span><strong>{frame?.ply ?? '—'}<small> / {frame?.max_plies ?? 600}</small></strong></div><div><span>Search</span><strong>{latest?.simulations ?? 128}<small> sims</small></strong></div></div>
        {state && <><div className="spectator-turn">{frame?.termination_reason === 'threefold_repetition' ? 'Draw by repetition' : state.status === 'playing' ? `${state.currentPlayer === 'red' ? 'Red' : 'Blue'} to act` : state.status.replace('-won', ' wins')}<span>Red {state.pieces.filter(x => x.owner === 'red').length} · Blue {state.pieces.filter(x => x.owner === 'blue').length} pieces</span></div><p className="spectator-action">{description}</p></>}
        <div className="spectator-playback">
        <button className={live ? 'spectator-live selected' : 'spectator-live'} onClick={() => {setLive(true);setPlaying(false);}}>● Live</button>
        <input aria-label="Replay position" type="range" min={0} max={Math.max(0, frames.length - 1)} value={cursor} onChange={e => seek(Number(e.target.value))} disabled={!frames.length} />
        <div className="spectator-controls"><button onClick={() => seek(0)} disabled={!frames.length} aria-label="Rewind">↤</button><button onClick={() => seek(cursor - 1)} disabled={!frames.length} aria-label="Previous action">←</button><button onClick={() => {if (cursor >= frames.length - 1) setCursor(0);setLive(false);setPlaying(!playing);}} disabled={!frames.length}>{playing ? 'Pause' : 'Play'}</button><button onClick={() => seek(cursor + 1)} disabled={!frames.length} aria-label="Next action">→</button></div>
        <select aria-label="Playback speed" value={speed} onChange={e => setSpeed(Number(e.target.value))}><option value={1000}>1×</option><option value={500}>2×</option><option value={200}>5×</option></select>
        </div>
        <p className="spectator-note">{frames.length ? `Action ${cursor}/${frames.length - 1}${live ? ` · ${age}s ago` : ' · buffered replay'}` : 'No moves yet'}</p>
        {!live && <p className="spectator-note">Playback speed is independent of training. Live rejoins the latest game.</p>}
        <details className="spectator-rules"><summary>Game settings</summary><p className="spectator-note">Discount {latest?.value_discount ?? .997} per piece move. Tunes are separate actions, up to {latest?.max_tuning_actions ?? 20} per turn. {latest?.max_plies ?? 600}-ply cap; threefold repetition is a draw.</p></details>
        {error && <p role="status">{error}</p>}
      </aside>
    </section>
    <section className="spectator-scoreboard" aria-label="Self-play statistics">
      <header className="scoreboard-heading"><div><span className="spectator-eyebrow">THE SCOREBOARD</span><h2>Self-play results</h2><p>{scores?.games ? `Iterations ${scores.first_iteration}–${scores.last_iteration} · ${number(scores.games)} completed games` : 'Waiting for completed training batches'}{statsError ? ' · Statistics reconnecting…' : ''}</p></div><div className="scoreboard-scope" aria-label="Statistics window"><button aria-pressed={scoreScope === 'current'} onClick={() => setScoreScope('current')}>Current rules</button><button aria-pressed={scoreScope === 'recent'} onClick={() => setScoreScope('recent')}>Last 20 batches</button></div></header>
      <div className="scoreboard-teams">
        <div className="scoreboard-team red"><span><i />Red wins</span><strong>{number(scores?.red_wins)}</strong><small>{scores?.games ? `${number(scores.red_wins / scores.games * 100, 1)}% of games` : '—'}</small></div>
        <div className="scoreboard-team draws"><span>Draws</span><strong>{number(scores?.draws)}</strong><small>{number(scores?.repetition_draws)} repetition · {number(scores?.capped)} capped</small></div>
        <div className="scoreboard-team blue"><span><i />Blue wins</span><strong>{number(scores?.blue_wins)}</strong><small>{scores?.games ? `${number(scores.blue_wins / scores.games * 100, 1)}% of games` : '—'}</small></div>
      </div>
      <div className="scoreboard-metrics">
        <div><span>Decisive games</span><strong>{number(scores?.decisive_rate == null ? null : scores.decisive_rate * 100, 1)}<small>%</small></strong></div>
        <div><span>Average game</span><strong>{number(scores?.mean_plies, 1)}<small> plies</small></strong></div>
        <div><span>Generation pace</span><strong>{number(scores?.games_per_hour)}<small> games/h</small></strong></div>
        <div><span>Average batch</span><strong>{number(scores?.batch_seconds, 1)}<small> sec</small></strong></div>
        <div><span>Tunes / move</span><strong>{number(scores?.tunes_per_move, 2)}</strong></div>
        <div><span>Large-batch updates</span><strong>{number(service.large_batch_updates)}<small> / {number(service.target_large_batch_updates)}</small></strong></div>
      </div>
      <div className="scoreboard-learning">
        <div><h3>Latest learning losses <small>iteration {losses?.iteration ?? '—'}</small></h3><dl><div><dt>Training</dt><dd>{number(losses?.loss, 3)}</dd></div><div><dt>Validation</dt><dd>{number(losses?.validation_loss, 3)}</dd></div><div><dt>Value / validation</dt><dd>{number(losses?.value, 3)} / {number(losses?.validation_value, 3)}</dd></div></dl><p>{number(losses?.validation_samples)} held-out positions · rolling validation set</p></div>
        <div><h3>Frozen checkpoint match <small>{evaluation ? `${evaluation.iteration} vs ${evaluation.opponent_iteration}` : 'awaiting evaluation'}</small></h3><dl><div><dt>Score</dt><dd>{number(evaluation ? evaluation.score * 100 : null, 1)}%</dd></div><div><dt>Wins / losses</dt><dd>{number(evaluation?.wins)} / {number(evaluation?.losses)}</dd></div><div><dt>Capped</dt><dd>{number(evaluation?.capped)} / {number(evaluation?.games)}</dd></div></dl><p>Network without MCTS · 600-ply cap · no repetition cutoff</p></div>
      </div>
      <section className="scoreboard-tactics" aria-label="Tactical statistics">
        <header><h3>Tactical play</h3><p>{tactics?.games ? `${number(tactics.games)} measured games · since iteration ${tactics.first_iteration}` : 'Collecting tactical measurements from new games…'}</p></header>
        <div className="scoreboard-metrics">
          <div><span>Rescue rate</span><strong>{number(tactics?.rescue_rate == null ? null : tactics.rescue_rate * 100, 1)}<small>%</small></strong><p>{number(tactics?.rescues)} / {number(tactics?.rescue_opportunities)} threatened turns saved</p><p>Red {number(tactics?.rescues_by_player.red)}/{number(tactics?.opportunities_by_player.red)} · Blue {number(tactics?.rescues_by_player.blue)}/{number(tactics?.opportunities_by_player.blue)}</p></div>
          <div><span>Pieces lost / game</span><strong><b className="tactic-red">{number(tactics?.pieces_lost_per_game.red, 2)}</b><small> / </small><b className="tactic-blue">{number(tactics?.pieces_lost_per_game.blue, 2)}</b></strong><p>Red / Blue · losses on either side's turn</p></div>
          <div><span>Turns with tuning</span><strong>{number(tactics?.tuning_turn_rate == null ? null : tactics.tuning_turn_rate * 100, 1)}<small>%</small></strong><p>At least one tuning action that turn</p></div>
          <div><span>Net tuning changes</span><strong>{number(tactics?.tuning_change_rate == null ? null : tactics.tuning_change_rate * 100, 1)}<small>%</small></strong><p>Net changed components / tuning actions</p></div>
          <div><span>Legal moves / position</span><strong>{number(tactics?.mean_legal_choices, 1)}</strong><p>Average over MCTS decision positions</p></div>
          <div><span>Decisive game length</span><strong>{number(tactics?.decisive_mean_plies, 1)}<small> plies</small></strong><p>Games ending in a win, excluding draws</p></div>
        </div>
        <p className="scoreboard-tactics-note">A rescue means every own piece unstable at the start of a turn survives and becomes stable by the end of that turn. Tactical measurements begin with the updated collector; earlier games are excluded. No opportunities is shown as —.</p>
      </section>
      <p className="scoreboard-footnote">Results include held-out games. Generation pace excludes neural training and evaluation time. Learning losses and checkpoint matches show the latest reading, independent of the selected results window.</p>
    </section>
    {service.state && <section className="spectator-service spectator-service-bottom" aria-label="Training service status">
      <div><strong>{service.heartbeat && now / 1000 - service.heartbeat > 35 ? 'Service heartbeat stale — possibly asleep' : ({waiting_for_checkpoint: 'Service running · checkpoint handoff pending', training: 'Training service running', recovering: 'Recovering training', target_complete: '100,000 large-batch updates complete', paused_low_disk: 'Training paused · low disk space', stopped: 'Service stopped'}[service.state] ?? service.state)}</strong>
        <span>{service.active_workers ?? 8} workers · effective batch {(service.active_effective_batch_size ?? 4096).toLocaleString()} · microbatch {service.microbatch_size ?? 128}</span></div>
      <div><span>Lifetime optimizer updates: {(service.optimizer_steps ?? 0).toLocaleString()}</span><span>{games.filter(x => !x.complete).length} games generating · {games.filter(x => x.complete).length} finished</span></div>
      <p>{service.pending_checkpoint && service.state === 'waiting_for_checkpoint' ? `Switching to 8 workers and batch 4,096 after checkpoint ${service.pending_checkpoint}. ` : ''}{service.repetition_limit === 3 ? 'Threefold repetition ends self-play as a draw. ' : ''}Recovery checkpoint: iteration {service.checkpoint_iteration ?? '—'}. Service starts: {service.start_count ?? 1}; computer reboots: {service.reboot_count ?? 0}. {service.last_start_at ? `Last start: ${new Date(service.last_start_at).toLocaleString()} (${service.start_reason ?? 'service restart'}). ` : ''}{service.unsaved_updates_repeated ? `Last recovery omitted ${service.unsaved_updates_repeated} updates that had not reached a checkpoint. ` : ''}{service.last_resume_at ? `Last inactivity resume: ${new Date(service.last_resume_at).toLocaleString()}. ` : ''}Pauses during Mac sleep; resumes on wake. After reboot, starts when you log in.{service.error ? ` ${service.error}` : ''}</p>
    </section>}
  </main>;
}
