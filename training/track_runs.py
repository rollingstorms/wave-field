"""Mirror existing JSONL training logs into MLflow without interrupting training."""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def event_metrics(row: dict[str, Any], state: dict[str, Any]) -> dict[str, float]:
    kind = row.get("event")
    metrics: dict[str, float] = {}
    if kind == "config":
        config = row["config"]
        state["batch_size"] = config.get("batch_size", 128)
        if config.get("effective_batch_size"):
            state["large_batch_start_step"] = config.get("large_batch_start_step", 0)
        if "optimizer_steps_start" in config:
            state["optimizer_steps"] = config["optimizer_steps_start"]
        for key in ("max_plies", "max_tuning_actions", "mcts_simulations", "mcts_repetition_limit", "mcts_value_discount", "color_augmentation", "spatial_augmentation", "mcts_depth", "mcts_top_k", "mcts_workers", "replay_capacity", "replay_sample_size", "effective_batch_size", "batch_size", "target_optimizer_steps", "large_batch_start_step"):
            if key in config:
                metrics[f"settings/{key}"] = config[key]
        state.pop("game_iteration", None)
    elif kind == "train_epoch":
        updates = row.get("optimizer_steps", math.ceil(row["samples"] / state.get("batch_size", 128)))
        state["optimizer_steps"] = state.get("optimizer_steps", 0) + updates
        for source, label in {"loss": "total", "kind": "action_kind", "policy": "move_policy", "tuning": "tuning_policy", "value": "value"}.items():
            metrics[f"loss/{label}"] = row[source]
            if f"validation_{source}" in row:
                metrics[f"validation_loss/{label}"] = row[f"validation_{source}"]
        metrics.update({"training/epoch_seconds": row["seconds"], "training/positions_per_epoch": row["samples"], "training/updates_per_epoch": updates})
        metrics['validation/positions'] = row.get('validation_samples', 0)
        metrics['training/color_augmentation'] = int(row.get('color_augmentation', False))
        metrics['training/augmentation_variants'] = row.get('augmentation_variants', 1)
        metrics['training/spatial_augmentation'] = int(row.get('spatial_augmentation', False))
        if 'original_training_positions' in row:
            metrics['training/original_positions_per_epoch'] = row['original_training_positions']
    elif kind == 'validation_split':
        metrics.update({'validation/heldout_games_per_batch': row['heldout_games'],
                        'validation/new_positions': row['new_validation_samples'],
                        'validation/positions': row['validation_samples'],
                        'validation/total_positions_seen': row['validation_total_seen']})
    elif kind == "learning_update":
        state["optimizer_steps"] = row["optimizer_steps_total"]
        state["iteration"] = row["iteration"]
        metrics["training/optimizer_steps"] = row["optimizer_steps_total"]
        metrics["progress/completed_iteration"] = row["iteration"]
        if "large_batch_start_step" in state:
            metrics["training/large_batch_optimizer_steps"] = max(0, row["optimizer_steps_total"] - state["large_batch_start_step"])
    elif kind == "train_batch":
        metrics.update({"replay/size": row.get("replay_size", 0), "replay/positions_seen": row.get("replay_total_seen", row["raw_samples"]), "training/new_positions": row["raw_samples"], "training/sampled_positions": row["training_samples"]})
    elif kind == "generate":
        summary = row.get("summary", {})
        games = summary.get("games", 0)
        if games:
            state["games_total"] = state.get("games_total", 0) + games
            wins = summary.get("wins", {})
            metrics["self_play/repetition_draw_rate"] = summary.get("repetition_draws", 0) / games
            metrics["self_play/plies_avoided_by_repetition"] = summary.get("plies_avoided_by_repetition", 0)
            tactics = summary.get('tactics')
            if tactics:
                metrics['self_play/rescues'] = tactics['rescues']
                metrics['self_play/rescue_opportunities'] = tactics['rescue_opportunities']
                if tactics['rescue_opportunities']:
                    metrics['self_play/rescue_rate'] = tactics['rescues'] / tactics['rescue_opportunities']
                for player in ('red', 'blue'):
                    metrics[f'self_play/{player}_pieces_lost_per_game'] = tactics['pieces_lost'][player] / games
                if tactics['turns']:
                    metrics['self_play/tuning_turn_rate'] = tactics['tuning_turns'] / tactics['turns']
                if tactics['tuning_actions']:
                    metrics['self_play/net_tuning_change_rate'] = tactics['net_tuning_changes'] / tactics['tuning_actions']
            metrics.update({"self_play/batch_seconds": row["seconds"], "self_play/mean_plies": summary["mean_plies"], "self_play/capped_rate": wins.get("capped", 0) / games, "self_play/red_win_rate": wins.get("red", 0) / games, "self_play/blue_win_rate": wins.get("blue", 0) / games, "self_play/games_total": state["games_total"], "self_play/positions_per_second": summary["samples"] / max(row["seconds"], 1e-9)})
            actions = summary.get("metadata", {}).get("action_kinds", {})
            metrics["self_play/tunes_per_move"] = actions.get("tune", 0) / max(actions.get("move", 0), 1)
            if "search_simulations" in row:
                metrics["self_play/search_simulations"] = row["search_simulations"]
    elif kind == "selfplay_game":
        if state.get("game_iteration") != row["iteration"]:
            state["game_iteration"] = row["iteration"]
            state["games_in_batch"] = 0
        state["games_in_batch"] += 1
        metrics.update({"live/games_finished_in_batch": state["games_in_batch"], "live/game_plies": row["plies"], "live/game_search_decisions": row["samples"]})
    elif kind == "checkpoint_eval":
        count = max(row["games"], 1)
        metrics.update({"evaluation/checkpoint_score": row["score"], "evaluation/win_rate": row["wins"] / count, "evaluation/loss_rate": row["losses"] / count, "evaluation/capped_rate": row["capped"] / count, "evaluation/opponent_iteration": row["opponent_iteration"]})
    elif kind == "baseline_eval":
        summary = row["summary"]
        count = max(summary["games"], 1)
        prefix = f"evaluation/vs_{row['opponent']}/{row['model_side']}"
        metrics[f"{prefix}/win_rate"] = summary["wins"].get(row["model_side"], 0) / count
        metrics[f"{prefix}/capped_rate"] = summary["wins"].get("capped", 0) / count
    return {key: float(value) for key, value in metrics.items() if math.isfinite(float(value))}


def read_complete_events(path: Path, offset: int, limit: int = 100) -> tuple[list[dict[str, Any]], int]:
    rows = []
    with path.open("rb") as handle:
        handle.seek(offset)
        for _ in range(limit):
            line = handle.readline()
            if not line or not line.endswith(b"\n"):
                break
            rows.append(json.loads(line))
            offset = handle.tell()
    return rows, offset


def event_timestamp(row: dict[str, Any], state: dict[str, Any], live: bool) -> int:
    if row.get("at"):
        stamp = int(datetime.fromisoformat(row["at"]).timestamp() * 1000)
    elif live:
        stamp = int(time.time() * 1000)
    else:
        # Old logs lack timestamps: reconstruct duration for backfill. Step is
        # the accurate chart axis; live records use their observed arrival time.
        kind = row.get("event")
        if kind == "config":
            state.pop("batch_started_ms", None)
        if kind in ("selfplay_game", "generate"):
            base = state.setdefault("batch_started_ms", state["clock_ms"])
            seconds = row.get("elapsed_seconds", 0) if kind == "selfplay_game" else row.get("seconds", 0)
            stamp = base + int(seconds * 1000)
            if kind == "generate":
                state.pop("batch_started_ms", None)
        else:
            stamp = state["clock_ms"] + int(row.get("seconds", 0) * 1000)
    state["clock_ms"] = max(state["clock_ms"], stamp)
    return stamp


class RunTracker:
    def __init__(self, tracking_uri: str, experiment: str, state_path: Path) -> None:
        from mlflow.tracking import MlflowClient
        self.client = MlflowClient(tracking_uri=tracking_uri)
        found = self.client.get_experiment_by_name(experiment)
        self.experiment_id = found.experiment_id if found else self.client.create_experiment(experiment)
        self.uri = tracking_uri.rstrip("/")
        self.state_path = state_path
        self.state = json.loads(state_path.read_text()) if state_path.exists() else {}

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2) + "\n")
        tmp.replace(self.state_path)

    def sync(self, directory: Path) -> int:
        from mlflow.entities import Metric, Param, RunTag
        directory = directory.resolve()
        events = directory / "events.jsonl"
        if not events.exists():
            return 0
        key = str(directory)
        if key not in self.state:
            existing = next((run for run in self.client.search_runs([self.experiment_id], max_results=1000) if run.data.tags.get("wavefield_run_dir") == key), None)
            created = int(getattr(events.stat(), "st_birthtime", events.stat().st_mtime) * 1000)
            run = existing or self.client.create_run(self.experiment_id, start_time=created, tags={"mlflow.runName": directory.name, "wavefield_run_dir": key, "mlflow.note.content": "Loss charts use optimizer updates on the Step axis. Historical timestamps are reconstructed; live points use arrival time. Capped games are unfinished, not confirmed draws."})
            self.state[key] = {"run_id": run.info.run_id, "offset": 0, "line": 0, "optimizer_steps": 0, "clock_ms": created, "caught_up": False}
        state = self.state[key]
        if "source_final_iteration" not in state:
            # Also works when upgrading a watcher whose cursor is already at EOF.
            for line in events.read_text().splitlines():
                try:
                    old = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if old.get("event") == "config":
                    state["source_final_iteration"] = old["config"].get("final_iteration", 0)
                elif old.get("event") == "saved":
                    state["source_saved_iteration"] = old["iteration"]
            state.setdefault("source_final_iteration", 0)
        if events.stat().st_size < state["offset"]:
            raise RuntimeError(f"Log was truncated: {events}. Use a new run directory or reset tracking state.")
        rows, offset = read_complete_events(events, state["offset"])
        metrics = []
        for row in rows:
            state["line"] += 1
            timestamp = event_timestamp(row, state, live=state["caught_up"])
            values = event_metrics(row, state)
            step = state.get("optimizer_steps", 0)
            metrics.extend(Metric(name, value, timestamp, step) for name, value in values.items())
            if row.get("event") == "config":
                config = row["config"]
                state["source_final_iteration"] = config.get("final_iteration", 0)
                if not state.get("configured"):
                    params = [Param(f"initial.{name}", json.dumps(value)[:5000]) for name, value in config.items()]
                    self.client.log_batch(state["run_id"], params=params)
                    state["configured"] = True
                tags = [RunTag(f"current.{name}", str(config[name])) for name in ("max_plies", "max_tuning_actions", "mcts_simulations", "mcts_depth", "replay_capacity", "rules_variant", "effective_batch_size", "batch_size", "target_optimizer_steps", "large_batch_start_step") if name in config]
                self.client.log_batch(state["run_id"], tags=tags)
                self.client.log_dict(state["run_id"], config, f"configs/event-{state['line']}.json")
            if row.get("event") == "saved":
                self.client.set_tag(state["run_id"], "latest_checkpoint", row["checkpoint"])
                state["source_saved_iteration"] = row["iteration"]
        for index in range(0, len(metrics), 500):
            self.client.log_batch(state["run_id"], metrics=metrics[index:index + 500])
        state["offset"] = offset
        if offset == events.stat().st_size:
            state["caught_up"] = True
        receipt_file = directory / "selfplay-settings.json"
        if receipt_file.exists():
            receipt = json.loads(receipt_file.read_text())
            discount = receipt["value_discount"]
            if state.get("active_value_discount") != discount:
                self.client.log_batch(state["run_id"], metrics=[
                    Metric("settings/mcts_value_discount", discount, int(time.time() * 1000), state.get("optimizer_steps", 0))
                ], tags=[RunTag("current.mcts_value_discount", str(discount)),
                         RunTag("value_discount_first_iteration", str(receipt["effective_iteration"]))])
                self.client.log_dict(state["run_id"], receipt, "configs/active-selfplay-settings.json")
                state["active_value_discount"] = discount
        service_file = directory.parent.parent / "service-status.json"
        if service_file.exists():
            service = json.loads(service_file.read_text())
            if service.get("run_dir") == str(directory):
                signature = [service.get("start_count"), service.get("inactivity_count"), service.get("state"), service.get("reboot_count")]
                if state.get("service_signature") != signature:
                    self.client.log_batch(state["run_id"], metrics=[
                        Metric("service/starts", service.get("start_count", 0), int(time.time() * 1000), state.get("optimizer_steps", 0)),
                        Metric("service/inactivity_resumes", service.get("inactivity_count", 0), int(time.time() * 1000), state.get("optimizer_steps", 0)),
                        Metric("service/computer_reboots", service.get("reboot_count", 0), int(time.time() * 1000), state.get("optimizer_steps", 0)),
                    ], tags=[RunTag("service.state", str(service.get("state"))),
                             RunTag("service.last_start", str(service.get("last_start_at"))),
                             RunTag("service.start_reason", str(service.get("start_reason"))),
                             RunTag("service.last_resume", str(service.get("last_resume_at"))),
                             RunTag("service.target_large_batch_updates", str(service.get("target_large_batch_updates")))])
                    state["service_signature"] = signature
        status_file = directory / "status.json"
        status = json.loads(status_file.read_text())["state"] if status_file.exists() else None
        if status is None and state["source_final_iteration"] and state.get("source_saved_iteration", 0) >= state["source_final_iteration"]:
            status = "complete"
        if status is not None:
            if state.get("status") != status:
                self.client.set_tag(state["run_id"], "training_status", status)
                if status == "complete":
                    self.client.set_terminated(state["run_id"], status="FINISHED")
                elif status == "failed":
                    self.client.set_terminated(state["run_id"], status="FAILED")
                elif status.startswith("stopped"):
                    self.client.set_terminated(state["run_id"], status="KILLED")
                elif status == "running":
                    self.client.update_run(state["run_id"], status="RUNNING")
                state["status"] = status
        state["url"] = f"{self.uri}/#/experiments/{self.experiment_id}/runs/{state['run_id']}"
        self.save()
        return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5050")
    parser.add_argument("--experiment", default="Wave Field Self-play")
    parser.add_argument("--state", type=Path, default=Path("training/tracking/sync-state.json"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    tracker = RunTracker(args.tracking_uri, args.experiment, args.state)
    while True:
        imported = 0
        for directory in args.run_dirs:
            try:
                count = tracker.sync(directory)
                while count == 100:
                    imported += count
                    count = tracker.sync(directory)
                imported += count
            except Exception as error:
                if not args.watch:
                    raise
                print(f"Sync failed for {directory}: {error}", flush=True)
                # Retry from the last persisted cursor, rather than a partially
                # mutated in-memory state after a network or database failure.
                tracker.state = json.loads(args.state.read_text()) if args.state.exists() else {}
        if imported:
            print(f"Imported {imported} events", flush=True)
        if not args.watch:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
