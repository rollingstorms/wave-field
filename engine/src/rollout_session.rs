use crate::{
    Field, GameState, GameStatus, Player, apply_move, evaluate_field,
    playable_training_action_indexes_with_field, training_action_to_move,
    training_legal_action_indexes, training_piece_slot, training_side,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(Default)]
pub struct RolloutSessionStore {
    next_id: u64,
    sessions: HashMap<u64, RolloutSession>,
}

struct RolloutSession {
    states: Vec<GameState>,
    plies: Vec<u64>,
    active: Vec<bool>,
    legal_action_cache: Vec<Option<Vec<usize>>>,
    max_plies: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateRolloutSessionResult {
    pub session_id: u64,
    pub games: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RolloutBatchPosition {
    pub game_index: usize,
    pub ply: u64,
    pub pieces: Vec<RolloutPiece>,
    pub field: Vec<f32>,
    pub side: Vec<f32>,
    pub legal_action_indexes: Vec<usize>,
    pub player: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RolloutPiece {
    pub slot: usize,
    pub x: i32,
    pub y: i32,
    pub unstable: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RolloutBatch {
    pub session_id: u64,
    pub active_games: usize,
    pub positions: Vec<RolloutBatchPosition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<RolloutBatchProfile>,
}

#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RolloutBatchProfile {
    pub field_ms: f64,
    pub legal_indexes_ms: f64,
    pub pieces_ms: f64,
    pub side_ms: f64,
    pub flatten_field_ms: f64,
    pub positions: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RolloutAction {
    pub game_index: usize,
    pub action_index: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApplyRolloutActionsResult {
    pub session_id: u64,
    pub applied: usize,
    pub active_games: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FinishedRolloutGame {
    pub game_index: usize,
    pub plies: u64,
    pub state: GameState,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FinishRolloutSessionResult {
    pub session_id: u64,
    pub games: Vec<FinishedRolloutGame>,
}

fn player_key(player: Player) -> &'static str {
    match player {
        Player::Red => "red",
        Player::Blue => "blue",
    }
}

fn compact_pieces(state: &GameState) -> Vec<RolloutPiece> {
    state
        .pieces
        .iter()
        .map(|piece| RolloutPiece {
            slot: training_piece_slot(&piece.id),
            x: piece.position.x,
            y: piece.position.y,
            unstable: piece.unstable,
        })
        .collect()
}

fn flattened_field(field: &Field) -> Vec<f32> {
    field
        .iter()
        .flat_map(|row| row.iter().map(|value| *value as f32))
        .collect()
}

impl RolloutSessionStore {
    pub fn create(&mut self, states: Vec<GameState>, max_plies: u64) -> CreateRolloutSessionResult {
        let session_id = self.next_id;
        self.next_id += 1;
        let games = states.len();
        let active = states
            .iter()
            .map(|state| state.status == GameStatus::Playing)
            .collect::<Vec<_>>();
        self.sessions.insert(
            session_id,
            RolloutSession {
                states,
                plies: vec![0; games],
                active,
                legal_action_cache: vec![None; games],
                max_plies,
            },
        );
        CreateRolloutSessionResult { session_id, games }
    }

    pub fn batch(
        &mut self,
        session_id: u64,
        profile_enabled: bool,
    ) -> Result<RolloutBatch, String> {
        let session = self
            .sessions
            .get_mut(&session_id)
            .ok_or_else(|| format!("unknown rollout session: {session_id}"))?;
        let mut positions = Vec::new();
        let mut profile = profile_enabled.then(RolloutBatchProfile::default);
        for game_index in 0..session.states.len() {
            if !session.active[game_index] {
                continue;
            }
            if session.plies[game_index] >= session.max_plies
                || session.states[game_index].status != GameStatus::Playing
            {
                session.active[game_index] = false;
                session.legal_action_cache[game_index] = None;
                continue;
            }
            let started_at = Instant::now();
            let field = evaluate_field(&session.states[game_index]);
            if let Some(profile) = &mut profile {
                profile.field_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }
            let started_at = Instant::now();
            let legal_action_indexes =
                playable_training_action_indexes_with_field(&session.states[game_index], &field);
            if let Some(profile) = &mut profile {
                profile.legal_indexes_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }
            if legal_action_indexes.is_empty() {
                session.active[game_index] = false;
                session.legal_action_cache[game_index] = None;
                continue;
            }
            session.legal_action_cache[game_index] = Some(legal_action_indexes.clone());
            let started_at = Instant::now();
            let pieces = compact_pieces(&session.states[game_index]);
            if let Some(profile) = &mut profile {
                profile.pieces_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }
            let started_at = Instant::now();
            let flattened = flattened_field(&field);
            if let Some(profile) = &mut profile {
                profile.flatten_field_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }
            let started_at = Instant::now();
            let side = training_side(&session.states[game_index]);
            if let Some(profile) = &mut profile {
                profile.side_ms += started_at.elapsed().as_secs_f64() * 1000.0;
                profile.positions += 1;
            }
            positions.push(RolloutBatchPosition {
                game_index,
                ply: session.plies[game_index],
                pieces,
                field: flattened,
                side,
                legal_action_indexes,
                player: player_key(session.states[game_index].current_player).to_owned(),
            });
        }
        Ok(RolloutBatch {
            session_id,
            active_games: session.active.iter().filter(|active| **active).count(),
            positions,
            profile,
        })
    }

    pub fn apply_actions(
        &mut self,
        session_id: u64,
        actions: Vec<RolloutAction>,
    ) -> Result<ApplyRolloutActionsResult, String> {
        let session = self
            .sessions
            .get_mut(&session_id)
            .ok_or_else(|| format!("unknown rollout session: {session_id}"))?;
        let mut seen = HashSet::new();
        for action in &actions {
            if action.game_index >= session.states.len() {
                return Err(format!("unknown rollout game: {}", action.game_index));
            }
            if !seen.insert(action.game_index) {
                return Err(format!(
                    "duplicate rollout game action: {}",
                    action.game_index
                ));
            }
        }

        let mut applied = 0;
        for action in actions {
            if !session.active[action.game_index] {
                continue;
            }
            let legal = session.legal_action_cache[action.game_index]
                .take()
                .unwrap_or_else(|| {
                    training_legal_action_indexes(&session.states[action.game_index])
                });
            if !legal.contains(&action.action_index) {
                return Err(format!(
                    "illegal rollout action {} for game {}",
                    action.action_index, action.game_index
                ));
            }
            let Some((piece_id, destination)) = training_action_to_move(action.action_index) else {
                return Err(format!("invalid rollout action: {}", action.action_index));
            };
            let result = apply_move(
                &piece_id,
                destination,
                session.states[action.game_index].clone(),
                false,
            );
            if !result.ok {
                return Err(result.reason.unwrap_or_else(|| {
                    format!("rollout action rejected for game {}", action.game_index)
                }));
            }
            session.states[action.game_index] = result.state;
            session.plies[action.game_index] += 1;
            applied += 1;
            session.legal_action_cache[action.game_index] = None;
            if session.plies[action.game_index] >= session.max_plies
                || session.states[action.game_index].status != GameStatus::Playing
            {
                session.active[action.game_index] = false;
            }
        }

        Ok(ApplyRolloutActionsResult {
            session_id,
            applied,
            active_games: session.active.iter().filter(|active| **active).count(),
        })
    }

    pub fn finish(&mut self, session_id: u64) -> Result<FinishRolloutSessionResult, String> {
        let session = self
            .sessions
            .remove(&session_id)
            .ok_or_else(|| format!("unknown rollout session: {session_id}"))?;
        Ok(FinishRolloutSessionResult {
            session_id,
            games: session
                .states
                .into_iter()
                .enumerate()
                .map(|(game_index, state)| FinishedRolloutGame {
                    game_index,
                    plies: session.plies[game_index],
                    state,
                })
                .collect(),
        })
    }
}
