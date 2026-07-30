use crate::{
    GameState, GameStatus, apply_move, training_action_to_move, training_legal_action_indexes,
    training_observation,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Default)]
pub struct RolloutSessionStore {
    next_id: u64,
    sessions: HashMap<u64, RolloutSession>,
}

struct RolloutSession {
    states: Vec<GameState>,
    plies: Vec<u64>,
    active: Vec<bool>,
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
    pub board: Vec<f32>,
    pub side: Vec<f32>,
    pub legal_action_indexes: Vec<usize>,
    pub player: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RolloutBatch {
    pub session_id: u64,
    pub active_games: usize,
    pub positions: Vec<RolloutBatchPosition>,
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
                max_plies,
            },
        );
        CreateRolloutSessionResult { session_id, games }
    }

    pub fn batch(&mut self, session_id: u64) -> Result<RolloutBatch, String> {
        let session = self
            .sessions
            .get_mut(&session_id)
            .ok_or_else(|| format!("unknown rollout session: {session_id}"))?;
        let mut positions = Vec::new();
        for game_index in 0..session.states.len() {
            if !session.active[game_index] {
                continue;
            }
            if session.plies[game_index] >= session.max_plies
                || session.states[game_index].status != GameStatus::Playing
            {
                session.active[game_index] = false;
                continue;
            }
            let observation = training_observation(&session.states[game_index]);
            if observation.legal_action_indexes.is_empty() {
                session.active[game_index] = false;
                continue;
            }
            positions.push(RolloutBatchPosition {
                game_index,
                ply: session.plies[game_index],
                board: observation.board,
                side: observation.side,
                legal_action_indexes: observation.legal_action_indexes,
                player: observation.player,
            });
        }
        Ok(RolloutBatch {
            session_id,
            active_games: session.active.iter().filter(|active| **active).count(),
            positions,
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
            let legal = training_legal_action_indexes(&session.states[action.game_index]);
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
