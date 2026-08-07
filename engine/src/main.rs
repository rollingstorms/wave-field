use std::io::{self, BufRead};

use serde_json::json;
use wave_field_engine::{
    AiTurnOptions, GameState, PieceType, Player, Position, RolloutAction, RolloutSessionStore,
    apply_closest_playable_hint, apply_move, apply_tuning, begin_turn, evaluate_field,
    generate_random_training_batch, get_legal_moves, get_playable_moves, is_king_unprotected,
    play_easy_turn, play_heuristic_turn, profile_random_games, profile_random_training_batch,
    randomize_tuning, reset_tuning, resign_in_check, simulate_ai_games, simulate_random_games,
    simulate_random_lean_games, unstable_pieces,
};

fn main() {
    let mut rollout_sessions = RolloutSessionStore::default();
    for line in io::stdin().lock().lines() {
        let line = line.expect("read request");
        let request: serde_json::Value = serde_json::from_str(&line).expect("valid request JSON");
        let method = request["method"].as_str().expect("method");
        if let Some(result) = handle_rollout_request(method, &request, &mut rollout_sessions) {
            println!("{}", serde_json::to_string(&result).unwrap());
            continue;
        }
        let state: GameState =
            serde_json::from_value(request["state"].clone()).expect("valid state");
        let result = match method {
            "evaluateField" => serde_json::to_value(evaluate_field(&state)).unwrap(),
            "legalMoves" => {
                let field = evaluate_field(&state);
                serde_json::to_value(get_legal_moves(
                    request["pieceId"].as_str().unwrap(),
                    &state,
                    &field,
                ))
                .unwrap()
            }
            "applyMove" => {
                let destination: Position =
                    serde_json::from_value(request["destination"].clone()).unwrap();
                let analyze = request["analyzeCheckmate"].as_bool().unwrap_or(true);
                serde_json::to_value(apply_move(
                    request["pieceId"].as_str().unwrap(),
                    destination,
                    state,
                    analyze,
                ))
                .unwrap()
            }
            "playableMoves" => serde_json::to_value(get_playable_moves(
                request["pieceId"].as_str().unwrap(),
                &state,
            ))
            .unwrap(),
            "playableActions" => {
                let actions = state
                    .pieces
                    .iter()
                    .filter(|piece| piece.owner == state.current_player)
                    .flat_map(|piece| {
                        get_playable_moves(&piece.id, &state)
                            .into_iter()
                            .map(|destination| json!({ "pieceId": piece.id, "destination": destination }))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                serde_json::to_value(actions).unwrap()
            }
            "beginTurn" => {
                let analyze = request["analyzeCheckmate"].as_bool().unwrap_or(true);
                serde_json::to_value(begin_turn(state, analyze)).unwrap()
            }
            "applyTuning" => {
                let player: Player = serde_json::from_value(request["player"].clone()).unwrap();
                let piece_type: PieceType =
                    serde_json::from_value(request["pieceType"].clone()).unwrap();
                serde_json::to_value(apply_tuning(
                    player,
                    piece_type,
                    request["componentIndex"].as_u64().unwrap() as usize,
                    request["value"].as_i64().unwrap() as i8,
                    state,
                ))
                .unwrap()
            }
            "resignInCheck" => serde_json::to_value(resign_in_check(state)).unwrap(),
            "applyClosestPlayableHint" => {
                serde_json::to_value(apply_closest_playable_hint(state)).unwrap()
            }
            "resetTuning" => serde_json::to_value(reset_tuning(state)).unwrap(),
            "randomizeTuning" => {
                let rolls: [f64; 4] = serde_json::from_value(request["rolls"].clone()).unwrap();
                serde_json::to_value(randomize_tuning(state, rolls)).unwrap()
            }
            "unstablePieceIds" => {
                let player: Player = serde_json::from_value(request["player"].clone()).unwrap();
                let field = evaluate_field(&state);
                let ids = unstable_pieces(player, &state, &field)
                    .into_iter()
                    .map(|piece| piece.id)
                    .collect::<Vec<_>>();
                serde_json::to_value(ids).unwrap()
            }
            "kingUnprotected" => {
                let player: Player = serde_json::from_value(request["player"].clone()).unwrap();
                let field = evaluate_field(&state);
                serde_json::to_value(is_king_unprotected(player, &state, &field)).unwrap()
            }
            "playHeuristicTurn" => {
                let player: Player = serde_json::from_value(request["player"].clone()).unwrap();
                serde_json::to_value(play_heuristic_turn(
                    state,
                    player,
                    AiTurnOptions {
                        seed: request["seed"].as_u64().map(|value| value as u32),
                        variety: request["variety"].as_f64(),
                        time_budget_ms: request["timeBudgetMs"].as_u64(),
                    },
                ))
                .unwrap()
            }
            "playEasyTurn" => {
                let player: Player = serde_json::from_value(request["player"].clone()).unwrap();
                serde_json::to_value(play_easy_turn(
                    state,
                    player,
                    AiTurnOptions {
                        seed: request["seed"].as_u64().map(|value| value as u32),
                        variety: request["variety"].as_f64(),
                        time_budget_ms: request["timeBudgetMs"].as_u64(),
                    },
                ))
                .unwrap()
            }
            "simulateAiGames" => serde_json::to_value(simulate_ai_games(
                &state,
                request["games"].as_u64().unwrap_or(100),
                request["maxPlies"].as_u64().unwrap_or(300),
                request["seed"].as_u64().unwrap_or(0) as u32,
                request["variety"].as_f64().unwrap_or(0.55),
                request["timeBudgetMs"].as_u64().unwrap_or(20),
            ))
            .unwrap(),
            "simulateRandomGames" => serde_json::to_value(simulate_random_games(
                &state,
                request["games"].as_u64().unwrap_or(100),
                request["maxPlies"].as_u64().unwrap_or(300),
                request["seed"].as_u64().unwrap_or(0) as u32,
            ))
            .unwrap(),
            "simulateRandomLeanGames" => serde_json::to_value(simulate_random_lean_games(
                &state,
                request["games"].as_u64().unwrap_or(100),
                request["maxPlies"].as_u64().unwrap_or(300),
                request["seed"].as_u64().unwrap_or(0) as u32,
            ))
            .unwrap(),
            "profileRandomGames" => serde_json::to_value(profile_random_games(
                &state,
                request["games"].as_u64().unwrap_or(100),
                request["maxPlies"].as_u64().unwrap_or(300),
                request["seed"].as_u64().unwrap_or(0) as u32,
            ))
            .unwrap(),
            "generateRandomTrainingBatch" => serde_json::to_value(generate_random_training_batch(
                &state,
                request["games"].as_u64().unwrap_or(100),
                request["maxPlies"].as_u64().unwrap_or(300),
                request["seed"].as_u64().unwrap_or(0) as u32,
                request["materialForCapped"].as_bool().unwrap_or(true),
            ))
            .unwrap(),
            "profileRandomTrainingBatch" => serde_json::to_value(profile_random_training_batch(
                &state,
                request["games"].as_u64().unwrap_or(100),
                request["maxPlies"].as_u64().unwrap_or(300),
                request["seed"].as_u64().unwrap_or(0) as u32,
                request["materialForCapped"].as_bool().unwrap_or(true),
            ))
            .unwrap(),
            method => panic!("unknown method: {method}"),
        };
        println!("{}", serde_json::to_string(&result).unwrap());
    }
}

fn handle_rollout_request(
    method: &str,
    request: &serde_json::Value,
    rollout_sessions: &mut RolloutSessionStore,
) -> Option<serde_json::Value> {
    let result = match method {
        "createRolloutSession" => {
            let states: Vec<GameState> =
                serde_json::from_value(request["states"].clone()).expect("valid rollout states");
            let max_plies = request["maxPlies"].as_u64().unwrap_or(160);
            serde_json::to_value(rollout_sessions.create(
                states,
                max_plies,
                request["collectPressure"].as_bool().unwrap_or(false),
            ))
            .unwrap()
        }
        "getRolloutBatch" => {
            let session_id = request["sessionId"].as_u64().expect("rollout sessionId");
            serde_json::to_value(
                rollout_sessions
                    .batch(session_id, request["profile"].as_bool().unwrap_or(false))
                    .unwrap(),
            )
            .unwrap()
        }
        "applyRolloutActions" => {
            let session_id = request["sessionId"].as_u64().expect("rollout sessionId");
            let actions: Vec<RolloutAction> =
                serde_json::from_value(request["actions"].clone()).expect("valid rollout actions");
            serde_json::to_value(rollout_sessions.apply_actions(session_id, actions).unwrap())
                .unwrap()
        }
        "finishRolloutSession" => {
            let session_id = request["sessionId"].as_u64().expect("rollout sessionId");
            serde_json::to_value(rollout_sessions.finish(session_id).unwrap()).unwrap()
        }
        _ => return None,
    };
    Some(result)
}
