use std::io::{self, BufRead};
use std::time::Instant;
use wave_field_engine::{
    GameState, PieceType, Player, Position, apply_closest_playable_hint, apply_move, apply_tuning,
    begin_turn, evaluate_field, get_legal_moves, get_playable_moves, is_king_unprotected,
    play_heuristic_turn, randomize_tuning, reset_tuning, resign_in_check, unstable_pieces,
    AiTurnOptions,
};

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct AiBatchSummary {
    games: u64,
    red_wins: u64,
    blue_wins: u64,
    capped: u64,
    total_plies: u64,
    mean_plies: f64,
    elapsed_ms: u128,
    plies_per_second: f64,
}

fn simulate_ai_games(
    initial_state: &GameState,
    games: u64,
    max_plies: u64,
    seed: u32,
    variety: f64,
    time_budget_ms: u64,
) -> AiBatchSummary {
    let started_at = Instant::now();
    let mut red_wins = 0;
    let mut blue_wins = 0;
    let mut capped = 0;
    let mut total_plies = 0;

    for game in 0..games {
        let mut state = initial_state.clone();
        let game_seed = seed.wrapping_add(game as u32);
        let mut plies = 0;
        while state.status == wave_field_engine::GameStatus::Playing && plies < max_plies {
            let player = state.current_player;
            state = play_heuristic_turn(
                state,
                player,
                AiTurnOptions {
                    seed: Some(game_seed),
                    variety: Some(variety),
                    time_budget_ms: Some(time_budget_ms),
                },
            );
            plies += 1;
        }
        total_plies += plies;
        match state.status {
            wave_field_engine::GameStatus::RedWon => red_wins += 1,
            wave_field_engine::GameStatus::BlueWon => blue_wins += 1,
            wave_field_engine::GameStatus::Playing => capped += 1,
        }
    }

    let elapsed_ms = started_at.elapsed().as_millis();
    AiBatchSummary {
        games,
        red_wins,
        blue_wins,
        capped,
        total_plies,
        mean_plies: if games == 0 {
            0.0
        } else {
            total_plies as f64 / games as f64
        },
        elapsed_ms,
        plies_per_second: if elapsed_ms == 0 {
            0.0
        } else {
            total_plies as f64 / (elapsed_ms as f64 / 1000.0)
        },
    }
}

fn main() {
    for line in io::stdin().lock().lines() {
        let line = line.expect("read request");
        let request: serde_json::Value = serde_json::from_str(&line).expect("valid request JSON");
        let state: GameState =
            serde_json::from_value(request["state"].clone()).expect("valid state");
        let result = match request["method"].as_str().expect("method") {
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
                let seed = request["seed"].as_u64().map(|value| value as u32);
                let variety = request["variety"].as_f64();
                let time_budget_ms = request["timeBudgetMs"].as_u64();
                serde_json::to_value(play_heuristic_turn(
                    state,
                    player,
                    AiTurnOptions {
                        seed,
                        variety,
                        time_budget_ms,
                    },
                ))
                .unwrap()
            }
            "simulateAiGames" => serde_json::to_value(simulate_ai_games(
                &state,
                request["games"].as_u64().unwrap_or(100),
                request["maxPlies"].as_u64().unwrap_or(160),
                request["seed"].as_u64().unwrap_or(0) as u32,
                request["variety"].as_f64().unwrap_or(0.55),
                request["timeBudgetMs"].as_u64().unwrap_or(20),
            ))
            .unwrap(),
            method => panic!("unknown method: {method}"),
        };
        println!("{}", serde_json::to_string(&result).unwrap());
    }
}
