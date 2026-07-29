use std::io::{self, BufRead};
use wave_field_engine::{
    AiTurnOptions, GameState, PieceType, Player, Position, apply_closest_playable_hint, apply_move,
    apply_tuning, begin_turn, evaluate_field, get_legal_moves, get_playable_moves,
    is_king_unprotected, play_heuristic_turn, randomize_tuning, reset_tuning, resign_in_check,
    simulate_ai_games, unstable_pieces,
};

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
