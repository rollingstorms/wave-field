use crate::*;
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn parse_state(json: &str) -> GameState {
    serde_json::from_str(json).expect("valid game state JSON")
}

fn json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("serializable engine result")
}

#[wasm_bindgen]
pub fn evaluate_field_json(state_json: &str) -> String {
    json(&evaluate_field(&parse_state(state_json)))
}

#[wasm_bindgen]
pub fn legal_moves_json(piece_id: &str, state_json: &str) -> String {
    let state = parse_state(state_json);
    let field = evaluate_field(&state);
    json(&get_legal_moves(piece_id, &state, &field))
}

#[wasm_bindgen]
pub fn playable_moves_json(piece_id: &str, state_json: &str) -> String {
    json(&get_playable_moves(piece_id, &parse_state(state_json)))
}

#[wasm_bindgen]
pub fn closest_playable_configuration_json(player: &str, state_json: &str) -> String {
    let player = serde_json::from_str(&format!("\"{player}\"")).expect("valid player");
    json(&find_closest_playable_configuration(
        player,
        &parse_state(state_json),
    ))
}

#[wasm_bindgen]
pub fn apply_move_json(
    piece_id: &str,
    x: i32,
    y: i32,
    state_json: &str,
    analyze_checkmate: bool,
) -> String {
    json(&apply_move(
        piece_id,
        Position { x, y },
        parse_state(state_json),
        analyze_checkmate,
    ))
}

#[wasm_bindgen]
pub fn begin_turn_json(state_json: &str, analyze_checkmate: bool) -> String {
    json(&begin_turn(parse_state(state_json), analyze_checkmate))
}

#[wasm_bindgen]
pub fn apply_tuning_json(
    player: &str,
    piece_type: &str,
    component_index: usize,
    value: i8,
    state_json: &str,
) -> String {
    let player = serde_json::from_str(&format!("\"{player}\"")).expect("valid player");
    let piece_type = serde_json::from_str(&format!("\"{piece_type}\"")).expect("valid piece type");
    json(&apply_tuning(
        player,
        piece_type,
        component_index,
        value,
        parse_state(state_json),
    ))
}

#[wasm_bindgen]
pub fn unstable_piece_ids_json(player: &str, state_json: &str) -> String {
    let player = serde_json::from_str(&format!("\"{player}\"")).expect("valid player");
    let state = parse_state(state_json);
    let field = evaluate_field(&state);
    let ids = unstable_pieces(player, &state, &field)
        .into_iter()
        .map(|piece| piece.id)
        .collect::<Vec<_>>();
    json(&ids)
}

#[wasm_bindgen]
pub fn king_unprotected_json(player: &str, state_json: &str) -> bool {
    let player = serde_json::from_str(&format!("\"{player}\"")).expect("valid player");
    let state = parse_state(state_json);
    is_king_unprotected(player, &state, &evaluate_field(&state))
}

#[wasm_bindgen]
pub fn mark_instability_json(state_json: &str) -> String {
    let state = parse_state(state_json);
    let field = evaluate_field(&state);
    json(&mark_instability(state, &field))
}

#[wasm_bindgen]
pub fn resign_in_check_json(state_json: &str) -> String {
    json(&resign_in_check(parse_state(state_json)))
}

#[wasm_bindgen]
pub fn apply_closest_playable_hint_json(state_json: &str) -> String {
    json(&apply_closest_playable_hint(parse_state(state_json)))
}

#[wasm_bindgen]
pub fn reset_tuning_json(state_json: &str) -> String {
    json(&reset_tuning(parse_state(state_json)))
}

#[wasm_bindgen]
pub fn randomize_tuning_json(rolls_json: &str, state_json: &str) -> String {
    let rolls = serde_json::from_str(rolls_json).expect("four random rolls");
    json(&randomize_tuning(parse_state(state_json), rolls))
}
