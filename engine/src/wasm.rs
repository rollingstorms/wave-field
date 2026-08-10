use crate::api;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn evaluate_field_json(state_json: &str) -> String {
    api::evaluate_field_json(state_json).expect("valid engine API call")
}

#[wasm_bindgen]
pub fn legal_moves_json(piece_id: &str, state_json: &str) -> String {
    api::legal_moves_json(piece_id, state_json).expect("valid engine API call")
}

#[wasm_bindgen]
pub fn playable_moves_json(piece_id: &str, state_json: &str) -> String {
    api::playable_moves_json(piece_id, state_json).expect("valid engine API call")
}

#[wasm_bindgen]
pub fn closest_playable_configuration_json(player: &str, state_json: &str) -> String {
    api::closest_playable_configuration_json(player, state_json).expect("valid engine API call")
}

#[wasm_bindgen]
pub fn hint_search_json(
    player: &str,
    focused_piece_id: &str,
    state_json: &str,
    max_tuning_states: u32,
    time_budget_ms: u32,
) -> String {
    api::hint_search_json(
        player,
        focused_piece_id,
        state_json,
        max_tuning_states,
        time_budget_ms,
    )
    .expect("valid engine API call")
}

#[wasm_bindgen]
pub fn apply_move_json(
    piece_id: &str,
    x: i32,
    y: i32,
    state_json: &str,
    analyze_checkmate: bool,
) -> String {
    api::apply_move_json(piece_id, x, y, state_json, analyze_checkmate)
        .expect("valid engine API call")
}

#[wasm_bindgen]
pub fn begin_turn_json(state_json: &str, analyze_checkmate: bool) -> String {
    api::begin_turn_json(state_json, analyze_checkmate).expect("valid engine API call")
}

#[wasm_bindgen]
pub fn apply_tuning_json(
    player: &str,
    piece_type: &str,
    component_index: usize,
    value: i8,
    state_json: &str,
) -> String {
    api::apply_tuning_json(player, piece_type, component_index, value, state_json)
        .expect("valid engine API call")
}

#[wasm_bindgen]
pub fn unstable_piece_ids_json(player: &str, state_json: &str) -> String {
    api::unstable_piece_ids_json(player, state_json).expect("valid engine API call")
}

#[wasm_bindgen]
pub fn king_unprotected_json(player: &str, state_json: &str) -> bool {
    api::king_unprotected_json(player, state_json).expect("valid engine API call")
}

#[wasm_bindgen]
pub fn mark_instability_json(state_json: &str) -> String {
    api::mark_instability_json(state_json).expect("valid engine API call")
}

#[wasm_bindgen]
pub fn resign_in_check_json(state_json: &str) -> String {
    api::resign_in_check_json(state_json).expect("valid engine API call")
}

#[wasm_bindgen]
pub fn apply_closest_playable_hint_json(state_json: &str) -> String {
    api::apply_closest_playable_hint_json(state_json).expect("valid engine API call")
}

#[wasm_bindgen]
pub fn reset_tuning_json(state_json: &str) -> String {
    api::reset_tuning_json(state_json).expect("valid engine API call")
}

#[wasm_bindgen]
pub fn randomize_tuning_json(rolls_json: &str, state_json: &str) -> String {
    api::randomize_tuning_json(rolls_json, state_json).expect("valid engine API call")
}

#[wasm_bindgen]
pub fn play_heuristic_turn_json(
    player: &str,
    state_json: &str,
    seed: u32,
    variety: f64,
    time_budget_ms: u32,
) -> String {
    api::play_heuristic_turn_json(player, state_json, seed, variety, time_budget_ms)
        .expect("valid engine API call")
}

#[wasm_bindgen]
pub fn play_easy_turn_json(
    player: &str,
    state_json: &str,
    seed: u32,
    variety: f64,
    time_budget_ms: u32,
) -> String {
    api::play_easy_turn_json(player, state_json, seed, variety, time_budget_ms)
        .expect("valid engine API call")
}
