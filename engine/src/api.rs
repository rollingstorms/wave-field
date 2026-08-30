use crate::*;
use serde::Serialize;

pub type ApiResult = Result<String, String>;

const PRODUCT_INITIAL_STATE: &str = include_str!("../tests/product-initial-state.json");

fn parse_state(json: &str) -> Result<GameState, String> {
    serde_json::from_str(json).map_err(|error| format!("invalid game state JSON: {error}"))
}

fn parse_json<T>(json: &str, label: &str) -> Result<T, String>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_str(json).map_err(|error| format!("invalid {label}: {error}"))
}

fn parse_token<T>(value: &str, label: &str) -> Result<T, String>
where
    T: serde::de::DeserializeOwned,
{
    parse_json(&format!("\"{value}\""), label)
}

fn json<T: Serialize>(value: &T) -> ApiResult {
    serde_json::to_string(value)
        .map_err(|error| format!("serializing engine result failed: {error}"))
}

pub fn new_game_json() -> ApiResult {
    json(&parse_state(PRODUCT_INITIAL_STATE)?)
}

pub fn undo_json(state_json: &str) -> ApiResult {
    let mut state = parse_state(state_json)?;
    let Some(previous) = state.history.pop() else {
        state.message = "Nothing to undo".to_owned();
        return json(&state);
    };

    state.pieces = previous.pieces;
    state.current_player = previous.current_player;
    state.components = previous.components;
    state.activation_orders = previous.activation_orders;
    state.status = previous.status;
    state.selected_piece_id = previous.selected_piece_id;
    state.turn_number = previous.turn_number;
    state.definitions = previous.definitions;
    state.wave_scales = previous.wave_scales;
    state.home_energy = previous.home_energy;
    state.message = format!("{} to move", state.current_player.name());
    json(&state)
}

pub fn evaluate_field_json(state_json: &str) -> ApiResult {
    json(&evaluate_field(&parse_state(state_json)?))
}

pub fn influence_contributors_json(x: i32, y: i32, state_json: &str) -> ApiResult {
    json(&influence_contributors_at(
        Position { x, y },
        &parse_state(state_json)?,
    ))
}

pub fn all_influence_contributors_json(state_json: &str) -> ApiResult {
    json(&all_influence_contributors(&parse_state(state_json)?))
}

pub fn instability_influence_links_json(threshold: f64, state_json: &str) -> ApiResult {
    json(&instability_influence_links(
        threshold,
        &parse_state(state_json)?,
    ))
}

pub fn preview_move_json(piece_id: &str, x: i32, y: i32, state_json: &str) -> ApiResult {
    let state = parse_state(state_json)?;
    let destination = Position { x, y };
    let result = apply_move(piece_id, destination, state.clone(), false);
    if result.ok {
        return json(&result.state);
    }

    let mut preview = state;
    if let Some(piece) = preview.pieces.iter_mut().find(|piece| piece.id == piece_id) {
        piece.position = destination;
    }
    let field = evaluate_field(&preview);
    json(&mark_instability(preview, &field))
}

pub fn piece_pattern_json(player: &str, piece_type: &str, state_json: &str) -> ApiResult {
    let player: Player = parse_token(player, "player")?;
    let piece_type: PieceType = parse_token(piece_type, "piece type")?;
    let mut state = parse_state(state_json)?;
    state.pieces = vec![Piece {
        id: format!(
            "{}-{}-preview",
            player.name().to_lowercase(),
            piece_type.name()
        ),
        owner: player,
        piece_type,
        position: Position { x: 3, y: 3 },
        unstable: false,
    }];
    json(&evaluate_field(&state))
}

pub fn legal_moves_json(piece_id: &str, state_json: &str) -> ApiResult {
    let state = parse_state(state_json)?;
    let field = evaluate_field(&state);
    json(&get_legal_moves(piece_id, &state, &field))
}

pub fn playable_moves_json(piece_id: &str, state_json: &str) -> ApiResult {
    json(&get_playable_moves(piece_id, &parse_state(state_json)?))
}

pub fn closest_playable_configuration_json(player: &str, state_json: &str) -> ApiResult {
    let player = parse_token(player, "player")?;
    json(&find_closest_playable_configuration(
        player,
        &parse_state(state_json)?,
    ))
}

pub fn hint_search_json(
    player: &str,
    focused_piece_id: &str,
    state_json: &str,
    max_tuning_states: u32,
    time_budget_ms: u32,
) -> ApiResult {
    let player = parse_token(player, "player")?;
    let focused_piece_id = if focused_piece_id.is_empty() {
        None
    } else {
        Some(focused_piece_id)
    };
    json(&hint_search(
        player,
        focused_piece_id,
        &parse_state(state_json)?,
        max_tuning_states,
        time_budget_ms,
    ))
}

pub fn apply_move_json(
    piece_id: &str,
    x: i32,
    y: i32,
    state_json: &str,
    analyze_checkmate: bool,
) -> ApiResult {
    json(&apply_move(
        piece_id,
        Position { x, y },
        parse_state(state_json)?,
        analyze_checkmate,
    ))
}

pub fn begin_turn_json(state_json: &str, analyze_checkmate: bool) -> ApiResult {
    json(&begin_turn(parse_state(state_json)?, analyze_checkmate))
}

pub fn apply_tuning_json(
    player: &str,
    piece_type: &str,
    component_index: usize,
    value: i8,
    state_json: &str,
) -> ApiResult {
    let player = parse_token(player, "player")?;
    let piece_type = parse_token(piece_type, "piece type")?;
    json(&apply_tuning(
        player,
        piece_type,
        component_index,
        value,
        parse_state(state_json)?,
    ))
}

pub fn unstable_piece_ids_json(player: &str, state_json: &str) -> ApiResult {
    let player = parse_token(player, "player")?;
    let state = parse_state(state_json)?;
    let field = evaluate_field(&state);
    let ids = unstable_pieces(player, &state, &field)
        .into_iter()
        .map(|piece| piece.id)
        .collect::<Vec<_>>();
    json(&ids)
}

pub fn king_unprotected_json(player: &str, state_json: &str) -> Result<bool, String> {
    let player = parse_token(player, "player")?;
    let state = parse_state(state_json)?;
    Ok(is_king_unprotected(player, &state, &evaluate_field(&state)))
}

pub fn mark_instability_json(state_json: &str) -> ApiResult {
    let state = parse_state(state_json)?;
    let field = evaluate_field(&state);
    json(&mark_instability(state, &field))
}

pub fn resign_in_check_json(state_json: &str) -> ApiResult {
    json(&resign_in_check(parse_state(state_json)?))
}

pub fn apply_closest_playable_hint_json(state_json: &str) -> ApiResult {
    json(&apply_closest_playable_hint(parse_state(state_json)?))
}

pub fn reset_tuning_json(state_json: &str) -> ApiResult {
    json(&reset_tuning(parse_state(state_json)?))
}

pub fn randomize_tuning_json(rolls_json: &str, state_json: &str) -> ApiResult {
    let rolls = parse_json(rolls_json, "four random rolls")?;
    json(&randomize_tuning(parse_state(state_json)?, rolls))
}

pub fn play_heuristic_turn_json(
    player: &str,
    state_json: &str,
    seed: u32,
    variety: f64,
    time_budget_ms: u32,
) -> ApiResult {
    let player = parse_token(player, "player")?;
    json(&play_heuristic_turn(
        parse_state(state_json)?,
        player,
        AiTurnOptions {
            seed: Some(seed),
            variety: Some(variety),
            time_budget_ms: Some(u64::from(time_budget_ms)),
        },
    ))
}

pub fn play_hard_turn_json(
    player: &str,
    state_json: &str,
    seed: u32,
    variety: f64,
    time_budget_ms: u32,
    conversion_weight: f64,
    trap_focus: f64,
    cycle_weight: f64,
) -> ApiResult {
    let player = parse_token(player, "player")?;
    json(&play_hard_turn_tuned(
        parse_state(state_json)?,
        player,
        AiTurnOptions {
            seed: Some(seed),
            variety: Some(variety),
            time_budget_ms: Some(u64::from(time_budget_ms)),
        },
        HardBotTuning {
            conversion_weight,
            trap_focus,
            cycle_weight,
        },
    ))
}

pub fn play_easy_turn_json(
    player: &str,
    state_json: &str,
    seed: u32,
    variety: f64,
    time_budget_ms: u32,
) -> ApiResult {
    let player = parse_token(player, "player")?;
    json(&play_easy_turn(
        parse_state(state_json)?,
        player,
        AiTurnOptions {
            seed: Some(seed),
            variety: Some(variety),
            time_budget_ms: Some(u64::from(time_budget_ms)),
        },
    ))
}
