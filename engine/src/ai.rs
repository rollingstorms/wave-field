use crate::board::*;
use crate::field::evaluate_field;
use crate::model::*;
use crate::rules::apply_move;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

const EXACT_CANDIDATE_LIMIT: usize = 6;
const DEFAULT_TIME_BUDGET_MS: u64 = 180;
const FULL_ANALYSIS_LIMIT: usize = 3;
const REPETITION_LOOKBACK: usize = 18;
const REPEATED_STATE_PENALTY: f64 = 900.0;
const IMMEDIATE_REVERSAL_PENALTY: f64 = 500.0;
const EASY_SEARCH_DEPTH: u8 = 2;
const EASY_WIN_PENALTY: f64 = 900_000.0;
const EASY_CHECK_PENALTY: f64 = 250_000.0;
const EASY_ENEMY_CAPTURE_PENALTY: f64 = 700.0;
const EASY_OWN_LOSS_BONUS: f64 = 650.0;
const HARD_DEFAULT_TIME_BUDGET_MS: u64 = 1_500;
const HARD_MAX_DEPTH: u8 = 4;
const HARD_ROOT_ACTION_LIMIT: usize = 18;
const HARD_BRANCH_ACTION_LIMIT: usize = 10;
const HARD_QUIESCENCE_ACTION_LIMIT: usize = 6;
const HARD_PROFILE_LIMIT: usize = 14;
const HARD_QUIESCENCE_DEPTH: u8 = 1;
const HARD_ROOT_TRAP_ANALYSIS_LIMIT: usize = 8;

#[cfg(not(target_arch = "wasm32"))]
type SearchStartedAt = Instant;

#[cfg(target_arch = "wasm32")]
type SearchStartedAt = ();

#[cfg(not(target_arch = "wasm32"))]
fn search_started_at() -> SearchStartedAt {
    Instant::now()
}

#[cfg(target_arch = "wasm32")]
fn search_started_at() -> SearchStartedAt {}

#[cfg(not(target_arch = "wasm32"))]
fn deadline_reached(started_at: &SearchStartedAt, deadline: Duration) -> bool {
    started_at.elapsed() >= deadline
}

#[cfg(target_arch = "wasm32")]
fn deadline_reached(_started_at: &SearchStartedAt, _deadline: Duration) -> bool {
    false
}

#[derive(Clone, Copy, Debug)]
pub struct AiTurnOptions {
    pub seed: Option<u32>,
    pub variety: Option<f64>,
    pub time_budget_ms: Option<u64>,
}

fn material_value(piece_type: PieceType) -> f64 {
    match piece_type {
        PieceType::Pawn => 2.0,
        PieceType::Rook => 4.0,
        PieceType::Spy => 3.0,
        PieceType::King => 100.0,
    }
}

fn base_tuning_strength(piece_type: PieceType) -> usize {
    match piece_type {
        PieceType::Pawn | PieceType::Spy => 1,
        PieceType::Rook | PieceType::King => 2,
    }
}

fn tuning_strength(piece_type: PieceType, count: usize) -> usize {
    base_tuning_strength(piece_type).min(count)
}

fn profile_after_control_change(
    state: &GameState,
    player: Player,
    piece_type: PieceType,
    component_index: usize,
    value: i8,
) -> Option<Vec<i8>> {
    let mut profile = state.components.get(player).get(piece_type).clone();
    let current_value = *profile.get(component_index)?;
    if current_value == value {
        return None;
    }

    let active_indices = profile
        .iter()
        .enumerate()
        .filter_map(|(index, coefficient)| (*coefficient != 0).then_some(index))
        .collect::<Vec<_>>();
    let mut next_order = state
        .activation_orders
        .get(player)
        .get(piece_type)
        .iter()
        .copied()
        .filter(|index| active_indices.contains(index) && *index != component_index)
        .collect::<Vec<_>>();
    for index in &active_indices {
        if *index != component_index && !next_order.contains(index) {
            next_order.push(*index);
        }
    }

    if current_value == 0 && active_indices.len() >= tuning_strength(piece_type, profile.len()) {
        if !next_order.is_empty() {
            let evicted = next_order.remove(0);
            profile[evicted] = 0;
        }
    }

    profile[component_index] = value;
    Some(profile)
}

fn tuning_candidates(
    state: &GameState,
    player: Player,
    max_candidates: usize,
) -> Vec<PlayerComponents> {
    let current = state.components.get(player);
    let mut candidates = vec![current.clone()];
    let mut seen = vec![serde_json::to_string(current).expect("serializable components")];

    for piece_type in PIECE_TYPES {
        let profile = current.get(piece_type);
        for index in 0..profile.len() {
            for value in [-1, 1] {
                let Some(next_profile) =
                    profile_after_control_change(state, player, piece_type, index, value)
                else {
                    continue;
                };
                let mut next = current.clone();
                *next.get_mut(piece_type) = next_profile;
                let key = serde_json::to_string(&next).expect("serializable components");
                if !seen.contains(&key) {
                    seen.push(key);
                    candidates.push(next);
                    if candidates.len() >= max_candidates {
                        return candidates;
                    }
                }
            }
        }
    }
    candidates
}

fn compatible(player: Player, value: f64) -> bool {
    match player {
        Player::Red => value >= -FIELD_EPSILON,
        Player::Blue => value <= FIELD_EPSILON,
    }
}

fn score_state(state: &GameState, player: Player, field: &Field) -> f64 {
    if state.status == win_status(player) {
        return 1_000_000.0;
    }
    if state.status != GameStatus::Playing {
        return -1_000_000.0;
    }

    let enemy = player.opponent();
    let mut score = 0.0;

    for piece in &state.pieces {
        let direction = if piece.owner == player { 1.0 } else { -1.0 };
        score += direction * material_value(piece.piece_type) * 120.0;
        let center_distance = (piece.position.x - 3).abs() + (piece.position.y - 3).abs();
        score += direction
            * f64::from(6 - center_distance)
            * if piece.piece_type == PieceType::Spy {
                2.0
            } else {
                1.0
            };
        score += direction * get_legal_moves(&piece.id, state, field).len() as f64 * 1.5;
    }

    for row in field {
        for value in row {
            if *value > 0.0 {
                score += if player == Player::Red { 2.0 } else { -2.0 };
            }
            if *value < 0.0 {
                score += if player == Player::Blue { 2.0 } else { -2.0 };
            }
        }
    }

    score -= unstable_pieces(player, state, field)
        .into_iter()
        .filter(|piece| piece.piece_type != PieceType::King)
        .count() as f64
        * 90.0;
    score += unstable_pieces(enemy, state, field)
        .into_iter()
        .filter(|piece| piece.piece_type != PieceType::King)
        .count() as f64
        * 45.0;

    if let Some(own_king) = state
        .pieces
        .iter()
        .find(|piece| piece.owner == player && piece.piece_type == PieceType::King)
    {
        let value = field[own_king.position.y as usize][own_king.position.x as usize];
        score += if compatible(player, value) {
            value.abs().min(4.0) * 25.0
        } else {
            -10_000.0
        };
    }

    if is_king_unprotected(enemy, state, field) {
        score += 400_000.0;
    }

    score
}

fn material_points(state: &GameState, player: Player) -> f64 {
    state
        .pieces
        .iter()
        .filter(|piece| piece.owner == player)
        .map(|piece| material_value(piece.piece_type))
        .sum()
}

fn hard_score_state(state: &GameState, player: Player) -> f64 {
    let field = evaluate_field(state);
    if state.status != GameStatus::Playing {
        return score_state(state, player, &field);
    }

    let enemy = player.opponent();
    let material_balance = material_points(state, player) - material_points(state, enemy);
    let own_unstable = unstable_pieces(player, state, &field)
        .into_iter()
        .filter(|piece| piece.piece_type != PieceType::King)
        .count() as f64;
    let enemy_unstable = unstable_pieces(enemy, state, &field)
        .into_iter()
        .filter(|piece| piece.piece_type != PieceType::King)
        .count() as f64;

    score_state(state, player, &field) + material_balance * 85.0 - own_unstable * 260.0
        + enemy_unstable * 120.0
}

fn win_status(player: Player) -> GameStatus {
    match player {
        Player::Red => GameStatus::RedWon,
        Player::Blue => GameStatus::BlueWon,
    }
}

fn player_name(player: Player) -> &'static str {
    match player {
        Player::Red => "Red",
        Player::Blue => "Blue",
    }
}

fn hash_unit(value: &str) -> f64 {
    let mut hash = 2166136261_u32;
    for byte in value.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(16777619);
    }
    f64::from(hash) / 4294967295.0
}

fn choice_noise(choice: &Choice, state: &GameState, player: Player, seed: u32) -> f64 {
    hash_unit(&format!(
        "{}:{}:{}:{}:{}:{}:{:.3}",
        seed,
        state.turn_number,
        player_key(player),
        choice.piece_id,
        choice.destination.x,
        choice.destination.y,
        choice.score
    ))
}

fn player_key(player: Player) -> &'static str {
    match player {
        Player::Red => "red",
        Player::Blue => "blue",
    }
}

fn status_key(status: GameStatus) -> &'static str {
    match status {
        GameStatus::Playing => "playing",
        GameStatus::RedWon => "red-won",
        GameStatus::BlueWon => "blue-won",
    }
}

fn components_key(components: &PlayerMap<PlayerComponents>) -> String {
    fn profile_key(profile: &PlayerComponents) -> String {
        format!(
            "{}/{}/{}/{}",
            profile
                .pawn
                .iter()
                .map(i8::to_string)
                .collect::<Vec<_>>()
                .join(","),
            profile
                .rook
                .iter()
                .map(i8::to_string)
                .collect::<Vec<_>>()
                .join(","),
            profile
                .spy
                .iter()
                .map(i8::to_string)
                .collect::<Vec<_>>()
                .join(","),
            profile
                .king
                .iter()
                .map(i8::to_string)
                .collect::<Vec<_>>()
                .join(","),
        )
    }
    format!(
        "blue:{}|red:{}",
        profile_key(&components.blue),
        profile_key(&components.red)
    )
}

fn snapshot_state_key(
    pieces: &[Piece],
    current_player: Player,
    status: GameStatus,
    components: &PlayerMap<PlayerComponents>,
) -> String {
    let mut pieces = pieces
        .iter()
        .map(|piece| {
            format!(
                "{}:{}:{}:{},{}",
                piece.id,
                player_key(piece.owner),
                piece_type_key(piece.piece_type),
                piece.position.x,
                piece.position.y
            )
        })
        .collect::<Vec<_>>();
    pieces.sort();
    format!(
        "{}|{}|{}|{}",
        player_key(current_player),
        status_key(status),
        pieces.join("|"),
        components_key(components)
    )
}

fn piece_type_key(piece_type: PieceType) -> &'static str {
    match piece_type {
        PieceType::Pawn => "pawn",
        PieceType::Rook => "rook",
        PieceType::Spy => "spy",
        PieceType::King => "king",
    }
}

fn state_key(state: &GameState) -> String {
    snapshot_state_key(
        &state.pieces,
        state.current_player,
        state.status,
        &state.components,
    )
}

fn history_key(entry: &GameSnapshot) -> String {
    snapshot_state_key(
        &entry.pieces,
        entry.current_player,
        entry.status,
        &entry.components,
    )
}

fn recent_state_counts(state: &GameState) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    let start = state.history.len().saturating_sub(REPETITION_LOOKBACK);
    for entry in &state.history[start..] {
        let key = history_key(entry);
        *counts.entry(key).or_insert(0) += 1;
    }
    *counts.entry(state_key(state)).or_insert(0) += 1;
    counts
}

fn same_position(left: Position, right: Position) -> bool {
    left == right
}

fn moved_piece(
    before: &GameSnapshot,
    after: &GameSnapshot,
) -> Option<(String, Position, Position)> {
    for piece in &before.pieces {
        if let Some(next) = after
            .pieces
            .iter()
            .find(|candidate| candidate.id == piece.id)
        {
            if !same_position(piece.position, next.position) {
                return Some((piece.id.clone(), piece.position, next.position));
            }
        }
    }
    None
}

fn last_move_by_piece(state: &GameState, player: Player) -> HashMap<String, (Position, Position)> {
    let mut timeline = state.history.clone();
    timeline.push(state.snapshot());
    let mut moves = HashMap::new();
    if timeline.len() < 2 {
        return moves;
    }
    for index in (0..timeline.len() - 1).rev() {
        let before = &timeline[index];
        if before.current_player != player {
            continue;
        }
        if let Some((piece_id, from, to)) = moved_piece(before, &timeline[index + 1]) {
            moves.entry(piece_id).or_insert((from, to));
        }
    }
    moves
}

fn loop_penalty(
    preview: &GameState,
    piece: &Piece,
    destination: Position,
    repetition_counts: &HashMap<String, usize>,
    recent_moves: &HashMap<String, (Position, Position)>,
) -> f64 {
    let repeat_count = repetition_counts
        .get(&state_key(preview))
        .copied()
        .unwrap_or(0) as f64;
    let reverses_last_move = recent_moves.get(&piece.id).is_some_and(|(from, to)| {
        same_position(*to, piece.position) && same_position(*from, destination)
    });
    repeat_count * REPEATED_STATE_PENALTY
        + if reverses_last_move {
            IMMEDIATE_REVERSAL_PENALTY
        } else {
            0.0
        }
}

#[derive(Clone)]
struct Choice {
    tuned: GameState,
    piece_id: String,
    destination: Position,
    preview: GameState,
    score: f64,
}

#[derive(Clone, Copy)]
struct CacheEntry {
    depth: u8,
    value: f64,
}

struct HardSearchContext {
    root_player: Player,
    started_at: SearchStartedAt,
    deadline: Duration,
    repetition_counts: HashMap<String, usize>,
    recent_moves: HashMap<String, (Position, Position)>,
    transpositions: HashMap<String, CacheEntry>,
    nodes: usize,
}

#[derive(Clone)]
struct MoveChoice {
    piece_id: String,
    destination: Position,
    preview: GameState,
    score: f64,
}

fn activation_order_for_profile(components: &PlayerComponents) -> PlayerActivationOrder {
    fn active(values: &[i8]) -> Vec<usize> {
        values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (*value != 0).then_some(index))
            .collect()
    }
    PlayerActivationOrder {
        pawn: active(&components.pawn),
        rook: active(&components.rook),
        spy: active(&components.spy),
        king: active(&components.king),
    }
}

fn sort_choices(choices: &mut [Choice]) {
    choices.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });
}

fn hard_action_choices(
    state: &GameState,
    player: Player,
    profile_limit: usize,
    action_limit: usize,
    trap_analysis_limit: usize,
    repetition_counts: &HashMap<String, usize>,
    recent_moves: &HashMap<String, (Position, Position)>,
) -> Vec<Choice> {
    let mut choices = Vec::new();
    let current_field = evaluate_field(state);
    let current_unstable_count = unstable_pieces(player, state, &current_field).len();

    for profile in tuning_candidates(state, player, profile_limit) {
        let mut tuned_base = state.clone();
        *tuned_base.components.get_mut(player) = profile;
        *tuned_base.activation_orders.get_mut(player) =
            activation_order_for_profile(tuned_base.components.get(player));
        tuned_base.selected_piece_id = None;
        let tuned_base_field = evaluate_field(&tuned_base);
        let tuned = mark_instability(tuned_base, &tuned_base_field);

        for piece in tuned
            .pieces
            .iter()
            .filter(|piece| piece.owner == player)
            .cloned()
            .collect::<Vec<_>>()
        {
            for destination in get_legal_moves(&piece.id, &tuned, &tuned_base_field) {
                let result = apply_move(&piece.id, destination, tuned.clone(), false);
                if !result.ok {
                    continue;
                }
                let field = evaluate_field(&result.state);
                let tactical_bonus = if result.state.status == win_status(player) {
                    500_000.0
                } else {
                    0.0
                };
                let rescue_bonus = if current_unstable_count == 0 {
                    0.0
                } else {
                    let after = unstable_pieces(player, &result.state, &field).len();
                    current_unstable_count.saturating_sub(after) as f64 * 2_000.0
                };
                let score = hard_score_state(&result.state, player) + tactical_bonus + rescue_bonus
                    - loop_penalty(
                        &result.state,
                        &piece,
                        destination,
                        repetition_counts,
                        recent_moves,
                    );
                choices.push(Choice {
                    tuned: tuned.clone(),
                    piece_id: piece.id.clone(),
                    destination,
                    preview: result.state,
                    score,
                });
            }
        }
    }

    sort_choices(&mut choices);
    for choice in choices.iter_mut().take(trap_analysis_limit) {
        let field = evaluate_field(&choice.preview);
        if !is_king_unprotected(player.opponent(), &choice.preview, &field) {
            continue;
        }
        let analyzed = apply_move(
            &choice.piece_id,
            choice.destination,
            choice.tuned.clone(),
            true,
        );
        if !analyzed.ok {
            continue;
        }
        if analyzed.state.status == win_status(player) {
            choice.score += 900_000.0;
            choice.preview = analyzed.state;
        }
    }
    sort_choices(&mut choices);
    choices.truncate(action_limit);
    choices
}

fn hard_position_is_volatile(state: &GameState) -> bool {
    if state.status != GameStatus::Playing {
        return false;
    }
    let field = evaluate_field(state);
    is_king_unprotected(state.current_player, state, &field)
        || is_king_unprotected(state.current_player.opponent(), state, &field)
        || !unstable_pieces(state.current_player, state, &field).is_empty()
        || !unstable_pieces(state.current_player.opponent(), state, &field).is_empty()
}

fn hard_cache_key(state: &GameState, depth: u8, quiescence_depth: u8) -> String {
    format!("{}|{}|{}", depth, quiescence_depth, state_key(state))
}

fn hard_search_score(
    state: &GameState,
    depth: u8,
    quiescence_depth: u8,
    mut alpha: f64,
    mut beta: f64,
    context: &mut HardSearchContext,
) -> f64 {
    context.nodes += 1;
    if context.nodes > 1 && deadline_reached(&context.started_at, context.deadline) {
        return hard_score_state(state, context.root_player);
    }

    if state.status != GameStatus::Playing {
        return hard_score_state(state, context.root_player);
    }

    let extending = depth == 0 && quiescence_depth > 0 && hard_position_is_volatile(state);
    let effective_depth = if extending { 1 } else { depth };
    let next_quiescence_depth = if extending {
        quiescence_depth.saturating_sub(1)
    } else {
        quiescence_depth
    };

    if effective_depth == 0 {
        return hard_score_state(state, context.root_player);
    }

    let cache_key = hard_cache_key(state, effective_depth, next_quiescence_depth);
    if let Some(entry) = context.transpositions.get(&cache_key) {
        if entry.depth >= effective_depth {
            return entry.value;
        }
    }

    let action_limit = if extending {
        HARD_QUIESCENCE_ACTION_LIMIT
    } else {
        HARD_BRANCH_ACTION_LIMIT
    };
    let choices = hard_action_choices(
        state,
        state.current_player,
        HARD_PROFILE_LIMIT,
        action_limit,
        0,
        &context.repetition_counts,
        &context.recent_moves,
    );
    if choices.is_empty() {
        return if state.current_player == context.root_player {
            -1_000_000.0
        } else {
            1_000_000.0
        };
    }

    let value = if state.current_player == context.root_player {
        let mut value = f64::NEG_INFINITY;
        for choice in choices {
            value = value.max(hard_search_score(
                &choice.preview,
                effective_depth - 1,
                next_quiescence_depth,
                alpha,
                beta,
                context,
            ));
            alpha = alpha.max(value);
            if alpha >= beta {
                break;
            }
        }
        value
    } else {
        let mut value = f64::INFINITY;
        for choice in choices {
            value = value.min(hard_search_score(
                &choice.preview,
                effective_depth - 1,
                next_quiescence_depth,
                alpha,
                beta,
                context,
            ));
            beta = beta.min(value);
            if alpha >= beta {
                break;
            }
        }
        value
    };

    context.transpositions.insert(
        cache_key,
        CacheEntry {
            depth: effective_depth,
            value,
        },
    );
    value
}

fn legal_move_choices(state: &GameState) -> Vec<MoveChoice> {
    let field = evaluate_field(state);
    state
        .pieces
        .iter()
        .filter(|piece| piece.owner == state.current_player)
        .cloned()
        .flat_map(|piece| {
            get_legal_moves(&piece.id, state, &field)
                .into_iter()
                .filter_map(move |destination| {
                    let result = apply_move(&piece.id, destination, state.clone(), false);
                    result.ok.then_some(MoveChoice {
                        piece_id: piece.id.clone(),
                        destination,
                        preview: result.state,
                        score: 0.0,
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn no_legal_move_state(mut state: GameState, player: Player) -> GameState {
    let field = evaluate_field(&state);
    let winner = player.opponent();
    let message = if is_king_unprotected(player, &state, &field) {
        format!("{} has no legal rescue", player_name(player))
    } else {
        format!("{} has no legal move", player_name(player))
    };
    let previous = state.snapshot();
    state.status = win_status(winner);
    state.selected_piece_id = None;
    state.history.push(previous);
    state.message = message;
    state
}

fn minimax_score(
    state: &GameState,
    root_player: Player,
    depth: u8,
    mut alpha: f64,
    mut beta: f64,
) -> f64 {
    if depth == 0 || state.status != GameStatus::Playing {
        let field = evaluate_field(state);
        return score_state(state, root_player, &field);
    }

    let choices = legal_move_choices(state);
    if choices.is_empty() {
        return if state.current_player == root_player {
            -1_000_000.0
        } else {
            1_000_000.0
        };
    }

    if state.current_player == root_player {
        let mut value = f64::NEG_INFINITY;
        for choice in choices {
            value = value.max(minimax_score(
                &choice.preview,
                root_player,
                depth - 1,
                alpha,
                beta,
            ));
            alpha = alpha.max(value);
            if alpha >= beta {
                break;
            }
        }
        value
    } else {
        let mut value = f64::INFINITY;
        for choice in choices {
            value = value.min(minimax_score(
                &choice.preview,
                root_player,
                depth - 1,
                alpha,
                beta,
            ));
            beta = beta.min(value);
            if alpha >= beta {
                break;
            }
        }
        value
    }
}

fn lost_material(before: &GameState, after: &GameState, owner: Player) -> f64 {
    before
        .pieces
        .iter()
        .filter(|piece| {
            piece.owner == owner
                && !after
                    .pieces
                    .iter()
                    .any(|remaining| remaining.id == piece.id)
        })
        .map(|piece| material_value(piece.piece_type))
        .sum()
}

fn easy_generosity_score(choice: &MoveChoice, state: &GameState, player: Player) -> f64 {
    let enemy = player.opponent();
    let self_score = minimax_score(
        &choice.preview,
        player,
        EASY_SEARCH_DEPTH.saturating_sub(1),
        f64::NEG_INFINITY,
        f64::INFINITY,
    );
    let preview_field = evaluate_field(&choice.preview);

    -self_score
        - if choice.preview.status == win_status(player) {
            EASY_WIN_PENALTY
        } else {
            0.0
        }
        - if is_king_unprotected(enemy, &choice.preview, &preview_field) {
            EASY_CHECK_PENALTY
        } else {
            0.0
        }
        - lost_material(state, &choice.preview, enemy) * EASY_ENEMY_CAPTURE_PENALTY
        + lost_material(state, &choice.preview, player) * EASY_OWN_LOSS_BONUS
}

fn sort_move_choices(choices: &mut [MoveChoice]) {
    choices.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });
}

pub fn play_easy_turn(state: GameState, player: Player, options: AiTurnOptions) -> GameState {
    if state.status != GameStatus::Playing || state.current_player != player {
        return state;
    }

    let repetition_counts = recent_state_counts(&state);
    let recent_moves = last_move_by_piece(&state, player);
    let mut choices = legal_move_choices(&state)
        .into_iter()
        .filter_map(|mut choice| {
            let piece = state
                .pieces
                .iter()
                .find(|piece| piece.id == choice.piece_id)?
                .clone();
            choice.score = easy_generosity_score(&choice, &state, player)
                - loop_penalty(
                    &choice.preview,
                    &piece,
                    choice.destination,
                    &repetition_counts,
                    &recent_moves,
                );
            Some(choice)
        })
        .collect::<Vec<_>>();

    if choices.is_empty() {
        return no_legal_move_state(state, player);
    }

    sort_move_choices(&mut choices);
    let seed = options.seed.unwrap_or(0);
    let variety = options.variety.unwrap_or(0.0).clamp(0.0, 1.0);
    if variety > 0.0 {
        let leader = choices[0].score;
        let mut candidate_window = choices
            .iter()
            .filter(|choice| leader - choice.score <= 80.0 + variety * 180.0)
            .cloned()
            .collect::<Vec<_>>();
        candidate_window.sort_by(|left, right| {
            let right_value = right.score
                + hash_unit(&format!(
                    "{}:{}:{}:{}:{}:{}:{:.3}",
                    seed,
                    state.turn_number,
                    player_key(player),
                    right.piece_id,
                    right.destination.x,
                    right.destination.y,
                    right.score
                )) * variety
                    * 120.0;
            let left_value = left.score
                + hash_unit(&format!(
                    "{}:{}:{}:{}:{}:{}:{:.3}",
                    seed,
                    state.turn_number,
                    player_key(player),
                    left.piece_id,
                    left.destination.x,
                    left.destination.y,
                    left.score
                )) * variety
                    * 120.0;
            right_value
                .partial_cmp(&left_value)
                .unwrap_or(Ordering::Equal)
        });
        choices.splice(0..candidate_window.len(), candidate_window);
    }

    choices
        .into_iter()
        .next()
        .map(|choice| choice.preview)
        .unwrap_or_else(|| no_legal_move_state(state, player))
}

pub fn play_heuristic_turn(state: GameState, player: Player, options: AiTurnOptions) -> GameState {
    if state.status != GameStatus::Playing || state.current_player != player {
        return state;
    }

    let mut choices: Vec<Choice> = Vec::new();
    let repetition_counts = recent_state_counts(&state);
    let recent_moves = last_move_by_piece(&state, player);
    let started_at = search_started_at();
    let deadline = Duration::from_millis(options.time_budget_ms.unwrap_or(DEFAULT_TIME_BUDGET_MS));
    let mut best_score = f64::NEG_INFINITY;

    fn remember_choice(
        choices: &mut Vec<Choice>,
        best_score: &mut f64,
        tuned: GameState,
        piece_id: String,
        destination: Position,
        preview: GameState,
        score: f64,
    ) {
        if choices.len() < EXACT_CANDIDATE_LIMIT || score > *best_score {
            choices.push(Choice {
                tuned,
                piece_id,
                destination,
                preview,
                score,
            });
            sort_choices(choices);
            choices.truncate(EXACT_CANDIDATE_LIMIT);
            *best_score = choices
                .last()
                .map_or(f64::NEG_INFINITY, |choice| choice.score);
        }
    }

    'search: for profile in tuning_candidates(&state, player, 28) {
        let mut tuned_base = state.clone();
        *tuned_base.components.get_mut(player) = profile;
        *tuned_base.activation_orders.get_mut(player) =
            activation_order_for_profile(tuned_base.components.get(player));
        tuned_base.selected_piece_id = None;
        let tuned_base_field = evaluate_field(&tuned_base);
        let tuned = mark_instability(tuned_base, &tuned_base_field);

        let pieces = tuned
            .pieces
            .iter()
            .filter(|piece| piece.owner == player)
            .cloned()
            .collect::<Vec<_>>();
        for piece in pieces {
            for destination in get_legal_moves(&piece.id, &tuned, &tuned_base_field) {
                let result = apply_move(&piece.id, destination, tuned.clone(), false);
                if !result.ok {
                    continue;
                }
                let result_field = evaluate_field(&result.state);
                let score = score_state(&result.state, player, &result_field)
                    - loop_penalty(
                        &result.state,
                        &piece,
                        destination,
                        &repetition_counts,
                        &recent_moves,
                    );
                remember_choice(
                    &mut choices,
                    &mut best_score,
                    tuned.clone(),
                    piece.id.clone(),
                    destination,
                    result.state,
                    score,
                );
                if deadline_reached(&started_at, deadline) && !choices.is_empty() {
                    break 'search;
                }
            }
        }
    }

    if choices.is_empty() {
        let field = evaluate_field(&state);
        let winner = player.opponent();
        let message = if is_king_unprotected(player, &state, &field) {
            format!("{} has no legal rescue", player_name(player))
        } else {
            format!("{} has no legal move", player_name(player))
        };
        let mut next = state.clone();
        next.status = win_status(winner);
        next.selected_piece_id = None;
        next.history.push(state.snapshot());
        next.message = message;
        return next;
    }

    let seed = options.seed.unwrap_or(0);
    let variety = options.variety.unwrap_or(0.0).clamp(0.0, 1.0);
    if variety > 0.0 {
        let leader = choices[0].score;
        let mut candidate_window = choices
            .iter()
            .filter(|choice| leader - choice.score <= 120.0 + variety * 280.0)
            .cloned()
            .collect::<Vec<_>>();
        candidate_window.sort_by(|left, right| {
            let right_value =
                right.score + choice_noise(right, &state, player, seed) * variety * 180.0;
            let left_value =
                left.score + choice_noise(left, &state, player, seed) * variety * 180.0;
            right_value
                .partial_cmp(&left_value)
                .unwrap_or(Ordering::Equal)
        });
        for (index, choice) in candidate_window.into_iter().enumerate() {
            choices[index] = choice;
        }
    }

    let analysis_limit = if deadline_reached(&started_at, deadline) {
        1
    } else {
        FULL_ANALYSIS_LIMIT
    };
    let mut fallback = None;
    for choice in choices.iter().take(analysis_limit) {
        let result = apply_move(
            &choice.piece_id,
            choice.destination,
            choice.tuned.clone(),
            true,
        );
        if !result.ok {
            continue;
        }
        if result.state.status == win_status(player) {
            let mut winning_state = result.state;
            winning_state.history = {
                let mut history = state.history.clone();
                history.push(state.snapshot());
                history
            };
            return winning_state;
        }
        if fallback.is_none() {
            fallback = Some(result.state);
        }
    }
    if let Some(mut fallback) =
        fallback.or_else(|| choices.first().map(|choice| choice.preview.clone()))
    {
        fallback.history = {
            let mut history = state.history.clone();
            history.push(state.snapshot());
            history
        };
        return fallback;
    }
    let mut next = state;
    next.message = format!("{} has no legal move", player_name(player));
    next
}

pub fn play_hard_turn(state: GameState, player: Player, options: AiTurnOptions) -> GameState {
    if state.status != GameStatus::Playing || state.current_player != player {
        return state;
    }

    let deadline = Duration::from_millis(
        options
            .time_budget_ms
            .unwrap_or(HARD_DEFAULT_TIME_BUDGET_MS)
            .max(50),
    );
    let mut context = HardSearchContext {
        root_player: player,
        started_at: search_started_at(),
        deadline,
        repetition_counts: recent_state_counts(&state),
        recent_moves: last_move_by_piece(&state, player),
        transpositions: HashMap::new(),
        nodes: 0,
    };

    let root_trap_analysis_limit = if deadline >= Duration::from_millis(500) {
        HARD_ROOT_TRAP_ANALYSIS_LIMIT
    } else {
        0
    };
    let mut root_choices = hard_action_choices(
        &state,
        player,
        HARD_PROFILE_LIMIT,
        HARD_ROOT_ACTION_LIMIT,
        root_trap_analysis_limit,
        &context.repetition_counts,
        &context.recent_moves,
    );
    if root_choices.is_empty() {
        return no_legal_move_state(state, player);
    }

    let seed = options.seed.unwrap_or(0);
    let variety = options.variety.unwrap_or(0.0).clamp(0.0, 1.0);
    if variety > 0.0 {
        let leader = root_choices[0].score;
        let mut candidate_window = root_choices
            .iter()
            .filter(|choice| leader - choice.score <= 80.0 + variety * 180.0)
            .cloned()
            .collect::<Vec<_>>();
        candidate_window.sort_by(|left, right| {
            let right_value =
                right.score + choice_noise(right, &state, player, seed) * variety * 80.0;
            let left_value = left.score + choice_noise(left, &state, player, seed) * variety * 80.0;
            right_value
                .partial_cmp(&left_value)
                .unwrap_or(Ordering::Equal)
        });
        for (index, choice) in candidate_window.into_iter().enumerate() {
            root_choices[index] = choice;
        }
    }

    let mut best_choice = root_choices[0].clone();
    let mut best_score = f64::NEG_INFINITY;

    for depth in 1..=HARD_MAX_DEPTH {
        if deadline_reached(&context.started_at, context.deadline) {
            break;
        }

        let mut depth_best_choice = None;
        let mut depth_best_score = f64::NEG_INFINITY;
        let mut completed_depth = true;

        for choice in &root_choices {
            let score = hard_search_score(
                &choice.preview,
                depth.saturating_sub(1),
                HARD_QUIESCENCE_DEPTH,
                f64::NEG_INFINITY,
                f64::INFINITY,
                &mut context,
            );
            if score > depth_best_score {
                depth_best_score = score;
                depth_best_choice = Some(choice.clone());
            }
            if deadline_reached(&context.started_at, context.deadline) {
                completed_depth = false;
                break;
            }
        }

        if completed_depth || depth_best_choice.is_some() && best_score == f64::NEG_INFINITY {
            if let Some(choice) = depth_best_choice {
                best_choice = choice;
                best_score = depth_best_score;
            }
        }

        if completed_depth {
            for choice in &mut root_choices {
                choice.score = hard_search_score(
                    &choice.preview,
                    depth.saturating_sub(1),
                    HARD_QUIESCENCE_DEPTH,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    &mut context,
                );
            }
            sort_choices(&mut root_choices);
        }
    }

    let result = apply_move(
        &best_choice.piece_id,
        best_choice.destination,
        best_choice.tuned,
        true,
    );
    if result.ok {
        let mut next = result.state;
        next.history = {
            let mut history = state.history.clone();
            history.push(state.snapshot());
            history
        };
        return next;
    }

    let mut fallback = best_choice.preview;
    fallback.history = {
        let mut history = state.history.clone();
        history.push(state.snapshot());
        history
    };
    fallback
}
