use crate::board::*;
use crate::field::evaluate_field;
use crate::model::*;
use crate::rules::{apply_known_legal_move, apply_move, apply_search_move};
use serde::Serialize;
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
const EASY_OWN_LOSS_PENALTY: f64 = 5_000.0;
const EASY_OWN_UNSTABLE_PENALTY: f64 = 500.0;
const EASY_OWN_KING_DANGER_PENALTY: f64 = 600_000.0;
const EASY_OWN_KING_MARGIN_BONUS: f64 = 35.0;
const HARD_DEFAULT_TIME_BUDGET_MS: u64 = 1_500;
const HARD_MAX_DEPTH: u8 = 4;
const HARD_ROOT_ACTION_LIMIT: usize = 18;
const HARD_BRANCH_ACTION_LIMIT: usize = 10;
const HARD_QUIESCENCE_ACTION_LIMIT: usize = 6;
const HARD_PROFILE_LIMIT: usize = 14;
const HARD_QUIESCENCE_DEPTH: u8 = 1;
const HARD_ROOT_TRAP_ANALYSIS_LIMIT: usize = 8;
const HARD_REPLY_SAFETY_PROFILE_LIMIT: usize = 1;
const HARD_REPLY_SAFETY_MOVE_LIMIT: usize = 12;
const HARD_CONVERSION_TIEBREAKER_SCALE: f64 = 0.2;

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

#[derive(Clone, Copy, Debug)]
pub struct HardBotTuning {
    pub conversion_weight: f64,
    pub trap_focus: f64,
    pub cycle_weight: f64,
}

impl Default for HardBotTuning {
    fn default() -> Self {
        Self {
            conversion_weight: 1.0,
            trap_focus: 1.0,
            cycle_weight: 1.0,
        }
    }
}

impl HardBotTuning {
    pub fn clamped(self) -> Self {
        Self {
            conversion_weight: self.conversion_weight.clamp(0.0, 3.0),
            trap_focus: self.trap_focus.clamp(0.0, 3.0),
            cycle_weight: self.cycle_weight.clamp(0.0, 4.0),
        }
    }
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

fn hard_score_state_with_field(state: &GameState, player: Player, field: &Field) -> f64 {
    if state.status != GameStatus::Playing {
        return score_state(state, player, field);
    }

    let enemy = player.opponent();
    let material_balance = material_points(state, player) - material_points(state, enemy);
    let own_unstable = unstable_pieces(player, state, field)
        .into_iter()
        .filter(|piece| piece.piece_type != PieceType::King)
        .count() as f64;
    let enemy_unstable = unstable_pieces(enemy, state, field)
        .into_iter()
        .filter(|piece| piece.piece_type != PieceType::King)
        .count() as f64;

    score_state(state, player, field) + material_balance * 85.0 - own_unstable * 260.0
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
    let mut moves = HashMap::new();
    let timeline_len = state.history.len() + 1;
    if timeline_len < 2 {
        return moves;
    }
    let current = state.snapshot();
    for index in (0..timeline_len - 1).rev() {
        let before = &state.history[index];
        if before.current_player != player {
            continue;
        }
        let after = if index + 1 == state.history.len() {
            &current
        } else {
            &state.history[index + 1]
        };
        if let Some((piece_id, from, to)) = moved_piece(before, after) {
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

fn material_delta(before: &GameState, after: &GameState, player: Player) -> f64 {
    let enemy = player.opponent();
    lost_material(before, after, enemy) - lost_material(before, after, player)
}

fn raw_king_moves(state: &GameState, player: Player, context: &mut HardSearchContext) -> usize {
    let field = cached_field(state, context);
    state
        .pieces
        .iter()
        .find(|piece| piece.owner == player && piece.piece_type == PieceType::King)
        .map(|king| get_legal_moves(&king.id, state, &field).len())
        .unwrap_or(0)
}

fn legal_action_count(state: &GameState, player: Player, context: &mut HardSearchContext) -> usize {
    let mut probe = state.clone();
    probe.current_player = player;
    let field = cached_field(&probe, context);
    probe
        .pieces
        .iter()
        .filter(|piece| piece.owner == player)
        .map(|piece| {
            get_legal_moves(&piece.id, &probe, &field)
                .into_iter()
                .filter(|destination| {
                    apply_search_move(&piece.id, *destination, probe.clone(), &field).ok
                })
                .count()
        })
        .sum()
}

fn player_has_forcing_reply(
    state: &GameState,
    player: Player,
    profile_limit: usize,
    context: &mut HardSearchContext,
) -> bool {
    if state.status != GameStatus::Playing {
        return state.status == win_status(player);
    }

    let mut probe = state.clone();
    probe.current_player = player;
    for profile in tuning_candidates(&probe, player, profile_limit) {
        if deadline_reached(&context.started_at, context.deadline) {
            return false;
        }
        let mut tuned_base = probe.clone();
        tuned_base.history.clear();
        *tuned_base.components.get_mut(player) = profile;
        *tuned_base.activation_orders.get_mut(player) =
            activation_order_for_profile(tuned_base.components.get(player));
        tuned_base.selected_piece_id = None;
        let tuned_base_field = cached_field(&tuned_base, context);
        let tuned = mark_instability(tuned_base, &tuned_base_field);

        let mut checked_moves = 0;
        for piece in tuned
            .pieces
            .iter()
            .filter(|piece| piece.owner == player)
            .cloned()
            .collect::<Vec<_>>()
        {
            for destination in get_legal_moves(&piece.id, &tuned, &tuned_base_field) {
                if checked_moves >= HARD_REPLY_SAFETY_MOVE_LIMIT
                    || deadline_reached(&context.started_at, context.deadline)
                {
                    return false;
                }
                checked_moves += 1;
                let result =
                    apply_search_move(&piece.id, destination, tuned.clone(), &tuned_base_field);
                if !result.ok {
                    continue;
                }
                if result.state.status == win_status(player) {
                    return true;
                }
                let reply_field = cached_field(&result.state, context);
                if is_king_unprotected(player.opponent(), &result.state, &reply_field) {
                    return true;
                }
            }
        }
    }

    false
}

fn support_score(state: &GameState, player: Player) -> i32 {
    let pieces = state
        .pieces
        .iter()
        .filter(|piece| piece.owner == player)
        .collect::<Vec<_>>();
    let mut support = 0;
    for left in 0..pieces.len() {
        for right in left + 1..pieces.len() {
            let distance = (pieces[left].position.x - pieces[right].position.x).abs()
                + (pieces[left].position.y - pieces[right].position.y).abs();
            if distance <= 2 {
                support += 1;
            }
        }
    }
    support
}

fn field_control_score(state: &GameState, player: Player, context: &mut HardSearchContext) -> i32 {
    let field = cached_field(state, context);
    let mut score = 0;
    for row in field {
        for value in row {
            if value > 0.0 {
                score += if player == Player::Red { 1 } else { -1 };
            } else if value < 0.0 {
                score += if player == Player::Blue { 1 } else { -1 };
            }
        }
    }
    score
}

fn abstract_state_key(state: &GameState, context: &mut HardSearchContext) -> String {
    let blue_king = state
        .pieces
        .iter()
        .find(|piece| piece.owner == Player::Blue && piece.piece_type == PieceType::King)
        .map(|piece| format!("{},{}", piece.position.x, piece.position.y))
        .unwrap_or_else(|| "x".to_owned());
    let red_king = state
        .pieces
        .iter()
        .find(|piece| piece.owner == Player::Red && piece.piece_type == PieceType::King)
        .map(|piece| format!("{},{}", piece.position.x, piece.position.y))
        .unwrap_or_else(|| "x".to_owned());
    format!(
        "{}|{}|{}|{:.0}|{:.0}|{}|{}|{}|{}|{}",
        player_key(state.current_player),
        blue_king,
        red_king,
        material_points(state, Player::Blue),
        material_points(state, Player::Red),
        raw_king_moves(state, Player::Blue, context),
        raw_king_moves(state, Player::Red, context),
        legal_action_count(state, Player::Blue, context) / 4,
        legal_action_count(state, Player::Red, context) / 4,
        field_control_score(state, Player::Red, context) / 6,
    )
}

fn snapshot_as_state(base: &GameState, snapshot: &GameSnapshot) -> GameState {
    GameState {
        pieces: snapshot.pieces.clone(),
        current_player: snapshot.current_player,
        components: snapshot.components.clone(),
        activation_orders: snapshot.activation_orders.clone(),
        default_components: base.default_components.clone(),
        status: snapshot.status,
        selected_piece_id: snapshot.selected_piece_id.clone(),
        turn_number: snapshot.turn_number,
        definitions: snapshot.definitions.clone(),
        wave_scales: snapshot.wave_scales.clone(),
        home_energy: snapshot.home_energy.clone(),
        history: Vec::new(),
        message: String::new(),
    }
}

fn recent_abstract_counts(
    state: &GameState,
    context: &mut HardSearchContext,
) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    let start = state.history.len().saturating_sub(24);
    for snapshot in &state.history[start..] {
        let probe = snapshot_as_state(state, snapshot);
        let key = abstract_state_key(&probe, context);
        *counts.entry(key).or_insert(0) += 1;
    }
    let current = abstract_state_key(state, context);
    *counts.entry(current).or_insert(0) += 1;
    counts
}

fn hard_conversion_score(
    choice: &Choice,
    state: &GameState,
    player: Player,
    tuning: HardBotTuning,
    context: &mut HardSearchContext,
) -> f64 {
    let enemy = player.opponent();
    let before_enemy_escapes = raw_king_moves(state, enemy, context) as f64;
    let after_enemy_escapes = raw_king_moves(&choice.preview, enemy, context) as f64;
    let before_own_escapes = raw_king_moves(state, player, context) as f64;
    let after_own_escapes = raw_king_moves(&choice.preview, player, context) as f64;
    let enemy_safe = legal_action_count(&choice.preview, enemy, context) as f64;
    let own_safe = legal_action_count(&choice.preview, player, context) as f64;
    let enemy_field = cached_field(&choice.preview, context);
    let enemy_forced = is_king_unprotected(enemy, &choice.preview, &enemy_field)
        || unstable_pieces(enemy, &choice.preview, &enemy_field)
            .into_iter()
            .any(|piece| piece.piece_type != PieceType::King);
    let recent_counts = recent_abstract_counts(state, context);
    let repeat_penalty = recent_counts
        .get(&abstract_state_key(&choice.preview, context))
        .copied()
        .unwrap_or(0) as f64
        * 900.0
        * tuning.cycle_weight;
    let reopens_trap_penalty =
        if before_enemy_escapes <= 2.0 && after_enemy_escapes > before_enemy_escapes {
            (after_enemy_escapes - before_enemy_escapes) * 220.0 * tuning.trap_focus
        } else {
            0.0
        };
    let trap_phase = before_enemy_escapes <= 3.0 || enemy_safe <= 14.0;
    let close_exit_bonus = (before_enemy_escapes - after_enemy_escapes).max(0.0)
        * if trap_phase { 180.0 } else { 65.0 }
        * tuning.trap_focus;
    let support_delta =
        f64::from(support_score(&choice.preview, player) - support_score(state, player));
    let field_delta = f64::from(
        field_control_score(&choice.preview, player, context)
            - field_control_score(state, player, context),
    );
    let material = material_delta(state, &choice.preview, player);
    let own_big_hat_is_tactically_thin = after_own_escapes <= 2.0 || before_own_escapes <= 2.0;
    let allows_forcing_reply = own_big_hat_is_tactically_thin
        && player_has_forcing_reply(
            &choice.preview,
            enemy,
            HARD_REPLY_SAFETY_PROFILE_LIMIT,
            context,
        );
    let score = if trap_phase {
        (if enemy_forced {
            450.0 * tuning.trap_focus
        } else {
            0.0
        }) + close_exit_bonus
            - after_enemy_escapes * 130.0 * tuning.trap_focus
            - enemy_safe * 6.0 * tuning.trap_focus
            + after_own_escapes.max(0.0) * 18.0
            + (own_safe - enemy_safe).max(0.0) * 3.0
            + material * 18.0
            - reopens_trap_penalty
            - repeat_penalty
    } else {
        close_exit_bonus + (after_own_escapes - before_own_escapes) * 28.0
            - (after_enemy_escapes - before_enemy_escapes) * 42.0
            + field_delta * 7.0
            + support_delta * 8.0
            + material * 14.0
            - repeat_penalty * 0.35
            - reopens_trap_penalty
    };
    let forcing_reply_penalty = if allows_forcing_reply {
        450_000.0 * tuning.trap_focus.max(0.5)
    } else {
        0.0
    };
    (score * tuning.conversion_weight - forcing_reply_penalty) * HARD_CONVERSION_TIEBREAKER_SCALE
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
    field_cache: HashMap<String, Field>,
    profile: HardSearchProfile,
    nodes: usize,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardSearchProfile {
    pub nodes: u64,
    pub transposition_hits: u64,
    pub transposition_stores: u64,
    pub field_cache_hits: u64,
    pub field_cache_misses: u64,
    pub generated_candidates: u64,
    pub applied_candidates: u64,
    pub rejected_candidates: u64,
    pub tuning_profiles: u64,
    pub alpha_beta_cutoffs: u64,
    pub deadline_cutoffs: u64,
    pub completed_depth: u8,
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

fn cached_field(state: &GameState, context: &mut HardSearchContext) -> Field {
    let key = state_key(state);
    if let Some(field) = context.field_cache.get(&key) {
        context.profile.field_cache_hits += 1;
        return field.clone();
    }
    context.profile.field_cache_misses += 1;
    let field = evaluate_field(state);
    context.field_cache.insert(key, field.clone());
    field
}

fn hard_action_choices(
    state: &GameState,
    player: Player,
    profile_limit: usize,
    action_limit: usize,
    trap_analysis_limit: usize,
    context: &mut HardSearchContext,
) -> Vec<Choice> {
    let mut choices = Vec::new();
    let current_field = cached_field(state, context);
    let current_unstable_count = unstable_pieces(player, state, &current_field).len();

    for profile in tuning_candidates(state, player, profile_limit) {
        context.profile.tuning_profiles += 1;
        let mut tuned_base = state.clone();
        tuned_base.history.clear();
        *tuned_base.components.get_mut(player) = profile;
        *tuned_base.activation_orders.get_mut(player) =
            activation_order_for_profile(tuned_base.components.get(player));
        tuned_base.selected_piece_id = None;
        let tuned_base_field = cached_field(&tuned_base, context);
        let tuned = mark_instability(tuned_base, &tuned_base_field);

        for piece in tuned
            .pieces
            .iter()
            .filter(|piece| piece.owner == player)
            .cloned()
            .collect::<Vec<_>>()
        {
            for destination in get_legal_moves(&piece.id, &tuned, &tuned_base_field) {
                context.profile.generated_candidates += 1;
                let result =
                    apply_search_move(&piece.id, destination, tuned.clone(), &tuned_base_field);
                if !result.ok {
                    context.profile.rejected_candidates += 1;
                    continue;
                }
                context.profile.applied_candidates += 1;
                let mut preview = result.state;
                preview.history.clear();
                let field = cached_field(&preview, context);
                let tactical_bonus = if preview.status == win_status(player) {
                    500_000.0
                } else {
                    0.0
                };
                let rescue_bonus = if current_unstable_count == 0 {
                    0.0
                } else {
                    let after = unstable_pieces(player, &preview, &field).len();
                    current_unstable_count.saturating_sub(after) as f64 * 2_000.0
                };
                let score = hard_score_state_with_field(&preview, player, &field)
                    + tactical_bonus
                    + rescue_bonus
                    - loop_penalty(
                        &preview,
                        &piece,
                        destination,
                        &context.repetition_counts,
                        &context.recent_moves,
                    );
                choices.push(Choice {
                    tuned: tuned.clone(),
                    piece_id: piece.id.clone(),
                    destination,
                    preview,
                    score,
                });
            }
        }
    }

    sort_choices(&mut choices);
    for choice in choices.iter_mut().take(trap_analysis_limit) {
        let field = cached_field(&choice.preview, context);
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

fn hard_position_is_volatile(state: &GameState, context: &mut HardSearchContext) -> bool {
    if state.status != GameStatus::Playing {
        return false;
    }
    let field = cached_field(state, context);
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
    context.profile.nodes += 1;
    if context.nodes > 1 && deadline_reached(&context.started_at, context.deadline) {
        context.profile.deadline_cutoffs += 1;
        let field = cached_field(state, context);
        return hard_score_state_with_field(state, context.root_player, &field);
    }

    if state.status != GameStatus::Playing {
        let field = cached_field(state, context);
        return hard_score_state_with_field(state, context.root_player, &field);
    }

    let extending = depth == 0 && quiescence_depth > 0 && hard_position_is_volatile(state, context);
    let effective_depth = if extending { 1 } else { depth };
    let next_quiescence_depth = if extending {
        quiescence_depth.saturating_sub(1)
    } else {
        quiescence_depth
    };

    if effective_depth == 0 {
        let field = cached_field(state, context);
        return hard_score_state_with_field(state, context.root_player, &field);
    }

    let cache_key = hard_cache_key(state, effective_depth, next_quiescence_depth);
    if let Some(entry) = context.transpositions.get(&cache_key) {
        if entry.depth >= effective_depth {
            context.profile.transposition_hits += 1;
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
        context,
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
                context.profile.alpha_beta_cutoffs += 1;
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
                context.profile.alpha_beta_cutoffs += 1;
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
    context.profile.transposition_stores += 1;
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

fn own_safety_score(state: &GameState, player: Player, field: &Field) -> f64 {
    let own_king_value = state
        .pieces
        .iter()
        .find(|piece| piece.owner == player && piece.piece_type == PieceType::King)
        .map(|piece| field[piece.position.y as usize][piece.position.x as usize])
        .unwrap_or(0.0);
    let own_unstable = unstable_pieces(player, state, field)
        .into_iter()
        .filter(|piece| piece.piece_type != PieceType::King)
        .count() as f64;

    -if is_king_unprotected(player, state, field) {
        EASY_OWN_KING_DANGER_PENALTY
    } else {
        0.0
    } - own_unstable * EASY_OWN_UNSTABLE_PENALTY
        + if compatible(player, own_king_value) {
            own_king_value.abs().min(4.0) * EASY_OWN_KING_MARGIN_BONUS
        } else {
            0.0
        }
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
        - lost_material(state, &choice.preview, player) * EASY_OWN_LOSS_PENALTY
        + own_safety_score(&choice.preview, player, &preview_field)
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
    play_hard_turn_profiled(state, player, options).state
}

pub fn play_hard_turn_tuned(
    state: GameState,
    player: Player,
    options: AiTurnOptions,
    tuning: HardBotTuning,
) -> GameState {
    play_hard_turn_profiled_tuned(state, player, options, tuning).state
}

pub struct HardTurnResult {
    pub state: GameState,
    pub profile: HardSearchProfile,
}

pub fn play_hard_turn_profiled(
    state: GameState,
    player: Player,
    options: AiTurnOptions,
) -> HardTurnResult {
    play_hard_turn_profiled_tuned(state, player, options, HardBotTuning::default())
}

pub fn play_hard_turn_profiled_tuned(
    state: GameState,
    player: Player,
    options: AiTurnOptions,
    tuning: HardBotTuning,
) -> HardTurnResult {
    if state.status != GameStatus::Playing || state.current_player != player {
        return HardTurnResult {
            state,
            profile: HardSearchProfile::default(),
        };
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
        field_cache: HashMap::new(),
        profile: HardSearchProfile::default(),
        nodes: 0,
    };
    let hard_tuning = tuning.clamped();

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
        &mut context,
    );
    if root_choices.is_empty() {
        return HardTurnResult {
            state: no_legal_move_state(state, player),
            profile: context.profile,
        };
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
            let score =
                hard_search_score(
                    &choice.preview,
                    depth.saturating_sub(1),
                    HARD_QUIESCENCE_DEPTH,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    &mut context,
                ) + hard_conversion_score(choice, &state, player, hard_tuning, &mut context);
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
                context.profile.completed_depth = depth;
            }
        }

        if completed_depth {
            for index in 0..root_choices.len() {
                let choice = root_choices[index].clone();
                root_choices[index].score =
                    hard_search_score(
                        &choice.preview,
                        depth.saturating_sub(1),
                        HARD_QUIESCENCE_DEPTH,
                        f64::NEG_INFINITY,
                        f64::INFINITY,
                        &mut context,
                    ) + hard_conversion_score(&choice, &state, player, hard_tuning, &mut context);
            }
            sort_choices(&mut root_choices);
        }
    }

    let tuned_field = cached_field(&best_choice.tuned, &mut context);
    let result = apply_known_legal_move(
        &best_choice.piece_id,
        best_choice.destination,
        best_choice.tuned,
        &tuned_field,
        true,
    );
    if result.ok {
        let mut next = result.state;
        next.history = {
            let mut history = state.history.clone();
            history.push(state.snapshot());
            history
        };
        return HardTurnResult {
            state: next,
            profile: context.profile,
        };
    }

    let mut fallback = best_choice.preview;
    fallback.history = {
        let mut history = state.history.clone();
        history.push(state.snapshot());
        history
    };
    HardTurnResult {
        state: fallback,
        profile: context.profile,
    }
}
