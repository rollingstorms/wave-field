use crate::board::*;
use crate::field::evaluate_field;
use crate::model::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

fn base_tuning_strength(piece_type: PieceType) -> usize {
    match piece_type {
        PieceType::Pawn | PieceType::Spy => 1,
        PieceType::Rook | PieceType::King => 2,
    }
}

fn tuning_strength(piece_type: PieceType, count: usize) -> usize {
    base_tuning_strength(piece_type).min(count)
}

fn remove_unrescued_pieces(
    player: Player,
    mut state: GameState,
    rescue_deadline_ids: &HashSet<String>,
) -> GameState {
    loop {
        let field = evaluate_field(&state);
        let lost_ids = unstable_pieces(player, &state, &field)
            .into_iter()
            .filter(|piece| {
                piece.piece_type != PieceType::King && rescue_deadline_ids.contains(&piece.id)
            })
            .map(|piece| piece.id)
            .collect::<HashSet<_>>();
        if lost_ids.is_empty() {
            return state;
        }
        state.pieces.retain(|piece| !lost_ids.contains(&piece.id));
    }
}

fn move_piece(mut state: GameState, piece_id: &str, destination: Position) -> GameState {
    if let Some(piece) = state.pieces.iter_mut().find(|piece| piece.id == piece_id) {
        piece.position = destination;
    }
    state
}

fn training_candidate_state(state: &GameState, piece_id: &str, destination: Position) -> GameState {
    let mut pieces = state.pieces.clone();
    if let Some(piece) = pieces.iter_mut().find(|piece| piece.id == piece_id) {
        piece.position = destination;
    }
    GameState {
        pieces,
        current_player: state.current_player,
        components: state.components.clone(),
        activation_orders: state.activation_orders.clone(),
        default_components: state.default_components.clone(),
        status: state.status,
        selected_piece_id: None,
        turn_number: state.turn_number,
        definitions: state.definitions.clone(),
        wave_scales: state.wave_scales.clone(),
        home_energy: state.home_energy.clone(),
        amp_squares: state.amp_squares.clone(),
        history: Vec::new(),
        message: String::new(),
    }
}

#[derive(Clone, Debug)]
pub struct TrainingSafetyContext {
    pub player: Player,
    deadline_ids: HashSet<String>,
}

impl TrainingSafetyContext {
    pub fn new(state: &GameState, field: &Field) -> Self {
        let mut deadline_ids = state
            .pieces
            .iter()
            .filter(|piece| {
                piece.owner == state.current_player
                    && piece.piece_type != PieceType::King
                    && piece.unstable
            })
            .map(|piece| piece.id.clone())
            .collect::<HashSet<_>>();
        for piece in unstable_pieces(state.current_player, state, field) {
            if piece.piece_type != PieceType::King {
                deadline_ids.insert(piece.id);
            }
        }
        Self {
            player: state.current_player,
            deadline_ids,
        }
    }
}

pub(crate) fn resolve_own_turn_consequences(
    player: Player,
    previous: &GameState,
    candidate: GameState,
) -> GameState {
    let previous_field = evaluate_field(previous);
    resolve_own_turn_consequences_with_field(player, previous, &previous_field, candidate)
}

fn resolve_own_turn_consequences_with_field(
    player: Player,
    previous: &GameState,
    previous_field: &Field,
    candidate: GameState,
) -> GameState {
    let mut deadlines = previous
        .pieces
        .iter()
        .filter(|piece| {
            piece.owner == player && piece.piece_type != PieceType::King && piece.unstable
        })
        .map(|piece| piece.id.clone())
        .collect::<HashSet<_>>();
    for piece in unstable_pieces(player, previous, &previous_field) {
        if piece.piece_type != PieceType::King {
            deadlines.insert(piece.id);
        }
    }
    let candidate_field = evaluate_field(&candidate);
    let marked = mark_instability(candidate, &candidate_field);
    let resolved = remove_unrescued_pieces(player, marked, &deadlines);
    let resolved_field = evaluate_field(&resolved);
    mark_instability(resolved, &resolved_field)
}

fn component_options(piece_type: PieceType, count: usize) -> Vec<Vec<i8>> {
    fn build(piece_type: PieceType, count: usize, values: &mut Vec<i8>, output: &mut Vec<Vec<i8>>) {
        if values.len() == count {
            if values.iter().filter(|value| **value != 0).count()
                == tuning_strength(piece_type, count)
            {
                output.push(values.clone());
            }
            return;
        }
        for value in [1, 0, -1] {
            values.push(value);
            build(piece_type, count, values, output);
            values.pop();
        }
    }
    let mut output = Vec::new();
    build(piece_type, count, &mut Vec::new(), &mut output);
    output
}

fn component_distance(left: &PlayerComponents, right: &PlayerComponents) -> usize {
    PIECE_TYPES
        .iter()
        .map(|piece_type| {
            left.get(*piece_type)
                .iter()
                .zip(right.get(*piece_type))
                .filter(|(left, right)| left != right)
                .count()
        })
        .sum()
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayableConfigurationHint {
    pub components: PlayerComponents,
    pub piece_id: String,
    pub piece_type: PieceType,
    pub destination: Position,
    pub changed_components: usize,
}

fn all_component_options(state: &GameState, player: Player) -> Vec<PlayerComponents> {
    let current = state.components.get(player);
    let pawn = component_options(PieceType::Pawn, current.pawn.len());
    let rook = component_options(PieceType::Rook, current.rook.len());
    let spy = component_options(PieceType::Spy, current.spy.len());
    let king = component_options(PieceType::King, current.king.len());
    let mut options = Vec::new();
    for pawn_values in &pawn {
        for rook_values in &rook {
            for spy_values in &spy {
                for king_values in &king {
                    options.push(PlayerComponents {
                        pawn: pawn_values.clone(),
                        rook: rook_values.clone(),
                        spy: spy_values.clone(),
                        king: king_values.clone(),
                    });
                }
            }
        }
    }
    options.sort_by_key(|candidate| component_distance(candidate, current));
    options
}

pub fn find_closest_playable_configuration(
    player: Player,
    state: &GameState,
) -> Option<PlayableConfigurationHint> {
    let current = state.components.get(player);
    for components in all_component_options(state, player) {
        let mut tuned = state.clone();
        *tuned.components.get_mut(player) = components.clone();
        let field = evaluate_field(&tuned);
        let pieces = tuned
            .pieces
            .iter()
            .filter(|piece| piece.owner == player)
            .cloned()
            .collect::<Vec<_>>();
        for piece in pieces {
            for destination in get_legal_moves(&piece.id, &tuned, &field) {
                let moved = move_piece(tuned.clone(), &piece.id, destination);
                let resolved = resolve_own_turn_consequences(player, &tuned, moved);
                let resolved_field = evaluate_field(&resolved);
                if !is_king_unprotected(player, &resolved, &resolved_field) {
                    return Some(PlayableConfigurationHint {
                        components,
                        piece_id: piece.id,
                        piece_type: piece.piece_type,
                        destination,
                        changed_components: component_distance(
                            current,
                            resolved.components.get(player),
                        ),
                    });
                }
            }
        }
    }
    None
}

fn has_playable_move_in_current_configuration(
    player: Player,
    state: &GameState,
    field: &Field,
) -> bool {
    let pieces = state
        .pieces
        .iter()
        .filter(|piece| piece.owner == player)
        .cloned()
        .collect::<Vec<_>>();
    for piece in pieces {
        for destination in get_legal_moves(&piece.id, state, field) {
            let moved = move_piece(state.clone(), &piece.id, destination);
            let resolved = resolve_own_turn_consequences(player, state, moved);
            let resolved_field = evaluate_field(&resolved);
            if !is_king_unprotected(player, &resolved, &resolved_field) {
                return true;
            }
        }
    }
    false
}

fn board_coordinate(position: Position) -> String {
    let file = char::from_u32('A' as u32 + position.x as u32).unwrap_or('?');
    format!("{file}{}", BOARD_SIZE - position.y)
}

fn win_status(player: Player) -> GameStatus {
    match player {
        Player::Red => GameStatus::RedWon,
        Player::Blue => GameStatus::BlueWon,
    }
}

pub fn begin_turn(state: GameState, analyze_checkmate: bool) -> GameState {
    begin_turn_with_options(state, analyze_checkmate, false)
}

fn begin_turn_quiet(state: GameState) -> GameState {
    begin_turn_with_options(state, false, true)
}

fn begin_turn_with_options(state: GameState, analyze_checkmate: bool, quiet: bool) -> GameState {
    if state.status != GameStatus::Playing {
        return state;
    }
    let field = evaluate_field(&state);
    let mut resolved = mark_instability(state, &field);
    let resolved_field = evaluate_field(&resolved);
    if is_king_unprotected(resolved.current_player, &resolved, &resolved_field) {
        if !analyze_checkmate {
            if !quiet {
                resolved.message =
                    format!("{} Big Hat is in check", resolved.current_player.name());
            }
            return resolved;
        }
        if let Some(rescue) =
            find_closest_playable_configuration(resolved.current_player, &resolved)
        {
            let hint = if rescue.changed_components > 0 {
                format!(
                    "tune, then move {} to {}",
                    piece_type_name(rescue.piece_type),
                    board_coordinate(rescue.destination)
                )
            } else {
                format!(
                    "move {} to {}",
                    piece_type_name(rescue.piece_type),
                    board_coordinate(rescue.destination)
                )
            };
            resolved.message = format!(
                "{} Big Hat is in check · {hint}",
                resolved.current_player.name()
            );
            return resolved;
        }
        resolved.status = win_status(resolved.current_player.opponent());
        resolved.selected_piece_id = None;
        resolved.message = format!(
            "{} Big Hat is in check · no legal rescue found",
            resolved.current_player.name()
        );
        return resolved;
    }
    if analyze_checkmate
        && !has_playable_move_in_current_configuration(
            resolved.current_player,
            &resolved,
            &resolved_field,
        )
    {
        let playable = find_closest_playable_configuration(resolved.current_player, &resolved);
        if playable.is_none() {
            resolved.status = win_status(resolved.current_player.opponent());
            resolved.selected_piece_id = None;
            if !quiet {
                resolved.message = format!("{} has no legal move", resolved.current_player.name());
            }
            return resolved;
        }
    }
    let unstable = unstable_pieces(resolved.current_player, &resolved, &resolved_field)
        .into_iter()
        .find(|piece| piece.piece_type != PieceType::King);
    if !quiet {
        resolved.message = match unstable {
            Some(piece) => format!(
                "{} must rescue an unstable {}",
                resolved.current_player.name(),
                piece_type_name(piece.piece_type)
            ),
            None => format!("{} to move", resolved.current_player.name()),
        };
    }
    resolved
}

fn piece_type_name(piece_type: PieceType) -> &'static str {
    match piece_type {
        PieceType::Pawn => "round hat",
        PieceType::Rook => "tower",
        PieceType::Spy => "triangle hat",
        PieceType::King => "big hat",
    }
}

pub fn apply_move(
    piece_id: &str,
    destination: Position,
    state: GameState,
    analyze_checkmate: bool,
) -> MoveResult {
    let field = evaluate_field(&state);
    apply_move_with_field(
        piece_id,
        destination,
        state,
        &field,
        analyze_checkmate,
        true,
        false,
    )
}

pub(crate) fn apply_known_legal_move(
    piece_id: &str,
    destination: Position,
    state: GameState,
    field: &Field,
    analyze_checkmate: bool,
) -> MoveResult {
    apply_move_with_field(
        piece_id,
        destination,
        state,
        field,
        analyze_checkmate,
        false,
        false,
    )
}

pub(crate) fn apply_search_move(
    piece_id: &str,
    destination: Position,
    state: GameState,
    field: &Field,
) -> MoveResult {
    apply_move_with_field(piece_id, destination, state, field, false, false, true)
}

fn apply_move_with_field(
    piece_id: &str,
    destination: Position,
    state: GameState,
    field: &Field,
    analyze_checkmate: bool,
    validate_destination: bool,
    quiet: bool,
) -> MoveResult {
    if state.status != GameStatus::Playing {
        return rejected(state, "The game is over.");
    }
    let Some(piece) = state.pieces.iter().find(|piece| piece.id == piece_id) else {
        return rejected(state, "Choose one of your pieces.");
    };
    if piece.owner != state.current_player {
        return rejected(state, "Choose one of your pieces.");
    }
    if validate_destination && !get_legal_moves(piece_id, &state, field).contains(&destination) {
        return rejected(state, "That square is not a legal move.");
    }

    let previous = state;
    let candidate = move_piece(previous.clone(), piece_id, destination);
    let mut resolved = resolve_own_turn_consequences_with_field(
        previous.current_player,
        &previous,
        field,
        candidate,
    );
    let resolved_field = evaluate_field(&resolved);
    if is_king_unprotected(previous.current_player, &resolved, &resolved_field) {
        return rejected(previous, "That move would leave your Big Hat unprotected.");
    }
    let losses = if quiet {
        Vec::new()
    } else {
        let remaining = resolved
            .pieces
            .iter()
            .map(|piece| piece.id.as_str())
            .collect::<HashSet<_>>();
        previous
            .pieces
            .iter()
            .filter(|piece| {
                piece.owner == previous.current_player
                    && piece.piece_type != PieceType::King
                    && !remaining.contains(piece.id.as_str())
            })
            .map(|piece| piece_type_name(piece.piece_type))
            .collect::<Vec<_>>()
    };
    resolved.current_player = previous.current_player.opponent();
    if previous.current_player == Player::Red {
        resolved.turn_number += 1;
    }
    resolved.selected_piece_id = None;
    if !quiet {
        resolved.history.push(previous.snapshot());
    }
    let mut next = if quiet {
        begin_turn_quiet(resolved)
    } else {
        begin_turn(resolved, analyze_checkmate)
    };
    if !losses.is_empty() && next.status == GameStatus::Playing {
        next.message = format!(
            "{} lost {} · {}",
            previous.current_player.name(),
            losses.join(", "),
            next.message
        );
    }
    MoveResult {
        ok: true,
        state: next,
        reason: None,
    }
}

pub fn apply_training_move(piece_id: &str, destination: Position, state: GameState) -> MoveResult {
    if state.status != GameStatus::Playing {
        return MoveResult {
            ok: false,
            state,
            reason: None,
        };
    }

    let previous = state;
    let candidate = move_piece(previous.clone(), piece_id, destination);
    let mut resolved = resolve_own_turn_consequences(previous.current_player, &previous, candidate);
    let resolved_field = evaluate_field(&resolved);
    if is_king_unprotected(previous.current_player, &resolved, &resolved_field) {
        return MoveResult {
            ok: false,
            state: previous,
            reason: None,
        };
    }
    resolved.current_player = previous.current_player.opponent();
    if previous.current_player == Player::Red {
        resolved.turn_number += 1;
    }
    resolved.selected_piece_id = None;
    let next = begin_turn(resolved, false);
    MoveResult {
        ok: true,
        state: next,
        reason: None,
    }
}

pub fn training_move_is_safe(piece_id: &str, destination: Position, state: &GameState) -> bool {
    let field = evaluate_field(state);
    training_move_is_safe_with_field(piece_id, destination, state, &field)
}

pub fn training_move_is_safe_with_field(
    piece_id: &str,
    destination: Position,
    state: &GameState,
    field: &Field,
) -> bool {
    let context = TrainingSafetyContext::new(state, field);
    training_move_is_safe_with_context(piece_id, destination, state, &context)
}

pub fn training_move_is_safe_with_context(
    piece_id: &str,
    destination: Position,
    state: &GameState,
    context: &TrainingSafetyContext,
) -> bool {
    if state.status != GameStatus::Playing {
        return false;
    }
    let mut candidate = training_candidate_state(state, piece_id, destination);

    loop {
        let candidate_field = evaluate_field(&candidate);
        let lost_ids = unstable_pieces(context.player, &candidate, &candidate_field)
            .into_iter()
            .filter(|piece| {
                piece.piece_type != PieceType::King && context.deadline_ids.contains(&piece.id)
            })
            .map(|piece| piece.id)
            .collect::<HashSet<_>>();
        if lost_ids.is_empty() {
            return !is_king_unprotected(context.player, &candidate, &candidate_field);
        }
        candidate
            .pieces
            .retain(|piece| !lost_ids.contains(&piece.id));
    }
}

pub fn get_playable_moves(piece_id: &str, state: &GameState) -> Vec<Position> {
    let field = evaluate_field(state);
    get_legal_moves(piece_id, state, &field)
        .into_iter()
        .filter(|destination| apply_move(piece_id, *destination, state.clone(), false).ok)
        .collect()
}

pub fn resign_in_check(state: GameState) -> MoveResult {
    if state.status != GameStatus::Playing {
        return rejected(state, "The game is over.");
    }
    let field = evaluate_field(&state);
    let mut resolved = mark_instability(state.clone(), &field);
    let resolved_field = evaluate_field(&resolved);
    if !is_king_unprotected(resolved.current_player, &resolved, &resolved_field) {
        return rejected(state, "You can resign only while your Big Hat is in check.");
    }
    resolved.status = win_status(resolved.current_player.opponent());
    resolved.selected_piece_id = None;
    resolved.history.push(state.snapshot());
    resolved.message = format!("{} resigned while in check", resolved.current_player.name());
    MoveResult {
        ok: true,
        state: resolved,
        reason: None,
    }
}

pub fn apply_closest_playable_hint(state: GameState) -> MoveResult {
    if state.status != GameStatus::Playing {
        return rejected(state, "The game is over.");
    }
    let field = evaluate_field(&state);
    let resolved = mark_instability(state.clone(), &field);
    let resolved_field = evaluate_field(&resolved);
    if !is_king_unprotected(resolved.current_player, &resolved, &resolved_field) {
        return rejected(
            state,
            "Hints are available only while your Big Hat is in check.",
        );
    }
    let Some(hint) = find_closest_playable_configuration(resolved.current_player, &resolved) else {
        return rejected(state, "No legal escape exists.");
    };
    let mut tuned = resolved;
    *tuned.components.get_mut(tuned.current_player) = hint.components;
    *tuned.activation_orders.get_mut(tuned.current_player) =
        activation_order_for_profile(tuned.components.get(tuned.current_player));
    let tuned_field = evaluate_field(&tuned);
    let mut marked = mark_instability(tuned, &tuned_field);
    let change_text = if hint.changed_components == 0 {
        "Current tuning works".to_owned()
    } else {
        format!(
            "{} control{} changed",
            hint.changed_components,
            if hint.changed_components == 1 {
                ""
            } else {
                "s"
            }
        )
    };
    marked.selected_piece_id = Some(hint.piece_id);
    marked.history.push(state.snapshot());
    marked.message = format!(
        "Hint · {change_text} · move {} to {}",
        piece_type_name(hint.piece_type),
        board_coordinate(hint.destination)
    );
    MoveResult {
        ok: true,
        state: marked,
        reason: None,
    }
}

pub fn reset_tuning(state: GameState) -> MoveResult {
    if state.status != GameStatus::Playing {
        return rejected(state, "The game is over.");
    }
    let player = state.current_player;
    if component_distance(state.components.get(player), &state.default_components) == 0 {
        return rejected(state, "Tuning already matches the defaults.");
    }
    let previous = state;
    let mut candidate = previous.clone();
    *candidate.components.get_mut(player) = candidate.default_components.clone();
    *candidate.activation_orders.get_mut(player) =
        activation_order_for_profile(&candidate.default_components);
    let field = evaluate_field(&candidate);
    let mut marked = mark_instability(candidate, &field);
    let marked_field = evaluate_field(&marked);
    marked.message = if is_king_unprotected(player, &marked, &marked_field) {
        format!("{} reset tuning · Big Hat remains in check", player.name())
    } else {
        format!(
            "{} reset tuning · move a piece to end the turn",
            player.name()
        )
    };
    marked.selected_piece_id = None;
    marked.history.push(previous.snapshot());
    MoveResult {
        ok: true,
        state: marked,
        reason: None,
    }
}

pub fn randomize_tuning(state: GameState, rolls: [f64; 4]) -> MoveResult {
    if state.status != GameStatus::Playing {
        return rejected(state, "The game is over.");
    }
    let player = state.current_player;
    let previous = state;
    let mut randomized = previous.components.get(player).clone();
    for (piece_type, roll) in PIECE_TYPES.into_iter().zip(rolls) {
        let options = component_options(piece_type, randomized.get(piece_type).len());
        let index = ((roll * options.len() as f64).floor() as usize).min(options.len() - 1);
        *randomized.get_mut(piece_type) = options[index].clone();
    }
    if component_distance(&randomized, previous.components.get(player)) == 0 {
        let alternatives = component_options(PieceType::Pawn, randomized.pawn.len())
            .into_iter()
            .filter(|profile| profile != &randomized.pawn)
            .collect::<Vec<_>>();
        randomized.pawn = alternatives[0].clone();
    }
    let mut candidate = previous.clone();
    *candidate.components.get_mut(player) = randomized.clone();
    *candidate.activation_orders.get_mut(player) = activation_order_for_profile(&randomized);
    let field = evaluate_field(&candidate);
    let mut marked = mark_instability(candidate, &field);
    let marked_field = evaluate_field(&marked);
    marked.message = if is_king_unprotected(player, &marked, &marked_field) {
        format!(
            "{} randomized tuning · Big Hat remains in check",
            player.name()
        )
    } else {
        format!(
            "{} randomized tuning · move a piece to end the turn",
            player.name()
        )
    };
    marked.selected_piece_id = None;
    marked.history.push(previous.snapshot());
    MoveResult {
        ok: true,
        state: marked,
        reason: None,
    }
}

pub fn apply_tuning(
    player: Player,
    piece_type: PieceType,
    component_index: usize,
    value: i8,
    state: GameState,
) -> MoveResult {
    if state.status != GameStatus::Playing || player != state.current_player {
        return rejected(state, "It is not that player's turn.");
    }
    if value == 0 {
        return rejected(state, "Controls must stay at full strength.");
    }
    let current_value = state
        .components
        .get(player)
        .get(piece_type)
        .get(component_index)
        .copied();
    if current_value == Some(value) {
        return rejected(state, "Choose a different sign.");
    }
    let Some(current_value) = current_value else {
        return rejected(state, "Unknown component.");
    };
    let previous = state;
    let mut candidate = previous.clone();
    let coefficients = candidate.components.get_mut(player).get_mut(piece_type);
    let active_indices = coefficients
        .iter()
        .enumerate()
        .filter_map(|(index, coefficient)| (*coefficient != 0).then_some(index))
        .collect::<Vec<_>>();
    let mut order = candidate
        .activation_orders
        .get(player)
        .get(piece_type)
        .iter()
        .copied()
        .filter(|index| active_indices.contains(index))
        .collect::<Vec<_>>();
    for index in &active_indices {
        if !order.contains(index) {
            order.push(*index);
        }
    }
    order.retain(|index| *index != component_index);
    if current_value == 0
        && active_indices.len() >= tuning_strength(piece_type, coefficients.len())
        && !order.is_empty()
    {
        let evicted = order.remove(0);
        coefficients[evicted] = 0;
    }
    coefficients[component_index] = value;
    order.push(component_index);
    *candidate
        .activation_orders
        .get_mut(player)
        .get_mut(piece_type) = order;
    let field = evaluate_field(&candidate);
    let mut marked = mark_instability(candidate, &field);
    let marked_field = evaluate_field(&marked);
    marked.message = if is_king_unprotected(player, &marked, &marked_field) {
        format!(
            "{} Big Hat is in check · move to rescue the Big Hat",
            player.name()
        )
    } else {
        format!("{} tuning · move a piece to end the turn", player.name())
    };
    marked.history.push(previous.snapshot());
    MoveResult {
        ok: true,
        state: marked,
        reason: None,
    }
}

fn rejected(state: GameState, reason: &str) -> MoveResult {
    MoveResult {
        ok: false,
        state,
        reason: Some(reason.to_owned()),
    }
}
