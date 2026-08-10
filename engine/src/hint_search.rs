use crate::board::*;
use crate::field::evaluate_field;
use crate::model::*;
use crate::rules::resolve_own_turn_consequences;
use serde::Serialize;
use std::collections::{HashSet, VecDeque};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
type SearchStartedAt = Instant;

#[cfg(target_arch = "wasm32")]
type SearchStartedAt = f64;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Date, js_name = now)]
    fn date_now() -> f64;
}

#[cfg(not(target_arch = "wasm32"))]
fn search_started_at() -> SearchStartedAt {
    Instant::now()
}

#[cfg(target_arch = "wasm32")]
fn search_started_at() -> SearchStartedAt {
    date_now()
}

#[cfg(not(target_arch = "wasm32"))]
fn deadline_reached(started_at: &SearchStartedAt, deadline: Duration) -> bool {
    started_at.elapsed() >= deadline
}

#[cfg(target_arch = "wasm32")]
fn deadline_reached(started_at: &SearchStartedAt, deadline: Duration) -> bool {
    date_now() - *started_at >= deadline.as_millis() as f64
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HintSearchSuccess {
    pub ok: bool,
    pub state: GameState,
    #[serde(rename = "pieceID")]
    pub piece_id: String,
    pub moves: Vec<Position>,
    pub safe: bool,
    pub loss_count: usize,
    pub tuning_distance: usize,
    pub tuned_kinds: Vec<String>,
    pub exhausted: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HintSearchFailure {
    pub ok: bool,
    pub reason: String,
    pub exhausted: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum HintSearchResult {
    Success(HintSearchSuccess),
    Failure(HintSearchFailure),
}

#[derive(Clone)]
struct TuningNode {
    state: GameState,
    distance: usize,
}

#[derive(Clone)]
struct MoveEvaluation {
    piece_id: String,
    destination: Position,
    loss_count: usize,
}

#[derive(Clone)]
struct Candidate {
    state: GameState,
    piece_id: String,
    moves: Vec<Position>,
    primary: Position,
    loss_count: usize,
    same_loss_moves: usize,
    tuning_distance: usize,
    sequence: usize,
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
) -> Option<(Vec<i8>, Vec<usize>)> {
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
    let mut order = state
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
        && active_indices.len() >= tuning_strength(piece_type, profile.len())
        && !order.is_empty()
    {
        let evicted = order.remove(0);
        profile[evicted] = 0;
    }

    profile[component_index] = value;
    order.push(component_index);
    Some((profile, order))
}

fn tuning_key(state: &GameState, player: Player) -> String {
    serde_json::to_string(&(
        state.components.get(player),
        state.activation_orders.get(player),
    ))
    .expect("serializable tuning key")
}

fn tuning_neighbors(state: &GameState, player: Player) -> Vec<GameState> {
    let mut neighbors = Vec::new();
    for piece_type in PIECE_TYPES {
        let count = state.components.get(player).get(piece_type).len();
        for component_index in 0..count {
            for value in [-1, 1] {
                let Some((profile, order)) =
                    profile_after_control_change(state, player, piece_type, component_index, value)
                else {
                    continue;
                };
                let mut next = state.clone();
                *next.components.get_mut(player).get_mut(piece_type) = profile;
                *next.activation_orders.get_mut(player).get_mut(piece_type) = order;
                next.selected_piece_id = None;
                neighbors.push(next);
            }
        }
    }
    neighbors
}

fn tuned_kinds(current: &PlayerComponents, tuned: &PlayerComponents) -> Vec<String> {
    PIECE_TYPES
        .iter()
        .filter_map(|piece_type| {
            (current.get(*piece_type) != tuned.get(*piece_type))
                .then_some(piece_type.name().to_owned())
        })
        .collect()
}

fn move_piece(mut state: GameState, piece_id: &str, destination: Position) -> GameState {
    if let Some(piece) = state.pieces.iter_mut().find(|piece| piece.id == piece_id) {
        piece.position = destination;
    }
    state
}

fn own_loss_count(player: Player, before: &GameState, after: &GameState) -> usize {
    let remaining = after
        .pieces
        .iter()
        .map(|piece| piece.id.as_str())
        .collect::<HashSet<_>>();
    before
        .pieces
        .iter()
        .filter(|piece| piece.owner == player && !remaining.contains(piece.id.as_str()))
        .count()
}

fn playable_evaluations(
    player: Player,
    state: &GameState,
    focused_piece_id: Option<&str>,
) -> Vec<MoveEvaluation> {
    let field = evaluate_field(state);
    let pieces = state
        .pieces
        .iter()
        .filter(|piece| {
            piece.owner == player && focused_piece_id.is_none_or(|piece_id| piece.id == piece_id)
        })
        .cloned()
        .collect::<Vec<_>>();

    let mut evaluations = Vec::new();
    for piece in pieces {
        for destination in get_legal_moves(&piece.id, state, &field) {
            let moved = move_piece(state.clone(), &piece.id, destination);
            let resolved = resolve_own_turn_consequences(player, state, moved);
            let resolved_field = evaluate_field(&resolved);
            if is_king_unprotected(player, &resolved, &resolved_field) {
                continue;
            }
            evaluations.push(MoveEvaluation {
                piece_id: piece.id.clone(),
                destination,
                loss_count: own_loss_count(player, state, &resolved),
            });
        }
    }
    evaluations
}

fn grouped_candidate(
    state: GameState,
    evaluations: &[MoveEvaluation],
    loss_count: usize,
    tuning_distance: usize,
    sequence: usize,
) -> Option<Candidate> {
    let same_loss = evaluations
        .iter()
        .filter(|evaluation| evaluation.loss_count == loss_count)
        .collect::<Vec<_>>();
    let primary = same_loss.first()?;
    let moves = same_loss
        .iter()
        .filter(|evaluation| evaluation.piece_id == primary.piece_id)
        .map(|evaluation| evaluation.destination)
        .collect::<Vec<_>>();
    Some(Candidate {
        state,
        piece_id: primary.piece_id.clone(),
        moves,
        primary: primary.destination,
        loss_count,
        same_loss_moves: same_loss.len(),
        tuning_distance,
        sequence,
    })
}

fn better_least_loss(left: &Candidate, right: &Candidate) -> bool {
    (
        left.loss_count,
        std::cmp::Reverse(left.same_loss_moves),
        left.tuning_distance,
        left.sequence,
        left.piece_id.as_str(),
        left.primary.y,
        left.primary.x,
    ) < (
        right.loss_count,
        std::cmp::Reverse(right.same_loss_moves),
        right.tuning_distance,
        right.sequence,
        right.piece_id.as_str(),
        right.primary.y,
        right.primary.x,
    )
}

fn success(
    player: Player,
    current: &PlayerComponents,
    candidate: Candidate,
    safe: bool,
    exhausted: bool,
) -> HintSearchResult {
    HintSearchResult::Success(HintSearchSuccess {
        ok: true,
        tuned_kinds: tuned_kinds(current, candidate.state.components.get(player)),
        state: candidate.state,
        piece_id: candidate.piece_id,
        moves: candidate.moves,
        safe,
        loss_count: candidate.loss_count,
        tuning_distance: candidate.tuning_distance,
        exhausted,
    })
}

fn hint_search_scope(
    player: Player,
    focused_piece_id: Option<&str>,
    state: &GameState,
    max_tuning_states: u32,
    time_budget_ms: u32,
) -> HintSearchResult {
    if state.status != GameStatus::Playing {
        return HintSearchResult::Failure(HintSearchFailure {
            ok: false,
            reason: "no playable moves".to_owned(),
            exhausted: false,
        });
    }

    let current = state.components.get(player).clone();
    let max_states = if max_tuning_states == 0 {
        usize::MAX
    } else {
        max_tuning_states as usize
    };
    let deadline = Duration::from_millis(u64::from(time_budget_ms));
    let enforce_deadline = time_budget_ms > 0;
    let started_at = search_started_at();
    let mut exhausted = false;
    let mut inspected = 0_usize;
    let mut sequence = 0_usize;
    let mut best: Option<Candidate> = None;
    let mut queue = VecDeque::from([TuningNode {
        state: state.clone(),
        distance: 0,
    }]);
    let mut seen = HashSet::from([tuning_key(state, player)]);

    while let Some(node) = queue.pop_front() {
        if inspected >= max_states {
            exhausted = true;
            break;
        }
        inspected += 1;

        let field = evaluate_field(&node.state);
        let tuned = mark_instability(node.state.clone(), &field);
        let evaluations = playable_evaluations(player, &tuned, focused_piece_id);
        if evaluations
            .iter()
            .any(|evaluation| evaluation.loss_count == 0)
        {
            return success(
                player,
                &current,
                grouped_candidate(tuned, &evaluations, 0, node.distance, sequence)
                    .expect("safe candidate exists"),
                true,
                exhausted,
            );
        }

        if let Some(min_loss) = evaluations
            .iter()
            .map(|evaluation| evaluation.loss_count)
            .min()
        {
            let candidate =
                grouped_candidate(tuned, &evaluations, min_loss, node.distance, sequence)
                    .expect("least-loss candidate exists");
            if best
                .as_ref()
                .is_none_or(|current_best| better_least_loss(&candidate, current_best))
            {
                best = Some(candidate);
            }
        }

        for neighbor in tuning_neighbors(&node.state, player) {
            let key = tuning_key(&neighbor, player);
            if seen.insert(key) {
                queue.push_back(TuningNode {
                    state: neighbor,
                    distance: node.distance + 1,
                });
            }
        }

        sequence += 1;
        if enforce_deadline && deadline_reached(&started_at, deadline) {
            exhausted = true;
            break;
        }
    }

    if let Some(candidate) = best {
        return success(player, &current, candidate, false, exhausted);
    }

    HintSearchResult::Failure(HintSearchFailure {
        ok: false,
        reason: "no playable moves".to_owned(),
        exhausted,
    })
}

pub fn hint_search(
    player: Player,
    focused_piece_id: Option<&str>,
    state: &GameState,
    max_tuning_states: u32,
    time_budget_ms: u32,
) -> HintSearchResult {
    let Some(piece_id) = focused_piece_id else {
        return hint_search_scope(player, None, state, max_tuning_states, time_budget_ms);
    };

    let focused = hint_search_scope(
        player,
        Some(piece_id),
        state,
        max_tuning_states,
        time_budget_ms,
    );
    if matches!(&focused, HintSearchResult::Success(hint) if hint.safe) {
        return focused;
    }

    let global = hint_search_scope(player, None, state, max_tuning_states, time_budget_ms);
    if matches!(&global, HintSearchResult::Success(hint) if hint.safe) {
        return global;
    }

    global
}
