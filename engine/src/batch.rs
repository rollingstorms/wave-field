use crate::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

pub const TRAINING_BOARD_CHANNELS: usize = 13;
pub const TRAINING_BOARD_SIZE: usize = 7;
pub const TRAINING_PIECE_IDS: [&str; 12] = [
    "blue-rook-1",
    "blue-king-1",
    "blue-rook-2",
    "blue-pawn-1",
    "blue-spy-1",
    "blue-pawn-2",
    "red-pawn-1",
    "red-spy-1",
    "red-pawn-2",
    "red-rook-1",
    "red-king-1",
    "red-rook-2",
];

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiBatchSummary {
    pub games: u64,
    pub red_wins: u64,
    pub blue_wins: u64,
    pub capped: u64,
    pub decisive: u64,
    pub total_plies: u64,
    pub mean_plies: f64,
    pub min_plies: u64,
    pub max_plies: u64,
    pub elapsed_ms: u128,
    pub ms_per_game: f64,
    pub ms_per_ply: f64,
    pub plies_per_second: f64,
    pub projected_500_games_ms: f64,
    pub decisive_rate: f64,
    pub cap_rate: f64,
    pub red_win_rate: f64,
    pub blue_win_rate: f64,
    pub blue_decisive_share: f64,
    pub first_loss_red: u64,
    pub first_loss_blue: u64,
    pub first_loss_team_wins: u64,
    pub first_loss_team_losses: u64,
    pub first_loss_team_capped: u64,
    pub first_loss_win_rate: f64,
    pub underdog_wins: u64,
    pub underdog_win_rate: f64,
    pub checks_created: u64,
    pub check_rate_per_ply: f64,
    pub unstable_created: u64,
    pub unstable_creation_rate_per_ply: f64,
    pub rescue_opportunities: u64,
    pub rescues: u64,
    pub rescue_rate: f64,
    pub material_losses: u64,
    pub losses_by_piece: HashMap<String, u64>,
    pub losses_by_owner: HashMap<String, u64>,
    pub winner_piece_counts: HashMap<String, u64>,
    pub repeated_piece_destinations: u64,
    pub repeated_piece_destination_rate: f64,
    pub avg_loser_first_loss_ply: f64,
    pub avg_winner_pieces: f64,
    pub avg_loser_pieces: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LeanBatchSummary {
    pub games: u64,
    pub red_wins: u64,
    pub blue_wins: u64,
    pub capped: u64,
    pub decisive: u64,
    pub total_plies: u64,
    pub mean_plies: f64,
    pub min_plies: u64,
    pub max_plies: u64,
    pub elapsed_ms: u128,
    pub ms_per_game: f64,
    pub ms_per_ply: f64,
    pub plies_per_second: f64,
    pub projected_500_games_ms: f64,
}

#[derive(Default)]
struct ProfileTotals {
    candidate_generation_ns: u128,
    apply_move_ns: u128,
    candidate_moves: u64,
    candidate_turns: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RandomProfileSummary {
    pub batch: LeanBatchSummary,
    pub candidate_generation_ms: f64,
    pub apply_move_ms: f64,
    pub avg_candidate_generation_ms_per_ply: f64,
    pub avg_apply_move_ms_per_ply: f64,
    pub avg_candidate_moves_per_ply: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingBatchSample {
    pub board: Vec<f32>,
    pub side: Vec<f32>,
    pub legal_action_indexes: Vec<usize>,
    pub action_index: usize,
    pub player: String,
    pub value: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingObservation {
    pub board: Vec<f32>,
    pub side: Vec<f32>,
    pub legal_action_indexes: Vec<usize>,
    pub player: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingBatchSummary {
    pub games: u64,
    pub samples: usize,
    pub red_wins: u64,
    pub blue_wins: u64,
    pub capped: u64,
    pub decisive: u64,
    pub total_plies: u64,
    pub mean_plies: f64,
    pub elapsed_ms: u128,
    pub samples_per_second: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingBatch {
    pub summary: TrainingBatchSummary,
    pub samples: Vec<TrainingBatchSample>,
}

#[derive(Default)]
struct TrainingProfileTotals {
    legal_scan_ns: u128,
    candidate_apply_ns: u128,
    candidate_generation_ns: u128,
    sample_encoding_ns: u128,
    value_assignment_ns: u128,
    total_candidates: u64,
    attempted_candidates: u64,
    candidate_turns: u64,
    encoded_samples: u64,
    legal_action_indexes: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingProfileSummary {
    pub batch: TrainingBatchSummary,
    pub legal_scan_ms: f64,
    pub candidate_apply_ms: f64,
    pub candidate_generation_ms: f64,
    pub sample_encoding_ms: f64,
    pub value_assignment_ms: f64,
    pub unprofiled_ms: f64,
    pub avg_candidate_generation_ms_per_ply: f64,
    pub avg_legal_scan_ms_per_ply: f64,
    pub avg_candidate_apply_ms_per_candidate: f64,
    pub avg_sample_encoding_ms_per_sample: f64,
    pub avg_value_assignment_ms_per_game: f64,
    pub avg_candidates_per_ply: f64,
    pub avg_attempted_candidates_per_ply: f64,
    pub avg_legal_indexes_per_sample: f64,
}

#[derive(Default)]
struct GameMetrics {
    first_loss_owner: Option<Player>,
    first_loss_ply: Option<u64>,
    checks_created: u64,
    unstable_created: u64,
    rescue_opportunities: u64,
    rescues: u64,
    material_losses: u64,
    losses_by_piece: HashMap<String, u64>,
    losses_by_owner: HashMap<String, u64>,
    repeated_piece_destinations: u64,
    piece_destinations: HashSet<String>,
}

fn player_key(player: Player) -> &'static str {
    match player {
        Player::Red => "red",
        Player::Blue => "blue",
    }
}

fn piece_type_key(piece_type: PieceType) -> &'static str {
    match piece_type {
        PieceType::Pawn => "pawn",
        PieceType::Rook => "rook",
        PieceType::Spy => "spy",
        PieceType::King => "king",
    }
}

fn piece_type_index(piece_type: PieceType) -> usize {
    match piece_type {
        PieceType::Pawn => 0,
        PieceType::Rook => 1,
        PieceType::Spy => 2,
        PieceType::King => 3,
    }
}

fn player_index(player: Player) -> usize {
    match player {
        Player::Red => 0,
        Player::Blue => 1,
    }
}

pub fn training_piece_slot(piece_id: &str) -> usize {
    TRAINING_PIECE_IDS
        .iter()
        .position(|candidate| *candidate == piece_id)
        .expect("known training piece id")
}

pub fn training_square_index(position: Position) -> usize {
    position.y as usize * TRAINING_BOARD_SIZE + position.x as usize
}

pub fn training_action_index(piece_id: &str, destination: Position) -> usize {
    training_piece_slot(piece_id) * TRAINING_BOARD_SIZE * TRAINING_BOARD_SIZE
        + training_square_index(destination)
}

pub fn training_action_to_move(index: usize) -> Option<(String, Position)> {
    let squares = TRAINING_BOARD_SIZE * TRAINING_BOARD_SIZE;
    let slot = index / squares;
    if slot >= TRAINING_PIECE_IDS.len() {
        return None;
    }
    let square = index % squares;
    Some((
        TRAINING_PIECE_IDS[slot].to_owned(),
        Position {
            x: (square % TRAINING_BOARD_SIZE) as i32,
            y: (square / TRAINING_BOARD_SIZE) as i32,
        },
    ))
}

fn push_components(values: &mut Vec<f32>, components: &PlayerComponents) {
    values.extend(components.pawn.iter().map(|value| f32::from(*value)));
    values.extend(components.rook.iter().map(|value| f32::from(*value)));
    values.extend(components.spy.iter().map(|value| f32::from(*value)));
    values.extend(components.king.iter().map(|value| f32::from(*value)));
}

pub fn training_side(state: &GameState) -> Vec<f32> {
    let mut values = Vec::with_capacity(19);
    push_components(&mut values, &state.components.red);
    push_components(&mut values, &state.components.blue);
    values.push(if state.current_player == Player::Red {
        1.0
    } else {
        -1.0
    });
    values
}

fn set_board_value(board: &mut [f32], channel: usize, position: Position, value: f32) {
    let index = channel * TRAINING_BOARD_SIZE * TRAINING_BOARD_SIZE
        + position.y as usize * TRAINING_BOARD_SIZE
        + position.x as usize;
    board[index] = value;
}

pub fn training_board(state: &GameState, field: &Field) -> Vec<f32> {
    let mut board = vec![0.0; TRAINING_BOARD_CHANNELS * TRAINING_BOARD_SIZE * TRAINING_BOARD_SIZE];
    for piece in &state.pieces {
        let occupancy_channel = player_index(piece.owner) * 4 + piece_type_index(piece.piece_type);
        set_board_value(&mut board, occupancy_channel, piece.position, 1.0);
        if piece.unstable {
            let unstable_channel = match piece.owner {
                Player::Red => 8,
                Player::Blue => 9,
            };
            set_board_value(&mut board, unstable_channel, piece.position, 1.0);
        }
    }
    for y in 0..TRAINING_BOARD_SIZE {
        for x in 0..TRAINING_BOARD_SIZE {
            let value = field[y][x] as f32;
            let position = Position {
                x: x as i32,
                y: y as i32,
            };
            set_board_value(&mut board, 10, position, (value / 8.0).clamp(-1.0, 1.0));
            set_board_value(
                &mut board,
                11,
                position,
                (value.abs() / 8.0).clamp(0.0, 1.0),
            );
            set_board_value(
                &mut board,
                12,
                position,
                if state.current_player == Player::Red {
                    1.0
                } else {
                    -1.0
                },
            );
        }
    }
    board
}

fn material_value(state: &GameState, player: Player) -> f32 {
    let own = piece_count(state, player) as f32;
    let opponent = piece_count(state, player.opponent()) as f32;
    ((own - opponent) / 6.0).clamp(-1.0, 1.0)
}

fn result_value_for_state(state: &GameState, player: Player, material_for_capped: bool) -> f32 {
    match state.status {
        GameStatus::RedWon => {
            if player == Player::Red {
                1.0
            } else {
                -1.0
            }
        }
        GameStatus::BlueWon => {
            if player == Player::Blue {
                1.0
            } else {
                -1.0
            }
        }
        GameStatus::Playing => {
            if material_for_capped {
                material_value(state, player)
            } else {
                0.0
            }
        }
    }
}

pub fn playable_training_candidates(state: &GameState) -> Vec<(String, Position)> {
    let field = evaluate_field(state);
    let mut candidates = Vec::new();
    for piece in state
        .pieces
        .iter()
        .filter(|piece| piece.owner == state.current_player)
    {
        for destination in get_legal_moves(&piece.id, state, &field) {
            if training_move_is_safe_with_field(&piece.id, destination, state, &field) {
                candidates.push((piece.id.clone(), destination));
            }
        }
    }
    candidates
}

pub fn training_legal_action_indexes(state: &GameState) -> Vec<usize> {
    playable_training_candidates(state)
        .iter()
        .map(|(piece_id, destination)| training_action_index(piece_id, *destination))
        .collect()
}

pub fn playable_training_action_indexes(state: &GameState) -> Vec<usize> {
    let field = evaluate_field(state);
    playable_training_action_indexes_with_field(state, &field)
}

pub fn playable_training_action_indexes_with_field(state: &GameState, field: &Field) -> Vec<usize> {
    let mut indexes = Vec::new();
    for piece in state
        .pieces
        .iter()
        .filter(|piece| piece.owner == state.current_player)
    {
        for destination in get_legal_moves(&piece.id, state, field) {
            if training_move_is_safe_with_field(&piece.id, destination, state, field) {
                indexes.push(training_action_index(&piece.id, destination));
            }
        }
    }
    indexes
}

pub fn training_observation(state: &GameState) -> TrainingObservation {
    let field = evaluate_field(state);
    let candidates = playable_training_candidates(state);
    TrainingObservation {
        board: training_board(state, &field),
        side: training_side(state),
        legal_action_indexes: candidates
            .iter()
            .map(|(piece_id, destination)| training_action_index(piece_id, *destination))
            .collect(),
        player: player_key(state.current_player).to_owned(),
    }
}

fn playable_training_candidates_profiled(
    state: &GameState,
    profile: &mut TrainingProfileTotals,
) -> Vec<(String, Position)> {
    let started_at = Instant::now();
    let legal_started_at = Instant::now();
    let field = evaluate_field(state);
    let mut legal_moves = Vec::new();
    for piece in state
        .pieces
        .iter()
        .filter(|piece| piece.owner == state.current_player)
    {
        for destination in get_legal_moves(&piece.id, state, &field) {
            legal_moves.push((piece.id.clone(), destination));
        }
    }
    profile.legal_scan_ns += legal_started_at.elapsed().as_nanos();

    let mut candidates = Vec::new();
    profile.attempted_candidates += legal_moves.len() as u64;
    for (piece_id, destination) in legal_moves {
        let apply_started_at = Instant::now();
        let safe = training_move_is_safe_with_field(&piece_id, destination, state, &field);
        profile.candidate_apply_ns += apply_started_at.elapsed().as_nanos();
        if safe {
            candidates.push((piece_id, destination));
        }
    }
    profile.candidate_generation_ns += started_at.elapsed().as_nanos();
    profile.candidate_turns += 1;
    profile.total_candidates += candidates.len() as u64;
    candidates
}

fn encode_training_sample(
    state: &GameState,
    candidates: &[(String, Position)],
    action_index: usize,
) -> TrainingBatchSample {
    let field = evaluate_field(state);
    let legal_action_indexes = candidates
        .iter()
        .map(|(piece_id, destination)| training_action_index(piece_id, *destination))
        .collect::<Vec<_>>();
    let (piece_id, destination) = &candidates[action_index];
    TrainingBatchSample {
        board: training_board(state, &field),
        side: training_side(state),
        legal_action_indexes,
        action_index: training_action_index(piece_id, *destination),
        player: player_key(state.current_player).to_owned(),
        value: 0.0,
    }
}

fn encode_training_sample_profiled(
    state: &GameState,
    candidates: &[(String, Position)],
    action_index: usize,
    profile: &mut TrainingProfileTotals,
) -> TrainingBatchSample {
    let started_at = Instant::now();
    let sample = encode_training_sample(state, candidates, action_index);
    profile.sample_encoding_ns += started_at.elapsed().as_nanos();
    profile.encoded_samples += 1;
    profile.legal_action_indexes += sample.legal_action_indexes.len() as u64;
    sample
}

fn piece_count(state: &GameState, player: Player) -> u64 {
    state
        .pieces
        .iter()
        .filter(|piece| piece.owner == player)
        .count() as u64
}

fn piece_map(state: &GameState) -> HashMap<String, (Player, PieceType, Position, bool)> {
    state
        .pieces
        .iter()
        .map(|piece| {
            (
                piece.id.clone(),
                (
                    piece.owner,
                    piece.piece_type,
                    piece.position,
                    piece.unstable,
                ),
            )
        })
        .collect()
}

fn update_game_metrics(before: &GameState, after: &GameState, ply: u64, metrics: &mut GameMetrics) {
    let before_by_id = piece_map(before);
    let after_by_id = piece_map(after);

    for (id, (owner, piece_type, _, _)) in &before_by_id {
        if !after_by_id.contains_key(id) {
            metrics.material_losses += 1;
            *metrics
                .losses_by_piece
                .entry(piece_type_key(*piece_type).to_owned())
                .or_insert(0) += 1;
            *metrics
                .losses_by_owner
                .entry(player_key(*owner).to_owned())
                .or_insert(0) += 1;
            if metrics.first_loss_owner.is_none() {
                metrics.first_loss_owner = Some(*owner);
                metrics.first_loss_ply = Some(ply);
            }
        }
    }

    for (id, (_, _, before_position, _)) in &before_by_id {
        if let Some((_, _, after_position, _)) = after_by_id.get(id) {
            if before_position != after_position {
                let key = format!("{id}:{},{}", after_position.x, after_position.y);
                if !metrics.piece_destinations.insert(key) {
                    metrics.repeated_piece_destinations += 1;
                }
            }
        }
    }

    for (id, (_, piece_type, _, before_unstable)) in &before_by_id {
        if *piece_type == PieceType::King {
            continue;
        }
        if let Some((_, _, _, after_unstable)) = after_by_id.get(id) {
            if !before_unstable && *after_unstable {
                metrics.unstable_created += 1;
            }
            if *before_unstable {
                metrics.rescue_opportunities += 1;
                if !after_unstable {
                    metrics.rescues += 1;
                }
            }
        } else if *before_unstable {
            metrics.rescue_opportunities += 1;
        }
    }

    let field = evaluate_field(after);
    if is_king_unprotected(before.current_player.opponent(), after, &field) {
        metrics.checks_created += 1;
    }
}

pub fn simulate_ai_games(
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
    let mut min_game_plies = u64::MAX;
    let mut max_game_plies = 0;
    let mut first_loss_red = 0;
    let mut first_loss_blue = 0;
    let mut first_loss_team_wins = 0;
    let mut first_loss_team_losses = 0;
    let mut first_loss_team_capped = 0;
    let mut underdog_wins = 0;
    let mut checks_created = 0;
    let mut unstable_created = 0;
    let mut rescue_opportunities = 0;
    let mut rescues = 0;
    let mut material_losses = 0;
    let mut losses_by_piece = HashMap::new();
    let mut losses_by_owner = HashMap::new();
    let mut winner_piece_counts = HashMap::new();
    let mut repeated_piece_destinations = 0;
    let mut loser_first_loss_ply_total = 0;
    let mut loser_first_loss_games = 0;
    let mut winner_piece_total = 0;
    let mut loser_piece_total = 0;
    let mut decisive_piece_count_games = 0;

    for game in 0..games {
        let mut state = initial_state.clone();
        let mut metrics = GameMetrics::default();
        let game_seed = seed.wrapping_add(game as u32);
        let mut plies = 0;
        while state.status == GameStatus::Playing && plies < max_plies {
            let before = state.clone();
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
            update_game_metrics(&before, &state, plies, &mut metrics);
        }

        total_plies += plies;
        min_game_plies = min_game_plies.min(plies);
        max_game_plies = max_game_plies.max(plies);
        match state.status {
            GameStatus::RedWon => red_wins += 1,
            GameStatus::BlueWon => blue_wins += 1,
            GameStatus::Playing => capped += 1,
        }

        if let Some(owner) = metrics.first_loss_owner {
            match owner {
                Player::Red => first_loss_red += 1,
                Player::Blue => first_loss_blue += 1,
            }
            match state.status {
                GameStatus::RedWon if owner == Player::Red => first_loss_team_wins += 1,
                GameStatus::BlueWon if owner == Player::Blue => first_loss_team_wins += 1,
                GameStatus::RedWon | GameStatus::BlueWon => first_loss_team_losses += 1,
                GameStatus::Playing => first_loss_team_capped += 1,
            }
            if matches!(
                (state.status, owner),
                (GameStatus::RedWon, Player::Blue) | (GameStatus::BlueWon, Player::Red)
            ) {
                loser_first_loss_ply_total += metrics.first_loss_ply.unwrap_or(0);
                loser_first_loss_games += 1;
            }
        }

        if state.status != GameStatus::Playing {
            let red_pieces = piece_count(&state, Player::Red);
            let blue_pieces = piece_count(&state, Player::Blue);
            let (winner_pieces, loser_pieces) = match state.status {
                GameStatus::RedWon => (red_pieces, blue_pieces),
                GameStatus::BlueWon => (blue_pieces, red_pieces),
                GameStatus::Playing => unreachable!(),
            };
            *winner_piece_counts
                .entry(winner_pieces.to_string())
                .or_insert(0) += 1;
            winner_piece_total += winner_pieces;
            loser_piece_total += loser_pieces;
            decisive_piece_count_games += 1;
            if winner_pieces < loser_pieces {
                underdog_wins += 1;
            }
        }

        checks_created += metrics.checks_created;
        unstable_created += metrics.unstable_created;
        rescue_opportunities += metrics.rescue_opportunities;
        rescues += metrics.rescues;
        material_losses += metrics.material_losses;
        repeated_piece_destinations += metrics.repeated_piece_destinations;
        for (piece_type, count) in metrics.losses_by_piece {
            *losses_by_piece.entry(piece_type).or_insert(0) += count;
        }
        for (owner, count) in metrics.losses_by_owner {
            *losses_by_owner.entry(owner).or_insert(0) += count;
        }
    }

    let elapsed_ms = started_at.elapsed().as_millis();
    let decisive = red_wins + blue_wins;
    let ms_per_game = ratio(elapsed_ms as f64, games);
    let ms_per_ply = ratio(elapsed_ms as f64, total_plies);

    AiBatchSummary {
        games,
        red_wins,
        blue_wins,
        capped,
        decisive,
        total_plies,
        mean_plies: ratio(total_plies as f64, games),
        min_plies: if games == 0 { 0 } else { min_game_plies },
        max_plies: max_game_plies,
        elapsed_ms,
        ms_per_game,
        ms_per_ply,
        plies_per_second: if elapsed_ms == 0 {
            0.0
        } else {
            total_plies as f64 / (elapsed_ms as f64 / 1000.0)
        },
        projected_500_games_ms: ms_per_game * 500.0,
        decisive_rate: ratio(decisive as f64, games),
        cap_rate: ratio(capped as f64, games),
        red_win_rate: ratio(red_wins as f64, games),
        blue_win_rate: ratio(blue_wins as f64, games),
        blue_decisive_share: ratio(blue_wins as f64, decisive),
        first_loss_red,
        first_loss_blue,
        first_loss_team_wins,
        first_loss_team_losses,
        first_loss_team_capped,
        first_loss_win_rate: ratio(
            first_loss_team_wins as f64,
            first_loss_team_wins + first_loss_team_losses,
        ),
        underdog_wins,
        underdog_win_rate: ratio(underdog_wins as f64, decisive),
        checks_created,
        check_rate_per_ply: ratio(checks_created as f64, total_plies),
        unstable_created,
        unstable_creation_rate_per_ply: ratio(unstable_created as f64, total_plies),
        rescue_opportunities,
        rescues,
        rescue_rate: ratio(rescues as f64, rescue_opportunities),
        material_losses,
        losses_by_piece,
        losses_by_owner,
        winner_piece_counts,
        repeated_piece_destinations,
        repeated_piece_destination_rate: ratio(repeated_piece_destinations as f64, total_plies),
        avg_loser_first_loss_ply: ratio(loser_first_loss_ply_total as f64, loser_first_loss_games),
        avg_winner_pieces: ratio(winner_piece_total as f64, decisive_piece_count_games),
        avg_loser_pieces: ratio(loser_piece_total as f64, decisive_piece_count_games),
    }
}

fn random_unit(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn no_move_loss(mut state: GameState) -> GameState {
    state.status = match state.current_player {
        Player::Red => GameStatus::BlueWon,
        Player::Blue => GameStatus::RedWon,
    };
    state.selected_piece_id = None;
    state.message = format!("{} has no legal move", player_key(state.current_player));
    let snapshot = state.snapshot();
    state.history.push(snapshot);
    state
}

fn random_play_turn(state: GameState, rng: &mut u64) -> GameState {
    if state.status != GameStatus::Playing {
        return state;
    }

    let field = evaluate_field(&state);
    let mut candidates = Vec::new();
    for piece in state
        .pieces
        .iter()
        .filter(|piece| piece.owner == state.current_player)
    {
        for destination in get_legal_moves(&piece.id, &state, &field) {
            candidates.push((piece.id.clone(), destination));
        }
    }
    if candidates.is_empty() {
        return no_move_loss(state);
    }

    let start = (random_unit(rng) as usize) % candidates.len();
    for offset in 0..candidates.len() {
        let (piece_id, destination) = &candidates[(start + offset) % candidates.len()];
        let result = apply_move(piece_id, *destination, state.clone(), false);
        if result.ok {
            return result.state;
        }
    }

    no_move_loss(state)
}

fn no_move_loss_training(mut state: GameState) -> GameState {
    state.status = match state.current_player {
        Player::Red => GameStatus::BlueWon,
        Player::Blue => GameStatus::RedWon,
    };
    state.selected_piece_id = None;
    state
}

fn random_play_training_turn(state: GameState, rng: &mut u64) -> GameState {
    if state.status != GameStatus::Playing {
        return state;
    }

    let field = evaluate_field(&state);
    let mut candidates = Vec::new();
    for piece in state
        .pieces
        .iter()
        .filter(|piece| piece.owner == state.current_player)
    {
        for destination in get_legal_moves(&piece.id, &state, &field) {
            candidates.push((piece.id.clone(), destination));
        }
    }
    if candidates.is_empty() {
        return no_move_loss_training(state);
    }

    let start = (random_unit(rng) as usize) % candidates.len();
    for offset in 0..candidates.len() {
        let (piece_id, destination) = &candidates[(start + offset) % candidates.len()];
        let result = apply_training_move(piece_id, *destination, state.clone());
        if result.ok {
            return result.state;
        }
    }

    no_move_loss_training(state)
}

fn random_play_turn_profile(
    state: GameState,
    rng: &mut u64,
    profile: &mut ProfileTotals,
) -> GameState {
    if state.status != GameStatus::Playing {
        return state;
    }

    let candidate_started_at = Instant::now();
    let field = evaluate_field(&state);
    let mut candidates = Vec::new();
    for piece in state
        .pieces
        .iter()
        .filter(|piece| piece.owner == state.current_player)
    {
        for destination in get_legal_moves(&piece.id, &state, &field) {
            candidates.push((piece.id.clone(), destination));
        }
    }
    profile.candidate_generation_ns += candidate_started_at.elapsed().as_nanos();
    profile.candidate_moves += candidates.len() as u64;
    profile.candidate_turns += 1;

    if candidates.is_empty() {
        return no_move_loss_training(state);
    }

    let start = (random_unit(rng) as usize) % candidates.len();
    for offset in 0..candidates.len() {
        let (piece_id, destination) = &candidates[(start + offset) % candidates.len()];
        let apply_started_at = Instant::now();
        let result = apply_training_move(piece_id, *destination, state.clone());
        profile.apply_move_ns += apply_started_at.elapsed().as_nanos();
        if result.ok {
            return result.state;
        }
    }

    no_move_loss_training(state)
}

fn summarize_lean(
    games: u64,
    red_wins: u64,
    blue_wins: u64,
    capped: u64,
    total_plies: u64,
    min_game_plies: u64,
    max_game_plies: u64,
    elapsed_ms: u128,
) -> LeanBatchSummary {
    let decisive = red_wins + blue_wins;
    let ms_per_game = ratio(elapsed_ms as f64, games);
    let ms_per_ply = ratio(elapsed_ms as f64, total_plies);
    LeanBatchSummary {
        games,
        red_wins,
        blue_wins,
        capped,
        decisive,
        total_plies,
        mean_plies: ratio(total_plies as f64, games),
        min_plies: if games == 0 { 0 } else { min_game_plies },
        max_plies: max_game_plies,
        elapsed_ms,
        ms_per_game,
        ms_per_ply,
        plies_per_second: if elapsed_ms == 0 {
            0.0
        } else {
            total_plies as f64 / (elapsed_ms as f64 / 1000.0)
        },
        projected_500_games_ms: ms_per_game * 500.0,
    }
}

pub fn simulate_random_lean_games(
    initial_state: &GameState,
    games: u64,
    max_plies: u64,
    seed: u32,
) -> LeanBatchSummary {
    let started_at = Instant::now();
    let mut red_wins = 0;
    let mut blue_wins = 0;
    let mut capped = 0;
    let mut total_plies = 0;
    let mut min_game_plies = u64::MAX;
    let mut max_game_plies = 0;

    for game in 0..games {
        let mut state = initial_state.clone();
        let mut rng = u64::from(seed).wrapping_add(game);
        let mut plies = 0;
        while state.status == GameStatus::Playing && plies < max_plies {
            state = random_play_training_turn(state, &mut rng);
            plies += 1;
        }

        total_plies += plies;
        min_game_plies = min_game_plies.min(plies);
        max_game_plies = max_game_plies.max(plies);
        match state.status {
            GameStatus::RedWon => red_wins += 1,
            GameStatus::BlueWon => blue_wins += 1,
            GameStatus::Playing => capped += 1,
        }
    }

    summarize_lean(
        games,
        red_wins,
        blue_wins,
        capped,
        total_plies,
        min_game_plies,
        max_game_plies,
        started_at.elapsed().as_millis(),
    )
}

pub fn generate_random_training_batch(
    initial_state: &GameState,
    games: u64,
    max_plies: u64,
    seed: u32,
    material_for_capped: bool,
) -> TrainingBatch {
    generate_random_training_batch_inner(
        initial_state,
        games,
        max_plies,
        seed,
        material_for_capped,
        None,
    )
}

fn generate_random_training_batch_inner(
    initial_state: &GameState,
    games: u64,
    max_plies: u64,
    seed: u32,
    material_for_capped: bool,
    mut profile: Option<&mut TrainingProfileTotals>,
) -> TrainingBatch {
    let started_at = Instant::now();
    let mut samples = Vec::new();
    let mut red_wins = 0;
    let mut blue_wins = 0;
    let mut capped = 0;
    let mut total_plies = 0;

    for game in 0..games {
        let mut state = initial_state.clone();
        let mut rng = u64::from(seed).wrapping_add(game);
        let game_sample_start = samples.len();
        let mut plies = 0;

        while state.status == GameStatus::Playing && plies < max_plies {
            let candidates = match profile.as_deref_mut() {
                Some(profile) => playable_training_candidates_profiled(&state, profile),
                None => playable_training_candidates(&state),
            };
            if candidates.is_empty() {
                state = no_move_loss_training(state);
                break;
            }

            let selected = (random_unit(&mut rng) as usize) % candidates.len();
            let sample = match profile.as_deref_mut() {
                Some(profile) => {
                    encode_training_sample_profiled(&state, &candidates, selected, profile)
                }
                None => encode_training_sample(&state, &candidates, selected),
            };
            samples.push(sample);
            let (piece_id, destination) = &candidates[selected];
            let result = apply_training_move(piece_id, *destination, state);
            state = if result.ok {
                result.state
            } else {
                no_move_loss_training(result.state)
            };
            plies += 1;
        }

        let value_started_at = Instant::now();
        for sample in &mut samples[game_sample_start..] {
            let player = if sample.player == "red" {
                Player::Red
            } else {
                Player::Blue
            };
            sample.value = result_value_for_state(&state, player, material_for_capped);
        }
        if let Some(profile) = profile.as_deref_mut() {
            profile.value_assignment_ns += value_started_at.elapsed().as_nanos();
        }

        total_plies += plies;
        match state.status {
            GameStatus::RedWon => red_wins += 1,
            GameStatus::BlueWon => blue_wins += 1,
            GameStatus::Playing => capped += 1,
        }
    }

    let elapsed_ms = started_at.elapsed().as_millis();
    let decisive = red_wins + blue_wins;
    let samples_per_second = if elapsed_ms == 0 {
        0.0
    } else {
        samples.len() as f64 / (elapsed_ms as f64 / 1000.0)
    };

    TrainingBatch {
        summary: TrainingBatchSummary {
            games,
            samples: samples.len(),
            red_wins,
            blue_wins,
            capped,
            decisive,
            total_plies,
            mean_plies: ratio(total_plies as f64, games),
            elapsed_ms,
            samples_per_second,
        },
        samples,
    }
}

pub fn profile_random_training_batch(
    initial_state: &GameState,
    games: u64,
    max_plies: u64,
    seed: u32,
    material_for_capped: bool,
) -> TrainingProfileSummary {
    let mut profile = TrainingProfileTotals::default();
    let batch = generate_random_training_batch_inner(
        initial_state,
        games,
        max_plies,
        seed,
        material_for_capped,
        Some(&mut profile),
    );
    let elapsed_ms = batch.summary.elapsed_ms as f64;
    let measured_ms = (profile.candidate_generation_ns
        + profile.sample_encoding_ns
        + profile.value_assignment_ns) as f64
        / 1_000_000.0;

    TrainingProfileSummary {
        legal_scan_ms: profile.legal_scan_ns as f64 / 1_000_000.0,
        candidate_apply_ms: profile.candidate_apply_ns as f64 / 1_000_000.0,
        candidate_generation_ms: profile.candidate_generation_ns as f64 / 1_000_000.0,
        sample_encoding_ms: profile.sample_encoding_ns as f64 / 1_000_000.0,
        value_assignment_ms: profile.value_assignment_ns as f64 / 1_000_000.0,
        unprofiled_ms: (elapsed_ms - measured_ms).max(0.0),
        avg_candidate_generation_ms_per_ply: ratio(
            profile.candidate_generation_ns as f64 / 1_000_000.0,
            profile.candidate_turns,
        ),
        avg_legal_scan_ms_per_ply: ratio(
            profile.legal_scan_ns as f64 / 1_000_000.0,
            profile.candidate_turns,
        ),
        avg_candidate_apply_ms_per_candidate: ratio(
            profile.candidate_apply_ns as f64 / 1_000_000.0,
            profile.attempted_candidates,
        ),
        avg_sample_encoding_ms_per_sample: ratio(
            profile.sample_encoding_ns as f64 / 1_000_000.0,
            profile.encoded_samples,
        ),
        avg_value_assignment_ms_per_game: ratio(
            profile.value_assignment_ns as f64 / 1_000_000.0,
            games,
        ),
        avg_candidates_per_ply: ratio(profile.total_candidates as f64, profile.candidate_turns),
        avg_attempted_candidates_per_ply: ratio(
            profile.attempted_candidates as f64,
            profile.candidate_turns,
        ),
        avg_legal_indexes_per_sample: ratio(
            profile.legal_action_indexes as f64,
            profile.encoded_samples,
        ),
        batch: batch.summary,
    }
}

pub fn profile_random_games(
    initial_state: &GameState,
    games: u64,
    max_plies: u64,
    seed: u32,
) -> RandomProfileSummary {
    let started_at = Instant::now();
    let mut red_wins = 0;
    let mut blue_wins = 0;
    let mut capped = 0;
    let mut total_plies = 0;
    let mut min_game_plies = u64::MAX;
    let mut max_game_plies = 0;
    let mut profile = ProfileTotals::default();

    for game in 0..games {
        let mut state = initial_state.clone();
        let mut rng = u64::from(seed).wrapping_add(game);
        let mut plies = 0;
        while state.status == GameStatus::Playing && plies < max_plies {
            state = random_play_turn_profile(state, &mut rng, &mut profile);
            plies += 1;
        }

        total_plies += plies;
        min_game_plies = min_game_plies.min(plies);
        max_game_plies = max_game_plies.max(plies);
        match state.status {
            GameStatus::RedWon => red_wins += 1,
            GameStatus::BlueWon => blue_wins += 1,
            GameStatus::Playing => capped += 1,
        }
    }

    let batch = summarize_lean(
        games,
        red_wins,
        blue_wins,
        capped,
        total_plies,
        min_game_plies,
        max_game_plies,
        started_at.elapsed().as_millis(),
    );
    RandomProfileSummary {
        candidate_generation_ms: profile.candidate_generation_ns as f64 / 1_000_000.0,
        apply_move_ms: profile.apply_move_ns as f64 / 1_000_000.0,
        avg_candidate_generation_ms_per_ply: ratio(
            profile.candidate_generation_ns as f64 / 1_000_000.0,
            profile.candidate_turns,
        ),
        avg_apply_move_ms_per_ply: ratio(
            profile.apply_move_ns as f64 / 1_000_000.0,
            profile.candidate_turns,
        ),
        avg_candidate_moves_per_ply: ratio(profile.candidate_moves as f64, profile.candidate_turns),
        batch,
    }
}

pub fn simulate_random_games(
    initial_state: &GameState,
    games: u64,
    max_plies: u64,
    seed: u32,
) -> AiBatchSummary {
    let started_at = Instant::now();
    let mut red_wins = 0;
    let mut blue_wins = 0;
    let mut capped = 0;
    let mut total_plies = 0;
    let mut min_game_plies = u64::MAX;
    let mut max_game_plies = 0;
    let mut first_loss_red = 0;
    let mut first_loss_blue = 0;
    let mut first_loss_team_wins = 0;
    let mut first_loss_team_losses = 0;
    let mut first_loss_team_capped = 0;
    let mut underdog_wins = 0;
    let mut checks_created = 0;
    let mut unstable_created = 0;
    let mut rescue_opportunities = 0;
    let mut rescues = 0;
    let mut material_losses = 0;
    let mut losses_by_piece = HashMap::new();
    let mut losses_by_owner = HashMap::new();
    let mut winner_piece_counts = HashMap::new();
    let mut repeated_piece_destinations = 0;
    let mut loser_first_loss_ply_total = 0;
    let mut loser_first_loss_games = 0;
    let mut winner_piece_total = 0;
    let mut loser_piece_total = 0;
    let mut decisive_piece_count_games = 0;

    for game in 0..games {
        let mut state = initial_state.clone();
        let mut rng = u64::from(seed).wrapping_add(game);
        let mut metrics = GameMetrics::default();
        let mut plies = 0;
        while state.status == GameStatus::Playing && plies < max_plies {
            let before = state.clone();
            state = random_play_turn(state, &mut rng);
            plies += 1;
            update_game_metrics(&before, &state, plies, &mut metrics);
        }

        total_plies += plies;
        min_game_plies = min_game_plies.min(plies);
        max_game_plies = max_game_plies.max(plies);
        match state.status {
            GameStatus::RedWon => red_wins += 1,
            GameStatus::BlueWon => blue_wins += 1,
            GameStatus::Playing => capped += 1,
        }

        if let Some(owner) = metrics.first_loss_owner {
            match owner {
                Player::Red => first_loss_red += 1,
                Player::Blue => first_loss_blue += 1,
            }
            match state.status {
                GameStatus::RedWon if owner == Player::Red => first_loss_team_wins += 1,
                GameStatus::BlueWon if owner == Player::Blue => first_loss_team_wins += 1,
                GameStatus::RedWon | GameStatus::BlueWon => first_loss_team_losses += 1,
                GameStatus::Playing => first_loss_team_capped += 1,
            }
            if matches!(
                (state.status, owner),
                (GameStatus::RedWon, Player::Blue) | (GameStatus::BlueWon, Player::Red)
            ) {
                loser_first_loss_ply_total += metrics.first_loss_ply.unwrap_or(0);
                loser_first_loss_games += 1;
            }
        }

        if state.status != GameStatus::Playing {
            let red_pieces = piece_count(&state, Player::Red);
            let blue_pieces = piece_count(&state, Player::Blue);
            let (winner_pieces, loser_pieces) = match state.status {
                GameStatus::RedWon => (red_pieces, blue_pieces),
                GameStatus::BlueWon => (blue_pieces, red_pieces),
                GameStatus::Playing => unreachable!(),
            };
            *winner_piece_counts
                .entry(winner_pieces.to_string())
                .or_insert(0) += 1;
            winner_piece_total += winner_pieces;
            loser_piece_total += loser_pieces;
            decisive_piece_count_games += 1;
            if winner_pieces < loser_pieces {
                underdog_wins += 1;
            }
        }

        checks_created += metrics.checks_created;
        unstable_created += metrics.unstable_created;
        rescue_opportunities += metrics.rescue_opportunities;
        rescues += metrics.rescues;
        material_losses += metrics.material_losses;
        repeated_piece_destinations += metrics.repeated_piece_destinations;
        for (piece_type, count) in metrics.losses_by_piece {
            *losses_by_piece.entry(piece_type).or_insert(0) += count;
        }
        for (owner, count) in metrics.losses_by_owner {
            *losses_by_owner.entry(owner).or_insert(0) += count;
        }
    }

    let elapsed_ms = started_at.elapsed().as_millis();
    let decisive = red_wins + blue_wins;
    let ms_per_game = ratio(elapsed_ms as f64, games);
    let ms_per_ply = ratio(elapsed_ms as f64, total_plies);

    AiBatchSummary {
        games,
        red_wins,
        blue_wins,
        capped,
        decisive,
        total_plies,
        mean_plies: ratio(total_plies as f64, games),
        min_plies: if games == 0 { 0 } else { min_game_plies },
        max_plies: max_game_plies,
        elapsed_ms,
        ms_per_game,
        ms_per_ply,
        plies_per_second: if elapsed_ms == 0 {
            0.0
        } else {
            total_plies as f64 / (elapsed_ms as f64 / 1000.0)
        },
        projected_500_games_ms: ms_per_game * 500.0,
        decisive_rate: ratio(decisive as f64, games),
        cap_rate: ratio(capped as f64, games),
        red_win_rate: ratio(red_wins as f64, games),
        blue_win_rate: ratio(blue_wins as f64, games),
        blue_decisive_share: ratio(blue_wins as f64, decisive),
        first_loss_red,
        first_loss_blue,
        first_loss_team_wins,
        first_loss_team_losses,
        first_loss_team_capped,
        first_loss_win_rate: ratio(
            first_loss_team_wins as f64,
            first_loss_team_wins + first_loss_team_losses,
        ),
        underdog_wins,
        underdog_win_rate: ratio(underdog_wins as f64, decisive),
        checks_created,
        check_rate_per_ply: ratio(checks_created as f64, total_plies),
        unstable_created,
        unstable_creation_rate_per_ply: ratio(unstable_created as f64, total_plies),
        rescue_opportunities,
        rescues,
        rescue_rate: ratio(rescues as f64, rescue_opportunities),
        material_losses,
        losses_by_piece,
        losses_by_owner,
        winner_piece_counts,
        repeated_piece_destinations,
        repeated_piece_destination_rate: ratio(repeated_piece_destinations as f64, total_plies),
        avg_loser_first_loss_ply: ratio(loser_first_loss_ply_total as f64, loser_first_loss_games),
        avg_winner_pieces: ratio(winner_piece_total as f64, decisive_piece_count_games),
        avg_loser_pieces: ratio(loser_piece_total as f64, decisive_piece_count_games),
    }
}

fn ratio(numerator: f64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator / denominator as f64
    }
}
