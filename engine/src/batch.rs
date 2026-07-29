use crate::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

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

fn ratio(numerator: f64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator / denominator as f64
    }
}
