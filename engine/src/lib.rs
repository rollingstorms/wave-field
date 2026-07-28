use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const BOARD_SIZE: i32 = 7;
pub const FIELD_EPSILON: f64 = 1e-9;
pub type Field = Vec<Vec<f64>>;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Player {
    Red,
    Blue,
}

impl Player {
    fn opponent(self) -> Self {
        match self {
            Self::Red => Self::Blue,
            Self::Blue => Self::Red,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Red => "Red",
            Self::Blue => "Blue",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PieceType {
    Pawn,
    Rook,
    Spy,
    King,
}

const PIECE_TYPES: [PieceType; 4] = [
    PieceType::Pawn,
    PieceType::Rook,
    PieceType::Spy,
    PieceType::King,
];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Piece {
    pub id: String,
    pub owner: Player,
    #[serde(rename = "type")]
    pub piece_type: PieceType,
    pub position: Position,
    pub unstable: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PlayerComponents {
    pub pawn: Vec<i8>,
    pub rook: Vec<i8>,
    pub spy: Vec<i8>,
    pub king: Vec<i8>,
}

impl PlayerComponents {
    fn get(&self, piece_type: PieceType) -> &Vec<i8> {
        match piece_type {
            PieceType::Pawn => &self.pawn,
            PieceType::Rook => &self.rook,
            PieceType::Spy => &self.spy,
            PieceType::King => &self.king,
        }
    }

    fn get_mut(&mut self, piece_type: PieceType) -> &mut Vec<i8> {
        match piece_type {
            PieceType::Pawn => &mut self.pawn,
            PieceType::Rook => &mut self.rook,
            PieceType::Spy => &mut self.spy,
            PieceType::King => &mut self.king,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PlayerMap<T> {
    pub red: T,
    pub blue: T,
}

impl<T> PlayerMap<T> {
    fn get(&self, player: Player) -> &T {
        match player {
            Player::Red => &self.red,
            Player::Blue => &self.blue,
        }
    }

    fn get_mut(&mut self, player: Player) -> &mut T {
        match player {
            Player::Red => &mut self.red,
            Player::Blue => &mut self.blue,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PlayerActivationOrder {
    pub pawn: Vec<usize>,
    pub rook: Vec<usize>,
    pub spy: Vec<usize>,
    pub king: Vec<usize>,
}

impl PlayerActivationOrder {
    fn get(&self, piece_type: PieceType) -> &Vec<usize> {
        match piece_type {
            PieceType::Pawn => &self.pawn,
            PieceType::Rook => &self.rook,
            PieceType::Spy => &self.spy,
            PieceType::King => &self.king,
        }
    }

    fn get_mut(&mut self, piece_type: PieceType) -> &mut Vec<usize> {
        match piece_type {
            PieceType::Pawn => &mut self.pawn,
            PieceType::Rook => &mut self.rook,
            PieceType::Spy => &mut self.spy,
            PieceType::King => &mut self.king,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FormulaPreset {
    Checkerboard,
    DiagonalStripes,
    HorizontalVersusVertical,
    Quadrants,
    ConstantBasin,
    SkippedRings,
    CompassRose,
    AxisFavor,
    DiagonalFavor,
    WideBullseye,
    PulseGap,
    BlockChecker,
    DiamondCore,
    Astigmatism,
    LocalFlip,
    AdjacentOpinion,
    Sink,
    DeepSink,
    FarCrown,
    SlowGovernance,
    DipoleX,
    DipoleY,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum BasisDefinition {
    Ring {
        name: String,
        geometry: String,
        #[serde(rename = "ringValues")]
        ring_values: Vec<i8>,
        repeat: bool,
        #[serde(rename = "decayBase")]
        decay_base: f64,
        #[serde(rename = "originScale")]
        origin_scale: f64,
    },
    Preset {
        name: String,
        preset: FormulaPreset,
        #[serde(rename = "decayBase")]
        decay_base: f64,
        #[serde(rename = "originScale")]
        origin_scale: f64,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ComponentDefinitions {
    pub pawn: Vec<BasisDefinition>,
    pub rook: Vec<BasisDefinition>,
    pub spy: Vec<BasisDefinition>,
    pub king: Vec<BasisDefinition>,
}

impl ComponentDefinitions {
    fn get(&self, piece_type: PieceType) -> &Vec<BasisDefinition> {
        match piece_type {
            PieceType::Pawn => &self.pawn,
            PieceType::Rook => &self.rook,
            PieceType::Spy => &self.spy,
            PieceType::King => &self.king,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct WaveScale {
    pub friendly: f64,
    pub hostile: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PieceTypeMap<T> {
    pub pawn: T,
    pub rook: T,
    pub spy: T,
    pub king: T,
}

impl<T> PieceTypeMap<T> {
    fn get(&self, piece_type: PieceType) -> &T {
        match piece_type {
            PieceType::Pawn => &self.pawn,
            PieceType::Rook => &self.rook,
            PieceType::Spy => &self.spy,
            PieceType::King => &self.king,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GameStatus {
    Playing,
    RedWon,
    BlueWon,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GameSnapshot {
    pub pieces: Vec<Piece>,
    pub current_player: Player,
    pub components: PlayerMap<PlayerComponents>,
    pub activation_orders: PlayerMap<PlayerActivationOrder>,
    pub status: GameStatus,
    pub selected_piece_id: Option<String>,
    pub turn_number: u32,
    pub definitions: ComponentDefinitions,
    pub wave_scales: PieceTypeMap<WaveScale>,
    pub home_energy: PieceTypeMap<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GameState {
    pub pieces: Vec<Piece>,
    pub current_player: Player,
    pub components: PlayerMap<PlayerComponents>,
    pub activation_orders: PlayerMap<PlayerActivationOrder>,
    pub default_components: PlayerComponents,
    pub status: GameStatus,
    pub selected_piece_id: Option<String>,
    pub turn_number: u32,
    pub definitions: ComponentDefinitions,
    pub wave_scales: PieceTypeMap<WaveScale>,
    pub home_energy: PieceTypeMap<f64>,
    pub history: Vec<GameSnapshot>,
    pub message: String,
}

impl GameState {
    fn snapshot(&self) -> GameSnapshot {
        GameSnapshot {
            pieces: self.pieces.clone(),
            current_player: self.current_player,
            components: self.components.clone(),
            activation_orders: self.activation_orders.clone(),
            status: self.status,
            selected_piece_id: self.selected_piece_id.clone(),
            turn_number: self.turn_number,
            definitions: self.definitions.clone(),
            wave_scales: self.wave_scales.clone(),
            home_energy: self.home_energy.clone(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MoveResult {
    pub ok: bool,
    pub state: GameState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

fn piece_strength(piece_type: PieceType) -> f64 {
    match piece_type {
        PieceType::Pawn => 1.0,
        PieceType::Rook | PieceType::Spy | PieceType::King => 2.0,
    }
}

fn tuning_strength(piece_type: PieceType) -> usize {
    match piece_type {
        PieceType::Pawn | PieceType::Spy => 1,
        PieceType::Rook | PieceType::King => 2,
    }
}

fn preset_sign(preset: &FormulaPreset, delta: Position, ring: i32) -> i8 {
    let x = delta.x.abs();
    let y = delta.y.abs();
    match preset {
        FormulaPreset::Checkerboard => {
            if (x + y) % 2 == 0 {
                1
            } else {
                -1
            }
        }
        FormulaPreset::DiagonalStripes => {
            if ((delta.x - delta.y).abs() / 2) % 2 == 0 {
                1
            } else {
                -1
            }
        }
        FormulaPreset::HorizontalVersusVertical => {
            if x >= y {
                1
            } else {
                -1
            }
        }
        FormulaPreset::Quadrants => {
            if delta.x * delta.y >= 0 {
                1
            } else {
                -1
            }
        }
        FormulaPreset::ConstantBasin => 1,
        FormulaPreset::SkippedRings => {
            if ring % 6 == 0 {
                1
            } else if ring % 6 == 3 {
                -1
            } else {
                0
            }
        }
        FormulaPreset::CompassRose => {
            if delta.x == 0 || delta.y == 0 || x == y {
                1
            } else {
                -1
            }
        }
        FormulaPreset::AxisFavor => {
            if delta.x == 0 || delta.y == 0 {
                1
            } else if x == y {
                -1
            } else {
                0
            }
        }
        FormulaPreset::DiagonalFavor => {
            if x == y {
                1
            } else if delta.x == 0 || delta.y == 0 {
                -1
            } else {
                0
            }
        }
        FormulaPreset::WideBullseye => {
            if (ring / 2) % 2 == 0 {
                1
            } else {
                -1
            }
        }
        FormulaPreset::PulseGap => {
            if ring % 4 == 0 {
                1
            } else if ring % 4 == 2 {
                -1
            } else {
                0
            }
        }
        FormulaPreset::BlockChecker => {
            if ((x / 2) + (y / 2)) % 2 == 0 {
                1
            } else {
                -1
            }
        }
        FormulaPreset::DiamondCore => {
            if x + y <= 2 {
                1
            } else {
                -1
            }
        }
        FormulaPreset::Astigmatism => {
            if x == y {
                0
            } else if x > y {
                1
            } else {
                -1
            }
        }
        FormulaPreset::LocalFlip => {
            if ring <= 1 {
                1
            } else {
                0
            }
        }
        FormulaPreset::AdjacentOpinion => {
            if ring == 1 {
                1
            } else {
                0
            }
        }
        FormulaPreset::Sink => {
            if ring == 0 {
                1
            } else {
                -1
            }
        }
        FormulaPreset::DeepSink => {
            if ring <= 1 {
                1
            } else {
                -1
            }
        }
        FormulaPreset::FarCrown => {
            if ring >= 3 {
                1
            } else {
                0
            }
        }
        FormulaPreset::SlowGovernance => {
            if ring <= 2 {
                1
            } else {
                -1
            }
        }
        FormulaPreset::DipoleX => {
            if delta.x == 0 {
                0
            } else if delta.x > 0 {
                1
            } else {
                -1
            }
        }
        FormulaPreset::DipoleY => {
            if delta.y == 0 {
                0
            } else if delta.y > 0 {
                1
            } else {
                -1
            }
        }
    }
}

pub fn evaluate_basis(definition: &BasisDefinition, delta: Position) -> f64 {
    let ring = delta.x.abs().max(delta.y.abs());
    let (sign, decay_base, origin_scale) = match definition {
        BasisDefinition::Preset {
            preset,
            decay_base,
            origin_scale,
            ..
        } => (preset_sign(preset, delta, ring), *decay_base, *origin_scale),
        BasisDefinition::Ring {
            ring_values,
            repeat,
            decay_base,
            origin_scale,
            ..
        } => {
            let index = if *repeat {
                ring as usize % ring_values.len()
            } else {
                ring as usize
            };
            (
                *ring_values.get(index).unwrap_or(&0),
                *decay_base,
                *origin_scale,
            )
        }
    };
    let multiplier = decay_base.powi(-ring) * if ring == 0 { origin_scale } else { 1.0 };
    f64::from(sign) * multiplier
}

pub fn evaluate_piece_contribution(piece: &Piece, square: Position, state: &GameState) -> f64 {
    let delta = Position {
        x: square.x - piece.position.x,
        y: square.y - piece.position.y,
    };
    if delta.x == 0 && delta.y == 0 {
        return *state.home_energy.get(piece.piece_type);
    }
    let coefficients = state.components.get(piece.owner).get(piece.piece_type);
    let definitions = state.definitions.get(piece.piece_type);
    let contribution = coefficients
        .iter()
        .enumerate()
        .map(|(index, coefficient)| {
            let value = f64::from(*coefficient) * evaluate_basis(&definitions[index], delta);
            let scale = state.wave_scales.get(piece.piece_type);
            value
                * if value >= 0.0 {
                    scale.friendly
                } else {
                    scale.hostile
                }
        })
        .sum::<f64>();
    piece_strength(piece.piece_type) * contribution
}

pub fn evaluate_field(state: &GameState) -> Field {
    (0..BOARD_SIZE)
        .map(|y| {
            (0..BOARD_SIZE)
                .map(|x| {
                    state
                        .pieces
                        .iter()
                        .map(|piece| {
                            let sign = if piece.owner == Player::Red {
                                1.0
                            } else {
                                -1.0
                            };
                            sign * evaluate_piece_contribution(piece, Position { x, y }, state)
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}

fn compatible(player: Player, value: f64) -> bool {
    match player {
        Player::Red => value >= -FIELD_EPSILON,
        Player::Blue => value <= FIELD_EPSILON,
    }
}

fn in_bounds(position: Position) -> bool {
    position.x >= 0 && position.x < BOARD_SIZE && position.y >= 0 && position.y < BOARD_SIZE
}

fn occupied(state: &GameState, position: Position) -> bool {
    state.pieces.iter().any(|piece| piece.position == position)
}

pub fn get_legal_moves(piece_id: &str, state: &GameState, field: &Field) -> Vec<Position> {
    let Some(piece) = state.pieces.iter().find(|piece| piece.id == piece_id) else {
        return Vec::new();
    };
    let mut moves = Vec::new();
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let mut destination = Position {
                x: piece.position.x + dx,
                y: piece.position.y + dy,
            };
            while in_bounds(destination) {
                let passable = piece.piece_type == PieceType::Spy
                    || compatible(
                        piece.owner,
                        field[destination.y as usize][destination.x as usize],
                    );
                if occupied(state, destination) || !passable {
                    break;
                }
                moves.push(destination);
                destination.x += dx;
                destination.y += dy;
            }
        }
    }
    moves
}

pub fn unstable_pieces(player: Player, state: &GameState, field: &Field) -> Vec<Piece> {
    state
        .pieces
        .iter()
        .filter(|piece| {
            piece.owner == player
                && !compatible(
                    player,
                    field[piece.position.y as usize][piece.position.x as usize],
                )
        })
        .cloned()
        .collect()
}

pub fn mark_instability(mut state: GameState, field: &Field) -> GameState {
    for piece in &mut state.pieces {
        piece.unstable = !compatible(
            piece.owner,
            field[piece.position.y as usize][piece.position.x as usize],
        );
    }
    state
}

pub fn is_king_unprotected(player: Player, state: &GameState, field: &Field) -> bool {
    state
        .pieces
        .iter()
        .find(|piece| piece.owner == player && piece.piece_type == PieceType::King)
        .is_some_and(|king| {
            !compatible(
                player,
                field[king.position.y as usize][king.position.x as usize],
            )
        })
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

fn resolve_own_turn_consequences(
    player: Player,
    previous: &GameState,
    candidate: GameState,
) -> GameState {
    let previous_field = evaluate_field(previous);
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
            if values.iter().filter(|value| **value != 0).count() == tuning_strength(piece_type) {
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
    if state.status != GameStatus::Playing {
        return state;
    }
    let field = evaluate_field(&state);
    let mut resolved = mark_instability(state, &field);
    let resolved_field = evaluate_field(&resolved);
    if is_king_unprotected(resolved.current_player, &resolved, &resolved_field) {
        if !analyze_checkmate {
            resolved.message = format!("{} king is in check", resolved.current_player.name());
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
                "{} king is in check · {hint}",
                resolved.current_player.name()
            );
            return resolved;
        }
        resolved.status = win_status(resolved.current_player.opponent());
        resolved.selected_piece_id = None;
        resolved.message = format!(
            "{} king is in check · no legal rescue found",
            resolved.current_player.name()
        );
        return resolved;
    }
    let unstable = unstable_pieces(resolved.current_player, &resolved, &resolved_field)
        .into_iter()
        .find(|piece| piece.piece_type != PieceType::King);
    resolved.message = match unstable {
        Some(piece) => format!(
            "{} must rescue an unstable {}",
            resolved.current_player.name(),
            piece_type_name(piece.piece_type)
        ),
        None => format!("{} to move", resolved.current_player.name()),
    };
    resolved
}

fn piece_type_name(piece_type: PieceType) -> &'static str {
    match piece_type {
        PieceType::Pawn => "pawn",
        PieceType::Rook => "rook",
        PieceType::Spy => "spy",
        PieceType::King => "king",
    }
}

pub fn apply_move(
    piece_id: &str,
    destination: Position,
    state: GameState,
    analyze_checkmate: bool,
) -> MoveResult {
    if state.status != GameStatus::Playing {
        return rejected(state, "The game is over.");
    }
    let field = evaluate_field(&state);
    let Some(piece) = state.pieces.iter().find(|piece| piece.id == piece_id) else {
        return rejected(state, "Choose one of your pieces.");
    };
    if piece.owner != state.current_player {
        return rejected(state, "Choose one of your pieces.");
    }
    if !get_legal_moves(piece_id, &state, &field).contains(&destination) {
        return rejected(state, "That square is not a legal move.");
    }

    let previous = state;
    let candidate = move_piece(previous.clone(), piece_id, destination);
    let mut resolved = resolve_own_turn_consequences(previous.current_player, &previous, candidate);
    let resolved_field = evaluate_field(&resolved);
    if is_king_unprotected(previous.current_player, &resolved, &resolved_field) {
        return rejected(previous, "That move would leave your king unprotected.");
    }
    let remaining = resolved
        .pieces
        .iter()
        .map(|piece| piece.id.as_str())
        .collect::<HashSet<_>>();
    let losses = previous
        .pieces
        .iter()
        .filter(|piece| {
            piece.owner == previous.current_player
                && piece.piece_type != PieceType::King
                && !remaining.contains(piece.id.as_str())
        })
        .map(|piece| piece_type_name(piece.piece_type))
        .collect::<Vec<_>>();
    resolved.current_player = previous.current_player.opponent();
    if previous.current_player == Player::Red {
        resolved.turn_number += 1;
    }
    resolved.selected_piece_id = None;
    resolved.history.push(previous.snapshot());
    let mut next = begin_turn(resolved, analyze_checkmate);
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

pub fn get_playable_moves(piece_id: &str, state: &GameState) -> Vec<Position> {
    let field = evaluate_field(state);
    get_legal_moves(piece_id, state, &field)
        .into_iter()
        .filter(|destination| apply_move(piece_id, *destination, state.clone(), true).ok)
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
        return rejected(state, "You can resign only while your king is in check.");
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
            "Hints are available only while your king is in check.",
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
        format!("{} reset tuning · king remains in check", player.name())
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
            "{} randomized tuning · king remains in check",
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
        && active_indices.len() >= tuning_strength(piece_type)
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
            "{} king is in check · move to rescue the king",
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

#[cfg(feature = "wasm")]
fn parse_state(json: &str) -> GameState {
    serde_json::from_str(json).expect("valid game state JSON")
}

#[cfg(feature = "wasm")]
fn json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("serializable engine result")
}

#[cfg(feature = "wasm")]
mod wasm {
    use super::*;
    use wasm_bindgen::prelude::*;

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
        let piece_type =
            serde_json::from_str(&format!("\"{piece_type}\"")).expect("valid piece type");
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> GameState {
        serde_json::from_str(include_str!("../tests/initial-state.json")).unwrap()
    }

    #[test]
    fn initial_field_is_rotationally_antisymmetric() {
        let field = evaluate_field(&fixture());
        for y in 0..7 {
            for x in 0..7 {
                assert!((field[y][x] + field[6 - y][6 - x]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn initial_moves_match_golden_fixture() {
        let state = fixture();
        let field = evaluate_field(&state);
        let actual = get_legal_moves("blue-spy-1", &state, &field);
        let expected: Vec<Position> =
            serde_json::from_str(include_str!("../tests/initial-blue-spy-moves.json")).unwrap();
        assert_eq!(actual, expected);
    }
}
