use serde::{Deserialize, Serialize};

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
    pub(crate) fn opponent(self) -> Self {
        match self {
            Self::Red => Self::Blue,
            Self::Blue => Self::Red,
        }
    }

    pub(crate) fn name(self) -> &'static str {
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

pub(crate) const PIECE_TYPES: [PieceType; 4] = [
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
    pub(crate) fn get(&self, piece_type: PieceType) -> &Vec<i8> {
        match piece_type {
            PieceType::Pawn => &self.pawn,
            PieceType::Rook => &self.rook,
            PieceType::Spy => &self.spy,
            PieceType::King => &self.king,
        }
    }

    pub(crate) fn get_mut(&mut self, piece_type: PieceType) -> &mut Vec<i8> {
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
    pub(crate) fn get(&self, player: Player) -> &T {
        match player {
            Player::Red => &self.red,
            Player::Blue => &self.blue,
        }
    }

    pub(crate) fn get_mut(&mut self, player: Player) -> &mut T {
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
    pub(crate) fn get(&self, piece_type: PieceType) -> &Vec<usize> {
        match piece_type {
            PieceType::Pawn => &self.pawn,
            PieceType::Rook => &self.rook,
            PieceType::Spy => &self.spy,
            PieceType::King => &self.king,
        }
    }

    pub(crate) fn get_mut(&mut self, piece_type: PieceType) -> &mut Vec<usize> {
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
    pub(crate) fn get(&self, piece_type: PieceType) -> &Vec<BasisDefinition> {
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
    pub(crate) fn get(&self, piece_type: PieceType) -> &T {
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
    pub(crate) fn snapshot(&self) -> GameSnapshot {
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
