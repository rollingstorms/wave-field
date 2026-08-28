use crate::model::*;
use serde::Serialize;

fn piece_strength(piece_type: PieceType) -> f64 {
    match piece_type {
        PieceType::Pawn => 1.0,
        PieceType::Rook | PieceType::Spy | PieceType::King => 2.0,
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
    if let BasisDefinition::Combo { components, .. } = definition {
        return components
            .iter()
            .map(|component| component.weight * evaluate_basis(&component.definition, delta))
            .sum();
    }
    let (sign, decay_base, origin_scale) = match definition {
        BasisDefinition::Preset {
            preset,
            decay_base,
            origin_scale,
            ..
        } => (
            f64::from(preset_sign(preset, delta, ring)),
            *decay_base,
            *origin_scale,
        ),
        BasisDefinition::Combo { .. } => unreachable!(),
        BasisDefinition::Grid {
            grid_values,
            decay_base,
            origin_scale,
            ..
        } => {
            let y = (delta.y + 3) as usize;
            let x = (delta.x + 3) as usize;
            let value = grid_values
                .get(y)
                .and_then(|row| row.get(x))
                .copied()
                .unwrap_or(0);
            (f64::from(value), *decay_base, *origin_scale)
        }
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
                f64::from(*ring_values.get(index).unwrap_or(&0)),
                *decay_base,
                *origin_scale,
            )
        }
    };
    let inverse_decay = 1.0 / decay_base;
    let multiplier = match ring {
        0 => origin_scale,
        1 => inverse_decay,
        2 => inverse_decay * inverse_decay,
        3 => inverse_decay * inverse_decay * inverse_decay,
        4 => inverse_decay * inverse_decay * inverse_decay * inverse_decay,
        5 => inverse_decay * inverse_decay * inverse_decay * inverse_decay * inverse_decay,
        _ => {
            inverse_decay
                * inverse_decay
                * inverse_decay
                * inverse_decay
                * inverse_decay
                * inverse_decay
        }
    };
    sign * multiplier
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InfluenceContributor {
    #[serde(rename = "pieceID")]
    pub piece_id: String,
    pub owner: Player,
    pub kind: PieceType,
    pub position: Position,
    pub value: f64,
    pub magnitude: f64,
    pub share_of_total_magnitude: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SquareInfluenceContributors {
    pub position: Position,
    pub total: f64,
    pub contributors: Vec<InfluenceContributor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub highest_negative_contributor: Option<InfluenceContributor>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstabilityInfluenceLink {
    pub target: Position,
    #[serde(rename = "targetPieceID")]
    pub target_piece_id: String,
    pub target_owner: Player,
    pub target_kind: PieceType,
    pub contributor: InfluenceContributor,
}

fn signed_piece_contribution(piece: &Piece, square: Position, state: &GameState) -> f64 {
    let sign = if piece.owner == Player::Red {
        1.0
    } else {
        -1.0
    };
    sign * evaluate_piece_contribution(piece, square, state)
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
    let mut positive_raw = 0.0;
    let mut negative_raw = 0.0;
    for (index, coefficient) in coefficients.iter().enumerate() {
        let value = f64::from(*coefficient) * evaluate_basis(&definitions[index], delta);
        if value > 0.0 {
            positive_raw += value;
        } else if value < 0.0 {
            negative_raw += value;
        }
    }
    let scale = state.wave_scales.get(piece.piece_type);
    piece_strength(piece.piece_type)
        * (positive_raw * scale.friendly + negative_raw * scale.hostile)
}

pub fn influence_contributors_at(
    position: Position,
    state: &GameState,
) -> SquareInfluenceContributors {
    let raw = state
        .pieces
        .iter()
        .map(|piece| (piece, signed_piece_contribution(piece, position, state)))
        .filter(|(_, value)| value.abs() > FIELD_EPSILON)
        .collect::<Vec<_>>();
    let total = raw.iter().map(|(_, value)| *value).sum::<f64>();
    let total_magnitude = raw.iter().map(|(_, value)| value.abs()).sum::<f64>();
    let contributors = raw
        .into_iter()
        .map(|(piece, value)| {
            let magnitude = value.abs();
            InfluenceContributor {
                piece_id: piece.id.clone(),
                owner: piece.owner,
                kind: piece.piece_type,
                position: piece.position,
                value,
                magnitude,
                share_of_total_magnitude: if total_magnitude > FIELD_EPSILON {
                    magnitude / total_magnitude
                } else {
                    0.0
                },
            }
        })
        .collect::<Vec<_>>();
    let highest_negative_contributor = contributors
        .iter()
        .filter(|contributor| contributor.value < -FIELD_EPSILON)
        .min_by(|left, right| {
            left.value
                .total_cmp(&right.value)
                .then_with(|| left.piece_id.cmp(&right.piece_id))
        })
        .cloned();

    SquareInfluenceContributors {
        position,
        total,
        contributors,
        highest_negative_contributor,
    }
}

pub fn all_influence_contributors(state: &GameState) -> Vec<Vec<SquareInfluenceContributors>> {
    (0..BOARD_SIZE)
        .map(|y| {
            (0..BOARD_SIZE)
                .map(|x| influence_contributors_at(Position { x, y }, state))
                .collect()
        })
        .collect()
}

pub fn instability_influence_links(
    threshold: f64,
    state: &GameState,
) -> Vec<InstabilityInfluenceLink> {
    let threshold = threshold.max(0.0);
    state
        .pieces
        .iter()
        .filter_map(|target_piece| {
            let square = influence_contributors_at(target_piece.position, state);
            let hostile = match target_piece.owner {
                Player::Red => square.total < -FIELD_EPSILON,
                Player::Blue => square.total > FIELD_EPSILON,
            };
            hostile.then_some((target_piece, square))
        })
        .flat_map(|(target_piece, square)| {
            square
                .contributors
                .into_iter()
                .filter(move |contributor| {
                    let hostile_contribution = match target_piece.owner {
                        Player::Red => contributor.value < -FIELD_EPSILON,
                        Player::Blue => contributor.value > FIELD_EPSILON,
                    };
                    hostile_contribution && contributor.share_of_total_magnitude >= threshold
                })
                .map(move |contributor| InstabilityInfluenceLink {
                    target: target_piece.position,
                    target_piece_id: target_piece.id.clone(),
                    target_owner: target_piece.owner,
                    target_kind: target_piece.piece_type,
                    contributor,
                })
        })
        .collect()
}

pub fn evaluate_field(state: &GameState) -> Field {
    let mut field = [[0.0; BOARD_LEN]; BOARD_LEN];
    for piece in &state.pieces {
        let sign = if piece.owner == Player::Red {
            1.0
        } else {
            -1.0
        };
        let strength = piece_strength(piece.piece_type);
        let scale = state.wave_scales.get(piece.piece_type);
        let coefficients = state.components.get(piece.owner).get(piece.piece_type);
        let definitions = state.definitions.get(piece.piece_type);
        let active_components = coefficients
            .iter()
            .enumerate()
            .filter_map(|(index, coefficient)| {
                (*coefficient != 0).then_some((*coefficient, &definitions[index]))
            })
            .collect::<Vec<_>>();

        for y in 0..BOARD_SIZE {
            for x in 0..BOARD_SIZE {
                let delta = Position {
                    x: x - piece.position.x,
                    y: y - piece.position.y,
                };
                let contribution = if delta.x == 0 && delta.y == 0 {
                    *state.home_energy.get(piece.piece_type)
                } else {
                    let mut positive_raw = 0.0;
                    let mut negative_raw = 0.0;
                    for (coefficient, definition) in &active_components {
                        let value = f64::from(*coefficient) * evaluate_basis(definition, delta);
                        if value > 0.0 {
                            positive_raw += value;
                        } else if value < 0.0 {
                            negative_raw += value;
                        }
                    }
                    strength * (positive_raw * scale.friendly + negative_raw * scale.hostile)
                };
                field[y as usize][x as usize] += sign * contribution;
            }
        }
    }
    field
}
