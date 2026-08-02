use crate::model::*;

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
        } => (preset_sign(preset, delta, ring), *decay_base, *origin_scale),
        BasisDefinition::Combo { .. } => unreachable!(),
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
