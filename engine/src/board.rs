use crate::model::*;

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
