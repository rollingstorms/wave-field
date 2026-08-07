use crate::*;

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

#[test]
fn training_move_matches_normal_move_without_ui_fields() {
    let state = fixture();
    let field = evaluate_field(&state);
    let destination = get_legal_moves("blue-spy-1", &state, &field)
        .into_iter()
        .next()
        .unwrap();

    let normal = apply_move("blue-spy-1", destination, state.clone(), false);
    let training = apply_training_move("blue-spy-1", destination, state);
    assert!(normal.ok);
    assert!(training.ok);

    let mut normal_state = normal.state;
    let mut training_state = training.state;
    normal_state.history.clear();
    training_state.history.clear();
    normal_state.message.clear();
    training_state.message.clear();

    assert_eq!(normal_state, training_state);
}

#[test]
fn training_safety_matches_normal_move_acceptance_for_initial_moves() {
    let state = fixture();
    let field = evaluate_field(&state);
    for piece in state
        .pieces
        .iter()
        .filter(|piece| piece.owner == state.current_player)
    {
        for destination in get_legal_moves(&piece.id, &state, &field) {
            let normal = apply_move(&piece.id, destination, state.clone(), false);
            let safe = training_move_is_safe_with_field(&piece.id, destination, &state, &field);
            assert_eq!(
                normal.ok, safe,
                "{} to ({}, {})",
                piece.id, destination.x, destination.y
            );
        }
    }
}

#[test]
fn easy_ai_turn_moves_from_initial_state() {
    let state = fixture();
    let next = play_easy_turn(
        state,
        Player::Blue,
        AiTurnOptions {
            seed: Some(7),
            variety: Some(0.0),
            time_budget_ms: Some(10),
        },
    );

    assert_eq!(next.status, GameStatus::Playing);
    assert_eq!(next.current_player, Player::Red);
    assert_eq!(next.history.len(), 1);
}

#[test]
fn begin_turn_declares_loss_when_player_has_no_legal_move() {
    let mut state = fixture();
    state.current_player = Player::Blue;
    state.pieces = (0..49)
        .map(|index| Piece {
            id: if index == 24 {
                "blue-king".to_owned()
            } else {
                format!("blue-pawn-{index}")
            },
            owner: Player::Blue,
            piece_type: if index == 24 {
                PieceType::King
            } else {
                PieceType::Pawn
            },
            position: Position {
                x: index % 7,
                y: index / 7,
            },
            unstable: false,
        })
        .collect();

    let started = begin_turn(state, true);

    assert_eq!(started.status, GameStatus::RedWon);
    assert!(started.message.contains("Blue has no legal move"));
}
