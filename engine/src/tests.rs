use crate::*;

fn fixture() -> GameState {
    serde_json::from_str(include_str!("../tests/initial-state.json")).unwrap()
}

#[test]
fn initial_state_uses_inverted_hat_tower_grid() {
    let state = fixture();
    let BasisDefinition::Grid {
        name, grid_values, ..
    } = &state.definitions.rook[0]
    else {
        panic!("expected tower C1 to use a grid basis");
    };

    assert_eq!(name, "Tower grid");
    assert_eq!(grid_values[4][6], 0);
    assert_eq!(
        evaluate_basis(&state.definitions.rook[0], Position { x: 3, y: 1 }),
        0.0
    );
    assert!(evaluate_basis(&state.definitions.rook[0], Position { x: -3, y: -1 }) > 0.0);

    assert_ne!(grid_values[4][6], grid_values[2][0]);
}

#[test]
fn influence_contributors_sum_to_field_with_magnitude_shares() {
    let state = fixture();
    let position = Position { x: 3, y: 4 };
    let square = influence_contributors_at(position, &state);
    let field = evaluate_field(&state);
    let total_magnitude = square
        .contributors
        .iter()
        .map(|contributor| contributor.magnitude)
        .sum::<f64>();

    assert!((square.total - field[position.y as usize][position.x as usize]).abs() < 1e-9);
    assert!(square.contributors.iter().all(|contributor| {
        contributor.value.abs() > FIELD_EPSILON
            && contributor.magnitude == contributor.value.abs()
            && (contributor.share_of_total_magnitude - contributor.magnitude / total_magnitude)
                .abs()
                < 1e-9
    }));
    assert_eq!(square.position, position);
}

#[test]
fn all_influence_contributors_returns_board_shaped_breakdowns() {
    let state = fixture();
    let breakdowns = all_influence_contributors(&state);

    assert_eq!(breakdowns.len(), BOARD_SIZE as usize);
    assert!(
        breakdowns
            .iter()
            .all(|row| row.len() == BOARD_SIZE as usize)
    );
    assert_eq!(
        breakdowns[4][3],
        influence_contributors_at(Position { x: 3, y: 4 }, &state)
    );
}

#[test]
fn instability_links_use_hostile_contributors_and_stable_magnitude_denominator() {
    let mut state = fixture();
    for components in [&mut state.components.red, &mut state.components.blue] {
        for values in [
            &mut components.pawn,
            &mut components.rook,
            &mut components.spy,
            &mut components.king,
        ] {
            values.fill(0);
        }
    }
    state.home_energy = PieceTypeMap {
        pawn: 1.0,
        rook: 0.0,
        spy: 0.0,
        king: 0.0,
    };
    state.pieces = vec![
        Piece {
            id: "red-spy-target".to_owned(),
            owner: Player::Red,
            piece_type: PieceType::Spy,
            position: Position { x: 3, y: 3 },
            unstable: false,
        },
        Piece {
            id: "blue-pawn-a".to_owned(),
            owner: Player::Blue,
            piece_type: PieceType::Pawn,
            position: Position { x: 3, y: 3 },
            unstable: false,
        },
        Piece {
            id: "blue-pawn-b".to_owned(),
            owner: Player::Blue,
            piece_type: PieceType::Pawn,
            position: Position { x: 3, y: 3 },
            unstable: false,
        },
        Piece {
            id: "red-pawn-cancel".to_owned(),
            owner: Player::Red,
            piece_type: PieceType::Pawn,
            position: Position { x: 3, y: 3 },
            unstable: false,
        },
    ];

    let square = influence_contributors_at(Position { x: 3, y: 3 }, &state);
    assert!((square.total + 1.0).abs() < 1e-9);
    assert!((square.contributors[0].share_of_total_magnitude - (1.0 / 3.0)).abs() < 1e-9);
    assert_eq!(
        square
            .highest_negative_contributor
            .as_ref()
            .map(|contributor| contributor.piece_id.as_str()),
        Some("blue-pawn-a")
    );

    let links = instability_influence_links(0.3, &state)
        .into_iter()
        .filter(|link| link.target_piece_id == "red-spy-target")
        .collect::<Vec<_>>();
    let contributor_ids = links
        .iter()
        .map(|link| link.contributor.piece_id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(contributor_ids, vec!["blue-pawn-a", "blue-pawn-b"]);
    assert!(links.iter().all(|link| {
        link.contributor.value < 0.0 && link.contributor.share_of_total_magnitude >= 0.3
    }));

    assert!(instability_influence_links(0.34, &state).is_empty());
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
fn easy_ai_prefers_safe_generosity_over_self_instability() {
    let mut state = fixture();
    state.current_player = Player::Red;
    state.pieces = vec![
        Piece {
            id: "red-king".to_string(),
            owner: Player::Red,
            piece_type: PieceType::King,
            position: Position { x: 3, y: 3 },
            unstable: false,
        },
        Piece {
            id: "red-rook".to_string(),
            owner: Player::Red,
            piece_type: PieceType::Rook,
            position: Position { x: 2, y: 3 },
            unstable: false,
        },
        Piece {
            id: "red-spy".to_string(),
            owner: Player::Red,
            piece_type: PieceType::Spy,
            position: Position { x: 1, y: 3 },
            unstable: false,
        },
        Piece {
            id: "blue-king".to_string(),
            owner: Player::Blue,
            piece_type: PieceType::King,
            position: Position { x: 6, y: 6 },
            unstable: false,
        },
        Piece {
            id: "blue-pawn".to_string(),
            owner: Player::Blue,
            piece_type: PieceType::Pawn,
            position: Position { x: 4, y: 3 },
            unstable: false,
        },
    ];
    state.components.red = PlayerComponents {
        pawn: vec![0],
        rook: vec![1, 0],
        spy: vec![1, 0, 0],
        king: vec![1, 0],
    };
    state.components.blue = PlayerComponents {
        pawn: vec![-1],
        rook: vec![0, 0],
        spy: vec![0, 0, 0],
        king: vec![0, 0],
    };
    state.activation_orders.red = PlayerActivationOrder {
        pawn: vec![],
        rook: vec![],
        spy: vec![],
        king: vec![],
    };
    state.activation_orders.blue = PlayerActivationOrder {
        pawn: vec![],
        rook: vec![],
        spy: vec![],
        king: vec![],
    };
    let next = play_easy_turn(
        state,
        Player::Red,
        AiTurnOptions {
            seed: Some(0),
            variety: Some(0.0),
            time_budget_ms: Some(10),
        },
    );
    let field = evaluate_field(&next);

    assert_eq!(next.current_player, Player::Blue);
    assert!(next.pieces.iter().any(|piece| piece.id == "red-spy"));
    assert!(unstable_pieces(Player::Red, &next, &field).is_empty());
    assert!(!is_king_unprotected(Player::Red, &next, &field));
}

#[test]
fn hint_search_returns_current_safe_move() {
    let state = fixture();
    let result = hint_search(Player::Blue, None, &state, 128, 100);
    let HintSearchResult::Success(hint) = result else {
        panic!("expected a hint search result");
    };

    assert!(hint.ok);
    assert!(hint.safe);
    assert_eq!(hint.loss_count, 0);
    assert_eq!(hint.tuning_distance, 0);
    assert!(hint.moves.len() >= 1);
    assert!(hint.tuned_kinds.is_empty());
    assert_eq!(hint.state.components, state.components);
    assert_eq!(
        hint.state
            .pieces
            .iter()
            .find(|piece| piece.id == hint.piece_id)
            .map(|piece| piece.owner),
        Some(Player::Blue)
    );
}

#[test]
fn hint_search_honors_focused_piece() {
    let state = fixture();
    let result = hint_search(Player::Blue, Some("blue-spy-1"), &state, 128, 100);
    let HintSearchResult::Success(hint) = result else {
        panic!("expected a focused hint search result");
    };

    assert_eq!(hint.piece_id, "blue-spy-1");
    assert!(hint.moves.len() >= 1);
}

#[test]
fn hint_search_falls_back_to_any_safe_piece_when_focus_has_no_result() {
    let state = fixture();
    let result = hint_search(Player::Blue, Some("blue-missing-piece"), &state, 128, 100);
    let HintSearchResult::Success(hint) = result else {
        panic!("expected a global hint search result");
    };

    assert!(hint.safe);
    assert_eq!(hint.loss_count, 0);
    assert_ne!(hint.piece_id, "blue-missing-piece");
    assert!(hint.moves.len() >= 1);
}

#[test]
fn hint_search_reports_exhausted_when_state_cap_stops_search() {
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

    let result = hint_search(Player::Blue, None, &state, 1, 0);
    let HintSearchResult::Failure(failure) = result else {
        panic!("expected no playable moves");
    };

    assert!(!failure.ok);
    assert_eq!(failure.reason, "no playable moves");
    assert!(failure.exhausted);
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
