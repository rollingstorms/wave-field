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
