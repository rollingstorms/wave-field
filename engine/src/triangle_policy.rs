use crate::{BasisDefinition, GameState, PlayerActivationOrder, PlayerComponents};

// Triangle Hat keeps the middle basis from the original three. Its positive and
// negative phases were tuning options 3 and 4 of the six-option sequence.
fn phase(values: &[i8]) -> i8 {
    values
        .get(1)
        .copied()
        .filter(|value| *value != 0)
        .or_else(|| values.iter().copied().find(|value| *value != 0))
        .unwrap_or(1)
        .signum()
}

fn normalize_components(components: &mut PlayerComponents) {
    components.spy = vec![phase(&components.spy)];
}

fn normalize_order(order: &mut PlayerActivationOrder) {
    order.spy = vec![0];
}

fn middle_basis(definitions: &[BasisDefinition]) -> Result<BasisDefinition, String> {
    definitions
        .get(if definitions.len() > 1 { 1 } else { 0 })
        .cloned()
        .ok_or_else(|| "Triangle Hat basis is missing".to_owned())
}

pub(crate) fn normalize(state: &mut GameState) -> Result<(), String> {
    state.definitions.spy = vec![middle_basis(&state.definitions.spy)?];
    normalize_components(&mut state.components.blue);
    normalize_components(&mut state.components.red);
    normalize_components(&mut state.default_components);
    normalize_order(&mut state.activation_orders.blue);
    normalize_order(&mut state.activation_orders.red);

    for previous in &mut state.history {
        previous.definitions.spy = vec![middle_basis(&previous.definitions.spy)?];
        normalize_components(&mut previous.components.blue);
        normalize_components(&mut previous.components.red);
        normalize_order(&mut previous.activation_orders.blue);
        normalize_order(&mut previous.activation_orders.red);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MoveResult, api};

    #[test]
    fn new_games_use_only_the_two_middle_triangle_phases() {
        let state: GameState = serde_json::from_str(&api::new_game_json().unwrap()).unwrap();
        assert_eq!(state.default_components.spy, vec![1]);
        assert_eq!(state.components.blue.spy, vec![1]);
        assert_eq!(state.components.red.spy, vec![1]);
        assert_eq!(state.definitions.spy.len(), 1);

        let tuned: MoveResult = serde_json::from_str(
            &api::apply_tuning_json("blue", "spy", 0, -1, &api::new_game_json().unwrap()).unwrap(),
        )
        .unwrap();
        assert!(tuned.ok);
        assert_eq!(tuned.state.components.blue.spy, vec![-1]);

        let removed: MoveResult = serde_json::from_str(
            &api::apply_tuning_json("blue", "spy", 1, 1, &api::new_game_json().unwrap()).unwrap(),
        )
        .unwrap();
        assert!(!removed.ok);
    }

    #[test]
    fn legacy_saves_and_undo_history_are_migrated() {
        let mut legacy: GameState =
            serde_json::from_str(include_str!("../tests/initial-state.json")).unwrap();
        legacy.components.blue.spy = vec![0, -1, 0];
        legacy.history.push(legacy.snapshot());
        legacy.components.blue.spy = vec![1, 0, 0];

        let migrated: GameState = serde_json::from_str(
            &api::normalize_state_json(&serde_json::to_string(&legacy).unwrap()).unwrap(),
        )
        .unwrap();
        assert_eq!(migrated.components.blue.spy, vec![1]);
        assert_eq!(migrated.history[0].components.blue.spy, vec![-1]);
        assert_eq!(migrated.activation_orders.blue.spy, vec![0]);
        assert_eq!(migrated.history[0].activation_orders.blue.spy, vec![0]);
        assert_eq!(migrated.definitions.spy.len(), 1);
        assert_eq!(migrated.history[0].definitions.spy.len(), 1);
    }
}
