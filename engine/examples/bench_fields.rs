use std::hint::black_box;
use std::time::Instant;
use wave_field_engine::{GameState, GameVariant, evaluate_field};

fn measure(states: &[GameState], iterations: usize) -> f64 {
    let start = Instant::now();
    for index in 0..iterations {
        black_box(evaluate_field(black_box(&states[index % states.len()])));
    }
    start.elapsed().as_nanos() as f64 / iterations as f64
}

fn main() {
    let opening: GameState =
        serde_json::from_str(include_str!("../tests/product-initial-state.json")).unwrap();
    let mut states = Vec::new();
    for profile in [[1, 1], [1, -1], [-1, 1], [-1, -1]] {
        let mut state = opening.clone();
        for components in [&mut state.components.red, &mut state.components.blue] {
            components.rook = profile.to_vec();
            components.king = profile.to_vec();
            components.pawn = vec![profile[0]];
            components.spy = vec![profile[1]];
        }
        states.push(state.clone());
        for (index, piece) in state.pieces.iter_mut().enumerate() {
            piece.position.x = (index % 3) as i32 * 3;
            piece.position.y = (index / 3) as i32 * 2;
        }
        states.push(state);
    }
    let mut classic = states.clone();
    for state in &mut classic {
        state.variant = GameVariant::Classic;
        state.home_energy.spy = 0.5;
    }
    for state in &mut states {
        state.variant = GameVariant::Sparse;
    }
    measure(&classic, 5_000);
    measure(&states, 5_000);
    let mut classic_samples = Vec::new();
    let mut sparse_samples = Vec::new();
    for _ in 0..5 {
        classic_samples.push(measure(&classic, 100_000));
        sparse_samples.push(measure(&states, 100_000));
    }
    classic_samples.sort_by(f64::total_cmp);
    sparse_samples.sort_by(f64::total_cmp);
    println!(
        "{{\"iterationsPerSample\":100000,\"positions\":8,\"classicNsPerField\":{:.1},\"discreteNsPerField\":{:.1},\"fieldSpeedup\":{:.2}}}",
        classic_samples[2],
        sparse_samples[2],
        classic_samples[2] / sparse_samples[2]
    );
}
