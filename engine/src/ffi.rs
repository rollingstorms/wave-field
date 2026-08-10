use crate::api;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

fn read_str<'a>(value: *const c_char) -> Result<&'a str, String> {
    if value.is_null() {
        return Err("received null string pointer".to_owned());
    }
    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map_err(|error| format!("received non-UTF-8 string: {error}"))
}

fn into_c_string(result: Result<String, String>) -> *mut c_char {
    let Ok(value) = result else {
        return ptr::null_mut();
    };
    CString::new(value)
        .map(CString::into_raw)
        .unwrap_or(ptr::null_mut())
}

fn call_json(operation: impl FnOnce() -> Result<String, String>) -> *mut c_char {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(result) => into_c_string(result),
        Err(_) => ptr::null_mut(),
    }
}

fn call_bool(operation: impl FnOnce() -> Result<bool, String>) -> bool {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(value)) => value,
        Ok(Err(_)) | Err(_) => false,
    }
}

/// Frees strings returned by the Wave Field C FFI.
///
/// Swift callers should pass every non-null returned `UnsafeMutablePointer<CChar>`
/// back to this function exactly once after copying it into a Swift `String`.
#[unsafe(no_mangle)]
pub extern "C" fn wf_string_free(value: *mut c_char) {
    if value.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(value);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_new_game_json() -> *mut c_char {
    call_json(api::new_game_json)
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_undo_json(state_json: *const c_char) -> *mut c_char {
    call_json(|| api::undo_json(read_str(state_json)?))
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_evaluate_field_json(state_json: *const c_char) -> *mut c_char {
    call_json(|| api::evaluate_field_json(read_str(state_json)?))
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_preview_move_json(
    piece_id: *const c_char,
    x: i32,
    y: i32,
    state_json: *const c_char,
) -> *mut c_char {
    call_json(|| api::preview_move_json(read_str(piece_id)?, x, y, read_str(state_json)?))
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_piece_pattern_json(
    player: *const c_char,
    piece_type: *const c_char,
    state_json: *const c_char,
) -> *mut c_char {
    call_json(|| {
        api::piece_pattern_json(
            read_str(player)?,
            read_str(piece_type)?,
            read_str(state_json)?,
        )
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_legal_moves_json(
    piece_id: *const c_char,
    state_json: *const c_char,
) -> *mut c_char {
    call_json(|| api::legal_moves_json(read_str(piece_id)?, read_str(state_json)?))
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_playable_moves_json(
    piece_id: *const c_char,
    state_json: *const c_char,
) -> *mut c_char {
    call_json(|| api::playable_moves_json(read_str(piece_id)?, read_str(state_json)?))
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_closest_playable_configuration_json(
    player: *const c_char,
    state_json: *const c_char,
) -> *mut c_char {
    call_json(|| api::closest_playable_configuration_json(read_str(player)?, read_str(state_json)?))
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_hint_search_json(
    player: *const c_char,
    focused_piece_id: *const c_char,
    state_json: *const c_char,
    max_tuning_states: u32,
    time_budget_ms: u32,
) -> *mut c_char {
    call_json(|| {
        api::hint_search_json(
            read_str(player)?,
            read_str(focused_piece_id)?,
            read_str(state_json)?,
            max_tuning_states,
            time_budget_ms,
        )
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_apply_move_json(
    piece_id: *const c_char,
    x: i32,
    y: i32,
    state_json: *const c_char,
    analyze_checkmate: bool,
) -> *mut c_char {
    call_json(|| {
        api::apply_move_json(
            read_str(piece_id)?,
            x,
            y,
            read_str(state_json)?,
            analyze_checkmate,
        )
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_begin_turn_json(
    state_json: *const c_char,
    analyze_checkmate: bool,
) -> *mut c_char {
    call_json(|| api::begin_turn_json(read_str(state_json)?, analyze_checkmate))
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_apply_tuning_json(
    player: *const c_char,
    piece_type: *const c_char,
    component_index: usize,
    value: i8,
    state_json: *const c_char,
) -> *mut c_char {
    call_json(|| {
        api::apply_tuning_json(
            read_str(player)?,
            read_str(piece_type)?,
            component_index,
            value,
            read_str(state_json)?,
        )
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_unstable_piece_ids_json(
    player: *const c_char,
    state_json: *const c_char,
) -> *mut c_char {
    call_json(|| api::unstable_piece_ids_json(read_str(player)?, read_str(state_json)?))
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_king_unprotected_json(
    player: *const c_char,
    state_json: *const c_char,
) -> bool {
    call_bool(|| api::king_unprotected_json(read_str(player)?, read_str(state_json)?))
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_mark_instability_json(state_json: *const c_char) -> *mut c_char {
    call_json(|| api::mark_instability_json(read_str(state_json)?))
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_resign_in_check_json(state_json: *const c_char) -> *mut c_char {
    call_json(|| api::resign_in_check_json(read_str(state_json)?))
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_apply_closest_playable_hint_json(state_json: *const c_char) -> *mut c_char {
    call_json(|| api::apply_closest_playable_hint_json(read_str(state_json)?))
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_reset_tuning_json(state_json: *const c_char) -> *mut c_char {
    call_json(|| api::reset_tuning_json(read_str(state_json)?))
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_randomize_tuning_json(
    rolls_json: *const c_char,
    state_json: *const c_char,
) -> *mut c_char {
    call_json(|| api::randomize_tuning_json(read_str(rolls_json)?, read_str(state_json)?))
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_play_heuristic_turn_json(
    player: *const c_char,
    state_json: *const c_char,
    seed: u32,
    variety: f64,
    time_budget_ms: u32,
) -> *mut c_char {
    call_json(|| {
        api::play_heuristic_turn_json(
            read_str(player)?,
            read_str(state_json)?,
            seed,
            variety,
            time_budget_ms,
        )
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_play_easy_turn_json(
    player: *const c_char,
    state_json: *const c_char,
    seed: u32,
    variety: f64,
    time_budget_ms: u32,
) -> *mut c_char {
    call_json(|| {
        api::play_easy_turn_json(
            read_str(player)?,
            read_str(state_json)?,
            seed,
            variety,
            time_budget_ms,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> CString {
        CString::new(include_str!("../tests/initial-state.json")).unwrap()
    }

    #[test]
    fn ffi_returns_json_for_valid_state() {
        let state = fixture();
        let result = wf_evaluate_field_json(state.as_ptr());
        assert!(!result.is_null());

        let value = unsafe { CStr::from_ptr(result) }
            .to_str()
            .expect("UTF-8 JSON");
        let field: Vec<Vec<f64>> = serde_json::from_str(value).expect("field JSON");
        assert_eq!(field.len(), 7);

        wf_string_free(result);
    }

    #[test]
    fn ffi_new_game_has_product_component_counts() {
        let result = wf_new_game_json();
        assert!(!result.is_null());
        let value = unsafe { CStr::from_ptr(result) }
            .to_str()
            .expect("UTF-8 JSON");
        let state: crate::GameState = serde_json::from_str(value).expect("game state JSON");
        assert_eq!(state.components.blue.pawn.len(), 1);
        assert_eq!(state.components.blue.rook.len(), 2);
        assert_eq!(state.components.blue.spy.len(), 2);
        assert_eq!(state.components.blue.king.len(), 2);
        wf_string_free(result);
    }

    #[test]
    fn ffi_returns_null_for_invalid_state() {
        let state = CString::new("{}").unwrap();
        let result = wf_evaluate_field_json(state.as_ptr());
        assert!(result.is_null());
    }
}
