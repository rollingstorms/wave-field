#ifndef WAVE_FIELD_ENGINE_H
#define WAVE_FIELD_ENGINE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void wf_string_free(char *value);

char *wf_new_game_json(void);
char *wf_undo_json(const char *state_json);
char *wf_evaluate_field_json(const char *state_json);
char *wf_piece_pattern_json(
    const char *player,
    const char *piece_type,
    const char *state_json
);
char *wf_legal_moves_json(const char *piece_id, const char *state_json);
char *wf_playable_moves_json(const char *piece_id, const char *state_json);
char *wf_closest_playable_configuration_json(const char *player, const char *state_json);
char *wf_apply_move_json(
    const char *piece_id,
    int32_t x,
    int32_t y,
    const char *state_json,
    bool analyze_checkmate
);
char *wf_begin_turn_json(const char *state_json, bool analyze_checkmate);
char *wf_apply_tuning_json(
    const char *player,
    const char *piece_type,
    uintptr_t component_index,
    int8_t value,
    const char *state_json
);
char *wf_unstable_piece_ids_json(const char *player, const char *state_json);
bool wf_king_unprotected_json(const char *player, const char *state_json);
char *wf_mark_instability_json(const char *state_json);
char *wf_resign_in_check_json(const char *state_json);
char *wf_apply_closest_playable_hint_json(const char *state_json);
char *wf_reset_tuning_json(const char *state_json);
char *wf_randomize_tuning_json(const char *rolls_json, const char *state_json);
char *wf_play_heuristic_turn_json(
    const char *player,
    const char *state_json,
    uint32_t seed,
    double variety,
    uint32_t time_budget_ms
);
char *wf_play_easy_turn_json(
    const char *player,
    const char *state_json,
    uint32_t seed,
    double variety,
    uint32_t time_budget_ms
);

#ifdef __cplusplus
}
#endif

#endif
