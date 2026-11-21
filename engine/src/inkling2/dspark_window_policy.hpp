#pragma once

#include <algorithm>
#include <array>

// Decisions use only completed rounds. The two samples are wall time per
// committed token, including drafting/rollback/injection where applicable.
struct DsparkWindowPolicy {
    enum Mode { Single, Speculative };
    struct Stats {
        long long rounds = 0, tokens = 0, proposals = 0, accepted = 0;
        double total_ms = 0.0, estimated_ms = 0.0;
        int windows = 0;
    };
    std::array<Stats, 2> modes{};
    Mode next = Speculative, preferred = Speculative;
    int switches = 0, single_selections = 0, probes = 0;
    int window_tokens = 0, window_rounds = 0, preferred_windows = 0;
    double window_ms = 0.0;
    int probe_interval = 4;

    void observe(Mode mode, int tokens, int proposals, int accepted, double elapsed_ms, bool comparable) {
        Stats& stats = modes[mode];
        stats.total_ms += elapsed_ms;
        stats.rounds++;
        stats.tokens += tokens;
        stats.proposals += proposals;
        stats.accepted += accepted;
        // The first call can instantiate CUDA graphs. Truncated terminal rounds
        // and a forced singleton at the end cannot characterize a full window.
        if (!comparable || stats.rounds == 1) return;
        window_tokens += tokens;
        window_rounds++;
        window_ms += elapsed_ms;
        // End a losing probe once three completed rounds show a substantial
        // excess cost. Only earlier outputs influence the next choice.
        const bool probing = mode != preferred && modes[preferred].windows > 0;
        const double reference = modes[preferred].estimated_ms * window_tokens;
        const bool losing_probe = probing && window_rounds >= 3 &&
            window_ms > reference * 1.20 && window_ms - reference > 16.0;
        const int target_tokens = stats.windows == 0 ? 12 : 24;
        if (!losing_probe && (window_tokens < target_tokens || window_rounds < 6)) return;
        const double cost = window_ms / window_tokens;
        stats.estimated_ms = stats.windows == 0 ? cost : 0.5 * (cost + stats.estimated_ms);
        ++stats.windows;
        window_tokens = window_rounds = 0;
        window_ms = 0.0;
        const Mode other = mode == Single ? Speculative : Single;
        if (modes[other].windows == 0) {
            next = other;  // Initial calibration of the other method.
        } else {
            const Mode challenger = preferred == Single ? Speculative : Single;
            if (modes[challenger].estimated_ms < 0.96 * modes[preferred].estimated_ms) {
                preferred = challenger;
                probe_interval = 4;
                preferred_windows = 0;
                if (preferred == Single) ++single_selections;
            } else if (probing && losing_probe) {
                // Long stretches of unprofitable drafting get progressively
                // fewer trials, while still allowing recovery later in a chat.
                probe_interval = std::min(32, 2 * probe_interval);
            } else if (probing) {
                probe_interval = 4;
            }
            next = preferred;
            if (mode == preferred && ++preferred_windows >= probe_interval) {
                // Re-measure the alternative on the current region of the text.
                next = other;
                preferred_windows = 0;
                ++probes;
            }
        }
        if (next != mode) ++switches;
    }
};
