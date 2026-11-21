#include "dspark_window_policy.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>

int main() {
    using Policy = DsparkWindowPolicy;
    Policy p;
    auto step = [&](double single_ms, double speculative_ms, bool equivalent = true) {
        const auto mode = p.next;
        const int n = mode == Policy::Single ? 1 : 2;
        p.observe(mode, n, n == 2 ? 1 : 0, n == 2 ? 1 : 0,
                  n * (mode == Policy::Single ? single_ms : speculative_ms), equivalent);
    };
    step(1e7, 1e7);  // CUDA graph capture belongs only to lifetime telemetry.
    assert(p.window_ms == 0.0);
    for (int i = 0; i < 1000; ++i) step(10, 20);
    assert(p.preferred == Policy::Single && p.probes >= 2);
    assert(p.modes[Policy::Single].estimated_ms == 10.0);
    assert(p.modes[Policy::Speculative].estimated_ms == 20.0);
    assert(p.single_selections == 1);
    const int before = p.window_tokens;
    step(1e7, 1e7, false);
    assert(p.window_tokens == before);
    for (int i = 0; i < 1500; ++i) step(10, 4);
    assert(p.preferred == Policy::Speculative && p.probes >= 4);
    assert(p.modes[Policy::Speculative].estimated_ms < 4.1);
    Policy close;
    for (int i = 0; i < 500; ++i) {
        const auto mode = close.next;
        close.observe(mode, 1, 0, 0, mode == Policy::Single ? 9.8 : 10.0, true);
    }
    assert(close.preferred == Policy::Speculative && close.single_selections == 0);
    puts("warm exclusion, fallback, probes, recovery, terminal exclusion and hysteresis pass");
}
