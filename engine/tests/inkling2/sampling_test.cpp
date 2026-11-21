#include "strata/inkling2/sampling.hpp"

#include <iostream>
#include <stdexcept>

using namespace strata::inkling2;

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    try {
        SamplingParams p;
        p.temp = 1.0f;
        std::mt19937_64 rng(1337);
        std::vector<float> flat(4096, 0.0f);
        std::vector<bool> seen(flat.size(), false);
        int distinct = 0;
        for (int i = 0; i < 512; ++i) {
            const auto id = sample_logits(flat, {}, p, rng);
            require(id >= 0 && id < 4096, "sample outside vocabulary");
            if (!seen[(size_t) id]) { seen[(size_t) id] = true; ++distinct; }
        }
        // Equal logits do not give partial_sort a stable choice of IDs. Count
        // distinct draws, so any fixed 256-candidate truncation must fail.
        require(distinct > 400, "disabled top-k still truncates the vocabulary");

        std::vector<float> peaked(4096, -10.0f);
        for (int i = 4064; i < 4096; ++i) peaked[i] = 0.0f;
        p.top_k = 32;
        for (int i = 0; i < 64; ++i)
            require(sample_logits(peaked, {}, p, rng) >= 4064, "explicit top-k was ignored");

        p.top_k = 0;
        p.top_p = 0.5f;
        std::vector<float> nucleus{std::log(0.7f), std::log(0.2f), std::log(0.1f)};
        for (int i = 0; i < 32; ++i)
            require(sample_logits(nucleus, {}, p, rng) == 0, "nucleus threshold was ignored");

        p.top_p = 1.0f;
        p.min_p = 0.5f;
        std::vector<float> relative{0.0f, std::log(0.75f), std::log(0.25f)};
        for (int i = 0; i < 64; ++i)
            require(sample_logits(relative, {}, p, rng) != 2, "min-p threshold was ignored");

        p.min_p = 0.0f;
        std::mt19937_64 first(42), second(42);
        for (int i = 0; i < 64; ++i)
            require(sample_logits(flat, {}, p, first) == sample_logits(flat, {}, p, second),
                    "fixed seed is not reproducible");

        p.temp = 0.0f;
        peaked[4000] = 10.0f;
        require(sample_logits(peaked, {}, p, rng) == 4000, "greedy selection changed");
        p.pres_pen = 20.0f;
        require(sample_logits(peaked, {4000}, p, rng) >= 4064, "presence penalty was ignored");
        std::cout << "Inkling sampling checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
