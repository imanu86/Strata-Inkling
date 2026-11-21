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

        // DSpark uses the same full-vocabulary distribution when top-k is disabled.
        p = SamplingParams{}; p.temp = 1.0f;
        std::vector<float> full(200058, 0.0f);
        const auto full_dist = build_sampling_dist(full.data(), (int) full.size(), {}, p);
        require(full_dist.idx.size() == full.size(), "speculative top-k=0 truncates full vocabulary");
        require(std::fabs(full_dist.prob_of(200057) - 1.0 / full.size()) < 1e-12,
                "speculative distribution lost a vocabulary tail token");
        // O(V) residual over a full vocabulary: equal p=q has no residual and falls back to p.
        const auto tail = sample_residual_dist(full_dist, full_dist, rng);
        require(tail >= 0 && tail < (int) full.size(), "full-vocabulary residual fallback is invalid");

        const std::vector<float> pl{std::log(0.1f), std::log(0.3f), std::log(0.6f)};
        const std::vector<float> ql{std::log(0.6f), std::log(0.3f), std::log(0.1f)};
        const auto pd = build_sampling_dist(pl.data(), 3, {}, p);
        const auto qd = build_sampling_dist(ql.data(), 3, {}, p);
        // All positive residual mass is at token 2; acceptance plus residual must reproduce p.
        for (int i = 0; i < 32; ++i)
            require(sample_residual_dist(pd, qd, rng) == 2, "rejection residual is not normalized max(p-q,0)");
        int counts[3]{};
        constexpr int trials = 50000;
        for (int i = 0; i < trials; ++i) {
            const int draft = sample_from_dist(qd, rng);
            const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
            const int accepted = u * qd.prob_of(draft) <= pd.prob_of(draft)
                                     ? draft : sample_residual_dist(pd, qd, rng);
            ++counts[accepted];
        }
        for (int i = 0; i < 3; ++i)
            require(std::fabs((double) counts[i] / trials - pd.prob_of(i)) < 0.012,
                    "speculative acceptance does not preserve the target distribution");
        std::cout << "Inkling sampling checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
