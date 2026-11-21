#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace strata::inkling2 {

struct SamplingParams {
    float temp = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    float min_p = 0.0f;
    float rep_pen = 1.0f;
    float freq_pen = 0.0f;
    float pres_pen = 0.0f;
    int pen_last_n = 64;
    uint64_t seed = 0;
    bool needs_logits() const {
        return temp > 0.0f || rep_pen != 1.0f || freq_pen != 0.0f || pres_pen != 0.0f;
    }
};

inline int32_t sample_logits(std::vector<float>& logits, const std::vector<int32_t>& history, const SamplingParams& smp, std::mt19937_64& rng) {
    const int V = (int) logits.size();
    if (smp.pen_last_n > 0 && (smp.rep_pen != 1.0f || smp.freq_pen != 0.0f || smp.pres_pen != 0.0f)) {
        std::vector<int> counts;
        std::vector<int32_t> uniq;
        const size_t start = history.size() > (size_t) smp.pen_last_n ? history.size() - (size_t) smp.pen_last_n : 0;
        for (size_t i = start; i < history.size(); ++i) {
            const int32_t tok = history[i];
            if (tok < 0 || tok >= V) continue;
            bool found = false;
            for (size_t j = 0; j < uniq.size(); ++j) {
                if (uniq[j] == tok) { ++counts[j]; found = true; break; }
            }
            if (!found) { uniq.push_back(tok); counts.push_back(1); }
        }
        for (size_t j = 0; j < uniq.size(); ++j) {
            float& l = logits[(size_t) uniq[j]];
            if (smp.rep_pen != 1.0f) l = (l <= 0.0f) ? (l * smp.rep_pen) : (l / smp.rep_pen);
            l -= (float) counts[j] * smp.freq_pen + smp.pres_pen;
        }
    }
    if (smp.temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < V; ++i) if (logits[(size_t) i] > logits[(size_t) best]) best = i;
        return best;
    }
    // A disabled top-k keeps the whole vocabulary. Sorting is unnecessary when
    // neither nucleus nor min-p truncation is requested (the model defaults).
    const int K = (smp.top_k > 0 && smp.top_k < V) ? smp.top_k : V;
    const bool ordered = K < V || smp.min_p > 0.0f || (smp.top_p > 0.0f && smp.top_p < 1.0f);
    std::vector<int32_t> idx((size_t) V);
    for (int i = 0; i < V; ++i) idx[(size_t) i] = i;
    if (ordered) {
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int32_t a, int32_t b) {
            return logits[(size_t) a] > logits[(size_t) b];
        });
    }
    const float max_l = ordered ? logits[(size_t) idx[0]] : *std::max_element(logits.begin(), logits.end());
    const float inv_t = 1.0f / smp.temp;
    std::vector<float> probs((size_t) K);
    double sum = 0.0;
    for (int i = 0; i < K; ++i) {
        const float p = std::exp((logits[(size_t) idx[(size_t) i]] - max_l) * inv_t);
        probs[(size_t) i] = p;
        sum += p;
    }
    int keep = K;
    if (smp.min_p > 0.0f) {
        const float thr = probs[0] * smp.min_p;
        while (keep > 1 && probs[(size_t) (keep - 1)] < thr) --keep;
    }
    if (smp.top_p > 0.0f && smp.top_p < 1.0f) {
        double csum = 0.0;
        const double target = sum * (double) smp.top_p;
        for (int i = 0; i < keep; ++i) {
            csum += probs[(size_t) i];
            if (csum >= target) { keep = i + 1; break; }
        }
    }
    double ksum = 0.0;
    for (int i = 0; i < keep; ++i) ksum += probs[(size_t) i];
    const double r = std::uniform_real_distribution<double>(0.0, ksum)(rng);
    double acc = 0.0;
    for (int i = 0; i < keep; ++i) {
        acc += probs[(size_t) i];
        if (r <= acc) return idx[(size_t) i];
    }
    return idx[0];
}

// DSpark acceptance uses full distributions; dense token lookup keeps residual sampling O(V).
struct SamplingDist {
    std::vector<int32_t> idx;
    std::vector<double> probs;
    std::vector<double> dense;
    double prob_of(int32_t tok) const { return tok >= 0 && (size_t) tok < dense.size() ? dense[(size_t) tok] : 0.0; }
};

inline SamplingDist build_sampling_dist(const float* raw_logits, int V, const std::vector<int32_t>& history, const SamplingParams& smp) {
    std::vector<float> adj(raw_logits, raw_logits + V);
    if (smp.pen_last_n > 0 && (smp.rep_pen != 1.0f || smp.freq_pen != 0.0f || smp.pres_pen != 0.0f)) {
        std::vector<int> counts;
        std::vector<int32_t> uniq;
        const size_t start = history.size() > (size_t) smp.pen_last_n ? history.size() - (size_t) smp.pen_last_n : 0;
        for (size_t i = start; i < history.size(); ++i) {
            const int32_t tok = history[i];
            if (tok < 0 || tok >= V) continue;
            bool found = false;
            for (size_t j = 0; j < uniq.size(); ++j) {
                if (uniq[j] == tok) { ++counts[j]; found = true; break; }
            }
            if (!found) { uniq.push_back(tok); counts.push_back(1); }
        }
        for (size_t j = 0; j < uniq.size(); ++j) {
            float& l = adj[(size_t) uniq[j]];
            if (smp.rep_pen != 1.0f) l = (l <= 0.0f) ? (l * smp.rep_pen) : (l / smp.rep_pen);
            l -= (float) counts[j] * smp.freq_pen + smp.pres_pen;
        }
    }
    SamplingDist d;
    d.dense.assign((size_t) V, 0.0);
    if (smp.temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < V; ++i) if (adj[(size_t) i] > adj[(size_t) best]) best = i;
        d.idx = {best};
        d.probs = {1.0f};
        d.dense[(size_t) best] = 1.0;
        return d;
    }
    const int K = (smp.top_k > 0 && smp.top_k < V) ? smp.top_k : V;
    std::vector<int32_t> idx((size_t) V);
    for (int i = 0; i < V; ++i) idx[(size_t) i] = i;
    const bool ordered = K < V || smp.min_p > 0.0f || (smp.top_p > 0.0f && smp.top_p < 1.0f);
    if (ordered) std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int32_t a, int32_t b) {
        return adj[(size_t) a] > adj[(size_t) b];
    });
    const float max_l = ordered ? adj[(size_t) idx[0]] : *std::max_element(adj.begin(), adj.end());
    const float inv_t = 1.0f / smp.temp;
    std::vector<float> probs((size_t) K);
    double sum = 0.0;
    for (int i = 0; i < K; ++i) {
        const float p = std::exp((adj[(size_t) idx[(size_t) i]] - max_l) * inv_t);
        probs[(size_t) i] = p;
        sum += p;
    }
    int keep = K;
    if (smp.min_p > 0.0f) {
        const float thr = probs[0] * smp.min_p;
        while (keep > 1 && probs[(size_t) (keep - 1)] < thr) --keep;
    }
    if (smp.top_p > 0.0f && smp.top_p < 1.0f) {
        double csum = 0.0;
        const double target = sum * (double) smp.top_p;
        for (int i = 0; i < keep; ++i) {
            csum += probs[(size_t) i];
            if (csum >= target) { keep = i + 1; break; }
        }
    }
    double ksum = 0.0;
    for (int i = 0; i < keep; ++i) ksum += probs[(size_t) i];
    const double inv_ksum = (ksum > 0.0) ? 1.0 / ksum : 1.0;
    d.idx.assign(idx.begin(), idx.begin() + keep);
    d.probs.resize((size_t) keep);
    for (int i = 0; i < keep; ++i) {
        d.probs[(size_t) i] = probs[(size_t) i] * inv_ksum;
        d.dense[(size_t) d.idx[(size_t) i]] = d.probs[(size_t) i];
    }
    return d;
}

inline int32_t sample_from_dist(const SamplingDist& d, std::mt19937_64& rng) {
    if (d.idx.size() <= 1) return d.idx[0];
    const double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double acc = 0.0;
    for (size_t i = 0; i < d.idx.size(); ++i) {
        acc += d.probs[i];
        if (r <= acc) return d.idx[i];
    }
    return d.idx[0];
}

inline int32_t sample_residual_dist(const SamplingDist& p, const SamplingDist& q, std::mt19937_64& rng) {
    std::vector<int32_t> r_idx;
    std::vector<double> r_w;
    r_idx.reserve(p.idx.size());
    r_w.reserve(p.idx.size());
    double sum = 0.0;
    for (size_t i = 0; i < p.idx.size(); ++i) {
        const double diff = p.probs[i] - q.prob_of(p.idx[i]);
        if (diff > 0.0f) {
            r_idx.push_back(p.idx[i]);
            r_w.push_back(diff);
            sum += diff;
        }
    }
    if (sum <= 0.0) return sample_from_dist(p, rng);
    const double u = std::uniform_real_distribution<double>(0.0, sum)(rng);
    double acc = 0.0;
    for (size_t i = 0; i < r_idx.size(); ++i) {
        acc += r_w[i];
        if (u <= acc) return r_idx[i];
    }
    return r_idx[0];
}



}  // namespace strata::inkling2
