#pragma once
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

namespace strata::inkling2 {
// Stable ties preserve expert ID; zero-observation entries are never called hot.
inline std::vector<int> replica_hot_order(const std::vector<double>& counts, int capacity) {
    std::vector<int> ids(counts.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return counts[a] > counts[b]; });
    ids.resize(std::min(ids.size(), (std::size_t) std::max(0, capacity)));
    ids.erase(std::remove_if(ids.begin(), ids.end(), [&](int e) { return counts[e] <= 0; }), ids.end());
    return ids;
}
} // namespace strata::inkling2
