#include "strata/inkling2/replica_policy.hpp"
#include <cassert>
#include <vector>
using strata::inkling2::replica_hot_order;
int main() {
    assert(replica_hot_order(std::vector<double>(128, 0), 32).empty());
    std::vector<double> counts(128, 0);
    counts[127] = 9; counts[19] = 4; counts[3] = 4; counts[45] = 1;
    assert((replica_hot_order(counts, 3) == std::vector<int>{127,3,19}));
    assert((replica_hot_order(counts, 128) == std::vector<int>{127,3,19,45}));
    assert(replica_hot_order(counts, 0).empty());
    // Mixed top-8 routing: residency and compute maps must form an exact partition.
    std::vector<int> local(128, -1), remote(128);
    for (int e=0;e<128;++e) remote[e]=e;
    const auto hot = replica_hot_order(counts, 3);
    for (int i=0;i<(int)hot.size();++i) { local[hot[i]]=i; remote[hot[i]]=-1; }
    const int top8[] = {127,0,19,81,3,46,45,5};
    float parts[8] = {};
    for (int i=0;i<8;++i) {
        const int e=top8[i];
        assert((local[e]>=0) != (remote[e]>=0));
        // Both compute owners scatter to the original top-k entry, never a partial sum.
        if(local[e]>=0) parts[i]=(float)e;
        if(remote[e]>=0) parts[i]=(float)e;
    }
    for (int i=0;i<8;++i) assert(parts[i]==(float)top8[i]);
}
