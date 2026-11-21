#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace strata::inkling2 {
// Counters stay on the primary device until a request boundary; inputs are router IDs.
__global__ void k_replica_route_stats(const int32_t* ids, const int32_t* slots, int32_t* miss,
                                    unsigned long long* frequencies, unsigned long long* hits,
                                    int entries, int experts) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < entries; i += blockDim.x * gridDim.x) {
        const int e = ids[i];
        if (e < 0 || e >= experts) { atomicAdd(hits + 2, 1ull); miss[i] = 1; continue; }
        const int cold = slots[e] < 0;
        miss[i] = cold;
        atomicAdd(frequencies + e, 1ull);
        atomicAdd(hits + cold, 1ull);
    }
}
} // namespace strata::inkling2
