// Standalone, bitwise oracle against unchanged one-block depth-mix kernels.
#include "inkling2_kernels.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>
using namespace strata::inkling2;
#define CHECK(x)                                                                                                  \
    do {                                                                                                          \
        auto e = (x);                                                                                             \
        if (e != cudaSuccess) {                                                                                   \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                        \
            std::exit(2);                                                                                         \
        }                                                                                                         \
    } while (0)
struct Arena {
    std::vector<void *> data;
    template <class T> T *upload(const std::vector<T> &v) {
        T *p = nullptr;
        CHECK(cudaMalloc(&p, v.size() * sizeof(T)));
        data.push_back(p);
        CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
        return p;
    }
    ~Arena() {
        for (auto p : data)
            cudaFree(p);
    }
};
static unsigned rng = 7449123;
static float random_float() {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return ((int)(rng & 65535) - 32768) / 32768.0f;
}
int main(int argc, char **argv) {
    CHECK(cudaSetDevice(argc > 1 ? std::atoi(argv[1]) : 0));
    constexpr int N = 2304, pad = 16;
    constexpr float guard = 1234567.0f;
    int cases = 0;
    for (int T : {1, 2, 4})
        for (int D = 1; D <= 8; ++D)
            for (int mode = 0; mode < 4; ++mode) {
                const int nq = mode == 0 || mode == 1 ? 1 : mode == 2 ? 3 : 4;
                const int nout = mode == 1 ? 3 : nq;
                Arena arena;
                NormMixPtrs np{};
                for (int d = 0; d < D; ++d) {
                    std::vector<float> x((size_t)T * N);
                    for (auto &v : x)
                        v = random_float();
                    np.p[d] = arena.upload(x);
                }
                std::vector<float> query((size_t)4 * N);
                for (auto &v : query)
                    v = random_float();
                auto q = arena.upload(query);
                auto scratch = arena.upload(std::vector<float>((size_t)T * 4 * 8 * 8 + 2 * pad, guard));
                MixOuts old{};
                MixOuts4 old4{}, out{};
                const size_t values = (size_t)T * N;
                std::vector<float *> old_alloc, new_alloc;
                for (int i = 0; i < nout; ++i) {
                    auto a = arena.upload(std::vector<float>(values + 2 * pad, guard));
                    auto b = arena.upload(std::vector<float>(values + 2 * pad, guard));
                    old_alloc.push_back(a);
                    new_alloc.push_back(b);
                    old4.p[i] = a + pad;
                    out.p[i] = b + pad;
                    if (i < 3)
                        old.p[i] = a + pad;
                }
                const float scale = 1.0f / std::sqrt((float)N);
                if (mode == 3)
                    k_mix_normed_4<<<dim3(4, T), kThreads>>>(np, D, q, q + 2 * N, old4, N, scale);
                else
                    k_mix_normed<<<dim3(nq, T), kThreads>>>(np, D, q, old, nq, N, scale, mode == 1);
                k_depth_mix_partials<<<dim3(8, D, nq * T), 32>>>(np, D, q, mode == 3 ? q + 2 * N : nullptr,
                                                                 scratch + pad, nq, N);
                k_depth_mix_rows<<<dim3((N + 127) / 128, nq, T), 32>>>(np, D, scratch + pad, out, nq, N, scale,
                                                                       mode == 1);
                CHECK(cudaGetLastError());
                CHECK(cudaDeviceSynchronize());
                for (int i = 0; i < nout; ++i) {
                    std::vector<float> a(values + 2 * pad), b(a.size());
                    CHECK(cudaMemcpy(a.data(), old_alloc[i], a.size() * 4, cudaMemcpyDeviceToHost));
                    CHECK(cudaMemcpy(b.data(), new_alloc[i], b.size() * 4, cudaMemcpyDeviceToHost));
                    if (std::memcmp(a.data(), b.data(), a.size() * 4)) {
                        size_t j = 0;
                        for (; j < a.size(); ++j)
                            if (std::memcmp(&a[j], &b[j], 4))
                                break;
                        std::fprintf(stderr, "FAIL T%d D%d mode%d out%d idx%zu old%.9g new%.9g\n", T, D, mode, i,
                                     j, a[j], b[j]);
                        return 1;
                    }
                    for (int j = 0; j < pad; ++j)
                        if (b[j] != guard || b[pad + values + j] != guard) {
                            std::fprintf(stderr, "FAIL guard\n");
                            return 1;
                        }
                }
                std::vector<float> w((size_t)T * 4 * 8 * 8 + 2 * pad);
                CHECK(cudaMemcpy(w.data(), scratch, w.size() * 4, cudaMemcpyDeviceToHost));
                for (int j = 0; j < pad; ++j)
                    if (w[j] != guard || w[w.size() - 1 - j] != guard) {
                        std::fprintf(stderr, "FAIL scratch guard\n");
                        return 1;
                    }
                ++cases;
            }
    std::printf("PASS %d bitwise cases N2304 T1/2/4 D1..8 query1/3/4 SWA broadcast; guards intact\n", cases);
}
