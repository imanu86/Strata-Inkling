#include <cuda_runtime.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x)                                                                                                     \
    do {                                                                                                          \
        cudaError_t e = (x);                                                                                      \
        if (e != cudaSuccess) {                                                                                   \
            std::fprintf(stderr, "%s: %s", #x, cudaGetErrorString(e));                                            \
            std::abort();                                                                                         \
        }                                                                                                         \
    } while (0)
#include "inkling2_pq.cuh"
int main() {
    const int n = 515, k = 2;
    std::vector<float> q(n), p(n * 3);
    for (int i = 0; i < n; ++i) {
        q[i] = 0.25f * float((i * 17) % 29 - 14);
        p[i] = q[i]; // equal distributions: always accept
        p[2 * n + i] = 0.0f;
    }
    for (int i = 0; i < n; ++i)
        p[n + i] = q[n - 1 - i];
    float *qd = nullptr, *pd = nullptr;
    CK(cudaMalloc(&qd, n * sizeof(float)));
    CK(cudaMalloc(&pd, p.size() * sizeof(float)));
    CK(cudaMemcpy(qd, q.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(pd, p.data(), p.size() * sizeof(float), cudaMemcpyHostToDevice));
    GpuPqSampler sampler(n, k);
    std::vector<double> qw(n), pw(n), res(n);
    double sumq = 0, sump = 0, sumr = 0;
    for (int i = 0; i < n; ++i) {
        qw[i] = std::exp(double(q[i]));
        pw[i] = std::exp(double(p[n + i]));
        sumq += qw[i];
        sump += pw[i];
    }
    for (int i = 0; i < n; ++i) {
        res[i] = std::fmax(0.0, pw[i] / sump - qw[i] / sumq);
        sumr += res[i];
    }
    auto inverse = [&](const std::vector<double> &w, double sum, double u) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) {
            acc += w[i];
            if (acc > u * sum)
                return i;
        }
        return n - 1;
    };
    int cases = 0;
    for (int j = 0; j < 97; ++j) {
        double u = (double(j) + 0.371) / 97;
        int y = sampler.sample_proposal(0, qd, u, 0);
        int z = sampler.sample_proposal(1, qd, 1.0 - u, 0);
        assert(y == inverse(qw, sumq, u) && z == inverse(qw, sumq, 1.0 - u));
        sampler.verify(pd, k, 0);
        assert(std::abs(sampler.host[0].acceptance - 1.0) < 1e-12);
        assert(std::abs(sampler.host[1].acceptance - std::fmin(1.0, (pw[z] / sump) / (qw[z] / sumq))) < 3e-6);
        assert(sampler.sample_residual(pd, 1, u, 0) == inverse(res, sumr, u));
        assert(sampler.sample_target(pd, 2, u, 0) == int(u * n));
        ++cases;
    }
    CK(cudaFree(pd));
    CK(cudaFree(qd));
    std::printf("passed %d CDF, acceptance, signed residual and bonus row cases, vocab=%d\n", cases, n);
}
