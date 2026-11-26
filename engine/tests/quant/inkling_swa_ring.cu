// Standalone SWA batched-store regression: no model, Engine, or cuBLAS required.
#include "../../src/inkling2/split/inkling2_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

static bool run_case(int prefix, int batch, int tmax, bool fixed) {
    constexpr int window = 1024, heads = 3, dim = 128;
    const int cap = fixed ? window + tmax - 1 : window;
    // Initialize only the last cap context rows. Their ring slots are unique,
    // even after multiple wraps; a single launch over [0,prefix) would race.
    const int context = std::min(prefix, cap);
    const int rows = std::max(context, batch);
    Buffer<float> k(size_t(rows) * dim), v(size_t(rows) * dim);
    Buffer<int8_t> kq(size_t(cap) * dim), vq(size_t(cap) * dim);
    Buffer<__half> ks(cap), vs(cap);
    Buffer<int32_t> pos(1);
    Buffer<float> q(size_t(batch) * heads * dim), gate(size_t(batch) * heads * dim);
    Buffer<float> sink(size_t(batch) * heads), rel(size_t(batch) * heads * window);
    Buffer<float> out(size_t(batch) * heads * dim);
    check(cudaMemset(k.p, 0, size_t(rows) * dim * sizeof(float)));
    std::vector<float> values(size_t(rows) * dim, -1.0f);
    check(cudaMemcpy(v.p, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    int32_t start = prefix - context;
    check(cudaMemcpy(pos.p, &start, sizeof(start), cudaMemcpyHostToDevice));
    strata::inkling2::k_clip_norm_kv_store_q8<<<dim3(1, context), 32>>>(
        k.p, v.p, kq.p, ks.p, vq.p, vs.p, 1, 1.0f, 0.0f, pos.p, cap, 1);
    check(cudaGetLastError());
    std::fill(values.begin(), values.end(), 1.0f);
    check(cudaMemcpy(v.p, values.data(), size_t(batch) * dim * sizeof(float), cudaMemcpyHostToDevice));
    start = prefix;
    check(cudaMemcpy(pos.p, &start, sizeof(start), cudaMemcpyHostToDevice));
    strata::inkling2::k_clip_norm_kv_store_q8<<<dim3(1, batch), 32>>>(
        k.p, v.p, kq.p, ks.p, vq.p, vs.p, 1, 1.0f, 0.0f, pos.p, cap, 1);
    check(cudaGetLastError());
    check(cudaMemset(q.p, 0, size_t(batch) * heads * dim * sizeof(float)));
    check(cudaMemset(gate.p, 0, size_t(batch) * heads * dim * sizeof(float)));
    check(cudaMemset(sink.p, 0, size_t(batch) * heads * sizeof(float)));
    check(cudaMemset(rel.p, 0, size_t(batch) * heads * window * sizeof(float)));
    strata::inkling2::k_attn_q8_gqa3_pp<<<dim3(1, batch), 128>>>(
        q.p, kq.p, ks.p, vq.p, vs.p, rel.p, sink.p, gate.p, out.p,
        heads, 1, window, cap, 1, window, pos.p, 1.0f / std::sqrt(float(dim)));
    check(cudaGetLastError());
    std::vector<float> actual(size_t(batch) * heads * dim);
    check(cudaMemcpy(actual.data(), out.p, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    // Zero logits give equal token weights plus one zero-valued sink;
    // sigmoid(0) halves the result. Include the actual FP16 Q8 scale rounding.
    const float amplitude = 127.0f * __half2float(__float2half_rn(1.0f / 127.0f));
    float max_error = 0, first_error = 0, last_error = 0;
    for (int t = 0; t < batch; ++t) {
        const int length = std::min(window, prefix + t + 1);
        const float expected = 0.5f * amplitude * float(2 * (t + 1) - length) / float(length + 1);
        for (int c = 0; c < heads * dim; ++c) {
            const float a = actual[size_t(t) * heads * dim + c];
            const float error = std::isfinite(a) ? std::abs(a - expected) : INFINITY;
            max_error = std::max(max_error, error);
            if (t == 0) first_error = std::max(first_error, error);
            if (t == batch - 1) last_error = std::max(last_error, error);
        }
    }
    const bool pass = fixed ? max_error <= 2e-5f : first_error > 1e-3f && last_error <= 2e-5f;
    std::printf("%s %s P=%d T=%d Tmax=%d cap=%d max=%g first=%g last=%g\n",
        pass ? "PASS" : "FAIL", fixed ? "fixed" : "old-negative-control",
        prefix, batch, tmax, cap, max_error, first_error, last_error);
    return pass;
}

int main(int argc, char** argv) {
    int device = -1;
    if (argc == 3 && std::string(argv[1]) == "--device") {
        char* end = nullptr;
        const long parsed = std::strtol(argv[2], &end, 10);
        if (end != argv[2] && *end == '\0' && parsed >= 0 && parsed <= 1024) device = int(parsed);
    }
    if (device < 0) {
        std::fprintf(stderr, "Usage: inkling_swa_ring --device N (select an inference GPU explicitly)\n");
        return 2;
    }
    int count = 0;
    const cudaError_t available = cudaGetDeviceCount(&count);
    if (available == cudaErrorNoDevice || available == cudaErrorInsufficientDriver || (available == cudaSuccess && count == 0)) {
        std::puts("SKIP: no CUDA device available");
        return 77;
    }
    try {
        check(available);
        if (device >= count) throw std::runtime_error("selected CUDA device does not exist");
        check(cudaSetDevice(device));
        bool pass = true;
        for (const int prefix : {1023, 1024, 1087, 2305}) {
            for (const int batch : {4, 8, 64}) {
                pass = run_case(prefix, batch, 64, false) && pass;
                pass = run_case(prefix, batch, 64, true) && pass;
            }
            pass = run_case(prefix, 8, 8, true) && pass;
        }
        return pass ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "CUDA regression failed: %s\n", e.what());
        return 1;
    }
}
