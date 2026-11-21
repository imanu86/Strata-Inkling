// Q3_K byte-layout and CPU/CUDA expert regression, no model or permanent dequantization.
// Link strata_quant. STRATA_INKLING_Q3_CUDA=1 additionally needs strata_kernels,
// strata_kernels_cpu and the CUDA runtime. --cpu-only never creates a CUDA context;
// --device N selects the physical inference card, rather than the Windows display card.
// CPU dots use the same exact Q8 values as the CUDA dots, removing activation-quantizer
// error from the comparison. The final SITU experts test the real activation quantizers too.
#include "strata/quant/dequant.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/fp16.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if STRATA_INKLING_Q3_CUDA
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#endif

namespace sq = strata::quant;
namespace {
constexpr int QK = 256, QB = 110;
int failures = 0;
void require(bool ok, const char* what) {
    if (!ok) { if (failures < 12) std::printf("FAIL: %s\n", what); ++failures; }
}

struct Block {
    std::array<uint8_t, QB> bytes{};
    std::array<float, QK> values{};
};

// Pack from logical signed integers, independently of the kernel's word-wise unpacking.
// ggml Q3_K: hmask[32], low bits[64], packed signed scales[12], fp16 d.
Block block(std::mt19937& rng, int scale_case = -1, int code_case = -1, uint16_t d = 0x2400) {
    Block b;
    std::array<int, 16> scales;
    for (int g = 0; g < 16; ++g) {
        const int u = scale_case < 0 ? (int) (rng() % 64) : (scale_case + g) % 64;
        scales[g] = u - 32;
        b.bytes[96 + g % 8] |= (uint8_t) ((u & 15) << (4 * (g / 8)));
        b.bytes[104 + g % 4] |= (uint8_t) ((u >> 4) << (2 * (g / 4)));
    }
    for (int i = 0; i < QK; ++i) {
        const int q = (code_case < 0 ? (int) (rng() % 8) : (code_case + i) % 8) - 4;
        const int j = (i / 128) * 32 + i % 32;
        b.bytes[32 + j] |= (uint8_t) ((q & 3) << (2 * ((i % 128) / 32)));
        if (q >= 0) b.bytes[i % 32] |= (uint8_t) (1 << (i / 32));
        b.values[i] = sq::fp16_to_fp32(d) * (float) scales[i / 16] * (float) q;
    }
    std::memcpy(b.bytes.data() + 108, &d, 2);
    return b;
}

struct Rows {
    int n, rows;
    std::vector<uint8_t> bytes;
    std::vector<float> values;
    Rows(int width, int count, std::mt19937& rng, bool corners = false, uint16_t d = 0x2400)
        : n(width), rows(count), bytes((size_t) count * (width / QK) * QB), values((size_t) count * width) {
        const uint16_t half_cases[] = {0, 0x8000, 1, 0x0400, 0x2400, 0xa400, 0x3c00};
        for (int r = 0; r < rows; ++r) for (int k = 0; k < n / QK; ++k) {
            const int sc = corners ? r % 64 : -1, qc = corners ? (r / 64) % 8 : -1;
            const Block b = block(rng, sc, qc, corners && r >= 512 ? half_cases[(r - 512) % 7] : d);
            std::memcpy(bytes.data() + ((size_t) r * (n / QK) + k) * QB, b.bytes.data(), QB);
            std::copy(b.values.begin(), b.values.end(), values.begin() + (size_t) r * n + k * QK);
        }
    }
    size_t row_bytes() const { return (size_t) (n / QK) * QB; }
};

struct Acts {
    int n, nt;
    std::vector<uint8_t> q8k, q81;
    std::vector<float> values;
    Acts(int width, int tokens, std::mt19937& rng)
        : n(width), nt(tokens), q8k((size_t) tokens * (width / QK) * 292),
          q81((size_t) tokens * (width / 32) * 36), values((size_t) tokens * width) {
        for (int t = 0; t < nt; ++t) for (int k = 0; k < n / QK; ++k) {
            uint8_t* a = q8k.data() + ((size_t) t * (n / QK) + k) * 292;
            const float d = std::ldexp(1.0f, -6 - ((t + k) % 4));
            std::memcpy(a, &d, 4);
            int16_t sums[16]{};
            for (int i = 0; i < QK; ++i) {
                const int8_t q = (int8_t) (t == 0 ? (i % 2 ? 127 : -127) : (int) (rng() % 255) - 127);
                a[4 + i] = (uint8_t) q;
                sums[i / 16] += q;
                values[(size_t) t * n + k * QK + i] = d * q;
                uint8_t* a32 = q81.data() + ((size_t) t * (n / 32) + k * 8 + i / 32) * 36;
                a32[4 + i % 32] = (uint8_t) q;
                if (i % 32 == 0) { const uint16_t dh = sq::fp32_to_fp16(d); std::memcpy(a32, &dh, 2); }
            }
            std::memcpy(a + 260, sums, sizeof sums);
        }
    }
};

double reference(const float* w, const float* a, int n, double& magnitude) {
    double v = 0; magnitude = 0;
    for (int i = 0; i < n; ++i) { const double p = (double) w[i] * a[i]; v += p; magnitude += std::fabs(p); }
    return v;
}

void check_cpu(const Rows& w, const Acts& a) {
    std::vector<float> deq(w.n);
    for (int r = 0; r < w.rows; ++r) {
        const uint8_t* row = w.bytes.data() + (size_t) r * w.row_bytes();
        require(sq::dequant_row(sq::Type::Q3_K, row, deq.data(), w.n), "Q3_K CPU dequant dispatch");
        for (int i = 0; i < w.n; ++i) require(deq[i] == w.values[(size_t) r * w.n + i], "Q3_K dequant vs logical codes/scales");
        for (sq::Arith ar : {sq::Arith::Generic, sq::Arith::Avx, sq::Arith::Avx2, sq::Arith::Neon}) {
            const sq::DotKernel* kernels[16]{};
            const int nk = sq::dot_kernels(sq::Type::Q3_K, ar, kernels, 16);
            for (int j = 0; j < nk; ++j) if (sq::isa_ok(kernels[j]->isa)) for (int t = 0; t < a.nt; ++t) {
                float got = 0;
                kernels[j]->fn(w.n, &got, row, a.q8k.data() + (size_t) t * (w.n / QK) * 292);
                double mag;
                const double want = reference(w.values.data() + (size_t) r * w.n, a.values.data() + (size_t) t * w.n, w.n, mag);
                require(std::isfinite(got) && std::fabs(got - want) <= 2e-6 * mag + 1e-12, "Q3_K CPU dot vs exact Q8 values");
            }
        }
    }
}

#if STRATA_INKLING_Q3_CUDA
namespace gpu = strata::kernels;
namespace cpu = strata::kernels::cpu;
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::printf("CUDA %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
struct Device {
    void* p = nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p, n), "malloc"); }
    ~Device() { cudaFree(p); }
    template<class T> T* as() { return static_cast<T*>(p); }
    void put(const void* h, size_t n) { ck(cudaMemcpy(p, h, n, cudaMemcpyHostToDevice), "upload"); }
    void get(void* h, size_t n) { ck(cudaMemcpy(h, p, n, cudaMemcpyDeviceToHost), "download"); }
};

void check_gpu_rows(const Rows& w, const Acts& a) {
    Device dw(w.bytes.size()), da(a.q81.size()), dy((size_t) w.rows * a.nt * 4), dd(w.values.size() * 4);
    dw.put(w.bytes.data(), w.bytes.size()); da.put(a.q81.data(), a.q81.size());
    gpu::iq_dequant_f32(11, dw.p, (int64_t) w.rows * w.n, dd.as<float>(), nullptr);
    std::vector<float> deq(w.values.size()), got((size_t) w.rows * a.nt);
    dd.get(deq.data(), deq.size() * 4);
    for (size_t i = 0; i < deq.size(); ++i) require(deq[i] == w.values[i], "CUDA Q3_K dequant vs logical codes/scales");
    for (bool old : {false, true}) {
        gpu::iq_set_old_kernels(old);
        gpu::iq_mmvq(11, dw.p, da.p, dy.as<float>(), w.n, w.rows, a.nt, nullptr);
        dy.get(got.data(), got.size() * 4);
        for (int t = 0; t < a.nt; ++t) for (int r = 0; r < w.rows; ++r) {
            double mag;
            const double want = reference(w.values.data() + (size_t) r * w.n, a.values.data() + (size_t) t * w.n, w.n, mag);
            const float v = got[(size_t) t * w.rows + r];
            require(std::isfinite(v) && std::fabs(v - want) <= 2e-6 * mag + 1e-12, "CUDA Q3_K dot vs exact Q8 values");
        }
    }
    gpu::iq_set_old_kernels(false);
}

double rel(const std::vector<float>& got, const std::vector<float>& want) {
    double n = 0, d = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) return HUGE_VAL;
        n += std::fabs((double) got[i] - want[i]); d += std::fabs(want[i]);
    }
    return n / (d + 1e-30);
}

void check_expert(int n, int ff, int down_type, std::mt19937& rng) {
    constexpr int nt = 3;
    cpu::NativeFmt f; std::string err;
    require(cpu::native_fmt(11, down_type, n, ff, f, err), "CPU Q3_K native format");
    require(gpu::native_expert_supported(11, down_type, n, ff), "CUDA Q3_K native format");
    if (f.bytes == 0) return;
    f.situ = true;
    auto l = gpu::native_expert_layout(11, down_type, n, ff); l.situ = true;
    require(l.bytes == f.bytes && l.gu_row == f.gu_row && l.down_off == f.down_off, "CPU/CUDA expert layouts agree");
    Rows gu(n, 2 * ff, rng, false, 0x1400);
    std::vector<uint8_t> blob(f.bytes);
    std::memcpy(blob.data(), gu.bytes.data(), gu.bytes.size());
    if (down_type == 11) {
        Rows down(ff, n, rng, false, 0x1400);
        std::memcpy(blob.data() + f.down_off, down.bytes.data(), down.bytes.size());
    } else {
        // Valid NVFP4: four UE4M3 scales and 32 E2M1 bytes in every 64-value block.
        for (size_t k = f.down_off; k < f.tail_off; k += 36) {
            for (int i = 0; i < 4; ++i) blob[k + i] = 0x18;
            for (int i = 4; i < 36; ++i) blob[k + i] = (uint8_t) rng();
        }
        const float tail[] = {1.f, 1.f, 0.75f, 0.f}; std::memcpy(blob.data() + f.tail_off, tail, sizeof tail);
    }
    std::vector<float> x((size_t) nt * n), h((size_t) nt * ff), ref((size_t) nt * n), dc((size_t) n * ff);
    for (auto& v : x) v = ((int) (rng() % 2001) - 1000) * 0.001f;
    require(sq::dequant_row((sq::Type) down_type, blob.data() + f.down_off, dc.data(), (int64_t) n * ff), "down reference dequant");
    for (int t = 0; t < nt; ++t) {
        for (int r = 0; r < ff; ++r) {
            double mag;
            const double g = reference(gu.values.data() + (size_t) r * n, x.data() + (size_t) t * n, n, mag);
            const double u = reference(gu.values.data() + (size_t) (ff + r) * n, x.data() + (size_t) t * n, n, mag);
            h[(size_t) t * ff + r] = (float) ((4.0 * std::tanh(g / 4.0) / (1.0 + std::exp(-g))) * (25.0 * std::tanh(u / 25.0)));
        }
        for (int r = 0; r < n; ++r) {
            double mag;
            ref[(size_t) t * n + r] = (float) (reference(dc.data() + (size_t) r * ff, h.data() + (size_t) t * ff, ff, mag) * (f.tail_off ? 0.75 : 1.0));
        }
    }
    std::vector<std::vector<uint8_t>> acts(nt, std::vector<uint8_t>(cpu::kNativeActBytes)), hq(nt, std::vector<uint8_t>(cpu::kNativeHBytes));
    std::vector<float> cpu_h((size_t) nt * ff), cpu_y((size_t) nt * n);
    const void* ap[nt]; const void* hp[nt]; float* fp[nt]; float* op[nt];
    for (int t = 0; t < nt; ++t) { cpu::native_quant_act(f, x.data() + (size_t) t * n, acts[t].data()); ap[t] = acts[t].data(); fp[t] = cpu_h.data() + (size_t) t * ff; }
    cpu::native_gu_rows(f, blob.data(), ap, nt, fp, 0, ff);
    for (int t = 0; t < nt; ++t) { cpu::native_quant_h(f, fp[t], hq[t].data()); hp[t] = hq[t].data(); op[t] = cpu_y.data() + (size_t) t * n; }
    cpu::native_down_rows(f, blob.data(), hp, nt, op, 0, n);
    const double cpu_err = rel(cpu_y, ref);
    require(cpu_err < 0.03, "CPU Q3_K SITU expert vs float reference");

    Device db(blob.size()), dx(x.size() * 4), dq((size_t) nt * (n / 32) * 36),
           dp(sizeof(unsigned long long)), ds(2 * 4), dg(4), dt(nt * 4), dd(nt * 4),
           scratch(gpu::native_expert_scratch_bytes(nt, ff)), dy(ref.size() * 4);
    db.put(blob.data(), blob.size()); dx.put(x.data(), x.size() * 4);
    const unsigned long long ptr = (unsigned long long) db.p;
    const int starts[] = {0, nt}, groups = 1, ids[] = {0, 1, 2};
    dp.put(&ptr, sizeof ptr); ds.put(starts, sizeof starts); dg.put(&groups, 4); dt.put(ids, sizeof ids); dd.put(ids, sizeof ids);
    gpu::quantize_q8_1_rows(dx.as<float>(), nt, n, dq.p, nullptr);
    std::vector<float> got(ref.size()), previous;
    for (bool old : {false, true}) {
        gpu::iq_set_old_kernels(old);
        gpu::native_expert_grouped(l, dp.as<unsigned long long>(), ds.as<int32_t>(), dg.as<int32_t>(), dd.as<int32_t>(), dt.as<int32_t>(),
                                  1, nt, dq.p, scratch.p, dy.as<float>(), nullptr);
        dy.get(got.data(), got.size() * 4);
        const double gpu_err = rel(got, ref);
        std::printf("Q3_K/%d SITU n=%d ff=%d old=%d: CPU rel %.3g CUDA rel %.3g\n", down_type, n, ff, (int) old, cpu_err, gpu_err);
        require(gpu_err < 0.03, "CUDA Q3_K SITU expert vs float reference");
        if (old) require(std::memcmp(got.data(), previous.data(), got.size() * 4) == 0, "Q3_K old/new grouped results bitwise equal");
        else previous = got;
    }
    gpu::iq_set_old_kernels(false);
}
#endif
}  // namespace

int main(int argc, char** argv) {
    bool cpu_only = false; int device = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cpu-only") == 0) cpu_only = true;
        else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = std::atoi(argv[++i]);
        else { std::printf("usage: inkling_q3_parity [--cpu-only] [--device N]\n"); return 2; }
    }
    int available = 0;
    for (sq::Arith ar : {sq::Arith::Generic, sq::Arith::Avx, sq::Arith::Avx2, sq::Arith::Neon}) {
        const sq::DotKernel* kernels[16]{};
        const int nk = sq::dot_kernels(sq::Type::Q3_K, ar, kernels, 16);
        for (int j = 0; j < nk; ++j) if (sq::isa_ok(kernels[j]->isa)) {
            std::printf("Q3_K CPU kernel: %s (%s / %s)\n", kernels[j]->name, sq::arith_name(ar), sq::isa_name(kernels[j]->isa));
            ++available;
        }
    }
    require(available > 0, "at least one runnable CPU Q3_K dot kernel");
    std::mt19937 rng(0x1a3b2026);
    Rows corners(256, 526, rng, true); Acts corner_a(256, 3, rng);
    check_cpu(corners, corner_a);
    for (int n : {512, 768, 2304}) { Rows w(n, 19, rng); Acts a(n, 3, rng); check_cpu(w, a); }
    std::printf("Q3_K CPU: all 64 sub-scales, all 8 codes, signed/zero/subnormal block scales, 1/2/3/9 blocks: %s\n", failures ? "FAIL" : "PASS");
#if STRATA_INKLING_Q3_CUDA
    if (!cpu_only) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count <= device || device < 0) { std::printf("CUDA device unavailable\n"); return failures ? 1 : 77; }
        ck(cudaSetDevice(device), "select device");
        cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop, device), "device properties");
        std::printf("CUDA device %d: %s (sm_%d%d)\n", device, prop.name, prop.major, prop.minor);
        check_gpu_rows(corners, corner_a);
        for (int n : {512, 768, 2304}) { Rows w(n, 19, rng); Acts a(n, 3, rng); check_gpu_rows(w, a); }
        require(!gpu::native_expert_supported(11, 11, 2304, 1152), "reject partial Q3_K down blocks");
        check_expert(768, 256, 11, rng);
        check_expert(2304, 1152, 40, rng);
    }
#else
    (void) cpu_only; (void) device;
#endif
    std::printf("inkling_q3_parity: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
