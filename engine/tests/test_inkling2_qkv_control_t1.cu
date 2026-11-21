// Standalone, no model or CMake dependency. Build with the engine's normal CUDA flags, e.g.:
// nvcc -std=c++20 -DNDEBUG -Xcompiler="/EHsc /O2 /Ob2 /MT" -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 engine/tests/test_inkling2_qkv_control_t1.cu -o test_qkv.exe
// Matches strata_inkling2 build.ninja: no --use_fast_math or --fmad override (NVCC defaults).
// Run once per GPU: test_qkv.exe DEVICE_INDEX. No timing claims; bitwise oracle only.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "../src/inkling2/inkling2_kernels.cuh"

using namespace strata::inkling2;
#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)

struct Buffer {
    float* p = nullptr;
    explicit Buffer(size_t n) { CUDA_OK(cudaMalloc(&p, n * sizeof(float))); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

static uint32_t rng = 0x725c51a3U;
static float random_value() {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (float(int(rng & 65535U) - 32768) / 32768.0f);
}

static void equal(const char* stage, const float* a, const float* b, size_t n,
                  int layer, int pattern, int alias, int replay) {
    std::vector<float> ha(n), hb(n);
    CUDA_OK(cudaMemcpy(ha.data(), a, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(hb.data(), b, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(ha[i]) || !std::isfinite(hb[i])) {
            std::fprintf(stderr, "FAIL nonfinite %s layer=%d pattern=%d alias=%d replay=%d index=%zu\n",
                         stage, layer, pattern, alias, replay, i);
            std::exit(1);
        }
    }
    if (std::memcmp(ha.data(), hb.data(), n * sizeof(float)) == 0) return;
    for (size_t i = 0; i < n; ++i) {
        uint32_t ua, ub;
        std::memcpy(&ua, &ha[i], 4); std::memcpy(&ub, &hb[i], 4);
        if (ua != ub) {
            std::fprintf(stderr, "FAIL %s layer=%d pattern=%d alias=%d replay=%d index=%zu old=%08x new=%08x\n",
                         stage, layer, pattern, alias, replay, i, ua, ub);
            std::exit(1);
        }
    }
}

int main(int argc, char** argv) {
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    CUDA_OK(cudaSetDevice(device));
    cudaDeviceProp prop{}; CUDA_OK(cudaGetDeviceProperties(&prop, device));
    std::printf("QKV F32 T1 bitwise: device=%d %s sm%d%d\n", device, prop.name, prop.major, prop.minor);
    constexpr int N = 2304, R = 64;
    constexpr float eps = 1e-6f;
    Buffer cur0(3*N), cur1(3*N), msg(3*N), wd(3*R*2*N), wg(3*N*R), wf(3*N*R);
    Buffer prep0(3*2*N), prep1(3*2*N), down0(3*R), down1(3*R), out0(3*N), out1(3*N);
    cudaStream_t stream; CUDA_OK(cudaStreamCreate(&stream));
    std::vector<float> hc(3*N), hm(3*N), hd(3*R*2*N), hg(3*N*R), hf(3*N*R);
    int cases = 0;
    for (int layer = 0; layer < 12; ++layer) {
        // Distinct F32 weights per physical core layer and per Q/K/V controller.
        for (auto& x : hd) x = random_value() * 0.02f;
        for (auto& x : hg) x = random_value() * 0.1f;
        for (auto& x : hf) x = random_value() * 0.1f;
        CUDA_OK(cudaMemcpy(wd.p, hd.data(), hd.size()*4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(wg.p, hg.data(), hg.size()*4, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(wf.p, hf.data(), hf.size()*4, cudaMemcpyHostToDevice));
        for (int pattern = 0; pattern < 3; ++pattern) {
            for (int i = 0; i < 3*N; ++i) {
                if (pattern == 0) { hc[i] = random_value(); hm[i] = random_value(); }
                // Alternating large/small signs exercises cancellation in down/up reductions.
                if (pattern == 1) { hc[i] = (i & 1 ? -1.0f : 1.0f) * (i % 4 < 2 ? 32.0f : 0.0001f); hm[i] = -hc[i] + random_value()*0.001f; }
                if (pattern == 2) { hc[i] = 0.0f; hm[i] = i % 97 == 0 ? random_value()*1e-7f : 0.0f; }
            }
            for (int alias = 0; alias < 3; ++alias) {
                // 0: distinct outputs; 1: out==cur; 2: shared cur input, distinct outputs (no residual query).
                QkvControlT1 old{}, batched{};
                for (int j = 0; j < 3; ++j) {
                    old.cur[j] = cur0.p + (alias == 2 ? 0 : j*N);
                    batched.cur[j] = cur1.p + (alias == 2 ? 0 : j*N);
                    old.msg[j] = batched.msg[j] = msg.p+j*N;
                    old.down[j] = batched.down[j] = wd.p+j*R*2*N;
                    old.gain[j] = batched.gain[j] = wg.p+j*N*R;
                    old.forget[j] = batched.forget[j] = wf.p+j*N*R;
                    old.out[j] = alias == 1 ? cur0.p+j*N : out0.p+j*N;
                    batched.out[j] = alias == 1 ? cur1.p+j*N : out1.p+j*N;
                }
                auto reset = [&] {
                    // 0xffffffff is a NaN: unwritten lanes must fail the finite checks above.
                    for (float* p : {prep0.p, prep1.p})
                        CUDA_OK(cudaMemsetAsync(p, 0xff, (size_t) 3*2*N*4, stream));
                    for (float* p : {down0.p, down1.p})
                        CUDA_OK(cudaMemsetAsync(p, 0xff, (size_t) 3*R*4, stream));
                    for (float* p : {out0.p, out1.p})
                        CUDA_OK(cudaMemsetAsync(p, 0xff, (size_t) 3*N*4, stream));
                    CUDA_OK(cudaMemcpyAsync(cur0.p, hc.data(), hc.size()*4, cudaMemcpyHostToDevice, stream));
                    CUDA_OK(cudaMemcpyAsync(cur1.p, hc.data(), hc.size()*4, cudaMemcpyHostToDevice, stream));
                    CUDA_OK(cudaMemcpyAsync(msg.p, hm.data(), hm.size()*4, cudaMemcpyHostToDevice, stream));
                    CUDA_OK(cudaStreamSynchronize(stream));
                };
                auto launch = [&] {
                    for (int j = 0; j < 3; ++j) {
                        k_control_prep<<<dim3(2,1),256,0,stream>>>(old.cur[j], old.msg[j], prep0.p+j*2*N,N,eps);
                        k_gemv_f32_cta<<<R,128,0,stream>>>(old.down[j],prep0.p+j*2*N,down0.p+j*R,2*N,R);
                        k_control_up_combine_t1<<<(N+31)/32,256,0,stream>>>(old.cur[j],old.msg[j],old.gain[j],old.forget[j],down0.p+j*R,old.out[j],R,N);
                    }
                    k_control_prep_qkv_t1<<<dim3(2,3),256,0,stream>>>(batched,prep1.p,N,eps);
                    k_control_down_qkv_t1<<<dim3(R,3),128,0,stream>>>(batched,prep1.p,down1.p,N);
                    k_control_up_combine_qkv_t1<<<dim3((N+31)/32,3),256,0,stream>>>(batched,down1.p,N);
                    CUDA_OK(cudaGetLastError());
                };
                auto check = [&](int replay) {
                    CUDA_OK(cudaStreamSynchronize(stream));
                    equal("prep",prep0.p,prep1.p,3*2*N,layer,pattern,alias,replay);
                    equal("down",down0.p,down1.p,3*R,layer,pattern,alias,replay);
                    equal("output",alias==1?cur0.p:out0.p,alias==1?cur1.p:out1.p,3*N,layer,pattern,alias,replay);
                };
                reset(); launch(); check(-1);
                cudaGraph_t graph; cudaGraphExec_t exec;
                CUDA_OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
                launch();
                CUDA_OK(cudaStreamEndCapture(stream,&graph));
                CUDA_OK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
                for (int replay=0; replay<3; ++replay) { reset(); CUDA_OK(cudaGraphLaunch(exec,stream)); check(replay); }
                CUDA_OK(cudaGraphExecDestroy(exec)); CUDA_OK(cudaGraphDestroy(graph));
                ++cases;
            }
        }
    }
    CUDA_OK(cudaStreamDestroy(stream));
    std::printf("PASS %d cases x eager + 3 graph replays; prep/down/output byte-identical\n",cases);
    return 0;
}
