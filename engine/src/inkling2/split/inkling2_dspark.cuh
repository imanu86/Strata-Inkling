// inkling2_dspark.cuh - Inkling-2-Mini DSpark speculative drafter (3-layer DFlare + recurrent Markov head).
// Included directly by inkling2_engine.cu inside namespace strata::inkling2.
#pragma once

namespace dspark_detail {

// Opt-in diagnostic artifacts only. The caller atomically claims a new
// directory, so existing captures are never overwritten. No files, copies or
// synchronizations are added when INKLING2_DSPARK_DUMP_PREFIX is unset.
struct DiagnosticCapture {
    std::string directory;
    std::FILE* manifest = nullptr;
    bool first_tensor = true;

    bool begin(const std::string& path, int64_t call, int pos, int anchor,
               int n_draft, int vocab, const std::vector<int>& caps) {
        std::error_code ec;
        if (!std::filesystem::create_directory(path, ec)) {
            std::fprintf(stderr, "strata-inkling2: DSpark dump needs a new directory %s (%s); dump skipped\n",
                         path.c_str(), ec ? ec.message().c_str() : "already exists");
            return false;
        }
        directory = path;
        manifest = std::fopen((directory + "/manifest.json").c_str(), "wb");
        if (!manifest) {
            std::fprintf(stderr, "strata-inkling2: cannot create DSpark dump manifest in %s\n", path.c_str());
            return false;
        }
        std::fprintf(manifest, "{\n\"version\":1,\"call\":%lld,\"pos\":%d,\"anchor\":%d,\"n_draft\":%d,\"vocab\":%d,\"kv_caps\":[",
                     (long long) call, pos, anchor, n_draft, vocab);
        for (size_t i = 0; i < caps.size(); ++i) std::fprintf(manifest, "%s%d", i ? "," : "", caps[i]);
        std::fprintf(manifest, "],\n\"tensors\":{\n");
        return true;
    }

    void write(const std::string& name, const void* source, const char* dtype,
               size_t item_size, std::initializer_list<int> shape, cudaStream_t st, bool on_device = true) {
        if (!manifest) return;
        size_t count = 1;
        for (int dim : shape) count *= (size_t) dim;
        std::vector<uint8_t> host;
        if (on_device) {
            host.resize(count * item_size);
            CK(cudaMemcpyAsync(host.data(), source, host.size(), cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            source = host.data();
        }
        const std::string file = name + ".bin";
        std::FILE* out = std::fopen((directory + "/" + file).c_str(), "wb");
        if (!out) { std::fprintf(stderr, "strata-inkling2: DSpark dump write failed: %s\n", file.c_str()); std::exit(1); }
        const bool ok = std::fwrite(source, item_size, count, out) == count;
        const int closed = std::fclose(out);
        if (!ok || closed != 0) { std::fprintf(stderr, "strata-inkling2: DSpark dump incomplete: %s\n", file.c_str()); std::exit(1); }
        std::fprintf(manifest, "%s\"%s\":{\"file\":\"%s\",\"dtype\":\"%s\",\"shape\":[",
                     first_tensor ? "" : ",\n", name.c_str(), file.c_str(), dtype);
        bool first_dim = true;
        for (int dim : shape) { std::fprintf(manifest, "%s%d", first_dim ? "" : ",", dim); first_dim = false; }
        std::fprintf(manifest, "]}");
        first_tensor = false;
    }

    void finish() {
        if (!manifest) return;
        std::fprintf(manifest, "\n}}\n");
        std::fclose(manifest);
        manifest = nullptr;
        std::fprintf(stderr, "strata-inkling2: DSpark diagnostic capture completed in %s\n", directory.c_str());
    }
    ~DiagnosticCapture() { if (manifest) std::fclose(manifest); }
};

__device__ __forceinline__ float bf16_to_f32_d(uint16_t u) {
    return __uint_as_float((uint32_t) u << 16);
}

__device__ __forceinline__ float warp_sum_d(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max_d(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/// Mix 4 target feature tensors via softmax(fuse_row[0..3]) and RMSNorm: grid (T), block 256.
__global__ void k_dspark_mix_features(const float* __restrict__ f0, const float* __restrict__ f1,
                                      const float* __restrict__ f2, const float* __restrict__ f3,
                                      const float* __restrict__ fuse_row, const float* __restrict__ norm_w,
                                      float* __restrict__ out, int N, float eps) {
    __shared__ float sw[4];
    __shared__ float ssum[8];
    const int t = blockIdx.x;
    if (threadIdx.x == 0) {
        float a[4] = {fuse_row[0], fuse_row[1], fuse_row[2], fuse_row[3]};
        float mx = fmaxf(fmaxf(a[0], a[1]), fmaxf(a[2], a[3]));
        float s = 0.0f;
        for (int k = 0; k < 4; ++k) { a[k] = expf(a[k] - mx); s += a[k]; }
        const float inv = 1.0f / s;
        for (int k = 0; k < 4; ++k) sw[k] = a[k] * inv;
    }
    __syncthreads();
    const float w0 = sw[0], w1 = sw[1], w2 = sw[2], w3 = sw[3];
    const size_t base = (size_t) t * N;
    float* dst = out + base;
    float acc = 0.0f;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
        const size_t i = base + c;
        const float x = w0 * f0[i] + w1 * f1[i] + w2 * f2[i] + w3 * f3[i];
        dst[c] = x;
        acc += x * x;
    }
    acc = warp_sum_d(acc);
    if ((threadIdx.x & 31) == 0) ssum[threadIdx.x >> 5] = acc;
    __syncthreads();
    float tot = 0.0f;
    if (threadIdx.x < 8) tot = ssum[threadIdx.x];
    if (threadIdx.x < 32) {
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
        if (threadIdx.x == 0) ssum[0] = rsqrtf(tot / (float) N + eps);
    }
    __syncthreads();
    const float sc = ssum[0];
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
        dst[c] = dst[c] * sc * norm_w[c];
    }
}

/// Fused per-head RMSNorm(k) + NeoX RoPE(k) + Q8 store for injected k_ctx, v_ctx.
/// Grid (n_head_kv, T), Block 64 (d = 0..63 pairs (d, d+64) for D=128).
__global__ void k_dspark_norm_rope_kv_store_q8(const float* __restrict__ K, const float* __restrict__ V,
                                               const float* __restrict__ k_norm_w,
                                               int8_t* __restrict__ kq, __half* __restrict__ ks,
                                               int8_t* __restrict__ vq, __half* __restrict__ vs,
                                               int n_head_kv, int pos0, int cap, int is_swa,
                                               float eps, float freq_base, int n_rot) {
    // Each reduction has its own storage: one warp must never overwrite a
    // previous reduction while the other warp is still reading it.
    __shared__ float s_sq[2], s_kmax[2], s_vmax[2];
    const int hkv = blockIdx.x, t = blockIdx.y;
    // Injection has no queries inside the batch. Keep only the newest ring-sized
    // suffix, so large prompt chunks never race older rows for the same SWA slot.
    if (is_swa && t < (int) gridDim.y - cap) return;
    const int d = threadIdx.x;   // 0..63
    const int warp = d >> 5;
    const int kv_dim = n_head_kv * 128;
    const size_t src_off = (size_t) t * kv_dim + hkv * 128;
    const float k0_raw = K[src_off + d];
    const float k1_raw = K[src_off + d + 64];
    const float v0 = V[src_off + d];
    const float v1 = V[src_off + d + 64];

    float sq = warp_sum_d(k0_raw * k0_raw + k1_raw * k1_raw);
    if ((d & 31) == 0) s_sq[warp] = sq;
    __syncthreads();
    const float rms = rsqrtf((s_sq[0] + s_sq[1]) / 128.0f + eps);
    float k0 = k0_raw * rms * k_norm_w[d];
    float k1 = k1_raw * rms * k_norm_w[d + 64];

    // NeoX RoPE on pairs (d, d + n_rot/2) when d < n_rot/2
    if (d * 2 < n_rot) {
        const float theta = (float) (pos0 + t) * powf(freq_base, -2.0f * (float) d / (float) n_rot);
        float sv, cv;
        __sincosf(theta, &sv, &cv);
        const float rk0 = k0 * cv - k1 * sv;
        const float rk1 = k0 * sv + k1 * cv;
        k0 = rk0;
        k1 = rk1;
    }

    float ak = warp_max_d(fmaxf(fabsf(k0), fabsf(k1)));
    float av = warp_max_d(fmaxf(fabsf(v0), fabsf(v1)));
    if ((d & 31) == 0) { s_kmax[warp] = ak; }
    __syncthreads();
    const float amax_k = fmaxf(s_kmax[0], s_kmax[1]);
    if ((d & 31) == 0) { s_vmax[warp] = av; }
    __syncthreads();
    const float amax_v = fmaxf(s_vmax[0], s_vmax[1]);

    const int pos = pos0 + t;
    const int slot = is_swa ? (pos % cap) : pos;
    const float dk = amax_k / 127.0f;
    const float dv = amax_v / 127.0f;
    const float idk = dk > 0.0f ? (127.0f / amax_k) : 0.0f;
    const float idv = dv > 0.0f ? (127.0f / amax_v) : 0.0f;

    const size_t dst_off = (size_t) slot * kv_dim + hkv * 128;
    kq[dst_off + d]      = (int8_t) __float2int_rn(k0 * idk);
    kq[dst_off + d + 64] = (int8_t) __float2int_rn(k1 * idk);
    vq[dst_off + d]      = (int8_t) __float2int_rn(v0 * idv);
    vq[dst_off + d + 64] = (int8_t) __float2int_rn(v1 * idv);
    if (d == 0) {
        ks[(size_t) slot * n_head_kv + hkv] = __float2half(dk);
        vs[(size_t) slot * n_head_kv + hkv] = __float2half(dv);
    }
}

/// Fused per-head RMSNorm + NeoX RoPE for Q (blocks 0..n_head-1) and K (blocks n_head..n_head+n_head_kv-1).
/// Grid (n_head + n_head_kv, T), Block 64 (D = 128).
__global__ void k_dspark_qk_norm_rope(float* __restrict__ Q, float* __restrict__ K,
                                      const float* __restrict__ q_norm_w, const float* __restrict__ k_norm_w,
                                      int n_head, int n_head_kv, int pos0, float eps, float freq_base, int n_rot) {
    __shared__ float s_sq[2];
    const int idx_h = blockIdx.x, t = blockIdx.y;
    const int d = threadIdx.x;
    const int warp = d >> 5;
    const bool is_k = (idx_h >= n_head);
    const int h = is_k ? (idx_h - n_head) : idx_h;
    const int nh = is_k ? n_head_kv : n_head;
    float* ptr = (is_k ? K : Q) + ((size_t) t * nh + h) * 128;
    const float* nw = is_k ? k_norm_w : q_norm_w;

    const float x0 = ptr[d];
    const float x1 = ptr[d + 64];
    float sq = warp_sum_d(x0 * x0 + x1 * x1);
    if ((d & 31) == 0) s_sq[warp] = sq;
    __syncthreads();
    const float rms = rsqrtf((s_sq[0] + s_sq[1]) / 128.0f + eps);
    float y0 = x0 * rms * nw[d];
    float y1 = x1 * rms * nw[d + 64];

    if (d * 2 < n_rot) {
        const float theta = (float) (pos0 + t) * powf(freq_base, -2.0f * (float) d / (float) n_rot);
        float sv, cv;
        __sincosf(theta, &sv, &cv);
        const float ry0 = y0 * cv - y1 * sv;
        const float ry1 = y0 * sv + y1 * cv;
        y0 = ry0;
        y1 = ry1;
    }
    ptr[d] = y0;
    ptr[d + 64] = y1;
}

/// Non-causal block-uniform attention over context Q8 KV [j0..P-1] + FP32 self KV [0..block_n-1].
/// Grid (n_head, block_n), Block 128 (4 warps).
__global__ void k_dspark_attn_q8(const float* __restrict__ Q,
                                 const int8_t* __restrict__ kq_ctx, const __half* __restrict__ ks_ctx,
                                 const int8_t* __restrict__ vq_ctx, const __half* __restrict__ vs_ctx,
                                 const float* __restrict__ K_self, const float* __restrict__ V_self,
                                 float* __restrict__ out,
                                 int n_head, int n_head_kv, int cap, int is_swa, int n_swa,
                                 int P, int block_n, float kq_scale) {
    __shared__ float sq[128];
    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_acc[4][128];

    const int h = blockIdx.x, t = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hkv = h / (n_head / n_head_kv);
    const int kv_dim = n_head_kv * 128;
    const float* qv = Q + ((size_t) t * n_head + h) * 128;
    sq[threadIdx.x] = qv[threadIdx.x];
    __syncthreads();

    const int j0 = is_swa ? max(0, P - n_swa + 1) : 0;
    float m = -1e30f;
    float l = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // 1. Context KV entries j in [j0 .. P-1]
    for (int j = j0 + warp; j < P; j += 4) {
        const int slot = is_swa ? (j % cap) : j;
        const size_t k_off = (size_t) slot * kv_dim + hkv * 128;
        const int8_t* kj = kq_ctx + k_off;
        const int8_t* vj = vq_ctx + k_off;
        const float ksc = __half2float(ks_ctx[(size_t) slot * n_head_kv + hkv]);
        const float vsc = __half2float(vs_ctx[(size_t) slot * n_head_kv + hkv]);
        float dot = sq[lane] * (float) kj[lane]
                  + sq[32 + lane] * (float) kj[32 + lane]
                  + sq[64 + lane] * (float) kj[64 + lane]
                  + sq[96 + lane] * (float) kj[96 + lane];
        dot = warp_sum_d(dot) * ksc;
        const float sc = dot * kq_scale;
        const float m_new = fmaxf(m, sc);
        const float alpha = l > 0.0f ? expf(m - m_new) : 0.0f;
        const float p = expf(sc - m_new);
        l = l * alpha + p;
        const float pv = p * vsc;
        acc0 = acc0 * alpha + pv * (float) vj[lane];
        acc1 = acc1 * alpha + pv * (float) vj[32 + lane];
        acc2 = acc2 * alpha + pv * (float) vj[64 + lane];
        acc3 = acc3 * alpha + pv * (float) vj[96 + lane];
        m = m_new;
    }

    // 2. Self block KV entries s in [0 .. block_n - 1] (non-causal)
    for (int s = warp; s < block_n; s += 4) {
        const size_t s_off = ((size_t) s * n_head_kv + hkv) * 128;
        const float* ks = K_self + s_off;
        const float* vs = V_self + s_off;
        float dot = sq[lane] * ks[lane]
                  + sq[32 + lane] * ks[32 + lane]
                  + sq[64 + lane] * ks[64 + lane]
                  + sq[96 + lane] * ks[96 + lane];
        dot = warp_sum_d(dot);
        const float sc = dot * kq_scale;
        const float m_new = fmaxf(m, sc);
        const float alpha = l > 0.0f ? expf(m - m_new) : 0.0f;
        const float p = expf(sc - m_new);
        l = l * alpha + p;
        acc0 = acc0 * alpha + p * vs[lane];
        acc1 = acc1 * alpha + p * vs[32 + lane];
        acc2 = acc2 * alpha + p * vs[64 + lane];
        acc3 = acc3 * alpha + p * vs[96 + lane];
        m = m_new;
    }

    if (lane == 0) { s_m[warp] = m; s_l[warp] = l; }
    s_acc[warp][lane]      = acc0;
    s_acc[warp][32 + lane] = acc1;
    s_acc[warp][64 + lane] = acc2;
    s_acc[warp][96 + lane] = acc3;
    __syncthreads();

    float mg = -1e30f;
    for (int w = 0; w < 4; ++w) if (s_l[w] > 0.0f && s_m[w] > mg) mg = s_m[w];
    float lg = 0.0f, og = 0.0f;
    for (int w = 0; w < 4; ++w) {
        if (s_l[w] > 0.0f) {
            const float c = expf(s_m[w] - mg);
            lg += s_l[w] * c;
            og += s_acc[w][threadIdx.x] * c;
        }
    }
    out[((size_t) t * n_head + h) * 128 + threadIdx.x] = lg > 0.0f ? (og / lg) : 0.0f;
}

__global__ void k_dspark_broadcast_tail(float* __restrict__ dst, const float* __restrict__ row, int total_rows, int N) {
    const int t = blockIdx.y + 1;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < total_rows && c < N) {
        dst[(size_t) t * N + c] = row[c];
    }
}

__global__ void k_dspark_swiglu(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

__global__ void k_dspark_add(float* __restrict__ dst, const float* __restrict__ a, const float* __restrict__ b, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] + b[i];
}

__global__ void k_dspark_add3(float* __restrict__ dst, const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] + b[i] + c[i];
}

__global__ void k_dspark_prep_joint(const float* __restrict__ state, const void* __restrict__ markov_w1, int w1_type,
                                    int prev_id, const float* __restrict__ hidden_i, const float* __restrict__ context,
                                    float* __restrict__ joint_inp, int rank, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = 2 * rank + 2 * N;
    if (i >= total) return;
    if (i < rank) {
        joint_inp[i] = state[i];
    } else if (i < 2 * rank) {
        const int c = i - rank;
        const size_t idx = (size_t) prev_id * rank + c;
        if (w1_type == 30) joint_inp[i] = bf16_to_f32_d(((const uint16_t*) markov_w1)[idx]);
        else if (w1_type == 1) joint_inp[i] = __half2float(((const __half*) markov_w1)[idx]);
        else joint_inp[i] = ((const float*) markov_w1)[idx];
    } else if (i < 2 * rank + N) {
        joint_inp[i] = hidden_i[i - 2 * rank];
    } else {
        joint_inp[i] = context[i - 2 * rank - N];
    }
}

__global__ void k_dspark_rnn_update(const float* __restrict__ joint_out, const float* __restrict__ joint_b,
                                    float* __restrict__ state, float* __restrict__ latent, int rank) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rank) return;
    const float g_raw = joint_out[i] + joint_b[i];
    const float c_raw = joint_out[rank + i] + joint_b[rank + i];
    const float l_raw = joint_out[2 * rank + i] + joint_b[2 * rank + i];
    const float gate = 1.0f / (1.0f + expf(-g_raw));
    const float cand = tanhf(c_raw);
    const float s = state[i];
    state[i] = cand + gate * (s - cand);
    latent[i] = tanhf(l_raw);
}

__global__ void k_dspark_copy_row(float* __restrict__ dst, const float* __restrict__ src, int row, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[(size_t) row * N + i];
}

__global__ void k_dspark_softmax_probs(const float* __restrict__ logits, const float* __restrict__ logits_buf_row,
                                       int V, float inv_temp, float max_val, float* __restrict__ tmp_sum) {
    // Wait: host-side sampling reuses Engine::sample_logits directly cleanly when probabilistic is active.
    (void) logits; (void) logits_buf_row; (void) V; (void) inv_temp; (void) max_val; (void) tmp_sum;
}

}  // namespace dspark_detail

struct DsparkHparams {
    int n_layer = 3;
    int block_size = 8;
    int n_embd = 2304;
    int n_ff = 9216;
    int n_head = 24;
    int n_head_kv = 8;
    int head_dim = 128;
    int n_rot = 128;
    int q_dim = 3072;
    int kv_dim = 1024;
    int n_swa = 1025;       // gguf sliding_window + 1
    int markov_rank = 256;
    int64_t vocab_padded = 201024;
    int64_t vocab_unpadded = 200058;
    float rms_eps = 1e-6f;
    float freq_base = 1e6f;
    float softcap = 30.0f;
    std::vector<uint8_t> swa_pattern;
    std::vector<int> target_layers = {5, 23, 41, 48};

    bool is_swa(int il) const {
        if ((size_t) il < swa_pattern.size()) return swa_pattern[(size_t) il] != 0;
        return il != 1;
    }
};

struct DsparkLayer {
    bool is_swa = true;
    float* dflare_norm = nullptr;
    float* attn_norm = nullptr;
    float* ffn_norm = nullptr;
    float* attn_q_norm = nullptr;
    float* attn_k_norm = nullptr;
    float* attn_k_ctx_norm = nullptr;

    DenseMat attn_k_ctx;
    DenseMat attn_v_ctx;
    DenseMat attn_q;
    DenseMat attn_k;
    DenseMat attn_v;
    DenseMat attn_output;
    DenseMat ffn_gate;
    DenseMat ffn_up;
    DenseMat ffn_down;
};

struct DsparkModel : GpuWeightStore {
    std::unique_ptr<strata::GgufModel> gguf;
    DsparkHparams hp;
    int device = -1;

    float* dspark_fuse = nullptr;       // [n_layer + 1, ntarget] = [4, 4]
    float* dspark_norm = nullptr;       // [n_embd]
    float* mask_embd = nullptr;         // [n_embd]
    float* output_norm = nullptr;       // [n_embd]
    void* markov_w1 = nullptr;          // [vocab_padded, markov_rank] on GPU
    int markov_w1_type = 30;
    DenseMat joint_proj;                // [2*rank + 2*N, 3*rank]
    float* joint_proj_b = nullptr;      // [3*rank]
    DenseMat hidden_out;                // [rank, N]
    float* hidden_out_b = nullptr;      // [N]
    std::vector<DsparkLayer> layers;

    ~DsparkModel() { free_store(); }

    const uint8_t* find_tensor(const char* name, const strata::TensorInfo** out = nullptr) const override {
        size_t shard = 0;
        const strata::TensorInfo* t = gguf->find(name, &shard);
        if (out) *out = t;
        return t ? gguf->shard(shard).tensor_data(*t) : nullptr;
    }
};

inline bool load_dspark_model(const std::string& path, const Hparams& target_hp, DsparkModel& dm, std::string& err) {
    if (dm.device < 0) CK(cudaGetDevice(&dm.device));
    DeviceScope owner(dm.device);
    const auto t0 = Clock::now();
    try {
        dm.gguf = std::make_unique<strata::GgufModel>(strata::gguf_split_paths(path));
    } catch (const std::exception& ex) {
        err = ex.what();
        return false;
    }
    const strata::GgufFile& meta = dm.gguf->meta();
    const auto* arch = meta.get("general.architecture");
    if (!arch || arch->s != "dflash") {
        err = "expected draft GGUF architecture dflash";
        return false;
    }
    auto get_i = [&](const char* k, int64_t def) -> int64_t {
        const auto* v = meta.get(k);
        return v && v->is_num() ? (int64_t) v->u : def;
    };
    auto get_f = [&](const char* k, float def) -> float {
        const auto* v = meta.get(k);
        return v ? (float) v->num() : def;
    };

    auto& hp = dm.hp;
    hp.n_layer = (int) get_i("dflash.block_count", 3);
    hp.block_size = (int) get_i("dflash.block_size", 8);
    hp.n_embd = (int) get_i("dflash.embedding_length", (int) target_hp.n_embd);
    hp.n_ff = (int) get_i("dflash.feed_forward_length", 9216);
    hp.n_head = (int) get_i("dflash.attention.head_count", 24);
    hp.n_head_kv = (int) get_i("dflash.attention.head_count_kv", 8);
    hp.head_dim = (int) get_i("dflash.attention.key_length", 128);
    hp.n_rot = (int) get_i("dflash.rope.dimension_count", hp.head_dim);
    hp.q_dim = hp.n_head * hp.head_dim;
    hp.kv_dim = hp.n_head_kv * hp.head_dim;
    hp.n_swa = (int) get_i("dflash.attention.sliding_window", 1024) + 1;   // anchor + window before it
    hp.markov_rank = (int) get_i("dflash.dspark.markov_rank", get_i("dspark.markov_rank", 256));
    hp.vocab_padded = get_i("dflash.vocab_size", target_hp.vocab_padded);
    hp.vocab_unpadded = get_i("dflash.vocab_size_unpadded", target_hp.vocab_unpadded);
    hp.rms_eps = get_f("dflash.attention.layer_norm_rms_epsilon", target_hp.rms_eps);
    hp.freq_base = get_f("dflash.rope.freq_base", 1e6f);
    hp.softcap = get_f("dflash.final_logit_softcapping", target_hp.softcap);

    if (hp.n_embd != target_hp.n_embd || hp.vocab_unpadded != target_hp.vocab_unpadded ||
        hp.vocab_padded != target_hp.vocab_padded || hp.n_head != 24 || hp.n_head_kv != 8 ||
        hp.n_ff != 9216 || hp.head_dim != 128 || hp.block_size != 8) {
        err = "dspark hyperparameters incompatible with target model";
        return false;
    }

    if (const auto* v = meta.get("dflash.attention.sliding_window_pattern")) {
        hp.swa_pattern.resize(v->items.size());
        for (size_t i = 0; i < v->items.size(); ++i) hp.swa_pattern[i] = v->items[i].u ? 1 : 0;
    }
    const auto* target_ids = meta.get("dflash.target_layers");
    if (!target_ids) target_ids = meta.get("dflash.target_layer_ids");
    if (const auto* v = target_ids) {
        hp.target_layers.clear();
        for (const auto& item : v->items) hp.target_layers.push_back((int) item.num());
    }
    if (hp.target_layers != std::vector<int>({5, 23, 41, 48}) || hp.n_layer != 3 || hp.n_rot != 128 || hp.markov_rank != 256) {
        err = "dspark requires the official 3-layer/rank-256 draft and target layers 5,23,41,48";
        return false;
    }

    const int64_t N = hp.n_embd;
    const int64_t rank = hp.markov_rank;
    dm.dspark_fuse = dm.upload_f32("dspark_fuse.weight", (size_t) (hp.n_layer + 1) * 4);
    dm.dspark_norm = dm.upload_f32("dspark_norm.weight", (size_t) N);
    dm.mask_embd = dm.upload_f32("mask_embd.weight", (size_t) N);
    dm.output_norm = dm.upload_f32("output_norm.weight", (size_t) N);

    {
        const strata::TensorInfo* wt = nullptr;
        const uint8_t* raw = dm.find_tensor("markov_w1.weight", &wt);
        if (!raw || !wt || wt->shape.size() != 2 || wt->shape[0] != (uint64_t) rank ||
            wt->shape[1] != (uint64_t) hp.vocab_padded || (wt->type != 0 && wt->type != 1 && wt->type != 30)) {
            err = "markov_w1.weight missing or unsupported type";
            return false;
        }
        dm.markov_w1_type = (int) wt->type;
        const size_t elem_sz = (wt->type == 0) ? sizeof(float) : sizeof(uint16_t);
        const size_t bytes = (size_t) wt->elements() * elem_sz;
        void* d = dm.gpu_alloc<uint8_t>(bytes);
        CK(cudaMemcpy(d, raw, bytes, cudaMemcpyHostToDevice));
        dm.markov_w1 = d;
    }

    dm.joint_proj = dm.upload_dense("joint_proj.weight", (size_t) (3 * rank), (size_t) (2 * rank + 2 * N));
    dm.joint_proj_b = dm.upload_f32("joint_proj.bias", (size_t) (3 * rank));
    dm.hidden_out = dm.upload_dense("dspark_hidden_out.weight", (size_t) N, (size_t) rank);
    dm.hidden_out_b = dm.upload_f32("dspark_hidden_out.bias", (size_t) N);

    size_t fr_vram = 0, tot_vram = 0;
    const bool pack_dspark_q8 = (cudaMemGetInfo(&fr_vram, &tot_vram) == cudaSuccess && tot_vram >= (20ull << 30));
    dm.layers.resize((size_t) hp.n_layer);
    char nm[128];
    for (int il = 0; il < hp.n_layer; ++il) {
        auto& L = dm.layers[(size_t) il];
        L.is_swa = hp.is_swa(il);
        auto tn = [&](const char* sfx) {
            std::snprintf(nm, sizeof nm, "blk.%d.%s", il, sfx);
            return nm;
        };
        L.dflare_norm = dm.upload_f32(tn("dflare_norm.weight"), (size_t) N);
        L.attn_norm = dm.upload_f32(tn("attn_norm.weight"), (size_t) N);
        L.ffn_norm = dm.upload_f32(tn("ffn_norm.weight"), (size_t) N);
        L.attn_q_norm = dm.upload_f32(tn("attn_q_norm.weight"), (size_t) hp.head_dim);
        L.attn_k_norm = dm.upload_f32(tn("attn_k_norm.weight"), (size_t) hp.head_dim);
        L.attn_k_ctx_norm = dm.upload_f32(tn("attn_k_ctx_norm.weight"), (size_t) hp.head_dim);

        L.attn_k_ctx = dm.upload_dense(tn("attn_k_ctx.weight"), (size_t) hp.kv_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_v_ctx = dm.upload_dense(tn("attn_v_ctx.weight"), (size_t) hp.kv_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_q = dm.upload_dense(tn("attn_q.weight"), (size_t) hp.q_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_k = dm.upload_dense(tn("attn_k.weight"), (size_t) hp.kv_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_v = dm.upload_dense(tn("attn_v.weight"), (size_t) hp.kv_dim, (size_t) N, false, pack_dspark_q8);
        L.attn_output = dm.upload_dense(tn("attn_output.weight"), (size_t) N, (size_t) hp.q_dim, false, pack_dspark_q8);
        L.ffn_gate = dm.upload_dense(tn("ffn_gate.weight"), (size_t) hp.n_ff, (size_t) N, false, pack_dspark_q8);
        L.ffn_up = dm.upload_dense(tn("ffn_up.weight"), (size_t) hp.n_ff, (size_t) N, false, pack_dspark_q8);
        L.ffn_down = dm.upload_dense(tn("ffn_down.weight"), (size_t) N, (size_t) hp.n_ff, false, pack_dspark_q8);
    }

    std::fprintf(stderr, "strata-inkling2: loaded dspark draft (%d layers, block=%d, rank=%d) in %.2f s\n",
                 hp.n_layer, hp.block_size, hp.markov_rank, ms_since(t0) / 1000.0);
    return true;
}

struct DsparkEngine {
    int device = -1;
    const DsparkModel* dm = nullptr;
    int ctx = 0;
    int tmax = 0;
    int pos = 0;

    // Context Q8 KV cache per layer
    std::vector<int> kv_cap;
    std::vector<int8_t*> kq, vq;
    std::vector<__half*> ks, vs;

    // Inject & draft scratch
    float* ctx_norm = nullptr;       // [tmax * N]
    float* ctx_k = nullptr;          // [tmax * kv_dim]
    float* ctx_v = nullptr;          // [tmax * kv_dim]
    float* f_rnn_buf = nullptr;      // [tmax * N]
    float* f_rnn_last = nullptr;     // [N]

    float* block_cur = nullptr;      // [block_size * N]
    float* x_norm = nullptr;         // [block_size * N]
    float* Q = nullptr;              // [block_size * q_dim]
    float* K = nullptr;              // [block_size * kv_dim]
    float* V = nullptr;              // [block_size * kv_dim]
    float* att = nullptr;            // [block_size * q_dim]
    float* wo = nullptr;             // [block_size * N]
    float* gate = nullptr;           // [block_size * n_ff]
    float* up = nullptr;             // [block_size * n_ff]

    float* rnn_state = nullptr;      // [rank]
    float* joint_inp = nullptr;      // [2*rank + 2*N]
    float* joint_out = nullptr;      // [3*rank]
    float* rnn_latent = nullptr;     // [rank]
    float* rnn_corr = nullptr;       // [N]
    float* h_pred = nullptr;         // [N]
    std::vector<float> h_anchor;     // [N]

    // Stats
    int64_t n_draft_calls = 0;
    int64_t n_drafted = 0;
    int64_t n_accepted = 0;
    double draft_ms = 0.0;
    double inject_ms = 0.0;

    std::string diagnostic_path;
    int64_t diagnostic_call = 0;
    bool diagnostic_attempted = false;
    bool diagnostic_capture_completed = false;
    bool diagnostic_verify_attempted = false;
    int diagnostic_anchor = -1;
    std::vector<float> diagnostic_features[4];
    bool diagnostic_history_enabled = false;
    int diagnostic_history_limit = 4096;
    int diagnostic_history_rows = 0;
    int diagnostic_history_generation = 0;
    // 0=complete so far, 1=gap, 2=row limit, 3=checkpoint restore, 4=end mismatch.
    int diagnostic_history_error = 0;
    std::vector<float> diagnostic_history[4];

    // Called once after the ordinary acceptance decision, just before commit.
    // These logits already live on the host; diagnostic capture adds no GPU work.
    void dump_verify_logits(int pos_pre, int K, int V, const std::vector<float>& logits, int accepted_prefix) {
        if (diagnostic_path.empty() || !diagnostic_capture_completed || diagnostic_verify_attempted ||
            n_draft_calls - 1 != diagnostic_call || K < 1 || V < 1 ||
            logits.size() != (size_t) (K + 1) * (size_t) V || accepted_prefix < 0 || accepted_prefix > K) return;
        diagnostic_verify_attempted = true;
        dspark_detail::DiagnosticCapture verify;
        if (!verify.begin(diagnostic_path + "/verify", n_draft_calls - 1, pos_pre,
                          diagnostic_anchor, K, V, kv_cap)) return;
        verify.write("target_logits", logits.data(), "f32", sizeof(float), {K + 1, V}, nullptr, false);
        verify.write("accepted_prefix", &accepted_prefix, "i32", sizeof(int32_t), {1}, nullptr, false);
        verify.finish();
    }

    ~DsparkEngine() {
        DeviceScope owner(device);
        for (auto* p : kq) cudaFree(p);
        for (auto* p : vq) cudaFree(p);
        for (auto* p : ks) cudaFree(p);
        for (auto* p : vs) cudaFree(p);
        cudaFree(ctx_norm);
        cudaFree(ctx_k);
        cudaFree(ctx_v);
        cudaFree(f_rnn_buf);
        cudaFree(f_rnn_last);
        cudaFree(block_cur);
        cudaFree(x_norm);
        cudaFree(Q);
        cudaFree(K);
        cudaFree(V);
        cudaFree(att);
        cudaFree(wo);
        cudaFree(gate);
        cudaFree(up);
        cudaFree(rnn_state);
        cudaFree(joint_inp);
        cudaFree(joint_out);
        cudaFree(rnn_latent);
        cudaFree(rnn_corr);
        cudaFree(h_pred);
    }

    void init(const DsparkModel& model, int max_ctx, int max_t) {
        device = model.device;
        DeviceScope owner(device);
        dm = &model;
        if (const char* path = std::getenv("INKLING2_DSPARK_DUMP_PREFIX")) {
            diagnostic_path = path;
            if (const char* call = std::getenv("INKLING2_DSPARK_DUMP_CALL"))
                diagnostic_call = std::max<int64_t>(0, std::strtoll(call, nullptr, 10));
        }
        if (!diagnostic_path.empty()) {
            if (const char* history = std::getenv("INKLING2_DSPARK_DUMP_HISTORY"))
                diagnostic_history_enabled = history[0] != '\0' && history[0] != '0';
            if (const char* limit = std::getenv("INKLING2_DSPARK_DUMP_HISTORY_MAX_ROWS"))
                diagnostic_history_limit = std::max(1, std::min(4096, std::atoi(limit)));
        }
        ctx = max_ctx;
        tmax = std::max(max_t, 16);
        const auto& hp = dm->hp;
        kq.resize((size_t) hp.n_layer, nullptr);
        vq.resize((size_t) hp.n_layer, nullptr);
        ks.resize((size_t) hp.n_layer, nullptr);
        vs.resize((size_t) hp.n_layer, nullptr);
        kv_cap.resize((size_t) hp.n_layer, 0);
        for (int il = 0; il < hp.n_layer; ++il) {
            const int cap = hp.is_swa(il) ? hp.n_swa : max_ctx;
            kv_cap[(size_t) il] = cap;
            CK(cudaMalloc(&kq[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(int8_t)));
            CK(cudaMalloc(&vq[(size_t) il], (size_t) cap * hp.kv_dim * sizeof(int8_t)));
            CK(cudaMalloc(&ks[(size_t) il], (size_t) cap * hp.n_head_kv * sizeof(__half)));
            CK(cudaMalloc(&vs[(size_t) il], (size_t) cap * hp.n_head_kv * sizeof(__half)));
        }
        auto d_f32 = [](float** p, size_t n) { CK(cudaMalloc(p, n * sizeof(float))); };
        const int B = hp.block_size;
        d_f32(&ctx_norm, (size_t) tmax * hp.n_embd);
        d_f32(&ctx_k, (size_t) tmax * hp.kv_dim);
        d_f32(&ctx_v, (size_t) tmax * hp.kv_dim);
        d_f32(&f_rnn_buf, (size_t) tmax * hp.n_embd);
        d_f32(&f_rnn_last, (size_t) hp.n_embd);

        d_f32(&block_cur, (size_t) B * hp.n_embd);
        d_f32(&x_norm, (size_t) B * hp.n_embd);
        d_f32(&Q, (size_t) B * hp.q_dim);
        d_f32(&K, (size_t) B * hp.kv_dim);
        d_f32(&V, (size_t) B * hp.kv_dim);
        d_f32(&att, (size_t) B * hp.q_dim);
        d_f32(&wo, (size_t) B * hp.n_embd);
        d_f32(&gate, (size_t) B * hp.n_ff);
        d_f32(&up, (size_t) B * hp.n_ff);

        d_f32(&rnn_state, (size_t) hp.markov_rank);
        d_f32(&joint_inp, (size_t) (2 * hp.markov_rank + 2 * hp.n_embd));
        d_f32(&joint_out, (size_t) (3 * hp.markov_rank));
        d_f32(&rnn_latent, (size_t) hp.markov_rank);
        d_f32(&rnn_corr, (size_t) hp.n_embd);
        d_f32(&h_pred, (size_t) hp.n_embd);
        h_anchor.resize((size_t) hp.n_embd);
        pos = 0;
    }

    void reset(cudaStream_t st) {
        DeviceScope owner(device);
        pos = 0;
        if (diagnostic_history_enabled && !diagnostic_attempted) {
            for (auto& history : diagnostic_history) history.clear();
            diagnostic_history_rows = 0;
            diagnostic_history_error = 0;
            ++diagnostic_history_generation;
        }
        CK(cudaMemsetAsync(f_rnn_last, 0, (size_t) dm->hp.n_embd * sizeof(float), st));
    }

    size_t checkpoint_bytes() const {
        const auto& hp = dm->hp;
        size_t total = (size_t) hp.n_embd * sizeof(float);   // f_rnn_last
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) continue;
            const size_t cap = (size_t) kv_cap[(size_t) il];
            total += cap * (size_t) hp.kv_dim * 2 + cap * (size_t) hp.n_head_kv * sizeof(__half) * 2;
        }
        return total;
    }

    void save_checkpoint(uint8_t* dst, cudaStream_t st) const {
        DeviceScope owner(device);
        const auto& hp = dm->hp;
        size_t off = 0;
        const size_t rnn_bytes = (size_t) hp.n_embd * sizeof(float);
        CK(cudaMemcpyAsync(dst + off, f_rnn_last, rnn_bytes, cudaMemcpyDeviceToHost, st));
        off += rnn_bytes;
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) continue;
            const size_t cap = (size_t) kv_cap[(size_t) il];
            const size_t qb = cap * (size_t) hp.kv_dim;
            const size_t sb = cap * (size_t) hp.n_head_kv * sizeof(__half);
            CK(cudaMemcpyAsync(dst + off, kq[(size_t) il], qb, cudaMemcpyDeviceToHost, st)); off += qb;
            CK(cudaMemcpyAsync(dst + off, vq[(size_t) il], qb, cudaMemcpyDeviceToHost, st)); off += qb;
            CK(cudaMemcpyAsync(dst + off, ks[(size_t) il], sb, cudaMemcpyDeviceToHost, st)); off += sb;
            CK(cudaMemcpyAsync(dst + off, vs[(size_t) il], sb, cudaMemcpyDeviceToHost, st)); off += sb;
        }
    }

    void restore_checkpoint(const uint8_t* src, int new_pos, cudaStream_t st) {
        DeviceScope owner(device);
        // The model checkpoint has no target-feature history. Conservatively
        // refuse a CPU history replay after restore; never infer missing rows.
        if (diagnostic_history_enabled && !diagnostic_attempted) {
            for (auto& history : diagnostic_history) history.clear();
            diagnostic_history_rows = 0;
            diagnostic_history_error = 3;
        }
        const auto& hp = dm->hp;
        size_t off = 0;
        const size_t rnn_bytes = (size_t) hp.n_embd * sizeof(float);
        CK(cudaMemcpyAsync(f_rnn_last, src + off, rnn_bytes, cudaMemcpyHostToDevice, st));
        off += rnn_bytes;
        for (int il = 0; il < hp.n_layer; ++il) {
            if (!hp.is_swa(il)) continue;
            const size_t cap = (size_t) kv_cap[(size_t) il];
            const size_t qb = cap * (size_t) hp.kv_dim;
            const size_t sb = cap * (size_t) hp.n_head_kv * sizeof(__half);
            CK(cudaMemcpyAsync(kq[(size_t) il], src + off, qb, cudaMemcpyHostToDevice, st)); off += qb;
            CK(cudaMemcpyAsync(vq[(size_t) il], src + off, qb, cudaMemcpyHostToDevice, st)); off += qb;
            CK(cudaMemcpyAsync(ks[(size_t) il], src + off, sb, cudaMemcpyHostToDevice, st)); off += sb;
            CK(cudaMemcpyAsync(vs[(size_t) il], src + off, sb, cudaMemcpyHostToDevice, st)); off += sb;
        }
        pos = new_pos;
    }

    /// Inject `T` committed target feature vectors (from pos0 .. pos0+T-1) into DSpark KV + update `f_rnn_last`.
    template <typename EngineT>
    void inject(EngineT& E, int pos0, int T) {
        DeviceScope owner(device);
        if (T <= 0) return;
        const auto t0 = Clock::now();
        const auto& hp = dm->hp;
        const float* f0 = E.d_tgt_feat[0];
        const float* f1 = E.d_tgt_feat[1];
        const float* f2 = E.d_tgt_feat[2];
        const float* f3 = E.d_tgt_feat[3];
        if (diagnostic_history_enabled && !diagnostic_attempted && diagnostic_history_error == 0) {
            if (pos0 != diagnostic_history_rows || pos0 < 0) {
                diagnostic_history_error = 1;
            } else if (pos0 > diagnostic_history_limit || T > diagnostic_history_limit - pos0) {
                diagnostic_history_error = 2;
            } else {
                // inject receives only the committed prefix (including anchor),
                // never the discarded suffix of the target verify batch.
                const float* features[4] = {f0, f1, f2, f3};
                const size_t offset = (size_t) pos0 * hp.n_embd;
                const size_t end = (size_t) (pos0 + T) * hp.n_embd;
                for (int i = 0; i < 4; ++i) {
                    diagnostic_history[i].resize(end);
                    CK(cudaMemcpyAsync(diagnostic_history[i].data() + offset, features[i],
                                       (size_t) T * hp.n_embd * sizeof(float), cudaMemcpyDeviceToHost, E.st));
                }
                CK(cudaStreamSynchronize(E.st));
                diagnostic_history_rows = pos0 + T;
            }
        }
        if (!diagnostic_path.empty() && !diagnostic_attempted && n_draft_calls == diagnostic_call) {
            const float* features[4] = {f0, f1, f2, f3};
            for (int i = 0; i < 4; ++i) {
                diagnostic_features[i].resize((size_t) hp.n_embd);
                CK(cudaMemcpyAsync(diagnostic_features[i].data(), features[i] + (size_t) (T - 1) * hp.n_embd,
                                   (size_t) hp.n_embd * sizeof(float), cudaMemcpyDeviceToHost, E.st));
            }
            CK(cudaStreamSynchronize(E.st));
        }
        for (int il = 0; il < hp.n_layer; ++il) {
            const auto& L = dm->layers[(size_t) il];
            dspark_detail::k_dspark_mix_features<<<(unsigned) T, 256, 0, E.st>>>(
                f0, f1, f2, f3, dm->dspark_fuse + il * 4, L.dflare_norm, ctx_norm, hp.n_embd, hp.rms_eps);
            E.prep_q8(ctx_norm, E.xq, T, hp.n_embd);
            E.project_preq(ctx_norm, T <= 8 ? E.xq : nullptr, L.attn_k_ctx, ctx_k, T, hp.kv_dim, hp.n_embd);
            E.project_preq(ctx_norm, T <= 8 ? E.xq : nullptr, L.attn_v_ctx, ctx_v, T, hp.kv_dim, hp.n_embd);
            dspark_detail::k_dspark_norm_rope_kv_store_q8<<<dim3((unsigned) hp.n_head_kv, (unsigned) T), 64, 0, E.st>>>(
                ctx_k, ctx_v, L.attn_k_ctx_norm,
                kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il],
                hp.n_head_kv, pos0, kv_cap[(size_t) il], hp.is_swa(il) ? 1 : 0,
                hp.rms_eps, hp.freq_base, hp.n_rot);
        }
        dspark_detail::k_dspark_mix_features<<<(unsigned) T, 256, 0, E.st>>>(
            f0, f1, f2, f3, dm->dspark_fuse + hp.n_layer * 4, dm->dspark_norm, f_rnn_buf, hp.n_embd, hp.rms_eps);
        CK(cudaMemcpyAsync(f_rnn_last, f_rnn_buf + (size_t) (T - 1) * hp.n_embd,
                           (size_t) hp.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, E.st));
        pos = pos0 + T;
        if (E.prof) {
            CK(cudaStreamSynchronize(E.st));
            inject_ms += ms_since(t0);
        }
    }

    /// Draft `n_draft` (1..7) speculative tokens from `anchor` (placed at `pos`, seeing context `0..pos-1`).
    /// When `out_logits != nullptr` (`n_draft` rows of `vocab_unpadded`), uses `sample_fn(row_vec)` for each draft step.
    template <typename EngineT, typename SampleFn>
    void draft(EngineT& E, int32_t anchor, int n_draft, std::vector<int32_t>& out_tokens,
               std::vector<std::vector<float>>* out_logits, SampleFn&& sample_fn, bool device_logits = false) {
        DeviceScope owner(device);
        const auto t0 = Clock::now();
        const auto& hp = dm->hp;
        const int block_n = hp.block_size;
        const int N = hp.n_embd;
        const int rank = hp.markov_rank;

        dspark_detail::DiagnosticCapture diagnostic;
        bool dump = false;
        if (!diagnostic_path.empty() && !diagnostic_attempted && n_draft_calls == diagnostic_call) {
            diagnostic_attempted = true;
            dump = diagnostic.begin(diagnostic_path, n_draft_calls, pos, anchor, n_draft,
                                    (int) hp.vocab_unpadded, kv_cap);
        }
        auto snap = [&](const std::string& name, const float* data, int rows, int cols) {
            if (dump) diagnostic.write(name, data, "f32", sizeof(float), {rows, cols}, E.st);
        };
        if (dump) {
            snap("f_rnn", f_rnn_last, 1, N);
            if (diagnostic_history_enabled) {
                const int error = diagnostic_history_error != 0 ? diagnostic_history_error :
                                  diagnostic_history_rows == pos ? 0 : 4;
                const int32_t status[4] = {diagnostic_history_generation, diagnostic_history_rows,
                                           error, diagnostic_history_limit};
                diagnostic.write("target_history_status", status, "i32", sizeof(int32_t), {4}, E.st, false);
                if (diagnostic_history_rows > 0) for (int i = 0; i < 4; ++i)
                    diagnostic.write("target_history" + std::to_string(i), diagnostic_history[i].data(),
                                     "f32", sizeof(float), {diagnostic_history_rows, N}, E.st, false);
            }
            for (int i = 0; i < 4; ++i) if (!diagnostic_features[i].empty())
                diagnostic.write("target_last" + std::to_string(i), diagnostic_features[i].data(),
                                 "f32", sizeof(float), {1, N}, E.st, false);
            for (int il = 0; il < hp.n_layer; ++il) {
                const int rows = std::min(pos, kv_cap[(size_t) il]);
                const std::string tag = "layer" + std::to_string(il);
                diagnostic.write(tag + ".cache_kq", kq[(size_t) il], "i8", sizeof(int8_t), {rows, hp.kv_dim}, E.st);
                diagnostic.write(tag + ".cache_vq", vq[(size_t) il], "i8", sizeof(int8_t), {rows, hp.kv_dim}, E.st);
                diagnostic.write(tag + ".cache_ks", ks[(size_t) il], "f16", sizeof(__half), {rows, hp.n_head_kv}, E.st);
                diagnostic.write(tag + ".cache_vs", vs[(size_t) il], "f16", sizeof(__half), {rows, hp.n_head_kv}, E.st);
            }
        }

        out_tokens.resize((size_t) n_draft);
        if (out_logits) out_logits->resize((size_t) n_draft);

        // 1. Anchor embedding (with target tok_embd_norm) + mask_embd broadcast to rows [1..block_n-1]
        E.m->dequant_tok(anchor, h_anchor.data());
        CK(cudaMemcpyAsync(x_norm, h_anchor.data(), (size_t) N * sizeof(float), cudaMemcpyHostToDevice, E.st));
        k_rms_norm<<<1, kThreads, 0, E.st>>>(x_norm, E.m->tok_embd_norm, block_cur, N, E.m->hp.rms_eps);
        dspark_detail::k_dspark_broadcast_tail<<<dim3((unsigned) ((N + 255) / 256), (unsigned) (block_n - 1)), 256, 0, E.st>>>(
            block_cur, dm->mask_embd, block_n, N);

        // 2. 3 DFlare transformer layers over block_cur [block_n, N]
        const float kq_scale = 1.0f / std::sqrt((float) hp.head_dim);
        for (int il = 0; il < hp.n_layer; ++il) {
            const auto& L = dm->layers[(size_t) il];
            const std::string tag = dump ? "layer" + std::to_string(il) : std::string();
            if (dump) snap(tag + ".input", block_cur, block_n, N);
            k_rms_norm<<<(unsigned) block_n, kThreads, 0, E.st>>>(block_cur, L.attn_norm, x_norm, N, hp.rms_eps);
            if (dump) snap(tag + ".attn_norm", x_norm, block_n, N);
            E.prep_q8(x_norm, E.xq, block_n, N);
            E.project_preq(x_norm, E.xq, L.attn_q, Q, block_n, hp.q_dim, N);
            E.project_preq(x_norm, E.xq, L.attn_k, K, block_n, hp.kv_dim, N);
            E.project_preq(x_norm, E.xq, L.attn_v, V, block_n, hp.kv_dim, N);
            if (dump) snap(tag + ".q_raw", Q, block_n, hp.q_dim);
            if (dump) snap(tag + ".k_raw", K, block_n, hp.kv_dim);
            if (dump) snap(tag + ".v_raw", V, block_n, hp.kv_dim);
            dspark_detail::k_dspark_qk_norm_rope<<<dim3((unsigned) (hp.n_head + hp.n_head_kv), (unsigned) block_n), 64, 0, E.st>>>(
                Q, K, L.attn_q_norm, L.attn_k_norm, hp.n_head, hp.n_head_kv, pos, hp.rms_eps, hp.freq_base, hp.n_rot);
            if (dump) snap(tag + ".q", Q, block_n, hp.q_dim);
            if (dump) snap(tag + ".k", K, block_n, hp.kv_dim);
            if (dump) snap(tag + ".v", V, block_n, hp.kv_dim);
            dspark_detail::k_dspark_attn_q8<<<dim3((unsigned) hp.n_head, (unsigned) block_n), 128, 0, E.st>>>(
                Q, kq[(size_t) il], ks[(size_t) il], vq[(size_t) il], vs[(size_t) il], K, V, att,
                hp.n_head, hp.n_head_kv, kv_cap[(size_t) il], hp.is_swa(il) ? 1 : 0, hp.n_swa, pos, block_n, kq_scale);
            if (dump) snap(tag + ".att", att, block_n, hp.q_dim);
            E.project(att, L.attn_output, wo, block_n, N, hp.q_dim);
            if (dump) snap(tag + ".wo", wo, block_n, N);
            dspark_detail::k_dspark_add<<<(unsigned) (((int64_t) block_n * N + 255) / 256), 256, 0, E.st>>>(
                block_cur, block_cur, wo, (int64_t) block_n * N);
            if (dump) snap(tag + ".after_attn", block_cur, block_n, N);

            k_rms_norm<<<(unsigned) block_n, kThreads, 0, E.st>>>(block_cur, L.ffn_norm, x_norm, N, hp.rms_eps);
            if (dump) snap(tag + ".ffn_norm", x_norm, block_n, N);
            E.prep_q8(x_norm, E.xq, block_n, N);
            E.project_preq(x_norm, E.xq, L.ffn_gate, gate, block_n, hp.n_ff, N);
            E.project_preq(x_norm, E.xq, L.ffn_up, up, block_n, hp.n_ff, N);
            if (dump) snap(tag + ".gate", gate, block_n, hp.n_ff);
            if (dump) snap(tag + ".up", up, block_n, hp.n_ff);
            dspark_detail::k_dspark_swiglu<<<(unsigned) (((int64_t) block_n * hp.n_ff + 255) / 256), 256, 0, E.st>>>(
                gate, up, gate, (int64_t) block_n * hp.n_ff);
            if (dump) snap(tag + ".swiglu", gate, block_n, hp.n_ff);
            E.project(gate, L.ffn_down, wo, block_n, N, hp.n_ff);
            if (dump) snap(tag + ".down", wo, block_n, N);
            dspark_detail::k_dspark_add<<<(unsigned) (((int64_t) block_n * N + 255) / 256), 256, 0, E.st>>>(
                block_cur, block_cur, wo, (int64_t) block_n * N);
            if (dump) snap(tag + ".output", block_cur, block_n, N);
        }
        k_rms_norm<<<(unsigned) block_n, kThreads, 0, E.st>>>(block_cur, dm->output_norm, block_cur, N, hp.rms_eps);
        if (dump) snap("hidden", block_cur, block_n, N);

        // 3. Recurrent Markov chain for i = 1 .. n_draft
        CK(cudaMemsetAsync(rnn_state, 0, (size_t) rank * sizeof(float), E.st));
        int32_t prev_id = anchor;
        const int joint_tot = 2 * rank + 2 * N;
        for (int i = 0; i < n_draft; ++i) {
            const std::string tag = dump ? "rnn" + std::to_string(i) : std::string();
            const float* hi = block_cur + (size_t) (i + 1) * N;
            dspark_detail::k_dspark_prep_joint<<<(unsigned) ((joint_tot + 255) / 256), 256, 0, E.st>>>(
                rnn_state, dm->markov_w1, dm->markov_w1_type, prev_id, hi, f_rnn_last, joint_inp, rank, N);
            if (dump) snap(tag + ".joint_input", joint_inp, 1, joint_tot);
            E.project(joint_inp, dm->joint_proj, joint_out, 1, 3 * rank, joint_tot);
            if (dump) snap(tag + ".joint_out", joint_out, 1, 3 * rank);
            dspark_detail::k_dspark_rnn_update<<<(unsigned) ((rank + 255) / 256), 256, 0, E.st>>>(
                joint_out, dm->joint_proj_b, rnn_state, rnn_latent, rank);
            if (dump) snap(tag + ".state", rnn_state, 1, rank);
            if (dump) snap(tag + ".latent", rnn_latent, 1, rank);
            E.project(rnn_latent, dm->hidden_out, rnn_corr, 1, N, rank);
            if (dump) snap(tag + ".correction", rnn_corr, 1, N);
            dspark_detail::k_dspark_add3<<<(unsigned) ((N + 255) / 256), 256, 0, E.st>>>(
                h_pred, hi, rnn_corr, dm->hidden_out_b, N);
            if (dump) snap(tag + ".prediction", h_pred, 1, N);
            E.project(h_pred, E.m->output_weight, E.logits, 1, hp.vocab_unpadded, N);
            if (out_logits) {
                k_softcap<<<(unsigned) ((hp.vocab_unpadded + 255) / 256), 256, 0, E.st>>>(
                    E.logits, hp.vocab_unpadded, hp.softcap);
                if (dump) snap(tag + ".logits", E.logits, 1, (int) hp.vocab_unpadded);
                auto& row_lg = (*out_logits)[(size_t) i];
                if (!device_logits) {
                    row_lg.resize((size_t) hp.vocab_unpadded);
                    CK(cudaMemcpyAsync(row_lg.data(), E.logits, (size_t) hp.vocab_unpadded * sizeof(float),
                                       cudaMemcpyDeviceToHost, E.st));
                    CK(cudaStreamSynchronize(E.st));
                }
                prev_id = sample_fn(i, row_lg);
            } else {
                if (dump) snap(tag + ".logits_raw", E.logits, 1, (int) hp.vocab_unpadded);
                k_argmax<<<1, 256, 0, E.st>>>(E.logits, E.pick_d, (int) hp.vocab_unpadded);
                int32_t next_id = 0;
                CK(cudaMemcpyAsync(&next_id, E.pick_d, sizeof(int32_t), cudaMemcpyDeviceToHost, E.st));
                CK(cudaStreamSynchronize(E.st));
                prev_id = next_id;
            }
            out_tokens[(size_t) i] = prev_id;
        }
        if (dump) {
            diagnostic.write("proposed_ids", out_tokens.data(), "i32", sizeof(int32_t), {n_draft}, E.st, false);
            diagnostic.finish();
            diagnostic_capture_completed = true;
            diagnostic_anchor = anchor;
        }
        ++n_draft_calls;
        n_drafted += n_draft;
        if (E.prof) draft_ms += ms_since(t0);
    }
};
