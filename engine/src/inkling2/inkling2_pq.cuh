#pragma once

// Full-vocabulary temperature-one sampling. Softcapped logits are bounded
// by an explicitly checked positive cap <=80; expf remains finite.
// Accumulate in FP64 and draw one uniform from the existing host RNG.
__global__ void full_vocab_tile_mass(const float *logits, double *mass, int vocab) {
    __shared__ double partial[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    partial[threadIdx.x] = i < vocab ? (double)expf(logits[i]) : 0.0;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (threadIdx.x < step)
            partial[threadIdx.x] += partial[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        mass[blockIdx.x] = partial[0];
}

__global__ void full_vocab_pick(const float *logits, const double *mass, int32_t *pick, int vocab, int tiles, double u) {
    // One thread selects a tile. Only its 256 probabilities are revisited.
    double total = 0.0;
    for (int t = 0; t < tiles; ++t)
        total += mass[t];
    double cut = u * total, before = 0.0;
    int t = 0;
    for (; t + 1 < tiles && before + mass[t] < cut; ++t)
        before += mass[t];
    int begin = t * 256, end = min(begin + 256, vocab);
    for (int i = begin; i < end; ++i) {
        before += (double)expf(logits[i]);
        if (before >= cut) {
            *pick = i;
            return;
        }
    }
    *pick = end - 1;
}

// Device-side masses for the full-vocabulary DSpark proposal and target rows.
// This file is included after the full-vocabulary categorical kernels.
struct PqTotals {
    double p;
    double q;
    double acceptance;
};

__global__ void pq_row_totals(const double* p_mass, const double* q_mass, const float* p,
                              const float* q, const int32_t* ids, int vocab, int tiles, PqTotals* totals) {
    int row = blockIdx.x;
    __shared__ double ps[256], qs[256];
    double a = 0.0, b = 0.0;
    for (int tile = threadIdx.x; tile < tiles; tile += blockDim.x) {
        a += p_mass[row * tiles + tile];
        b += q_mass[row * tiles + tile];
    }
    ps[threadIdx.x] = a;
    qs[threadIdx.x] = b;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (threadIdx.x < step) {
            ps[threadIdx.x] += ps[threadIdx.x + step];
            qs[threadIdx.x] += qs[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int index = row * vocab + ids[row];
        double px = (double)expf(p[index]) / ps[0];
        double qx = (double)expf(q[index]) / qs[0];
        totals[row] = {ps[0], qs[0], fmin(1.0, px / qx)};
    }
}

__global__ void pq_residual_mass(const float* p, const float* q, double p_sum, double q_sum,
                                 double* residual, double* mass, int vocab) {
    __shared__ double partial[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double w = 0.0;
    if (i < vocab) {
        w = fmax(0.0, (double)expf(p[i]) / p_sum - (double)expf(q[i]) / q_sum);
        residual[i] = w;
    }
    partial[threadIdx.x] = w;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (threadIdx.x < step) partial[threadIdx.x] += partial[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) mass[blockIdx.x] = partial[0];
}

__global__ void pq_pick_residual(const double* residual, const double* mass, int32_t* id,
                                 int vocab, int tiles, double u) {
    double sum = 0.0;
    for (int t = 0; t < tiles; ++t) sum += mass[t];
    if (!(sum > 0.0)) { *id = -1; return; }
    double cut = u * sum, before = 0.0;
    int tile = 0;
    while (tile + 1 < tiles && before + mass[tile] <= cut) before += mass[tile++];
    int end = min((tile + 1) * 256, vocab);
    int last_positive = -1;
    for (int i = tile * 256; i < end; ++i) {
        if (residual[i] > 0.0) last_positive = i;
        before += residual[i];
        if (before > cut) { *id = i; return; }
    }
    *id = last_positive;
}

struct GpuPqSampler {
    int vocab, tiles;
    float* proposal = nullptr;
    double *q_mass = nullptr, *p_mass = nullptr, *residual = nullptr, *residual_mass = nullptr;
    int32_t *proposal_ids = nullptr, *pick = nullptr;
    PqTotals* totals = nullptr;
    PqTotals host[7]{};

    GpuPqSampler(int v, int k) : vocab(v), tiles((v + 255) / 256) {
        CK(cudaMalloc(&proposal, (size_t)k * v * sizeof(float)));
        CK(cudaMalloc(&q_mass, (size_t)k * tiles * sizeof(double)));
        CK(cudaMalloc(&p_mass, (size_t)(k + 1) * tiles * sizeof(double)));
        CK(cudaMalloc(&proposal_ids, k * sizeof(int32_t)));
        CK(cudaMalloc(&totals, k * sizeof(PqTotals)));
        CK(cudaMalloc(&residual, v * sizeof(double)));
        CK(cudaMalloc(&residual_mass, tiles * sizeof(double)));
        CK(cudaMalloc(&pick, sizeof(int32_t)));
    }
    ~GpuPqSampler() {
        cudaFree(pick); cudaFree(residual_mass); cudaFree(residual);
        cudaFree(totals); cudaFree(proposal_ids); cudaFree(p_mass); cudaFree(q_mass); cudaFree(proposal);
    }
    int read_id(const int32_t* ptr, cudaStream_t st) {
        int32_t id;
        CK(cudaMemcpyAsync(&id, ptr, sizeof(id), cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return id;
    }
    int sample_proposal(int row, const float* logits, double u, cudaStream_t st) {
        float* q = proposal + (size_t)row * vocab;
        double* mass = q_mass + (size_t)row * tiles;
        CK(cudaMemcpyAsync(q, logits, vocab * sizeof(float), cudaMemcpyDeviceToDevice, st));
        full_vocab_tile_mass<<<tiles, 256, 0, st>>>(q, mass, vocab);
        full_vocab_pick<<<1, 1, 0, st>>>(q, mass, proposal_ids + row, vocab, tiles, u);
        return read_id(proposal_ids + row, st);
    }
    void verify(const float* logits, int k, cudaStream_t st) {
        for (int row = 0; row <= k; ++row)
            full_vocab_tile_mass<<<tiles, 256, 0, st>>>(logits + (size_t)row * vocab, p_mass + (size_t)row * tiles, vocab);
        pq_row_totals<<<k, 256, 0, st>>>(p_mass, q_mass, logits, proposal, proposal_ids, vocab, tiles, totals);
        CK(cudaMemcpyAsync(host, totals, k * sizeof(PqTotals), cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
    int sample_target(const float* logits, int row, double u, cudaStream_t st) {
        full_vocab_pick<<<1, 1, 0, st>>>(logits + (size_t)row * vocab, p_mass + (size_t)row * tiles, pick, vocab, tiles, u);
        return read_id(pick, st);
    }
    int sample_residual(const float* logits, int row, double u, cudaStream_t st) {
        size_t start = (size_t)row * vocab;
        pq_residual_mass<<<tiles, 256, 0, st>>>(logits + start, proposal + start, host[row].p, host[row].q,
                                                residual, residual_mass, vocab);
        pq_pick_residual<<<1, 1, 0, st>>>(residual, residual_mass, pick, vocab, tiles, u);
        int id = read_id(pick, st);
        return id >= 0 ? id : sample_target(logits, row, u, st);
    }
};
