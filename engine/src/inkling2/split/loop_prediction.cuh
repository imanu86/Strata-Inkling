// Inside Engine: three 12-layer visits of Inkling 2 Mini, greedy or sampled
// autoregressive decode. The caller always prefills with the original model.
bool predict_during_decode = false;
bool predicted_boundary = false;
int predict_after = -1;
float *loop_weights = nullptr, *loop_bias = nullptr, *loop_input = nullptr;
float *loop_prediction = nullptr, *loop_residual = nullptr, *loop_saved_cur = nullptr;
cudaGraphExec_t loop_graphs[4]{};
ForwardState loop_meta[4];

bool load_loop_predictor(int cut, const std::string& file) {
    DeviceScope owner(device);
    const int n = (int)m->hp.n_embd;
    const int d = 2 * n;
    const size_t weights = (size_t)d * d;
    std::ifstream in(std::filesystem::u8path(file), std::ios::binary | std::ios::ate);
    if (!core_all_gpu || n != 2304 || !in || in.tellg() != (std::streamoff)((weights + d) * sizeof(float))) {
        std::fprintf(stderr, "Inkling loop predictor requires resident core experts and a matching FP32 matrix: %s\n",
                     file.c_str());
        return false;
    }
    in.seekg(0);
    std::vector<float> host(weights + d);
    if (!in.read(reinterpret_cast<char*>(host.data()), host.size() * sizeof(float)) ||
        !std::all_of(host.begin(), host.end(), [](float x) { return std::isfinite(x); })) {
        std::fprintf(stderr, "Could not read finite loop prediction weights\n");
        return false;
    }
    predict_after = cut;
    CK(cudaMalloc(&loop_weights, weights * sizeof(float)));
    CK(cudaMalloc(&loop_bias, d * sizeof(float)));
    CK(cudaMemcpy(loop_weights, host.data(), weights * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(loop_bias, host.data() + weights, d * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&loop_input, d * sizeof(float)));
    CK(cudaMalloc(&loop_prediction, d * sizeof(float)));
    CK(cudaMalloc(&loop_residual, d * sizeof(float)));
    CK(cudaMalloc(&loop_saved_cur, n * sizeof(float)));
    clear_loop_feedback();
    std::fprintf(stderr, "Inkling approximate loop predictor: cut %d/12, weights %s\n", cut, file.c_str());
    return true;
}

void clear_loop_feedback() {
    DeviceScope owner(device);
    CK(cudaMemsetAsync(loop_residual, 0, 2 * m->hp.n_embd * sizeof(float), st));
}

// Each slot records one fixed layer range and the associated residual-buffer
// metadata. Token positions are read through pos0_d when the graph runs.
void predicted_range(ForwardState& state, int slot, int begin, int end) {
    if (begin == end)
        return;
    auto& graph = loop_graphs[slot];
    if (!graph) {
        cudaGraph_t capture{};
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        run_layers(state, begin, end);
        CK(cudaStreamEndCapture(st, &capture));
        CK(cudaGraphInstantiate(&graph, capture, nullptr, nullptr, 0));
        CK(cudaGraphDestroy(capture));
        loop_meta[slot] = state;
    } else {
        state = loop_meta[slot];
        hy_gpu += (int64_t)(end - begin) * m->hp.n_expert_used;
    }
    CK(cudaGraphLaunch(graph, st));
}

void forward_predicted_core(ForwardState& state, Engine& outer, ForwardState& output) {
    DeviceScope owner(device);
    const int n = (int)m->hp.n_embd, d = 2 * n;
    const size_t row = n * sizeof(float);
    predicted_range(state, 0, 6, 6 + predict_after);
    if (predict_after == 0)
        CK(cudaMemcpyAsync(prelude_cur_buf, cur, row, cudaMemcpyDeviceToDevice, st));
    CK(cudaMemcpyAsync(loop_saved_cur, cur, row, cudaMemcpyDeviceToDevice, st));
    CK(cudaMemcpyAsync(loop_input, state.norm_blocks.back(), row, cudaMemcpyDeviceToDevice, st));
    k_rms_norm<<<1, kThreads, 0, st>>>(cur, nullptr, loop_input + n, n, m->hp.rms_eps);
    const float one = 1, zero = 0;
    CKB(cublasSgemv(blas, CUBLAS_OP_N, d, d, &one, loop_weights, d, loop_input, 1, &zero, loop_prediction, 1));
    add_loop_bias<<<(d + 255) / 256, 256, 0, st>>>(loop_prediction, loop_bias, loop_residual, d);
    // Visit 3 reads the prelude, the predicted residual block and normalized hidden.
    // Slot 3 in norm_blocks is skipped by run_layers when building the bank.
    ForwardState last;
    last.T = 1;
    last.next_blk_buf = 5;
    last.prelude_norm_blocks = {norm_blk_buf[0]};
    last.norm_blocks = {norm_blk_buf[0], norm_blk_buf[3], norm_blk_buf[4]};
    CK(cudaMemcpyAsync(norm_blk_buf[4], loop_prediction, row, cudaMemcpyDeviceToDevice, st));
    CK(cudaMemcpyAsync(cur, loop_prediction + n, row, cudaMemcpyDeviceToDevice, st));
    predicted_boundary = true;
    predicted_range(last, 1, 30, 42);
    predicted_boundary = false;
    handoff(last, outer, output);
    // Cur and the first visit's blocks remain private to this GPU. Carry out
    // the unfinished exact work while the other GPU consumes the returned state.
    CK(cudaMemcpyAsync(cur, loop_saved_cur, row, cudaMemcpyDeviceToDevice, st));
    predicted_range(state, 2, 6 + predict_after, 18);
    predicted_range(state, 3, 18, 30);
    CK(cudaMemcpyAsync(loop_input, norm_blk_buf[4], row, cudaMemcpyDeviceToDevice, st));
    k_rms_norm<<<1, kThreads, 0, st>>>(cur, nullptr, loop_input + n, n, m->hp.rms_eps);
    update_loop_residual<<<(d + 255) / 256, 256, 0, st>>>(loop_residual, loop_input, loop_prediction, d);
}
