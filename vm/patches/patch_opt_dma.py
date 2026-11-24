"""MoE DMA engine optimizations (OPT_DMA_PATCH, after patch_dmaalign.py):
1. Raise MC_DMA_MAX_IDS from 24 to 48 (8 tokens x 6 experts, matching MMVQ_MAX_BATCH_SIZE=8 and patch_hcfuse <=8)
   so 2-stream concurrent decode (6 tokens = 36 ids) and draft n_max=3 (up to 8 tokens = 48 ids) stay on the hardware DMA engine.
2. Submit each pass (gate+up, down) via a single CUDA 12.8+/13.0 cudaMemcpyBatchAsync call (LLAMA_MOE_DEVCACHE_DMA_BATCH, default 1)
   instead of up to 3*nm separate cudaMemcpyAsync driver calls per layer.
3. Start proxy queue scan at the next expected layer (q+1) % nq instead of scanning from 0 every time.
Usage: python3 patch_opt_dma.py LLAMA_SRC_DIR
"""
import os, sys

src = sys.argv[1]
MC = 'ggml/src/ggml-cuda/moe-cache.cu'


def replace(rel, old, new, mark):
    p = os.path.join(src, rel)
    s = open(p).read()
    if mark in s:
        print('already patched', rel, mark)
        return
    assert s.count(old) == 1, (rel, old[:80])
    open(p, 'w').write(s.replace(old, new))
    print('patched', rel, mark)


# 1. Expand MC_DMA_MAX_IDS from 24 (4 tokens) to 48 (8 tokens)
replace(
    MC,
    '#define MC_DMA_MAX_IDS 24\n',
    '#define MC_DMA_MAX_IDS 48 // OPT_DMA_PATCH: 8 tokens x 6 experts (matches MMVQ_MAX_BATCH_SIZE=8)\n',
    'OPT_DMA_PATCH: 8 tokens',
)

# 2. Use cudaMemcpyBatchAsync (CUDA >= 12.8) + next-layer-first scan in mc_dma_state::run()
OLD_RUN = '''    void run() {
        cudaSetDevice(device);
        const bool align = ggml_cuda_moe_copy_align() != 0; // DMAALIGN_PATCH
        if (align) {
            fprintf(stderr, "moe-cache DMA: the plan kernel copies the bytes before the first 128-byte line of each source, the DMA the aligned rest\\n");
        }
        std::vector<uint32_t> seen(MC_DMA_LAYERS, 0);
        auto last = std::chrono::steady_clock::now();
        while (!stop.load(std::memory_order_relaxed)) {
            bool busy = false;
            const int nq = n_q.load(std::memory_order_acquire);
            for (int q = 0; q < nq; ++q) {
                const mc_dma_desc & d = desc[q];
                const volatile int32_t * p = hq + (size_t) q*MC_HQ_STRIDE;
                const uint32_t s = (uint32_t) p[0];
                if (s == seen[q] || !d.ready.load(std::memory_order_acquire)) {
                    continue;
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                seen[q] = s;
                busy    = true;
                n_pub++;
                const int nm    = std::min((int) p[1], MC_DMA_MAX_IDS);
                const auto t_seen = std::chrono::steady_clock::now();
                const int nb_gu = d.n_banks > 1 ? d.n_banks - 1 : d.n_banks; // gate and up (or gate_up) first, down last
                for (int pass = 0; pass < 2; ++pass) {
                    const int b0 = pass == 0 ? 0 : nb_gu;
                    const int b1 = pass == 0 ? nb_gu : d.n_banks;
                    for (int m = 0; m < nm; ++m) {
                        const int e  = p[2 + 2*m];
                        const int sl = p[3 + 2*m];
                        if (e < 0 || sl < 0) {
                            continue;
                        }
                        for (int b = b0; b < b1; ++b) {
                            const char * x    = d.h[b] + (size_t) e*d.n[b];
                            const size_t head = align ? mc_head_bytes(x, d.n[b]) : 0; // DMAALIGN_PATCH
                            n_fail  += cudaMemcpyAsync(d.b[b] + (size_t) sl*d.n[b] + head, x + head, d.n[b] - head, cudaMemcpyHostToDevice, stream) != cudaSuccess;
                            n_copy  += 1;
                            n_bytes += d.n[b];
                        }
                    }
                    val[2*q + pass] = s;
                    n_fail += cudaMemcpyAsync(done + 2*q + pass, val + 2*q + pass, sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess;
                }'''

NEW_RUN = '''    void run() {
        cudaSetDevice(device);
        const bool align = ggml_cuda_moe_copy_align() != 0; // DMAALIGN_PATCH
        const char * bv = getenv("LLAMA_MOE_DEVCACHE_DMA_BATCH");
        const bool use_batch = bv == nullptr || atoi(bv) != 0; // OPT_DMA_BATCH_PATCH
        if (align) {
            fprintf(stderr, "moe-cache DMA: the plan kernel copies the bytes before the first 128-byte line of each source, the DMA the aligned rest\\n");
        }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
        cudaMemcpyAttributes batch_attr = {};
        batch_attr.srcAccessOrder   = cudaMemcpySrcAccessOrderStream;
        batch_attr.srcLocHint.type  = cudaMemLocationTypeHost;
        batch_attr.srcLocHint.id    = 0;
        batch_attr.dstLocHint.type  = cudaMemLocationTypeDevice;
        batch_attr.dstLocHint.id    = device;
        batch_attr.flags            = 0;
        size_t batch_attr_idx       = 0;
        void       * b_dst[3*MC_DMA_MAX_IDS];
        const void * b_src[3*MC_DMA_MAX_IDS];
        size_t       b_len[3*MC_DMA_MAX_IDS];
        fprintf(stderr, "moe-cache DMA: OPT_DMA_PATCH max_ids=%d batch_api=%s\\n", MC_DMA_MAX_IDS, use_batch ? "cudaMemcpyBatchAsync" : "off");
#else
        (void) use_batch;
        fprintf(stderr, "moe-cache DMA: OPT_DMA_PATCH max_ids=%d batch_api=unavailable\\n", MC_DMA_MAX_IDS);
#endif
        std::vector<uint32_t> seen(MC_DMA_LAYERS, 0);
        int start_q = 0;
        auto last = std::chrono::steady_clock::now();
        while (!stop.load(std::memory_order_relaxed)) {
            bool busy = false;
            const int nq = n_q.load(std::memory_order_acquire);
            for (int step = 0; step < nq; ++step) {
                const int q = (start_q + step) % nq;
                const mc_dma_desc & d = desc[q];
                const volatile int32_t * p = hq + (size_t) q*MC_HQ_STRIDE;
                const uint32_t s = (uint32_t) p[0];
                if (s == seen[q] || !d.ready.load(std::memory_order_acquire)) {
                    continue;
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                seen[q] = s;
                start_q = (q + 1) % nq;
                busy    = true;
                n_pub++;
                const int nm    = std::min((int) p[1], MC_DMA_MAX_IDS);
                const auto t_seen = std::chrono::steady_clock::now();
                const int nb_gu = d.n_banks > 1 ? d.n_banks - 1 : d.n_banks; // gate and up (or gate_up) first, down last
                for (int pass = 0; pass < 2; ++pass) {
                    const int b0 = pass == 0 ? 0 : nb_gu;
                    const int b1 = pass == 0 ? nb_gu : d.n_banks;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
                    if (use_batch) {
                        size_t nb = 0;
                        for (int m = 0; m < nm; ++m) {
                            const int e  = p[2 + 2*m];
                            const int sl = p[3 + 2*m];
                            if (e < 0 || sl < 0) {
                                continue;
                            }
                            for (int b = b0; b < b1; ++b) {
                                const char * x    = d.h[b] + (size_t) e*d.n[b];
                                const size_t head = align ? mc_head_bytes(x, d.n[b]) : 0;
                                const size_t rem  = d.n[b] - head;
                                if (rem > 0) {
                                    b_dst[nb] = d.b[b] + (size_t) sl*d.n[b] + head;
                                    b_src[nb] = x + head;
                                    b_len[nb] = rem;
                                    nb++;
                                }
                                n_copy  += 1;
                                n_bytes += d.n[b];
                            }
                        }
                        if (nb > 1) {
                            n_fail += cudaMemcpyBatchAsync(b_dst, b_src, b_len, nb, &batch_attr, &batch_attr_idx, 1, stream) != cudaSuccess;
                        } else if (nb == 1) {
                            n_fail += cudaMemcpyAsync(b_dst[0], b_src[0], b_len[0], cudaMemcpyHostToDevice, stream) != cudaSuccess;
                        }
                    } else
#endif
                    {
                        for (int m = 0; m < nm; ++m) {
                            const int e  = p[2 + 2*m];
                            const int sl = p[3 + 2*m];
                            if (e < 0 || sl < 0) {
                                continue;
                            }
                            for (int b = b0; b < b1; ++b) {
                                const char * x    = d.h[b] + (size_t) e*d.n[b];
                                const size_t head = align ? mc_head_bytes(x, d.n[b]) : 0; // DMAALIGN_PATCH
                                n_fail  += cudaMemcpyAsync(d.b[b] + (size_t) sl*d.n[b] + head, x + head, d.n[b] - head, cudaMemcpyHostToDevice, stream) != cudaSuccess;
                                n_copy  += 1;
                                n_bytes += d.n[b];
                            }
                        }
                    }
                    val[2*q + pass] = s;
                    n_fail += cudaMemcpyAsync(done + 2*q + pass, val + 2*q + pass, sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess;
                }'''

replace(MC, OLD_RUN, NEW_RUN, 'OPT_DMA_BATCH_PATCH')
