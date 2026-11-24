"""Overlap MoE DMA gate+up transfers with shared-expert (ffn_shexp) compute on DeepSeek-V4.1 (OPT_SHEXP_OVERLAP_PATCH):
- Defers the gate+up mc_dma_spin from k_moe_cache_plan into ggml_cuda_moe_dma_wait (before the first gate/up mmvq),
  controlled by LLAMA_MOE_DEVCACHE_DMA_OVERLAP (default 1).
- On DeepSeek-V4.1 decode ubatches (n_tokens <= 8), expands ffn_shexp immediately after build_moe_cache_slots (which
  triggers the host proxy DMA) and before routed gate/up/down matvecs, hiding PCIe Gen5 transfer latency behind
  dense shared-expert GPU compute with zero extra prefill VRAM.
Usage: python3 patch_opt_shexp_overlap.py LLAMA_SRC_DIR
"""
import os, sys

src = sys.argv[1]


def replace(rel, old, new, mark):
    p = os.path.join(src, rel)
    s = open(p).read()
    if mark in s:
        print('already patched', rel, mark)
        return
    assert s.count(old) == 1, (rel, old[:80])
    open(p, 'w').write(s.replace(old, new))
    print('patched', rel, mark)


# 1. ggml/src/ggml-cuda/moe-cache.cu: defer gate+up DMA wait from k_moe_cache_plan to ggml_cuda_moe_dma_wait
MC = 'ggml/src/ggml-cuda/moe-cache.cu'
replace(
    MC,
    '''struct mc_heads {
    const int4 * h[3];
    int4       * b[3];
    int64_t      n[3];
    int          n_banks;
    int          on;
};

static mc_heads mc_heads_of(const int4 * const * h, int4 * const * b, const int64_t * n, const int n_banks) {
    mc_heads r;
    for (int i = 0; i < 3; ++i) {
        r.h[i] = h[i];
        r.b[i] = b[i];
        r.n[i] = n[i];
    }
    r.n_banks = n_banks;
    r.on      = ggml_cuda_moe_copy_align();
    return r;
}''',
    '''static bool mc_dma_overlap_env() { // OPT_SHEXP_OVERLAP_PATCH
    static const bool on = getenv("LLAMA_MOE_DEVCACHE_DMA_OVERLAP") == nullptr || atoi(getenv("LLAMA_MOE_DEVCACHE_DMA_OVERLAP")) != 0;
    return on;
}

struct mc_heads {
    const int4 * h[3];
    int4       * b[3];
    int64_t      n[3];
    int          n_banks;
    int          on;
    int          wait_gu;
};

static mc_heads mc_heads_of(const int4 * const * h, int4 * const * b, const int64_t * n, const int n_banks) {
    mc_heads r;
    for (int i = 0; i < 3; ++i) {
        r.h[i] = h[i];
        r.b[i] = b[i];
        r.n[i] = n[i];
    }
    r.n_banks = n_banks;
    r.on      = ggml_cuda_moe_copy_align();
    r.wait_gu = !mc_dma_overlap_env();
    return r;
}''',
    'OPT_SHEXP_OVERLAP_PATCH',
)

replace(
    MC,
    '''    if (dseq != 0) {
        mc_dma_spin(dma_done, dseq, cnt + 3, dma_stat); // DMA_PATCH: gate and up of every miss are in VRAM before the next kernel starts
    }''',
    '''    if (dseq != 0 && hd.wait_gu) { // OPT_SHEXP_OVERLAP_WAIT
        mc_dma_spin(dma_done, dseq, cnt + 3, dma_stat); // DMA_PATCH: gate and up of every miss are in VRAM before the next kernel starts
    }''',
    'OPT_SHEXP_OVERLAP_WAIT',
)

replace(
    MC,
    '''    const ggml_tensor * down = n_banks > 1 ? lk->src[3 + n_banks - 1] : nullptr;
    if (down == nullptr || src0->data != down->data) {
        return; // gate and up were waited for in the plan kernel
    }
    mc_dma_state * s = mc_dma_get();
    uint32_t * err = (uint32_t *) lk->src[2]->data + 2*n_slots + n_dyn + 1 + 2*MC_MAX_MISS + 3;
    const ggml_cuda_kernel_launch_params launch_params(dim3(1), dim3(32), 0, ctx.stream());
    ggml_cuda_kernel_launch(k_moe_dma_wait, launch_params, (const uint32_t *) (s->seq + il), (const uint32_t *) (s->done + 2*il + 1), err, s->stat + 2);''',
    '''    const ggml_tensor * down = n_banks > 1 ? lk->src[3 + n_banks - 1] : nullptr;
    const bool is_down = down != nullptr && src0->data == down->data;
    if (!is_down && !mc_dma_overlap_env()) { // OPT_SHEXP_OVERLAP_DISPATCH
        return; // gate and up were waited for in the plan kernel
    }
    mc_dma_state * s = mc_dma_get();
    uint32_t * err = (uint32_t *) lk->src[2]->data + 2*n_slots + n_dyn + 1 + 2*MC_MAX_MISS + 3;
    const ggml_cuda_kernel_launch_params launch_params(dim3(1), dim3(32), 0, ctx.stream());
    const int pass = is_down ? 1 : 0;
    ggml_cuda_kernel_launch(k_moe_dma_wait, launch_params,
        (const uint32_t *) (s->seq + il), (const uint32_t *) (s->done + 2*il + pass), err, s->stat + 2*pass);''',
    'OPT_SHEXP_OVERLAP_DISPATCH',
)

# 2. src/llama-graph.h: add mutable overlap slot on llm_graph_context
replace(
    'src/llama-graph.h',
    '''    llama_moe_cache              * moe_cache;

    const llama_prec_policy * prec_policy;''',
    '''    llama_moe_cache              * moe_cache;
    mutable ggml_tensor          * moe_dma_overlap = nullptr; // OPT_SHEXP_OVERLAP_H

    const llama_prec_policy * prec_policy;''',
    'OPT_SHEXP_OVERLAP_H',
)

# 3. src/llama-graph.cpp: expand overlap node right after build_moe_cache_slots (which kicks off DMA)
replace(
    'src/llama-graph.cpp',
    '    ggml_tensor * slots = build_moe_cache_slots(selected_experts, up_exps, gate_exps, down_exps, gate_up_exps, il);\n',
    '    ggml_tensor * slots = build_moe_cache_slots(selected_experts, up_exps, gate_exps, down_exps, gate_up_exps, il);\n'
    '    if (moe_dma_overlap != nullptr) { // OPT_SHEXP_OVERLAP_CPP\n'
    '        if (slots != nullptr) {\n'
    '            ggml_build_forward_expand(gf, moe_dma_overlap);\n'
    '        }\n'
    '        moe_dma_overlap = nullptr;\n'
    '    }\n',
    'OPT_SHEXP_OVERLAP_CPP',
)

# 4. src/models/deepseek41.cpp: schedule ffn_shexp between k_moe_cache_plan and routed gate/up for small ubatches (<= 8 tokens)
replace(
    'src/models/deepseek41.cpp',
    '''        // the reference normalizes only when it routes to more than one expert
        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm && hparams.n_expert_used() > 1,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il, logits);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);''',
    '''        ggml_tensor * ffn_shexp = build_ffn(cur, // OPT_SHEXP_OVERLAP_DS41
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);
        if (n_tokens <= 8) {
            moe_dma_overlap = ffn_shexp;
        }

        // the reference normalizes only when it routes to more than one expert
        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm && hparams.n_expert_used() > 1,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il, logits);
        moe_dma_overlap = nullptr;
        cb(moe_out, "ffn_moe_out", il);''',
    'OPT_SHEXP_OVERLAP_DS41',
)
