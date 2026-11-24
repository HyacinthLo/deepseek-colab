"""Backport upstream PR #32943: fuse DeepSeek-V4.1 top-k MoE weight norm with additive sum bias (SCALE scale=1, bias=eps).
Eliminates 4 unfused kernels x 40 layers = 160 kernel launches per decode step.
Runtime kill-switch: GGML_CUDA_TOPK_MOE_BIAS=0 disables SCALE-bias norm fusion (default 1).
usage: python3 patch_opt_topk_fuse.py LLAMA_SRC_DIR"""
import os, sys

src = sys.argv[1]


def patch_file(rel, marker, edits):
    p = os.path.join(src, rel)
    s = open(p).read()
    if marker in s:
        print('already patched', rel, marker)
        return
    for old, new in edits:
        assert s.count(old) == 1, f'{rel}: expected 1 match for:\n{old[:180]}'
        s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('patched', rel, marker)


patch_file(
    'ggml/src/ggml-cuda/topk-moe.cuh',
    'OPT_TOPK_FUSE_PR32943',
    [
        (
            '''    bool norm{};
    bool scale{};''',
            '''    bool norm{};
    bool norm_bias{};    // OPT_TOPK_FUSE_PR32943: weight sum gets SCALE (sum + bias) instead of CLAMP
    bool scale{};''',
        ),
        (
            '''void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);''',
            '''void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             denom,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);''',
        ),
    ],
)

patch_file(
    'ggml/src/ggml-cuda/topk-moe.cu',
    'OPT_TOPK_FUSE_PR32943',
    [
        (
            '''struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
    int  n_groups;
    int  n_group_used;
};''',
            '''struct topk_moe_config {
    bool  use_sigmoid;
    bool  use_sqrt_softplus;
    bool  with_norm;
    bool  delayed_softmax;
    int   n_groups;
    int   n_group_used;
    float norm_bias; // OPT_TOPK_FUSE_PR32943
};''',
        ),
        (
            '''    if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;''',
            '''    if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val) + config.norm_bias;
        const float inv_sum = 1.0f / wt_sum;''',
        ),
        (
            '''void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {''',
            '''void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             denom,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {''',
        ),
        (
            '''    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;
    config.n_groups          = args.n_groups;
    config.n_group_used      = args.n_group_used;''',
            '''    const bool with_norm = denom != nullptr;

    // OPT_TOPK_FUSE_PR32943: sum is clamped from below (CLAMP) or gets additive bias (SCALE with scale 1)
    float clamp_val = -INFINITY;
    float norm_bias = 0.0f;
    if (denom && denom->op == GGML_OP_SCALE) {
        norm_bias = ggml_get_op_params_f32(denom, 1);
    } else if (denom) {
        clamp_val = ggml_get_op_params_f32(denom, 0);
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;
    config.n_groups          = args.n_groups;
    config.n_group_used      = args.n_group_used;
    config.norm_bias         = norm_bias;''',
        ),
    ],
)

patch_file(
    'ggml/src/ggml-cuda/ggml-cuda.cu',
    'OPT_TOPK_FUSE_PR32943',
    [
        (
            '''    args.prob_bias       = false;
    args.norm            = false;
    args.n_groups        = 0;''',
            '''    args.prob_bias       = false;
    args.norm            = false;
    args.norm_bias       = false; // OPT_TOPK_FUSE_PR32943
    args.n_groups        = 0;''',
        ),
        (
            '''    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }''',
            '''    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        // OPT_TOPK_FUSE_PR32943: check RESHAPE->SUM_ROWS->CLAMP|SCALE->DIV->RESHAPE
        static const bool allow_bias = []() {
            const char * v = getenv("GGML_CUDA_TOPK_MOE_BIAS");
            return !v || atoi(v) != 0;
        }();
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (node_idx < n_nodes && nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        const ggml_tensor * denom = node_idx < n_nodes ? nodes[node_idx] : nullptr;
        const bool is_clamp = denom && denom->op == GGML_OP_CLAMP;
        const bool is_bias  = allow_bias && denom && denom->op == GGML_OP_SCALE && ggml_get_op_params_f32(denom, 0) == 1.0f;
        if ((!is_clamp && !is_bias) || denom->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }
        args.norm_bias = is_bias;
        node_idx++;

        // DIV <- CLAMP|SCALE, RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }''',
        ),
        (
            '''            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;''',
            '''            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * denom   = nullptr;
            const ggml_tensor * scale   = nullptr;''',
        ),
        (
            '''                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }''',
            '''                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, args.norm_bias ? GGML_OP_SCALE : GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    denom = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, denom, scale, bias, args);
                    return ops.size() - 1;
                }''',
        ),
        (
            '''                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }''',
            '''                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, denom, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }''',
        ),
        (
            '''                        if (args.norm) {
                            ops.insert(ops.end(),
                                       { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                        }''',
            '''                        if (args.norm) {
                            ops.insert(ops.end(),
                                       { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, args.norm_bias ? GGML_OP_SCALE : GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                        }''',
        ),
    ],
)
