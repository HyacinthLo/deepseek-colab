"""Enable multi-token (ncols_dst = 2..8) bias-free gate+up+GLU MMVQ fusion for GGML_OP_MUL_MAT (OPT_SHEXP_FUSE_PATCH).
Fuses DeepSeek-V4.1's 40 shared experts (ffn_shexp) and DSpark's 3 shared experts from 3 kernels into 1 kernel
during speculative verify (ncols_dst=3), DSpark draft (ncols_dst=5), and 2-stream decode (ncols_dst=6).
Instantiates ncols_dst=2..8 fusion only for common dense quant types to keep NVCC compile time fast.
Runtime kill-switch: GGML_CUDA_MMVQ_FUSE_BATCH=0 restores ncols_dst==1 only (default 1).
usage: python3 patch_opt_shexp_fuse.py LLAMA_SRC_DIR"""
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
    'ggml/src/ggml-cuda/mmvq.cu',
    'OPT_SHEXP_FUSE_PATCH',
    [
        (
            '''template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>
static void mul_mat_vec_q_switch_fusion(''',
            '''static constexpr bool mmvq_batch_fusion_type(ggml_type type) { // OPT_SHEXP_FUSE_PATCH
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q8_0 ||
           type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K ||
           type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_MXFP4;
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>
static void mul_mat_vec_q_switch_fusion(''',
        ),
        (
            '''    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");''',
            '''    if constexpr (c_ncols_dst == 1 || mmvq_batch_fusion_type(type)) {
        if (has_fusion) {
            if constexpr (c_ncols_dst > 1) {
                GGML_ASSERT(!fusion.x_bias && !fusion.gate_bias && !fusion.x_scale && !fusion.gate_scale);
            }
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1 or batch-fusion quant types");''',
        ),
        (
            '        GGML_ASSERT(  ids || dst->ne[1] == 1);',
            '        GGML_ASSERT(  ids || dst->ne[1] == 1 || (dst->ne[1] <= MMVQ_MAX_BATCH_SIZE && !fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale)); // OPT_SHEXP_FUSE_PATCH',
        ),
    ],
)

patch_file(
    'ggml/src/ggml-cuda/ggml-cuda.cu',
    'OPT_SHEXP_FUSE_PATCH',
    [
        (
            '''static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }''',
            '''static bool ggml_cuda_mmvq_batch_fusion_supported(ggml_type type) { // OPT_SHEXP_FUSE_PATCH
    static const bool on = []() {
        const char * v = getenv("GGML_CUDA_MMVQ_FUSE_BATCH");
        return !v || atoi(v) != 0;
    }();
    if (!on) {
        return false;
    }
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q8_0 ||
           type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K ||
           type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_MXFP4;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor, bool has_bias_or_scale = true) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    // OPT_SHEXP_FUSE_PATCH: allow ncols_dst = 2..8 for bias-free/scale-free gate+up+GLU on dense MUL_MAT
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        if (has_bias_or_scale || dst->ne[1] > MMVQ_MAX_BATCH_SIZE || !ggml_cuda_mmvq_batch_fusion_supported(src0->type)) {
            return false;
        }
    }''',
        ),
        (
            '''            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);''',
            '''            if (ggml_cuda_should_fuse_mul_mat_vec_q(up, /*has_bias_or_scale=*/false)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);''',
        ),
    ],
)
