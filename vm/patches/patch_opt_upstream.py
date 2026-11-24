"""Upstream CUDA graph & lightning-indexer micro-optimizations backported to b11963:
- PR #32004: skip O(nodes) ggml_cuda_graph_check_compability walk when graph->uid already matches (5,651 nodes on DeepSeek-V4.1)
- PR #30662: avoid evicting the graph about to be used and grant a 10s grace window after idle pauses
- PR #31803 (adapted to b11963 WMMA kernel): pad qk_shared inner stride by +8 floats to eliminate shared-memory bank conflicts
Usage: python3 patch_opt_upstream.py LLAMA_SRC_DIR
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


# 1. PR #32004: skip graph compatibility scan on reused CUDA graphs
replace(
    'ggml/src/ggml-cuda/ggml-cuda.cu',
    '        const bool graph_compatible = ggml_cuda_graph_check_compability(cgraph);\n',
    '        // OPT_UPSTREAM_PR32004: graph->uid is only set for a compatible graph, so a reused graph needs no re-check\n'
    '        const bool graph_reused     = cgraph->uid != 0 && cgraph->uid == graph->uid;\n'
    '        const bool graph_compatible = graph_reused || ggml_cuda_graph_check_compability(cgraph);\n',
    'OPT_UPSTREAM_PR32004',
)

# 2. PR #30662: prevent premature eviction of graphs about to be used
replace(
    'ggml/src/ggml-cuda/common.cuh',
    '''    int64_t last_graph_eviction_sweep = 0;

    ggml_cuda_graph * cuda_graph(const void * first_node_ptr) {
        const int64_t time_now = ggml_time_us();

        // sweep every 5s, evicting cuda graphs unused for >=10s
        if (time_now - last_graph_eviction_sweep >= 5'000'000) {
            last_graph_eviction_sweep = time_now;
            for (auto it = cuda_graphs.begin(); it != cuda_graphs.end(); ) {
                if (time_now - it->second->last_used_time >= 10'000'000) {
                    it = cuda_graphs.erase(it);
                } else {
                    ++it;
                }
            }
        }''',
    '''    int64_t next_graph_eviction_time = 0; // OPT_UPSTREAM_PR30662
    int64_t last_graph_use_time = 0;

    ggml_cuda_graph * cuda_graph(const void * first_node_ptr) {
        const int64_t time_now = ggml_time_us();

        // Give all cached graphs 10s to be reused after an idle period
        if (!cuda_graphs.empty() && time_now - last_graph_use_time >= 10'000'000) {
            next_graph_eviction_time = time_now + 10'000'000;
        }
        last_graph_use_time = time_now;

        // Sweep every 5s during active use, evicting graphs unused for >=10s (never the graph about to be used)
        if (time_now >= next_graph_eviction_time) {
            next_graph_eviction_time = time_now + 5'000'000;
            for (auto it = cuda_graphs.begin(); it != cuda_graphs.end(); ) {
                if (it->first != first_node_ptr && time_now - it->second->last_used_time >= 10'000'000) {
                    it = cuda_graphs.erase(it);
                } else {
                    ++it;
                }
            }
        }''',
    'OPT_UPSTREAM_PR30662',
)

# 3. PR #31803 (adapted): pad qk_shared stride by +8 floats to avoid bank conflicts in wmma::store_matrix_sync
replace(
    'ggml/src/ggml-cuda/lightning-indexer.cu',
    '''    __shared__ float qk_shared[WARPS_PER_BLOCK][HEADS_PER_INNER_LOOP][K_VECS_PER_BLOCK];''',
    '''    constexpr int K_VECS_PADDED = K_VECS_PER_BLOCK + 8; // OPT_UPSTREAM_PR31803: avoid smem bank conflicts
    __shared__ float qk_shared[WARPS_PER_BLOCK][HEADS_PER_INNER_LOOP][K_VECS_PADDED];''',
    'OPT_UPSTREAM_PR31803',
)
replace(
    'ggml/src/ggml-cuda/lightning-indexer.cu',
    '''        wmma::store_matrix_sync((float*) &qk_shared[i_warp][0][0], frag_acc, K_VECS_PER_BLOCK, wmma::mem_row_major);''',
    '''        wmma::store_matrix_sync((float*) &qk_shared[i_warp][0][0], frag_acc, K_VECS_PADDED, wmma::mem_row_major); // OPT_UPSTREAM_PR31803_LDM''',
    'OPT_UPSTREAM_PR31803_LDM',
)
