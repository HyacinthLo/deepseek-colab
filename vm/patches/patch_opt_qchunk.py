"""Per-stream contiguous query chunking for DeepSeek-V4.1 prefill/reserve (adapted from upstream PR #32883/#32885).
Fixes two huge compute-buffer blowups on split-cache (-np 2, n_stream=2) runs:
  1. Slices host input masks (dsv4_hca/csa_kq_mask, cand_pin, raw_mask) per stream (n_stream=1 per chunk) so every
     chunk view is 100% contiguous (128 MiB instead of 640 MiB strided span across streams).
  2. Calls ggml_build_forward_expand() on each indexer chunk's top_k / cand_sel inside the loop so ggml-alloc frees
     idx_score (256 MiB) immediately after each chunk instead of keeping all N chunks (2,048 MiB) live at once.
  3. Chunks dense sparse attention (build_top_k_mask + concat + build_attn_mha) per stream into a single 2D acc buffer.
Leaves decode (nt <= 32) untouched with zero overhead.
Runtime control: LLAMA_DSV41_QCHUNK (tokens per chunk per stream; default 256, 0 disables).
usage: python3 patch_opt_qchunk.py LLAMA_SRC_DIR"""
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
    'src/models/models.h',
    'OPT_QCHUNK_PR32885',
    [
        (
            '''        std::map<std::pair<int32_t, int32_t>, ggml_tensor *> gather_k_of;
        std::map<int32_t, ggml_tensor *> gather_sel_of;
        std::map<int32_t, ggml_tensor *> gather_mask_of;
    };''',
            '''        std::map<std::pair<int32_t, int32_t>, ggml_tensor *> gather_k_of;
        std::map<int32_t, ggml_tensor *> gather_sel_of;
        std::map<int32_t, ggml_tensor *> gather_mask_of;

        // OPT_QCHUNK_PR32885: cache contiguous per-stream chunk views of host input masks once per graph
        std::map<std::tuple<ggml_tensor *, int64_t, int64_t, int64_t>, ggml_tensor *> mask_chunk_of;
        ggml_tensor * get_mask_chunk(ggml_tensor * mask, int64_t s, int64_t t0, int64_t nc);
    };''',
        ),
    ],
)

patch_file(
    'src/llama-context.cpp',
    'OPT_QCHUNK_PR32885',
    [
        (
            '''    // every sampler builds n_outputs_max_per_seq chains, see llm_graph_context::build_sampling''',
            '''    // OPT_QCHUNK_PR32885: query chunking adds nodes per chunk per layer on deepseek41
    if (model.arch == LLM_ARCH_DEEPSEEK41) {
        const char * qc_env = getenv("LLAMA_DSV41_QCHUNK");
        const int64_t qchunk = qc_env ? atoll(qc_env) : 256;
        if (qchunk > 0 && n_tokens > (uint32_t) qchunk) {
            const uint32_t n_chunks = (uint32_t) ((n_tokens + qchunk - 1) / qchunk);
            res += (uint32_t) model.layers.size() * n_chunks * 128u;
        }
    }

    // every sampler builds n_outputs_max_per_seq chains, see llm_graph_context::build_sampling''',
        ),
    ],
)

patch_file(
    'src/models/deepseek41.cpp',
    'OPT_QCHUNK_PR32885',
    [
        (
            '''ggml_tensor * llama_model_deepseek41::graph::build_attention41(''',
            '''ggml_tensor * llama_model_deepseek41::graph::get_mask_chunk(ggml_tensor * mask, int64_t s, int64_t t0, int64_t nc) {
    // OPT_QCHUNK_PR32885: contiguous single-stream slice cached per (mask, s, t0, nc) across all 40 layers
    if (!mask || (s == 0 && t0 == 0 && nc == mask->ne[1] && mask->ne[3] == 1)) {
        return mask;
    }
    auto & entry = mask_chunk_of[{ mask, s, t0, nc }];
    if (!entry) {
        entry = ggml_view_4d(ctx0, mask, mask->ne[0], nc, mask->ne[2], 1,
                mask->nb[1], mask->nb[2], mask->nb[3], s * mask->nb[3] + t0 * mask->nb[1]);
    }
    return entry;
}

ggml_tensor * llama_model_deepseek41::graph::build_attention41(''',
        ),
        (
            '''            static const bool    cbdiet        = dsv41_env_on("LLAMA_DSV41_CBDIET");
            static const int64_t idx_chunk_mib = dsv41_env_int("LLAMA_DSV41_IDX_CHUNK_MIB", 0);
            const int64_t chunk = fused_idx ? (cbdiet && idx_chunk_mib > 0 ? std::clamp<int64_t>((idx_chunk_mib << 18)/(n_comp*n_stream), 1, n_tok) : n_tok) :
                std::max({ (n_tok + 63)/64, (INT64_C(1) << 26)/(n_idx_head*n_comp*n_stream), INT64_C(1) });

            const int32_t cand_src = hparams.dsv41_cand_src;
            const bool    pick_blk = cand_src == il && cand_pin;              // level one: pick the candidate blocks
            const bool    use_blk  = cand_src >= 0 && cand_src < il && cand_sel; // level two: score inside them
            const int64_t bs       = hparams.dsv41_cand_block_size;
            const int64_t n_blk    = pick_blk || use_blk ? (n_comp + bs - 1)/bs : 0;
            GGML_ASSERT(!pick_blk || cand_pin->ne[0] == n_blk);
            GGML_ASSERT(!use_blk  || cand_n_comp == n_comp);

            ggml_tensor * ik_p = fused_idx ? nullptr : ggml_permute(ctx0, ik, 0, 2, 1, 3);
            std::vector<ggml_tensor *> parts_top_k;
            std::vector<ggml_tensor *> parts_sel;
            for (int64_t t0 = 0; t0 < n_tok; t0 += chunk) {
                const int64_t nc = std::min(chunk, n_tok - t0);
                const auto rows = [&](ggml_tensor * t) {
                    return ggml_view_4d(ctx0, t, t->ne[0], nc, 1, n_stream, t->nb[1], t->nb[2], t->nb[3], t0*t->nb[1]);
                };
                ggml_tensor * score = nullptr;
                if (fused_idx) {
                    score = nc == n_tok ? ggml_lightning_indexer(ctx0, iq, ik, iw, ci.kq_mask) :
                        ggml_lightning_indexer(ctx0, ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], nc, n_stream, iq->nb[1], iq->nb[2], iq->nb[3], t0*iq->nb[2]),
                                ik, rows(iw), rows(ci.kq_mask));
                    res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
                } else {
                    ggml_tensor * iqc = ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], nc, n_stream, iq->nb[1], iq->nb[2], iq->nb[3], t0*iq->nb[2]);

                    ggml_tensor * kq = ggml_mul_mat(ctx0, ik_p, ggml_permute(ctx0, iqc, 0, 2, 1, 3));
                    kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));
                    score = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_relu(ctx0, kq), rows(iw)));
                    ggml_tensor * mask  = rows(ci.kq_mask);
                    score = ggml_add(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3)),
                            mask->type == GGML_TYPE_F32 ? mask : ggml_cast(ctx0, mask, GGML_TYPE_F32));
                }
                cb(score, "idx_score", il);

                if (pick_blk) {
                    // best position per block, pin the newest block, keep the top blocks
                    ggml_tensor * s2 = ggml_reshape_2d(ctx0, score, n_comp, nc*n_stream);
                    if (n_blk*bs != n_comp) {
                        s2 = ggml_concat(ctx0, s2, ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_blk*bs - n_comp, nc*n_stream), -INFINITY), 0);
                    }
                    ggml_tensor * bmax = ggml_pool_1d(ctx0, s2, GGML_OP_POOL_MAX, bs, bs, 0);
                    bmax = ggml_add(ctx0, ggml_reshape_4d(ctx0, bmax, n_blk, nc, 1, n_stream), rows(cand_pin));
                    cb(bmax, "cand_block_score", il);
                    parts_sel.push_back(ggml_top_k(ctx0, bmax, std::min<int64_t>(hparams.dsv41_cand_topk_blocks, n_blk)));
                } else if (use_blk && cbdiet && n_blk*bs == n_comp) {
                    // CBDIET_PATCH: the same 0/-inf block mask added by broadcast over the block size; FILL never reads its source, a view of the score
                    ggml_tensor * bsel = ggml_cont(ctx0, rows(cand_sel));
                    const int64_t k_b = bsel->ne[0];
                    const size_t  es  = ggml_element_size(score);
                    ggml_tensor * bm    = ggml_fill(ctx0, ggml_view_4d(ctx0, score, 1, n_blk, nc*n_stream, 1, es, es*n_blk, es*n_blk*nc*n_stream, 0), -INFINITY);
                    ggml_tensor * zeros = ggml_fill(ctx0, ggml_view_4d(ctx0, score, 1, k_b, nc*n_stream, 1, es, es*k_b, es*k_b*nc*n_stream, 0), 0.0f);
                    bm = ggml_set_rows(ctx0, bm, zeros, ggml_reshape_3d(ctx0, bsel, k_b, nc*n_stream, 1));
                    cb(bm, "cand_mask", il);
                    score = ggml_reshape_4d(ctx0, ggml_add_inplace(ctx0, ggml_reshape_4d(ctx0, score, bs, n_blk, nc, n_stream),
                            ggml_reshape_4d(ctx0, bm, 1, n_blk, nc, n_stream)), n_comp, nc, 1, n_stream);
                } else if (use_blk) {
                    // own scores, restricted to the source's candidate blocks: 0 inside them, -inf elsewhere
                    ggml_tensor * bsel = ggml_cont(ctx0, rows(cand_sel));
                    const int64_t k_b = bsel->ne[0];
                    ggml_tensor * bm    = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_blk, nc*n_stream, 1), -INFINITY);
                    ggml_tensor * zeros = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, k_b, nc*n_stream, 1), 0.0f);
                    bm = ggml_set_rows(ctx0, bm, zeros, ggml_reshape_3d(ctx0, bsel, k_b, nc*n_stream, 1));
                    bm = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, bm, bs, n_blk, nc*n_stream, 1), bs*n_blk, nc*n_stream);
                    bm = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, bm, n_comp, nc*n_stream, bm->nb[1], 0)), n_comp, nc, 1, n_stream);
                    cb(bm, "cand_mask", il);
                    score = ggml_add(ctx0, score, bm);
                }

                parts_top_k.push_back(ggml_top_k(ctx0, score, std::min<int64_t>(n_comp, hparams.indexer_top_k)));
            }

            const auto join = [&](std::vector<ggml_tensor *> v) {
                while (v.size() > 1) {
                    std::vector<ggml_tensor *> w;
                    for (size_t i = 0; i < v.size(); i += 2) {
                        w.push_back(i + 1 < v.size() ? ggml_concat(ctx0, v[i], v[i + 1], 1) : v[i]);
                    }
                    v.swap(w);
                }
                return v[0];
            };
            if (pick_blk) {
                cand_sel    = join(parts_sel);
                cand_n_comp = n_comp;
            }
            top_k = ggml_cont(ctx0, join(parts_top_k));''',
            '''            static const bool    cbdiet        = dsv41_env_on("LLAMA_DSV41_CBDIET");
            static const int64_t idx_chunk_mib = dsv41_env_int("LLAMA_DSV41_IDX_CHUNK_MIB", 0);
            static const int64_t qchunk_env    = dsv41_env_int("LLAMA_DSV41_QCHUNK", 256); // OPT_QCHUNK_PR32885
            const int64_t idx_mib_lim = (cbdiet && idx_chunk_mib > 0) ? std::clamp<int64_t>((idx_chunk_mib << 18)/(n_comp*n_stream), 1, n_tok) : n_tok;
            const int64_t qchunk_lim  = (qchunk_env > 0 && n_tok > qchunk_env) ? qchunk_env : n_tok;
            const bool    per_stream_chunk = fused_idx && qchunk_env > 0 && (n_tok > qchunk_env || (n_stream > 1 && nt > qchunk_env));
            const int64_t chunk = per_stream_chunk ? std::min({ n_tok, idx_mib_lim, qchunk_env }) :
                (fused_idx ? std::min({ n_tok, idx_mib_lim, qchunk_lim }) :
                    std::max({ (n_tok + 63)/64, (INT64_C(1) << 26)/(n_idx_head*n_comp*n_stream), INT64_C(1) }));

            const int32_t cand_src = hparams.dsv41_cand_src;
            const bool    pick_blk = cand_src == il && cand_pin;              // level one: pick the candidate blocks
            const bool    use_blk  = cand_src >= 0 && cand_src < il && cand_sel; // level two: score inside them
            const int64_t bs       = hparams.dsv41_cand_block_size;
            const int64_t n_blk    = pick_blk || use_blk ? (n_comp + bs - 1)/bs : 0;
            GGML_ASSERT(!pick_blk || cand_pin->ne[0] == n_blk);
            GGML_ASSERT(!use_blk  || cand_n_comp == n_comp);

            const auto join_dim = [&](std::vector<ggml_tensor *> v, int dim) {
                while (v.size() > 1) {
                    std::vector<ggml_tensor *> w;
                    for (size_t i = 0; i < v.size(); i += 2) {
                        w.push_back(i + 1 < v.size() ? ggml_concat(ctx0, v[i], v[i + 1], dim) : v[i]);
                    }
                    v.swap(w);
                }
                return v[0];
            };

            if (per_stream_chunk) {
                // OPT_QCHUNK_PR32885: iterate per stream so every mask/cand_pin slice is contiguous, and expand each chunk so idx_score is freed immediately
                std::vector<ggml_tensor *> streams_top_k;
                std::vector<ggml_tensor *> streams_sel;
                for (int64_t s = 0; s < n_stream; ++s) {
                    ggml_tensor * ik_s = n_stream == 1 ? ik :
                        ggml_view_4d(ctx0, ik, ik->ne[0], ik->ne[1], ik->ne[2], 1, ik->nb[1], ik->nb[2], ik->nb[3], s * ik->nb[3]);
                    std::vector<ggml_tensor *> parts_top_k;
                    std::vector<ggml_tensor *> parts_sel;
                    for (int64_t t0 = 0; t0 < n_tok; t0 += chunk) {
                        const int64_t nc = std::min(chunk, n_tok - t0);
                        ggml_tensor * iqc = ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], nc, 1,
                                iq->nb[1], iq->nb[2], iq->nb[3], s * iq->nb[3] + t0 * iq->nb[2]);
                        ggml_tensor * iwc = ggml_view_4d(ctx0, iw, iw->ne[0], nc, 1, 1,
                                iw->nb[1], iw->nb[2], iw->nb[3], s * iw->nb[3] + t0 * iw->nb[1]);
                        ggml_tensor * msk = get_mask_chunk(ci.kq_mask, s, t0, nc);
                        ggml_tensor * score = ggml_lightning_indexer(ctx0, iqc, ik_s, iwc, msk);
                        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
                        cb(score, "idx_score", il);

                        if (pick_blk) {
                            ggml_tensor * s2 = ggml_reshape_2d(ctx0, score, n_comp, nc);
                            if (n_blk*bs != n_comp) {
                                s2 = ggml_concat(ctx0, s2, ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_blk*bs - n_comp, nc), -INFINITY), 0);
                            }
                            ggml_tensor * bmax = ggml_pool_1d(ctx0, s2, GGML_OP_POOL_MAX, bs, bs, 0);
                            bmax = ggml_add(ctx0, ggml_reshape_4d(ctx0, bmax, n_blk, nc, 1, 1), get_mask_chunk(cand_pin, s, t0, nc));
                            cb(bmax, "cand_block_score", il);
                            ggml_tensor * sel_c = ggml_top_k(ctx0, bmax, std::min<int64_t>(hparams.dsv41_cand_topk_blocks, n_blk));
                            ggml_build_forward_expand(gf, sel_c);
                            parts_sel.push_back(sel_c);
                        } else if (use_blk && cbdiet && n_blk*bs == n_comp) {
                            ggml_tensor * bsel = ggml_view_4d(ctx0, cand_sel, cand_sel->ne[0], nc, 1, 1,
                                    cand_sel->nb[1], cand_sel->nb[2], cand_sel->nb[3], s * cand_sel->nb[3] + t0 * cand_sel->nb[1]);
                            const int64_t k_b = bsel->ne[0];
                            const size_t  es  = ggml_element_size(score);
                            ggml_tensor * bm    = ggml_fill(ctx0, ggml_view_4d(ctx0, score, 1, n_blk, nc, 1, es, es*n_blk, es*n_blk*nc, 0), -INFINITY);
                            ggml_tensor * zeros = ggml_fill(ctx0, ggml_view_4d(ctx0, score, 1, k_b, nc, 1, es, es*k_b, es*k_b*nc, 0), 0.0f);
                            bm = ggml_set_rows(ctx0, bm, zeros, ggml_reshape_3d(ctx0, bsel, k_b, nc, 1));
                            cb(bm, "cand_mask", il);
                            score = ggml_reshape_4d(ctx0, ggml_add_inplace(ctx0, ggml_reshape_4d(ctx0, score, bs, n_blk, nc, 1),
                                    ggml_reshape_4d(ctx0, bm, 1, n_blk, nc, 1)), n_comp, nc, 1, 1);
                        } else if (use_blk) {
                            ggml_tensor * bsel = ggml_view_4d(ctx0, cand_sel, cand_sel->ne[0], nc, 1, 1,
                                    cand_sel->nb[1], cand_sel->nb[2], cand_sel->nb[3], s * cand_sel->nb[3] + t0 * cand_sel->nb[1]);
                            const int64_t k_b = bsel->ne[0];
                            ggml_tensor * bm    = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_blk, nc, 1), -INFINITY);
                            ggml_tensor * zeros = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, k_b, nc, 1), 0.0f);
                            bm = ggml_set_rows(ctx0, bm, zeros, ggml_reshape_3d(ctx0, bsel, k_b, nc, 1));
                            bm = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, bm, bs, n_blk, nc, 1), bs*n_blk, nc);
                            bm = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, bm, n_comp, nc, bm->nb[1], 0)), n_comp, nc, 1, 1);
                            cb(bm, "cand_mask", il);
                            score = ggml_add(ctx0, score, bm);
                        }

                        ggml_tensor * tk_c = ggml_top_k(ctx0, score, std::min<int64_t>(n_comp, hparams.indexer_top_k));
                        ggml_build_forward_expand(gf, tk_c);
                        parts_top_k.push_back(tk_c);
                    }
                    if (pick_blk) {
                        streams_sel.push_back(join_dim(parts_sel, 1));
                    }
                    streams_top_k.push_back(join_dim(parts_top_k, 1));
                }
                if (pick_blk) {
                    cand_sel    = ggml_cont(ctx0, join_dim(streams_sel, 3));
                    cand_n_comp = n_comp;
                }
                top_k = ggml_cont(ctx0, join_dim(streams_top_k, 3));
            } else {
                ggml_tensor * ik_p = fused_idx ? nullptr : ggml_permute(ctx0, ik, 0, 2, 1, 3);
                std::vector<ggml_tensor *> parts_top_k;
                std::vector<ggml_tensor *> parts_sel;
                for (int64_t t0 = 0; t0 < n_tok; t0 += chunk) {
                    const int64_t nc = std::min(chunk, n_tok - t0);
                    const auto rows = [&](ggml_tensor * t) {
                        return ggml_view_4d(ctx0, t, t->ne[0], nc, 1, n_stream, t->nb[1], t->nb[2], t->nb[3], t0*t->nb[1]);
                    };
                    ggml_tensor * score = nullptr;
                    if (fused_idx) {
                        score = nc == n_tok ? ggml_lightning_indexer(ctx0, iq, ik, iw, ci.kq_mask) :
                            ggml_lightning_indexer(ctx0, ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], nc, n_stream, iq->nb[1], iq->nb[2], iq->nb[3], t0*iq->nb[2]),
                                    ik, rows(iw), rows(ci.kq_mask));
                        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
                    } else {
                        ggml_tensor * iqc = ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], nc, n_stream, iq->nb[1], iq->nb[2], iq->nb[3], t0*iq->nb[2]);

                        ggml_tensor * kq = ggml_mul_mat(ctx0, ik_p, ggml_permute(ctx0, iqc, 0, 2, 1, 3));
                        kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));
                        score = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_relu(ctx0, kq), rows(iw)));
                        ggml_tensor * mask  = rows(ci.kq_mask);
                        score = ggml_add(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3)),
                                mask->type == GGML_TYPE_F32 ? mask : ggml_cast(ctx0, mask, GGML_TYPE_F32));
                    }
                    cb(score, "idx_score", il);

                    if (pick_blk) {
                        ggml_tensor * s2 = ggml_reshape_2d(ctx0, score, n_comp, nc*n_stream);
                        if (n_blk*bs != n_comp) {
                            s2 = ggml_concat(ctx0, s2, ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_blk*bs - n_comp, nc*n_stream), -INFINITY), 0);
                        }
                        ggml_tensor * bmax = ggml_pool_1d(ctx0, s2, GGML_OP_POOL_MAX, bs, bs, 0);
                        bmax = ggml_add(ctx0, ggml_reshape_4d(ctx0, bmax, n_blk, nc, 1, n_stream), rows(cand_pin));
                        cb(bmax, "cand_block_score", il);
                        ggml_tensor * sel_c = ggml_top_k(ctx0, bmax, std::min<int64_t>(hparams.dsv41_cand_topk_blocks, n_blk));
                        ggml_build_forward_expand(gf, sel_c);
                        parts_sel.push_back(sel_c);
                    } else if (use_blk && cbdiet && n_blk*bs == n_comp) {
                        ggml_tensor * bsel = ggml_cont(ctx0, rows(cand_sel));
                        const int64_t k_b = bsel->ne[0];
                        const size_t  es  = ggml_element_size(score);
                        ggml_tensor * bm    = ggml_fill(ctx0, ggml_view_4d(ctx0, score, 1, n_blk, nc*n_stream, 1, es, es*n_blk, es*n_blk*nc*n_stream, 0), -INFINITY);
                        ggml_tensor * zeros = ggml_fill(ctx0, ggml_view_4d(ctx0, score, 1, k_b, nc*n_stream, 1, es, es*k_b, es*k_b*nc*n_stream, 0), 0.0f);
                        bm = ggml_set_rows(ctx0, bm, zeros, ggml_reshape_3d(ctx0, bsel, k_b, nc*n_stream, 1));
                        cb(bm, "cand_mask", il);
                        score = ggml_reshape_4d(ctx0, ggml_add_inplace(ctx0, ggml_reshape_4d(ctx0, score, bs, n_blk, nc, n_stream),
                                ggml_reshape_4d(ctx0, bm, 1, n_blk, nc, n_stream)), n_comp, nc, 1, n_stream);
                    } else if (use_blk) {
                        ggml_tensor * bsel = ggml_cont(ctx0, rows(cand_sel));
                        const int64_t k_b = bsel->ne[0];
                        ggml_tensor * bm    = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_blk, nc*n_stream, 1), -INFINITY);
                        ggml_tensor * zeros = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, k_b, nc*n_stream, 1), 0.0f);
                        bm = ggml_set_rows(ctx0, bm, zeros, ggml_reshape_3d(ctx0, bsel, k_b, nc*n_stream, 1));
                        bm = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, bm, bs, n_blk, nc*n_stream, 1), bs*n_blk, nc*n_stream);
                        bm = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, bm, n_comp, nc*n_stream, bm->nb[1], 0)), n_comp, nc, 1, n_stream);
                        cb(bm, "cand_mask", il);
                        score = ggml_add(ctx0, score, bm);
                    }

                    ggml_tensor * tk_c = ggml_top_k(ctx0, score, std::min<int64_t>(n_comp, hparams.indexer_top_k));
                    ggml_build_forward_expand(gf, tk_c);
                    parts_top_k.push_back(tk_c);
                }
                if (pick_blk) {
                    cand_sel    = join_dim(parts_sel, 1);
                    cand_n_comp = n_comp;
                }
                top_k = ggml_cont(ctx0, join_dim(parts_top_k, 1));
            }''',
        ),
        (
            '''            ggml_tensor * sel_mask = build_top_k_mask(ci.kq_mask, top_k, "comp_top_k_mask", il);
            ggml_tensor * kq_mask  = ggml_concat(ctx0, raw_mask, sel_mask, 0);
            cb(kq_mask, "comp_kq_mask", il);

            // at most window + top-k visible columns per query: lets CUDA FA gather only those
            const int64_t n_kv_max = sparse ? std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + k_sel : 0;
            out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks, nullptr, n_kv_max, kq_scale, il);''',
            '''            // at most window + top-k visible columns per query: lets CUDA FA gather only those
            const int64_t n_kv_max = sparse ? std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + k_sel : 0;

            // OPT_QCHUNK_PR32885: per-stream contiguous chunking of sparse attention mask + FA pass
            static const int64_t qchunk_attn = dsv41_env_int("LLAMA_DSV41_QCHUNK", 256);
            const int64_t n_tok_a = nt / n_stream;
            const bool    do_chunk_attn = qchunk_attn > 0 && nt <= 16384 && (n_tok_a > qchunk_attn || (n_stream > 1 && nt > qchunk_attn));

            if (!do_chunk_attn) {
                ggml_tensor * sel_mask = build_top_k_mask(ci.kq_mask, top_k, "comp_top_k_mask", il);
                ggml_tensor * kq_mask  = ggml_concat(ctx0, raw_mask, sel_mask, 0);
                cb(kq_mask, "comp_kq_mask", il);
                out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks, nullptr, n_kv_max, kq_scale, il);
            } else {
                ggml_tensor * acc = nullptr;
                for (int64_t s = 0; s < n_stream; ++s) {
                    ggml_tensor * k_all_s = n_stream == 1 ? k_all :
                        ggml_view_4d(ctx0, k_all, k_all->ne[0], k_all->ne[1], k_all->ne[2], 1,
                                k_all->nb[1], k_all->nb[2], k_all->nb[3], s * k_all->nb[3]);
                    for (int64_t t0 = 0; t0 < n_tok_a; t0 += qchunk_attn) {
                        const int64_t nc   = std::min(qchunk_attn, n_tok_a - t0);
                        const int64_t row0 = s * n_tok_a + t0;
                        ggml_tensor * ci_m  = get_mask_chunk(ci.kq_mask, s, t0, nc);
                        ggml_tensor * raw_m = get_mask_chunk(raw_mask, s, t0, nc);
                        ggml_tensor * tk_c  = ggml_view_4d(ctx0, top_k, top_k->ne[0], nc, 1, 1,
                                top_k->nb[1], top_k->nb[2], top_k->nb[3], s * top_k->nb[3] + t0 * top_k->nb[1]);
                        ggml_tensor * sel_mask = build_top_k_mask(ci_m, tk_c, "comp_top_k_mask", il);
                        ggml_tensor * kq_mask  = ggml_concat(ctx0, raw_m, sel_mask, 0);
                        cb(kq_mask, "comp_kq_mask", il);

                        ggml_tensor * q_c   = ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], nc, q->nb[1], q->nb[2], row0 * q->nb[2]);
                        ggml_tensor * out_c = build_attn_mha(q_c, k_all_s, k_all_s, nullptr, kq_mask, layer.attn_sinks, nullptr, n_kv_max, kq_scale, il);

                        if (acc == nullptr) {
                            acc = ggml_pad(ctx0, out_c, 0, (int) (nt - nc), 0, 0);
                        } else {
                            acc = ggml_set_2d_inplace(ctx0, acc, out_c, acc->nb[1], row0 * acc->nb[1]);
                        }
                        ggml_build_forward_expand(gf, acc);
                    }
                }
                out = acc;
            }''',
        ),
    ],
)
