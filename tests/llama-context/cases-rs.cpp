// recurrent state cases

#include "test-case.h"

#include "test-log.h"
#include "test-util.h"

#include "common.h"

#include "../../src/llama-ext.h" // llama_arch_from_string

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// roll back multiple sequences, then replay them in a single batch whose per-seq token count exceeds n_ubatch: each
// seq's replay spans several ubatches while its rollback restore is still pending. Compared against a reference context
// that never advanced past the rollback point and decodes the identical replay batch. Only the first replayed token of
// each seq is compared.
struct case_multi_seq_split_replay : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    // the suite has this many models with a recurrent cache
    size_t min_models() const override { return 14; }

    const char * name() const override { return "replay"; }

    const char * description() const override {
        return "replay rolled-back sequences in a batch wider than n_ubatch, against a reference context";
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t  n_seqs     = 2;
        constexpr uint32_t  n_ubatch   = 16;
        constexpr uint32_t  n_prompt   = 20;
        constexpr uint32_t  n_rollback = 3;
        constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
        constexpr llama_pos p0         = n_prompt - n_rollback;

        // both contexts have the same layout, only what they decode differs
        const auto make_ctx_multi = [&]() {
            ctx_opts opts;
            opts.n_ctx      = 512;
            opts.n_batch    = 256;
            opts.n_ubatch   = n_ubatch;
            opts.n_seq_max  = n_seqs;
            opts.n_rs_seq   = 8;
            opts.kv_unified = 0;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx_roll = make_ctx_multi();
        llama_context_ptr ctx_ref  = make_ctx_multi();
        if (!ctx_roll || !ctx_ref) {
            return case_result::fail();
        }

        if (llama_n_rs_seq(ctx_roll.get()) < n_rollback) {
            return case_result::skip("n_rs_seq is too small");
        }

        const auto tok = [&](uint32_t seq, llama_pos pos) {
            return gen_token(n_vocab, (llama_seq_id) seq, pos);
        };

        bool ok = true;

        // keep the anchor at p0 - 1 when rolling back the tail
        for (uint32_t s = 0; s < n_seqs && ok; ++s) {
            ok = ok && decode_gen(ctx_roll.get(), n_vocab, (llama_seq_id) s, 0, (llama_pos) p0 - 1);
            ok = ok && decode_gen(ctx_ref.get(),  n_vocab, (llama_seq_id) s, 0, (llama_pos) p0 - 1);

            ok = ok && decode_gen(ctx_roll.get(), n_vocab, (llama_seq_id) s, (llama_pos) p0 - 1, (llama_pos) n_prompt);
            ok = ok && decode_gen(ctx_ref.get(),  n_vocab, (llama_seq_id) s, (llama_pos) p0 - 1, (llama_pos) p0);

            ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0, -1);

            // a second partial removal while one is pending must be refused
            ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0 - 1, -1);
        }
        CTX_REQUIRE(ok, "the multi-seq prefill / rollback failed");

        // all seqs replay in a single batch
        const auto decode_replay = [&](llama_context * ctx) {
            common_batch batch(ctx);
            for (uint32_t s = 0; s < n_seqs; ++s) {
                for (uint32_t i = 0; i < n_replay; ++i) {
                    const llama_pos pos = p0 + (llama_pos) i;
                    batch.add(tok(s, pos), pos, (llama_seq_id) s, true);
                }
            }
            return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
        };

        ok = decode_replay(ctx_roll.get()) && decode_replay(ctx_ref.get());
        CTX_REQUIRE(ok, "the multi-seq replay decode failed");

        // the first replayed token of each seq reads the state right after the rollback, so its
        //   logits should match; the later tokens are not compared because small state
        //   differences at the start grow with each replayed token
        // random dummy models can still drift up to ~1.7e-5, so the bound is 1e-4
        constexpr float nmse_eps = 1e-4f;

        logits_diff diff;
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const uint32_t i = s*n_replay; // first replayed token of this seq

            const float * l_roll = llama_get_logits_ith(ctx_roll.get(), i);
            const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  i);
            if (l_roll == nullptr || l_ref == nullptr) {
                TLOG_ERR("%s: missing multi-seq logits at index %u\n", __func__, i);
                return case_result::fail();
            }

            diff.add(l_roll, l_ref, n_vocab);
        }

        CTX_REQUIRE(diff.nmse() <= nmse_eps,
                "multi-seq split replay logits mismatch (max diff " + std::to_string(diff.diff_max) +
                ", nmse " + std::to_string(diff.nmse()) + ", first difference at index " + std::to_string(diff.i_first) + ")");

        TLOG_INF("%s: multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff.diff_max, diff.nmse());

        // the replay above can leave the 2 contexts in slightly different states, and the tail
        //   below assumes they start from the same state - so copy the roll state into ref
        for (uint32_t s = 0; s < n_seqs && ok; ++s) {
            std::vector<uint8_t> state;

            ok = ok && get_seq_state(ctx_roll.get(), (llama_seq_id) s, 0, state);
            ok = ok && set_seq_state(ctx_ref.get(),  (llama_seq_id) s, 0, state);

            // do a second round of state save/load to negate any SWA-positioning effects
            if (ok) {
                std::vector<uint8_t> restored;
                ok = ok && get_seq_state(ctx_ref.get(), (llama_seq_id) s, 0, restored);

                // what was loaded into ref must come back out bit-identical
                CTX_REQUIRE(ok && state == restored,
                        "the seq " + std::to_string(s) + " state did not round-trip bit-exact (sizes " +
                        std::to_string(state.size()) + "/" + std::to_string(restored.size()) + ")");

                ok = ok && set_seq_state(ctx_roll.get(), (llama_seq_id) s, 0, restored);
            }
        }
        CTX_REQUIRE(ok, "the multi-seq state copy failed");

        // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
        //   in ctx_ref only, then compare identical seq-1-only continuations
        constexpr uint32_t n_tail = 4;

        {
            common_batch batch_tail(ctx_ref.get());
            for (uint32_t i = 0; i < n_tail; ++i) {
                const llama_pos pos = p0 + (llama_pos) (n_replay + i);
                batch_tail.add(tok(0, pos + 7), pos, 0, false);
            }
            ok = llama_process(ctx_ref.get(), LLAMA_PROCESS_TYPE_DECODE, batch_tail.get()) == 0;
        }

        logits_diff diff_tail;
        for (uint32_t i = 0; i < n_tail && ok; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            ok = decode_one(ctx_roll.get(), tok(1, pos), pos, 1) &&
                 decode_one(ctx_ref.get(),  tok(1, pos), pos, 1);
            if (!ok) {
                break;
            }

            const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
            const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
            if (l_roll == nullptr || l_ref == nullptr) {
                TLOG_ERR("%s: missing tail logits at step %u\n", __func__, i);
                return case_result::fail();
            }

            diff_tail.add(l_roll, l_ref, n_vocab);
        }

        CTX_REQUIRE(ok && diff_tail.nmse() <= 0.0f,
                "a seq-1-only decode leaked the state of seq 0 (nmse " + std::to_string(diff_tail.nmse()) + ")");

        TLOG_INF("%s: seq-1-only decode is independent of seq 0 (max diff %g, nmse %g)\n",
                __func__, (double) diff_tail.diff_max, diff_tail.nmse());

        return case_result::pass();
    }
};

// save a rolled-back single-seq state, restore it into a fresh and into a dirty context, and
// verify exact logit matches on replay
struct case_rollback : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    // the suite has this many models with a recurrent cache
    size_t min_models() const override { return 14; }

    const char * name() const override { return "rollback"; }

    const char * description() const override {
        return "a rolled-back recurrent state restores into fresh and dirty contexts with identical replay";
    }

    case_result run(model_run & mr) const override {
        const llama_vocab * vocab   = llama_model_get_vocab(mr.model);
        const int           n_vocab = llama_vocab_n_tokens(vocab);

        const auto make_rs_ctx = [&]() {
            ctx_opts opts;
            opts.n_seq_max = 1;
            opts.n_rs_seq  = 8;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx_src = make_rs_ctx();
        llama_context_ptr ctx_dst = make_rs_ctx();
        if (!ctx_src || !ctx_dst) {
            return case_result::fail();
        }

        const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src.get());
        if (n_rs_seq == 0) {
            return case_result::skip("n_rs_seq is disabled");
        }

        constexpr uint32_t n_rollback = 3;
        if (n_rs_seq < n_rollback) {
            return case_result::skip("n_rs_seq is too small");
        }

        llama_tokens tokens;
        if (!mr.has_vocab()) {
            tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        } else {
            tokens = common_tokenize(ctx_src.get(), "The quick brown fox jumps over the lazy dog", true);
        }
        if (tokens.empty()) {
            TLOG_ERR("%s: not enough prompt tokens\n", __func__);
            return case_result::fail();
        }
        tokens.resize(n_rs_seq + 1, tokens.back());

        const llama_pos rollback_pos = (llama_pos) tokens.size() - n_rollback;

        // Decode the full prompt on the source, then roll back three positions.
        // Replaying them crosses DSV4's ratio-4 compressor boundary.
        // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
        if (!decode_tokens(ctx_src.get(), tokens)) {
            TLOG_ERR("%s: failed to decode the prompt\n", __func__);
            return case_result::fail();
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1)) {
            TLOG_ERR("%s: rollback failed\n", __func__);
            return case_result::fail();
        }

        // Save the rolled-back state and restore it into a fresh context.
        common_prompt_checkpoint ckpt;
        ckpt.update_tgt(ctx_src.get(), 0, 0);
        ckpt.load_tgt(ctx_dst.get(), 0, 0);

        // exact match - the same context configuration replaying the same tokens
        constexpr float nmse_eps = 0.0;

        std::vector<std::vector<float>> logits_src_replay(n_rollback);
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src.get(), tokens[pos], pos) ||
                !decode_one(ctx_dst.get(), tokens[pos], pos)) {
                TLOG_ERR("%s: replay failed at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const float * logits_src = llama_get_logits_ith(ctx_src.get(), 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst.get(), 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                TLOG_ERR("%s: missing replay logits at position %d\n", __func__, pos);
                return case_result::fail();
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);

            const logits_diff diff = compare_logits(logits_src, logits_dst, n_vocab);
            CTX_REQUIRE(diff.nmse() <= nmse_eps,
                    "replay logits mismatch at position " + std::to_string(pos) + ", nmse " + std::to_string(diff.nmse()));
        }

        // the replay ran in single-token ubatches, which hold no snapshot that deep
        CTX_REQUIRE(!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1) &&
                    !llama_memory_seq_rm(llama_get_memory(ctx_dst.get()), 0, rollback_pos, -1),
                "a rollback past the snapshots of single-token ubatches was accepted");

        // Repeat the load into a context that already has its own rollback state:
        // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
        // non-zero at load time. The restore must wipe that state and still match.
        llama_context_ptr ctx_dirty = make_rs_ctx();
        if (!ctx_dirty) {
            return case_result::fail();
        }

        llama_tokens noise = tokens;
        for (auto & tok : noise) {
            tok = (tok + 1) % n_vocab;
            if (tok < 0) {
                tok = 0;
            }
        }
        if (!decode_tokens(ctx_dirty.get(), noise)) {
            TLOG_ERR("%s: failed to decode the dirty prompt\n", __func__);
            return case_result::fail();
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty.get()), 0, rollback_pos, -1)) {
            TLOG_ERR("%s: dirty rollback failed\n", __func__);
            return case_result::fail();
        }

        ckpt.load_tgt(ctx_dirty.get(), 0, 0);

        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_dirty.get(), tokens[pos], pos)) {
                TLOG_ERR("%s: dirty replay failed at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const float * logits_dirty = llama_get_logits_ith(ctx_dirty.get(), 0);
            if (logits_dirty == nullptr) {
                TLOG_ERR("%s: missing dirty logits at position %d\n", __func__, pos);
                return case_result::fail();
            }

            const logits_diff diff = compare_logits(logits_src_replay[i].data(), logits_dirty, n_vocab);
            CTX_REQUIRE(diff.nmse() <= nmse_eps,
                    "dirty-ctx logits mismatch at position " + std::to_string(pos) + ", nmse " + std::to_string(diff.nmse()));
        }

        TLOG_INF("%s: recurrent rollback checkpoint restored successfully\n", __func__);

        return case_result::pass();
    }
};

// the DSV4 cache rebuilds snapshot d from the state before the ubatch, so d <= n is exact there,
//   while the recurrent memory gets snapshots 0 .. n - 1 from ggml_gated_delta_net, only for
//   n <= n_rs_seq + 1
static bool rs_from_ubatch_start(const llama_model * model) {
    char arch_name[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch_name, sizeof(arch_name));
    const llm_arch arch = llama_arch_from_string(arch_name);

    return arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41;
}

// the deepest rollback a cell of this memory type guarantees after a ubatch of n tokens
static uint32_t rs_rollback_bound(bool from_ubatch_start, uint32_t n, uint32_t n_rs_seq) {
    return from_ubatch_start ? std::min(n, n_rs_seq) : (n <= n_rs_seq + 1 ? n - 1 : 0);
}

// after a ubatch of n tokens, probe rollbacks from deepest to shallowest: every rollback past the
// snapshots that ubatch guarantees is refused and the deepest guaranteed one is accepted
struct case_rollback_bound : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    // the suite has this many models with a recurrent cache
    size_t min_models() const override { return 14; }

    const char * name() const override { return "rollback_bound"; }

    const char * description() const override {
        return "seq_rm accepts exactly the rollbacks whose snapshots the last ubatch wrote";
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr llama_pos n_prefix = 16;

        const bool from_ubatch_start = rs_from_ubatch_start(mr.model);

        const auto make_ctx = [&]() {
            ctx_opts opts;
            opts.n_batch   = 64;
            opts.n_ubatch  = 64;
            opts.n_seq_max = 1;
            opts.n_rs_seq  = 8;
            return mr.ctx(opts);
        };

        llama_context_ptr ctx = make_ctx();
        if (!ctx) {
            return case_result::fail();
        }

        const uint32_t n_rs_seq = llama_n_rs_seq(ctx.get());
        if (n_rs_seq == 0) {
            return case_result::skip("n_rs_seq is disabled");
        }

        for (const uint32_t n : { 1u, 2u, n_rs_seq + 1, n_rs_seq + 2 }) {
            const uint32_t r_max = rs_rollback_bound(from_ubatch_start, n, n_rs_seq);

            ctx = make_ctx();
            if (!ctx) {
                return case_result::fail();
            }

            const llama_pos pos_end = n_prefix + (llama_pos) n;

            CTX_REQUIRE(decode_gen(ctx.get(), n_vocab, 0, 0, n_prefix) &&
                        decode_gen(ctx.get(), n_vocab, 0, n_prefix, pos_end),
                    "the prefix / ubatch decode failed");

            for (uint32_t r = n_rs_seq + 1; r > r_max; --r) {
                CTX_REQUIRE(!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, pos_end - (llama_pos) r, -1),
                        "rollback " + std::to_string(r) + " after a " + std::to_string(n) + "-token ubatch was accepted");
            }

            if (r_max > 0) {
                CTX_REQUIRE(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, pos_end - (llama_pos) r_max, -1),
                        "rollback " + std::to_string(r_max) + " after a " + std::to_string(n) + "-token ubatch was refused");
            }
        }

        TLOG_INF("%s: rollback bounds hold for ubatches of 1, 2, %u and %u tokens\n", __func__, n_rs_seq + 1, n_rs_seq + 2);

        return case_result::pass();
    }
};

// decode a prompt into seq 0, share its cells with a second sequence, then keep decoding both
struct case_shared_seq_reserve : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    // this case also runs on models without a recurrent cache
    size_t min_models() const override { return 22; }

    const char * name() const override { return "shared"; }

    const char * description() const override {
        return "sequences sharing cells keep decoding correctly after a seq_cp";
    }

    std::string skip_reason(model_run & mr) const override {
        if (arch_reserves_single_seq(mr.model)) {
            return "the shared-seq batch is a multi-seq graph";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t n_seqs     = 2;
        constexpr uint32_t n_prompt   = 128;
        constexpr uint32_t n_continue = 32;

        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_batch    = 256;
        opts.n_ubatch   = 64;
        opts.n_seq_max  = n_seqs;
        opts.kv_unified = 1; // only a unified cache shares cells on seq_cp

        auto ctx = mr.ctx(opts);
        if (!ctx) {
            return case_result::fail();
        }

        if (!decode_gen(ctx.get(), n_vocab, 0, 0, (llama_pos) n_prompt)) {
            TLOG_ERR("%s: prompt decode failed\n", __func__);
            return case_result::fail();
        }

        // this is what llama-batched-bench does for -pps
        llama_memory_seq_cp(llama_get_memory(ctx.get()), 0, 1, -1, -1);

        for (uint32_t i = 0; i < n_continue; ++i) {
            const llama_pos pos = (llama_pos) (n_prompt + i);

            common_batch batch(ctx.get());
            for (uint32_t s = 0; s < n_seqs; ++s) {
                batch.add(gen_token(n_vocab, (llama_seq_id) s, pos), pos, (llama_seq_id) s, true);
            }
            CTX_REQUIRE(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0,
                    "shared-seq decode failed at step " + std::to_string(i));

            for (uint32_t s = 0; s < n_seqs; ++s) {
                const float * logits = llama_get_logits_ith(ctx.get(), (int) s);
                if (logits == nullptr) {
                    TLOG_ERR("%s: missing shared-seq logits at index %u\n", __func__, s);
                    return case_result::fail();
                }

                bool finite = true;
                for (int t_ = 0; t_ < n_vocab; ++t_) {
                    finite = finite && std::isfinite(logits[t_]);
                }

                CTX_REQUIRE(finite, "non-finite shared-seq logit at step " + std::to_string(i) + ", seq " + std::to_string(s));
            }
        }

        TLOG_INF("%s: shared-seq decode succeeded (%u tokens after seq_cp)\n", __func__, n_continue*n_seqs);

        return case_result::pass();
    }
};

// how many bytes of two serialized states differ, and where the first one does
struct state_delta {
    size_t n_diff  = 0;
    size_t i_first = 0;

    bool equal() const { return n_diff == 0; }
};

static state_delta state_diff(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    state_delta res;

    if (a.size() != b.size()) {
        res.n_diff  = std::max(a.size(), b.size());
        res.i_first = std::min(a.size(), b.size());

        return res;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            if (res.n_diff == 0) {
                res.i_first = i;
            }
            ++res.n_diff;
        }
    }

    return res;
}

// a ubatch that carries the other sequences makes find_slot relocate the cell of an idle
// sequence: the rollback snapshots of that cell must move with it, so that a speculative rollback
// on the relocated sequence holds the state of a context that never relocated
//
// the comparison runs on the serialized sequence state rather than on the logits, because the
//   random test models produce logits that barely depend on the recurrent state
struct case_relocate : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    // the suite has this many models with a recurrent cache
    size_t min_models() const override { return 14; }

    const char * name() const override { return "relocate"; }

    const char * description() const override {
        return "a sequence relocated by another ubatch keeps its rollback snapshots and bound";
    }

    std::string skip_reason(model_run & mr) const override {
        if (arch_reserves_single_seq(mr.model)) {
            return "the relocation batch is a multi-seq graph";
        }

        return ctx_test::skip_reason(mr);
    }

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        // the splitter accepts only consecutive sequence ids in one ubatch, so the ubatch carries
        //   seq 0 and seq 1 and seq 2 stays idle. Decoding the sequences out of order puts the
        //   cell of seq 2 between the cells of the other two, so the re-order step of find_slot
        //   has to swap it out of the contiguous range of the ubatch
        constexpr llama_seq_id seq_a     = 0;
        constexpr llama_seq_id seq_b     = 1;
        constexpr llama_seq_id seq_moved = 2;

        constexpr uint32_t n_prefix = 8;
        // the last ubatch of a sequence writes snapshots 0 .. n_tail - 1, so its cell guarantees
        //   a rollback of n_tail - 1 tokens
        constexpr uint32_t n_tail     = 4;
        constexpr uint32_t n_rollback = 1;
        constexpr uint32_t n_ubatch   = 16;

        const auto make_ctx_n = [&](uint32_t n_seq) {
            ctx_opts opts;
            opts.n_ctx      = 256;
            opts.n_batch    = 64;
            opts.n_ubatch   = n_ubatch;
            opts.n_seq_max  = n_seq;
            opts.n_rs_seq   = 8;
            opts.kv_unified = 0;
            return mr.ctx(opts);
        };

        const auto make_ctx = [&]() { return make_ctx_n(3); };

        llama_context_ptr ctx_roll = make_ctx();
        llama_context_ptr ctx_ref  = make_ctx();
        if (!ctx_roll || !ctx_ref) {
            return case_result::fail();
        }

        const uint32_t n_rs_seq = llama_n_rs_seq(ctx_roll.get());
        if (n_rs_seq < n_tail) {
            return case_result::skip("n_rs_seq is too small");
        }

        // deepseek4 and deepseek41 keep the pending flag of a sequence that is outside the ubatch
        //   and refuse a stacked rollback, so a bound assertion cannot exercise the cell there
        const bool from_ubatch_start = rs_from_ubatch_start(mr.model);

        // the last n_tok tokens of a sequence go into a ubatch of their own, so that its cell
        //   keeps the snapshots of that ubatch
        const auto prefill = [&](llama_context * ctx, llama_seq_id seq_id, uint32_t n_tok) {
            const llama_pos p_mid = (llama_pos) n_prefix;
            return decode_gen(ctx, n_vocab, seq_id, 0, p_mid) &&
                   decode_gen(ctx, n_vocab, seq_id, p_mid, p_mid + (llama_pos) n_tok);
        };

        // one ubatch carrying two sequences, which relocates the cell of the sequence between them
        const auto relocate = [&](llama_context * ctx, llama_pos pos_a, llama_pos pos_b) {
            common_batch batch(ctx);
            batch.add(gen_token(n_vocab, seq_a, pos_a), pos_a, seq_a, false);
            batch.add(gen_token(n_vocab, seq_b, pos_b), pos_b, seq_b, false);

            return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
        };

        // the state read must not depend on the cell index: the same sequence decoded into two
        //   different cells, with nothing relocated, serializes to the same bytes
        {
            llama_context_ptr ctx_same = make_ctx();
            llama_context_ptr ctx_other = make_ctx();
            bool control_ok = ctx_same && ctx_other;

            for (const llama_seq_id seq_id : { seq_a, seq_moved, seq_b }) {
                control_ok = control_ok && prefill(ctx_same.get(), seq_id, n_tail);
            }
            // the same sequences, decoded in an order that gives seq_moved another cell
            for (const llama_seq_id seq_id : { seq_a, seq_b, seq_moved }) {
                control_ok = control_ok && prefill(ctx_other.get(), seq_id, n_tail);
            }

            std::vector<uint8_t> st_same;
            std::vector<uint8_t> st_other;
            control_ok = control_ok && get_seq_state(ctx_same.get(),  seq_moved, 0, st_same) &&
                                      get_seq_state(ctx_other.get(), seq_moved, 0, st_other);

            CTX_REQUIRE(control_ok, "the cell-index control could not run");

            const state_delta control = state_diff(st_same, st_other);

            TLOG_INF("%s: cell-index control %zu bytes, %zu differ\n", __func__, st_same.size(), control.n_diff);

            CTX_REQUIRE(control.equal(),
                    "the sequence state read depends on the cell index (" + std::to_string(control.n_diff) + " of " +
                    std::to_string(st_same.size()) + " bytes differ, first at offset " + std::to_string(control.i_first) + ")");
        }

        // each sequence gets its own cell, in sequence order
        bool ok = true;
        for (const llama_seq_id seq_id : { seq_a, seq_moved, seq_b }) {
            ok = ok && prefill(ctx_roll.get(), seq_id, n_tail) &&
                       prefill(ctx_ref.get(),  seq_id, n_tail);
        }
        CTX_REQUIRE(ok, "the three-sequence prefill failed");

        const llama_pos p_end = (llama_pos) (n_prefix + n_tail);

        ok = relocate(ctx_roll.get(), p_end, p_end);
        // the reference decodes the same tokens one sequence at a time, so no cell is relocated
        ok = ok && decode_gen(ctx_ref.get(), n_vocab, seq_a, p_end, p_end + 1) &&
                   decode_gen(ctx_ref.get(), n_vocab, seq_b, p_end, p_end + 1);
        CTX_REQUIRE(ok, "the relocation ubatch failed");

        // one more ubatch of the two sequences leaves the moved cell out: the cell that seq_moved
        //   used to occupy is overwritten by its new owner, and the cell index the memory copies
        //   from settles on the new cell of seq_moved
        {
            common_batch batch(ctx_roll.get());
            for (const llama_seq_id seq_id : { seq_a, seq_b }) {
                for (llama_pos pos = p_end + 1; pos <= p_end + (llama_pos) n_tail; ++pos) {
                    batch.add(gen_token(n_vocab, seq_id, pos), pos, seq_id, false);
                }
            }

            ok = llama_process(ctx_roll.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
        }
        ok = ok && decode_gen(ctx_ref.get(), n_vocab, seq_a, p_end + 1, p_end + 1 + (llama_pos) n_tail) &&
                   decode_gen(ctx_ref.get(), n_vocab, seq_b, p_end + 1, p_end + 1 + (llama_pos) n_tail);
        CTX_REQUIRE(ok, "the ubatch after the relocation failed");

        // the speculative rollback of the last token of the moved sequence
        const llama_pos p_replay = p_end - (llama_pos) n_rollback;

        CTX_REQUIRE(llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), seq_moved, p_replay, -1),
                "the rollback of the relocated sequence was refused");
        CTX_REQUIRE(llama_memory_seq_rm(llama_get_memory(ctx_ref.get()), seq_moved, p_replay, -1),
                "the rollback of the reference sequence was refused");

        // the pending rollback is part of the serialized state, so the state of the moved sequence
        //   must be the state of a sequence that never relocated
        std::vector<uint8_t> st_roll;
        std::vector<uint8_t> st_ref;

        CTX_REQUIRE(get_seq_state(ctx_roll.get(), seq_moved, 0, st_roll) &&
                    get_seq_state(ctx_ref.get(),  seq_moved, 0, st_ref),
                "the sequence state could not be read");

        const state_delta delta = state_diff(st_roll, st_ref);

        // replay several tokens: a state restored from the wrong snapshot is already different at
        //   the first token and diverges further with every one
        constexpr uint32_t n_replay = 8;

        logits_diff diff;
        for (uint32_t i = 0; i < n_replay && ok; ++i) {
            const llama_pos pos = p_replay + (llama_pos) i;
            const llama_token tok = gen_token(n_vocab, seq_moved, pos);

            ok = decode_one(ctx_roll.get(), tok, pos, seq_moved) &&
                 decode_one(ctx_ref.get(),  tok, pos, seq_moved);
            if (!ok) {
                break;
            }

            const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
            const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
            if (l_roll == nullptr || l_ref == nullptr) {
                TLOG_ERR("%s: missing relocated rollback logits at step %u\n", __func__, i);
                return case_result::fail();
            }

            diff.add(l_roll, l_ref, n_vocab);
        }
        CTX_REQUIRE(ok, "the post-rollback decode failed");

        CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD,
                "the relocated sequence replayed from different snapshots (max diff " + std::to_string(diff.diff_max) +
                ", nmse " + std::to_string(diff.nmse()) + ", first difference at index " + std::to_string(diff.i_first) + ")");

        TLOG_INF("%s: relocated rollback matched (max diff %g, nmse %g)\n", __func__, (double) diff.diff_max, diff.nmse());

        TLOG_INF("%s: relocated state %zu bytes, %zu differ, first at %zu\n", __func__,
                st_roll.size(), delta.n_diff, delta.i_first);

        CTX_REQUIRE(delta.equal(),
                "the relocated sequence carries a different state (" + std::to_string(delta.n_diff) + " of " +
                std::to_string(st_roll.size()) + " bytes differ, first at offset " + std::to_string(delta.i_first) + ")");

        // a second layout: the moved sequence serves fewer rollback tokens than the cell that
        //   hosts it used to, so a rollback past its own snapshots must be refused
        constexpr uint32_t n_tail_moved = 2;
        constexpr uint32_t n_tail_other = 5;

        const uint32_t n_bound       = rs_rollback_bound(from_ubatch_start, n_tail_moved, n_rs_seq);
        const uint32_t n_bound_other = rs_rollback_bound(from_ubatch_start, n_tail_other, n_rs_seq);

        llama_context_ptr ctx_bnd = make_ctx();
        if (!ctx_bnd) {
            return case_result::fail();
        }

        ok = prefill(ctx_bnd.get(), seq_a, n_tail) &&
             prefill(ctx_bnd.get(), seq_moved, n_tail_moved) &&
             prefill(ctx_bnd.get(), seq_b, n_tail_other);
        CTX_REQUIRE(ok, "the second three-sequence prefill failed");

        const llama_pos p_end_moved = (llama_pos) (n_prefix + n_tail_moved);

        ok = relocate(ctx_bnd.get(), (llama_pos) (n_prefix + n_tail), (llama_pos) (n_prefix + n_tail_other));
        CTX_REQUIRE(ok, "the second relocation ubatch failed");

        // the leg only separates a carried bound from an inherited one while the hosting cell
        //   guarantees more than the moved sequence does
        CTX_REQUIRE(n_bound_other > n_bound,
                "the hosting cell does not guarantee a deeper rollback than the moved sequence");

        CTX_REQUIRE(!llama_memory_seq_rm(llama_get_memory(ctx_bnd.get()), seq_moved,
                        p_end_moved - (llama_pos) (n_bound + 1), -1),
                "a rollback past the snapshots of the relocated sequence was accepted");

        // the refused rollback changed nothing, so the deepest one the cell guarantees still holds
        CTX_REQUIRE(n_bound > 0 && llama_memory_seq_rm(llama_get_memory(ctx_bnd.get()), seq_moved,
                        p_end_moved - (llama_pos) n_bound, -1),
                "the rollback the relocated cell guarantees was refused");

        // a third layout: the displaced sequence holds a pending rollback when an unrelated ubatch
        //   takes its cell. s_copy runs for every entry of the block, so that ubatch applies the
        //   rollback and writes the rolled-back state into the cell's plane 0, which leaves the
        //   cell no snapshot for a further rollback
        constexpr llama_seq_id seq_other = 3;
        constexpr uint32_t     n_pending = n_tail - 1; // within the bound the sequence earned above

        const auto prefill_order = [&](llama_context * ctx, const std::vector<llama_seq_id> & order) {
            bool res = true;
            for (const llama_seq_id seq_id : order) {
                res = res && prefill(ctx, seq_id, n_tail);
            }
            return res;
        };

        // both contexts run the same ubatch composition. The decode order decides whether the batch
        //   displaces seq_moved: in the roll context its cell sits in the destination range of two
        //   cells that are further apart than two, in the reference it sits inside the block of the
        //   ubatch but outside that range
        llama_context_ptr ctx_pend     = make_ctx_n(4);
        llama_context_ptr ctx_pend_ref = make_ctx_n(4);
        if (!ctx_pend || !ctx_pend_ref) {
            return case_result::fail();
        }

        ok = prefill_order(ctx_pend.get(),     { seq_a, seq_moved, seq_other, seq_b }) &&
             prefill_order(ctx_pend_ref.get(), { seq_a, seq_other, seq_moved, seq_b });
        CTX_REQUIRE(ok, "the four-sequence prefill failed");

        const llama_pos p_end4   = (llama_pos) (n_prefix + n_tail);
        const llama_pos p_rolled = p_end4 - (llama_pos) n_pending;

        // the sequence has to serve this rollback before the relocation: without it the leg would
        //   report a pass for a setup that never had a pending rollback
        CTX_REQUIRE(llama_memory_seq_rm(llama_get_memory(ctx_pend.get()), seq_moved, p_rolled, -1) &&
                    llama_memory_seq_rm(llama_get_memory(ctx_pend_ref.get()), seq_moved, p_rolled, -1),
                "the pending rollback is not served before the relocation");

        ok = relocate(ctx_pend.get(), p_end4, p_end4) &&
             relocate(ctx_pend_ref.get(), p_end4, p_end4);
        CTX_REQUIRE(ok, "the ubatch that takes the pending cell failed");

        // the gather applied the pending plane into plane 0 of the cell
        std::vector<uint8_t> st_pend;
        std::vector<uint8_t> st_pend_ref;

        CTX_REQUIRE(get_seq_state(ctx_pend.get(), seq_moved, 0, st_pend) &&
                    get_seq_state(ctx_pend_ref.get(),  seq_moved, 0, st_pend_ref),
                "the pending sequence state could not be read");

        const state_delta d_pend = state_diff(st_pend, st_pend_ref);

        TLOG_INF("%s: pending cell %zu bytes, %zu differ, first at %zu\n", __func__,
                st_pend.size(), d_pend.n_diff, d_pend.i_first);

        CTX_REQUIRE(d_pend.equal(),
                "the displaced sequence did not carry its pending rollback (" + std::to_string(d_pend.n_diff) + " of " +
                std::to_string(st_pend.size()) + " bytes differ, first at offset " + std::to_string(d_pend.i_first) + ")");

        // the cell wrote no snapshot of its own, so its bound drops to zero. The refusal does not
        //   prove that on deepseek4 and deepseek41: a sequence outside the ubatch keeps its pending
        //   flag there (llama-kv-cache-dsv4.cpp:1859) and a stacked rollback is refused anyway
        //   (:1609), so the assertion would pass without the bound being enforced by relocation
        const llama_pos p_last = p_end4 - 1 - (llama_pos) n_pending;

        if (!from_ubatch_start) {
            CTX_REQUIRE(!llama_memory_seq_rm(llama_get_memory(ctx_pend.get()), seq_moved, p_last, -1),
                    "a rollback past the displaced cell was accepted after its pending rollback was applied");

            // the zero also lands on a pending cell that the ubatch never displaced
            CTX_REQUIRE(!llama_memory_seq_rm(llama_get_memory(ctx_pend_ref.get()), seq_moved, p_last, -1),
                    "a rollback past the in-place cell was accepted after its pending rollback was applied");
        }

        // one re-decoded token puts the sequence back on the reference's trajectory
        ok = decode_one(ctx_pend.get(), gen_token(n_vocab, seq_moved, p_rolled), p_rolled, seq_moved) &&
             decode_one(ctx_pend_ref.get(), gen_token(n_vocab, seq_moved, p_rolled), p_rolled, seq_moved);
        CTX_REQUIRE(ok, "the sequence did not decode after the refused rollback");

        st_pend.clear();
        st_pend_ref.clear();

        CTX_REQUIRE(get_seq_state(ctx_pend.get(), seq_moved, 0, st_pend) &&
                    get_seq_state(ctx_pend_ref.get(),  seq_moved, 0, st_pend_ref),
                "the resumed sequence state could not be read");

        const state_delta d_resumed = state_diff(st_pend, st_pend_ref);

        TLOG_INF("%s: resumed cell %zu bytes, %zu differ\n", __func__, st_pend.size(), d_resumed.n_diff);

        CTX_REQUIRE(d_resumed.equal(),
                "the sequence did not resume like the reference (" + std::to_string(d_resumed.n_diff) + " of " +
                std::to_string(st_pend.size()) + " bytes differ, first at offset " + std::to_string(d_resumed.i_first) + ")");

        // a ubatch of its own earns the bound back: the zero is a window of one round trip, not a
        //   permanent state of the cell
        const llama_pos p_regain = p_rolled + 1;
        const llama_pos p_regain_end = p_regain + (llama_pos) n_tail;

        ok = decode_gen(ctx_pend.get(), n_vocab, seq_moved, p_regain, p_regain_end) &&
             decode_gen(ctx_pend_ref.get(), n_vocab, seq_moved, p_regain, p_regain_end);
        CTX_REQUIRE(ok, "the sequence could not decode after the resumed token");

        CTX_REQUIRE(llama_memory_seq_rm(llama_get_memory(ctx_pend.get()), seq_moved, p_regain_end - 1, -1),
                "the displaced cell did not earn its rollback bound back");
        CTX_REQUIRE(llama_memory_seq_rm(llama_get_memory(ctx_pend_ref.get()), seq_moved, p_regain_end - 1, -1),
                "the reference cell was refused the same rollback");

        st_pend.clear();
        st_pend_ref.clear();

        CTX_REQUIRE(get_seq_state(ctx_pend.get(), seq_moved, 0, st_pend) &&
                    get_seq_state(ctx_pend_ref.get(),  seq_moved, 0, st_pend_ref),
                "the re-earned sequence state could not be read");

        const state_delta d_regain = state_diff(st_pend, st_pend_ref);

        TLOG_INF("%s: re-earned cell %zu bytes, %zu differ\n", __func__, st_pend.size(), d_regain.n_diff);

        CTX_REQUIRE(d_regain.equal(),
                "the sequence did not keep the reference's trajectory after earning its bound (" +
                std::to_string(d_regain.n_diff) + " of " + std::to_string(st_pend.size()) +
                " bytes differ, first at offset " + std::to_string(d_regain.i_first) + ")");

        TLOG_INF("%s: the relocated sequence keeps its rollback bound\n", __func__);

        return case_result::pass();
    }
};

// counts the polls of the compute abort callback, and aborts on poll abort_at
struct abort_counter {
    int64_t n_polls  = 0;
    int64_t abort_at = -1; // -1 = never

    static bool cb(void * data) {
        auto * a = (abort_counter *) data;
        const bool res = a->abort_at >= 0 && a->n_polls >= a->abort_at;
        ++a->n_polls;
        return res;
    }
};

// a failed ubatch: the compute abort callback stops a graph that may already have written the memory, and the test
//   hook llama_test_fail_before_graph stops a ubatch after its memory apply and before its graph. Afterwards every
//   sequence is either gone from the memory or holds exactly the state it had before the decode, and the context goes
//   on to match a reference that never attempted the failed ubatch
struct case_failed_ubatch : ctx_test {
    test_group group() const override { return TEST_GROUP_RS; }

    size_t min_models() const override { return 14; }

    const char * name() const override { return "failed_ub"; }

    const char * description() const override {
        return "a failed ubatch leaves every sequence removed or as it was, and the context recovers";
    }

    std::string skip_reason(model_run & mr) const override {
        if (arch_reserves_single_seq(mr.model)) {
            return "the failure batches are multi-seq graphs";
        }

        return ctx_test::skip_reason(mr);
    }

    struct seq_snap {
        llama_pos pos_min = -1;
        llama_pos pos_max = -1;
        std::vector<uint8_t> st;
    };

    case_result run(model_run & mr) const override {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

        constexpr uint32_t n_prefix = 8;
        constexpr uint32_t n_tail   = 4;
        constexpr uint32_t n_ubatch = 16;
        constexpr uint32_t n_replay = 6;

        const llama_pos p_end = (llama_pos) (n_prefix + n_tail);

        const auto make_ctx_n = [&](uint32_t n_seq, int unified) {
            ctx_opts opts;
            opts.n_ctx      = 256;
            opts.n_batch    = 64;
            opts.n_ubatch   = n_ubatch;
            opts.n_seq_max  = n_seq;
            opts.n_rs_seq   = 8;
            opts.kv_unified = unified;
            return mr.ctx(opts);
        };

        const auto prefill = [&](llama_context * ctx, llama_seq_id seq_id) {
            const llama_pos p_mid = (llama_pos) n_prefix;
            return decode_gen(ctx, n_vocab, seq_id, 0, p_mid) &&
                   decode_gen(ctx, n_vocab, seq_id, p_mid, p_mid + (llama_pos) n_tail);
        };

        const auto snap = [&](llama_context * ctx, llama_seq_id seq_id, seq_snap & out) {
            llama_memory_t mem = llama_get_memory(ctx);
            out.pos_min = llama_memory_seq_pos_min(mem, seq_id);
            out.pos_max = llama_memory_seq_pos_max(mem, seq_id);
            out.st.clear();
            return out.pos_max < 0 || get_seq_state(ctx, seq_id, 0, out.st);
        };

        // empty when the sequence is either removed or as it was, the reason otherwise
        const auto check_seq = [&](llama_context * ctx, llama_seq_id seq_id, const seq_snap & before,
                                   bool & dropped) -> std::string {
            seq_snap after;
            if (!snap(ctx, seq_id, after)) {
                return "the state of seq " + std::to_string(seq_id) + " could not be read";
            }

            dropped = after.pos_min < 0 && after.pos_max < 0;
            if (dropped) {
                return {};
            }

            if (after.pos_min != before.pos_min || after.pos_max != before.pos_max) {
                return "seq " + std::to_string(seq_id) + " kept positions it did not have: [" +
                    std::to_string(before.pos_min) + ", " + std::to_string(before.pos_max) + "] became [" +
                    std::to_string(after.pos_min) + ", " + std::to_string(after.pos_max) + "]";
            }

            const state_delta d = state_diff(after.st, before.st);
            if (!d.equal()) {
                return "seq " + std::to_string(seq_id) + " kept its positions but not its state (" +
                    std::to_string(d.n_diff) + " of " + std::to_string(before.st.size()) +
                    " bytes differ, first at offset " + std::to_string(d.i_first) + ")";
            }

            return {};
        };

        // decode the same tokens in both contexts, the logits and then the state have to match
        const auto replay_match = [&](llama_context * ctx, llama_context * ref, llama_seq_id seq_id,
                                      llama_pos p0) -> std::string {
            logits_diff diff;
            for (uint32_t i = 0; i < n_replay; ++i) {
                const llama_pos   pos = p0 + (llama_pos) i;
                const llama_token tok = gen_token(n_vocab, seq_id, pos);

                if (!decode_one(ctx, tok, pos, seq_id) || !decode_one(ref, tok, pos, seq_id)) {
                    return "the replay of seq " + std::to_string(seq_id) + " failed to decode at pos " +
                        std::to_string(pos);
                }

                const float * l_ctx = llama_get_logits_ith(ctx, 0);
                const float * l_ref = llama_get_logits_ith(ref, 0);
                if (l_ctx == nullptr || l_ref == nullptr) {
                    return "the replay of seq " + std::to_string(seq_id) + " has no logits";
                }

                diff.add(l_ctx, l_ref, n_vocab);
            }

            if (diff.nmse() > NMSE_THRESHOLD) {
                return "seq " + std::to_string(seq_id) + " replayed differently from the reference (nmse " +
                    std::to_string(diff.nmse()) + ")";
            }

            std::vector<uint8_t> st_ctx;
            std::vector<uint8_t> st_ref;
            if (!get_seq_state(ctx, seq_id, 0, st_ctx) || !get_seq_state(ref, seq_id, 0, st_ref)) {
                return "the replayed state of seq " + std::to_string(seq_id) + " could not be read";
            }

            const state_delta d = state_diff(st_ctx, st_ref);
            if (!d.equal()) {
                return "seq " + std::to_string(seq_id) + " replayed into a different state (" +
                    std::to_string(d.n_diff) + " of " + std::to_string(st_ref.size()) +
                    " bytes differ, first at offset " + std::to_string(d.i_first) + ")";
            }

            return {};
        };

        const auto seq_rm_ok = [](llama_context * ctx, llama_seq_id seq_id, llama_pos p0) {
            return llama_memory_seq_rm(llama_get_memory(ctx), seq_id, p0, -1);
        };

        {
            llama_context_ptr probe = make_ctx_n(4, 0);
            if (!probe) {
                return case_result::fail();
            }
            if (llama_n_rs_seq(probe.get()) < n_tail) {
                return case_result::skip("n_rs_seq is too small");
            }
        }

        // the order puts the cell of seq_moved between the two members of the failing ubatch, and seq_idle after them
        constexpr llama_seq_id seq_a     = 0;
        constexpr llama_seq_id seq_b     = 1;
        constexpr llama_seq_id seq_moved = 2;
        constexpr llama_seq_id seq_idle  = 3;

        const std::vector<llama_seq_id> order4 = { seq_a, seq_moved, seq_b, seq_idle };

        // leg 1: the abort lands in a ubatch that relocates seq_moved
        {
            llama_context_ptr ctx = make_ctx_n(4, 0);
            llama_context_ptr ref = make_ctx_n(4, 0);
            CTX_REQUIRE(ctx && ref, "the contexts of the abort leg could not be created");

            bool ok = true;
            for (const llama_seq_id s : order4) {
                ok = ok && prefill(ctx.get(), s) && prefill(ref.get(), s);
            }
            CTX_REQUIRE(ok, "the four-sequence prefill failed");

            std::vector<seq_snap> before(4);
            for (const llama_seq_id s : order4) {
                CTX_REQUIRE(snap(ctx.get(), s, before[s]), "the state before the abort could not be read");
            }

            abort_counter ac;
            ac.abort_at = 0;
            llama_set_abort_callback(ctx.get(), abort_counter::cb, &ac);

            int ret = 0;
            {
                common_batch batch(ctx.get());
                batch.add(gen_token(n_vocab, seq_a, p_end), p_end, seq_a, true);
                batch.add(gen_token(n_vocab, seq_b, p_end), p_end, seq_b, true);
                ret = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
            }
            llama_set_abort_callback(ctx.get(), nullptr, nullptr);

            CTX_REQUIRE(ret == 2, "the aborted decode returned " + std::to_string(ret) + ", expected 2");

            std::vector<bool> dropped(4, false);
            for (const llama_seq_id s : order4) {
                bool d = false;
                const std::string why = check_seq(ctx.get(), s, before[s], d);
                CTX_REQUIRE(why.empty(), "after the abort: " + why);
                dropped[s] = d;
            }

            TLOG_INF("%s: abort leg dropped a=%d b=%d moved=%d idle=%d\n", __func__,
                    (int) dropped[seq_a], (int) dropped[seq_b], (int) dropped[seq_moved], (int) dropped[seq_idle]);

            CTX_REQUIRE(dropped[seq_a] && dropped[seq_b], "a member of the aborted ubatch kept its state");
            CTX_REQUIRE(!dropped[seq_idle], "a sequence outside the aborted ubatch was dropped");

            // the reference drops and prefills the same sequences, so that the comparison isolates the failure:
            //   a removal clears tensor rows that a prefill does not write (DSV4 under a dirty cache)
            for (const llama_seq_id s : order4) {
                if (dropped[s]) {
                    CTX_REQUIRE(prefill(ctx.get(), s), "a dropped sequence could not be prefilled again");
                    CTX_REQUIRE(seq_rm_ok(ref.get(), s, -1) && prefill(ref.get(), s),
                            "the reference could not drop and prefill");
                }
            }
            // the survivor keeps its rollback bound and planes: a rollback of its last token serves in both contexts
            CTX_REQUIRE(seq_rm_ok(ctx.get(), seq_idle, p_end - 1) && seq_rm_ok(ref.get(), seq_idle, p_end - 1),
                    "after the abort, the survivor lost the rollback its last ubatch earned");

            for (const llama_seq_id s : order4) {
                const std::string why = replay_match(ctx.get(), ref.get(), s, s == seq_idle ? p_end - 1 : p_end);
                CTX_REQUIRE(why.empty(), "after the abort: " + why);
            }
        }

        // leg 2: three ubatches, the abort lands in the second. The first stays committed, and the third never ran,
        //   so the pending rollback of its sequence survives
        {
            constexpr llama_seq_id s1 = 0;
            constexpr llama_seq_id s2 = 2;
            constexpr llama_seq_id s3 = 4;

            llama_context_ptr ctx = make_ctx_n(6, 0);
            llama_context_ptr ref = make_ctx_n(6, 0);
            llama_context_ptr cal = make_ctx_n(6, 0);
            CTX_REQUIRE(ctx && ref && cal, "the contexts of the split leg could not be created");

            bool ok = true;
            for (const llama_seq_id s : { s1, s2, s3 }) {
                ok = ok && prefill(ctx.get(), s) && prefill(ref.get(), s) && prefill(cal.get(), s);
            }
            CTX_REQUIRE(ok, "the three-sequence prefill failed");

            const llama_pos p_roll = p_end - 1;
            CTX_REQUIRE(seq_rm_ok(ctx.get(), s3, p_roll) && seq_rm_ok(ref.get(), s3, p_roll),
                    "the pending rollback of the third sequence was refused");

            // the polls of the first ubatch, a single token of s1
            abort_counter cal_ac;
            llama_set_abort_callback(cal.get(), abort_counter::cb, &cal_ac);
            ok = decode_one(cal.get(), gen_token(n_vocab, s1, p_end), p_end, s1);
            llama_set_abort_callback(cal.get(), nullptr, nullptr);
            CTX_REQUIRE(ok && cal_ac.n_polls > 0, "the calibration decode failed or polled nothing");

            std::vector<seq_snap> before(6);
            for (const llama_seq_id s : { s1, s2, s3 }) {
                CTX_REQUIRE(snap(ctx.get(), s, before[s]), "the state before the split abort could not be read");
            }

            abort_counter ac;
            ac.abort_at = cal_ac.n_polls;
            llama_set_abort_callback(ctx.get(), abort_counter::cb, &ac);

            int ret = 0;
            {
                common_batch batch(ctx.get());
                batch.add(gen_token(n_vocab, s1, p_end),  p_end,  s1, true);
                batch.add(gen_token(n_vocab, s2, p_end),  p_end,  s2, true);
                batch.add(gen_token(n_vocab, s3, p_roll), p_roll, s3, true);
                ret = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
            }
            llama_set_abort_callback(ctx.get(), nullptr, nullptr);

            CTX_REQUIRE(ret == 2, "the split abort returned " + std::to_string(ret) + ", expected 2");
            CTX_REQUIRE(ac.n_polls > ac.abort_at, "the split abort never fired");

            // the first ubatch is committed: the reference decodes the same token on its own
            CTX_REQUIRE(decode_one(ref.get(), gen_token(n_vocab, s1, p_end), p_end, s1),
                    "the reference first ubatch failed");
            {
                seq_snap after;
                seq_snap after_ref;
                CTX_REQUIRE(snap(ctx.get(), s1, after) && snap(ref.get(), s1, after_ref),
                        "the committed state could not be read");
                CTX_REQUIRE(after.pos_max == p_end, "the first ubatch was not kept (pos_max " +
                        std::to_string(after.pos_max) +
                        ", calibrated polls " + std::to_string(cal_ac.n_polls) + ")");
                const state_delta d = state_diff(after.st, after_ref.st);
                CTX_REQUIRE(d.equal(), "the committed first ubatch differs from the reference (" +
                        std::to_string(d.n_diff) + " bytes)");
            }

            bool dropped2 = false;
            bool dropped3 = false;
            std::string why = check_seq(ctx.get(), s2, before[s2], dropped2);
            CTX_REQUIRE(why.empty(), "after the split abort: " + why);
            CTX_REQUIRE(dropped2, "the sequence of the aborted second ubatch kept its state");

            why = check_seq(ctx.get(), s3, before[s3], dropped3);
            CTX_REQUIRE(why.empty(), "after the split abort: " + why);
            CTX_REQUIRE(!dropped3, "the sequence of the third ubatch, which never ran, was dropped");

            CTX_REQUIRE(prefill(ctx.get(), s2), "the dropped sequence could not be prefilled again");
            CTX_REQUIRE(seq_rm_ok(ref.get(), s2, -1) && prefill(ref.get(), s2),
                    "the reference could not drop and prefill");

            why = replay_match(ctx.get(), ref.get(), s1, p_end + 1);
            CTX_REQUIRE(why.empty(), "after the split abort: " + why);
            why = replay_match(ctx.get(), ref.get(), s2, p_end);
            CTX_REQUIRE(why.empty(), "after the split abort: " + why);
            why = replay_match(ctx.get(), ref.get(), s3, p_roll);
            CTX_REQUIRE(why.empty(), "after the split abort, the pending rollback: " + why);
        }

        // leg 3: a failure after the memory apply and before the graph, in the relocating layout with a pending
        //   rollback on the relocated sequence. Nothing was computed, so nothing may be lost
        {
            llama_context_ptr ctx = make_ctx_n(4, 0);
            llama_context_ptr ref = make_ctx_n(4, 0);
            CTX_REQUIRE(ctx && ref, "the contexts of the pre-graph leg could not be created");

            bool ok = true;
            for (const llama_seq_id s : order4) {
                ok = ok && prefill(ctx.get(), s) && prefill(ref.get(), s);
            }
            CTX_REQUIRE(ok, "the four-sequence prefill failed");

            const llama_pos p_roll = p_end - 1;
            CTX_REQUIRE(seq_rm_ok(ctx.get(), seq_moved, p_roll) && seq_rm_ok(ref.get(), seq_moved, p_roll),
                    "the pending rollback of the relocated sequence was refused");

            std::vector<seq_snap> before(4);
            for (const llama_seq_id s : order4) {
                CTX_REQUIRE(snap(ctx.get(), s, before[s]), "the state before the pre-graph failure could not be read");
            }

            llama_test_fail_before_graph(ctx.get(), 0);
            int ret = 0;
            {
                common_batch batch(ctx.get());
                batch.add(gen_token(n_vocab, seq_a, p_end), p_end, seq_a, true);
                batch.add(gen_token(n_vocab, seq_b, p_end), p_end, seq_b, true);
                ret = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
            }
            llama_test_fail_before_graph(ctx.get(), -1);

            CTX_REQUIRE(ret == -2, "the pre-graph failure returned " + std::to_string(ret) + ", expected -2");

            for (const llama_seq_id s : order4) {
                bool d = false;
                const std::string why = check_seq(ctx.get(), s, before[s], d);
                CTX_REQUIRE(why.empty(), "after the pre-graph failure: " + why);
                CTX_REQUIRE(!d, "a failure before the graph dropped seq " + std::to_string(s));
            }

            // the members of the failed ubatch keep their state but no rollback, since an SWA cache may have
            //   evicted rows for them, while a sequence outside it keeps its bound
            CTX_REQUIRE(!seq_rm_ok(ctx.get(), seq_a, p_end - 1) && !seq_rm_ok(ctx.get(), seq_b, p_end - 1),
                    "after the pre-graph failure, a member of the failed ubatch still accepts a rollback");
            CTX_REQUIRE(seq_rm_ok(ctx.get(), seq_idle, p_end - 1) && seq_rm_ok(ref.get(), seq_idle, p_end - 1),
                    "after the pre-graph failure, the survivor lost the rollback its last ubatch earned");

            for (const llama_seq_id s : order4) {
                const llama_pos p0 = s == seq_moved ? p_roll : (s == seq_idle ? p_end - 1 : p_end);
                const std::string why = replay_match(ctx.get(), ref.get(), s, p0);
                CTX_REQUIRE(why.empty(), "after the pre-graph failure: " + why);
            }
        }

        // leg 4: one token shared by two sequences of one cell, which only a unified cache builds. A failure before
        //   the graph has to give the position back for both owners, an abort has to drop both
        // qwen4exp's PLE n-gram embeddings assert on a shared token, and a probe context finds the archs whose
        //   memory refuses one
        char arch_name[64] = {};
        llama_model_meta_val_str(mr.model, "general.architecture", arch_name, sizeof(arch_name));
        bool shares_tokens = llama_arch_from_string(arch_name) != LLM_ARCH_QWEN4EXP;
        if (shares_tokens) {
            llama_context_ptr probe = make_ctx_n(3, 1);
            CTX_REQUIRE(probe && prefill(probe.get(), 0), "the shared probe could not be prefilled");
            llama_memory_seq_cp(llama_get_memory(probe.get()), 0, 1, -1, -1);

            common_batch batch(probe.get());
            batch.add(gen_token(n_vocab, 0, p_end), p_end, std::vector<llama_seq_id>{ 0, 1 }, true);
            const int ret = llama_process(probe.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
            shares_tokens = ret == 0;
            if (!shares_tokens) {
                TLOG_INF("%s: the memory refuses a shared token (%d), skipping the shared leg\n", __func__, ret);
            }
        }

        for (const bool abort : { false, true }) {
            if (!shares_tokens) {
                break;
            }

            llama_context_ptr ctx = make_ctx_n(3, 1);
            llama_context_ptr ref = make_ctx_n(3, 1);
            CTX_REQUIRE(ctx && ref, "the contexts of the shared leg could not be created");

            // DSV4 queues the copy of its compressor state for the next decode, so one token of an unrelated
            //   sequence makes the copy happen before the state is read
            const auto setup = [&](llama_context * c) {
                if (!prefill(c, 0)) {
                    return false;
                }
                llama_memory_seq_cp(llama_get_memory(c), 0, 1, -1, -1);
                const llama_pos p2 = llama_memory_seq_pos_max(llama_get_memory(c), 2) + 1;
                return decode_gen(c, n_vocab, 2, p2, p2 + 1);
            };
            CTX_REQUIRE(setup(ctx.get()) && setup(ref.get()), "the shared prefill failed");

            std::vector<seq_snap> before(2);
            for (const llama_seq_id s : { 0, 1 }) {
                CTX_REQUIRE(snap(ctx.get(), s, before[s]), "the state before the shared failure could not be read");
            }

            abort_counter ac;
            if (abort) {
                ac.abort_at = 0;
                llama_set_abort_callback(ctx.get(), abort_counter::cb, &ac);
            } else {
                llama_test_fail_before_graph(ctx.get(), 0);
            }

            int ret = 0;
            {
                common_batch batch(ctx.get());
                batch.add(gen_token(n_vocab, 0, p_end), p_end, std::vector<llama_seq_id>{ 0, 1 }, true);
                ret = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
            }
            llama_set_abort_callback(ctx.get(), nullptr, nullptr);
            llama_test_fail_before_graph(ctx.get(), -1);

            const int ret_want = abort ? 2 : -2;
            const std::string phase = std::string("the shared ") + (abort ? "abort" : "pre-graph failure");
            CTX_REQUIRE(ret == ret_want, phase + " returned " +
                    std::to_string(ret) + ", expected " + std::to_string(ret_want));

            for (const llama_seq_id s : { 0, 1 }) {
                bool d = false;
                const std::string why = check_seq(ctx.get(), s, before[s], d);
                CTX_REQUIRE(why.empty(), "after " + phase + ": " + why);
                CTX_REQUIRE(d == abort, std::string("after the shared ") + (abort ? "abort" : "pre-graph failure") +
                        ", seq " + std::to_string(s) + (d ? " was dropped" : " was kept"));
            }

            if (abort) {
                CTX_REQUIRE(setup(ctx.get()), "the shared sequences could not be prefilled again");
                CTX_REQUIRE(seq_rm_ok(ref.get(), 0, -1) && seq_rm_ok(ref.get(), 1, -1) && setup(ref.get()),
                        "the reference could not drop and prefill");
            }

            for (const llama_seq_id s : { 0, 1 }) {
                const std::string why = replay_match(ctx.get(), ref.get(), s, p_end);
                CTX_REQUIRE(why.empty(), "after " + phase + ": " + why);
            }
        }

        // leg 5: the abort cases of the fork the revert comes from (llama.cpp-old fix/recurrent-rollback,
        //   tests/test-recurrent-state-rollback.cpp test_failed_ubatch), so the port meets the same cases. Each
        //   prefix sequence decodes P tokens in the given order, so cells need not follow seq ids
        {
            constexpr llama_pos P = 12;

            struct abort_case {
                const char *              name;
                uint32_t                  n_seq_max;
                bool                      no_rs;        // a context without snapshots
                std::vector<llama_seq_id> prefix_order;
                llama_seq_id              pending;      // verifies 4 tokens, keeps 2 and rolls back 2 first (-1: none)
                bool                      shared;       // seq 1 becomes a copy of seq 0, so both own one cell
                bool                      copy_to_2;    // seq 2 becomes a copy of seq 0
                std::vector<llama_seq_id> coupled;      // one token owned by all of these goes first
                std::vector<llama_seq_id> abort_seqs;   // then n_tokens[i] tokens for abort_seqs[i]
                std::vector<int>          n_tokens;
                int                       countdown;    // abort at this poll, 0 is after the first node
                std::vector<llama_seq_id> check;
            };
            const abort_case cases[] = {
                // control: no abort, the caller drops the sequence, so the restart comparison is checked on its own
                { "control: 16-token ubatch completes, then a drop", 1, false, { 0 }, -1, false, false,
                  {}, { 0 }, { 16 }, -1, { 0 } },
                { "16-token ubatch", 1, false, { 0 }, -1, false, false,
                  {}, { 0 }, { 16 }, 0, { 0 } },
                { "4-token ubatch without snapshots", 1, true, { 0 }, -1, false, false,
                  {}, { 0 }, { 4 }, 0, { 0 } },
                { "4-token verify ubatch mid-graph", 1, false, { 0 }, -1, false, false,
                  {}, { 0 }, { 4 }, 40, { 0 } },
                { "decode after a pending rollback", 1, false, { 0 }, 0, false, false,
                  {}, { 0 }, { 1 }, 0, { 0 } },
                { "ubatch that relocates a non-member", 3, false, { 1, 0, 2 }, -1, false, false,
                  {}, { 1, 2 }, { 1, 1 }, 0, { 0, 1, 2 } },
                { "decode of a seq that shares its cell", 2, false, { 0 }, -1, true, false,
                  {}, { 1 }, { 1 }, 0, { 0, 1 } },
                { "coupled token landing in a freed cell", 3, false, { 0, 1 }, -1, false, true,
                  { 0, 1 }, {}, {}, 40, { 0, 1, 2 } },
                { "coupled token freeing a cell in the range", 5, false, { 0, 3, 2, 4, 1 }, -1, false, false,
                  { 0, 2 }, { 1 }, { 1 }, 40, { 0, 1, 2, 3, 4 } },
                { "first ubatch, the next one's seq pending", 2, false, { 0, 1 }, 1, false, false,
                  {}, { 0, 1 }, { 4, 1 }, 0, { 0, 1 } },
                { "ordered cells, nothing moves", 3, false, { 0, 1, 2 }, -1, false, false,
                  {}, { 1, 2 }, { 1, 1 }, 0, { 0 } },
            };

            // a restart is compared by its replayed logits: a reused cell can keep a stale row that nothing reads
            //   (an orphaned pooled slot of the indexer cache, llama-memory-hybrid-idx.cpp), so its serialized bytes
            //   can differ from a fresh context even after an ordinary removal, which the control row shows

            // DSV4 queues the copy of its compressor state for the next decode, a token of a spare sequence makes it
            const bool queued_copy = rs_from_ubatch_start(mr.model);

            // the indexer pool step pads its new pools with cells that are not reps of complete pools, and writes a
            //   pooled key into their pooled slot, which nothing reads (set_input_kpool, llama-memory-hybrid-idx.cpp).
            //   Such a cell can belong to a surviving sequence, so there its bytes can change while its state holds:
            //   it is then compared by the logits of a replay against a context that never attempted the batch
            const llm_arch arch_lf = llama_arch_from_string(arch_name);
            const bool idx_padding = arch_lf == LLM_ARCH_QWEN4EXP || arch_lf == LLM_ARCH_GLM5_NEXT;

            for (const auto & c : cases) {
                const bool coupled = !c.coupled.empty() || c.shared || c.copy_to_2;
                const bool control = c.countdown < 0;
                GGML_UNUSED(control);
                const bool flush   = queued_copy && (c.shared || c.copy_to_2);
                if (!c.coupled.empty() && !shares_tokens) {
                    continue;
                }

                ctx_opts opts;
                opts.n_ctx      = 256;
                opts.n_batch    = 64;
                opts.n_ubatch   = 64;
                opts.n_seq_max  = c.n_seq_max + (flush ? 1 : 0);
                opts.n_rs_seq   = c.no_rs ? 0 : 8;
                opts.kv_unified = coupled ? 1 : 0;

                llama_context_ptr ctx = mr.ctx(opts);
                CTX_REQUIRE(ctx, std::string(c.name) + ": the context could not be created");
                if (c.no_rs && llama_n_rs_seq(ctx.get()) != 0) {
                    TLOG_INF("%s: %s: skipped, the context keeps snapshots by default\n", __func__, c.name);
                    continue;
                }
                llama_memory_t mem = llama_get_memory(ctx.get());

                std::vector<llama_pos> pos(c.n_seq_max, P - 1);
                const auto setup_case = [&](llama_context * cx) {
                    llama_memory_t mx = llama_get_memory(cx);
                    bool res = true;
                    for (const llama_seq_id sq : c.prefix_order) {
                        res = res && decode_gen(cx, n_vocab, sq, 0, P);
                    }
                    if (c.pending >= 0) {
                        res = res && decode_gen(cx, n_vocab, c.pending, P, P + 4) &&
                              llama_memory_seq_rm(mx, c.pending, P + 2, -1);
                        pos[c.pending] = P + 1;
                    }
                    if (c.shared) {
                        res = res && llama_memory_seq_rm(mx, 1, -1, -1);
                        llama_memory_seq_cp(mx, 0, 1, -1, -1);
                        pos[1] = pos[0];
                    }
                    if (c.copy_to_2) {
                        llama_memory_seq_cp(mx, 0, 2, -1, -1);
                        pos[2] = pos[0];
                    }
                    if (flush) {
                        res = res && decode_gen(cx, n_vocab, (llama_seq_id) c.n_seq_max, 0, 1);
                    }
                    llama_synchronize(cx);
                    return res;
                };
                CTX_REQUIRE(setup_case(ctx.get()), std::string(c.name) + ": the setup failed");

                std::vector<seq_snap> before(c.n_seq_max);
                for (const llama_seq_id sq : c.check) {
                    CTX_REQUIRE(snap(ctx.get(), sq, before[sq]),
                            std::string(c.name) + ": the state before the abort could not be read");
                }

                abort_counter ac;
                ac.abort_at = c.countdown;
                llama_set_abort_callback(ctx.get(), abort_counter::cb, &ac);
                int ret = 0;
                {
                    common_batch batch(ctx.get());
                    if (!c.coupled.empty()) {
                        const llama_pos pc = pos[c.coupled[0]] + 1;
                        batch.add(gen_token(n_vocab, c.coupled[0], pc), pc, c.coupled, true);
                    }
                    for (size_t i = 0; i < c.abort_seqs.size(); ++i) {
                        const llama_seq_id sq = c.abort_seqs[i];
                        for (int k = 0; k < c.n_tokens[i]; ++k) {
                            const llama_pos pk = pos[sq] + 1 + k;
                            batch.add(gen_token(n_vocab, sq, pk), pk, sq, true);
                        }
                    }
                    ret = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
                }
                llama_set_abort_callback(ctx.get(), nullptr, nullptr);

                if (c.countdown < 0) {
                    CTX_REQUIRE(ret == 0, std::string(c.name) + ": the control decode failed");
                    for (const llama_seq_id sq : c.check) {
                        llama_memory_seq_rm(mem, sq, -1, -1);
                    }
                } else if (ret == 0 && c.countdown > 0) {
                    TLOG_INF("%s: %s: skipped, the graph polled fewer than %d times\n", __func__, c.name, c.countdown);
                    continue;
                } else if (ret == 1 && !c.coupled.empty()) {
                    // DSV41 refuses a coupled token whose sequences do not share their history, before any graph
                    TLOG_INF("%s: %s: skipped, the memory refuses the coupled token\n", __func__, c.name);
                    continue;
                }
                else {
                    CTX_REQUIRE(ret == 2, std::string(c.name) + ": the decode returned " + std::to_string(ret) +
                            ", expected 2");
                }

                std::vector<llama_seq_id> dropped;
                std::vector<llama_seq_id> soft;
                for (const llama_seq_id sq : c.check) {
                    bool d = false;
                    const std::string why = check_seq(ctx.get(), sq, before[sq], d);
                    const bool bytes_only = why.find("kept its positions but not its state") != std::string::npos;
                    if (!why.empty() && idx_padding && bytes_only) {
                        soft.push_back(sq);
                        continue;
                    }
                    CTX_REQUIRE(why.empty(), std::string(c.name) + ": " + why);
                    if (d) {
                        dropped.push_back(sq);
                    }
                }

                if (!soft.empty()) {
                    llama_context_ptr ref = mr.ctx(opts);
                    CTX_REQUIRE(ref && setup_case(ref.get()), std::string(c.name) + ": the reference setup failed");
                    for (const llama_seq_id sq : soft) {
                        logits_diff diff;
                        for (llama_pos pr = pos[sq] + 1; pr < pos[sq] + 1 + (llama_pos) n_replay; ++pr) {
                            const llama_token tok = gen_token(n_vocab, sq, pr);
                            CTX_REQUIRE(decode_one(ctx.get(), tok, pr, sq) && decode_one(ref.get(), tok, pr, sq),
                                    std::string(c.name) + ": the replay of a surviving sequence failed");
                            diff.add(llama_get_logits_ith(ctx.get(), 0), llama_get_logits_ith(ref.get(), 0), n_vocab);
                        }
                        CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD, std::string(c.name) + ": seq " + std::to_string(sq) +
                                " survived with a different state (nmse " + std::to_string(diff.nmse()) + ")");
                        TLOG_INF("%s: %s: seq %d survived with unread pooled bytes changed, its replay matches\n",
                                __func__, c.name, sq);
                    }
                }

                // a dropped sequence must be gone from every memory, or its restart sees old attention. The reference
                //   sees the same removal, which for DSV4 also zeroes the rows of the stream
                if (!dropped.empty()) {
                    llama_context_ptr ref = mr.ctx(opts);
                    CTX_REQUIRE(ref, std::string(c.name) + ": the reference could not be created");
                    for (const llama_seq_id sq : dropped) {
                        llama_memory_seq_rm(llama_get_memory(ref.get()), sq, -1, -1);
                        CTX_REQUIRE(decode_gen(ctx.get(), n_vocab, sq, 0, 5) &&
                                    decode_gen(ref.get(), n_vocab, sq, 0, 5),
                                std::string(c.name) + ": the restart of a dropped sequence failed");

                        // old attention left behind would change the logits of the replay
                        logits_diff diff;
                        for (llama_pos pr = 5; pr < 5 + (llama_pos) n_replay; ++pr) {
                            const llama_token tok = gen_token(n_vocab, sq, pr);
                            CTX_REQUIRE(decode_one(ctx.get(), tok, pr, sq) && decode_one(ref.get(), tok, pr, sq),
                                    std::string(c.name) + ": the replay of a restarted sequence failed");
                            diff.add(llama_get_logits_ith(ctx.get(), 0), llama_get_logits_ith(ref.get(), 0), n_vocab);
                        }
                        CTX_REQUIRE(diff.nmse() <= NMSE_THRESHOLD, std::string(c.name) + ": seq " + std::to_string(sq) +
                                " restarted after the drop replays differently from a fresh context (nmse " +
                                std::to_string(diff.nmse()) + ")");

                        std::vector<uint8_t> st_ctx;
                        std::vector<uint8_t> st_ref;
                        CTX_REQUIRE(get_seq_state(ctx.get(), sq, 0, st_ctx) && get_seq_state(ref.get(), sq, 0, st_ref),
                                std::string(c.name) + ": the restarted state could not be read");
                        const state_delta d = state_diff(st_ctx, st_ref);
                        if (!d.equal()) {
                            TLOG_INF("%s: %s: seq %d restarted with %zu unread bytes unlike a fresh context\n",
                                    __func__, c.name, sq, d.n_diff);
                        }
                    }
                }

                TLOG_INF("%s: %s: %zu of %zu checked sequences dropped\n", __func__, c.name, dropped.size(),
                        c.check.size());
            }
        }

        TLOG_INF("%s: every failed ubatch left its sequences removed or intact (shared leg %s)\n", __func__,
                shares_tokens ? "run" : "skipped");

        return case_result::pass();
    }
};

std::vector<ctx_test_ptr> make_rs_cases() {
    std::vector<ctx_test_ptr> cases;

    cases.push_back(std::make_unique<case_rollback>());
    cases.push_back(std::make_unique<case_rollback_bound>());
    cases.push_back(std::make_unique<case_relocate>());
    cases.push_back(std::make_unique<case_multi_seq_split_replay>());
    cases.push_back(std::make_unique<case_shared_seq_reserve>());
    cases.push_back(std::make_unique<case_failed_ubatch>());

    return cases;
}
