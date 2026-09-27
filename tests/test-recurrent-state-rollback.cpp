#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include "../src/llama-io.h"
#include "../src/llama-memory.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <vector>

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

static llama_context * init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context * ctx = llama_init_from_model(model, cparams);
    // without rollback there are no snapshot planes to poison
    if (ctx == nullptr || fill == 0 || llama_n_rs_seq(ctx) == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx);
    if (!decode_tokens(ctx, std::vector<llama_token>(n_tokens, 0), n_tokens)) {
        llama_free(ctx);
        return nullptr;
    }
    llama_synchronize(ctx);
    cache_buffer_collector collector;
    llama_get_memory(ctx)->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (collector.buffers.empty()) {
        fprintf(stderr, "%s : no cache buffers found\n", __func__);
        llama_free(ctx);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint8_t fill) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return init_ctx(model, cparams, fill);
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

static double nmse(const float * a, const float * b, int n) {
    double mse_ab = 0.0;
    double mse_a0 = 0.0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<double>::infinity();
        }
        const double diff = (double) a[i] - b[i];
        mse_ab += diff*diff;
        mse_a0 += (double) a[i]*a[i];
    }
    return mse_a0 == 0.0 ? (mse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : mse_ab/mse_a0;
}

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
// The rolled-back tail is one ubatch of n_rollback + 1 tokens, so the anchor
// token stays and the snapshot planes can serve the removal.
static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab, uint8_t fill) {
    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;
    constexpr llama_pos p_tail     = p0 - 1; // the anchor

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return init_ctx(model, cparams, fill);
    };

    llama_context * ctx_roll = make_ctx_multi();
    llama_context * ctx_ref  = make_ctx_multi();
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init multi-seq contexts\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_roll);
        llama_free(ctx_ref);
    };

    if (llama_n_rs_seq(ctx_roll) < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // ctx_ref decodes the [0, p0) prefill; ctx_roll stops one token earlier and
    // decodes the anchor with the tail, which is then rolled back so its restore
    // is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < (llama_pos) p0; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_ref, batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = 0; pos < p_tail; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = p_tail; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        llama_batch_free(batch);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0, -1);

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        fprintf(stderr, "%s : multi-seq prefill/rollback failed\n", __func__);
        cleanup();
        return false;
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = llama_decode(ctx_roll, batch) == 0;
    ok = ok && llama_decode(ctx_ref, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        fprintf(stderr, "%s : multi-seq replay decode failed\n", __func__);
        cleanup();
        return false;
    }

    // identical ubatch shapes should produce identical states, but the larger
    // stdev makes the model sensitive to backend scheduling/rounding noise
    constexpr float nmse_eps = 1e-5f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    double   nmse_ab   = 0.0;
    double   nmse_a0   = 0.0;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll, i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing multi-seq logits at index %u\n", __func__, i);
            cleanup();
            return false;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            const float diff = logit_diff(r, f);
            if (diff > 0.0f && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_ab += d*d;
                nmse_a0 += (double) r*r;
            } else {
                nmse_ab = std::numeric_limits<double>::infinity();
                nmse_a0 = 1.0;
            }
        }
    }
    const double nmse_val = nmse_a0 == 0.0 ? (nmse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_ab/nmse_a0;

    if (nmse_val > nmse_eps) {
        fprintf(stderr, "%s : multi-seq split replay logits mismatch (max diff %g, nmse %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, nmse_val, seq_first, pos_first);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff_max, nmse_val);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref, batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    double nmse_tail_ab = 0.0;
    double nmse_tail_a0 = 0.0;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll, batch_one) == 0;
        ok = ok && llama_decode(ctx_ref, batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            diff_tail = std::max(diff_tail, logit_diff(r, f));
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_tail_ab += d*d;
                nmse_tail_a0 += (double) r*r;
            } else {
                nmse_tail_ab = std::numeric_limits<double>::infinity();
                nmse_tail_a0 = 1.0;
            }
        }
    }
    const double nmse_tail = nmse_tail_a0 == 0.0 ? (nmse_tail_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_tail_ab/nmse_tail_a0;

    if (!ok || nmse_tail > nmse_eps) {
        fprintf(stderr, "%s : seq-1-only decode leaked seq 0 state (ok=%d, max diff %g, nmse %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail, nmse_tail);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (max diff %g, nmse %g)\n", __func__, (double) diff_tail, nmse_tail);
    cleanup();
    return true;
}

static int test_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    // TODO: use smart pointers
    llama_context * ctx_src = make_ctx(params, model, fill);
    llama_context * ctx_dst = make_ctx(params, model, fill);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src, "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    constexpr float nmse_eps = 0.0;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src, tokens[pos], pos) ||
                !decode_one(ctx_dst, tokens[pos], pos)) {
                fprintf(stderr, "%s : %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src, 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                fprintf(stderr, "%s : missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            const double nmse_val = nmse(logits_src, logits_dst, n_vocab);
            int token_first = -1;
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > 0.0f && token_first < 0) {
                    token_first = token;
                }
            }
            if (nmse_val > nmse_eps) {
                fprintf(stderr, "%s : %s logits mismatch at position %d, first token %d, nmse %g\n",
                        __func__, mode, pos, token_first, nmse_val);
                return false;
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return 1;
    }

    // TODO: this test is invalid because RS rollback is only correct once after a ubatch with more than n_rs_seq tokens
    //       this is not the case here. add asserts and guardrails to prevent such attempts
    //if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
    //    !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
    //    fprintf(stderr, "%s : partial rollback failed\n", __func__);
    //    return 1;
    //}

    //constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    //common_prompt_checkpoint ckpt_partial;
    //ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    //ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    //if (!replay_and_compare("partial")) {
    //    return 1;
    //}

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model, fill);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty, tokens[pos], pos)) {
            fprintf(stderr, "%s : dirty replay failed at position %d\n", __func__, pos);
            return 1;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
        if (logits_dirty == nullptr) {
            fprintf(stderr, "%s : missing dirty logits at position %d\n", __func__, pos);
            return 1;
        }

        const double nmse_dirty = nmse(logits_src_replay[i].data(), logits_dirty, n_vocab);
        int token_first = -1;
        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_src_replay[i][token], logits_dirty[token]) > 0.0f && token_first < 0) {
                token_first = token;
            }
        }
        if (nmse_dirty > nmse_eps) {
            fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, first token %d, nmse %g\n",
                    __func__, pos, token_first, nmse_dirty);
            return 1;
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);

    if (!test_multi_seq_split_replay(params, model, n_vocab, fill)) {
        return 1;
    }

    return 0;
}

//
// multi-sequence lifecycle tests
//
// Each test drives a rolled-back context and a reference context through decodes of identical ubatch shapes and
// compares the recurrent state a sequence continues from, as exported by the public state API, bitwise.

struct tokspec {
    llama_seq_id seq;
    llama_pos    pos;
    llama_token  tok;
};

// deterministic token per (seq, pos); a non-zero salt gives a different stream, e.g. rejected drafts
static llama_token tok_at(int n_vocab, llama_seq_id seq, llama_pos pos, int salt = 0) {
    return (llama_token) ((7u*(uint32_t) pos + 31u*(uint32_t) seq + 101u*(uint32_t) salt + 1u) % (uint32_t) n_vocab);
}

static std::vector<tokspec> tok_run(int n_vocab, llama_seq_id seq, llama_pos p0, int n, int salt = 0) {
    std::vector<tokspec> v;
    for (int i = 0; i < n; ++i) {
        v.push_back({ seq, p0 + i, tok_at(n_vocab, seq, p0 + i, salt) });
    }
    return v;
}

static std::vector<tokspec> tok_cat(std::vector<tokspec> a, const std::vector<tokspec> & b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

static int decode_specs(llama_context * ctx, const std::vector<tokspec> & toks) {
    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (const auto & t : toks) {
        common_batch_add(batch, t.tok, t.pos, { t.seq }, true);
    }
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return ret;
}

// The recurrent rows one sequence continues from, in state export order and honouring a pending rollback index.
// Attention KV rows are skipped so that hybrid models compare only the state under test.
struct rs_row {
    ggml_type type;
    std::vector<uint8_t> bytes;
};

struct rs_state_collector : llama_io_write_i {
    std::vector<rs_row> rows;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t n) override {
        size += n;
        const std::string name = ggml_get_name(tensor);
        const bool recurrent = name.rfind("cache_r_l", 0) == 0 || name.rfind("cache_s_l", 0) == 0 ||
                               name.rfind("cache_ple_r_l", 0) == 0 || name.rfind("dsv4_", 0) == 0;
        if (!recurrent || n == 0) {
            return;
        }
        rs_row row { tensor->type, std::vector<uint8_t>(n) };
        ggml_backend_tensor_get(tensor, row.bytes.data(), offset, n);
        rows.push_back(std::move(row));
    }

    size_t n_bytes() override {
        return size;
    }
};

static std::vector<rs_row> seq_state(llama_context * ctx, llama_seq_id seq) {
    llama_synchronize(ctx);
    rs_state_collector collector;
    llama_get_memory(ctx)->state_write(collector, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    return collector.rows;
}

struct state_cmp {
    bool        same = false;
    double      max_abs = std::numeric_limits<double>::infinity();
    double      nmse    = std::numeric_limits<double>::infinity();
    std::string text; // empty when the states are equal
};

// Exact comparison, except that -0 and +0 are equal: the zeroed state is made by scaling the cell's previous
// content by 0, which keeps the sign of negative values.
static state_cmp compare_states(const std::vector<rs_row> & a, const std::vector<rs_row> & b) {
    state_cmp res;
    if (a.size() != b.size()) {
        res.text = "(row count differs)";
        return res;
    }
    size_t n_rows  = 0;
    size_t n_elems = 0;
    size_t first_row  = 0;
    size_t first_elem = 0;
    double max_abs = 0.0;
    double sum_sq_diff = 0.0;
    double sum_sq_ref  = 0.0;
    for (size_t r = 0; r < a.size(); ++r) {
        if (a[r].type != b[r].type || a[r].bytes.size() != b[r].bytes.size()) {
            res.text = "(row layout differs)";
            return res;
        }
        const size_t es = ggml_type_size(a[r].type);
        size_t n_diff = 0;
        for (size_t i = 0; i < a[r].bytes.size()/es; ++i) {
            const uint8_t * x = a[r].bytes.data() + i*es;
            const uint8_t * y = b[r].bytes.data() + i*es;
            bool equal;
            if (a[r].type == GGML_TYPE_F32 || a[r].type == GGML_TYPE_F16) {
                float fx, fy;
                if (a[r].type == GGML_TYPE_F32) {
                    memcpy(&fx, x, sizeof(fx));
                    memcpy(&fy, y, sizeof(fy));
                } else {
                    ggml_fp16_t hx, hy;
                    memcpy(&hx, x, sizeof(hx));
                    memcpy(&hy, y, sizeof(hy));
                    fx = ggml_fp16_to_fp32(hx);
                    fy = ggml_fp16_to_fp32(hy);
                }
                equal = fx == fy;
                const double d = (double) fx - fy;
                max_abs = std::isfinite(d) ? std::max(max_abs, std::fabs(d)) : std::numeric_limits<double>::infinity();
                sum_sq_diff += d*d;
                sum_sq_ref  += (double) fy*fy;
            } else {
                equal = memcmp(x, y, es) == 0;
            }
            if (!equal) {
                if (n_elems == 0 && n_diff == 0) {
                    first_row  = r;
                    first_elem = i;
                }
                ++n_diff;
            }
        }
        if (n_diff > 0) {
            ++n_rows;
            n_elems += n_diff;
        }
    }
    res.max_abs = max_abs;
    res.nmse    = sum_sq_ref > 0.0 ? sum_sq_diff/sum_sq_ref : (sum_sq_diff > 0.0 ? std::numeric_limits<double>::infinity() : 0.0);
    res.same    = n_rows == 0;
    if (!res.same) {
        char buf[192];
        snprintf(buf, sizeof(buf), "(%zu rows, %zu elements differ, first at row %zu element %zu, max abs %g, nmse %g)",
                 n_rows, n_elems, first_row, first_elem, res.max_abs, res.nmse);
        res.text = buf;
    }
    return res;
}

static std::string state_diff(const std::vector<rs_row> & a, const std::vector<rs_row> & b) {
    return compare_states(a, b).text;
}

struct ctx_spec {
    uint32_t n_seq_max;
    uint32_t n_rs_seq;
    uint32_t n_ubatch;
    llama_attention_type attention_type = LLAMA_ATTENTION_TYPE_UNSPECIFIED;
};

static llama_context * make_ctx_multi(const common_params & params, llama_model * model, uint8_t fill, const ctx_spec & spec) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max      = spec.n_seq_max;
    cparams.n_rs_seq       = spec.n_rs_seq;
    cparams.n_ctx          = 1024;
    cparams.n_batch        = 512;
    cparams.n_ubatch       = spec.n_ubatch;
    cparams.kv_unified     = true;
    cparams.attention_type = spec.attention_type;
    return init_ctx(model, cparams, fill);
}

// the exported state must tell two histories apart, otherwise every bitwise comparison below is vacuous
static bool test_state_instrument(const common_params & params, llama_model * model, int n_vocab, uint8_t fill) {
    const ctx_spec spec = { 1, 8, 64 };
    llama_context * a = make_ctx_multi(params, model, fill, spec);
    llama_context * b = make_ctx_multi(params, model, fill, spec);
    if (a == nullptr || b == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return false;
    }
    bool ok = decode_specs(a, tok_run(n_vocab, 0, 0, 12)) == 0 && decode_specs(b, tok_run(n_vocab, 0, 0, 12)) == 0;
    const auto sa = seq_state(a, 0);
    ok = ok && !sa.empty() && state_diff(sa, seq_state(b, 0)).empty();
    ok = ok && decode_specs(b, tok_run(n_vocab, 0, 12, 1)) == 0;
    ok = ok && !state_diff(sa, seq_state(b, 0)).empty();
    llama_free(a);
    llama_free(b);
    if (!ok) {
        fprintf(stderr, "%s : exported state does not discriminate histories\n", __func__);
        return false;
    }
    return true;
}

// A pending rollback index must not outlive the sequence's stay in its cell. After seq_keep or a finite seq_rm the
// next conversation on that seq id must start from the zero state, and a seq_cp from a rolled-back sequence must
// continue from the rolled-back state. Each variant runs once without a pending index as its control.
static bool test_pending_index_lifecycle(const common_params & params, llama_model * model, int n_vocab, uint8_t fill) {
    constexpr llama_pos P = 12;
    constexpr llama_pos R = 2;
    const ctx_spec spec = { 2, 8, 64 };

    enum op_t { OP_SEQ_KEEP, OP_FINITE_RM, OP_SEQ_CP_ONTO, OP_SEQ_CP_FROM };
    const std::pair<op_t, const char *> variants[] = {
        { OP_SEQ_KEEP,    "seq_keep"           },
        { OP_FINITE_RM,   "finite seq_rm"      },
        { OP_SEQ_CP_ONTO, "seq_cp onto seq 1"  },
        { OP_SEQ_CP_FROM, "seq_cp from seq 0"  },
    };

    bool all_ok = true;
    for (const auto & [op, name] : variants) {
        for (bool pending : { false, true }) {
            llama_context * roll = make_ctx_multi(params, model, fill, spec);
            llama_context * ref  = make_ctx_multi(params, model, fill, spec);
            if (roll == nullptr || ref == nullptr) {
                fprintf(stderr, "%s : failed to init contexts\n", __func__);
                return false;
            }
            if (llama_n_rs_seq(roll) < (uint32_t) R) {
                fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
                llama_free(roll);
                llama_free(ref);
                return true;
            }
            auto * mem = llama_get_memory(roll);

            bool ok = decode_specs(roll, tok_run(n_vocab, 0, 0, P)) == 0 &&
                      decode_specs(roll, tok_run(n_vocab, 1, 0, P)) == 0;

            // the rolled-back sequence keeps its anchor, so the request is one the snapshots can serve
            const llama_seq_id seq_pending = op == OP_SEQ_CP_FROM ? 0 : 1;
            if (pending) {
                ok = ok && llama_memory_seq_rm(mem, seq_pending, P - R, -1);
            }

            std::vector<tokspec> next_roll;
            std::vector<tokspec> next_ref;
            llama_seq_id seq_ref = 0;
            switch (op) {
                case OP_SEQ_KEEP:
                case OP_FINITE_RM:
                    if (op == OP_SEQ_KEEP) {
                        llama_memory_seq_keep(mem, 0);
                    } else if (!llama_memory_seq_rm(mem, 1, 0, 100000)) {
                        fprintf(stderr, "%s : %s%s: skipped, this memory refuses a finite removal\n", __func__, pending ? "" : "control ", name);
                        llama_free(roll);
                        llama_free(ref);
                        continue;
                    }
                    // a new conversation starts on seq 1 from the zero state
                    next_roll = { { 1, 0, tok_at(n_vocab, 1, 0, 3) } };
                    next_ref  = { { 1, 0, tok_at(n_vocab, 1, 0, 3) } };
                    seq_ref   = 1;
                    break;
                case OP_SEQ_CP_ONTO:
                    // seq 1 becomes a copy of seq 0 and continues seq 0's conversation. The removal first is the
                    // server's protocol: the unified KV cache's seq_cp only adds the destination to the source cells.
                    ok = ok && llama_memory_seq_rm(mem, 1, -1, -1);
                    llama_memory_seq_cp(mem, 0, 1, -1, -1);
                    ok = ok && decode_specs(ref, tok_run(n_vocab, 0, 0, P)) == 0;
                    next_roll = { { 1, P, tok_at(n_vocab, 0, P) } };
                    next_ref  = { { 0, P, tok_at(n_vocab, 0, P) } };
                    break;
                case OP_SEQ_CP_FROM: {
                    // seq 1 becomes a copy of seq 0 while seq 0's rollback is pending, and continues from there
                    ok = ok && llama_memory_seq_rm(mem, 1, -1, -1);
                    llama_memory_seq_cp(mem, 0, 1, -1, -1);
                    ok = ok && decode_specs(ref, tok_run(n_vocab, 0, 0, P)) == 0;
                    if (pending) {
                        ok = ok && llama_memory_seq_rm(llama_get_memory(ref), 0, P - R, -1);
                    }
                    const llama_pos p = pending ? P - R : P;
                    next_roll = { { 1, p, tok_at(n_vocab, 0, p) } };
                    next_ref  = { { 0, p, tok_at(n_vocab, 0, p) } };
                    break;
                }
            }

            ok = ok && decode_specs(roll, next_roll) == 0;
            ok = ok && decode_specs(ref,  next_ref)  == 0;
            if (!ok) {
                fprintf(stderr, "%s : %s%s: setup failed\n", __func__, pending ? "" : "control ", name);
                llama_free(roll);
                llama_free(ref);
                return false;
            }

            const std::string diff = state_diff(seq_state(roll, 1), seq_state(ref, seq_ref));
            llama_free(roll);
            llama_free(ref);

            fprintf(stderr, "%s : %s%s: state after continue %s the reference %s\n", __func__,
                    pending ? "" : "control ", name, diff.empty() ? "matches" : "DIFFERS from", diff.c_str());
            all_ok = all_ok && diff.empty();
        }
    }
    return all_ok;
}

struct abort_ctl {
    int countdown = -1; // number of callback checks to let through before aborting, -1 for never
};

static bool abort_cb(void * data) {
    auto * ctl = (abort_ctl *) data;
    if (ctl->countdown < 0) {
        return false;
    }
    if (ctl->countdown == 0) {
        ctl->countdown = -1;
        return true;
    }
    --ctl->countdown;
    return false;
}

// After a decode fails part-way, a sequence must either be gone or stand exactly where it stood before the failed
// ubatch, with the same state. Claiming positions the graph never computed is the failure this checks for.
static bool check_after_failure(llama_context * ctx, llama_seq_id seq, llama_pos pos_before,
                                const std::vector<rs_row> & state_before, const char * what) {
    const llama_pos pmax = llama_memory_seq_pos_max(llama_get_memory(ctx), seq);
    if (pmax == -1) {
        fprintf(stderr, "%s : %s: seq %d was dropped\n", __func__, what, seq);
        return true;
    }
    if (pmax != pos_before) {
        fprintf(stderr, "%s : %s: seq %d claims positions up to %d that were never computed\n", __func__, what, seq, pmax);
        return false;
    }
    const std::string diff = state_diff(seq_state(ctx, seq), state_before);
    if (!diff.empty()) {
        fprintf(stderr, "%s : %s: seq %d kept its position but its state changed %s\n", __func__, what, seq, diff.c_str());
        return false;
    }
    fprintf(stderr, "%s : %s: seq %d is intact\n", __func__, what, seq);
    return true;
}

// A ubatch that fails after find_slot has claimed its positions, and may have moved or partly written cells.
// Afterwards every sequence must be either gone or exactly as it was. The cases cover a failed span longer than the
// snapshot depth, no snapshots at all, a verify ubatch, a decode with a pending rollback, and a ubatch that
// relocates a sequence which is not in it.
static bool test_failed_ubatch(const common_params & params, llama_model * model, int n_vocab, uint8_t fill) {
    constexpr llama_pos P = 12;

    struct abort_case {
        const char *              name;
        uint32_t                  n_seq_max;
        uint32_t                  n_rs_seq;
        std::vector<llama_seq_id> prefix_order; // each decodes P tokens, in this order, so cells need not follow seq ids
        bool                      pending;      // seq 0 verifies 4 tokens, accepts 2 and rolls back 2 first
        bool                      shared;       // seq 1 becomes a copy of seq 0, so both own one cell
        std::vector<llama_seq_id> abort_seqs;   // the aborted ubatch: n_tokens for each of these
        int                       n_tokens;
        int                       countdown;    // abort at this callback check: 0 is before any node
        std::vector<llama_seq_id> check;
    };
    const abort_case cases[] = {
        { "abort of a 16-token ubatch with n_rs_seq 8",       1, 8, { 0 },       false, false, { 0 },    16,  0, { 0 } },
        { "abort of a 4-token ubatch with n_rs_seq 0",        1, 0, { 0 },       false, false, { 0 },     4,  0, { 0 } },
        { "abort of a 4-token verify ubatch",                 1, 8, { 0 },       false, false, { 0 },     4,  0, { 0 } },
        { "abort of a 4-token verify ubatch mid-graph",       1, 8, { 0 },       false, false, { 0 },     4, 40, { 0 } },
        { "abort of a decode after a pending rollback",       1, 8, { 0 },       true,  false, { 0 },     1,  0, { 0 } },
        { "abort of a ubatch that relocates a non-member",    3, 8, { 1, 0, 2 }, false, false, { 1, 2 },  1,  0, { 0, 1, 2 } },
        { "abort of a decode of a seq that shares its cell",  2, 8, { 0 },       false, true,  { 1 },     1,  0, { 0, 1 } },
        { "control: abort with ordered cells, nothing moves", 3, 8, { 0, 1, 2 }, false, false, { 1, 2 },  1,  0, { 0 } },
    };

    bool all_ok = true;
    for (const auto & c : cases) {
        llama_context * ctx = make_ctx_multi(params, model, fill, { c.n_seq_max, c.n_rs_seq, 64 });
        if (ctx == nullptr) {
            fprintf(stderr, "%s : failed to init context\n", __func__);
            return false;
        }
        auto * mem = llama_get_memory(ctx);
        if (c.pending && llama_n_rs_seq(ctx) == 0) {
            fprintf(stderr, "%s : %s: skipped, rollback is disabled for this context\n", __func__, c.name);
            llama_free(ctx);
            continue;
        }
        abort_ctl ctl;
        llama_set_abort_callback(ctx, abort_cb, &ctl);

        bool ok = true;
        for (const llama_seq_id s : c.prefix_order) {
            ok = ok && decode_specs(ctx, tok_run(n_vocab, s, 0, P)) == 0;
        }
        llama_pos pos0 = P - 1;
        if (c.pending) {
            ok = ok && decode_specs(ctx, tok_cat(tok_run(n_vocab, 0, P, 2), tok_run(n_vocab, 0, P + 2, 2, 1))) == 0;
            ok = ok && llama_memory_seq_rm(mem, 0, P + 2, -1);
            pos0 = P + 1;
        }
        if (c.shared) {
            ok = ok && llama_memory_seq_rm(mem, 1, -1, -1);
            llama_memory_seq_cp(mem, 0, 1, -1, -1);
        }
        std::vector<std::vector<rs_row>> before;
        for (const llama_seq_id s : c.check) {
            before.push_back(seq_state(ctx, s));
        }

        std::vector<tokspec> aborted;
        for (const llama_seq_id s : c.abort_seqs) {
            aborted = tok_cat(aborted, tok_run(n_vocab, s, s == 0 ? pos0 + 1 : P, c.n_tokens, 1));
        }
        ctl.countdown = c.countdown;
        const int rc = decode_specs(ctx, aborted);
        ctl.countdown = -1;
        if (!ok) {
            fprintf(stderr, "%s : %s: setup failed\n", __func__, c.name);
            llama_free(ctx);
            return false;
        }
        if (rc == 0) {
            fprintf(stderr, "%s : %s: skipped, the decode was not aborted\n", __func__, c.name);
            llama_free(ctx);
            continue;
        }
        for (size_t i = 0; i < c.check.size(); ++i) {
            const llama_seq_id s = c.check[i];
            all_ok = check_after_failure(ctx, s, s == 0 ? pos0 : P - 1, before[i], c.name) && all_ok;
        }
        llama_free(ctx);
    }

    // control: the same 16-token ubatch completes and leaves the same state as a reference
    {
        llama_context * roll = make_ctx_multi(params, model, fill, { 1, 8, 64 });
        llama_context * ref  = make_ctx_multi(params, model, fill, { 1, 8, 64 });
        if (roll == nullptr || ref == nullptr) {
            fprintf(stderr, "%s : failed to init contexts\n", __func__);
            return false;
        }
        abort_ctl ctl;
        llama_set_abort_callback(roll, abort_cb, &ctl);
        bool ok = decode_specs(roll, tok_run(n_vocab, 0, 0, P)) == 0 && decode_specs(ref, tok_run(n_vocab, 0, 0, P)) == 0;
        ok = ok && decode_specs(roll, tok_run(n_vocab, 0, P, 16, 1)) == 0 && decode_specs(ref, tok_run(n_vocab, 0, P, 16, 1)) == 0;
        const std::string diff = ok ? state_diff(seq_state(roll, 0), seq_state(ref, 0)) : "(decode failed)";
        llama_free(roll);
        llama_free(ref);
        fprintf(stderr, "%s : control 16-token ubatch without abort: state %s the reference %s\n", __func__,
                diff.empty() ? "matches" : "DIFFERS from", diff.c_str());
        all_ok = all_ok && diff.empty();
    }

    return all_ok;
}

// One llama_decode at the server's shape: a verify ubatch for one sequence (accepted tokens, then rejected drafts)
// and a one-token ubatch for the others, whose find_slot may relocate the verified sequence. The verify trim then
// has to read the relocated sequence's own snapshot. A context whose cells do not follow seq ids runs the same
// batches as a reference with ordered cells, so every state must match exactly, before and after the trim is used.
struct reloc_round {
    llama_seq_id              verify;
    std::vector<llama_seq_id> others;
    int                       accepted;
    std::vector<llama_seq_id> others2 = {}; // a second one-token ubatch after the trim, so the trimmed sequence
                                            // is relocated or gathered while its rollback is pending
};

static bool run_relocation_rounds(const common_params & params, llama_model * model, int n_vocab, uint8_t fill,
                                  const char * name, uint32_t n_seq_max, const std::vector<llama_seq_id> & order,
                                  const std::vector<reloc_round> & rounds) {
    constexpr llama_pos P = 12;
    constexpr uint32_t  K = 3; // the server's draft n_max

    llama_context * roll = make_ctx_multi(params, model, fill, { n_seq_max, K, 512 });
    llama_context * ref  = make_ctx_multi(params, model, fill, { n_seq_max, K, 512 });
    if (roll == nullptr || ref == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return false;
    }
    if (llama_n_rs_seq(roll) < K) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        llama_free(roll);
        llama_free(ref);
        return true;
    }

    bool ok = true;
    for (const llama_seq_id s : order) {
        ok = ok && decode_specs(roll, tok_run(n_vocab, s, 0, P)) == 0;
    }
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        ok = ok && decode_specs(ref, tok_run(n_vocab, (llama_seq_id) s, 0, P)) == 0;
    }

    const auto compare_all = [&](const char * when, int round) {
        bool same = true;
        for (uint32_t s = 0; s < n_seq_max; ++s) {
            const std::string diff = state_diff(seq_state(roll, (llama_seq_id) s), seq_state(ref, (llama_seq_id) s));
            if (!diff.empty()) {
                fprintf(stderr, "%s : %s round %d: seq %u %s DIFFERS from the reference %s\n", __func__, name, round, s, when, diff.c_str());
                same = false;
            }
        }
        return same;
    };

    bool all_ok = true;
    std::vector<llama_pos> pos(n_seq_max, P);
    for (size_t i = 0; i < rounds.size() && ok; ++i) {
        const auto & r = rounds[i];
        const llama_pos  p = pos[r.verify];
        const int rollback = (int) K + 1 - r.accepted;

        std::vector<tokspec> batch = tok_cat(tok_run(n_vocab, r.verify, p, r.accepted),
                                             tok_run(n_vocab, r.verify, p + r.accepted, rollback, 1));
        for (const llama_seq_id o : r.others) {
            batch = tok_cat(batch, tok_run(n_vocab, o, pos[o], 1));
            pos[o] += 1;
        }
        ok = ok && decode_specs(roll, batch) == 0 && decode_specs(ref, batch) == 0;
        ok = ok && llama_memory_seq_rm(llama_get_memory(roll), r.verify, p + r.accepted, -1);
        ok = ok && llama_memory_seq_rm(llama_get_memory(ref),  r.verify, p + r.accepted, -1);
        pos[r.verify] = p + r.accepted;
        if (!r.others2.empty()) {
            std::vector<tokspec> again;
            for (const llama_seq_id o : r.others2) {
                again = tok_cat(again, tok_run(n_vocab, o, pos[o], 1));
                pos[o] += 1;
            }
            ok = ok && decode_specs(roll, again) == 0 && decode_specs(ref, again) == 0;
        }
        if (!ok) {
            break;
        }
        bool same = compare_all("after the trim", (int) i);

        // use the trimmed state
        const std::vector<tokspec> next = { { r.verify, pos[r.verify], tok_at(n_vocab, r.verify, pos[r.verify]) } };
        ok = ok && decode_specs(roll, next) == 0 && decode_specs(ref, next) == 0;
        pos[r.verify] += 1;
        if (!ok) {
            break;
        }
        same = compare_all("after continuing", (int) i) && same;

        fprintf(stderr, "%s : %s round %d: trim of %d after relocation %s the reference\n", __func__, name, (int) i, rollback,
                same ? "matches" : "DIFFERS from");
        all_ok = all_ok && same;
    }
    if (!ok) {
        fprintf(stderr, "%s : %s: setup failed\n", __func__, name);
        all_ok = false;
    }

    fprintf(stderr, "%s : %s: graphs reused %d times (reference %d)\n", __func__, name,
            llama_perf_context(roll).n_reused, llama_perf_context(ref).n_reused);

    llama_free(roll);
    llama_free(ref);
    return all_ok;
}

static bool test_relocation(const common_params & params, llama_model * model, int n_vocab, uint8_t fill) {
    bool all_ok = true;

    // cells seq1, seq0, seq2: the {seq1, seq2} ubatch gathers seq2 into seq0's cell and pushes seq0 out
    all_ok = run_relocation_rounds(params, model, n_vocab, fill, "server shape 4/1/1, accepted 1", 3, { 1, 0, 2 }, { { 0, { 1, 2 }, 1 } }) && all_ok;
    all_ok = run_relocation_rounds(params, model, n_vocab, fill, "server shape 4/1/1, accepted 2", 3, { 1, 0, 2 }, { { 0, { 1, 2 }, 2 } }) && all_ok;

    // cells seq1, seq2, seq0: gathering seq0 then seq1 moves seq2 into seq0's old cell, a three-cell cycle
    all_ok = run_relocation_rounds(params, model, n_vocab, fill, "three-cell cycle", 3, { 1, 2, 0 }, { { 2, { 0, 1 }, 1 } }) && all_ok;

    // Each round: a verify ubatch (no carry), a pair ubatch that relocates the verified sequence (carry), the trim,
    // the same pair again (same sequences, nothing left to relocate: a graph reused here would still hold the carry
    // and must be rejected), then a one-token ubatch that uses the trimmed state. A graph can only be reused by a
    // consecutive ubatch of the same sequences, and after a gather those sit contiguously, so a reused graph never
    // has a different carry list; the size check guards the transitions in both directions.
    all_ok = run_relocation_rounds(params, model, n_vocab, fill, "repeated relocations", 3, { 1, 0, 2 },
            { { 0, { 2, 1 }, 1, { 2, 1 } }, { 1, { 0, 2 }, 2, { 0, 2 } }, { 2, { 1, 0 }, 1, { 1, 0 } },
              { 0, { 2, 1 }, 2, { 2, 1 } }, { 1, { 0, 2 }, 1, { 0, 2 } }, { 2, { 1, 0 }, 2, { 1, 0 } } }) && all_ok;

    // cells seq1, seq0, seq2, seq3: three members gather and the verified sequence is displaced twice, two cells along
    all_ok = run_relocation_rounds(params, model, n_vocab, fill, "three members, displaced twice", 4, { 1, 0, 2, 3 },
            { { 0, { 1, 2, 3 }, 1, { 1, 2, 0 } }, { 2, { 1, 0, 3 }, 2, { 1, 0, 2 } } }) && all_ok;

    // control: ordered cells, nothing is relocated
    all_ok = run_relocation_rounds(params, model, n_vocab, fill, "control: ordered cells", 3, { 0, 1, 2 }, { { 0, { 1, 2 }, 1 } }) && all_ok;

    return all_ok;
}

// A partial seq_rm is served by the snapshot planes only when the cell belongs to the sequence alone, nothing is
// pending, and the removal stays inside the last ubatch with the anchor kept. Everything else must be refused and
// leave the memory untouched: the whole ubatch, a removal reaching past the last ubatch, a removal right after a
// state load, one on a shared cell, one after a ubatch with repeated positions, and one after the attention was made
// non-causal. Accepted removals continue from the anchor's state; the reference never decoded the removed tokens,
// so its ubatch shapes differ and the comparison uses a tolerance far below the defect size.
static bool test_refusal(const common_params & params, llama_model * model, int n_vocab, uint8_t fill) {
    constexpr llama_pos P = 12;
    constexpr double    nmse_ok = 1e-4;
    const ctx_spec spec = { 2, 8, 64 };

    bool all_ok = true;

    const char * fn = __func__;
    const auto report_refused = [&](llama_context * ctx, llama_seq_id seq, bool accepted, llama_pos pmax_before,
                                    const std::vector<rs_row> & before, const char * name) {
        if (accepted) {
            fprintf(stderr, "%s : %s: accepted a removal the snapshots cannot serve\n", fn, name);
            return false;
        }
        const llama_pos pmax = llama_memory_seq_pos_max(llama_get_memory(ctx), seq);
        const std::string diff = state_diff(seq_state(ctx, seq), before);
        if (pmax != pmax_before || !diff.empty()) {
            fprintf(stderr, "%s : %s: refused but changed the memory (pos %d -> %d) %s\n", fn, name, pmax_before, pmax, diff.c_str());
            return false;
        }
        fprintf(stderr, "%s : %s: refused, memory unchanged\n", fn, name);
        return true;
    };

    // ubatches of `steps` tokens after the prefix; tokens beyond the kept ones are rejected drafts (salt 1)
    struct step_case {
        const char *     name;
        std::vector<int> steps;
        int              rollback;
        bool             expect_accept;
    };
    const step_case step_cases[] = {
        { "whole ubatch of 1 by 1",                     { 1 },       1, false },
        { "whole ubatch of 3 by 3",                     { 3 },       3, false },
        { "whole ubatch of 8 by 8",                     { 8 },       8, false },
        { "past the last ubatch, steps (3,2) by 4",     { 3, 2 },    4, false },
        { "past the last ubatch, steps (1,1,1) by 2",   { 1, 1, 1 }, 2, false },
        { "anchor in an earlier ubatch, steps (4,1) by 3", { 4, 1 }, 3, false },
        { "control: anchored, ubatch of 3 by 2",        { 3 },       2, true  },
        { "control: anchored, ubatch of 9 by 8",        { 9 },       8, true  },
        { "control: anchored, steps (3,2) by 1",        { 3, 2 },    1, true  },
    };
    for (const auto & c : step_cases) {
        llama_context * roll = make_ctx_multi(params, model, fill, spec);
        llama_context * ref  = make_ctx_multi(params, model, fill, spec);
        if (roll == nullptr || ref == nullptr) {
            fprintf(stderr, "%s : failed to init contexts\n", __func__);
            return false;
        }
        if (llama_n_rs_seq(roll) < 8) {
            fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
            llama_free(roll);
            llama_free(ref);
            return true;
        }
        int total = 0;
        for (const int n : c.steps) {
            total += n;
        }
        const int keep = total - c.rollback;

        bool ok = decode_specs(roll, tok_run(n_vocab, 0, 0, P)) == 0 && decode_specs(ref, tok_run(n_vocab, 0, 0, P)) == 0;
        llama_pos p = P;
        for (const int n : c.steps) {
            std::vector<tokspec> step;
            for (int i = 0; i < n; ++i, ++p) {
                step.push_back({ 0, p, tok_at(n_vocab, 0, p, p < P + keep ? 0 : 1) });
            }
            ok = ok && decode_specs(roll, step) == 0;
        }
        if (keep > 0) {
            ok = ok && decode_specs(ref, tok_run(n_vocab, 0, P, keep)) == 0;
        }
        if (!ok) {
            fprintf(stderr, "%s : %s: setup failed\n", __func__, c.name);
            llama_free(roll);
            llama_free(ref);
            return false;
        }

        const llama_pos pmax_before = llama_memory_seq_pos_max(llama_get_memory(roll), 0);
        const auto before = seq_state(roll, 0);
        const bool accepted = llama_memory_seq_rm(llama_get_memory(roll), 0, P + keep, -1);

        if (!c.expect_accept) {
            all_ok = report_refused(roll, 0, accepted, pmax_before, before, c.name) && all_ok;
        } else if (!accepted) {
            fprintf(stderr, "%s : %s: refused an anchored removal\n", __func__, c.name);
            all_ok = false;
        } else {
            const std::vector<tokspec> next = { { 0, P + keep, tok_at(n_vocab, 0, P + keep) } };
            ok = decode_specs(roll, next) == 0 && decode_specs(ref, next) == 0;
            const state_cmp cmp = ok ? compare_states(seq_state(roll, 0), seq_state(ref, 0)) : state_cmp{};
            const bool pass = ok && cmp.nmse <= nmse_ok;
            fprintf(stderr, "%s : %s: accepted, state after continuing %s the reference (nmse %g)\n", __func__, c.name,
                    pass ? "matches" : "DIFFERS from", cmp.nmse);
            all_ok = all_ok && pass;
        }
        llama_free(roll);
        llama_free(ref);
    }

    // after a state load only plane 0 is present; a later ubatch grants depth again
    {
        llama_context * src = make_ctx_multi(params, model, fill, spec);
        llama_context * dst = make_ctx_multi(params, model, fill, spec);
        if (src == nullptr || dst == nullptr) {
            fprintf(stderr, "%s : failed to init contexts\n", __func__);
            return false;
        }
        bool ok = decode_specs(src, tok_run(n_vocab, 0, 0, P)) == 0;
        std::vector<uint8_t> blob(llama_state_seq_get_size(src, 0));
        ok = ok && llama_state_seq_get_data(src, blob.data(), blob.size(), 0) == blob.size();
        ok = ok && llama_state_seq_set_data(dst, blob.data(), blob.size(), 0) == blob.size();
        if (!ok) {
            fprintf(stderr, "%s : after load: setup failed\n", __func__);
            llama_free(src);
            llama_free(dst);
            return false;
        }
        const llama_pos pmax_before = llama_memory_seq_pos_max(llama_get_memory(dst), 0);
        const auto before = seq_state(dst, 0);
        all_ok = report_refused(dst, 0, llama_memory_seq_rm(llama_get_memory(dst), 0, P - 2, -1), pmax_before, before, "right after a state load") && all_ok;

        ok = decode_specs(dst, tok_cat(tok_run(n_vocab, 0, P, 2), tok_run(n_vocab, 0, P + 2, 2, 1))) == 0;
        const bool accepted = ok && llama_memory_seq_rm(llama_get_memory(dst), 0, P + 2, -1);
        fprintf(stderr, "%s : control: anchored removal after a load and a 4-token ubatch: %s\n", __func__, accepted ? "accepted" : "REFUSED");
        all_ok = all_ok && accepted;
        llama_free(src);
        llama_free(dst);
    }

    // a cell shared after seq_cp: the removal would move both sequences
    {
        llama_context * ctx = make_ctx_multi(params, model, fill, spec);
        if (ctx == nullptr) {
            fprintf(stderr, "%s : failed to init context\n", __func__);
            return false;
        }
        auto * mem = llama_get_memory(ctx);
        bool ok = decode_specs(ctx, tok_run(n_vocab, 0, 0, P)) == 0;
        ok = ok && llama_memory_seq_rm(mem, 1, -1, -1);
        llama_memory_seq_cp(mem, 0, 1, -1, -1);
        if (!ok) {
            fprintf(stderr, "%s : shared cell: setup failed\n", __func__);
            llama_free(ctx);
            return false;
        }
        const llama_pos pmax_before = llama_memory_seq_pos_max(mem, 0);
        const auto before = seq_state(ctx, 0);
        all_ok = report_refused(ctx, 0, llama_memory_seq_rm(mem, 1, P - 2, -1), pmax_before, before, "on a shared cell") && all_ok;

        // once the copy leaves, the owner may roll back again
        ok = llama_memory_seq_rm(mem, 1, -1, -1);
        const bool accepted = ok && llama_memory_seq_rm(mem, 0, P - 2, -1);
        fprintf(stderr, "%s : control: anchored removal after the copy left the cell: %s\n", __func__, accepted ? "accepted" : "REFUSED");
        all_ok = all_ok && accepted;
        llama_free(ctx);
    }

    // repeated positions in one ubatch: a position difference no longer counts tokens
    {
        llama_context * ctx = make_ctx_multi(params, model, fill, spec);
        if (ctx == nullptr) {
            fprintf(stderr, "%s : failed to init context\n", __func__);
            return false;
        }
        bool ok = decode_specs(ctx, tok_run(n_vocab, 0, 0, P)) == 0;
        const std::vector<tokspec> repeated = { { 0, P, tok_at(n_vocab, 0, P) }, { 0, P + 1, tok_at(n_vocab, 0, P + 1, 1) }, { 0, P + 1, tok_at(n_vocab, 0, P + 1, 2) } };
        const int rc = ok ? decode_specs(ctx, repeated) : -1;
        if (!ok) {
            fprintf(stderr, "%s : repeated positions: setup failed\n", __func__);
            llama_free(ctx);
            return false;
        }
        if (rc != 0) {
            fprintf(stderr, "%s : repeated positions: skipped, the batch was rejected (rc %d)\n", __func__, rc);
        } else {
            const llama_pos pmax_before = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
            const auto before = seq_state(ctx, 0);
            all_ok = report_refused(ctx, 0, llama_memory_seq_rm(llama_get_memory(ctx), 0, P + 1, -1), pmax_before, before, "after repeated positions") && all_ok;
        }
        llama_free(ctx);
    }

    // non-causal attention: the planes written since are not rollback history, causal again grants depth again
    {
        llama_context * ctx = make_ctx_multi(params, model, fill, spec);
        if (ctx == nullptr) {
            fprintf(stderr, "%s : failed to init context\n", __func__);
            return false;
        }
        auto * mem = llama_get_memory(ctx);
        const auto verify = [&](llama_pos p) {
            return decode_specs(ctx, tok_cat(tok_run(n_vocab, 0, p, 2), tok_run(n_vocab, 0, p + 2, 2, 1))) == 0;
        };
        bool ok = decode_specs(ctx, tok_run(n_vocab, 0, 0, P)) == 0 && verify(P);
        if (!ok) {
            fprintf(stderr, "%s : non-causal: setup failed\n", __func__);
            llama_free(ctx);
            return false;
        }
        llama_set_causal_attn(ctx, false);
        const llama_pos pmax_before = llama_memory_seq_pos_max(mem, 0);
        const auto before = seq_state(ctx, 0);
        all_ok = report_refused(ctx, 0, llama_memory_seq_rm(mem, 0, P + 2, -1), pmax_before, before, "after llama_set_causal_attn(false)") && all_ok;

        llama_set_causal_attn(ctx, true);
        ok = verify(P + 4);
        const bool accepted = ok && llama_memory_seq_rm(mem, 0, P + 6, -1);
        fprintf(stderr, "%s : control: anchored removal after causal attention is back: %s\n", __func__, accepted ? "accepted" : "REFUSED");
        all_ok = all_ok && accepted;
        llama_free(ctx);
    }

    return all_ok;
}

// Under non-causal attention every token's state has seen the tokens after it, so the snapshot planes are not the
// history a rollback needs. A context that is non-causal, by the model's metadata or by attention_type, must get no
// rollback depth at all: n_rs_seq clamps to 0 and partial removals are refused.
static bool test_non_causal(const common_params & params, llama_model * model, int n_vocab, uint8_t fill) {
    constexpr llama_pos P = 12;

    char arch[64] = {0};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    const std::string key = std::string(arch) + ".attention.causal";
    char val[16] = {0};
    const bool model_causal = llama_model_meta_val_str(model, key.c_str(), val, sizeof(val)) < 0 || strcmp(val, "false") != 0;

    bool all_ok = true;
    for (const bool non_causal_type : { false, true }) {
        const ctx_spec spec = { 1, 8, 64, non_causal_type ? LLAMA_ATTENTION_TYPE_NON_CAUSAL : LLAMA_ATTENTION_TYPE_UNSPECIFIED };
        llama_context * ctx = make_ctx_multi(params, model, fill, spec);
        if (ctx == nullptr) {
            fprintf(stderr, "%s : failed to init context\n", __func__);
            return false;
        }
        const bool expect_rollback = model_causal && !non_causal_type;
        const char * what = non_causal_type ? "attention_type non-causal" : (model_causal ? "control: causal model" : "non-causal model");

        bool ok = decode_specs(ctx, tok_run(n_vocab, 0, 0, P)) == 0;
        ok = ok && decode_specs(ctx, tok_cat(tok_run(n_vocab, 0, P, 2), tok_run(n_vocab, 0, P + 2, 2, 1))) == 0;
        if (!ok) {
            fprintf(stderr, "%s : %s: setup failed\n", __func__, what);
            llama_free(ctx);
            return false;
        }
        const uint32_t n_rs_seq = llama_n_rs_seq(ctx);
        const bool accepted = llama_memory_seq_rm(llama_get_memory(ctx), 0, P + 2, -1);
        llama_free(ctx);

        const bool pass = expect_rollback ? (n_rs_seq == 8 && accepted) : (n_rs_seq == 0 && !accepted);
        fprintf(stderr, "%s : %s: n_rs_seq %u, anchored removal %s%s\n", __func__, what, n_rs_seq, accepted ? "accepted" : "refused",
                pass ? "" : (expect_rollback ? " (expected rollback)" : " (expected no rollback: snapshots written non-causally are not history)"));
        all_ok = all_ok && pass;
    }
    return all_ok;
}

static bool model_is_deepseek4(llama_model * model) {
    char arch[64] = {0};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    return strcmp(arch, "deepseek4") == 0;
}

static int test_lifecycle(const common_params & params, llama_model * model, uint8_t fill) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    if (model_is_deepseek4(model)) {
        // DeepSeek V4 keeps its rollback state in llama_kv_cache_dsv4, which does not implement these rules yet
        fprintf(stderr, "%s : skipping for DeepSeek V4\n", __func__);
        return 0;
    }
    if (!test_state_instrument(params, model, n_vocab, fill)) {
        return 1;
    }
    int ret = 0;
    if (!test_pending_index_lifecycle(params, model, n_vocab, fill)) {
        ret = 1;
    }
    if (!test_failed_ubatch(params, model, n_vocab, fill)) {
        ret = 1;
    }
    if (!test_relocation(params, model, n_vocab, fill)) {
        ret = 1;
    }
    if (!test_refusal(params, model, n_vocab, fill)) {
        ret = 1;
    }
    if (!test_non_causal(params, model, n_vocab, fill)) {
        ret = 1;
    }
    return ret;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    int ret = 0;
    for (uint8_t fill : { 0, 0x3e }) {
        fprintf(stderr, "%s : testing with cache fill 0x%02x\n", __func__, fill);
        if (test_rollback(params, model, fill) != 0) {
            ret = 1;
        }
        if (test_lifecycle(params, model, fill) != 0) {
            ret = 1;
        }
    }

    return ret;
}
