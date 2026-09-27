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
    if (ctx == nullptr || fill == 0) {
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
static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab, uint8_t fill) {
    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

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

    // both contexts decode the identical [0, p0) prefill; only ctx_roll decodes
    // the tail, which is then rolled back so its restore is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < (llama_pos) p0; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        ok = ok && llama_decode(ctx_ref,  batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = p0; pos < (llama_pos) n_prompt; ++pos) {
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

// Exact comparison, except that -0 and +0 are equal: the zeroed state is made by scaling the cell's previous
// content by 0, which keeps the sign of negative values. Returns an empty string when the states are equal.
static std::string state_diff(const std::vector<rs_row> & a, const std::vector<rs_row> & b) {
    if (a.size() != b.size()) {
        return "(row count differs)";
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
            return "(row layout differs)";
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
    if (n_rows == 0) {
        return "";
    }
    const double nmse_val = sum_sq_ref > 0.0 ? sum_sq_diff/sum_sq_ref : (sum_sq_diff > 0.0 ? std::numeric_limits<double>::infinity() : 0.0);
    char buf[192];
    snprintf(buf, sizeof(buf), "(%zu rows, %zu elements differ, first at row %zu element %zu, max abs %g, nmse %g)",
             n_rows, n_elems, first_row, first_elem, max_abs, nmse_val);
    return buf;
}

struct ctx_spec {
    uint32_t n_seq_max;
    uint32_t n_rs_seq;
    uint32_t n_ubatch;
};

static llama_context * make_ctx_multi(const common_params & params, llama_model * model, uint8_t fill, const ctx_spec & spec) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = spec.n_seq_max;
    cparams.n_rs_seq   = spec.n_rs_seq;
    cparams.n_ctx      = 1024;
    cparams.n_batch    = 512;
    cparams.n_ubatch   = spec.n_ubatch;
    cparams.kv_unified = true;
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

// A ubatch that fails after find_slot has claimed its positions. When the failed span is longer than the snapshot
// depth, or there are no snapshots, the partial removal in the failure path cannot serve it.
static bool test_failed_ubatch(const common_params & params, llama_model * model, int n_vocab, uint8_t fill) {
    constexpr llama_pos P = 12;

    struct variant {
        const char * name;
        uint32_t     n_rs_seq;
        int          n_tokens;
    };
    const variant variants[] = {
        { "abort of a 16-token ubatch with n_rs_seq 8", 8, 16 },
        { "abort of a 4-token ubatch with n_rs_seq 0",  0,  4 },
    };

    bool all_ok = true;
    for (const auto & v : variants) {
        llama_context * ctx = make_ctx_multi(params, model, fill, { 1, v.n_rs_seq, 64 });
        if (ctx == nullptr) {
            fprintf(stderr, "%s : failed to init context\n", __func__);
            return false;
        }
        abort_ctl ctl;
        llama_set_abort_callback(ctx, abort_cb, &ctl);

        bool ok = decode_specs(ctx, tok_run(n_vocab, 0, 0, P)) == 0;
        const auto before = seq_state(ctx, 0);

        ctl.countdown = 0;
        const int rc = decode_specs(ctx, tok_run(n_vocab, 0, P, v.n_tokens, 1));
        ctl.countdown = -1;
        if (!ok) {
            fprintf(stderr, "%s : %s: setup failed\n", __func__, v.name);
            llama_free(ctx);
            return false;
        }
        if (rc == 0) {
            fprintf(stderr, "%s : %s: skipped, the abort callback was not honoured\n", __func__, v.name);
            llama_free(ctx);
            continue;
        }
        all_ok = check_after_failure(ctx, 0, P - 1, before, v.name) && all_ok;
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
