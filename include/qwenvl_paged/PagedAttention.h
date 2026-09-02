#pragma once

#include "qwenvl_paged/CacheLayout.h"
#include "qwenvl_paged/KVCacheManager.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace qwenvl_paged {

/**
 * @brief Per-call parameters for the reference attention path.
 *
 * `context_len` is supplied by the caller rather than read from the cache
 * manager: the cache manager owns reserved *capacity*, while the engine owns
 * how many of those slots hold committed tokens. This mirrors how a batched GPU
 * kernel receives a per-sequence length array.
 */
struct PagedAttentionParams {
    std::uint32_t layer{0};
    std::uint32_t num_query_heads{0};
    std::uint32_t context_len{0};
    float scale{1.0F};
};

/**
 * @brief Correctness-only CPU PagedAttention for a single query position.
 *
 * Computes softmax(scale * Q Kᵀ) V for one query token against `context_len`
 * cached tokens, walking the sequence block table entry by entry so the cache
 * may be physically scattered across the block pool. There is no blocking, no
 * vectorization, and no online softmax; this exists to pin down the expected
 * result that a CUDA or Triton kernel must reproduce.
 *
 * Grouped-query attention is supported: `num_query_heads` must be a positive
 * multiple of `BlockShape::num_kv_heads`, and query head `h` reads kv head
 * `h / (num_query_heads / num_kv_heads)`.
 *
 * Causal prefill is this function applied per prompt position `p` with
 * `context_len = p + 1`; no separate prefill entry point is needed.
 *
 * @param view   Read-only cache view for the sequence.
 * @param query  `num_query_heads * head_dim` elements, head-major.
 * @param params Layer, head count, context length, and softmax scale.
 * @param out    `num_query_heads * head_dim` elements, head-major.
 * @return False without touching `out` when the arguments are inconsistent with
 *         the cache shape or when any required block is unmapped or swapped out.
 */
template <typename T>
[[nodiscard]] bool paged_attention_decode(
    const CacheView& view,
    const T* query,
    const PagedAttentionParams& params,
    T* out) {
    if (!view.valid() || query == nullptr || out == nullptr) {
        return false;
    }

    const BlockShape& shape = view.layout.shape;
    if (sizeof(T) != shape.bytes_per_element || params.context_len == 0 ||
        params.layer >= shape.num_layers || params.num_query_heads == 0 ||
        params.num_query_heads % shape.num_kv_heads != 0) {
        return false;
    }

    const std::uint32_t head_dim = shape.head_dim;
    const std::uint32_t group_size = params.num_query_heads / shape.num_kv_heads;

    // Walk the page table for the whole context before computing anything.
    // Writing part of the output and only then discovering a swapped-out block
    // would leave the caller unable to tell a finished result from a truncated
    // one, so resolution failures must be detected while `out` is still clean.
    const std::size_t resolved = static_cast<std::size_t>(shape.num_kv_heads) * params.context_len;
    std::vector<const T*> keys(resolved, nullptr);
    std::vector<const T*> values(resolved, nullptr);
    for (std::uint32_t kv_head = 0; kv_head < shape.num_kv_heads; ++kv_head) {
        for (std::uint32_t token = 0; token < params.context_len; ++token) {
            const std::size_t index = static_cast<std::size_t>(kv_head) * params.context_len + token;
            keys[index] = view.slot<T>(KVStream::Key, params.layer, token, kv_head);
            values[index] = view.slot<T>(KVStream::Value, params.layer, token, kv_head);
            if (keys[index] == nullptr || values[index] == nullptr) {
                return false;
            }
        }
    }

    std::vector<float> scores(params.context_len, 0.0F);
    std::vector<float> accumulator(head_dim, 0.0F);

    for (std::uint32_t q_head = 0; q_head < params.num_query_heads; ++q_head) {
        const std::size_t base = static_cast<std::size_t>(q_head / group_size) * params.context_len;
        const T* q = query + static_cast<std::size_t>(q_head) * head_dim;

        float max_score = -std::numeric_limits<float>::infinity();
        for (std::uint32_t token = 0; token < params.context_len; ++token) {
            const T* key = keys[base + token];
            float dot = 0.0F;
            for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
                dot += static_cast<float>(q[dim]) * static_cast<float>(key[dim]);
            }
            scores[token] = dot * params.scale;
            max_score = std::max(max_score, scores[token]);
        }

        std::fill(accumulator.begin(), accumulator.end(), 0.0F);
        float denominator = 0.0F;
        for (std::uint32_t token = 0; token < params.context_len; ++token) {
            const T* value = values[base + token];
            const float weight = std::exp(scores[token] - max_score);
            denominator += weight;
            for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
                accumulator[dim] += weight * static_cast<float>(value[dim]);
            }
        }

        T* destination = out + static_cast<std::size_t>(q_head) * head_dim;
        for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
            destination[dim] = static_cast<T>(accumulator[dim] / denominator);
        }
    }

    return true;
}

/**
 * @brief Same contract as `paged_attention_decode`, one physical block per
 *        inner iteration.
 *
 * The page-table walk is hoisted to the logical-block loop. The two
 * context-length pointer vectors go away. Math is still scalar and still
 * two-pass; those are separate week-11 levers.
 */
template <typename T>
[[nodiscard]] bool paged_attention_decode_blocked(
    const CacheView& view,
    const T* query,
    const PagedAttentionParams& params,
    T* out) {
    if (!view.valid() || query == nullptr || out == nullptr) {
        return false;
    }

    const BlockShape& shape = view.layout.shape;
    if (sizeof(T) != shape.bytes_per_element || params.context_len == 0 ||
        params.layer >= shape.num_layers || params.num_query_heads == 0 ||
        params.num_query_heads % shape.num_kv_heads != 0) {
        return false;
    }

    const std::uint32_t head_dim = shape.head_dim;
    const std::uint32_t group_size = params.num_query_heads / shape.num_kv_heads;
    const std::uint32_t tokens_per_block = shape.tokens_per_block;
    const std::uint32_t logical_blocks =
        (params.context_len + tokens_per_block - 1) / tokens_per_block;

    const bool per_layer = shape.per_layer_frames();
    const std::uint32_t offset_layer = per_layer ? 0 : params.layer;
    std::vector<const std::byte*> frames(logical_blocks, nullptr);
    for (std::uint32_t logical = 0; logical < logical_blocks; ++logical) {
        frames[logical] =
            per_layer ? view.block_bytes(logical, params.layer) : view.block_bytes(logical);
        if (frames[logical] == nullptr) {
            return false;
        }
    }

    std::vector<float> scores(params.context_len, 0.0F);
    std::vector<float> accumulator(head_dim, 0.0F);

    for (std::uint32_t q_head = 0; q_head < params.num_query_heads; ++q_head) {
        const std::uint32_t kv_head = q_head / group_size;
        const T* q = query + static_cast<std::size_t>(q_head) * head_dim;

        float max_score = -std::numeric_limits<float>::infinity();
        for (std::uint32_t logical = 0; logical < logical_blocks; ++logical) {
            const T* base = reinterpret_cast<const T*>(frames[logical]);
            const std::uint32_t begin = logical * tokens_per_block;
            const std::uint32_t end = std::min(begin + tokens_per_block, params.context_len);
            for (std::uint32_t token = begin; token < end; ++token) {
                const std::optional<std::size_t> offset = view.layout.element_offset(
                    offset_layer, KVStream::Key, token - begin, kv_head);
                if (!offset.has_value()) {
                    return false;
                }
                const T* key = base + *offset;
                float dot = 0.0F;
                for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
                    dot += static_cast<float>(q[dim]) * static_cast<float>(key[dim]);
                }
                scores[token] = dot * params.scale;
                max_score = std::max(max_score, scores[token]);
            }
        }

        std::fill(accumulator.begin(), accumulator.end(), 0.0F);
        float denominator = 0.0F;
        for (std::uint32_t logical = 0; logical < logical_blocks; ++logical) {
            const T* base = reinterpret_cast<const T*>(frames[logical]);
            const std::uint32_t begin = logical * tokens_per_block;
            const std::uint32_t end = std::min(begin + tokens_per_block, params.context_len);
            for (std::uint32_t token = begin; token < end; ++token) {
                const std::optional<std::size_t> offset = view.layout.element_offset(
                    offset_layer, KVStream::Value, token - begin, kv_head);
                if (!offset.has_value()) {
                    return false;
                }
                const T* value = base + *offset;
                const float weight = std::exp(scores[token] - max_score);
                denominator += weight;
                for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
                    accumulator[dim] += weight * static_cast<float>(value[dim]);
                }
            }
        }

        T* destination = out + static_cast<std::size_t>(q_head) * head_dim;
        for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
            destination[dim] = static_cast<T>(accumulator[dim] / denominator);
        }
    }

    return true;
}

namespace detail {

template <typename T>
float load_as_float(const T* ptr) {
    return static_cast<float>(*ptr);
}

template <typename T, bool kAvx>
float dot_head(const T* query, const T* key, std::uint32_t head_dim) {
#if defined(__AVX512F__)
    if constexpr (kAvx) {
        if (head_dim % 16 == 0) {
            __m512 acc = _mm512_setzero_ps();
            for (std::uint32_t dim = 0; dim < head_dim; dim += 16) {
                __m512 qv;
                __m512 kv;
                if constexpr (sizeof(T) == 4) {
                    qv = _mm512_loadu_ps(reinterpret_cast<const float*>(query + dim));
                    kv = _mm512_loadu_ps(reinterpret_cast<const float*>(key + dim));
                } else {
                    const __m256i q16 =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query + dim));
                    const __m256i k16 =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(key + dim));
                    qv = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(q16));
                    kv = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(k16));
                }
                acc = _mm512_fmadd_ps(qv, kv, acc);
            }
            return _mm512_reduce_add_ps(acc);
        }
    }
#endif
    float dot = 0.0F;
    for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
        dot += load_as_float(query + dim) * load_as_float(key + dim);
    }
    return dot;
}

template <typename T, bool kAvx>
void axpy_head(float* accumulator, const T* value, float weight, std::uint32_t head_dim) {
#if defined(__AVX512F__)
    if constexpr (kAvx) {
        if (head_dim % 16 == 0) {
            const __m512 scale = _mm512_set1_ps(weight);
            for (std::uint32_t dim = 0; dim < head_dim; dim += 16) {
                __m512 vv;
                if constexpr (sizeof(T) == 4) {
                    vv = _mm512_loadu_ps(reinterpret_cast<const float*>(value + dim));
                } else {
                    const __m256i v16 =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(value + dim));
                    vv = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(v16));
                }
                const __m512 acc = _mm512_loadu_ps(accumulator + dim);
                _mm512_storeu_ps(accumulator + dim, _mm512_fmadd_ps(scale, vv, acc));
            }
            return;
        }
    }
#endif
    for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
        accumulator[dim] += weight * load_as_float(value + dim);
    }
}

template <bool kAvx>
void scale_head(float* accumulator, float scale, std::uint32_t head_dim) {
#if defined(__AVX512F__)
    if constexpr (kAvx) {
        if (head_dim % 16 == 0) {
            const __m512 factor = _mm512_set1_ps(scale);
            for (std::uint32_t dim = 0; dim < head_dim; dim += 16) {
                _mm512_storeu_ps(
                    accumulator + dim, _mm512_mul_ps(_mm512_loadu_ps(accumulator + dim), factor));
            }
            return;
        }
    }
#endif
    for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
        accumulator[dim] *= scale;
    }
}

} // namespace detail

/**
 * @brief Week-11 optimized decode: optional GQA fusion, online softmax, AVX-512.
 *
 * Same contract and numeric target as `paged_attention_decode`. The scalar
 * reference stays the oracle; this is selected by the caller.
 */
template <typename T, bool kFuseGqa, bool kOnline, bool kAvx>
[[nodiscard]] bool paged_attention_decode_opt(
    const CacheView& view,
    const T* query,
    const PagedAttentionParams& params,
    T* out) {
    if (!view.valid() || query == nullptr || out == nullptr) {
        return false;
    }

    const BlockShape& shape = view.layout.shape;
    if (sizeof(T) != shape.bytes_per_element || params.context_len == 0 ||
        params.layer >= shape.num_layers || params.num_query_heads == 0 ||
        params.num_query_heads % shape.num_kv_heads != 0) {
        return false;
    }

    const std::uint32_t head_dim = shape.head_dim;
    const std::uint32_t group_size = params.num_query_heads / shape.num_kv_heads;
    const std::uint32_t tokens_per_block = shape.tokens_per_block;
    const std::uint32_t logical_blocks =
        (params.context_len + tokens_per_block - 1) / tokens_per_block;

    const bool per_layer = shape.per_layer_frames();
    const std::uint32_t offset_layer = per_layer ? 0 : params.layer;
    std::vector<const std::byte*> frames(logical_blocks, nullptr);
    for (std::uint32_t logical = 0; logical < logical_blocks; ++logical) {
        frames[logical] =
            per_layer ? view.block_bytes(logical, params.layer) : view.block_bytes(logical);
        if (frames[logical] == nullptr) {
            return false;
        }
    }

    auto key_ptr = [&](const T* base, std::uint32_t token_in_block, std::uint32_t kv_head) -> const T* {
        const std::optional<std::size_t> offset =
            view.layout.element_offset(offset_layer, KVStream::Key, token_in_block, kv_head);
        return offset.has_value() ? base + *offset : nullptr;
    };
    auto value_ptr = [&](const T* base, std::uint32_t token_in_block, std::uint32_t kv_head) -> const T* {
        const std::optional<std::size_t> offset =
            view.layout.element_offset(offset_layer, KVStream::Value, token_in_block, kv_head);
        return offset.has_value() ? base + *offset : nullptr;
    };

    if constexpr (!kFuseGqa) {
        return paged_attention_decode_blocked<T>(view, query, params, out);
    }

    const std::uint32_t kv_heads = shape.num_kv_heads;
    std::vector<float> scores;
    if constexpr (!kOnline) {
        scores.resize(static_cast<std::size_t>(group_size) * params.context_len);
    }
    std::vector<float> max_score(group_size, -std::numeric_limits<float>::infinity());
    std::vector<float> denom(group_size, 0.0F);
    std::vector<float> accumulator(static_cast<std::size_t>(group_size) * head_dim, 0.0F);

    for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
        std::fill(max_score.begin(), max_score.end(), -std::numeric_limits<float>::infinity());
        std::fill(denom.begin(), denom.end(), 0.0F);
        std::fill(accumulator.begin(), accumulator.end(), 0.0F);

        auto q_of = [&](std::uint32_t g) {
            return query + static_cast<std::size_t>(kv_head * group_size + g) * head_dim;
        };
        auto acc_of = [&](std::uint32_t g) { return accumulator.data() + static_cast<std::size_t>(g) * head_dim; };

        if constexpr (kOnline) {
            for (std::uint32_t logical = 0; logical < logical_blocks; ++logical) {
                const T* base = reinterpret_cast<const T*>(frames[logical]);
                const std::uint32_t begin = logical * tokens_per_block;
                const std::uint32_t end = std::min(begin + tokens_per_block, params.context_len);
                for (std::uint32_t token = begin; token < end; ++token) {
                    const T* key = key_ptr(base, token - begin, kv_head);
                    const T* value = value_ptr(base, token - begin, kv_head);
                    if (key == nullptr || value == nullptr) {
                        return false;
                    }
                    for (std::uint32_t g = 0; g < group_size; ++g) {
                        const float score = params.scale * detail::dot_head<T, kAvx>(q_of(g), key, head_dim);
                        const float new_max = std::max(max_score[g], score);
                        const float rescale = std::exp(max_score[g] - new_max);
                        const float weight = std::exp(score - new_max);
                        if (max_score[g] != new_max && denom[g] != 0.0F) {
                            detail::scale_head<kAvx>(acc_of(g), rescale, head_dim);
                        }
                        denom[g] = denom[g] * rescale + weight;
                        max_score[g] = new_max;
                        detail::axpy_head<T, kAvx>(acc_of(g), value, weight, head_dim);
                    }
                }
            }
        } else {
            for (std::uint32_t logical = 0; logical < logical_blocks; ++logical) {
                const T* base = reinterpret_cast<const T*>(frames[logical]);
                const std::uint32_t begin = logical * tokens_per_block;
                const std::uint32_t end = std::min(begin + tokens_per_block, params.context_len);
                for (std::uint32_t token = begin; token < end; ++token) {
                    const T* key = key_ptr(base, token - begin, kv_head);
                    if (key == nullptr) {
                        return false;
                    }
                    for (std::uint32_t g = 0; g < group_size; ++g) {
                        const float score = params.scale * detail::dot_head<T, kAvx>(q_of(g), key, head_dim);
                        scores[static_cast<std::size_t>(g) * params.context_len + token] = score;
                        max_score[g] = std::max(max_score[g], score);
                    }
                }
            }
            for (std::uint32_t logical = 0; logical < logical_blocks; ++logical) {
                const T* base = reinterpret_cast<const T*>(frames[logical]);
                const std::uint32_t begin = logical * tokens_per_block;
                const std::uint32_t end = std::min(begin + tokens_per_block, params.context_len);
                for (std::uint32_t token = begin; token < end; ++token) {
                    const T* value = value_ptr(base, token - begin, kv_head);
                    if (value == nullptr) {
                        return false;
                    }
                    for (std::uint32_t g = 0; g < group_size; ++g) {
                        const float weight = std::exp(
                            scores[static_cast<std::size_t>(g) * params.context_len + token] - max_score[g]);
                        denom[g] += weight;
                        detail::axpy_head<T, kAvx>(acc_of(g), value, weight, head_dim);
                    }
                }
            }
        }

        for (std::uint32_t g = 0; g < group_size; ++g) {
            T* destination =
                out + static_cast<std::size_t>(kv_head * group_size + g) * head_dim;
            const float* acc = acc_of(g);
            const float inv = 1.0F / denom[g];
            for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
                destination[dim] = static_cast<T>(acc[dim] * inv);
            }
        }
    }

    return true;
}

template <typename T>
[[nodiscard]] bool paged_attention_decode_fused(
    const CacheView& view, const T* query, const PagedAttentionParams& params, T* out) {
    return paged_attention_decode_opt<T, true, false, false>(view, query, params, out);
}

template <typename T>
[[nodiscard]] bool paged_attention_decode_fast(
    const CacheView& view, const T* query, const PagedAttentionParams& params, T* out) {
    return paged_attention_decode_opt<T, true, true, true>(view, query, params, out);
}

} // namespace qwenvl_paged
