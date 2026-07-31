#pragma once

#include "qwenvl_paged/CacheLayout.h"
#include "qwenvl_paged/KVCacheManager.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

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

} // namespace qwenvl_paged
