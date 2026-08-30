/**
 * @brief Exports golden PagedAttention fixtures for cross-backend validation.
 *
 * Roadmap weeks 15-16 require one fixture set that validates both the CPU
 * reference kernel and a future Triton kernel, dumped before any GPU time is
 * rented so paid hours go to the kernel rather than to plumbing. Each scenario
 * becomes a directory of .npy arrays holding the whole block pool, the
 * flattened block table, the queries, and the reference kernel's output;
 * tools/golden_export/pack_npz.py folds each directory into a single .npz.
 *
 * The fixtures carry KVBlockLayout's four strides explicitly instead of a
 * pre-unpacked tensor. The architecture doc's "Backend Integration Points"
 * claims those strides plus the flattened block table are all a GPU kernel
 * needs; a consumer that has to be handed anything else falsifies the claim,
 * which is the point of exporting them in this shape.
 *
 * What these fixtures cannot catch: decode attention sums over context tokens,
 * so it is permutation-invariant as long as each K stays paired with its V. A
 * backend that visited entirely full blocks in the wrong order still produces
 * the golden output, and no correct kernel is obliged to distinguish that case.
 * Reordering that moves the partially filled tail block *is* caught, because
 * the misordered read then reaches slots the fixture never wrote, so every
 * scenario but the minimal smoke test leaves a partial tail. Detecting a
 * misordering confined to full blocks would need positional information the
 * cache does not carry, since RoPE is applied before K is cached; that belongs
 * to a prefill fixture with a causal mask, not here.
 */

#include "qwenvl_paged/CacheLayout.h"
#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/PagedAttention.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwenvl_paged {
namespace {

namespace fs = std::filesystem;

constexpr float kCrossCheckTolerance = 1e-4F;

void require(bool condition, const std::string& what) {
    if (!condition) {
        throw std::runtime_error(what);
    }
}

// --- .npy output ---------------------------------------------------------

void write_npy(
    const fs::path& path,
    const std::string& descr,
    const std::vector<std::size_t>& shape,
    const void* data,
    std::size_t element_bytes) {
    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (";
    std::size_t count = 1;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        header += std::to_string(shape[i]);
        if (i + 1 < shape.size()) {
            header += ", ";
        } else if (shape.size() == 1) {
            header += ",";
        }
        count *= shape[i];
    }
    header += "), }";

    // numpy requires the 10-byte prefix plus the newline-terminated header to
    // land on a 64-byte boundary.
    constexpr std::size_t kPrefixBytes = 10;
    const std::size_t unpadded = kPrefixBytes + header.size() + 1;
    const std::size_t padded = ((unpadded + 63) / 64) * 64;
    header.append(padded - unpadded, ' ');
    header += '\n';

    std::ofstream out(path, std::ios::binary);
    require(static_cast<bool>(out), "cannot open " + path.string());

    const char magic[8] = {'\x93', 'N', 'U', 'M', 'P', 'Y', '\x01', '\x00'};
    const std::uint16_t header_len = static_cast<std::uint16_t>(header.size());
    const char header_len_bytes[2] = {
        static_cast<char>(header_len & 0xFFU), static_cast<char>((header_len >> 8) & 0xFFU)};

    out.write(magic, sizeof(magic));
    out.write(header_len_bytes, sizeof(header_len_bytes));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(count * element_bytes));
    require(static_cast<bool>(out), "write failed for " + path.string());
}

void write_f32(
    const fs::path& dir,
    const std::string& name,
    const std::vector<std::size_t>& shape,
    const std::vector<float>& data) {
    write_npy(dir / (name + ".npy"), "<f4", shape, data.data(), sizeof(float));
}

void write_i32(
    const fs::path& dir,
    const std::string& name,
    const std::vector<std::size_t>& shape,
    const std::vector<std::int32_t>& data) {
    write_npy(dir / (name + ".npy"), "<i4", shape, data.data(), sizeof(std::int32_t));
}

/**
 * @brief Writes a 0-dimensional array, so consumers can use plain int(array).
 */
void write_scalar_i32(const fs::path& dir, const std::string& name, std::int64_t value) {
    require(
        value >= std::numeric_limits<std::int32_t>::min() &&
            value <= std::numeric_limits<std::int32_t>::max(),
        "scalar " + name + " does not fit in int32");
    const std::vector<std::int32_t> data{static_cast<std::int32_t>(value)};
    write_i32(dir, name, {}, data);
}

void write_scalar_f32(const fs::path& dir, const std::string& name, float value) {
    write_f32(dir, name, {}, std::vector<float>{value});
}

// --- Deterministic fixture contents --------------------------------------

/**
 * @brief Deterministic pseudo-random value in [-1, 1) from a coordinate tuple.
 */
float mix(std::initializer_list<std::uint32_t> parts) {
    std::uint32_t hashed = 2166136261U;
    for (const std::uint32_t part : parts) {
        hashed = (hashed ^ part) * 16777619U;
    }
    return static_cast<float>(hashed % 2000U) / 1000.0F - 1.0F;
}

/**
 * @brief Cache contents, salted by sequence so batched backends cannot cheat.
 *
 * A kernel that resolved every batch entry through sequence 0's block table
 * would still pass if all sequences held identical bytes, so the sequence index
 * has to participate in the pattern.
 */
float cache_value(
    std::uint32_t sequence,
    std::uint32_t layer,
    KVStream stream,
    std::uint32_t token,
    std::uint32_t head,
    std::uint32_t dim) {
    const std::uint32_t stream_id = (stream == KVStream::Key) ? 0U : 1U;
    return mix({0U, sequence, layer, stream_id, token, head, dim});
}

float query_value(std::uint32_t sequence, std::uint32_t query_head, std::uint32_t dim) {
    return mix({1U, sequence, query_head, dim});
}

// --- Scenarios -----------------------------------------------------------

struct Scenario {
    std::string name;
    std::string purpose;
    std::uint32_t tokens_per_block;
    std::uint32_t num_layers;
    std::uint32_t num_kv_heads;
    std::uint32_t head_dim;
    std::uint32_t num_query_heads;
    std::uint32_t layer;
    float scale;
    std::vector<std::uint32_t> context_lens;
    bool scatter;
};

/**
 * @brief Textbook attention recomputed straight from the fixture pattern.
 *
 * This guards the exporter, not the kernel. Writing K or V into the wrong slot
 * would otherwise produce a self-consistent fixture that pins down the wrong
 * answer for every backend that later validates against it.
 */
void cross_check(
    const Scenario& scenario,
    const BlockShape& shape,
    std::uint32_t sequence,
    const float* query,
    const float* actual) {
    const std::uint32_t head_dim = shape.head_dim;
    const std::uint32_t group_size = scenario.num_query_heads / shape.num_kv_heads;
    const std::uint32_t context_len = scenario.context_lens[sequence];

    for (std::uint32_t query_head = 0; query_head < scenario.num_query_heads; ++query_head) {
        const std::uint32_t kv_head = query_head / group_size;
        const float* q = query + static_cast<std::size_t>(query_head) * head_dim;

        std::vector<float> scores(context_len, 0.0F);
        float max_score = -std::numeric_limits<float>::infinity();
        for (std::uint32_t token = 0; token < context_len; ++token) {
            float dot = 0.0F;
            for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
                dot += q[dim] *
                       cache_value(sequence, scenario.layer, KVStream::Key, token, kv_head, dim);
            }
            scores[token] = dot * scenario.scale;
            max_score = std::max(max_score, scores[token]);
        }

        std::vector<float> accumulator(head_dim, 0.0F);
        float denominator = 0.0F;
        for (std::uint32_t token = 0; token < context_len; ++token) {
            const float weight = std::exp(scores[token] - max_score);
            denominator += weight;
            for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
                accumulator[dim] +=
                    weight *
                    cache_value(sequence, scenario.layer, KVStream::Value, token, kv_head, dim);
            }
        }

        for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
            const float expected = accumulator[dim] / denominator;
            const float measured = actual[static_cast<std::size_t>(query_head) * head_dim + dim];
            require(
                std::fabs(expected - measured) <= kCrossCheckTolerance,
                scenario.name + ": sequence " + std::to_string(sequence) + " query head " +
                    std::to_string(query_head) + " dim " + std::to_string(dim) +
                    " disagrees with the contiguous reference");
        }
    }
}

/**
 * @brief Returns true when a block table's frames do not form one contiguous run.
 */
bool is_scattered(const BlockTable& table) {
    const std::vector<BlockTableEntry>& entries = table.entries();
    if (entries.size() < 2) {
        return false;
    }
    PhysicalBlockId lowest = entries.front().physical_id;
    PhysicalBlockId highest = entries.front().physical_id;
    for (const BlockTableEntry& entry : entries) {
        lowest = std::min(lowest, entry.physical_id);
        highest = std::max(highest, entry.physical_id);
    }
    return highest - lowest + 1 != entries.size();
}

/**
 * @brief Runs the reference kernel over a populated cache and writes the .npy set.
 *
 * Everything from "the cache holds its contents" onward is independent of how it
 * got that way, so both exporters share it. `sequences` gives the fixture's
 * sequence ids in the order they should appear in the batch; their salts are
 * their positions in that vector, which is what cross_check recomputes from.
 */
void write_fixture(
    const Scenario& scenario,
    const BlockShape& shape,
    const KVBlockLayout& layout,
    MemoryAllocator& allocator,
    KVCacheManager& cache,
    const std::vector<SequenceId>& sequences,
    std::uint32_t pool_blocks,
    std::uint32_t max_blocks_per_seq,
    const fs::path& root) {
    const std::size_t num_seqs = sequences.size();
    const std::size_t query_elements =
        static_cast<std::size_t>(scenario.num_query_heads) * shape.head_dim;
    std::vector<float> queries(num_seqs * query_elements, 0.0F);
    std::vector<float> outputs(num_seqs * query_elements, 0.0F);
    std::vector<std::int32_t> block_table(num_seqs * max_blocks_per_seq, -1);
    std::vector<std::int32_t> context_lens(num_seqs, 0);

    for (std::size_t i = 0; i < num_seqs; ++i) {
        const std::uint32_t salt = static_cast<std::uint32_t>(i);
        float* query = queries.data() + i * query_elements;
        for (std::uint32_t query_head = 0; query_head < scenario.num_query_heads; ++query_head) {
            for (std::uint32_t dim = 0; dim < shape.head_dim; ++dim) {
                query[static_cast<std::size_t>(query_head) * shape.head_dim + dim] =
                    query_value(salt, query_head, dim);
            }
        }

        const std::optional<CacheView> view = cache.cache_view(sequences[i]);
        require(view.has_value(), "cache_view failed");
        if (scenario.scatter) {
            require(
                is_scattered(*view->block_table),
                scenario.name + ": sequence " + std::to_string(i) + " did not scatter");
        }

        PagedAttentionParams params;
        params.layer = scenario.layer;
        params.num_query_heads = scenario.num_query_heads;
        params.context_len = scenario.context_lens[i];
        params.scale = scenario.scale;

        float* out = outputs.data() + i * query_elements;
        require(
            paged_attention_decode<float>(*view, query, params, out),
            "paged_attention_decode rejected the fixture");
        cross_check(scenario, shape, salt, query, out);

        const std::vector<BlockTableEntry>& entries = view->block_table->entries();
        for (std::size_t j = 0; j < entries.size(); ++j) {
            block_table[i * max_blocks_per_seq + j] =
                static_cast<std::int32_t>(entries[j].physical_id);
        }
        context_lens[i] = static_cast<std::int32_t>(scenario.context_lens[i]);
    }

    const std::size_t elements_per_block = layout.element_count();
    std::vector<float> pool(pool_blocks * elements_per_block, 0.0F);
    for (PhysicalBlockId id = 0; id < pool_blocks; ++id) {
        const PhysicalBlock* block = allocator.block(id);
        std::memcpy(
            pool.data() + static_cast<std::size_t>(id) * elements_per_block,
            block->data(),
            elements_per_block * sizeof(float));
    }

    const fs::path dir = root / scenario.name;
    fs::create_directories(dir);

    write_f32(dir, "kv_pool", {pool_blocks, elements_per_block}, pool);
    write_f32(dir, "query", {num_seqs, scenario.num_query_heads, shape.head_dim}, queries);
    write_f32(dir, "output", {num_seqs, scenario.num_query_heads, shape.head_dim}, outputs);
    write_i32(dir, "block_table", {num_seqs, max_blocks_per_seq}, block_table);
    write_i32(dir, "context_lens", {num_seqs}, context_lens);

    write_scalar_i32(dir, "tokens_per_block", shape.tokens_per_block);
    write_scalar_i32(dir, "num_layers", shape.num_layers);
    write_scalar_i32(dir, "num_kv_heads", shape.num_kv_heads);
    write_scalar_i32(dir, "head_dim", shape.head_dim);
    write_scalar_i32(dir, "num_query_heads", scenario.num_query_heads);
    write_scalar_i32(dir, "layer", scenario.layer);
    write_scalar_i32(dir, "head_stride", static_cast<std::int64_t>(layout.head_stride()));
    write_scalar_i32(dir, "token_stride", static_cast<std::int64_t>(layout.token_stride()));
    write_scalar_i32(dir, "stream_stride", static_cast<std::int64_t>(layout.stream_stride()));
    write_scalar_i32(dir, "layer_stride", static_cast<std::int64_t>(layout.layer_stride()));
    write_scalar_f32(dir, "scale", scenario.scale);

    std::cout << scenario.name << ": " << num_seqs << " seq, pool " << pool_blocks << " blocks, "
              << (pool.size() * sizeof(float)) / 1024 << " KiB  -- " << scenario.purpose << "\n";
}

/**
 * @brief Writes one sequence's whole context into the cache, honoring copy-on-write.
 */
void fill_sequence(
    const BlockShape& shape,
    const KVBlockLayout& layout,
    MemoryAllocator& allocator,
    KVCacheManager& cache,
    SequenceId sequence,
    std::uint32_t salt,
    std::uint32_t context_len) {
    for (std::uint32_t layer = 0; layer < shape.num_layers; ++layer) {
        for (const KVStream stream : {KVStream::Key, KVStream::Value}) {
            for (std::uint32_t token = 0; token < context_len; ++token) {
                const std::optional<PhysicalBlockId> physical =
                    cache.ensure_token_writable(sequence, token);
                require(physical.has_value(), "ensure_token_writable failed");
                PhysicalBlock* block = allocator.block(*physical);
                require(block != nullptr, "writable frame missing");
                float* base = reinterpret_cast<float*>(block->data());

                for (std::uint32_t head = 0; head < shape.num_kv_heads; ++head) {
                    const std::optional<std::size_t> offset = layout.element_offset(
                        layer, stream, token % shape.tokens_per_block, head);
                    require(offset.has_value(), "element_offset out of range");
                    for (std::uint32_t dim = 0; dim < shape.head_dim; ++dim) {
                        base[*offset + dim] = cache_value(salt, layer, stream, token, head, dim);
                    }
                }
            }
        }
    }
}

void export_scenario(const Scenario& scenario, const fs::path& root) {
    BlockShape shape;
    shape.tokens_per_block = scenario.tokens_per_block;
    shape.num_layers = scenario.num_layers;
    shape.num_kv_heads = scenario.num_kv_heads;
    shape.head_dim = scenario.head_dim;
    shape.bytes_per_element = static_cast<std::uint32_t>(sizeof(float));

    KVBlockLayout layout;
    layout.shape = shape;

    const std::size_t num_seqs = scenario.context_lens.size();
    std::vector<std::uint32_t> blocks_per_seq(num_seqs, 0);
    std::uint32_t mapped_blocks = 0;
    for (std::size_t i = 0; i < num_seqs; ++i) {
        blocks_per_seq[i] =
            (scenario.context_lens[i] + scenario.tokens_per_block - 1) / scenario.tokens_per_block;
        mapped_blocks += blocks_per_seq[i];
    }
    const std::uint32_t max_blocks_per_seq =
        *std::max_element(blocks_per_seq.begin(), blocks_per_seq.end());

    // A spacer sequence claims one frame for every frame the fixture sequences
    // claim, so the pool needs double capacity when scattering.
    const std::uint32_t pool_blocks = scenario.scatter ? mapped_blocks * 2 : mapped_blocks;

    AllocatorConfig config;
    config.block_shape = shape;
    config.max_blocks = pool_blocks;
    MemoryAllocator allocator(config);
    KVCacheManager cache(allocator);

    // Frames come from aligned_alloc, so frames the fixture never writes hold
    // indeterminate bytes. Zeroing first keeps repeated exports byte-identical.
    for (PhysicalBlockId id = 0; id < pool_blocks; ++id) {
        PhysicalBlock* block = allocator.block(id);
        require(block != nullptr, "pool frame missing");
        std::memset(block->data(), 0, block->size_bytes());
    }

    constexpr SequenceId kSpacerSequence = 9999;
    const auto sequence_id = [](std::size_t index) -> SequenceId {
        return static_cast<SequenceId>(index) + 1;
    };

    for (std::size_t i = 0; i < num_seqs; ++i) {
        require(
            cache.create_sequence(SequenceMetadata{sequence_id(i), sequence_id(i), {}, {}}),
            "create_sequence failed");
    }
    if (scenario.scatter) {
        require(
            cache.create_sequence(SequenceMetadata{kSpacerSequence, kSpacerSequence, {}, {}}),
            "create spacer sequence failed");
    }

    // Reserve one block per sequence per round, taking a spacer frame in
    // between, so every sequence ends up physically scattered and interleaved
    // with its neighbours rather than sitting in one clean run.
    for (std::uint32_t round = 0; round < max_blocks_per_seq; ++round) {
        for (std::size_t i = 0; i < num_seqs; ++i) {
            if (round >= blocks_per_seq[i]) {
                continue;
            }
            require(
                cache.reserve_tokens(sequence_id(i), scenario.tokens_per_block),
                "reserve_tokens failed");
            if (scenario.scatter) {
                require(
                    cache.reserve_tokens(kSpacerSequence, scenario.tokens_per_block),
                    "reserve spacer tokens failed");
            }
        }
    }
    if (scenario.scatter) {
        cache.release_sequence(kSpacerSequence);
    }

    std::vector<SequenceId> sequences(num_seqs);
    for (std::size_t i = 0; i < num_seqs; ++i) {
        sequences[i] = sequence_id(i);
        fill_sequence(
            shape, layout, allocator, cache, sequences[i], static_cast<std::uint32_t>(i),
            scenario.context_lens[i]);
    }

    write_fixture(
        scenario, shape, layout, allocator, cache, sequences, pool_blocks, max_blocks_per_seq, root);
}

/**
 * @brief Returns the physical frames a sequence currently maps, in logical order.
 */
std::vector<PhysicalBlockId> mapped_frames(KVCacheManager& cache, SequenceId sequence) {
    const std::optional<CacheView> view = cache.cache_view(sequence);
    require(view.has_value(), "cache_view failed");

    std::vector<PhysicalBlockId> frames;
    for (const BlockTableEntry& entry : view->block_table->entries()) {
        frames.push_back(entry.physical_id);
    }
    return frames;
}

/**
 * @brief Exports a fixture whose pool state is the product of allocator churn.
 *
 * Every other scenario fills a freshly zeroed pool exactly once, so all four
 * validate the kernel against cache contents that no fork, copy-on-write, or
 * frame recycle ever touched. That leaves the half of a paged allocator worth
 * having unchecked. A fixture built by a clean fill still passes if a fork
 * corrupts its parent, if copy-on-write leaves a block table pointing at the
 * pre-copy frame, or if a recycled frame's stale bytes leak past the context
 * length of a partially filled block.
 *
 * Three sequences go into the batch:
 *
 *   0  the fork parent, which must still read its own contents after the child
 *      has written over what it thought it shared
 *   1  the child, every one of whose blocks was materialized by copy-on-write
 *   2  a sequence built on frames recycled from a released one, ending mid-block
 *      so the tail of its last frame still holds the dead sequence's KV
 *
 * The pool is prefilled with a deterministic non-zero pattern instead of being
 * zeroed. Stale data the kernel must never read is then real data rather than
 * zeros, which a masking bug could hide behind: summing exp(q.0)*0 over a
 * zeroed tail perturbs only the softmax denominator, while a garbage tail moves
 * the output somewhere unmistakable.
 */
void export_churn_scenario(const fs::path& root) {
    const Scenario scenario{
        "cow_and_recycled_frames",
        "a fork parent, its copy-on-write child, and a sequence on recycled frames",
        4, 2, 2, 8, 4, 1, 0.35F, {10, 10, 6}, false};

    BlockShape shape;
    shape.tokens_per_block = scenario.tokens_per_block;
    shape.num_layers = scenario.num_layers;
    shape.num_kv_heads = scenario.num_kv_heads;
    shape.head_dim = scenario.head_dim;
    shape.bytes_per_element = static_cast<std::uint32_t>(sizeof(float));

    KVBlockLayout layout;
    layout.shape = shape;

    // Peak demand is parent 3 + child 3 + recycled 2; the victim's three frames
    // are back on the free list before any of those are claimed. The spare two
    // are never allocated, so they keep the prefill pattern and a kernel that
    // wanders out of its block table lands in obvious garbage.
    constexpr std::uint32_t kPoolBlocks = 10;
    constexpr std::uint32_t kMaxBlocksPerSeq = 3;

    // Distinct from every exported sequence's salt, so the victim's leftovers
    // can never be mistaken for a correct read.
    constexpr std::uint32_t kVictimSalt = 7;

    constexpr SequenceId kVictim = 100;
    constexpr SequenceId kParent = 1;
    constexpr SequenceId kChild = 2;
    constexpr SequenceId kRecycled = 3;

    AllocatorConfig config;
    config.block_shape = shape;
    config.max_blocks = kPoolBlocks;
    MemoryAllocator allocator(config);
    KVCacheManager cache(allocator);

    for (PhysicalBlockId id = 0; id < kPoolBlocks; ++id) {
        PhysicalBlock* block = allocator.block(id);
        require(block != nullptr, "pool frame missing");
        float* base = reinterpret_cast<float*>(block->data());
        const std::size_t count = block->size_bytes() / sizeof(float);
        for (std::size_t i = 0; i < count; ++i) {
            base[i] = mix({2U, id, static_cast<std::uint32_t>(i)});
        }
    }

    require(
        cache.create_sequence(SequenceMetadata{kParent, kParent, {}, {}}), "create parent failed");
    require(cache.reserve_tokens(kParent, scenario.context_lens[0]), "reserve parent failed");
    fill_sequence(shape, layout, allocator, cache, kParent, 0, scenario.context_lens[0]);

    require(
        cache.fork_sequence(kParent, SequenceMetadata{kChild, kChild, {}, {}}),
        "fork_sequence failed");

    // The child shares every one of the parent's blocks right now, so each first
    // touch below has to materialize a private copy.
    fill_sequence(shape, layout, allocator, cache, kChild, 1, scenario.context_lens[1]);

    // A sequence that lives just long enough to dirty three frames and hand them
    // back. The free list is LIFO, so this has to happen immediately before the
    // recycled sequence allocates or some earlier sequence takes the frames
    // instead -- which is what the reuse check below caught the first time.
    require(
        cache.create_sequence(SequenceMetadata{kVictim, kVictim, {}, {}}), "create victim failed");
    require(cache.reserve_tokens(kVictim, scenario.context_lens[0]), "reserve victim failed");
    fill_sequence(shape, layout, allocator, cache, kVictim, kVictimSalt, scenario.context_lens[0]);
    const std::vector<PhysicalBlockId> victim_frames = mapped_frames(cache, kVictim);
    cache.release_sequence(kVictim);

    require(
        cache.create_sequence(SequenceMetadata{kRecycled, kRecycled, {}, {}}),
        "create recycled failed");
    require(cache.reserve_tokens(kRecycled, scenario.context_lens[2]), "reserve recycled failed");
    fill_sequence(shape, layout, allocator, cache, kRecycled, 2, scenario.context_lens[2]);

    // Guard the fixture's premises. Without these the scenario could quietly
    // decay into three ordinary sequences and still pass, pinning down a weaker
    // property than its name claims.
    const std::vector<PhysicalBlockId> parent_frames = mapped_frames(cache, kParent);
    const std::vector<PhysicalBlockId> child_frames = mapped_frames(cache, kChild);
    const std::vector<PhysicalBlockId> recycled_frames = mapped_frames(cache, kRecycled);

    for (const PhysicalBlockId frame : child_frames) {
        require(
            std::find(parent_frames.begin(), parent_frames.end(), frame) == parent_frames.end(),
            "child still shares a frame with its parent: copy-on-write did not happen");
    }
    require(
        std::any_of(
            recycled_frames.begin(),
            recycled_frames.end(),
            [&victim_frames](PhysicalBlockId frame) {
                return std::find(victim_frames.begin(), victim_frames.end(), frame) !=
                       victim_frames.end();
            }),
        "recycled sequence drew only fresh frames: the fixture is not testing reuse");
    require(
        scenario.context_lens[2] % scenario.tokens_per_block != 0,
        "recycled sequence must end mid-block to leave a stale tail exposed");

    write_fixture(
        scenario, shape, layout, allocator, cache, {kParent, kChild, kRecycled}, kPoolBlocks,
        kMaxBlocksPerSeq, root);
}

std::vector<Scenario> scenarios() {
    return {
        Scenario{
            "single_head_contiguous",
            "smallest case that exercises nothing but the softmax and the dot product",
            4, 1, 1, 4, 1, 0, 1.0F, {8}, false},
        Scenario{
            "scattered_partial_block",
            "non-zero layer over scattered frames with a partially filled last block",
            4, 2, 2, 4, 2, 1, 0.5F, {10}, true},
        Scenario{
            "grouped_query_heads",
            "group size 4, so a kernel that ignores the query-head mapping fails",
            4, 1, 2, 4, 8, 0, 0.5F, {11}, true},
        Scenario{
            "batched_head_dim_128",
            "ragged batch at the real head_dim and GQA ratio, with block-table padding",
            16, 2, 2, 128, 4, 1, 0.088388348F, {40, 17, 63}, true},
    };
}

} // namespace
} // namespace qwenvl_paged

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::path("fixtures");

    try {
        fs::create_directories(root);
        for (const qwenvl_paged::Scenario& scenario : qwenvl_paged::scenarios()) {
            qwenvl_paged::export_scenario(scenario, root);
        }
        qwenvl_paged::export_churn_scenario(root);
    } catch (const std::exception& error) {
        std::cerr << "golden_export failed: " << error.what() << "\n";
        return 1;
    }

    std::cout << "wrote fixtures to " << root << "\n";
    return 0;
}
