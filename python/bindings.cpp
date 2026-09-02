/**
 * @brief pybind11 bindings for the paged KV cache core.
 *
 * The point of this module is the seam. Weeks 15-16 measured a Triton kernel
 * against exported .npz snapshots, and weeks 1-9 measured an allocator, but the
 * kernel had never read a pool the allocator was actually managing. These
 * bindings let Python own the pool's storage, hand it to the allocator, drive
 * copy-on-write and recycling through the real C++ paths, and then point the
 * kernel at the same bytes.
 *
 * Ownership runs one way only: Python allocates the slab and keeps it alive,
 * and MemoryAllocator adopts the pointer. That keeps CUDA out of the C++ core
 * and lets the pool be a torch tensor the kernel can take directly. The
 * `keep_alive` on the buffer constructor makes the contract the header
 * describes unbreakable from Python rather than merely documented. The
 * by-address constructor cannot do that, for the reason given at its
 * definition, and is the one place here where Python can still get it wrong.
 *
 * Scheduler is deliberately not exposed. Nothing in the seam needs it, and it
 * belongs with the serving-policy work rather than here.
 */

#include "qwenvl_paged/Block.h"
#include "qwenvl_paged/CacheLayout.h"
#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/PagedAttention.h"
#include "qwenvl_paged/SwapBackend.h"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace qwenvl_paged;

namespace {

/**
 * @brief Returns a writable, contiguous pointer to a buffer large enough for a pool.
 *
 * Adopting a short or strided buffer would put the last frames off the end of
 * it, which is a segfault a long way from its cause, so both are refused here.
 */
std::byte* adoptable_pointer(const py::buffer& storage, const AllocatorConfig& config) {
    const py::buffer_info info = storage.request(true);

    if (!(info.ndim == 1 || info.ndim == 0)) {
        throw std::invalid_argument("pool storage must be a flat buffer");
    }

    const std::size_t bytes = static_cast<std::size_t>(info.size) * static_cast<std::size_t>(info.itemsize);
    const std::size_t required = pool_bytes_for(config);
    if (bytes < required) {
        throw std::invalid_argument(
            "pool storage holds " + std::to_string(bytes) + " bytes but this config needs " +
            std::to_string(required) + "; size it with pool_bytes_for(config)");
    }

    return static_cast<std::byte*>(info.ptr);
}

std::vector<PhysicalBlockId> block_table_of(KVCacheManager& cache, SequenceId sequence) {
    const std::optional<CacheView> view = cache.cache_view(sequence);
    if (!view.has_value()) {
        throw std::invalid_argument("no cache view for sequence " + std::to_string(sequence));
    }

    std::vector<PhysicalBlockId> frames;
    for (const BlockTableEntry& entry : view->block_table->entries()) {
        frames.push_back(entry.physical_id);
    }
    return frames;
}

/**
 * @brief Runs the CPU reference kernel over a sequence and returns its output.
 *
 * This is the oracle the GPU kernel is checked against, so it is exposed as the
 * same call Python makes rather than reimplemented on the Python side.
 */
py::array_t<float> decode(
    KVCacheManager& cache,
    SequenceId sequence,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& query,
    std::uint32_t layer,
    std::uint32_t num_query_heads,
    std::uint32_t context_len,
    float scale) {
    const std::optional<CacheView> view = cache.cache_view(sequence);
    if (!view.has_value()) {
        throw std::invalid_argument("no cache view for sequence " + std::to_string(sequence));
    }

    const std::uint32_t head_dim = view->layout.shape.head_dim;
    const auto expected = static_cast<py::ssize_t>(num_query_heads) * head_dim;
    if (query.size() != expected) {
        throw std::invalid_argument(
            "query holds " + std::to_string(query.size()) + " elements, expected " +
            std::to_string(expected));
    }

    py::array_t<float> out({num_query_heads, head_dim});

    PagedAttentionParams params;
    params.layer = layer;
    params.num_query_heads = num_query_heads;
    params.context_len = context_len;
    params.scale = scale;

    if (!paged_attention_decode<float>(*view, query.data(), params, out.mutable_data())) {
        throw std::runtime_error(
            "paged_attention_decode refused the call: a block is unmapped or swapped out");
    }
    return out;
}

} // namespace

PYBIND11_MODULE(qwenvl_paged, module) {
    module.doc() = "Paged KV cache allocator and CPU reference kernel.";

    py::enum_<CacheKind>(module, "CacheKind")
        .value("TextKV", CacheKind::TextKV)
        .value("VisionKV", CacheKind::VisionKV)
        .value("RopeState", CacheKind::RopeState)
        .value("Auxiliary", CacheKind::Auxiliary);

    py::enum_<BlockState>(module, "BlockState")
        .value("Free", BlockState::Free)
        .value("Active", BlockState::Active)
        .value("Shared", BlockState::Shared)
        .value("Swapped", BlockState::Swapped);

    py::enum_<KVStream>(module, "KVStream")
        .value("Key", KVStream::Key)
        .value("Value", KVStream::Value);

    py::class_<BlockShape>(module, "BlockShape")
        .def(py::init<>())
        .def_readwrite("tokens_per_block", &BlockShape::tokens_per_block)
        .def_readwrite("num_layers", &BlockShape::num_layers)
        .def_readwrite("num_kv_heads", &BlockShape::num_kv_heads)
        .def_readwrite("head_dim", &BlockShape::head_dim)
        .def_readwrite("bytes_per_element", &BlockShape::bytes_per_element)
        .def_readwrite("layers_per_frame", &BlockShape::layers_per_frame)
        .def("byte_size", &BlockShape::byte_size);

    py::class_<HostMemoryOptions>(module, "HostMemoryOptions")
        .def(py::init<>())
        .def_readwrite("alignment_bytes", &HostMemoryOptions::alignment_bytes)
        .def_readwrite("prefer_pinned_memory", &HostMemoryOptions::prefer_pinned_memory);

    py::class_<AllocatorConfig>(module, "AllocatorConfig")
        .def(py::init<>())
        .def_readwrite("block_shape", &AllocatorConfig::block_shape)
        .def_readwrite("memory_options", &AllocatorConfig::memory_options)
        .def_readwrite("max_blocks", &AllocatorConfig::max_blocks);

    py::class_<AllocatorStats>(module, "AllocatorStats")
        .def_readonly("total_blocks", &AllocatorStats::total_blocks)
        .def_readonly("free_blocks", &AllocatorStats::free_blocks)
        .def_readonly("active_blocks", &AllocatorStats::active_blocks)
        .def_readonly("shared_blocks", &AllocatorStats::shared_blocks)
        .def_readonly("swapped_blocks", &AllocatorStats::swapped_blocks)
        .def_readonly("bytes_reserved", &AllocatorStats::bytes_reserved);

    py::class_<PhysicalBlockInfo>(module, "PhysicalBlockInfo")
        .def_readonly("id", &PhysicalBlockInfo::id)
        .def_readonly("state", &PhysicalBlockInfo::state)
        .def_readonly("ref_count", &PhysicalBlockInfo::ref_count)
        .def_readonly("generation", &PhysicalBlockInfo::generation);

    py::class_<KVBlockLayout>(module, "KVBlockLayout")
        .def(py::init<>())
        .def_readwrite("shape", &KVBlockLayout::shape)
        .def("head_stride", &KVBlockLayout::head_stride)
        .def("token_stride", &KVBlockLayout::token_stride)
        .def("stream_stride", &KVBlockLayout::stream_stride)
        .def("layer_stride", &KVBlockLayout::layer_stride)
        .def("element_count", &KVBlockLayout::element_count)
        .def("element_offset", &KVBlockLayout::element_offset);

    module.def("block_stride_for", &block_stride_for);
    module.def("pool_bytes_for", &pool_bytes_for);

    py::class_<MemoryAllocator>(module, "MemoryAllocator")
        .def(
            py::init([](const AllocatorConfig& config, const py::buffer& storage) {
                return std::make_unique<MemoryAllocator>(config, adoptable_pointer(storage, config));
            }),
            py::arg("config"),
            py::arg("storage"),
            // The allocator holds a raw pointer into the caller's buffer, so the
            // buffer has to outlive it. Tying the lifetimes here turns the
            // header's contract into something Python cannot get wrong.
            py::keep_alive<1, 3>(),
            "Adopts a caller-owned slab; size it with pool_bytes_for(config).")
        .def(
            // The device path. A CUDA tensor exposes no buffer protocol, so the
            // only thing crossing is its address, and with it goes the
            // keep_alive the overload above relies on: nothing here can tie the
            // allocator's lifetime to storage it cannot see. The caller has to
            // hold the tensor. `nbytes` is required so that at least the size
            // mistake is still caught here rather than as a later stray write.
            py::init([](const AllocatorConfig& config, std::uintptr_t address, std::size_t nbytes) {
                const std::size_t required = pool_bytes_for(config);
                if (nbytes < required) {
                    throw std::invalid_argument(
                        "pool storage holds " + std::to_string(nbytes) +
                        " bytes but this config needs " + std::to_string(required) +
                        "; size it with pool_bytes_for(config)");
                }
                if (address == 0) {
                    throw std::invalid_argument("pool storage address is null");
                }
                return std::make_unique<MemoryAllocator>(
                    config, reinterpret_cast<std::byte*>(address));
            }),
            py::arg("config"),
            py::arg("address"),
            py::arg("nbytes"),
            "Adopts a slab by address, for storage the CPU cannot read. The caller "
            "must keep it alive and must set a copy hook before forking.")
        .def(
            "set_copy_hook",
            &MemoryAllocator::set_copy_hook,
            py::arg("hook"),
            "Replaces the host memcpy behind copy-on-write with hook(source_id, "
            "destination_id). Required for a pool the CPU cannot address.")
        .def("allocate", &MemoryAllocator::allocate)
        .def("release", &MemoryAllocator::release)
        .def("retain", &MemoryAllocator::retain)
        .def("copy_block", &MemoryAllocator::copy_block)
        .def("info", &MemoryAllocator::info, py::return_value_policy::reference_internal)
        .def("stats", &MemoryAllocator::stats)
        .def("can_allocate", &MemoryAllocator::can_allocate)
        .def("block_stride_bytes", &MemoryAllocator::block_stride_bytes)
        .def("pool_bytes", &MemoryAllocator::pool_bytes)
        .def("owns_pool", &MemoryAllocator::owns_pool);

    py::class_<KVCacheManager>(module, "KVCacheManager")
        .def(
            py::init<MemoryAllocator&>(),
            py::arg("allocator"),
            py::keep_alive<1, 2>())
        .def(
            "create_sequence",
            [](KVCacheManager& cache, SequenceId sequence, RequestId request) {
                return cache.create_sequence(SequenceMetadata{sequence, request, {}, {}});
            },
            py::arg("sequence_id"),
            py::arg("request_id"))
        .def(
            "fork_sequence",
            [](KVCacheManager& cache, SequenceId parent, SequenceId child, RequestId request) {
                return cache.fork_sequence(parent, SequenceMetadata{child, request, {}, {}});
            },
            py::arg("parent_id"),
            py::arg("child_id"),
            py::arg("request_id"))
        .def("reserve_tokens", &KVCacheManager::reserve_tokens)
        .def(
            "ensure_token_writable",
            &KVCacheManager::ensure_token_writable,
            py::arg("sequence_id"),
            py::arg("token_position"),
            py::arg("cache_kind") = CacheKind::TextKV,
            py::arg("layer") = 0)
        .def("swap_out_sequence", &KVCacheManager::swap_out_sequence)
        .def("swap_in_sequence", &KVCacheManager::swap_in_sequence)
        .def("release_sequence", &KVCacheManager::release_sequence)
        .def("contains", &KVCacheManager::contains)
        .def("block_table", &block_table_of, py::arg("sequence_id"))
        .def(
            "decode",
            &decode,
            py::arg("sequence_id"),
            py::arg("query"),
            py::arg("layer"),
            py::arg("num_query_heads"),
            py::arg("context_len"),
            py::arg("scale"),
            "Runs the CPU reference kernel, the oracle the GPU kernel is checked against.");

    py::class_<HostSwapBackend>(module, "HostSwapBackend")
        .def(py::init<std::size_t>(), py::arg("capacity_slots"))
        .def("resident_slots", &HostSwapBackend::resident_slots)
        .def("capacity_slots", &HostSwapBackend::capacity_slots);

    module.def(
        "set_swap_backend",
        [](MemoryAllocator& allocator, HostSwapBackend* backend) {
            allocator.set_swap_backend(backend);
        },
        py::arg("allocator"),
        py::arg("backend"),
        py::keep_alive<1, 2>());
}
