"""A transformers cache backed by the paged KV allocator.

This is the piece that puts a real model on top of the memory subsystem. The
roadmap describes it as subclassing the `Cache` interface, which was accurate
for transformers 4.x; 5.x restructured `Cache` into a container of per-layer
`CacheLayerMixin` objects, so the extension point is now one layer object per
decoder layer. That turns out to fit better than the thing it replaced, because
`BlockShape` already carries `num_layers` and a physical block already holds
every layer's slice of a token range. One sequence's block table therefore backs
all layers at once, and a `PagedLayer` is just a window onto its own layer's
stride within those same frames.

What this does *not* do is put the Triton kernel in the forward pass. `update`
has to hand back contiguous K/V for torch's attention to consume, so every step
gathers the sequence's blocks into a dense tensor. That gather is the cost the
roadmap asked to have measured rather than hidden, and it is the concrete
argument for a native paged kernel. Validating that kernel against a model is a
separate problem from validating this cache against one.

Restrictions, all deliberate and all asserted rather than silently handled:
batch size 1 and greedy decode per sequence, no beam reordering. Several
sequences can share one pool, which is what `fork_paged_cache` is for, but they
are stepped one at a time rather than batched. This exists to answer whether
paged storage produces identical tokens, not to be a serving path.
"""

import numpy as np
import torch
from transformers.cache_utils import Cache, CacheLayerMixin

import qwenvl_paged as qp

DEFAULT_TOKENS_PER_BLOCK = 16
SEQUENCE_ID = 1


class PagedPool:
    """Allocator, cache manager, and slab shared by every layer of one sequence.

    Reservation is the reason this is shared rather than per-layer. A physical
    block spans all layers, so the sequence's capacity must grow once per token
    position, not once per layer per token position. `ensure_capacity` is
    idempotent so all `num_layers` callers can ask and only the first does work.
    """

    def __init__(self, *, num_layers, num_kv_heads, head_dim, max_blocks,
                 tokens_per_block=DEFAULT_TOKENS_PER_BLOCK, scatter=False,
                 dtype=torch.float32):
        config = qp.AllocatorConfig()
        config.block_shape.tokens_per_block = tokens_per_block
        config.block_shape.num_layers = num_layers
        config.block_shape.num_kv_heads = num_kv_heads
        config.block_shape.head_dim = head_dim
        config.block_shape.bytes_per_element = dtype.itemsize
        config.max_blocks = max_blocks

        self.config = config
        self.tokens_per_block = tokens_per_block
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # The slab is bytes, which is what C++ thinks it is, and torch reads it
        # through a zero-copy view at the model's dtype. Going through numpy at
        # the element dtype instead would rule out bfloat16, which numpy has no
        # type for and every real checkpoint here is stored in.
        self.storage = np.zeros(qp.pool_bytes_for(config), dtype=np.uint8)
        self.allocator = qp.MemoryAllocator(config, self.storage)
        self.manager = qp.KVCacheManager(self.allocator)
        self.frames = torch.from_numpy(self.storage).view(dtype).reshape(
            max_blocks, qp.block_stride_for(config) // dtype.itemsize)

        self.layout = qp.KVBlockLayout()
        self.layout.shape = config.block_shape

        # Tokens of one layer and one stream inside one block are contiguous:
        # the layout is [layer][K|V][token][kv_head][dim] with dim innermost, so
        # a block's slice for a given layer and stream is exactly
        # [tokens_per_block, num_kv_heads, head_dim]. Writes and gathers are
        # therefore whole-block array operations rather than per-token loops.
        self.block_span = tokens_per_block * num_kv_heads * head_dim

        self.root_id = SEQUENCE_ID
        assert self.manager.create_sequence(self.root_id, self.root_id)

        # A second sequence that takes a frame between each of ours and is never
        # released, leaving the block table pointing at frames that are not a
        # contiguous run. Without it the frames come out adjacent and a gather
        # that ignored the block table entirely would still return the right
        # bytes, so the test would pin down nothing.
        self.scatter = scatter
        self.spacer_id = self.root_id + 1000
        if scatter:
            assert self.manager.create_sequence(self.spacer_id, self.spacer_id)

    def fork(self, parent_id, child_id):
        """Branches a sequence, sharing every one of its blocks until written."""
        assert self.manager.fork_sequence(parent_id, child_id, child_id)

    def ensure_capacity(self, sequence_id, total_tokens):
        """Grows a block table to cover `total_tokens`, if it does not already.

        Counted in blocks rather than tokens on purpose. `reserve_tokens` rounds
        up and always appends, so it cannot top up a partially filled tail
        block: asking it for the one new token of a decode step appends a whole
        new block and strands the fifteen slots already reserved. Asking only
        for the blocks the table is actually missing is what makes this both
        correct and idempotent across the `num_layers` callers.
        """
        have = len(self.block_table(sequence_id))
        need = -(-total_tokens // self.tokens_per_block)
        for _ in range(need - have):
            assert self.manager.reserve_tokens(sequence_id, self.tokens_per_block), (
                f"pool exhausted growing to {need} blocks; raise max_blocks")
            if self.scatter:
                assert self.manager.reserve_tokens(self.spacer_id, self.tokens_per_block), (
                    "pool exhausted reserving a spacer frame; raise max_blocks")

    def stream_view(self, frame, layer_idx, stream):
        """The [tokens_per_block, num_kv_heads, head_dim] slice of one frame."""
        base = self.layout.element_offset(layer_idx, stream, 0, 0)
        assert base is not None, "element_offset out of range"
        return self.frames[frame, base:base + self.block_span].reshape(
            self.tokens_per_block, self.num_kv_heads, self.head_dim)

    def writable_frame(self, sequence_id, token):
        """Resolves a token's frame, honoring copy-on-write before any write."""
        frame = self.manager.ensure_token_writable(sequence_id, token)
        assert frame is not None, f"ensure_token_writable failed at token {token}"
        return frame

    def block_table(self, sequence_id):
        return self.manager.block_table(sequence_id)

    def frames_in_use(self):
        """Distinct frames the pool has handed out, counting a shared one once.

        `active_blocks` alone is not this. The allocator splits the two states:
        Active is a frame exactly one sequence holds, Shared is one that several
        do, and a frame every branch is reading is in the second bucket.
        """
        stats = self.allocator.stats()
        return stats.active_blocks + stats.shared_blocks


class PagedLayer(CacheLayerMixin):
    """One decoder layer's window onto the shared paged sequence."""

    def __init__(self, pool: PagedPool, layer_idx: int, sequence_id: int):
        super().__init__()
        self.pool = pool
        self.layer_idx = layer_idx
        self.sequence_id = sequence_id
        self.length = 0
        self.is_initialized = False
        self.gathered_elements = 0

    def lazy_initialization(self, key_states, value_states):
        assert key_states.dtype == self.pool.dtype, (
            f"model runs in {key_states.dtype} but the pool was sized for {self.pool.dtype}")
        assert key_states.shape[0] == 1, "PagedLayer handles batch size 1"
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        new_tokens = key_states.shape[-2]
        start, end = self.length, self.length + new_tokens
        self.pool.ensure_capacity(self.sequence_id, end)

        # [1, kv_heads, tokens, dim] -> [tokens, kv_heads, dim], matching the
        # in-block layout so each block is one assignment.
        keys = key_states[0].permute(1, 0, 2).to("cpu")
        values = value_states[0].permute(1, 0, 2).to("cpu")

        tokens_per_block = self.pool.tokens_per_block
        position = start
        while position < end:
            offset = position % tokens_per_block
            count = min(tokens_per_block - offset, end - position)
            frame = self.pool.writable_frame(self.sequence_id, position)
            source = slice(position - start, position - start + count)

            self.pool.stream_view(frame, self.layer_idx, qp.KVStream.Key)[offset:offset + count] = keys[source]
            self.pool.stream_view(frame, self.layer_idx, qp.KVStream.Value)[offset:offset + count] = values[source]
            position += count

        self.length = end
        return self._gather()

    def _gather(self):
        """Reassembles the sequence into the contiguous tensors attention wants.

        This is the paging tax on a torch attention path, and it is why a native
        paged kernel is worth having: the blocks are already the right bytes in
        the right order, and this copies all of them anyway.
        """
        frames = self.pool.block_table(self.sequence_id)
        tokens_per_block = self.pool.tokens_per_block

        key_blocks, value_blocks, remaining = [], [], self.length
        for frame in frames:
            if remaining <= 0:
                break
            count = min(tokens_per_block, remaining)
            key_blocks.append(self.pool.stream_view(frame, self.layer_idx, qp.KVStream.Key)[:count])
            value_blocks.append(self.pool.stream_view(frame, self.layer_idx, qp.KVStream.Value)[:count])
            remaining -= count

        keys = torch.cat(key_blocks)
        values = torch.cat(value_blocks)
        self.gathered_elements += keys.numel() + values.numel()

        # Concatenated on the host first so the whole context crosses to the
        # device once per stream rather than once per block.
        def to_model(tensor):
            return tensor.permute(1, 0, 2).unsqueeze(0).to(self.device)

        return to_model(keys), to_model(values)

    def get_seq_length(self) -> int:
        return self.length

    def get_max_length(self) -> int:
        return -1

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def reset(self) -> None:
        self.length = 0

    def reorder_cache(self, beam_idx) -> None:
        raise NotImplementedError("PagedLayer is greedy-decode only")

    def offload(self):
        pass

    def prefetch(self):
        pass


def build_paged_cache(text_config, max_blocks, tokens_per_block=DEFAULT_TOKENS_PER_BLOCK,
                      scatter=False, dtype=torch.float32):
    """Returns a transformers Cache backed by one paged pool, and the pool."""
    head_dim = getattr(text_config, "head_dim", None) or (
        text_config.hidden_size // text_config.num_attention_heads)

    pool = PagedPool(
        num_layers=text_config.num_hidden_layers,
        num_kv_heads=text_config.num_key_value_heads,
        head_dim=head_dim,
        max_blocks=max_blocks,
        tokens_per_block=tokens_per_block,
        scatter=scatter,
        dtype=dtype,
    )
    layers = [PagedLayer(pool, index, pool.root_id)
              for index in range(text_config.num_hidden_layers)]
    return Cache(layers=layers), pool


def fork_paged_cache(parent, pool, child_id):
    """Branches a filled cache into a child that shares its blocks.

    This is the parallel-sampling shape: one prompt, several continuations. The
    child starts at the parent's length holding none of its own memory, and only
    materializes a private copy of a block when it writes into one -- which for
    a decode is just the partially filled tail block, not the whole prompt.
    """
    pool.fork(parent.layers[0].sequence_id, child_id)

    layers = []
    for source in parent.layers:
        layer = PagedLayer(pool, source.layer_idx, child_id)
        layer.length = source.length
        layer.is_initialized = source.is_initialized
        layer.dtype, layer.device = source.dtype, source.device
        layers.append(layer)
    return Cache(layers=layers)
