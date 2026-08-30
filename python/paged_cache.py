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

Prefill still gathers: the Triton kernel is decode-only, so a prompt longer than
one token uses torch attention over a dense K/V. Decode writes the new token
into its page and, when `use_kernel` is set, skips the gather. A registered
attention implementation then launches Triton against the slab and the block
table. That is the path that removes the 12 ms/step measured in week 19.

Restrictions, all deliberate and all asserted rather than silently handled:
batch size 1 and greedy decode per sequence, no beam reordering. Several
sequences can share one pool, which is what `fork_paged_cache` is for, but they
are stepped one at a time rather than batched.
"""

import pathlib
import sys

import numpy as np
import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward

import qwenvl_paged as qp

_KERNEL = pathlib.Path(__file__).resolve().parent.parent / "tools" / "triton_kernel"
if str(_KERNEL) not in sys.path:
    sys.path.insert(0, str(_KERNEL))

from paged_attention_decode import paged_attention_decode_partitioned

DEFAULT_TOKENS_PER_BLOCK = 16
SEQUENCE_ID = 1


class PagedPool:
    """Allocator, cache manager, and slab shared by every layer of one sequence.

    Reservation is the reason this is shared rather than per-layer. A physical
    block spans all layers, so the sequence's capacity must grow once per token
    position, not once per layer per token position. `ensure_capacity` is
    idempotent so all `num_layers` callers can ask and only the first does work.

    With `device="cuda"` the slab is device-resident and the allocator adopts its
    address, which keeps writes and gathers off the PCIe bus. Two things stay
    host-only in that mode and will fault if used: the swap backend, and the CPU
    reference kernel reached through `KVCacheManager.decode`. Copy-on-write does
    work, because `set_copy_hook` redirects the one site that moves bytes.
    """

    def __init__(self, *, num_layers, num_kv_heads, head_dim, max_blocks,
                 tokens_per_block=DEFAULT_TOKENS_PER_BLOCK, scatter=False,
                 dtype=torch.float32, device="cpu", use_kernel=False):
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
        self.device = torch.device(device)
        self.use_kernel = use_kernel
        if use_kernel:
            assert self.device.type == "cuda", "the Triton kernel reads a device pool"

        # The slab is bytes, which is what C++ thinks it is, and torch reads it
        # through a zero-copy view at the model's dtype. Going through numpy at
        # the element dtype instead would rule out bfloat16, which numpy has no
        # type for and every real checkpoint here is stored in.
        pool_bytes = qp.pool_bytes_for(config)
        if self.device.type == "cpu":
            self.storage = np.zeros(pool_bytes, dtype=np.uint8)
            self.allocator = qp.MemoryAllocator(config, self.storage)
            frames = torch.from_numpy(self.storage)
        else:
            # Only the address crosses, so the allocator cannot keep this alive
            # for us; holding it on the pool is what stops it being collected
            # out from under C++.
            self.storage = torch.zeros(pool_bytes, dtype=torch.uint8, device=self.device)
            self.allocator = qp.MemoryAllocator(config, self.storage.data_ptr(), pool_bytes)
            frames = self.storage

        self.manager = qp.KVCacheManager(self.allocator)
        self.frames = frames.view(dtype).reshape(
            max_blocks, qp.block_stride_for(config) // dtype.itemsize)

        if self.device.type != "cpu":
            # Copy-on-write is the one path that moves bytes, and its default is
            # a host memcpy that would fault on this storage. Row `i` of frames
            # is frame `i`, because a block id is an offset into the slab.
            self.allocator.set_copy_hook(
                lambda source, destination: self.frames[destination].copy_(self.frames[source]))

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

        # Device tensors shared by every layer of a step. Invalidated when a
        # write remaps a frame or the table grows; see decode_inputs.
        self._host_frames = {}
        self._dev_table = {}
        self._dev_context = {}
        self._cached_length = {}
        self._table_len = {}
        self._writable = None
        self._scratch = {}

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
        have = self._table_len.get(sequence_id)
        if have is None:
            have = len(self.block_table(sequence_id))
        need = -(-total_tokens // self.tokens_per_block)
        for _ in range(need - have):
            assert self.manager.reserve_tokens(sequence_id, self.tokens_per_block), (
                f"pool exhausted growing to {need} blocks; raise max_blocks")
            if self.scatter:
                assert self.manager.reserve_tokens(self.spacer_id, self.tokens_per_block), (
                    "pool exhausted reserving a spacer frame; raise max_blocks")
            have += 1
            self._invalidate_table(sequence_id)
        self._table_len[sequence_id] = have

    def stream_base(self, layer_idx, stream):
        """Element offset of one layer's key or value slice within a frame."""
        base = self.layout.element_offset(layer_idx, stream, 0, 0)
        assert base is not None, "element_offset out of range"
        return base

    def stream_view(self, frame, layer_idx, stream):
        """The [tokens_per_block, num_kv_heads, head_dim] slice of one frame."""
        base = self.stream_base(layer_idx, stream)
        return self.frames[frame, base:base + self.block_span].reshape(
            self.tokens_per_block, self.num_kv_heads, self.head_dim)

    def _invalidate_table(self, sequence_id):
        self._host_frames.pop(sequence_id, None)
        self._dev_table.pop(sequence_id, None)
        self._dev_context.pop(sequence_id, None)
        self._cached_length.pop(sequence_id, None)

    def decode_inputs(self, sequence_id, length):
        """Block table and context length for a decode launch, cached per sequence.

        Every layer of a step reads the same frames. Rebuilding that tensor 28
        times was 3.4 ms/step; the host list is what lets a remapping write
        invalidate without a device-to-host sync.
        """
        needed = -(-length // self.tokens_per_block)
        table = self._dev_table.get(sequence_id)
        if table is None or table.shape[1] != needed:
            host = self.block_table(sequence_id)[:needed]
            self._host_frames[sequence_id] = host
            table = torch.tensor(host, dtype=torch.int32, device=self.device).unsqueeze(0)
            self._dev_table[sequence_id] = table

        context = self._dev_context.get(sequence_id)
        if context is None or self._cached_length[sequence_id] != length:
            # New tensor, not fill_: a previous layer's kernel may still be
            # reading the old one.
            context = torch.tensor([length], dtype=torch.int32, device=self.device)
            self._dev_context[sequence_id] = context
            self._cached_length[sequence_id] = length
        return table, context

    def frame_index(self, sequence_id, count):
        """The first `count` frames of a sequence, as an index tensor."""
        table, _ = self.decode_inputs(sequence_id, count * self.tokens_per_block)
        return table[0, :count].to(torch.long)

    def writable_frame(self, sequence_id, token):
        """Resolves a token's frame, honoring copy-on-write before any write.

        Cached for the current token: all layers of a decode step write the
        same position, and the first call is the one that materializes a
        shared block.
        """
        hit = self._writable
        if hit is not None and hit[0] == sequence_id and hit[1] == token:
            return hit[2]

        frame = self.manager.ensure_token_writable(sequence_id, token)
        assert frame is not None, f"ensure_token_writable failed at token {token}"
        host = self._host_frames.get(sequence_id)
        if host is not None:
            logical = token // self.tokens_per_block
            if logical >= len(host) or host[logical] != frame:
                self._invalidate_table(sequence_id)
        self._writable = (sequence_id, token, frame)
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
        self._key_base = self.pool.stream_base(self.layer_idx, qp.KVStream.Key)
        self._val_base = self.pool.stream_base(self.layer_idx, qp.KVStream.Value)
        self._token_elems = self.pool.num_kv_heads * self.pool.head_dim
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        new_tokens = key_states.shape[-2]
        start, end = self.length, self.length + new_tokens
        self.pool.ensure_capacity(self.sequence_id, end)

        if new_tokens == 1:
            # One token, already [kv_heads, dim] in the model's layout. The
            # prefill loop below permutes a whole span and walks block
            # boundaries; that was 0.11 ms/layer to store 4 KiB.
            self._write_one(key_states, value_states, start)
            self.length = end
            if self.pool.use_kernel:
                return key_states, value_states
            return self._gather()

        # [1, kv_heads, tokens, dim] -> [tokens, kv_heads, dim], matching the
        # in-block layout so each block is one assignment. This crosses to the
        # host only when the pool is there.
        keys = key_states[0].permute(1, 0, 2).to(self.pool.device)
        values = value_states[0].permute(1, 0, 2).to(self.pool.device)

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

    def _write_one(self, key_states, value_states, position):
        frame = self.pool.writable_frame(self.sequence_id, position)
        offset = position % self.pool.tokens_per_block
        tok = self._token_elems
        key_at = self._key_base + offset * tok
        val_at = self._val_base + offset * tok
        key_row = key_states[0, :, 0, :].reshape(-1)
        value_row = value_states[0, :, 0, :].reshape(-1)
        if key_row.device != self.pool.frames.device:
            key_row = key_row.to(self.pool.frames.device)
            value_row = value_row.to(self.pool.frames.device)
        self.pool.frames[frame, key_at:key_at + tok].copy_(key_row)
        self.pool.frames[frame, val_at:val_at + tok].copy_(value_row)

    def attend_decode(self, query, scale):
        """Runs Triton over this layer's pages. `query` is [batch, heads, 1, dim]."""
        q = query[:, :, 0, :].contiguous()
        table, context = self.pool.decode_inputs(self.sequence_id, self.length)
        layout = self.pool.layout
        # Leave num_partitions to the wrapper. Forcing 1 made launch 2.6 -> 7 ms;
        # see docs/performance.md week 21 finding 6.
        out = paged_attention_decode_partitioned(
            self.pool.frames,
            table,
            context,
            q,
            layer=self.layer_idx,
            layer_stride=layout.layer_stride(),
            stream_stride=layout.stream_stride(),
            token_stride=layout.token_stride(),
            head_stride=layout.head_stride(),
            tokens_per_block=self.pool.tokens_per_block,
            num_kv_heads=self.pool.num_kv_heads,
            scale=scale,
            scratch=self.pool._scratch,
        )
        # Kernel emits [batch, heads, dim]; eager/sdpa emit [batch, q, heads, dim].
        return out.unsqueeze(1).contiguous()

    def _gather(self):
        """Reassembles the sequence into the contiguous tensors attention wants.

        This is the paging tax on a torch attention path, and it is why a native
        paged kernel is worth having: the blocks are already the right bytes in
        the right order, and this copies all of them anyway.
        """
        tokens_per_block = self.pool.tokens_per_block
        needed = -(-self.length // tokens_per_block)
        index = self.pool.frame_index(self.sequence_id, needed)

        def stream(kind):
            # One indexed read of every block at once. Slicing the blocks
            # individually and concatenating them is the same bytes, but it is
            # a few thousand tensor ops per step at this context length, and at
            # that point the launch overhead costs more than the copy does.
            base = self.pool.stream_base(self.layer_idx, kind)
            rows = self.pool.frames[index, base:base + self.pool.block_span]
            return rows.reshape(-1, self.pool.num_kv_heads, self.pool.head_dim)[:self.length]

        keys, values = stream(qp.KVStream.Key), stream(qp.KVStream.Value)
        self.gathered_elements += keys.numel() + values.numel()

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


def paged_attention_forward(module, query, key, value, attention_mask, scaling,
                            dropout=0.0, **kwargs):
    """Decode through Triton; prefill through the dense implementation.

    Bound onto each text attention module by `install_paged_attention`. Vision
    layers never get a `_paged_layer` and fall through, so this can be the
    process-wide `qwenvl_paged` implementation without touching the ViT.
    """
    layer = getattr(module, "_paged_layer", None)
    if layer is not None and layer.pool.use_kernel and query.shape[-2] == 1:
        return layer.attend_decode(query, scale=scaling), None
    dense = ALL_ATTENTION_FUNCTIONS.get("sdpa", eager_attention_forward)
    return dense(module, query, key, value, attention_mask, scaling=scaling,
                 dropout=dropout, **kwargs)


def bind_paged_layers(model, cache):
    """Points each text attention module at this cache's `PagedLayer`.

    Must be called again after `fork_paged_cache`. The attention implementation
    is process-wide; the layer object is per sequence. Leaving the parent bound
    makes decode write the child and Triton read the parent.
    """
    for decoder_layer, paged_layer in zip(model.model.language_model.layers, cache.layers):
        decoder_layer.self_attn._paged_layer = paged_layer
        decoder_layer.self_attn.config._attn_implementation = "qwenvl_paged"


def install_paged_attention(model, cache):
    """Registers the decode kernel and binds this cache's layers."""
    ALL_ATTENTION_FUNCTIONS.register("qwenvl_paged", paged_attention_forward)
    bind_paged_layers(model, cache)


def build_paged_cache(text_config, max_blocks, tokens_per_block=DEFAULT_TOKENS_PER_BLOCK,
                      scatter=False, dtype=torch.float32, device="cpu", use_kernel=False):
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
        device=device,
        use_kernel=use_kernel,
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
        if source.is_initialized:
            layer._token_elems = source._token_elems
            layer._key_base = source._key_base
            layer._val_base = source._val_base
        layers.append(layer)
    return Cache(layers=layers)
