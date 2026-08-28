"""Triton PagedAttention decode kernel, validated against the CPU golden vectors.

This is the device counterpart to include/qwenvl_paged/PagedAttention.h. It
consumes exactly what docs/architecture.md's "Backend Integration Points"
promises an execution backend: KVBlockLayout's four strides and the flattened
[num_seqs, max_blocks_per_seq] block table, nothing else. The fixtures under
fixtures/*.npz exist so that promise, and this kernel's numerics, are checked
before any further GPU work is built on top of them.

One Triton program handles one (sequence, query head) pair and streams over
that sequence's logical blocks with an online (single-pass) softmax, since the
whole context does not necessarily fit in registers at head_dim=128.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_decode_kernel(
    kv_pool_ptr,
    block_table_ptr,
    context_lens_ptr,
    query_ptr,
    out_ptr,
    num_query_heads,
    max_blocks_per_seq,
    layer,
    layer_stride,
    stream_stride,
    token_stride,
    head_stride,
    elements_per_block,
    group_size,
    scale,
    HEAD_DIM: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // group_size

    context_len = tl.load(context_lens_ptr + seq_idx)
    num_logical_blocks = (context_len + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK

    dim_offsets = tl.arange(0, HEAD_DIM)
    query = tl.load(query_ptr + (seq_idx * num_query_heads + q_head_idx) * HEAD_DIM + dim_offsets)

    running_max = float("-inf")
    running_sum = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    token_offsets = tl.arange(0, TOKENS_PER_BLOCK)
    kv_head_base = layer * layer_stride + kv_head_idx * head_stride

    for logical_block in range(0, max_blocks_per_seq):
        block_active = logical_block < num_logical_blocks
        physical_id = tl.load(
            block_table_ptr + seq_idx * max_blocks_per_seq + logical_block,
            mask=block_active,
            other=0,
        )

        tokens_here = context_len - logical_block * TOKENS_PER_BLOCK
        tokens_here = tl.minimum(tl.maximum(tokens_here, 0), TOKENS_PER_BLOCK)
        token_mask = token_offsets < tokens_here

        # int64: at the real 2B block shape a block is 917,504 elements, so
        # physical block 2,341 already overflows int32, and roadmap-phase2.md
        # sizes a 4 GiB pool at exactly 2,340 blocks. Measured free on an L4.
        block_base = physical_id.to(tl.int64) * elements_per_block + kv_head_base
        addr = block_base + token_offsets[:, None] * token_stride + dim_offsets[None, :]

        keys = tl.load(kv_pool_ptr + addr, mask=token_mask[:, None], other=0.0)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(token_mask, scores, float("-inf"))

        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        # exp(-inf - -inf) is nan; on the first iteration running_max is -inf
        # and no token has been seen yet, so the correction factor is exactly 0.
        alpha = tl.where(running_max == float("-inf"), 0.0, tl.exp(running_max - new_max))

        weights = tl.where(token_mask, tl.exp(scores - new_max), 0.0)
        values = tl.load(kv_pool_ptr + addr + stream_stride, mask=token_mask[:, None], other=0.0)

        acc = acc * alpha + tl.sum(weights[:, None] * values, axis=0)
        running_sum = running_sum * alpha + tl.sum(weights, axis=0)
        running_max = new_max

    out = acc / running_sum
    tl.store(out_ptr + (seq_idx * num_query_heads + q_head_idx) * HEAD_DIM + dim_offsets, out)


def paged_attention_decode_triton(
    kv_pool: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    query: torch.Tensor,
    *,
    layer: int,
    layer_stride: int,
    stream_stride: int,
    token_stride: int,
    head_stride: int,
    tokens_per_block: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Runs the Triton decode kernel over a fixture's tensors.

    All tensors are expected on CUDA: kv_pool [num_blocks, elements_per_block]
    fp32, block_table [num_seqs, max_blocks_per_seq] int32, context_lens
    [num_seqs] int32, query [num_seqs, num_query_heads, head_dim] fp32.
    """
    num_seqs, num_query_heads, head_dim = query.shape
    assert num_query_heads % num_kv_heads == 0, "query heads must group evenly onto kv heads"
    group_size = num_query_heads // num_kv_heads
    max_blocks_per_seq = block_table.shape[1]
    elements_per_block = kv_pool.shape[1]

    out = torch.empty_like(query)
    grid = (num_seqs, num_query_heads)
    _paged_attention_decode_kernel[grid](
        kv_pool,
        block_table,
        context_lens,
        query,
        out,
        num_query_heads,
        max_blocks_per_seq,
        layer,
        layer_stride,
        stream_stride,
        token_stride,
        head_stride,
        elements_per_block,
        group_size,
        scale,
        HEAD_DIM=head_dim,
        TOKENS_PER_BLOCK=tokens_per_block,
    )
    return out
