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
    # Query head is the fast axis so that the group_size query heads sharing a KV
    # head are consecutive in launch order and stay co-resident. They read the
    # same K/V, so the later one hits L2 instead of DRAM. With sequence on the
    # fast axis they sit num_seqs apart, and past ~512 that exceeds what fits on
    # the device at once: measured L2 hit rate falls 50% -> 11% at batch 512.
    q_head_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
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
    grid = (num_query_heads, num_seqs)
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


@triton.jit
def _paged_attention_partial_kernel(
    kv_pool_ptr,
    block_table_ptr,
    context_lens_ptr,
    query_ptr,
    partial_out_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    num_query_heads,
    max_blocks_per_seq,
    num_partitions,
    partition_blocks,
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
    """One partition of one (sequence, query head), left unnormalized.

    Identical to the single-program kernel except for where the block loop
    starts and stops, and that the softmax denominator is handed to the reduce
    kernel rather than divided out here. A partition holding no tokens exits
    with running_max -inf and running_sum 0, which the reduce weights to zero.
    """
    q_head_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    partition_idx = tl.program_id(2)
    kv_head_idx = q_head_idx // group_size

    context_len = tl.load(context_lens_ptr + seq_idx)
    num_logical_blocks = (context_len + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK
    first_block = partition_idx * partition_blocks

    dim_offsets = tl.arange(0, HEAD_DIM)
    query = tl.load(query_ptr + (seq_idx * num_query_heads + q_head_idx) * HEAD_DIM + dim_offsets)

    running_max = float("-inf")
    running_sum = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    token_offsets = tl.arange(0, TOKENS_PER_BLOCK)
    kv_head_base = layer * layer_stride + kv_head_idx * head_stride

    for offset in range(0, partition_blocks):
        logical_block = first_block + offset
        block_active = logical_block < num_logical_blocks
        physical_id = tl.load(
            block_table_ptr + seq_idx * max_blocks_per_seq + logical_block,
            mask=block_active,
            other=0,
        )

        tokens_here = context_len - logical_block * TOKENS_PER_BLOCK
        tokens_here = tl.minimum(tl.maximum(tokens_here, 0), TOKENS_PER_BLOCK)
        token_mask = token_offsets < tokens_here

        block_base = physical_id.to(tl.int64) * elements_per_block + kv_head_base
        addr = block_base + token_offsets[:, None] * token_stride + dim_offsets[None, :]

        keys = tl.load(kv_pool_ptr + addr, mask=token_mask[:, None], other=0.0)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(token_mask, scores, float("-inf"))

        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        alpha = tl.where(running_max == float("-inf"), 0.0, tl.exp(running_max - new_max))

        weights = tl.where(token_mask, tl.exp(scores - new_max), 0.0)
        values = tl.load(kv_pool_ptr + addr + stream_stride, mask=token_mask[:, None], other=0.0)

        acc = acc * alpha + tl.sum(weights[:, None] * values, axis=0)
        running_sum = running_sum * alpha + tl.sum(weights, axis=0)
        # An all-masked block leaves new_max at -inf, which is the empty-partition
        # signal the reduce expects, so it is carried rather than clamped.
        running_max = new_max

    slot = (seq_idx * num_query_heads + q_head_idx) * num_partitions + partition_idx
    tl.store(partial_out_ptr + slot * HEAD_DIM + dim_offsets, acc)
    tl.store(partial_max_ptr + slot, running_max)
    tl.store(partial_sum_ptr + slot, running_sum)


@triton.jit
def _paged_attention_reduce_kernel(
    partial_out_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    out_ptr,
    num_query_heads,
    num_partitions,
    HEAD_DIM: tl.constexpr,
    MAX_PARTITIONS: tl.constexpr,
):
    """Merges one (sequence, query head)'s partitions into the final vector.

    Standard online-softmax merge: rescale each partition by exp(m_i - max m),
    then divide the summed numerators by the summed denominators.
    """
    q_head_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    row = seq_idx * num_query_heads + q_head_idx

    partition_offsets = tl.arange(0, MAX_PARTITIONS)
    partition_mask = partition_offsets < num_partitions
    slots = row * num_partitions + partition_offsets

    partial_max = tl.load(partial_max_ptr + slots, mask=partition_mask, other=float("-inf"))
    partial_sum = tl.load(partial_sum_ptr + slots, mask=partition_mask, other=0.0)

    global_max = tl.max(partial_max, axis=0)
    # Empty partitions carry -inf; -inf - global_max is -inf under IEEE but nan
    # if every partition were empty, which context_len >= 1 rules out.
    rescale = tl.where(partial_max == float("-inf"), 0.0, tl.exp(partial_max - global_max))

    dim_offsets = tl.arange(0, HEAD_DIM)
    partial_out = tl.load(
        partial_out_ptr + slots[:, None] * HEAD_DIM + dim_offsets[None, :],
        mask=partition_mask[:, None],
        other=0.0,
    )

    out = tl.sum(partial_out * rescale[:, None], axis=0) / tl.sum(partial_sum * rescale, axis=0)
    tl.store(out_ptr + row * HEAD_DIM + dim_offsets, out)


# A [MAX_PARTITIONS, HEAD_DIM] fp32 tile lives in the reduce kernel's registers,
# so the cap is a register-pressure limit, not an algorithmic one. 32 is already
# more than enough to fill this device from a single sequence: 16 query heads x
# 32 partitions is 512 programs against 58 SMs.
MAX_PARTITIONS = 32

# Programs needed to saturate DRAM, read off the batch sweep in
# docs/performance.md: batch 32 (512 programs) reaches 95% of peak, batch 8
# (128 programs) only 59%.
TARGET_PROGRAMS = 512


def choose_num_partitions(num_seqs: int, num_query_heads: int, max_logical_blocks: int) -> int:
    """Splits the context just far enough to fill the device, and no further.

    Every extra partition costs a slice of the reduce kernel's serial work and
    another round trip through the scratch buffers, so this stops as soon as the
    grid is large enough. At batch 32 and above it returns 1, which skips the
    partitioned path entirely -- that regime is already at 95% of peak DRAM and
    has nothing to gain from more parallelism.
    """
    wanted = -(-TARGET_PROGRAMS // (num_seqs * num_query_heads))
    return max(1, min(wanted, MAX_PARTITIONS, max_logical_blocks))


def paged_attention_decode_partitioned(
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
    num_partitions: int | None = None,
) -> torch.Tensor:
    """Decode attention with the context split across programs.

    Same inputs and same output as paged_attention_decode_triton. That kernel
    gives one program the whole context, so a single request launches only
    num_query_heads programs and leaves most of the device idle. Splitting the
    context multiplies the grid by num_partitions at the cost of a second pass
    to merge the partial softmaxes.

    Passing num_partitions=1 runs the single-partition path directly and skips
    the merge, which makes it the baseline plus one predicate.
    """
    num_seqs, num_query_heads, head_dim = query.shape
    assert num_query_heads % num_kv_heads == 0, "query heads must group evenly onto kv heads"
    group_size = num_query_heads // num_kv_heads
    max_blocks_per_seq = block_table.shape[1]
    elements_per_block = kv_pool.shape[1]

    # block_table.shape[1] is sized to the longest sequence in the batch, so it
    # bounds the logical blocks any program can walk. Deriving the split from
    # context_lens.max() instead would read a device tensor on the host and
    # synchronize on every call, which measured ~50 us -- more than the whole
    # kernel at small batch.
    if num_partitions is None:
        num_partitions = choose_num_partitions(num_seqs, num_query_heads, max_blocks_per_seq)
    assert 1 <= num_partitions <= MAX_PARTITIONS, f"num_partitions must be 1..{MAX_PARTITIONS}"

    if num_partitions == 1:
        return paged_attention_decode_triton(
            kv_pool,
            block_table,
            context_lens,
            query,
            layer=layer,
            layer_stride=layer_stride,
            stream_stride=stream_stride,
            token_stride=token_stride,
            head_stride=head_stride,
            tokens_per_block=tokens_per_block,
            num_kv_heads=num_kv_heads,
            scale=scale,
        )

    # Re-derived from the rounded-up partition size so a split that does not
    # divide evenly drops the trailing all-empty partitions instead of paying
    # for programs that exit immediately.
    partition_blocks = -(-max_blocks_per_seq // num_partitions)
    num_partitions = -(-max_blocks_per_seq // partition_blocks)

    partial_out = torch.empty(
        num_seqs, num_query_heads, num_partitions, head_dim, device=query.device, dtype=torch.float32
    )
    partial_max = torch.empty(num_seqs, num_query_heads, num_partitions, device=query.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)

    _paged_attention_partial_kernel[(num_query_heads, num_seqs, num_partitions)](
        kv_pool,
        block_table,
        context_lens,
        query,
        partial_out,
        partial_max,
        partial_sum,
        num_query_heads,
        max_blocks_per_seq,
        num_partitions,
        partition_blocks,
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

    out = torch.empty_like(query)
    _paged_attention_reduce_kernel[(num_query_heads, num_seqs)](
        partial_out,
        partial_max,
        partial_sum,
        out,
        num_query_heads,
        num_partitions,
        HEAD_DIM=head_dim,
        MAX_PARTITIONS=triton.next_power_of_2(num_partitions),
    )
    return out
