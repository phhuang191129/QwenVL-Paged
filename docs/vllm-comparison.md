# QwenVL-Paged vs vLLM's Block Manager

This project re-implements, from scratch and on a CPU-first contract, the
block manager vLLM runs in production. The designs agree on the paging
abstraction and diverge on prefix identity, store layout, and what the
scheduler is allowed to promise. No head-to-head throughput number was
collected: this box has no GPU session open, and the CPU kernel is still
9 GB/s of a 32 GB/s STREAM roof.

Numbers cited here live in [`performance.md`](performance.md).

## Where the designs agree

**A block is 16 tokens.** Kwon et al., vLLM's default `block_size`, and
`BlockShape::tokens_per_block` here. That is small enough that an image-length
prompt is tens of blocks and large enough that the page table is not the
inner-loop cost (week 11: hoisting the walk was 6%).

**The kernel sees a page table, not a pointer vector.** vLLM's CUDA / Triton
kernels take a `[num_seqs, max_blocks_per_seq]` integer map. `CacheView`
exposes the same thing: `BlockTable::entries()` is kept sorted so a device
backend can flatten it. Logical tokens stay contiguous; physical frames
do not have to be.

**Copy-on-write on fork.** Parallel sampling (`n > 1`) shares the prompt's
frames and copies on the first write. vLLM's `RefCounter` / `CopyOnWrite` and
`KVCacheManager::ensure_token_writable` are the same rule: a write to
`ref_count > 1` allocates a new frame, copies, remaps the writer's entry, and
drops the shared ref. The last branch standing inherits the original.

**Chunked prefill plus a batch token budget.** vLLM's
`max_num_batched_tokens` and our `max_batch_tokens` both split a long prompt
and then count decode tokens against the same budget. Week 14 found that
under a token-cost TTFT clock, chunking a 1,292-token image then a 326-token
text does not move p99 (1,620 = 1,620): the clock charges every scheduled
token, and in-progress prefills fill the budget before decode. vLLM's
wall-clock story can differ because a 256-token step is cheaper than a
1,618-token step; this replay does not have that clock.

**Bookkeeping is not storage.** vLLM's `BlockSpaceManager` / `KVCacheCoordinator`
talks in block ids; `CacheEngine` owns the bytes. `KVCacheManager` and
`MemoryAllocator` are that split. Swap-out here returns the frame immediately
and swap-in lands on a different one — the same "id is not an address" rule
vLLM needs once a block can move.

## Where they differ

### Store layout

Classic PagedAttention (the paper, and early vLLM `CacheEngine`) stores
**all layers in one block**: `block_size` tokens × `num_layers` × K/V. That
is this project's default (`layers_per_frame == 0`), and it is a 1.75 MiB
frame on Qwen3-VL-2B.

The decode kernel, in both systems, runs **one layer at a time**. The useful
traffic per call is 64 KiB per 16-token block, then a jump to the next
frame. Week 11 measured that isolated gap as 1,630 µs vs 670 µs packed
(**2.43×**) and called it a stride tax.

`layers_per_frame = 1` is the other vLLM-adjacent layout: one frame holds one
layer × `tokens_per_block` (64 KiB), and `reserve_tokens` fills tables
layer-major so L0's frames sit next to each other. The allocator did not
change. Serving stays on the default so week 9 / 14 pool counts stay valid.

Re-measurement (same fast kernel, ctx 1,280):

| Layout | Fast regime | Slow regime |
| --- | --: | --: |
| Packed 1-layer (5 MiB pool) | 580–620 µs | not seen |
| Layer-major, 28 × 64 KiB | 587–602 µs | 1,400–1,470 µs |
| Default 1.75 MiB frames | 606–700 µs | 850–954 µs |

Layer-major matches packed when the 5 MiB of one layer stays resident. The
slow mode is the original 2.4×, on consecutive frames. The paper's claim
that paging is nearly free is true in the hot regime and false in the cold
one; the gap is residency, not the page walk. Kwon et al. measured a GPU
kernel whose working set is device DRAM. This CPU number should not be
quoted as a rebuttal of that GPU result.

Per-layer frames do change CoW granularity: a decode write copies 64 KiB
instead of 1.75 MiB. That is a real difference from all-layers-in-one-block
vLLM, and it is why the layout exists even though it did not delete the tax.

### Prefix identity

vLLM's automatic prefix cache hashes **each block's token ids** and holds a
tree (later, a block-hash allocator). A hit can be a proper prefix of a
proper prefix; the tail stays private; eviction is per-block.

This project hashes **the published frames' bytes** under a caller-supplied
key (`publish_prefix` / `attach_prefix`). One key, all-or-nothing. Two
different contents on the same key are a collision and `attach` returns 0.
The published frames are pinned until the last user drops.

Week 14, `repeated-prefix`, pool 912: 232 hits, 16,606 blocks saved,
utilization 1.098. Peak went **702 → 830**. Publish pins the partial tail;
the first decode CoW copies it, so a live prefix holds the original and the
copy. vLLM's per-block hash plus "only complete blocks are cached" does not
pay that pin. Publishing only complete blocks here would close the gap; that
is not what shipped.

Token-id hashes also survive a cache that was filled by a different kernel
or dtype. A content hash of our `uint16` fixture bytes would miss a bf16
fill of the same tokens. That is acceptable on a CPU prototype and would be
wrong as a serving default.

### Admission and preemption

vLLM's default scheduler is FCFS with optional priority, preemption by
recompute or swap, and a watermark on free blocks. It does not, by default,
reserve `max_tokens` for the whole decode lifetime at admit time.

Week 14 admits against `prompt + max_decode_tokens` (minus a usable prefix)
and keeps that as a soft reservation. The week-9 image-heavy sweep went from
72 preemptions / 3,441 failed resumes at pool 256 to **0 / 0**. The cost is
packing: +38% steps, peak batch 4 → 3. Size-aware skip (`admission_skip_limit`)
did nothing at the 912-block pool and **−3.5%** steps at 256 — not the
tail-latency win a size-aware vLLM policy is sometimes sold as, because this
mix's p99 is late closed-loop admits, not one large head blocking one small
request.

Preempt / resume / cancel are request-scoped, so a fork's private frames
come back with the request. vLLM's request object is the same grain; the
difference is that this engine refuses to admit a decode budget it cannot
finish.

### Concurrency

vLLM overlaps GPU kernels with the next schedule step. This engine's
documented contract is single-threaded mutation: a frame may not be recycled
while a kernel still reads it, and `CacheView` is invalidated by CoW, swap,
or release. Week 12's thread pool stays closed because one thread is at
9 GB/s of 32. A GPU port has to add a completion event before
`release` / `swap_out` / `preempt`, which is the constraint vLLM already
lives under.

## What this project measured that the vLLM paper does not

1. **An isolated paging-tax number**, then a second measurement that
   reclassified it. Week 11: 2.43× at a 1,280-token image, blamed on stride.
   Layer-major: the same 2.4× appears on consecutive 64 KiB frames when the
   working set is cold, and disappears when it is hot. Packed (a 5 MiB pool
   that *is* the working set) almost never leaves the hot regime.

2. **Decode-aware admission vs recover-after-the-fact.** vLLM can preempt
   mid-decode. Doing that on the week-9 VLM mix without a lifetime
   reservation was 3,441 failed resumes at pool 256. Reserving the decode
   budget costs 38% more steps and removes the hang.

3. **Prefix pin + CoW peak tax.** Sharing saved 16,606 blocks and raised
   peak by 128 frames. A block-hash tree that refuses incomplete tails
   would not show that peak; a content-hash pin of the whole table does.

## What this is not

It is not a claim that this cache is faster than vLLM. It is not a GPU
recipe. It is not an argument for turning `layers_per_frame = 1` on in
replay. It is a map of the same four decisions — block size, layout, prefix
identity, admit-time reservation — with the numbers this repo actually
produced.
