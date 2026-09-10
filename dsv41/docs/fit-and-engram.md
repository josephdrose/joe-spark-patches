# Fix 1: the model does not fit, and CPU offload frees nothing

## The failure

No exception. Arithmetic.

| Quantity | GiB | GiB per rank at TP=4 |
|---|--:|--:|
| Installed memory, 4 boxes | 486.8 | 121.7 |
| V4.1-Flash, all 48 shards | 475.25 | 118.81 |
| Engram table alone | 189.13 | 47.28 |
| Backbone without Engram | 286.12 | 71.53 |

118.81 GiB per rank against 121.7 GiB physical leaves 2.89 GiB. That has to hold
the KV cache, the activations, the NCCL buffers and the CUDA context.

SGLang hit the same wall as a runtime error, before it read a checkpoint byte:

```
[TP0 EP0] Load weight begin. avail mem=108.11 GB
RuntimeError: Rank 0 scheduler died during initialization (exit code: -15)
NVRM: _iovaspaceCreateMappingDataFromMemDesc: failed to allocate 0x21c028 bytes
NVRM: Check failed: Out of memory [NV_ERR_NO_MEMORY] @ mem_desc.c:1359
```

## The cause

vLLM already offloads Engram. `EngramConfig.cpu_offload` defaults to `True`. The
table goes to `torch.empty(..., device="cpu", pin_memory=True)`, and a Triton
kernel reads it over UVA.

**On GB10 the CPU and the GPU share one physical pool.** Pinned host memory is
the same DRAM the GPU allocates from.

Measured on a GB10 while another model served:

```
device      : NVIDIA GB10  cc=12.1  SMs=48
total mem   : 121.7 GiB
is_uva_available()                -> True
GPU free BEFORE pinned alloc      : 11.88 GiB
GPU free AFTER  4 GiB pinned      :  7.79 GiB   delta = +4.09 GiB
cudaHostGetDevicePointer rc=0
device-side read of host buffer   : verified=True
```

A 4 GiB pinned allocation cost 4.09 GiB of GPU-visible memory. The mechanism
works. It frees zero bytes.

`nvidia-smi --query-gpu=memory.total` reports `N/A` on GB10. There is no separate
VRAM counter to report.

## Options rejected

| Option | Per rank | Verdict |
|---|--:|---|
| Everything resident | 118.81 GiB | Does not fit |
| `cpu_offload` pinned host plus UVA | 118.81 GiB | Frees zero bytes, measured above |
| SGLang host table | 118.81 GiB | Same, and it needs one PID namespace. TP=4 spans four machines |
| 2-bit Engram quantization | 83.4 GiB | Fits. Quality unknown. No such quantization exists |
| **Engram on NVMe** | **71.53 GiB** | **This document** |

Four more approaches were rejected after reading or measuring:

- **`cudaHostRegister` on a file mapping.** The driver either rejects the
  non-anonymous VMA, or accepts it by faulting in and pinning all 47.28 GiB.
- **ATS demand paging from NVMe.** A GPU access to a non-resident file page needs
  an SMMU fault escalated through PRI to the CPU. One serialized 357 us major
  fault per row is unusable.
- **Transparent huge pages.** `MADV_HUGEPAGE` and `MADV_COLLAPSE` cover anonymous
  and shmem mappings only. A regular file gets neither.
- **Co-locating an n-gram's head rows.** The 8 head rows come from
  `rolling % prime[h] + offset[h]` over 8 different primes. They land at
  independent positions in 8 disjoint ranges. Any static layout that makes one
  n-gram contiguous scatters every other n-gram.

## The change

Keep the 189.13 GiB Engram table on NVMe. Gather rows on the CPU. Dequantize on
the CPU. Copy the result to the device.

Patch: `patch/engram-disk-table.patch`, 4 files, +980 / -22 against PR commit
`e47aa780`.

### The row file

The checkpoint stores fp8 values and ue8m0 block scales in two disjoint regions.
Reading a row from the checkpoint costs two reads.

`build_row_file` writes one interleaved fixed-stride file instead:

```
row = [ dim bytes fp8_e4m3fn ] [ dim/32 bytes ue8m0 ]
    = 256 + 8
    = 264 bytes
```

One row is one contiguous read.

### The read path

- `O_DIRECT`, so the reads never occupy the page cache. On a unified-memory box
  every cached page is memory the model cannot have.
- Alignment is **probed at open**. The code assumes no value. This NVMe reports
  `logical_block_size` 512, which makes a row read 1024 bytes rather than 8192.
- Reads are dispatched once per gather to a persistent worker set. Each worker
  claims a contiguous slice of the row list from an `itertools.count`.
- One `preadv` carries many iovecs.

Three alternatives were measured and rejected: one task per row, sort-and-merge
into contiguous runs, and one large buffer with `torch.gather` extraction.
Random rows almost never share a 512-byte block, so merging finds nothing.

### The prefetch

A disk read cannot be captured inside a CUDA graph. Without prefetch the Engram
layers are forced eager.

`Engram.prefetch(hash_ids)` issues the read. `_take_prefetched` returns rows only
when the ids match exactly. `_disk_lookup` falls back to a synchronous gather on
a miss.

Under a 600-second load test at TP=4, batch 8, all four ranks reported **zero
misses**:

| Rank | Ready | Late | Missed | Ready % | Wait ms |
|--:|--:|--:|--:|--:|--:|
| 0 | 45,788 | 9,612 | **0** | 82.65 | 0.407 |
| 1 | 44,102 | 11,298 | **0** | 79.61 | 0.435 |
| 2 | 45,595 | 9,805 | **0** | 82.30 | 0.421 |
| 3 | 45,744 | 9,656 | **0** | 82.57 | 0.429 |

A late arrival costs its 0.41 ms wait, about 2% of a step.

That test used the reference model with 5 layers and random weights. It does not
speak to the real backbone.

## Measured cost

Cold, against one rank's real 23.6 GiB shard, on a box with 15.4 GiB of free RAM.
Most reads therefore miss the page cache.

| Rows per gather | Median ms | p95 ms | tok/s ceiling | Share of a 33.3 ms step |
|--:|--:|--:|--:|--:|
| 6 | 1.07 | 1.95 | 935 | 3.2% |
| **12** | **1.13** | 1.70 | 887 | **3.4%** |
| 24 | 1.22 | 1.54 | 821 | 3.7% |
| 48 | 1.89 | 2.15 | 529 | 5.7% |

**12 rows is the real per-rank decode load at TP=4.** That is 6 hash heads times
2 Engram layers.

The 25 MiB synthetic table measured 1.12 ms at 12 rows. The real 23.6 GiB table
measures 1.13 ms. At this queue depth the read is latency-bound, so the page
cache hit rate barely moves it.

## The ue8m0 scale byte

**The scale byte is the exponent field of an fp32 number.** Decode it as
`(byte << 23)` bitcast to `float32`.

A `torch.float8_e8m0fnu` cast gives a different answer at byte 0. A negative
control measured **118,311 differing values** between the two decodes.

An exhaustive count over all **768,022,850 rows** of both Engram layers found
**zero** scale bytes equal to 0. The case never fires on this checkpoint. The
correct decode still ships, and the test still forces the case.

## Correctness evidence

- Logits from the disk path are **bitwise equal** to logits from the resident
  path, on a reference model with the table on NVMe.
- 64 random owned rows per layer per rank were read from
  `model-000{47,48}-of-00048.safetensors` by raw `pread` at header offsets. They
  were bitwise equal to the gather, at the start and at the end of the load test,
  on different rows.
- The CPU hasher matched the module's GPU hash ids both times.

Read [silent-corruption.md](silent-corruption.md) before you trust any of this.
Only bitwise parity catches the failure mode these bugs have.

## Attribution

The dequantization arithmetic is ported from `sgl-project/sglang#38798`,
`python/sglang/srt/layers/engram.py`. Their tests assert it is bitwise equal to
the Triton kernel. Both projects are Apache-2.0.
