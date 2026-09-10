# SPDX-License-Identifier: Apache-2.0
"""Build one TP rank's real Engram row file from the staged V4.1 checkpoint.

Every latency number measured so far used a 25 MiB toy table that sits in page
cache. This builds the real thing so the read path can be measured cold.

At TP=4 a rank owns `cdiv(24, 4) = 6` of the 24 hash-head buckets per Engram
layer, so its share is about 23.6 GiB per layer and 47.3 GiB for both. The
checkpoint keeps weights and scales in two disjoint regions; `build_row_file`
interleaves them, so this streams both slices in chunks rather than loading
94 GiB of tensor.

Layer 1's Engram lives entirely in shard 47 and layer 14's in shard 48, so each
layer reads exactly one file.

    python3 build_real_engram_table.py --out /big/engram --layer 1 --rank 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engram_disk import FP8_BLOCK_SIZE, row_stride  # noqa: E402

# The released config. The toy self-test reproduces `engram_num_embeddings`
# exactly from these, so the prime search is confirmed against the real thing.
VOCAB = 16_000_000
LAYERS = (1, 14)
NGRAM, HEADS, HEAD_DIM = 4, 8, 256
SHARD = {1: "model-00047-of-00048.safetensors", 14: "model-00048-of-00048.safetensors"}


def head_sizes(layer: int) -> list[int]:
    """The 24 prime bucket sizes for one layer, in checkpoint (ngram, head) order."""
    from engram import EngramLayout

    class _Args:
        engram_layer_ids = LAYERS
        engram_max_ngram_size = NGRAM
        engram_n_heads = HEADS
        engram_head_dim = HEAD_DIM
        engram_vocab_size = VOCAB
        engram_num_embeddings = ()

    layout = EngramLayout.from_args(_Args())
    per_ngram = layout.primes[LAYERS.index(layer)]
    return [p for order in per_ngram for p in order]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.path.expanduser(
        "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots"))
    ap.add_argument("--out", required=True, help="directory for the row file")
    ap.add_argument("--layer", type=int, default=1, choices=LAYERS)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=1 << 20, help="rows per read")
    a = ap.parse_args()

    snaps = [os.path.join(a.snapshot, d) for d in os.listdir(a.snapshot)]
    snap = max(snaps, key=os.path.getmtime)
    shard = os.path.join(snap, SHARD[a.layer])
    print(f"[src] {shard}")

    sizes = head_sizes(a.layer)
    per_rank_cols = -(-len(sizes) // a.tp)
    lo_head = a.rank * per_rank_cols
    hi_head = min(lo_head + per_rank_cols, len(sizes))
    start = sum(sizes[:lo_head])
    rows = sum(sizes[lo_head:hi_head])
    stride = row_stride(HEAD_DIM, FP8_BLOCK_SIZE)
    print(f"[shard] layer {a.layer} rank {a.rank}/{a.tp}: heads {lo_head}..{hi_head - 1}, "
          f"rows {start:,}..{start + rows:,} ({rows:,} rows, {rows * stride / 2**30:.1f} GiB)")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"engram_L{a.layer}_r{a.rank}of{a.tp}.bin")
    n_scales = HEAD_DIM // FP8_BLOCK_SIZE

    t0 = time.time()
    written = 0
    with safe_open(shard, framework="pt") as f:
        w = f.get_slice(f"layers.{a.layer}.engram.embed.weight")
        s = f.get_slice(f"layers.{a.layer}.engram.embed.scale")
        total = w.get_shape()[0]
        assert w.get_shape()[1] == HEAD_DIM, w.get_shape()
        assert start + rows <= total, (start, rows, total)
        with open(path, "wb", buffering=1 << 22) as out:
            for lo in range(start, start + rows, a.chunk):
                hi = min(lo + a.chunk, start + rows)
                wb = w[lo:hi].view(torch.uint8)
                sb = s[lo:hi].view(torch.uint8)
                assert sb.shape[1] == n_scales, sb.shape
                out.write(torch.cat([wb, sb], dim=1).contiguous().numpy().tobytes())
                written += hi - lo
                if written % (32 * a.chunk) == 0 or hi == start + rows:
                    el = time.time() - t0
                    gib = written * stride / 2**30
                    print(f"  {written:,}/{rows:,} rows  {gib:.1f} GiB  "
                          f"{gib / max(el, 1e-9):.2f} GiB/s", flush=True)

    size = os.path.getsize(path)
    assert size == rows * stride, (size, rows * stride)
    print(f"[done] {path} {size / 2**30:.1f} GiB in {time.time() - t0:.0f}s")
    print(f"[use]  DiskEngramTable(path, {HEAD_DIM}, row_start={start}, row_count={rows})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
