# SPDX-License-Identifier: Apache-2.0
"""Can the Engram forward be CUDA-graph captured with the table on NVMe?

The whole disk design rests on this. A disk read cannot be captured, so the
rows must be staged into a persistent fixed-address GPU buffer BEFORE the
captured region runs. vLLM already has that shape: `prepare_embeddings()`
writes `staged_rows` and `embed()` only reads it.

Three questions, in order of importance:

  1. Does capture succeed at all when `embed` returns a persistent buffer?
  2. Is a replay bitwise equal to the eager path when the buffer is refilled
     in place between replays? A graph that replays stale rows is worse than
     no graph.
  3. Does the CPU gather actually overlap with `graph.replay()`? The eager
     toy forward is Python and serialises against the gather, which is why
     prefetched decode sits at 11.5% while the bound with the rows already
     landed is 3.0%. A graph replay parks the host thread in a C call, so the
     GIL should stop mattering. This measures whether it does.

Run it from inside the fetched `inference/` directory, same as the other
scripts here:

    python3 engram_graph_capture.py --tokenizer /w/tok --cache /w/cache/token_map.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model  # noqa: E402  (from the fetched inference/ tree)
from engram_selftest import (  # noqa: E402
    FP8_BLOCK,
    load_token_map,
    num_embeddings_for,
)
from engram import EngramLayout  # noqa: E402
from engram_disk import DiskEngramTable, build_row_file  # noqa: E402


class _ArgsShim:
    """Only the fields EngramLayout.from_args reads."""

    def __init__(self, layer_ids, ngram, heads, head_dim, vocab):
        self.engram_layer_ids = layer_ids
        self.engram_max_ngram_size = ngram
        self.engram_n_heads = heads
        self.engram_head_dim = head_dim
        self.engram_vocab_size = vocab
        self.engram_num_embeddings = ()


def build_engram(args_obj, layout, dim, hc_mult, dev, gen):
    """One Engram module with finite weights, plus its table on disk."""
    m_args = model.ModelArgs(
        dim=dim,
        hc_mult=hc_mult,
        engram_layer_ids=layout.layer_ids,
        engram_max_ngram_size=args_obj.engram_max_ngram_size,
        engram_n_heads=args_obj.engram_n_heads,
        engram_head_dim=args_obj.engram_head_dim,
        engram_vocab_size=args_obj.engram_vocab_size,
        engram_num_embeddings=tuple(layout.num_embeddings),
    )
    eng = model.Engram(m_args, layout.layer_ids[0], layout)
    # Only the wide dtypes; the fp8 table weight and its e8m0 scales are set
    # below as raw bytes, and uniform_ is not implemented for either.
    for p in eng.parameters():
        if p.dtype in (torch.bfloat16, torch.float16, torch.float32):
            p.data.uniform_(-0.05, 0.05, generator=gen)
    # fp8 table bytes: keep them away from e4m3 NaN, scales at exponent 127 = 1.0
    w = torch.randint(0, 256, tuple(eng.embed.weight.shape), dtype=torch.uint8,
                      device=dev, generator=gen)
    w = torch.where((w & 0x7F) == 0x7F, torch.full_like(w, 0x3C), w)
    eng.embed.weight.data = w.view(torch.float8_e4m3fn)
    eng.embed.scale.data = torch.full(tuple(eng.embed.scale.shape), 127,
                                      dtype=torch.uint8, device=dev).view(
                                          eng.embed.scale.dtype)
    return eng


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/w/tok")
    ap.add_argument("--cache", default="/w/cache/token_map.pt")
    ap.add_argument("--table-dir", default="/w/tables")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--tokens", type=int, default=4, help="tokens per step; 4 = BS1 with MTP-3")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--threads", type=int, default=64)
    a = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(7)
    os.makedirs(a.table_dir, exist_ok=True)

    layer_ids, ngram, heads, head_dim, dim, hc_mult = (1,), 4, 8, 256, 1024, 4
    shim = _ArgsShim(layer_ids, ngram, heads, head_dim, a.vocab)
    load_token_map(a.tokenizer, a.cache, False)
    # from_args copies num_embeddings straight from the config, it does not
    # derive them, so build the layout once to get the primes, read the row
    # counts off it, then rebuild with those filled in.
    shim.engram_num_embeddings = num_embeddings_for(EngramLayout.from_args(shim))
    layout = EngramLayout.from_args(shim)
    eng = build_engram(shim, layout, dim, hc_mult, dev, gen)
    n_cols = (ngram - 1) * heads
    print(f"[setup] engram dim={dim} hc_mult={hc_mult} cols={n_cols} "
          f"rows={eng.embed.num_embeddings:,}")

    path = os.path.join(a.table_dir, "graph_table.bin")
    build_row_file(path, eng.embed.weight.data.cpu(), eng.embed.scale.data.cpu(), FP8_BLOCK)
    table = DiskEngramTable(path, head_dim, row_start=eng.embed.vocab_start_idx,
                            row_count=eng.embed.part_num_embeddings,
                            block_size=FP8_BLOCK, threads=a.threads)
    print(f"[setup] table on disk: {os.path.getsize(path) / 2**20:.1f} MiB")

    T = a.tokens
    x = torch.randn(T, hc_mult, dim, generator=gen)
    hash_ids = torch.randint(0, eng.embed.num_embeddings, (T, n_cols),
                             device=dev, generator=gen)

    # The persistent staging buffer. Its address must never change: the graph
    # bakes in the pointer at capture and reads whatever is there on replay.
    staged = torch.zeros(T, n_cols, head_dim, dtype=torch.bfloat16, device=dev)
    eng.embed.forward = lambda _idx, _s=staged: _s

    def refill(ids: torch.Tensor) -> None:
        """Prefetch equivalent: gather off disk and land the rows in place."""
        staged.copy_(table.gather(ids.cpu()).to(dev))

    fails = 0

    # --- eager reference ---------------------------------------------------
    refill(hash_ids)
    eager = eng(x, hash_ids).clone()
    assert torch.isfinite(eager.float()).all(), "eager engram output is not finite"

    # --- capture -----------------------------------------------------------
    # Warm up on a side stream first; capture on a fresh graph.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            eng(x, hash_ids)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = eng(x, hash_ids)
        print("PASS capture succeeded")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL capture raised: {type(e).__name__}: {e}")
        table.close()
        return 1

    graph.replay()
    torch.cuda.synchronize()
    ok = torch.equal(captured, eager)
    print(f"{'PASS' if ok else 'FAIL'} replay matches eager on the same rows")
    fails += not ok

    # --- does a replay see rows written after capture? ---------------------
    ids2 = torch.randint(0, eng.embed.num_embeddings, (T, n_cols), device=dev, generator=gen)
    refill(ids2)
    eager2 = eng(x, ids2).clone()
    refill(ids2)
    graph.replay()
    torch.cuda.synchronize()
    ok = torch.equal(captured, eager2)
    print(f"{'PASS' if ok else 'FAIL'} replay picks up rows refilled after capture")
    fails += not ok
    if not ok:
        d = (captured.float() - eager2.float()).abs()
        print(f"     max|d|={d.max():.6f} mean|d|={d.mean():.6f}")

    # --- question 3: does the gather overlap a replay? ---------------------
    def bench(with_gather: bool) -> float:
        stop = threading.Event()

        def churn() -> None:
            ids = hash_ids.cpu()
            while not stop.is_set():
                table.gather(ids)

        th = threading.Thread(target=churn, daemon=True) if with_gather else None
        if th:
            th.start()
            time.sleep(0.05)
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        lat = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
        stop.set()
        if th:
            th.join(timeout=2)
        lat.sort()
        return lat[len(lat) // 2] * 1e3

    alone = bench(False)
    loaded = bench(True)
    print(f"[overlap] replay alone {alone:.3f} ms, with a gather churning "
          f"{loaded:.3f} ms, cost {loaded - alone:+.3f} ms")

    # Same question against the eager path, for contrast: this is the case that
    # measured 8.5 ms of GIL serialisation in the toy model.
    def bench_eager(with_gather: bool) -> float:
        stop = threading.Event()

        def churn() -> None:
            ids = hash_ids.cpu()
            while not stop.is_set():
                table.gather(ids)

        th = threading.Thread(target=churn, daemon=True) if with_gather else None
        if th:
            th.start()
            time.sleep(0.05)
        lat = []
        for _ in range(a.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            eng(x, hash_ids)
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
        stop.set()
        if th:
            th.join(timeout=2)
        lat.sort()
        return lat[len(lat) // 2] * 1e3

    e_alone = bench_eager(False)
    e_loaded = bench_eager(True)
    print(f"[overlap] eager alone {e_alone:.3f} ms, with a gather churning "
          f"{e_loaded:.3f} ms, cost {e_loaded - e_alone:+.3f} ms")

    table.close()
    print("ALL PASS" if fails == 0 else f"{fails} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
