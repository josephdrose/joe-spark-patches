# SPDX-License-Identifier: Apache-2.0
"""Parity, decode, concurrency and throughput checks for `engram_disk`.

Run inside any vLLM image (needs torch, no CUDA):
    python3 test_engram_disk.py [scratch_dir]
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engram_disk import (  # noqa: E402
    FP8_BLOCK_SIZE,
    DiskEngramTable,
    build_row_file,
    count_zero_exponents,
    dequantize_reference,
    row_stride,
)

DIM = 256
ROWS = 40_000
ROWS_PER_TOK = 12
TRIALS = 40


def make_table(rows: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(3)
    w = torch.randint(0, 256, (rows, dim), dtype=torch.uint8, generator=g)
    # Avoid e4m3 NaN (0x7f / 0xff): it makes parity assertions meaningless.
    w = torch.where((w & 0x7F) == 0x7F, torch.full_like(w, 0x3C), w)
    # e8m0 exponents in a sane band, and deliberately include byte 0.
    s = torch.randint(100, 150, (rows, dim // FP8_BLOCK_SIZE), dtype=torch.uint8, generator=g)
    s[0, 0] = 0
    return w.view(torch.float8_e4m3fn), s.view(torch.float8_e8m0fnu)


def check_ue8m0_decode(scratch: str) -> int:
    """The ue8m0 byte is the fp32 exponent field, not a float8_e8m0fnu value.

    Asserted against literal expected numbers, not against another decode, so
    it cannot pass by sharing a bug with the reference. torch's e8m0 cast
    disagrees at byte 0 (2**-127 instead of 0.0) and at byte 255 (NaN instead
    of +inf); both are checked to be genuinely different, so the test is known
    to be able to fail.
    """
    bytes_ = [0, 127, 128, 129, 126, 1, 254, 255]
    want = [0.0, 1.0, 2.0, 4.0, 0.5, 2.0**-126, 2.0**127, float("inf")]
    n_scales = DIM // FP8_BLOCK_SIZE
    assert len(bytes_) == n_scales
    w = torch.full((1, DIM), 0x38, dtype=torch.uint8)  # e4m3 1.0
    s = torch.tensor([bytes_], dtype=torch.uint8)
    path = os.path.join(scratch, "engram_decode.bin")
    build_row_file(path, w.view(torch.float8_e4m3fn), s.view(torch.float8_e8m0fnu))
    with DiskEngramTable(path, DIM, row_count=1) as t:
        got = t.gather(torch.zeros(1, 1, dtype=torch.int64))
    blocks = got.view(n_scales, FP8_BLOCK_SIZE)[:, 0].float().tolist()
    ok = blocks == want
    print(f"{'PASS' if ok else 'FAIL'} ue8m0 decodes as the fp32 exponent field")
    if not ok:
        print(f"     got  {blocks}\n     want {want}")
    cast = s.view(torch.float8_e8m0fnu).float().tolist()[0]
    differs = sum(1 for a, b in zip(cast, want) if not (a == b))
    print(f"{'PASS' if differs else 'FAIL'} float8_e8m0fnu cast is a different "
          f"decode ({differs}/{n_scales} bytes disagree, incl. byte 0 -> {cast[0]})")
    return int(not ok) + int(not differs)


def check_decode_sizes(path: str, weight, scale) -> int:
    """Gathers smaller than the worker pool, which is every decode step.

    `_ReadPool` wakes `min(threads, rows) - 1` workers and each claims a slice
    when it wakes. An earlier version here keyed the slice to the worker's own
    thread id instead. Releasing k permits wakes k ARBITRARY workers out of the
    pool, so whenever fewer workers woke than the pool holds, the slices they
    owned went unread and the gather returned zeros for those rows with no
    error. It only shows up when rows < threads, so the 768-row parity check
    above never saw it.

    Trials use fresh random indices, because one bad draw is not proof and one
    good draw is not either.
    """
    fails = 0
    g = torch.Generator().manual_seed(101)
    for threads in (16, 20, 24, 32, 64, 96):
        bad = {}
        for rows in (12, 24, 48):
            n_bad = 0
            with DiskEngramTable(path, DIM, row_count=ROWS, threads=threads) as t:
                for _ in range(TRIALS):
                    idx = torch.randint(0, ROWS, (1, rows), generator=g)
                    want = dequantize_reference(weight, scale, idx, 0, ROWS)
                    if not torch.equal(t.gather(idx), want):
                        n_bad += 1
            if n_bad:
                bad[rows] = n_bad
        ok = not bad
        print(f"{'PASS' if ok else 'FAIL'} threads={threads:<3} "
              f"decode sizes 12/24/48, {TRIALS} trials each"
              + ("" if ok else f"  bad: {bad}"))
        fails += not ok

    # Same shapes, several callers at once: the pool serializes gathers, and a
    # slice claimed by the wrong job would show up here.
    for threads in (16, 32, 64, 96):
        n_bad = 0
        with DiskEngramTable(path, DIM, row_count=ROWS, threads=threads) as t:
            for _ in range(TRIALS // 2):
                idx = torch.randint(0, ROWS, (1, 48), generator=g)
                want = dequantize_reference(weight, scale, idx, 0, ROWS)
                out: list[torch.Tensor | None] = [None] * 8
                def one(k: int, _i=idx) -> None:
                    out[k] = t.gather(_i)
                ts = [threading.Thread(target=one, args=(k,)) for k in range(8)]
                for th in ts:
                    th.start()
                for th in ts:
                    th.join()
                n_bad += sum(o is None or not torch.equal(o, want) for o in out)
        print(f"{'PASS' if not n_bad else 'FAIL'} threads={threads:<3} "
              f"8 concurrent callers x 48 rows, {TRIALS // 2} trials"
              + ("" if not n_bad else f"  {n_bad} bad gathers"))
        fails += bool(n_bad)

    # Two handles in flight at a decode size, which is what the prefetch does.
    for threads in (32, 96):
        n_bad = 0
        with DiskEngramTable(path, DIM, row_count=ROWS, threads=threads) as t:
            for _ in range(TRIALS):
                idx = torch.randint(0, ROWS, (2, 24), generator=g)
                want = dequantize_reference(weight, scale, idx, 0, ROWS)
                a, b = t.submit(idx[:1]), t.submit(idx[1:])
                got = torch.cat([t.wait(a), t.wait(b)])
                n_bad += not torch.equal(got, want)
        print(f"{'PASS' if not n_bad else 'FAIL'} threads={threads:<3} "
              f"two handles outstanding x 24 rows, {TRIALS} trials"
              + ("" if not n_bad else f"  {n_bad} bad"))
        fails += bool(n_bad)
    return fails


def check_async(path: str, idx: torch.Tensor, want: torch.Tensor) -> int:
    """submit/wait must give what gather gives, including two in flight and
    several caller threads at once."""
    fails = 0
    with DiskEngramTable(path, DIM, row_count=ROWS) as t:
        ok = torch.equal(t.wait(t.submit(idx)), want)
        print(f"{'PASS' if ok else 'FAIL'} submit/wait matches gather")
        fails += not ok

        half = idx.shape[0] // 2
        a, b = t.submit(idx[:half]), t.submit(idx[half:])
        both = torch.cat([t.wait(a), t.wait(b)])
        ok = torch.equal(both, want)
        print(f"{'PASS' if ok else 'FAIL'} two handles outstanding at once")
        fails += not ok

        # Same regression the per-thread scratch buffer exists for: concurrent
        # readers must not see each other's bytes.
        out: list[torch.Tensor | None] = [None] * 16
        def one(k: int) -> None:
            out[k] = t.gather(idx)
        ts = [threading.Thread(target=one, args=(k,)) for k in range(16)]
        for th in ts:
            th.start()
        for th in ts:
            th.join()
        ok = all(o is not None and torch.equal(o, want) for o in out)
        print(f"{'PASS' if ok else 'FAIL'} 16 threads gathering at once agree")
        fails += not ok
    return fails


def main(scratch: str) -> int:
    assert row_stride(DIM) == 264, row_stride(DIM)
    weight, scale = make_table(ROWS, DIM)
    path = os.path.join(scratch, "engram_rows.bin")
    stride = build_row_file(path, weight, scale)
    size = os.path.getsize(path)
    print(f"built {path}: {ROWS} rows x {stride} B = {size / 1024**2:.1f} MiB")
    print(f"zero e8m0 exponents in this table: {count_zero_exponents(scale)}")

    fails = 0

    # 1. Full-shard parity, including out-of-range indices.
    g = torch.Generator().manual_seed(11)
    idx = torch.randint(-50, ROWS + 50, (64, ROWS_PER_TOK), generator=g)
    with DiskEngramTable(path, DIM, row_start=0, row_count=ROWS) as t:
        got = t.gather(idx)
        direct = t._direct
    want = dequantize_reference(weight, scale, idx, 0, ROWS)
    if torch.equal(got, want):
        print(f"PASS full-shard parity: {idx.numel()} rows bitwise equal")
    else:
        bad = (got != want).any(-1).sum().item()
        print(f"FAIL full-shard parity: {bad}/{idx.numel()} rows differ")
        fails += 1
    print(f"     O_DIRECT engaged on this filesystem: {direct}")

    # 2. Sharded parity: rank owns a middle slice, everything else must be zero.
    row_start, row_count = 10_000, 12_000
    with DiskEngramTable(path, DIM, row_start=row_start, row_count=row_count) as t:
        got = t.gather(idx)
    want_sharded = dequantize_reference(weight, scale, idx, row_start, row_count)
    owned = ((idx - row_start >= 0) & (idx - row_start < row_count)).sum().item()
    if torch.equal(got, want_sharded):
        print(f"PASS sharded parity: {owned}/{idx.numel()} rows owned, rest zeroed")
    else:
        print("FAIL sharded parity")
        fails += 1

    # 3. Buffered read must agree with O_DIRECT.
    with DiskEngramTable(path, DIM, row_count=ROWS, use_odirect=False) as t:
        got_buf = t.gather(idx)
    if torch.equal(got_buf, want):
        print("PASS buffered path matches O_DIRECT path")
    else:
        print("FAIL buffered path")
        fails += 1

    # 4. A single reader must agree with a wide one.
    with DiskEngramTable(path, DIM, row_count=ROWS, threads=1) as t:
        if torch.equal(t.gather(idx), want):
            print("PASS single-threaded reader matches")
        else:
            print("FAIL single-threaded reader")
            fails += 1

    # 5. ue8m0 decode.
    fails += check_ue8m0_decode(scratch)

    # 6. The async API.
    fails += check_async(path, idx, want)

    # 7. Decode-sized gathers against pools bigger than the gather.
    fails += check_decode_sizes(path, weight, scale)

    # 8. Decode-shaped latency: one token's 12 rows per call.
    with DiskEngramTable(path, DIM, row_count=ROWS) as t:
        one = torch.randint(0, ROWS, (1, ROWS_PER_TOK), generator=g)
        for _ in range(20):
            t.gather(one)
        lat = []
        for _ in range(200):
            one = torch.randint(0, ROWS, (1, ROWS_PER_TOK), generator=g)
            t0 = time.perf_counter()
            t.gather(one)
            lat.append(time.perf_counter() - t0)
    lat.sort()
    med, p95 = lat[len(lat) // 2] * 1e3, lat[int(len(lat) * 0.95)] * 1e3
    print(f"decode gather: median {med:.2f} ms, p95 {p95:.2f} ms, "
          f"{1000 / med:.0f} tok/s ceiling, {med / 33.3 * 100:.1f}% of a 33.3 ms step")

    print("ALL PASS" if fails == 0 else f"{fails} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    os.makedirs(d, exist_ok=True)
    raise SystemExit(main(d))
