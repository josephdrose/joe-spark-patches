# SPDX-License-Identifier: Apache-2.0
"""Checks on `engram_disk` that the vLLM test file has no place for.

`tests/kernels/test_engram.py` asserts the disk path equals the resident path.
This asserts the things underneath that: that O_DIRECT is really engaged, that
its buffered fallback returns the same bytes, that the reader threads do not
race, that the async API agrees with `gather`, and it measures latency against
the read path this module replaced. Run inside a vLLM image:

    python3 engram_disk_harness.py [scratch_dir] [big_file]

`scratch_dir` must be a real filesystem. /dev/shm is used deliberately for the
fallback case, since tmpfs rejects O_DIRECT at open.

`big_file` is optional and turns on the before/after latency table. It must be
several GiB so the page cache cannot answer the reads; any large safetensors
file works, the bytes are only read for timing. Without it the timing section
runs against the small table this harness builds, which the cache does hold.
"""

from __future__ import annotations

import importlib.util
import inspect
import mmap
import os
import pathlib
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

HERE = pathlib.Path(__file__).parent
# The patch tree puts the module under its real package path; run from the
# repo directory it sits next to this file instead.
_CAND = [
    HERE / "vllm/models/deepseek_v4_1/common/engram_disk.py",
    HERE / "engram_disk.py",
]
_PATH = next(p for p in _CAND if p.exists())
_spec = importlib.util.spec_from_file_location("engram_disk", _PATH)
engram_disk = importlib.util.module_from_spec(_spec)
sys.modules["engram_disk"] = engram_disk
_spec.loader.exec_module(engram_disk)

DiskEngramTable = engram_disk.DiskEngramTable
build_row_file = engram_disk.build_row_file
# The two copies name the O_DIRECT switch differently.
_DIRECT_KW = ("direct_io" if "direct_io" in
              inspect.signature(DiskEngramTable.__init__).parameters else "use_odirect")
_ASYNC = hasattr(DiskEngramTable, "submit")

DIM = 256
BLOCK = 32
ROWS = 40_000
COLS = 12
STRIDE = DIM + DIM // BLOCK
PAGE = 4096
failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"{'PASS' if ok else 'FAIL'} {name}{' ' + detail if detail else ''}")
    failures += not ok


def make_table(rows: int, dim: int):
    """fp8 values with no NaN, and scales that include the edge bytes."""
    g = torch.Generator().manual_seed(3)
    w = torch.randint(0, 256, (rows, dim), dtype=torch.uint8, generator=g)
    # e4m3 NaN is 0x7f / 0xff, and NaN != NaN would make every check vacuous.
    w = torch.where((w & 0x7F) == 0x7F, torch.full_like(w, 0x3C), w)
    s = torch.randint(100, 150, (rows, dim // BLOCK), dtype=torch.uint8, generator=g)
    s[::11] = 0
    return w.view(torch.float8_e4m3fn), s.view(torch.float8_e8m0fnu)


def reference(weight, scale, indices, row_start, row_count):
    """In-memory equivalent of `DiskEngramTable.gather`."""
    flat = indices.reshape(-1).to(torch.int64)
    local = flat - row_start
    owned = (local >= 0) & (local < row_count)
    rows = weight[local.masked_fill(~owned, 0)].float()
    exps = scale[local.masked_fill(~owned, 0)].view(torch.uint8)
    exps = (exps.to(torch.int32) << 23).view(torch.float32)
    values = (rows.unflatten(-1, (-1, BLOCK)) * exps.unsqueeze(-1)).flatten(-2)
    out = values.to(torch.bfloat16).masked_fill(~owned.unsqueeze(-1), 0)
    return out.view(*indices.shape, weight.shape[1])


class LegacyReader:
    """The read path this module replaced: one `preadv` of the two pages
    around each row, one pool task per row, one bytes copy per row.

    The dequant is the current one, so a before/after difference is the read
    path and nothing else. Kept only so the numbers come from the same box,
    the same file and the same run. Not used by anything else.
    """

    def __init__(self, path: str, threads: int = 12) -> None:
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self.pool = ThreadPoolExecutor(threads)
        self.local = threading.local()

    def _read_row(self, row: int) -> bytes:
        off = row * STRIDE
        base = (off // PAGE) * PAGE
        buf = getattr(self.local, "buf", None)
        if buf is None:
            buf = self.local.buf = memoryview(mmap.mmap(-1, 2 * PAGE))
        os.preadv(self.fd, [buf], base)
        lo = off - base
        return bytes(buf[lo:lo + STRIDE])

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.reshape(-1).to(torch.int64)
        rows = flat.tolist()
        n = len(rows)
        raw = bytearray(n * STRIDE)

        def one(k: int) -> None:
            raw[k * STRIDE:(k + 1) * STRIDE] = self._read_row(rows[k])

        list(self.pool.map(one, range(n)))
        buf = torch.frombuffer(raw, dtype=torch.uint8).view(n, STRIDE)
        values = buf[:, :DIM].view(torch.float8_e4m3fn).float()
        exps = (buf[:, DIM:].to(torch.int32) << 23).view(torch.float32)
        out = (values.unflatten(-1, (-1, BLOCK)) * exps.unsqueeze(-1)).flatten(-2)
        return out.to(torch.bfloat16).view(*indices.shape, DIM)

    def close(self) -> None:
        self.pool.shutdown(wait=False)
        os.close(self.fd)


def latency_table(path: str, sizes=(12, 24, 48, 96, 512), reps: int = 120) -> None:
    """Median gather latency, old read path vs new, same file, same run.

    Measurements are interleaved rather than run in blocks: this box also
    serves a model, so a block of one variant can land in a quiet minute and
    the other in a busy one.
    """
    n_file = os.path.getsize(path) // STRIDE
    print(f"\nlatency on {path} "
          f"({os.path.getsize(path) / 2**30:.1f} GiB, {n_file:,} rows), "
          f"median of {reps}, ms")
    try:
        old12 = LegacyReader(path, threads=12)
    except OSError as err:
        print(f"     skipped: O_DIRECT refused on {path} ({err})")
        return
    old64 = LegacyReader(path, threads=64)
    new = DiskEngramTable(path, DIM, 0, n_file, BLOCK)
    ways = [("old thr=12", old12.gather), ("old thr=64", old64.gather),
            ("new", new.gather)]
    if _ASYNC:
        ways.append(("new submit/wait", lambda t: new.wait(new.submit(t))))

    g = torch.Generator().manual_seed(5)
    probe = torch.randint(0, n_file, (37,), generator=g)
    # Compared as bit patterns: an arbitrary file read as fp8 contains NaN,
    # and NaN never compares equal, which would make this check vacuous.
    check("old and new read paths return the same rows",
          torch.equal(old12.gather(probe).view(torch.int16),
                      new.gather(probe).view(torch.int16)))

    lat = {name: {n: [] for n in sizes} for name, _ in ways}
    for n in sizes:
        rows = [torch.randint(0, n_file, (n,), generator=g) for _ in range(reps)]
        for name, fn in ways:
            for _ in range(3):
                fn(rows[0])
        for i in range(reps):
            for name, fn in ways:
                t0 = time.perf_counter()
                fn(rows[i])
                lat[name][n].append((time.perf_counter() - t0) * 1e3)
    old12.close()
    old64.close()
    new.close()

    head = "  ".join(f"{n:>7}" for n in sizes)
    print(f"{'':<18}{head}")
    for name, _ in ways:
        print(f"{name:<18}" + "  ".join(
            f"{statistics.median(lat[name][n]):7.2f}" for n in sizes))
    print(f"{'speedup vs 12':<18}" + "  ".join(
        f"{statistics.median(lat['old thr=12'][n]) / statistics.median(lat['new'][n]):6.2f}x"
        for n in sizes))
    print(f"{'speedup vs 64':<18}" + "  ".join(
        f"{statistics.median(lat['old thr=64'][n]) / statistics.median(lat['new'][n]):6.2f}x"
        for n in sizes))


def main(scratch: str, big: str | None) -> int:
    weight, scale = make_table(ROWS, DIM)
    path = os.path.join(scratch, "engram_rows.bin")
    stride = build_row_file(path, weight, scale, BLOCK)
    check("row stride is dim + dim/block", stride == 264, f"({stride})")
    check(
        "file is rows x stride",
        os.path.getsize(path) == ROWS * stride,
        f"({os.path.getsize(path)} B)",
    )

    g = torch.Generator().manual_seed(11)
    idx = torch.randint(-50, ROWS + 50, (64, COLS), generator=g)
    want = reference(weight, scale, idx, 0, ROWS)

    table = DiskEngramTable(path, DIM, 0, ROWS, BLOCK)
    check("O_DIRECT engaged on this filesystem", table._direct)
    got = table.gather(idx)
    check("O_DIRECT gather matches reference", torch.equal(got, want))
    if _ASYNC:
        check("submit/wait matches gather",
              torch.equal(table.wait(table.submit(idx)), want))
        a, b = table.submit(idx[:32]), table.submit(idx[32:])
        check(
            "two handles outstanding at once",
            torch.equal(torch.cat([table.wait(a), table.wait(b)]), want),
        )
    table.close()

    table = DiskEngramTable(path, DIM, 0, ROWS, BLOCK, **{_DIRECT_KW: False})
    check("buffered gather matches O_DIRECT", torch.equal(table.gather(idx), want))
    table.close()

    # Single reader vs 32: the aligned buffer is per worker thread, and a
    # per-task buffer raced here silently, wrong rows and no error.
    table = DiskEngramTable(path, DIM, 0, ROWS, BLOCK, threads=1)
    serial = table.gather(idx)
    table.close()
    table = DiskEngramTable(path, DIM, 0, ROWS, BLOCK, threads=32)
    wide = torch.randint(0, ROWS, (512, COLS), generator=g)
    parallel = table.gather(wide)
    check("single reader matches reference", torch.equal(serial, want))
    check(
        "32 readers match reference on 6144 rows",
        torch.equal(parallel, reference(weight, scale, wide, 0, ROWS)),
    )
    # Same thing from many caller threads, which is what the async API invites.
    out: list[torch.Tensor | None] = [None] * 16
    def one(k: int) -> None:
        out[k] = table.gather(idx)
    ts = [threading.Thread(target=one, args=(k,)) for k in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    check(
        "16 caller threads gathering at once agree",
        all(o is not None and torch.equal(o, want) for o in out),
    )
    table.close()

    # Sharding: a rank owning a middle slice must zero everything else.
    start, count = 10_000, 12_000
    table = DiskEngramTable(path, DIM, start, count, BLOCK)
    owned = ((idx >= start) & (idx < start + count)).sum().item()
    check(
        "sharded gather zeroes rows this rank does not own",
        torch.equal(table.gather(idx), reference(weight, scale, idx, start, count)),
        f"({owned}/{idx.numel()} owned)",
    )
    table.close()

    # A filesystem that refuses O_DIRECT must fall back on its own. Forced
    # rather than taken from tmpfs, which accepts O_DIRECT on recent kernels.
    real_open = os.open

    def refuse_direct(path, flags, *args, **kwargs):
        if flags & engram_disk._O_DIRECT:
            raise OSError(22, "Invalid argument")
        return real_open(path, flags, *args, **kwargs)

    os.open = refuse_direct
    try:
        table = DiskEngramTable(path, DIM, 0, ROWS, BLOCK)
    finally:
        os.open = real_open
    check("falls back when O_DIRECT is refused", not table._direct)
    check("fallback gather matches reference", torch.equal(table.gather(idx), want))
    table.close()

    # ue8m0 byte 255 decodes to +inf here, matching the kernel's exponent-field
    # shift; torch's float8_e8m0fnu cast calls it NaN. Not asserted with
    # torch.equal because inf * 0 is NaN and NaN never compares equal.
    edge_scale = scale.view(torch.uint8).clone()
    edge_scale[:8] = 255
    edge_path = os.path.join(scratch, "engram_edge.bin")
    build_row_file(edge_path, weight[:8], edge_scale[:8], BLOCK)
    table = DiskEngramTable(edge_path, DIM, 0, 8, BLOCK)
    edge = table.gather(torch.arange(8).reshape(1, 8))
    table.close()
    print(
        f"     ue8m0 byte 255 decodes to inf/nan: "
        f"{int(edge.isinf().sum())} inf, {int(edge.isnan().sum())} nan of {edge.numel()}"
    )

    # Decode-shaped latency: one token's worth of rows per call.
    table = DiskEngramTable(path, DIM, 0, ROWS, BLOCK)
    one_idx = torch.randint(0, ROWS, (1, COLS), generator=g)
    for _ in range(20):
        table.gather(one_idx)
    lat = []
    for _ in range(200):
        one_idx = torch.randint(0, ROWS, (1, COLS), generator=g)
        t0 = time.perf_counter()
        table.gather(one_idx)
        lat.append(time.perf_counter() - t0)
    table.close()
    lat.sort()
    med, p95 = lat[len(lat) // 2] * 1e3, lat[int(len(lat) * 0.95)] * 1e3
    print(
        f"     decode gather: median {med:.2f} ms, p95 {p95:.2f} ms, "
        f"{1000 / med:.0f} lookups/s"
    )

    latency_table(big or path)

    print("ALL PASS" if not failures else f"{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    os.makedirs(d, exist_ok=True)
    raise SystemExit(main(d, sys.argv[2] if len(sys.argv) > 2 else None))
