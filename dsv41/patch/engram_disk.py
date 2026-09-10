# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-backed Engram table for DeepSeek-V4.1-Flash.

vLLM's `ParallelEngramEmbedding` keeps the n-gram hash table in pinned host
memory and reads it from a Triton kernel over UVA. On GB10 the CPU and GPU
share one 128 GB pool, so that offload frees nothing and the model does not
fit. This module keeps the table on NVMe instead and gathers rows on the CPU.

Three pieces:
  `build_row_file`   converts the checkpoint's two disjoint regions (fp8
                     weights, ue8m0 block scales) into one interleaved row file,
                     so a row is ONE contiguous read instead of two.
  `DiskEngramTable`  gathers rows with O_DIRECT `preadv` and dequantizes to
                     bf16.
  `submit`/`wait`    the same gather split in two, so a caller can start the
                     reads and go do something else.

The dequant arithmetic is ported from sgl-project/sglang#38798
`python/sglang/srt/layers/engram.py:_owned_rows`, whose tests assert it is
bitwise equal to the Triton kernel.

Read amplification is structural. The 8 head-rows of one n-gram come from
`rolling % prime[h] + offset[h]` over 8 different primes, so no static layout
co-locates them, and a 264-byte row (dim 256) is smaller than a sector. Every
row costs at least one device read.

What the read path is shaped by, all measured on spark-1 against a 40.9 GiB
file so the page cache cannot answer the reads:

  - The device is the floor at large gathers. 512 random rows cost about
    5.9 ms of reads, roughly 87k reads/s, which is already past the 58k IOPS
    a standalone 4 KiB random-read benchmark got on this NVMe. There is
    nothing left to win there.
  - What was winnable is the Python around the reads. `ThreadPoolExecutor.map`
    over one task per row costs more than the reads themselves at decode
    sizes. Workers here are woken once per gather and each claims a
    contiguous slice of the row list.
  - Handing each worker a slice by its own thread id instead of having it
    claim one is WRONG, and wrong silently: releasing k permits wakes k
    arbitrary workers, and a fast worker can take a second permit, so slices
    go unread and those rows come back as zeros. See `_ReadPool`.
  - A shared work counter, claimed per row or per small chunk, measured the
    same as the static slice at every row count from 12 to 512. The static
    slice is simpler, so that is what this uses.
  - Sorting the rows and merging adjacent reads into one `preadv` was not
    faster on the 40.9 GiB file or on a 10 MiB one. Random rows almost never
    share a block, so the sort is overhead. Not used.
  - Reading into one big buffer and cutting the rows out with `torch.gather`
    cost about 1 ms more per 512 rows than having each worker copy its own
    264 bytes into the compact destination. Not used.
  - `io_uring` was not measured. The image has no liburing binding and
    CPython does not expose `preadv2`, so there is no stdlib path to it and
    nothing may be installed.

Measured before and after, median of 120, milliseconds, by
`engram_disk_harness.py` on that file:

    rows            12     24     48     96    512
    old           1.25   1.67   2.41   3.93  19.57
    new           1.04   1.31   1.99   2.51   8.94
    speedup       1.20x  1.28x  1.22x  1.57x  2.19x
"""

from __future__ import annotations

import itertools
import mmap
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import torch

FP8_BLOCK_SIZE = 32
SECTOR = 512
PAGE = 4096
_O_DIRECT = getattr(os, "O_DIRECT", 0)


def row_stride(dim: int, block_size: int = FP8_BLOCK_SIZE) -> int:
    """Bytes per interleaved row: `dim` fp8 values then `dim // block` scales."""
    assert dim % block_size == 0, (dim, block_size)
    return dim + dim // block_size


def build_row_file(
    path: str,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
    chunk_rows: int = 1 << 16,
) -> int:
    """Write `weight`/`scale` to `path` as interleaved fixed-stride rows.

    `weight` is [n, dim] float8_e4m3fn, `scale` is [n, dim // block_size]
    float8_e8m0fnu. Returns the row stride in bytes.
    """
    n, dim = weight.shape
    assert scale.shape == (n, dim // block_size), (scale.shape, n, dim)
    stride = row_stride(dim, block_size)
    w = weight.cpu().view(torch.uint8)
    s = scale.cpu().view(torch.uint8)
    with open(path, "wb") as f:
        for lo in range(0, n, chunk_rows):
            hi = min(lo + chunk_rows, n)
            f.write(torch.cat([w[lo:hi], s[lo:hi]], dim=1).contiguous().numpy().tobytes())
    return stride


class DiskEngramTable:
    """One rank's shard of the Engram table, resident on NVMe.

    Rows are addressed globally; `row_start` and `row_count` bound what this
    rank owns. Indices outside that range gather zeros, matching the masked
    behaviour of the sharded GPU path.

    `gather` is `wait(submit(...))`. Both are safe to call from several threads;
    the reads of one gather use the whole worker set, so gathers queue.
    """

    def __init__(
        self,
        path: str,
        dim: int,
        row_start: int = 0,
        row_count: int | None = None,
        block_size: int = FP8_BLOCK_SIZE,
        threads: int = 64,
        use_odirect: bool = True,
    ) -> None:
        self.dim = dim
        self.block_size = block_size
        self.stride = row_stride(dim, block_size)
        self.row_start = row_start
        self.n_scales = dim // block_size
        self.num_scales = self.n_scales  # name used by the vLLM-side caller

        self._fd = -1
        self._direct = False
        if use_odirect and _O_DIRECT:
            try:
                self._fd = os.open(path, os.O_RDONLY | _O_DIRECT)
                self._direct = True
            except OSError:
                # Some filesystems refuse O_DIRECT at open. Clearing the flag
                # matters: the aligned path is correct on a buffered fd, so a
                # stale flag would pay for alignment forever and never say so.
                pass
        if not self._direct:
            self._fd = os.open(path, os.O_RDONLY)
        size = os.fstat(self._fd).st_size
        total_rows = size // self.stride
        self.row_count = total_rows if row_count is None else min(row_count, total_rows)

        # O_DIRECT wants offset, length and buffer address aligned to the
        # device's logical block size. 512 on the Spark NVMe, but 4096 on
        # plenty of hardware, and a wrong guess is EINVAL on every read, so
        # probe once instead of assuming.
        self._align = self._probe_align(path) if self._direct else 1
        # A row can start anywhere in a block, so cover it with one extra block.
        # This is the byte count of every row read, so keep it tight.
        blocks = (self.stride + self._align - 1) // self._align
        self._span = (blocks + 1) * self._align

        self._pool = _ReadPool(threads)
        self._jobs = ThreadPoolExecutor(2, thread_name_prefix="engram-gather")

    def _probe_align(self, path: str) -> int:
        buf = memoryview(mmap.mmap(-1, PAGE))
        for align in (SECTOR, PAGE):
            try:
                os.preadv(self._fd, [buf[:align]], 0)
                return align
            except OSError:
                continue
        # Nothing aligned works, so O_DIRECT is not usable on this file after
        # all. Reopen buffered rather than fail every read with EINVAL.
        os.close(self._fd)
        self._fd = os.open(path, os.O_RDONLY)
        self._direct = False
        return 1

    def close(self) -> None:
        self._jobs.shutdown(wait=False)
        self._pool.close()
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "DiskEngramTable":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def submit(self, indices: torch.Tensor) -> Future:
        """Start the gather. Returns a handle; the reads run in the background."""
        return self._jobs.submit(self._gather, indices)

    def wait(self, handle: Future) -> torch.Tensor:
        """Block until `handle`'s gather is done and return its tensor."""
        return handle.result()

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize the rows named by `indices` [..., k] to bf16 [..., k, dim].

        Out-of-shard indices return zeros.
        """
        return self.wait(self.submit(indices))

    def _gather(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.reshape(-1).to(torch.int64)
        n = flat.numel()
        if n == 0:
            return torch.zeros(*indices.shape, self.dim, dtype=torch.bfloat16)
        local = flat - self.row_start
        owned = (local >= 0) & (local < self.row_count)
        stride = self.stride

        off = local * stride
        if self._direct:
            lo = off & (self._align - 1)
            base = off - lo
            # -1 marks a row this rank does not own; the workers skip it and it
            # stays zero, which masked_fill would overwrite anyway.
            base = torch.where(owned, base, torch.full_like(base, -1))
            starts, offsets = base.tolist(), lo.tolist()
        else:
            starts = torch.where(owned, off, torch.full_like(off, -1)).tolist()
            offsets = None

        dst = bytearray(n * stride)
        self._pool.run(self._fd, starts, offsets, dst, stride, self._span)

        buf = torch.frombuffer(dst, dtype=torch.uint8).view(n, stride)
        values = buf[:, :self.dim].view(torch.float8_e4m3fn).float()
        # ue8m0 is a power of two, so its byte IS the fp32 exponent field.
        # Decoded the way the kernel does it, including byte 0, which this
        # gives 0.0 and torch's float8_e8m0fnu cast gives 2**-127.
        exps = (buf[:, self.dim:].to(torch.int32) << 23).view(torch.float32)
        out = values.unflatten(-1, (self.n_scales, self.block_size))
        out = (out * exps.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
        return out.masked_fill(~owned.unsqueeze(-1), 0).view(*indices.shape, self.dim)


class _ReadPool:
    """Worker threads woken once per gather, each claiming a slice of the rows.

    One gather at a time: a gather already spreads over every worker, so
    overlapping two of them would only add contention. Concurrent callers
    queue on `_busy`.

    A worker claims its slice when it wakes; it does NOT own a fixed slice.
    Releasing k permits wakes k arbitrary workers out of the pool, so slices
    keyed by thread identity leave rows unread whenever k is less than the
    pool size. That failed silently, with zeros in the result and no error.
    """

    def __init__(self, threads: int) -> None:
        self._n = max(threads - 1, 0)  # the calling thread takes a slice too
        self._busy = threading.Lock()
        self._local = threading.local()
        self._job = None
        self._step = 1
        self._claim = itertools.count(1).__next__
        self._left = 0
        self._error: BaseException | None = None
        self._tally = threading.Lock()
        self._go = threading.Semaphore(0)
        self._done = threading.Event()
        self._stop = False
        self._workers = [
            threading.Thread(target=self._loop, daemon=True, name=f"engram-read-{i}")
            for i in range(self._n)
        ]
        for w in self._workers:
            w.start()

    def close(self) -> None:
        self._stop = True
        for _ in self._workers:
            self._go.release()
        for w in self._workers:
            w.join(timeout=1.0)

    def run(self, fd, starts, offsets, dst, stride, span) -> None:
        n = len(starts)
        with self._busy:
            self._job = (fd, starts, offsets, dst, stride, span, n)
            parts = min(self._n + 1, n)
            self._step = (n + parts - 1) // parts
            # itertools.count.__next__ is one C call, so it is atomic under the
            # GIL and costs far less than a lock. It is claimed once per worker,
            # not once per row.
            self._claim = itertools.count(1).__next__
            helpers = parts - 1
            self._left = helpers
            self._error = None
            if helpers:
                self._done.clear()
                for _ in range(helpers):
                    self._go.release()
            self._slice(0)
            if helpers:
                self._done.wait()
            # A worker that died left its rows unread. Without this the gather
            # returns zeros there and says nothing, which is the failure mode
            # this whole file keeps tripping over.
            if self._error is not None:
                raise self._error

    def _loop(self) -> None:
        while True:
            self._go.acquire()
            if self._stop:
                return
            try:
                self._slice(self._claim())
            except BaseException as err:  # noqa: BLE001  re-raised in run()
                self._error = err
            finally:
                with self._tally:
                    self._left -= 1
                    if self._left == 0:
                        self._done.set()

    def _slice(self, part: int) -> None:
        fd, starts, offsets, dst, stride, span, n = self._job
        lo = part * self._step
        if lo >= n:
            return
        hi = min(lo + self._step, n)
        preadv = os.preadv
        if offsets is None:
            # Buffered: read the row straight into its place in the result.
            view = memoryview(dst)
            for k in range(lo, hi):
                start = starts[k]
                if start >= 0:
                    preadv(fd, [view[k * stride:(k + 1) * stride]], start)
            return
        # O_DIRECT: read the aligned block(s) covering the row, then copy the
        # row out. The scratch buffer is per WORKER THREAD, never per task: a
        # pool of N threads runs tasks k and k+N at the same time, so a
        # per-task buffer races and silently returns another row's bytes.
        buf = getattr(self._local, "buf", None)
        if buf is None or len(buf) < span:
            buf = self._local.buf = memoryview(mmap.mmap(-1, span))
        for k in range(lo, hi):
            start = starts[k]
            if start < 0:
                continue
            preadv(fd, [buf], start)
            o = offsets[k]
            dst[k * stride:(k + 1) * stride] = buf[o:o + stride]


def dequantize_reference(
    weight: torch.Tensor, scale: torch.Tensor, indices: torch.Tensor, row_start: int, row_count: int
) -> torch.Tensor:
    """In-memory equivalent of `DiskEngramTable.gather`, for parity tests.

    Ported from sglang `_owned_rows`, with the kernel's ue8m0 decode.
    """
    flat = indices.reshape(-1).to(torch.int64)
    local = flat - row_start
    owned = (local >= 0) & (local < row_count)
    local = local.masked_fill(~owned, 0)
    rows = weight[local].float().unflatten(-1, (-1, FP8_BLOCK_SIZE))
    exps = (scale.view(torch.uint8)[local].to(torch.int32) << 23).view(torch.float32)
    values = (rows * exps.unsqueeze(-1)).flatten(-2)
    out = values.to(torch.bfloat16).masked_fill(~owned.unsqueeze(-1), 0)
    return out.view(*indices.shape, weight.shape[1])


def count_zero_exponents(scale: torch.Tensor) -> int:
    """Rows carrying an e8m0 byte of 0, where sglang and vllm disagree."""
    return int((scale.view(torch.uint8) == 0).sum())
