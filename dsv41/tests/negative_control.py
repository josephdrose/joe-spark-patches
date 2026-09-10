# SPDX-License-Identifier: Apache-2.0
"""Negative controls for the disk table and its prefetch.

A parity assertion that cannot fail proves nothing. The first three runs are
the same comparison unchanged, with one byte of the row file flipped, and with
the ue8m0 decode swapped for torch's float8_e8m0fnu cast; only the first may
pass. The last two do the same for the prefetch guard: prefetch one id set,
look up another, once with the real id key and once with the key weakened to a
shape comparison. The weakened one must serve the wrong rows.
"""
import importlib.util
import os
import pathlib
import sys
import tempfile

import conftest  # noqa: F401  registers the PR modules under vllm.*
import torch

from vllm.models.deepseek_v4_1.common import engram_disk  # noqa: E402

HERE = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "test_engram", HERE / "tests/kernels/test_engram.py"
)
test_engram = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(test_engram)
_attach_disk_table = test_engram._attach_disk_table
_make_embedding = test_engram._make_embedding


def run(tmpdir, mutate_file=False, mutate_decode=False):
    layer = _make_embedding(False)
    layer.weight_scale_inv[::7] = 0
    cols, rows = 24, layer.part_num_embeddings
    ids = torch.randint(0, rows, (256, cols), dtype=torch.int32, device="cuda")
    shape = (256, cols, layer.dim)
    resident = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    layer.lookup(ids, resident)
    _attach_disk_table(layer, tmpdir)
    if mutate_file:
        with open(os.path.join(tmpdir, "engram.bin"), "r+b") as f:
            f.seek(37 * 264 + 5)
            byte = f.read(1)
            f.seek(37 * 264 + 5)
            f.write(bytes([byte[0] ^ 0x08]))
        layer._table.close()
        layer._table = engram_disk.DiskEngramTable(
            os.path.join(tmpdir, "engram.bin"),
            layer.dim,
            row_start=layer.vocab_start_idx,
            row_count=layer.weight.shape[0],
            block_size=layer.block_size,
        )
    if mutate_decode:
        original = engram_disk.DiskEngramTable.gather

        def cast_decode(self, indices):
            # What the disk path would do if it trusted torch's e8m0 cast:
            # byte 0 becomes 2**-127 where the kernel gives 0.0.
            #
            # Reads the whole row file rather than going through the table's
            # reader. The control is about the decode, so it must not depend on
            # the reader's internals; an earlier version called a private
            # helper that later stopped existing, and the check died with a
            # traceback the runner did not fail on.
            flat = indices.reshape(-1).to(torch.int64)
            local = flat - self.row_start
            owned = (local >= 0) & (local < self.row_count)
            with open(os.path.join(tmpdir, "engram.bin"), "rb") as fh:
                blob = bytearray(fh.read())
            table = torch.frombuffer(blob, dtype=torch.uint8).view(-1, self.stride)
            buf = table[local.masked_fill(~owned, 0)]
            values = buf[:, : self.dim].view(torch.float8_e4m3fn).float()
            scale = buf[:, self.dim :].view(torch.float8_e8m0fnu).float()
            out = values.unflatten(-1, (self.num_scales, self.block_size))
            out = (out * scale.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
            return out.masked_fill(~owned.unsqueeze(-1), 0).view(*indices.shape, self.dim)

        engram_disk.DiskEngramTable.gather = cast_decode
    try:
        disk = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        layer.lookup(ids, disk)
    finally:
        if mutate_decode:
            engram_disk.DiskEngramTable.gather = original
    return resident, disk


def prefetch_guard(tmpdir, shape_only=False):
    """Prefetch ids `a`, then look up ids `b`. The rows must be `b`'s.

    With `shape_only` the id comparison is replaced by a shape comparison,
    which is what a weaker key would do. That must hand back `a`'s rows, or
    the id key is not what makes the real check pass.
    """
    layer = _attach_disk_table(_make_embedding(False), tmpdir)
    cols, rows = 24, layer.part_num_embeddings
    a = torch.randint(0, rows, (8, cols), dtype=torch.int32, device="cuda")
    b = torch.randint(0, rows, (8, cols), dtype=torch.int32, device="cuda")
    shape = (8, cols, layer.dim)
    want_a = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    want_b = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    layer.lookup(a, want_a)
    layer.lookup(b, want_b)

    original = type(layer)._take_prefetched
    if shape_only:

        def shape_match(self, cols):
            pending, self._pending = self._pending, None
            if pending is None:
                return None
            ids, handle = pending
            got = self._table.wait(handle)
            return got if ids.shape == cols.shape else None

        type(layer)._take_prefetched = shape_match
    try:
        got = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        layer.prefetch(a)
        layer.lookup(b, got)
    finally:
        type(layer)._take_prefetched = original
    return got, want_a, want_b


tmp = tempfile.mkdtemp(dir=sys.argv[1] if len(sys.argv) > 1 else None)
fails = 0

resident, disk = run(tempfile.mkdtemp(dir=tmp))
nonzero = int(torch.count_nonzero(resident))
print(f"{'PASS' if torch.equal(disk, resident) else 'FAIL'} unmutated: equal, "
      f"{nonzero}/{resident.numel()} values nonzero")
fails += not torch.equal(disk, resident) or nonzero == 0

resident, disk = run(tempfile.mkdtemp(dir=tmp), mutate_file=True)
differ = int((disk != resident).sum())
print(f"{'PASS' if differ else 'FAIL'} one flipped byte in the row file is caught "
      f"({differ} values differ)")
fails += not differ

resident, disk = run(tempfile.mkdtemp(dir=tmp), mutate_decode=True)
differ = int((disk != resident).sum())
print(f"{'PASS' if differ else 'FAIL'} float8_e8m0fnu cast decode is caught "
      f"({differ} values differ)")
fails += not differ

got, want_a, want_b = prefetch_guard(tempfile.mkdtemp(dir=tmp))
distinct = not torch.equal(want_a, want_b)
ok = distinct and torch.equal(got, want_b)
print(f"{'PASS' if ok else 'FAIL'} a prefetch for other ids is not served "
      f"(the two id sets differ: {distinct})")
fails += not ok

got, want_a, want_b = prefetch_guard(tempfile.mkdtemp(dir=tmp), shape_only=True)
ok = torch.equal(got, want_a)
print(f"{'PASS' if ok else 'FAIL'} keying the prefetch on shape alone serves "
      "the wrong rows")
fails += not ok

print("ALL PASS" if not fails else f"{fails} FAILURE(S)")
raise SystemExit(1 if fails else 0)
