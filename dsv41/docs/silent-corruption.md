# Three silent-corruption bugs

Read this before you trust any disk-backed weight reader, including this one.

Three bugs were found in this work. All three have the same shape:

- Wrong rows returned.
- No exception.
- A wrong-row rate low enough to read as a quantization artifact.

Coherent output is no evidence against any of them. Only bitwise parity against
the checkpoint catches them.

## Bug 1: the raced read buffer

**Symptom.** 169 of 768 rows wrong, about 22%.

**Cause.** Aligned read buffers were indexed by task index. A
`ThreadPoolExecutor` with N threads runs tasks `k` and `k+N` at the same time.
Both wrote the same buffer. The gather returned another row's bytes.

**Fix.** Buffers became `threading.local()`. Each worker thread now owns one.

**Second form of the same bug.** A later batched rewrite keyed worker slices to
thread id. Releasing k permits wakes k arbitrary workers, so a fast worker could
take a second permit and leave slices unread. Those rows returned as **zeros**,
with no error.

**Fix.** Workers claim slices from an `itertools.count` at wake.

**Regression test.** `tests/test_engram_disk.py` runs 12, 24 and 48 rows at 16,
20, 24, 32, 64 and 96 threads, 40 trials each, plus 8 concurrent callers and two
outstanding handles. A negative control against the broken pool failed 38 to 40
of 40 trials at every thread count. The test is proven able to fail.

## Bug 2: the ue8m0 decode

**Symptom.** 118,311 differing values against the reference decode.

**Cause.** The scale byte is the **exponent field of an fp32 number**. The
correct decode is `(byte << 23)` bitcast to `float32`. A
`torch.float8_e8m0fnu` cast gives a different answer at byte 0.

**Fix.** The `(byte << 23)` decode ships. The negative control forces the case.

**Scope on this checkpoint.** An exhaustive count over all **768,022,850 rows**
of both Engram layers found **zero** scale bytes equal to 0. The case never fires
on this checkpoint. The correct decode still ships.

**A verification bug found alongside it.** The ue8m0 negative control had been
dying on an `AttributeError` for a helper an earlier rewrite removed. The remote
script had no `set -e`, so the traceback scrolled past above a block of `PASS`
lines, and the run exited 0.

Two fixes were applied. The control now reads the row file directly. Every step
in `tests/check_engram_disk.sh` echoes `### rc=N`, so a dead step cannot hide
behind earlier passes.

Any "ALL PASS" from that harness before this change covered fewer assertions than
it claimed.

## Bug 3: the missing rank offset

**Found in prior art.** This reader does not carry it.
`tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark` reports it in their boot log.

**Symptom.** Ranks 1, 2 and 3 read rank 0's rows. No error.

**Cause.** The disk reader ignored each rank's row offset.

**Why it matters here.** At TP=4 each rank owns 6 of the 24 hash-head buckets per
Engram layer. A reader that starts at row 0 on every rank returns rank 0's
weights to all four. Three quarters of the table is then never read.

**Status in this reader.** Checked by reading the code. No test covers it. That
is a weaker check than the other two bugs got.

A related sharding trap, found by reading: **vLLM shards Engram by hash-head
bucket. SGLang shards by arithmetic row range.** The two schemes differ by 947 to
1,344 rows per rank seam. A vLLM-shaped table read with SGLang's row range
returns other tokens' rows for about 0.0014% of lookups, with no error. Random
sampling misses that. Test at the seams.

## What catches these

| Check | Catches |
|---|---|
| Bitwise parity against the checkpoint, by raw `pread` at header offsets | All three |
| Negative controls that flip one byte, or swap the decode | Bugs 1 and 2 |
| Seam-boundary tests at each rank's first and last owned row | Bug 3, and the sharding trap |
| Coherent model output | None of them |

`tests/negative_control.py` runs six controls. All six fire.

## A fourth trap, in the verification itself

`safetensors.get_slice` **materializes the whole tensor** under a CUDA default
device. A 64-row parity read allocated the entire `[384006168, 256]` fp8 tensor
on the GPU. That is 94,513 MiB. The box dropped to 2 GiB free. It raised no
error.

The parity check now uses raw `pread` at header offsets, which is also a more
independent check.
