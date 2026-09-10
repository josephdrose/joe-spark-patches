# Fix 5: capture the decode step in a CUDA graph

## The failure

No exception. The serve runs. The eager decode step costs about 83 ms. Every
earlier run in this repository used `--enforce-eager`.

Dropping `--enforce-eager` on the disk-backed table raises:

```
RuntimeError: The Engram table is on disk, and a disk lookup cannot be
captured in a CUDA graph because the read is issued by the CPU. Stage the rows
before capture: call prepare_embeddings outside the captured region so it only
reads staged_rows. Engram.forward itself captures fine that way, and a replay
picks up rows refilled in the staging buffer afterwards.
```

`patch/engram-disk-table.patch` raises that. It is a deliberate guard. Capture
would record the device copy at most. Every replay would then serve the rows of
whichever ids were live at capture time.

## The cause

`prepare_embeddings` reads the Engram rows off NVMe inside the model forward.
A disk read is CPU work. Capture records device work only.

The forward therefore holds a host call, inside the region the graph must
record.

## The change

Two parts.

1. Move the read out of the forward. `patch/vlspeed-prestage.py` adds a stager
   that runs in `prepare_inputs`. See [engram-prestage.md](engram-prestage.md).
2. Give every decode batch an exact graph. Run
   `launch/vlspeed-tp4-4node-up.sh` with `EAGER=0`. It sets:

| Flag | Value |
|---|---|
| `cudagraph_mode` | `FULL_AND_PIECEWISE` |
| `cudagraph_capture_sizes` | `[5,6,10,12,15,18,20,24]` at k=5 |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `1` |
| `enable_adaptive_verification` | `false` |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` |
| `MAX_JOBS` | `2` |

The capture sizes come from the DSpark block size. At `DSPARK=k` a decode batch
carries `num_reqs * k` draft tokens, or `num_reqs * (k + 1)` target tokens. The
launcher captures every multiple of `k` and of `k + 1`, up to
`MAXSEQS * (k + 1)`. A padded speculative batch can hang SM120 sparse MLA
(FlashInfer issue #5015).

`enable_adaptive_verification` stays off for the same reason. It forces varlen
decode graphs. Their padded rows are that trigger.

All six settings are taken from the prior art. See the credits in the
[README](../README.md).

## Capture verification

`tests/engram_graph_capture.py` runs against the DeepSeek reference `Engram`
module with its table on NVMe:

```
[setup] engram dim=1024 hc_mult=4 cols=24 rows=100,776
[setup] table on disk: 25.4 MiB
PASS capture succeeded
PASS replay matches eager on the same rows
PASS replay picks up rows refilled after capture
[overlap] replay alone 0.287 ms, with a gather churning 0.446 ms, cost +0.160 ms
[overlap] eager  alone 0.374 ms, with a gather churning 2.202 ms, cost +1.827 ms
```

Three results:

1. The Engram forward captures when `embed` returns a persistent fixed-address
   buffer. vLLM already has that shape. `prepare_embeddings` writes
   `staged_rows`, and `embed` only reads it.
2. A replay picks up rows refilled after capture. The graph reads the buffer
   contents at replay time. It does not replay a snapshot taken at capture. The
   whole prestage design rests on that property.
3. A background gather costs +0.160 ms against a replay. It costs +1.827 ms
   against an eager forward, which is 11x. During a replay the host thread sits
   in a C call and releases the GIL. During an eager forward it sits in Python.

## Result

Bench conditions: prompt set v1 from the prior-art repository, byte-identical
prompts, streaming, temperature 0, thinking off, after warmup. Token counts come
from the server `usage` block. 16,384 max-model-len, `--max-num-seqs 4`, gmu
0.78, DSpark k=5, TP=4. C1 is the mean over 8 categories with counting excluded.
All figures are tok/s.

| Config | C1 agg | C1/stream | C2 agg | C4 agg | Counting C1 | Counting C4 | Prefill 8K |
|---|--:|--:|--:|--:|--:|--:|--:|
| Eager, k=5 | 40.15 | 44.74 | 66.58 | 95.15 | 72.07 | 175.13 | 1552 |
| **Graphs, k=5** | **46.31** | **52.53** | **71.14** | **97.98** | **85.19** | **194.68** | **1896** |
| Graphs, k=10 | 34.09 | 37.36 | 46.85 | 65.24 | 96.23 | 201.92 | 1599 |

Run-to-run spread is about 2%.

Graphs are worth 1.15x on the mixed set. Capture cost 0.30 GiB and 4 s on that
serve. At the default 1,048,576 the graph pool is 0.44 GiB.

## Why the gain here is 1.15x

The prior art reports 5.1 to 41.5 tok/s from the same change. This fleet went
40.15 to 46.31.

The eager path here was never as host-bound as theirs. Their eager step was
about 200 ms, with idle GPUs. This one was 83 ms. Graphs removed about 13 ms.

Their stall came from an Engram reader on NFS doing two serial reads per row.
The reader here is local NVMe with `O_DIRECT`. It does one interleaved read per
row, and it is already parallel. Their 4x came from removing a cost this recipe
does not have.

## Not covered

That every decode batch hit an exact FULL graph. The capture sizes were
derived. No per-batch trace confirmed them.
