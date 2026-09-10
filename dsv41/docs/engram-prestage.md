# Fix 6: stage the Engram rows before the forward

## The failure

```
RuntimeError: The Engram table is on disk, and a disk lookup cannot be
captured in a CUDA graph because the read is issued by the CPU. Stage the rows
before capture: call prepare_embeddings outside the captured region so it only
reads staged_rows. Engram.forward itself captures fine that way, and a replay
picks up rows refilled in the staging buffer afterwards.
```

The same guard raises on `prefetch`. Both halves of the disk path are CPU work.

## The cause

`Engram.prepare_embeddings` calls `lookup`, and `lookup` reads NVMe. It is
called from inside the model forward.

The seam for a fix already exists in the patched `engram.py`.
`ParallelEngramEmbedding.prefetch(ids)` starts a read. A later `lookup` for
exactly those ids takes the result without reading again. Its docstring records
that no caller uses it. This fix adds one.

## The change

`patch/vlspeed-prestage.py`. Six edits to an installed tree, plus one appended
class. No kernel work.

| File | Edit |
|---|---|
| `common/engram.py` | Add `self.prestage`, gated on a disk table, the V2 runner, and `VL41_ENGRAM_PRESTAGE` |
| `common/engram.py` | `staged_rows` becomes `torch.zeros` |
| `common/engram.py` | `prepare_embeddings` returns early under prestage |
| `common/engram.py` | Append `EngramDiskStager` and `_verify_staged` |
| `nvidia/model_state.py` | Import the three engram names |
| `nvidia/model_state.py` | Build the stager in `__init__`, from the model's `Engram` modules |
| `nvidia/model_state.py` | Call `stage()` at the end of `prepare_inputs` |

`EngramDiskStager.stage()` runs once per step, outside the forward. It runs the
same `NgramHashState` the forward runs, on the same step inputs. It then submits
both Engram layers' reads before waiting on either. Each layer has its own
table, its own fd and its own reader pool, so the two reads overlap. The rows
land in each layer's persistent `staged_rows` buffer.

`prepare_embeddings` then only has to leave that buffer alone. The forward holds
no host call, so it captures.

`staged_rows` changed from `torch.empty` to `torch.zeros` for a reason. The
stager covers this step's unpadded tokens only. CUDA-graph padding rows and the
tail of a padded prefill batch are never staged, and `wkv` still multiplies
them. Uninitialized memory there is a silent wrong answer.

The `model_state.py` wiring is taken from the prior art. The stager body is not
portable from theirs, because the readers differ. This one is a single
interleaved row file per layer per rank, `O_DIRECT`, addressed by global row id.
The helpers their version needs have no counterpart here.

## In-serve verification

`VL41_ENGRAM_PRESTAGE_VERIFY=N` makes `prepare_embeddings` do the lookup again
inside the forward, on the ids the forward hashed. It compares the result
bitwise against the staged rows. After N calls it turns itself off and logs that
it did. One eager boot therefore gives both the proof and a clean benchmark.

The check is eager-only. Under capture it would be the host call this whole fix
removes, and the guard raises there.

One boot at `VL41_ENGRAM_PRESTAGE_VERIFY=2000`, all four ranks:

| Item | Result |
|---|---|
| Lookups compared | 2,000 |
| Token-rows compared | 41,698 |
| Mismatches | 0 |

The 2,000 calls covered single-stream decode, 4 concurrent streams, an
8,427-token chunked prefill, and DSpark verification batches.

## What the verification proves

The staged rows arrive through `prefetch`, on ids the stager hashed in
`prepare_inputs`. The comparison is a fresh gather on the ids the forward
hashed. Equal rows mean two things. The two hashes agree, and the answer landed
in the right buffer.

It does not prove that either hash is the right hash. It says nothing about
what is in the row file. The hash-id surface stays unprotected. See the
[README](../README.md) section "Not proven".

## Not covered

`_ReadPool._busy` contention. The stager issues at most one gather per table,
and the two layers hold separate tables with separate pools, so the test was
not run.
