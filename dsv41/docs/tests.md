# Tests: what runs, and what each result proves

## Running them

```bash
./tests/check_engram_disk.sh
```

The script fetches three files from vLLM at commit `e47aa780`, applies
`patch/engram-disk-table.patch`, drops the harness in beside them, and runs
everything inside a vLLM image on a GPU.

Set `HOST` and `SPARK` first. The defaults are placeholders.

The work tree must sit on a real filesystem. `tmpfs` rejects `O_DIRECT` at open.

## Results on this fleet

| Step | Result |
|---|---|
| `pytest tests/kernels/test_engram.py` | 1 failed, 87 passed, 1 skipped |
| `pytest -k disk` | 22 passed |
| `engram_disk_harness.py` | ALL PASS |
| `negative_control.py` | ALL PASS, 6 of 6 controls fired |
| `config_checks.py` | ALL PASS |

**The one failure imports `vllm.models.deepseek_v4_1.nvidia`.** That is a PR
module the harness does not fetch. The failure predates this work and is
unrelated to it.

## What each file covers

| File | Covers |
|---|---|
| `test_engram_disk.py` | Parity, decode shapes, concurrency, throughput. No vLLM import needed |
| `engram_disk_harness.py` | That `O_DIRECT` is engaged, that the buffered fallback returns the same bytes, that reader threads do not race, that the async API agrees with `gather` |
| `negative_control.py` | Six controls that must fail. See below |
| `config_checks.py` | The `EngramConfig` fields and `compute_hash` |
| `conftest.py` | Registers the PR modules under `vllm.*`, since the base image predates the PR |
| `build_real_engram_table.py` | Builds one rank's real shard, and checks it against the checkpoint tensors |

## The negative controls

A parity assertion that cannot fail proves nothing. Six controls run:

1. The comparison unchanged. This one must pass.
2. The same comparison with one byte of the row file flipped. Must fail.
3. The same comparison with the ue8m0 decode swapped for a
   `torch.float8_e8m0fnu` cast. Must fail. It measured 118,311 differing values.
4. Prefetch one id set, look up another, with the real id key. Must serve correct
   rows.
5. The same, with the key weakened to a shape comparison. Must serve wrong rows.
6. The raced worker pool. It failed 38 to 40 of 40 trials at every thread count.

All six fire.

## Read the exit codes

Every step in `check_engram_disk.sh` echoes `### rc=N`.

This exists because a control once died on an `AttributeError` and the run still
exited 0. The remote script had no `set -e`, so the traceback scrolled past above
a block of `PASS` lines.

Check every `rc=` line. A step that printed `PASS` lines before dying still looks
clean in the tail of the log.

## What the tests do not cover

- The Engram lookup inside a served vLLM forward pass. Nothing compared it to a
  reference there.
- Quality. No benchmark and no evaluation run.
- CUDA graphs. Every serve used `--enforce-eager`.
- Concurrency at the endpoint. One stream only.
- Context past 16,384.
- The vision path.
- The rank row offset. That was checked by reading the code. No test covers it.
  See
  [silent-corruption.md](silent-corruption.md).
