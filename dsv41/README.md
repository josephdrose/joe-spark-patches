# DeepSeek-V4.1-Flash on four DGX Sparks (vLLM, TP=4, Engram on NVMe)

**Status: serving.** The endpoint answers prompts and returns parsed tool calls.
Run date 2026-09-10.

`deepseek-ai/DeepSeek-V4.1-Flash` is 475.25 GiB of weights. Four DGX Sparks hold
121.7 GiB each. At TP=4 that is 118.81 GiB per rank against 121.7 GiB physical.
This directory holds the seven changes that make the model fit, serve, and run
with CUDA graphs.

The recipe ports no kernel and compiles no vLLM from source. The layer that
compiles the op shim takes 12.8 seconds. The two Python layers take about a
second each.

## Hardware

| Item | Value |
|---|---|
| Nodes | 4 x NVIDIA DGX Spark |
| GPU | GB10, sm_121, compute capability 12.1 |
| SMs per GPU | 48 |
| Shared memory | 100 KiB per SM |
| Memory per box | 121.7 GiB. CPU and GPU share it. |
| Interconnect | ConnectX-7, RoCE v2 |
| NVMe | SAMSUNG MZALC4T0HBL1-00B07, 3.7 TB, 512-byte logical blocks |

The shared pool causes the memory problem. Read
[docs/fit-and-engram.md](docs/fit-and-engram.md).

## The model

| Item | Value |
|---|---|
| Repo | `deepseek-ai/DeepSeek-V4.1-Flash` |
| Revision | `df42c109f1defefcbfcedbe7d905718a12266e40` |
| License | MIT |
| Shards | 48 safetensors, 475.25 GiB |
| Parameters | 552B backbone plus 196B Engram |
| Active | 8B prefill, 16B decode |
| Context | 1,048,576 |

Sizes below are measured from all 48 safetensors headers.

| Component | GiB |
|---|--:|
| Routed experts, already E2M1 fp4 | 268.9 |
| Engram n-gram table | 189.13 |
| MTP | 7.4 |
| Attention and dense | 4.9 |
| Embedding, head, vision | 4.0 |

## Measured

Every number below comes from one serve on this fleet.

| Item | Value |
|---|---|
| Endpoint | `spark-2:8410`, served model `deepseek-v41-flash` |
| Fingerprint | `vllm-0.28.1rc1.dev391+g29af8bd67-tp4` |
| Image | `vlspeed-eng:4` |
| Weights resident | 81.36 GiB per rank, with the DSpark draft layers |
| Load time | 391 s |
| CUDA graphs | on. 0.30 GiB captured in 4 s |
| KV cache | 476,844 tokens at 9.03 GiB |
| Context | 16,384 |
| Concurrency | 29.1x at 16,384 |
| Host memory free while serving | 7 to 9 GiB per box |

Speed, DSpark k=5, `--max-num-seqs 4`, gmu 0.78. C1 is one stream, C4 is four.
The aggregate is the mean over 8 categories with counting excluded. All figures
are tok/s.

| Config | C1 agg | C4 agg | Counting C1 |
|---|--:|--:|--:|
| Eager | 40.15 | 95.15 | 72.07 |
| **CUDA graphs** | **46.31** | **97.98** | **85.19** |

Run-to-run spread is about 2%. Confirmed by hand on the final serve: 70.52 tok/s
single stream on a counting prompt, 206 tokens in 2.92 s, `finish_reason: stop`.
The full table, including k=10 and 8K prefill, is in
[docs/cuda-graphs.md](docs/cuda-graphs.md).

A 3,853-token prompt was answered correctly.

Verified output:

```json
{"model": "deepseek-v41-flash",
 "content": "Mercury, Venus, Earth, Mars.",
 "reasoning": "We need answer. Need name four inner planets in order from Sun...",
 "finish_reason": "stop"}
```

A tool call through the `deepseek_v41` parser returns `finish_reason:
"tool_calls"` and `{"name": "get_weather", "arguments": "{\"city\": \"Los
Angeles\"}"}`.

Engram read cost, measured cold against one rank's real 23.6 GiB shard:

| Rows per gather | Median ms | Share of a 33.3 ms step |
|--:|--:|--:|
| 12 (the TP=4 decode load) | 1.13 | 3.4% |
| 48 | 1.89 | 5.7% |

## Quickstart

Read [docs/RECIPE.md](docs/RECIPE.md) for the full sequence. The short form:

```bash
# 1. Stage the checkpoint on all four boxes, 475 GiB each.
#    Pin revision df42c109f1defefcbfcedbe7d905718a12266e40.

# 2. Build the image. Three layers, no compile of vLLM from source.
./build/vl41-build-image.sh        # vl41-eng:2
./build/vlpage-build-image.sh      # vlpage-eng:3
./build/vlspeed-build-image.sh     # vlspeed-eng:4

# 3. Build each rank's Engram row files. Run per box, per layer.
python3 tests/build_real_engram_table.py --out $HOME/table --layer 1 --rank 0
python3 tests/build_real_engram_table.py --out $HOME/table --layer 14 --rank 0

# 4. Serve. Edit NODE_TS and NODE_LAN first.
DSPARK=5 GPU_UTIL=0.78 ./launch/vlspeed-tp4-4node-up.sh
```

The IP addresses in `launch/vlspeed-tp4-4node-up.sh` are RFC 5737 documentation
ranges. Replace them with your own. Host paths use `$HOME`.

## What had to be fixed

Seven changes, in boot order. Each document states the failure verbatim, the
cause, and the change.

| # | Problem | Fix | Document |
|--:|---|---|---|
| 1 | 118.81 GiB per rank against 121.7 GiB installed. `EngramConfig.cpu_offload=True` frees zero bytes on a unified pool. | Keep the 189.13 GiB Engram table on NVMe. Gather rows on the CPU. | [fit-and-engram.md](docs/fit-and-engram.md) |
| 2 | No aarch64 vLLM wheel exists for PR #56214. A from-source build costs hours. | Take the official wheel for the PR's parent commit. Copy the PR's 87 changed `vllm/*.py` on top. | [image-build.md](docs/image-build.md) |
| 3 | The PR widens three op schemas with `apply_q_norm`. The parent wheel's `_C` has no such argument. | Compile the PR's own `.cu` out of tree for sm_121. Register it as `torch.ops.vl41`. | [op-shim-apply-q-norm.md](docs/op-shim-apply-q-norm.md) |
| 4 | `SM120 sparse-MLA has no decode kernel for this shape ... page_block_size=32`. | Put every KV page on 64 states. Three spec changes in Python. | [page-size-64.md](docs/page-size-64.md) |
| 5 | The decode step costs 83 ms eager. Dropping `--enforce-eager` raises, because the disk read sits inside the forward. | Stage the rows first, then capture with exact DSpark batch sizes. Worth 1.15x. | [cuda-graphs.md](docs/cuda-graphs.md) |
| 6 | `prepare_embeddings` reads NVMe inside the forward, so the forward holds a host call. | Read the rows in `prepare_inputs` instead, both layers in parallel. | [engram-prestage.md](docs/engram-prestage.md) |
| 7 | `persistent_topk` is reported to die on GB10 at a 1M-wide logits buffer. It never failed here. | Route sm12x decode to `top_k_per_row_decode`. Insurance for long context. | [topk-swap.md](docs/topk-swap.md) |

Three further documents cover the rest:

- [silent-corruption.md](docs/silent-corruption.md) reports three bugs that
  returned wrong rows with no exception. Read this before you trust any disk
  reader.
- [spec-decode.md](docs/spec-decode.md) records that classic MTP is refused by
  source, and that DSpark takes its place.
- [tests.md](docs/tests.md) states what the harness runs and what each result
  proves.

## Not proven

State of the work on 2026-09-10. Nothing below was measured.

- **Quality.** No benchmark and no evaluation run. No greedy reference, no
  garble gate, no needle test. Four correct prompts do not measure quality.
- **Engram hash ids.** The reader was verified against the checkpoint, and the
  staged rows were verified inside the serve. Both are below. Neither checks
  that the model computes the right row ids: `rolling % prime[h] + offset[h]`,
  the compressed token map, or `EngramLayout.offsets`. A wrong id reads a
  correct row and raises nothing.
- **Context past 16,384.** Every run was at 16,384. Nothing exercised the KV
  pool at 300K or 1M, the `persistent_topk` failure mode, or FlashInfer #5015.
- **Concurrency above 4.** The bench ran 1, 2 and 4 streams.
- **That every decode batch hit an exact FULL graph.** The capture sizes were
  derived. No per-batch trace confirmed them.
- **DSpark k other than 5 and 10.**
- **This prestage against the prior art's, head to head.**
- **Vision.** The vision path was never exercised.

## Engram reader, verified

144 rows across 8 shards. 0 differ. The serve ran throughout and was not
restarted.

Two comparisons, both bitwise:

1. The `.bin` row files against the checkpoint, read by `pread` at computed
   offsets. 8 of 8 shards, 18 of 18 rows byte-identical.
2. The reader's own `gather()` output against a reference dequantized from
   checkpoint bytes. 8 of 8 shards equal, 0 bad values.

14 of the 18 rows per shard were seams: the first and last row of each of the 6
owned buckets, plus the shard's own first and last. Random interior rows do not
catch an offset error.

Four negative controls. Each had to fail, and did:

| Control | Result |
|---|---|
| Local row ids instead of global | 18 of 18 differ on ranks 1, 2, 3 |
| Reader built with `row_start=0` | 16 of 16 differ on ranks 1-3 |
| Reader built with `row_start+1` | 16 of 16 differ on every rank |
| Gather shifted by one row | 18 of 18 differ |

The comparison is index-sensitive at one-row granularity.

144 rows of 768,022,850 is seam coverage. It does not detect sparse random
corruption.

## Staged rows inside the serve, verified

One boot at `VL41_ENGRAM_PRESTAGE_VERIFY=2000` made `prepare_embeddings` redo
the lookup inside the forward and compare bitwise against the staged rows. All
four ranks: 2,000 lookups, 41,698 token-rows, 0 mismatches.

The 2,000 calls covered single-stream decode, 4 concurrent streams, an
8,427-token chunked prefill, and DSpark verification batches.

This proves the two hashes agree and the rows land in the right buffer. It
proves nothing about whether either hash is the right hash. See
[docs/engram-prestage.md](docs/engram-prestage.md).

## Layout

| Path | Contents |
|---|---|
| `docs/RECIPE.md` | Bare fleet to serving endpoint, in order |
| `docs/*.md` | One file per fix |
| `patch/` | The exact files, with `md5sums.txt` and `mounts.txt` |
| `build/` | The three image layers |
| `launch/` | The 4-node bring-up script |
| `tests/` | The harness, the negative controls, the table builder |

## Credits

- The **vLLM team** for the day-0 `dsv41-feat` branch and PR #56214. Everything
  here sits on top of their work.
- **tonyd2wild** and **Kai** for
  [DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark).
  They reached the same three KV-spec changes independently. We found their
  repository after deriving ours. Fixes 5, 6 and 7 here take their graph flags,
  their `model_state.py` wiring and their `sparse_attn_indexer.py` edit.
- **eugr / local-inference-lab** for the b12x sm120 and sm121 kernels. The base
  image is theirs. Without it nothing here runs.

Apache-2.0, per the repository root `LICENSE`. The model weights are MIT and
belong to DeepSeek.
