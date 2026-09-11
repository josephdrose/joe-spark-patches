# DeepSeek-V4.1-Flash on four DGX Sparks (vLLM, TP=4, Engram on NVMe)

**Status: serving.** The endpoint answers prompts and returns parsed tool calls.
Run date 2026-09-11.

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

Every number below comes from this fleet on 2026-09-11, at the full 1,048,576
window. The serve is `vlspeed-eng:4` at TP=4 with the b12x bf16 MoE backend.
The launcher defaults reproduce it.

### The serve

| Item | Value |
|---|---|
| Endpoint | `spark-2:8410`, served model `deepseek-v41-flash` |
| Fingerprint | `vllm-0.28.1rc1.dev391+g29af8bd67-tp4` |
| Image | `vlspeed-eng:4` |
| Context | 1,048,576 |
| Memory and batching | `--gpu-memory-utilization 0.78 --max-num-seqs 8 --max-num-batched-tokens 8192` |
| MoE backend | `--moe-backend b12x` plus `VLLM_B12X_MOE_FP4_FORCE_A16=1` |
| Backend line in the head log | `Using 'B12X_MXFP4_BF16' Mxfp4 MoE backend.` |
| Speculative decode | DSpark k=5 |
| CUDA graphs | FULL_AND_PIECEWISE |
| Tokenizer | `--tokenizer-mode deepseek_v41` |
| Tool calls | `--enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41` |
| Engram table | 189.13 GiB, on NVMe |
| Docker memory cap | None. Read finding 3. |

Boot and pool, head rank:

| Item | Value |
|---|--:|
| Weight load | 289.73 s |
| Model loading | 368.06 s |
| Available KV cache memory | 12.84 GiB |
| KV cache | 2,900,475 tokens |
| Max concurrency at 1,048,576 | 2.77x |
| Engine init | 99.61 s |

### Single stream

Temperature 0.4, `reasoning_effort` 75, 3 repetitions, median decode tok/s.
TTFT is excluded from the rate.

| Workload | tok/s |
|---|--:|
| counting | 96.41 |
| code | 74.54 |
| code, thinking on | 67.79 |
| prose | 38.59 |

### Concurrency

Mixed set of 8 categories. Counting is excluded from the aggregate. The two
boots ran the same prompt set and differ in `--max-num-seqs`.

| Streams | agg tok/s, seqs=4 | agg tok/s, seqs=8 | TTFT s, seqs=4 | TTFT s, seqs=8 |
|--:|--:|--:|--:|--:|
| 1 | 51.33 | 51.58 | 0.303 | 0.313 |
| 2 | 81.03 | 81.26 | 0.376 | 0.379 |
| 4 | 113.85 | 116.06 | 0.522 | 0.496 |
| 6 | 100.35 | 141.74 | 2.249 | 0.573 |
| 8 | 110.34 | 159.56 | 3.212 | 0.660 |

At `seqs=4` the 6-stream and 8-stream points fall below the 4-stream point. The
extra streams queue, and TTFT rises to 3.2 s. At `seqs=8` the aggregate keeps
rising to 159.56 tok/s.

The counting prompt is the ceiling. At `seqs=8` its aggregate is 271.84 tok/s on
6 streams and 311.52 tok/s on 8 streams.

The `seqs=4` boot allocated 3,709,285 KV tokens. The `seqs=8` boot allocated
2,900,475.

### MoE backends

Single stream at 1,048,576, temperature 0.4, `reasoning_effort` 75, median of 3.

| Workload | DeepGEMM | b12x W4A8 | b12x bf16 |
|---|--:|--:|--:|
| counting | 88.07 | 97.45 | 95.49 |
| code | 62.86 | 69.24 | 68.40 |
| code, thinking on | 59.08 | 65.91 | 64.83 |
| prose | 39.27 | 44.24 | 46.46 |
| KV tokens | 1,372,132 | 1,175,588 | 3,709,285 |

The KV row is `--max-num-seqs 4` on all three columns. At `--max-num-seqs 8` the
bf16 backend gives 2,900,475 tokens.

### Quality

tool-eval-bench 2.6.1, 88 scenarios, 3 trials each, seed 42, temperature 0.4,
`reasoning_effort` 75.

| Serve | Trial scores | Mean | SD |
|---|---|--:|--:|
| b12x W4A8 | 91, 89, 93 | 91.0 | 2.0 |
| b12x bf16 | 90, 91, 90 | 90.3 | 0.6 |
| DeepGEMM | 90, 90, 91 | 90.3 | 0.6 |
| GLM-5.3-Flash, a different model for reference | 89, 94, 91 | 91.3 | 2.5 |

Scenario counts run both ways. bf16 against W4A8 is 3 better, 5 worse and 80
tied. Either b12x backend against DeepGEMM is 5 better, 6 worse and 77 tied.

This benchmark cannot separate the three backends. The choice rests on speed and
memory.

### Needle recall

Measured 2026-09-10 at the same 1,048,576 window, on the DeepGEMM backend.
7 runs, 7 pass. The needle string was `COPPER-LANTERN-8315`. Depth is the
fraction of the prompt the needle sits at. The rate column is prefill only.

| Target | Depth | Prompt tokens | TTFT s | Prefill tok/s | Result |
|--:|--:|--:|--:|--:|---|
| 32,768 | 0.1 | 32,376 | 17.9 | 1806.5 | pass |
| 32,768 | 0.5 | 32,351 | 17.5 | 1853.3 | pass |
| 32,768 | 0.9 | 32,330 | 16.0 | 2021.2 | pass |
| 131,072 | 0.1 | 130,422 | 77.1 | 1692.7 | pass |
| 131,072 | 0.5 | 130,258 | 75.1 | 1735.1 | pass |
| 131,072 | 0.9 | 130,219 | 68.7 | 1895.8 | pass |
| 262,144 | 0.5 | 260,119 | 164.3 | 1583.4 | pass |

That boot carried one extra patch, a KV-accounting logger, which changes no
serving path. Its pool came out at 1,150,699 tokens and 5.55 GiB.

### Verified output

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

### Earlier numbers at 16,384

The eager against CUDA-graphs comparison was measured on 2026-09-10, at
`--max-model-len 16384`, `--max-num-seqs 4`, and the DeepGEMM backend. Graphs
were worth 1.15x there. Those figures do not compare with the tables above.
They are kept in [docs/cuda-graphs.md](docs/cuda-graphs.md).

## Findings

Five results from the b12x work on 2026-09-11.

### 1. `--moe-backend b12x` selects W4A8 on its own

The bf16 variant needs `VLLM_B12X_MOE_FP4_FORCE_A16=1` as well. The selection
sits in `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, in
`_get_requested_backends`. Its comment reads: "W4A8 is the high-throughput b12x
path and is preferred when the model does not request an activation format."

Check the head log. It prints `Using 'B12X_MXFP4_MXFP8' Mxfp4 MoE backend.` for
W4A8 and `Using 'B12X_MXFP4_BF16' Mxfp4 MoE backend.` for bf16.

### 2. The shipped b12x GB10 profile predates this model

`b12x/policy/_profiles/data/nvidia.gb10.48sm.json.gz` in the image is dated
2026-09-04. This model reached the fleet on 2026-09-10.

The profile does not cover these shapes, so the serve logs a policy fallback:

```
b12x policy fallback: moe.decode is using a heuristic on nvidia gb10
(compute capability 12.1, 48 SMs) because profile 'nvidia.gb10.48sm' does not
cover the query; query={'activation': 'silu', 'hidden_size': 5120,
'intermediate_size': 576, 'num_experts': 384, 'num_tokens': 8192,
'quant_mode': 'w4a16', 'routed_rows': 49152, 'source_format': 'fp4_e8m0_k32',
'top_k': 6}
```

The W4A8 serve logs the same fallback with `quant_mode: 'w4a8_mx'`. Every b12x
number on this page therefore comes from the heuristic. No tuned profile covers
these shapes.

### 3. A docker `--memory` cap starves the KV pool at 1M

b12x needs about 1.16 GiB more resident memory than DeepGEMM. With
`--memory 112g` the 1M boot died:

```
ValueError: To serve at least one request with the model's max seq len
(1048576), 3.42 GiB KV cache is needed, which is larger than the available KV
cache memory (1.74 GiB). Based on the available memory, the estimated maximum
model length is 491456.
```

DeepGEMM got 2.9 GiB under the same cap and died the same way. Uncapped, the
DeepGEMM boot got 5.22 GiB and 1,372,132 KV tokens. The launcher now clears the
cap above `CTX` 262144.

### 4. b12x weight prep is sensitive to host memory fragmentation

`_canonicalize_fp4_zero_signs_` in
`vllm/model_executor/layers/fused_moe/b12x.py` makes 8 full-size uint8
temporaries per MoE weight tensor. About 4 are live at once. Per rank per layer
w13 is 1.06 GiB and w2 is 0.53 GiB, across 40 layers.

On a box 10 weeks into its uptime, that prep took 870 s. Three identical nodes
took 80 s, 85 s and 128 s. The slow node ran at 100% system time with the GPU at
0%, at 190 memory compactions per second, of which 39% failed.

The other three ranks then blocked in `broadcast_object_list`, inside
`_init_message_queues`. That collective has no timeout. The boot presents as a
hang, with every GPU at 2%.

The fix is `drop_caches` plus `compact_memory` on all four boxes, between
teardown and start, while the 81 GiB of weights is released. The launcher does
this by default. Prep spread went from 10.9x to 1.08x, and the slow node came
down to 88 s.

### 5. The nearest published recipe stops at 300,000 context

The nearest published recipe caps at `--max-model-len 300000`, and records that
1M has not been re-run on that stack. The b12x tables above are at 1,048,576.

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
DSPARK=5 ./launch/vlspeed-tp4-4node-up.sh                  # 1,048,576, the default
DSPARK=5 CTX=16384 ./launch/vlspeed-tp4-4node-up.sh        # 16,384
```

The first command is the serve every table above was measured on. It defaults to
the b12x bf16 backend, `--max-num-seqs 8` and no docker memory cap.

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

State of the work on 2026-09-11. Nothing below was measured.

- **Quality outside tool use.** tool-eval-bench 2.6.1 scores the three backends
  at 90.3 to 91.0 over 88 scenarios and 3 trials. Those scenarios are tool-use
  tasks at short context. No long-context quality run, no greedy reference and
  no garble gate.
- **Engram hash ids.** The reader was verified against the checkpoint, and the
  staged rows were verified inside the serve. Both are below. Neither checks
  that the model computes the right row ids: `rolling % prime[h] + offset[h]`,
  the compressed token map, or `EngramLayout.offsets`. A wrong id reads a
  correct row and raises nothing.
- **Context between 262,144 and 1,048,576.** The needle passed at 262,144. No
  prompt above it was run, so the top of the declared window is unproven. The
  needle runs used the DeepGEMM backend. No needle was run on b12x.
- **The `persistent_topk` failure mode.** `top_k_per_row_decode` was installed
  for every long-context run here, so nothing exercised the kernel it replaces.
- **Concurrency above 8.** The bench ran 1, 2, 4, 6 and 8 streams.
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
