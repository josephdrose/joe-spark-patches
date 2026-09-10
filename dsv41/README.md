# DeepSeek-V4.1-Flash on four DGX Sparks (vLLM, TP=4, Engram on NVMe)

**Status: serving.** The endpoint answers prompts and returns parsed tool calls.
Run date 2026-09-10.

`deepseek-ai/DeepSeek-V4.1-Flash` is 475.25 GiB of weights. Four DGX Sparks hold
121.7 GiB each. At TP=4 that is 118.81 GiB per rank against 121.7 GiB physical.
This directory holds the four changes that make the model fit and serve.

The recipe ports no kernel and compiles no vLLM from source. The image build
takes 12.8 seconds.

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
| Weights resident | 78.95 GiB per rank |
| Load time | 314 s |
| KV cache | 620,493 tokens at 13.04 GiB |
| Context | 16,384 |
| Graphs | off, `--enforce-eager` |
| Decode, eager | 14.94 tok/s |
| Decode, DSpark k=5 | 61.90 tok/s |

Both throughput figures use the same prompt. A 3,853-token prompt was answered
correctly.

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

# 2. Build the image. Two layers, no compile from source.
./build/vl41-build-image.sh        # vl41-eng:2
./build/vlpage-build-image.sh      # vlpage-eng:3

# 3. Build each rank's Engram row files. Run per box, per layer.
python3 tests/build_real_engram_table.py --out $HOME/table --layer 1 --rank 0
python3 tests/build_real_engram_table.py --out $HOME/table --layer 14 --rank 0

# 4. Serve. Edit NODE_TS and NODE_LAN first.
DSPARK=5 ./launch/vlpage-tp4-4node-up.sh
```

The IP addresses in `launch/vlpage-tp4-4node-up.sh` are RFC 5737 documentation
ranges. Replace them with your own. Host paths use `$HOME`.

## What had to be fixed

Four changes, in boot order. Each document states the failure verbatim, the
cause, and the change.

| # | Problem | Fix | Document |
|--:|---|---|---|
| 1 | 118.81 GiB per rank against 121.7 GiB installed. `EngramConfig.cpu_offload=True` frees zero bytes on a unified pool. | Keep the 189.13 GiB Engram table on NVMe. Gather rows on the CPU. | [fit-and-engram.md](docs/fit-and-engram.md) |
| 2 | No aarch64 vLLM wheel exists for PR #56214. A from-source build costs hours. | Take the official wheel for the PR's parent commit. Copy the PR's 87 changed `vllm/*.py` on top. | [image-build.md](docs/image-build.md) |
| 3 | The PR widens three op schemas with `apply_q_norm`. The parent wheel's `_C` has no such argument. | Compile the PR's own `.cu` out of tree for sm_121. Register it as `torch.ops.vl41`. | [op-shim-apply-q-norm.md](docs/op-shim-apply-q-norm.md) |
| 4 | `SM120 sparse-MLA has no decode kernel for this shape ... page_block_size=32`. | Put every KV page on 64 states. Three spec changes in Python. | [page-size-64.md](docs/page-size-64.md) |

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

- **Quality.** No benchmark and no evaluation run. Four correct prompts do not
  measure quality.
- **Engram numerics inside the model.** The lookup was checked bitwise against
  the checkpoint outside vLLM. Nothing compared it to a reference inside a
  served forward pass. In progress.
- **CUDA graphs.** Every run used `--enforce-eager`. In progress.
- **Concurrency.** One stream only.
- **Context past 16,384.** The KV pool holds 620,493 tokens. No long prompt was
  run.
- **Vision.** The vision path was never exercised.

## Layout

| Path | Contents |
|---|---|
| `docs/RECIPE.md` | Bare fleet to serving endpoint, in order |
| `docs/*.md` | One file per fix |
| `patch/` | The exact files, with `md5sums.txt` and `mounts.txt` |
| `build/` | The two image layers |
| `launch/` | The 4-node bring-up script |
| `tests/` | The harness, the negative controls, the table builder |

## Credits

- The **vLLM team** for the day-0 `dsv41-feat` branch and PR #56214. Everything
  here sits on top of their work.
- **tonyd2wild** and **Kai** for
  [DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark).
  They reached the same three KV-spec changes independently. We found their
  repository after deriving ours. Their repository also carries `persistent_topk`
  and CUDA-graph prestage work that this one does not.
- **eugr / local-inference-lab** for the b12x sm120 and sm121 kernels. The base
  image is theirs. Without it nothing here runs.

Apache-2.0, per the repository root `LICENSE`. The model weights are MIT and
belong to DeepSeek.
