# joe-spark-patches

Out-of-tree code and patches for running **DeepSeek** models on a 4-node
**DGX Spark (GB10 / sm_121)** cluster with vLLM.

`dsv41/` is a full recipe for **DeepSeek-V4.1-Flash** at TP=4. Status: serving.

Everything else came out of **DeepSeek-V4-Flash-DSpark**. Collected while getting
DSpark speculative decoding to actually engage — posted in case they save
someone else the time.

V4-Flash context: aidendle B12X image, sm_121 / CUDA 12.1, TP across all 4 nodes,
prefix-caching on, ~76-90 tok/s on code. **B12X stays on** — DSpark and B12X are
not mutually exclusive; you do not need to disable B12X to get DSpark working.

> Throughput numbers are from our own runs, not a controlled benchmark. The
> proposer is first-pass code, shared as-is — read it before you run it.

## dsv41/

DeepSeek-V4.1-Flash at TP=4 on four Sparks. Serving.

475.25 GiB of weights against 121.7 GiB per box. At TP=4 that is 118.81 GiB per
rank. vLLM's Engram `cpu_offload` frees zero bytes on GB10, because the CPU and
the GPU share one memory pool. This recipe keeps the 189.13 GiB Engram table on
NVMe and gathers rows on the CPU.

Measured on one serve: 391 s load, 476,844 KV tokens, 46.31 tok/s on one stream
with CUDA graphs and DSpark k=5, 97.98 tok/s aggregate at four streams. That was
16,384 context.

Seven fixes, each documented with its failure verbatim: the memory fit, the
image build, the `apply_q_norm` op schema, the KV page size, CUDA graph capture,
the Engram prestage, and the indexer top-k. `dsv41/patch/` carries the exact
files with md5s and a `mounts.txt`. `dsv41/docs/RECIPE.md` runs from a bare
fleet to a serving endpoint.

Unproven and listed as such in `dsv41/README.md`: quality, concurrency above 4
streams, context past 16,384, and the vision path.

## dspark/ — custom speculative-decode proposer

An out-of-tree vLLM `custom_class` proposer running DeepSeek's 3-stage DSpark
draft cascade. Registered via:

```
--speculative-config '{"method":"custom_class",
   "model":"dspark.proposer.DSparkProposer","num_speculative_tokens":5}'
```

Two things it gets right that a naive drafter doesn't:

- **Batched decode.** Keeps per-request draft-window state keyed by `req_id`,
  and captures the draft+sample cudagraph once at `B = max_num_seqs`, replayed
  every step. Without this the drafter no-ops the moment a second concurrent
  request joins the batch — and still pays draft overhead, so aggregate
  throughput drops *below* the no-spec baseline.
- **Prefix-cache / chunked prefill.** Detects the new-sequence boundary from
  each request's seqlen trajectory rather than a banked-vs-seqlen delta, so
  cached and chunked-prefill contexts don't reset-storm the draft window (which
  collapses acceptance to ~1 = base speed, progressively worse with depth).

Debug: `DSPARK_TRACE=1` + `dspark/replay_banking.py` replays a captured trace
offline (zero weights) to inspect the banking state. Other knobs: `DSPARK_GRAPH=0`
(eager-draft fallback), `DSPARK_TIMING=1` (draft-forward timing).

## parsers/ — DSML tool + reasoning parser patches

DSpark drops a variable-length chunk of the tool-call opener tag at
draft-rejection boundaries, so tool calls leak out as raw markup unless the
parser is made spec-decode-aware and can recover from a truncated opener. Stock
parsers assume the opener arrives intact. Also handles the reasoning/tool split
for the DSML format. Derived from vLLM's parsers (Apache-2.0).

These are **drop-in replacements** for files in vLLM's tree — each file's header
says which one it replaces (e.g. `vllm/reasoning/deepseek_v3_reasoning_parser.py`).
The relative imports resolve when the file sits in `vllm/reasoning/` or
`vllm/tool_parsers/`, not when the folder is imported as a standalone package.

*(Regression tests exist but are redacted — the fixtures were captured live
sessions containing private data.)*

## cluster-fabric/ — 4-node launcher + switchless RoCE ring notes

`dsv4-flash-4node-up.sh` is the native multi-node vLLM launcher (worker-first,
then head) with a `NET=ring|mesh|switch` fabric matrix. See
`cluster-fabric/README.md` for the switchless ring topology: the cabling loop,
the uncabled-diagonal relay, `NCCL_SKIP_TREE_CONNECT`, and the PP=2/TP=2
diagonal-free rank swizzle. The IPs and host paths in the script are
placeholders (RFC5737 documentation ranges) — fill in your own.

> `ring`/`mesh` modes LD_PRELOAD a patched NCCL — a source fork of NVIDIA/nccl
> v2.30.7 with a 2-hop RoCE relay overlay, at
> github.com/josephdrose/nccl-spark-switchless. Build it and preload the `.so`;
> the binary isn't shipped here. `NET=switch` needs only stock NCCL.

## sparkrun-cx7/ — multi-node interface pinning

If `ray` / NCCL / GLOO bind the default-route interface instead of the fast
interconnect (on a cluster with no shared management LAN), the placement group
just hangs — or NCCL builds every QP then deadlocks at the first data transfer,
a silent hang. These pin the node IP, socket interfaces, IB HCA and
`VLLM_HOST_IP` to the direct link. See `sparkrun-cx7/README.md` for the
four-layer root cause.

## License

Apache-2.0. The parsers derive from the vLLM project (Apache-2.0). The `dsv41/`
patches derive from vLLM PR #56214 (Apache-2.0), with dequantization arithmetic
ported from sgl-project/sglang (Apache-2.0). Credits for `dsv41/` are in
`dsv41/README.md`.
