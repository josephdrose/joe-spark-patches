# Recipe: bare fleet to a serving endpoint

Target: `deepseek-ai/DeepSeek-V4.1-Flash` at TP=4 on four DGX Sparks, with an
OpenAI-compatible endpoint on the head node.

Result on this fleet at the default 1,048,576 window: a 1,180,171-token KV
pool, and a needle at 262,144 that comes back correct.

At `CTX=16384` the same image gives 391 s load, 476,844 KV tokens, and 46.31
tok/s on one stream with CUDA graphs and DSpark k=5.

## 0. Prerequisites

| Item | Requirement |
|---|---|
| Nodes | 4 x DGX Spark, GB10, sm_121 |
| Free disk per box | 476 GiB for the checkpoint, plus 48 GiB for the Engram tables |
| Network | A control-plane LAN, plus ConnectX-7 for NCCL |
| SSH | Key-based login from the launch box to all four nodes |
| Docker | Working `--gpus all`, plus `/dev/infiniband` passthrough |
| Base image | `eugr/spark-vllm-b12x:latest`, pulled on all four boxes |

Check that the base image has the same digest on every box. A mixed image set
produces NCCL errors that resemble fabric faults.

## 1. Stage the checkpoint

Pull revision `df42c109f1defefcbfcedbe7d905718a12266e40` once. Copy it to the
other three boxes over the fast fabric. One WAN pull of 88 files took 43 minutes
here.

Each box needs the full 475.25 GiB. Every rank reads its own Engram shard from
the local copy.

## 2. Build the image

Three layers. None compiles vLLM from source.

```bash
./build/vl41-build-image.sh      # vl41-eng:2
./build/vlpage-build-image.sh    # vlpage-eng:3
./build/vlspeed-build-image.sh   # vlspeed-eng:4
```

`vl41-build-image.sh` does five things:

1. Downloads the PR #56214 source tree at commit `e47aa780`.
2. Downloads the official aarch64 wheel for that commit's **parent**,
   `29af8bd67`. See [image-build.md](image-build.md) for why the parent.
3. Copies the PR's 87 changed `vllm/*.py` over the installed package.
4. Applies `patch/engram-disk-table.patch`, which puts the Engram rows on disk.
5. Compiles `patch/vl41-ops-bindings.cpp` plus the PR's own kernel `.cu` into
   `vl41_ops.so`. See [op-shim-apply-q-norm.md](op-shim-apply-q-norm.md).

`vlpage-build-image.sh` runs `patch/vlpage-page64.py` against the installed
tree. That is Python only and takes about one second per box. See
[page-size-64.md](page-size-64.md).

`vlspeed-build-image.sh` runs `patch/vlspeed-topk.py` and
`patch/vlspeed-prestage.py`. Python only, about a second per box. Together they
are what CUDA graphs need. See [engram-prestage.md](engram-prestage.md),
[topk-swap.md](topk-swap.md) and [cuda-graphs.md](cuda-graphs.md).

Set `WORK` to a path with 20 GiB free. The script defaults to
`$HOME/cc-scratch/vllm-v41`.

## 3. Build the Engram row files

Each rank owns `ceil(24 / 4) = 6` of the 24 hash-head buckets per Engram layer.
That is 23.6 GiB per layer per rank, and 47.2 GiB per box.

Layer 1's Engram sits entirely in `model-00047-of-00048`. Layer 14's sits in
`model-00048-of-00048`. Each layer therefore reads one shard.

Run this on every box, with that box's rank:

```bash
python3 tests/build_real_engram_table.py --out $HOME/table --layer 1  --rank $R
python3 tests/build_real_engram_table.py --out $HOME/table --layer 14 --rank $R
```

Each file takes about 27 s to write at 0.87 GiB/s. The builder streams in
1M-row chunks and never holds the full 94 GiB tensor.

**Run this with the default torch device on CPU.** `safetensors.get_slice`
materializes the whole tensor under a CUDA default device. A 64-row read
allocated 94,513 MiB and left one box with 2 GiB free. It raised no error.

Verify the rank offset before you serve. A reader that ignores it gives ranks 1
to 3 rank 0's rows, silently. See [silent-corruption.md](silent-corruption.md).

## 4. Serve

Edit `launch/vlspeed-tp4-4node-up.sh` first:

| Variable | Meaning |
|---|---|
| `NODE_TS` | The addresses the launcher reaches by SSH |
| `NODE_LAN` | The addresses used for `VLLM_HOST_IP` and the rendezvous |
| `TABLE_HOST` | The directory holding this box's two row files |
| `SNAP` | The checkpoint snapshot path inside the container |

Both address arrays ship as RFC 5737 documentation ranges. Replace them.

```bash
DRYRUN=1 ./launch/vlspeed-tp4-4node-up.sh                  # print the per-rank scripts
DSPARK=5 ./launch/vlspeed-tp4-4node-up.sh                  # bring up at 1,048,576
DSPARK=5 CTX=16384 ./launch/vlspeed-tp4-4node-up.sh        # bring up at 16,384
```

`CTX` sets `--max-model-len`. The default is 1,048,576, which matches the live
serve and is where the needle table in the [README](../README.md) was measured.
`CTX=16384` is where the speed table was measured.

The script starts ranks 3, 2 and 1 headless, then rank 0. It polls
`/v1/models` on the head for up to 30 minutes. Load took 391 s at `CTX=16384`
here.

`DSPARK` must be a multiple of 5. See [spec-decode.md](spec-decode.md).

CUDA graphs are on by default. `EAGER=1` turns them off, and the launcher then
drops the graph flags with them. See [cuda-graphs.md](cuda-graphs.md).

## 5. Confirm

```bash
curl -s localhost:8410/v1/models | grep deepseek-v41-flash

curl -s localhost:8410/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"deepseek-v41-flash",
       "messages":[{"role":"user","content":"Name the four inner planets."}]}'
```

A correct response carries `content`, a populated `reasoning` field, and
`finish_reason: "stop"`.

Test a tool call as well. The parser is `deepseek_v41`. A correct response
carries `finish_reason: "tool_calls"`.

## Operational notes

- **Stop every node before a relaunch.** A new worker otherwise joins the dead
  head's rendezvous.
- **A dead head leaves the three workers up**, each holding about 111 GiB.
  Force-remove them before the next attempt.
- **Co-tenancy fails startup.** Any other process on any box triggers `Free
  memory on device cuda:0 ... less than desired GPU memory utilization`.
- **The tokenizer mode is `deepseek_v41`.** The V4 parsers do not carry over.
- **`reasoning_effort` is an integer from 1 to 100** in this model. Clients that
  send `high` or `max` break.
