# cluster-fabric — 4-node vLLM over a switchless RoCE ring

`dsv4-flash-4node-up.sh` launches DeepSeek-V4-Flash across 4 DGX Sparks with
native multi-node vLLM (`--nnodes 4`, **not** sparkrun/ray). It supports three
fabric modes via `NET=`:

- **`switch`** — a real RoCE switch, every pair directly reachable (the simple
  case; current production topology). NCCL runs Tree.
- **`ring`** — a **switchless** loop: each box cabled only to two neighbors,
  the two diagonals uncabled. `NCCL_SKIP_TREE_CONNECT=1` avoids the missing
  diagonal QPs so NCCL runs ring-only over the neighbor links.
- **`mesh`** — the switchless loop plus a **relay overlay** (`NCCL_RELAY_ENABLE=1`)
  that bridges the two uncabled diagonals with a 2-hop relay, so NCCL can run
  Tree across all four instead of ring-only.

## Ring cabling

Physical loop (no switch): `s2 <-> s1 <-> s3 <-> s4 <-> s2`. The diagonals
`s2<->s3` and `s1<->s4` have **no cable**.

For **TP=4** the ring all-reduce just walks the loop neighbor-to-neighbor, so
plain ring order (`s2,s1,s3,s4`) is fine.

For **PP=2 / TP=2** there's a rank-swizzle trick (`RANK_IDX="0 1 3 2"`, box order
`s2,s1,s4,s3`): vLLM's TP-inner layout then puts both TP groups `{s2,s1}` and
`{s4,s3}` on ring edges **and** both PP groups `{s2,s4}` and `{s1,s3}` on ring
edges — so every heavy TP all-reduce and both PP send/recv hops ride a direct
neighbor link, zero diagonal/relay.

## Load-bearing gotchas (these silently hang if wrong)

- **`GLOO_SOCKET_IFNAME` + `VLLM_HOST_IP` pinned to each node's fast-NIC IP.**
  Otherwise Gloo/NCCL bind the out-of-band management IP and hang. (See the
  `sparkrun-cx7/` writeup for the same failure under sparkrun's auto-detection.)
- **`--ulimit nofile=1048576`.** A full-mesh NCCL setup FD-starves at the 1024
  default and every worker dies with `Too many open files`.
- **Worker-first launch.** Start ranks 3,2,1 (`--headless`), then the head
  (rank 0) which serves the API.
- **`NCCL_DEBUG=INFO` on first bring-up.** A "hang" during NCCL bootstrap is
  often a data-plane deadlock (QPs build, first transfer stalls) from the wrong
  interface/IP — the log just goes silent. Debug output distinguishes "slow
  bootstrap, wait" from "wrong IP/port, fix."

## The relay patch (NCCL source)

`mesh` mode's 2-hop diagonal relay is a **source patch on NCCL** — a fork of
NVIDIA/nccl v2.30.7 that adds `src/transport/net_ib/relay.{cc,h}` (QP setup,
RoCE GID discovery, a forwarding pump thread, and the
`NCCL_RELAY_ENABLE`/`_DIAG`/`_BENCH` gates). Source:
https://github.com/josephdrose/nccl-spark-switchless — build it and LD_PRELOAD
the resulting `libnccl.so.2`. The compiled binary isn't shipped here. On a real
RoCE **switch** (`NET=switch`), stock NCCL works and the relay isn't needed.

## Placeholders

`NODE_TS` / `NODE_LAN` are RFC5737 documentation IPs — replace with your tailnet
and LAN addresses. Host artifact paths use `$HOME/...`. `IMAGE=` points at an
aidendle-derived DSV4 image tag.
