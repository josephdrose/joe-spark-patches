# sparkrun patches — 2-Spark CX7 cluster (no shared management LAN)

Local patches that let `sparkrun` form **and serve** a 2-node TP=2 cluster across
the two DGX Sparks, whose only inter-node link is a direct CX7 cable (no shared
management LAN). **Re-apply after every `sparkrun update` / reinstall** — these
edit the installed pip package + the eugr cache, which updates overwrite.

```bash
./apply.sh
# cluster must be registered on the CX7 host IPs:
sparkrun cluster update my-cluster --hosts 192.168.1.74,192.168.1.49 --user $USER
sparkrun run @eugr/nemotron-3-super-nvfp4 --hosts 192.168.1.74,192.168.1.49
```

## Root cause (one class of bug, four layers)

sparkrun assumes every node is reachable on a routable **management network** and
auto-detects the interface/IP from the **default route** (`ip route get 8.8.8.8`).
the Sparks have no shared mgmt LAN — the default route is the OOB NIC `enP7s7`
(`192.168.0.x`), which does **not** route box-to-box. The only inter-node path is
the CX7 cable. So sparkrun guesses the wrong interface/IP at *every* layer:

| Layer | File | Symptom | Fix |
|---|---|---|---|
| 1. ray GCS node IP | `ray_head.sh`, `ray_worker.sh` | ray binds GCS to `192.168.0.74`; worker can't reach it; placement group hangs on "Waiting for creating a placement group" forever | prefer CX7 iface for `NODE_IP` |
| 2. torch rendezvous + container NODE_IP | `ib_detect.sh` (`DEFAULT_IF`) | feeds both `DETECTED_SOCKET_IFNAME` (GLOO/TP rendezvous) and `DETECTED_MGMT_IP` (→ container `NODE_IP`); TP init hangs *after* ray forms | prefer CX7 iface for `DEFAULT_IF` |
| 3. **vLLM distributed host IP** | `infiniband.py` | **the nasty one** — without `VLLM_HOST_IP`, vLLM workers auto-detect the OOB IP for the collective. NCCL builds all QPs over the correct socket, then **deadlocks at the first real data transfer** (idle GPUs, no error, looks like a plain hang) | set `VLLM_HOST_IP` = CX7 IP per-host |
| 4. eugr launcher | `launch-cluster.sh` | `HEAD_IP ∈ nodes` check fails when registered on CX7 host IPs | pin `LOCAL_IP` to CX7 |

`infiniband.py` also pins `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / the IB HCA
to the one CX7 device (belt-and-suspenders on top of the `ib_detect.sh` fix).

## Which CX7 port — not obvious

The box exposes **two** CX7 ports, both `ACTIVE`/`LINK_UP` and both routable over
TCP (`ping` succeeds on `192.168.1.x` *and* `192.168.100.x`). But only
**`rocep1s0f0` / `enp1s0f0np0` / `192.168.1.x`** actually carries RDMA data.
`roceP2p1s0f0` / `192.168.100.x` pings fine over TCP but its RoCE **data plane**
deadlocks NCCL mid-bootstrap (QPs create, first transfer hangs). All four patches
pin to `rocep1s0f0` — the link the homegrown cluster (`dashboard/cluster-launch.sh`)
has long proven serves.

## Debugging note

`NCCL_DEBUG=INFO` was the turning point: the "hang" wasn't a hang during NCCL
bootstrap — the log just goes silent there. With debug on you can see NCCL build
every QP and then stall at the first data transfer, which is what distinguishes
"slow bootstrap" (wait) from "data-plane deadlock" (wrong IP/port → fix). Idle
GPUs alone are NOT proof of a wedge during this phase.

## Upstream (worth a PR / issue to spark-arena)

On a node with **no shared management LAN** (only a direct RDMA link between
boxes), sparkrun's default-route auto-detection picks an unroutable interface at
every layer, and the missing `VLLM_HOST_IP` produces a silent NCCL deadlock that's
brutal to diagnose. A cluster-level option to declare the coordination
interface/HCA/IP explicitly — e.g. `--coordination-iface enp1s0f0np0`
`--rdma-hca rocep1s0f0` — would set `NODE_IP`, the socket ifnames, the IB HCA, and
`VLLM_HOST_IP` consistently and fix all four layers cleanly.
