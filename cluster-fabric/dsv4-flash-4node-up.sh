#!/usr/bin/env bash
# DeepSeek-V4-Flash on the 4-Spark switchless RoCE RING (or mesh) via native vLLM multi-node.
# Experiment rig: measure TP=4 vs PP=2/TP=2 (and ring vs mesh) on DSV4-Flash against the
# 2-Spark baseline (41 tok/s, MTP n=2, fp8 KV, FULL_DECODE_ONLY — tests/eval/scoreboard.md).
#
# MODEL flags = dsv4-flash-compose.yaml (deepseek_v4 tokenizer/tool/reasoning parsers, MTP n=2,
# fp8 KV, block-size 256, FULL_DECODE_ONLY cudagraphs, prefix-cache, flashinfer-autotune,
# thinking=high). FABRIC = the MM3 4-node ring skeleton (mm3-vllm-4node-up.sh): patched libnccl
# LD_PRELOAD + NCCL_ALGO=Ring + NCCL_SKIP_TREE_CONNECT=1 (skip the uncabled diagonals), control
# plane on the 10GbE LAN, NCCL data on the 200G RoCE ring, --disable-custom-all-reduce so the
# cross-node all-reduce rides pynccl(->patched->ring), not a torch NCCL PG(->bundled->hang).
#
# PATCHED NCCL: $HOME/nccl-patched/libnccl.so.2 (glibc 2.39) loads in the DSV4 image
# (Debian-trixie / glibc 2.41, forward-compat). NOT the -u22 build (that's for the Ubuntu-22 MM3 image).
#
# RING CABLING (physical loop): spark-2 <-> spark-1 <-> spark-3 <-> spark-4 <-> spark-2.
# Diagonals s2<->s3 and s1<->s4 have NO cable (skip-tree-connect avoids needing them).
#
# PP=2/TP=2 RANK TRICK (RANK_IDX "0 1 3 2" = box order s2,s1,s4,s3): vLLM's TP-inner layout
# puts TP groups {r0,r1}={s2,s1} and {r2,r3}={s4,s3} on ring edges, AND PP groups {r0,r2}={s2,s4}
# and {r1,r3}={s1,s3} ALSO on ring edges -- so every heavy TP all-reduce and both PP send/recv
# hops ride a 200G neighbor link, zero diagonal/relay. TP=4 uses the plain ring order s2,s1,s3,s4
# (the ring all-reduce walks the loop neighbor-to-neighbor).
#
# Toggles (env): TP=4 PP=1 | TP=2 PP=2 ; NET=ring|mesh|switch ; CTX=262144 ; GPU_UTIL=0.78 ;
#   MTP=1 ; EAGER=0 ; MAXSEQS=4 ; MAXBATCH=8192 ; RANK_IDX=auto
#   NET=switch = the CRS812 switched mesh (f1 HCAs, stock-NCCL Tree, no relay) — the live topology.
#   ring/mesh assume the OLD switchless cabling (relay for the uncabled diagonals) and are stale.
#
# Worker-first launch (ranks 3,2,1 --headless) then head (rank 0 = spark-2 serves :8000),
# reached by clients via spark-1:8013 -> spark-2:8000 (vllm-cluster-proxy), same path as the DD.
set -euo pipefail
export PATH="/snap/bin:$HOME/.local/bin:$PATH"

IMAGE="${IMAGE:-dsv4-vllm:patched-42879}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
SERVED="${SERVED:-deepseek-v4-flash}"
PATCHED_NCCL="${PATCHED_NCCL:-$HOME/nccl-patched/libnccl.so.2}"
MESH_PLUGIN="${MESH_PLUGIN:-$HOME/mesh-plugin/libnccl-net.so}"
PORT=8000
MASTER_PORT="${MASTER_PORT:-25000}"

# ── node table (table-index 0..3 = s2,s1,s3,s4; rank 0 must land on s2 = head serving :8000) ─
NODE_TS=(  192.0.2.2   192.0.2.1  192.0.2.3    192.0.2.4 )   # tailnet (ssh launch): s2,s1,s3,s4
NODE_LAN=( 198.51.100.2   198.51.100.1   198.51.100.3    198.51.100.4 )   # 10GbE LAN: dist-init + control + VLLM_HOST_IP
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -i $SSH_KEY"
sshto()   { ssh $SSH_OPTS "${USER}@${NODE_TS[$1]}" "$2"; }                                  # run a cmd on table-index $1
sshpipe() { ssh $SSH_OPTS "${USER}@${NODE_TS[$1]}" 'cat > /tmp/dsv4-launch.sh && bash /tmp/dsv4-launch.sh'; }  # stdin -> remote script

# DD defaults (4-node): 512k context + util 0.70. GROWN 2026-06-28 from 0.55 for a bigger KV cache.
# The KV pool is the slack above a ~0.36 floor (155 GiB weights /4 + ~10 GiB/node engine overhead) at
# ~49 KiB/token (MLA latent + fp8 KV over 61 layers): 0.70 gives ~174 GiB aggregate KV pool ≈ 3.7M
# tokens ≈ ~7× full-512k contexts (0.55 was ~98 GiB ≈ 2.1M ≈ 4×). The CEILING is spark-1, the shared
# services box (cc-tmux fleet + fleet-dashboard + playwright MCP + socat proxies all run there): util 0.85
# OOMs its GB10 unified memory (NVRM NV_ERR_NO_MEMORY -> rank-1 worker dies -> head hangs, 2026-06-28).
# 0.70 sits below the historically-safe 0.78 but ~30 GiB tighter on spark-1 than 0.55 — VERIFY spark-1
# survives capture on restart (fall back to 0.65/0.60 if it OOMs). TP needs every rank so the weakest box
# sets the ceiling.
# Override CTX/GPU_UTIL per experiment.
NET="${NET:-ring}"; TP="${TP:-4}"; PP="${PP:-1}"; CTX="${CTX:-524288}"; GPU_UTIL="${GPU_UTIL:-0.70}"
MTP="${MTP:-1}"; EAGER="${EAGER:-0}"; MAXSEQS="${MAXSEQS:-4}"; MAXBATCH="${MAXBATCH:-8192}"; PROFILE="${PROFILE:-0}"
NET_IFACE="${NET_IFACE:-enP7s7}"; DBG="${NCCL_DEBUG:-WARN}"
HEAD_DIST="${NODE_LAN[0]}"   # control-plane rendezvous over 10GbE LAN (s2)

# rank -> node-table-index. PP=2/TP=2 ring gets the diagonal-free swizzle; else plain ring order.
if [ "${RANK_IDX:-auto}" = auto ]; then
  if [ "$TP" = 2 ] && [ "$PP" = 2 ]; then RANK_IDX="0 1 3 2"; else RANK_IDX="0 1 2 3"; fi
fi
read -r -a R2I <<< "$RANK_IDX"   # R2I[rank] = index into NODE_TS / NODE_LAN

# ── fabric env per NET mode ─
if [ "$NET" = ring ]; then
  NCCL_MODE="-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=${IB_HCA:-rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1} -e NCCL_IB_GID_INDEX=3 -e NCCL_NET_GDR_LEVEL=5 -e NCCL_CROSS_NIC=1 -e NCCL_ALGO=${ALGO:-Ring} -e NCCL_NET_PLUGIN=none -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_WIN_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_SKIP_TREE_CONNECT=1 -e NCCL_IB_TIMEOUT=${IB_TIMEOUT:-18} -e NCCL_IB_RETRY_CNT=${IB_RETRY:-7} -e NCCL_SOCKET_IFNAME=$NET_IFACE -e GLOO_SOCKET_IFNAME=$NET_IFACE -e NCCL_DEBUG=$DBG"
  NCCL_SO_MOUNT="-v $PATCHED_NCCL:/opt/nccl-patched/libnccl.so.2:ro"
  NCCL_LD="-e LD_PRELOAD=/opt/nccl-patched/libnccl.so.2 -e VLLM_NCCL_SO_PATH=/opt/nccl-patched/libnccl.so.2"
  MESH_MOUNT=""
elif [ "$NET" = mesh ]; then
  # "mesh" = the PATCHED libnccl's RELAY overlay (NOT the SGLANG nccl-mesh-plugin — that uses stock
  # NCCL as the NET provider and err-110s on the uncabled diagonal, 2026-06-23). Same patched .so as
  # the ring path (proven to load in the trixie image, run A), but NCCL_RELAY_ENABLE=1 turns the
  # relay ON (run A logged it OFF) so the 2-hop diagonal relay provides the s2<->s3 / s1<->s4 links,
  # letting NCCL run Tree across all 4 instead of ring-only. Proven vLLM mechanism: mm3-vllm-chthonic.
  # ALGO is UNSET by default = NCCL auto-picks Ring/Tree per op (the relay provides the diagonal so Tree
  # is available for the small latency-bound decode all-reduces, Ring for the rest). Do NOT force
  # NCCL_ALGO=Tree — vLLM issues an int8 AllGather that Tree can't service ("No algorithm/protocol
  # available for AllGather ncclInt8", 2026-06-23), and forcing Tree removes the Ring fallback -> crash.
  # Override ALGO=Ring for relay-on-ring (≈ ring, diagonal unused). MAXNCH caps channels if the relay
  # deadlocks under too many legs. SKIP_TREE=1 skips the direct diagonal QP if it's attempted pre-relay.
  NCCL_MODE="-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=${IB_HCA:-rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1} -e NCCL_IB_GID_INDEX=3 -e NCCL_NET_GDR_LEVEL=5 -e NCCL_CROSS_NIC=1 ${ALGO:+-e NCCL_ALGO=$ALGO} -e NCCL_NET_PLUGIN=none -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_WIN_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_RELAY_ENABLE=1 ${RELAY_DIAG:+-e NCCL_RELAY_DIAG=1} ${SKIP_TREE:+-e NCCL_SKIP_TREE_CONNECT=1} ${MAXNCH:+-e NCCL_MAX_NCHANNELS=$MAXNCH -e NCCL_MIN_NCHANNELS=$MAXNCH} -e NCCL_IB_TIMEOUT=${IB_TIMEOUT:-18} -e NCCL_IB_RETRY_CNT=${IB_RETRY:-7} -e NCCL_SOCKET_IFNAME=$NET_IFACE -e GLOO_SOCKET_IFNAME=$NET_IFACE -e NCCL_DEBUG=$DBG"
  NCCL_SO_MOUNT="-v $PATCHED_NCCL:/opt/nccl-patched/libnccl.so.2:ro"
  NCCL_LD="-e LD_PRELOAD=/opt/nccl-patched/libnccl.so.2 -e VLLM_NCCL_SO_PATH=/opt/nccl-patched/libnccl.so.2"
  MESH_MOUNT=""
elif [ "$NET" = switch ]; then
  # The CRS812 SWITCHED mesh (post-2026-06-27 cutover): every box -> switch, so ALL pairs are directly
  # reachable — the ring/relay paths above are OBSOLETE here (no uncabled diagonals to bridge). Minimal
  # diff from NET=ring: same patched .so (known to load in the trixie image), but (a) f1-cage HCAs ONLY
  # (the switch cabling; the f0 cage is DARK — banding it would hang on dead links), (b) RELAY stays OFF
  # (default; dormant with every link direct), (c) DROP NCCL_ALGO=Ring -> auto so NCCL picks Tree for the
  # small latency-bound decode all-reduce, (d) DROP NCCL_SKIP_TREE_CONNECT (we WANT the tree the switch
  # supports), (e) NCCL_IB_TC=104 = the RoCE PFC class the host+switch QoS is set for. Mirrors the
  # sglang-mesh + mm3-chthonic env that logged "Connected binomial trees" on this fabric. Bring up with
  # NCCL_DEBUG=INFO once to confirm Tree + direct f1 links before trusting the number.
  NCCL_MODE="-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=${IB_HCA:-rocep1s0f1,roceP2p1s0f1} -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_TC=${IB_TC:-104} -e NCCL_NET_GDR_LEVEL=5 -e NCCL_CROSS_NIC=1 ${ALGO:+-e NCCL_ALGO=$ALGO} -e NCCL_NET_PLUGIN=none -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_WIN_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 ${MAXNCH:+-e NCCL_MAX_NCHANNELS=$MAXNCH -e NCCL_MIN_NCHANNELS=$MAXNCH} -e NCCL_IB_TIMEOUT=${IB_TIMEOUT:-18} -e NCCL_IB_RETRY_CNT=${IB_RETRY:-7} -e NCCL_SOCKET_IFNAME=$NET_IFACE -e GLOO_SOCKET_IFNAME=$NET_IFACE -e NCCL_DEBUG=$DBG"
  NCCL_SO_MOUNT="-v $PATCHED_NCCL:/opt/nccl-patched/libnccl.so.2:ro"
  NCCL_LD="-e LD_PRELOAD=/opt/nccl-patched/libnccl.so.2 -e VLLM_NCCL_SO_PATH=/opt/nccl-patched/libnccl.so.2"
  MESH_MOUNT=""
else echo "bad NET=$NET (want ring|mesh|switch)"; exit 2; fi

# spec-decode (MTP n=2) + cudagraph flags. JSON is SINGLE-quoted so the remote bash (running the
# piped /tmp/dsv4-launch.sh) strips the single quotes and hands docker intact double-quoted JSON.
SPEC=""; [ "$MTP" = 1 ] && SPEC="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'"
if [ "$EAGER" = 1 ]; then GRAPH="--enforce-eager"; else GRAPH="--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'"; fi

# DSPARK=1: serve DeepSeek-V4-Flash-DSpark with our out-of-tree DSparkProposer
# (method=custom_class). The dspark package lives at $HOME/dspark-pkg/dspark
# on each box (distribute it there before launch); mounted ro + on PYTHONPATH.
DSPARK="${DSPARK:-0}"; DSPARK_MOUNT=""; DSPARK_ENV=""
if [ "$DSPARK" = 1 ]; then
  MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash-DSpark}"
  [ "$MODEL" = "deepseek-ai/DeepSeek-V4-Flash" ] && MODEL="deepseek-ai/DeepSeek-V4-Flash-DSpark"
  SPEC="--speculative-config '{\"method\":\"custom_class\",\"model\":\"dspark.proposer.DSparkProposer\",\"num_speculative_tokens\":${DSPARK_NSPEC:-5}}'"
  DSPARK_MOUNT="-v $HOME/dspark-pkg:/opt/dspark:ro"
  # DSPARK_GRAPH=0 disables the draft-cudagraph capture (eager 3-stage draft);
  # default 1 graphs the draft cascade. Toggle here for debug without a rebuild.
  DSPARK_ENV="-e PYTHONPATH=/opt/dspark -e DSPARK_GRAPH=${DSPARK_GRAPH:-1} -e DSPARK_TIMING=${DSPARK_TIMING:-0} -e DSPARK_DIV=${DSPARK_DIV:-0} -e DSPARK_TRACE=${DSPARK_TRACE:-0} -e DSPARK_TRACE_MAX=${DSPARK_TRACE_MAX:-600}"
fi

# DSML tool/reasoning parser patches (the streaming DSML tool-call-leak fix:
# deepseekv32 tool parser headless/opener repair + dsv4 tool-aware reasoning
# split). Baked into the image too, but mounting the repo copies makes the repo
# the source of truth and lets us iterate the parser fix WITHOUT an image
# rebuild. Files distributed to $DSML_DIR on each host (distribute before
# launch). DSML_TRACE=1 adds per-delta parser tracing to stderr.
DSML_DIR="${DSML_DIR:-$HOME/dsml-patch}"
VLLM_SP="/opt/env/lib/python3.12/site-packages/vllm"
DSML_MOUNT="-v $DSML_DIR/deepseekv32_tool_parser_patched.py:$VLLM_SP/tool_parsers/deepseekv32_tool_parser.py:ro -v $DSML_DIR/dsv4-reasoning-tool-aware.py:$VLLM_SP/reasoning/deepseek_v3_reasoning_parser.py:ro -v $DSML_DIR/dsv4_dsml_split.py:$VLLM_SP/reasoning/dsv4_dsml_split.py:ro"
DSML_TRACE_ENV="-e DSML_TRACE=${DSML_TRACE:-0}"
# PROFILE=1: enable vLLM's torch profiler via --profiler-config FLAGS (this build IGNORES the
# VLLM_TORCH_PROFILER_DIR env — it routes profiling through profiler_config). Idle until the /start_profile
# endpoint is hit; traces dump to the mounted HF cache -> host ~/.cache/huggingface/torchprof per rank.
# For measuring the decode-step heatmap (attention vs expert GEMM vs all-reduce vs norm/sampling).
PROF_FLAGS=""; [ "$PROFILE" = 1 ] && PROF_FLAGS="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/cache/huggingface/torchprof"

runscript() {  # $1 = node_rank ; emits the full remote launch script (rm + docker run) on stdout
  local r="$1" idx="${R2I[$1]}" hl=""
  [ "$r" != 0 ] && hl="--headless"
  cat <<EOF
docker rm -f dsv4-vllm mm3-vllm mm3-sglang dsv4-sglang >/dev/null 2>&1 || true
docker run -d --name dsv4-vllm --network host --ipc host --shm-size 10g --gpus all \\
  --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \\
  --device /dev/infiniband:/dev/infiniband --restart no \\
  -v \${HF_CACHE:-\$HOME/.cache/huggingface}:/cache/huggingface \\
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \\
  $NCCL_SO_MOUNT $MESH_MOUNT $DSPARK_MOUNT $DSML_MOUNT \\
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache $DSPARK_ENV $DSML_TRACE_ENV \\
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \\
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e TORCHINDUCTOR_COMPILE_THREADS=1 \\
  -e VLLM_HOST_IP=${NODE_LAN[$idx]} \\
  $NCCL_LD $NCCL_MODE -e NCCL_NVLS_ENABLE=0 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \\
  $IMAGE /usr/local/bin/dsv4-vllm-entrypoint serve $MODEL \\
    --served-model-name $SERVED --host 0.0.0.0 --port $PORT --trust-remote-code \\
    --tensor-parallel-size $TP --pipeline-parallel-size $PP --kv-cache-dtype fp8 --block-size 256 \\
    --max-model-len $CTX --max-num-seqs $MAXSEQS --max-num-batched-tokens $MAXBATCH --gpu-memory-utilization $GPU_UTIL \\
    --tokenizer-mode deepseek_v4 --distributed-executor-backend mp \\
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \\
    --default-chat-template-kwargs.thinking=true --default-chat-template-kwargs.reasoning_effort=high \\
    --enable-prefix-caching --disable-custom-all-reduce \\
    $SPEC --enable-flashinfer-autotune $GRAPH $PROF_FLAGS \\
    --nnodes 4 --node-rank $r --master-addr $HEAD_DIST --master-port $MASTER_PORT $hl
EOF
}

# DRYRUN=1 prints the generated per-rank launch script (no ssh, no teardown) and exits.
if [ "${DRYRUN:-0}" = 1 ]; then
  echo "### DRYRUN  NET=$NET TP=$TP PP=$PP CTX=$CTX MTP=$MTP EAGER=$EAGER  RANK_IDX='$RANK_IDX'"
  for r in 0 1 2 3; do echo "===== rank $r  ->  ${NODE_TS[${R2I[$r]}]} (LAN ${NODE_LAN[${R2I[$r]}]}) ====="; runscript "$r"; echo; done
  exit 0
fi

echo ">>> [1/3] freeing all 4 boxes (drop DD + stale dsv4)"
for i in 0 1 2 3; do sshto "$i" 'docker rm -f dsv4-vllm mm3-vllm mm3-sglang dsv4-sglang >/dev/null 2>&1 || true'; done
sleep 2

echo ">>> [2/3] NET=$NET TP=$TP PP=$PP CTX=$CTX MTP=$MTP EAGER=$EAGER  rank->box: $(for r in 0 1 2 3; do printf 'r%s=s%s ' "$r" "$(h=${NODE_TS[${R2I[$r]}]}; case $h in *.27)echo 2;; *.128)echo 1;; *.94)echo 3;; *.72)echo 4;; esac)"; done)"
echo ">>> starting workers (ranks 3,2,1 --headless) then head (rank 0 = s2)…"
for r in 3 2 1; do echo "    rank $r -> ${NODE_TS[${R2I[$r]}]}"; runscript "$r" | sshpipe "${R2I[$r]}"; sleep 4; done
echo "    rank 0 (head) -> ${NODE_TS[${R2I[0]}]}"; runscript 0 | sshpipe "${R2I[0]}"

echo ">>> [3/3] waiting for serve on head=s2:$PORT (149G load /4 across nodes + engine init + graph capture)…"
for i in $(seq 1 90); do
  sleep 12
  if sshto 0 "curl -s --max-time 4 localhost:$PORT/v1/models 2>/dev/null" | grep -q "$SERVED"; then
    echo ">>> OK: DeepSeek-V4-Flash UP after ~$((i*12))s (NET=$NET TP=$TP PP=$PP CTX=$CTX MTP=$MTP). On :8013 socat (OpenCode/LibreChat path)."
    exit 0
  fi
  hs=$(sshto 0 "docker inspect -f '{{.State.Status}}' dsv4-vllm 2>/dev/null" 2>/dev/null || true)
  [ "$hs" = exited ] && { echo ">>> FAIL: head container exited. Logs: ssh s2 docker logs dsv4-vllm"; exit 1; }
done
echo ">>> FAIL: timed out (~18min) waiting for :$PORT. Logs: ssh s2 docker logs dsv4-vllm (+ each worker)"; exit 2
