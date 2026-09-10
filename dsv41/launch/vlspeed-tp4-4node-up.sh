#!/usr/bin/env bash
# vlspeed-tp4-4node-up.sh: DeepSeek-V4.1-Flash on four DGX Sparks, with CUDA
# graphs. The serve is the one in docs/RECIPE.md. The differences are the image
# (vlspeed-eng:4 = vlpage-eng:3 + the top-k swap + the Engram prestage) and the
# graph flags below.
#
# EAGER=0, the default here, needs the prestage. Without it the disk Engram
# lookup is a host call inside the forward and engram.py refuses capture.
# See docs/cuda-graphs.md and docs/engram-prestage.md.
#
# CG_SIZES: with DSpark k, a decode batch is num_reqs*k draft tokens or
# num_reqs*(k+1) target tokens, so capturing every multiple of k and of k+1 up
# to MAXSEQS*(k+1) gives each batch an exact graph. A padded speculative batch
# can hang SM120 sparse MLA (FlashInfer #5015).
#
# IMAGE vlspeed-eng:4 is built by build/vl41-build-image.sh, then
# build/vlpage-build-image.sh, then build/vlspeed-build-image.sh. Layers, in
# order:
#   eugr/spark-vllm-b12x:latest  (torch 2.13.0+cu130, flashinfer 0.6.18, tilelang 0.1.12)
#   + the upstream aarch64 wheel at PR #56214's parent commit 29af8bd67
#   + the PR's 87 changed vllm/*.py            -> reproduces the PR head python tree
#   + engram-disk-table.patch                  -> Engram rows read from /table
#   + vl41_ops.so                              -> docs/op-shim-apply-q-norm.md
#   + vlpage-page64.py                         -> docs/page-size-64.md
#   + vlspeed-topk.py                          -> docs/topk-swap.md
#   + vlspeed-prestage.py                      -> docs/engram-prestage.md
# csrc and rust otherwise stay at the parent commit.
#
# Fabric conventions, all load-bearing: control plane on the LAN, NCCL on the f1
# CX7 HCAs, NCCL_IB_ROCE_VERSION_NUM=2, --ulimit nofile=1048576, workers before
# head. See ../cluster-fabric/README.md and ../sparkrun-cx7/README.md.
#
# NODE_TS and NODE_LAN below are RFC 5737 documentation ranges. Replace them.
#
# Usage:  vlspeed-tp4-4node-up.sh            # bring up
#         DRYRUN=1 vlspeed-tp4-4node-up.sh   # print per-rank scripts, change nothing
set -uo pipefail

IMAGE="${IMAGE:-vlspeed-eng:4}"
SNAP="${SNAP:-/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/df42c109f1defefcbfcedbe7d905718a12266e40}"
SERVED="${SERVED:-deepseek-v41-flash}"
PORT="${PORT:-8410}"
MASTER_PORT="${MASTER_PORT:-25410}"
NAME="${NAME:-vlspeed-tp4}"

TP="${TP:-4}"
CTX="${CTX:-16384}"
GPU_UTIL="${GPU_UTIL:-0.78}"
MAXSEQS="${MAXSEQS:-4}"
MAXBATCH="${MAXBATCH:-4096}"
EAGER="${EAGER:-0}"                 # 1 = --enforce-eager, skips graph capture
CGMODE="${CGMODE:-FULL_AND_PIECEWISE}"
CG_SIZES="${CG_SIZES:-}"            # comma list; empty = derive from DSPARK/MAXSEQS
VERIFY="${VERIFY:-0}"               # N = verify the first N engram lookups, eager only
PRESTAGE="${PRESTAGE:-1}"            # 0 = disable the prestage, for an A/B
TABLE_HOST="${TABLE_HOST:-$HOME/cc-scratch/vllm-v41/table}"
DISK_THREADS="${DISK_THREADS:-12}"
DISK_ODIRECT="${DISK_ODIRECT:-true}"
EXTRA="${EXTRA:-}"
# DSPARK=k enables DeepSeek-V4.1 speculative decoding. V4.1 has no classic
# MTP draft (config/speculative.py rejects method="mtp" for it), and one
# DSpark round drafts dspark_block_size=5 tokens, so k must be a multiple of 5.
DSPARK="${DSPARK:-}"

NODE_TS=(  192.0.2.2      192.0.2.1      192.0.2.3       192.0.2.4 )   # s2,s1,s3,s4
NODE_LAN=( 198.51.100.2   198.51.100.1   198.51.100.3    198.51.100.4 )
RANK_IDX="${RANK_IDX:-0 1 2 3}"; read -r -a R2I <<< "$RANK_IDX"
HEAD_DIST="${NODE_LAN[${R2I[0]}]}"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -i $SSH_KEY"
sshto()   { ssh $SSH_OPTS "${USER}@${NODE_TS[$1]}" "$2"; }
sshpipe() { ssh $SSH_OPTS "${USER}@${NODE_TS[$1]}" 'cat > /tmp/vlspeed-launch.sh && bash /tmp/vlspeed-launch.sh'; }

NCCL_MODE="-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_TC=104 -e NCCL_NET_GDR_LEVEL=5 \
 -e NCCL_CROSS_NIC=1 -e NCCL_NET_PLUGIN=none -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_MERGE_NICS=0 \
 -e NCCL_CUMEM_ENABLE=0 -e NCCL_WIN_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_IB_TIMEOUT=22 \
 -e NCCL_IB_RETRY_CNT=7 -e NCCL_SOCKET_IFNAME=enP7s7 -e GLOO_SOCKET_IFNAME=enP7s7 \
 -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=${NCCL_DEBUG:-WARN}"

ARCH_ENV="-e CUTE_DSL_ARCH=sm_121a -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

EAGER_ARG=""; GRAPH_ARG=""; GRAPH_ENV=""
if [ "$EAGER" = 1 ]; then
  EAGER_ARG="--enforce-eager"
else
  if [ -z "$CG_SIZES" ]; then
    if [ -n "$DSPARK" ]; then
      CG_SIZES=$( { seq "$DSPARK" "$DSPARK" $((DSPARK * MAXSEQS));
                    seq $((DSPARK + 1)) $((DSPARK + 1)) $(((DSPARK + 1) * MAXSEQS)); } \
                  | sort -n -u | paste -sd, - )
    else
      CG_SIZES=$(seq 1 "$MAXSEQS" | paste -sd, -)
    fi
  fi
  GRAPH_ARG='--compilation-config "{\"cudagraph_mode\":\"'"$CGMODE"'\",\"cudagraph_capture_sizes\":['"$CG_SIZES"']}"'
  # The eager-break decorator binds when the model imports, so this has to be
  # in the environment of every rank, not just the head.
  GRAPH_ENV="-e VLLM_USE_BREAKABLE_CUDAGRAPH=1"
fi
VERIFY_ENV="-e VL41_ENGRAM_PRESTAGE_VERIFY=$VERIFY -e VL41_ENGRAM_PRESTAGE=$PRESTAGE"
SPEC_ARG=""
# enable_adaptive_verification stays off: it forces varlen decode graphs, whose
# padded rows are the FlashInfer #5015 trigger on SM120 sparse MLA.
[ -n "$DSPARK" ] && SPEC_ARG='--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":'"$DSPARK"',\"enable_adaptive_verification\":false}"'

runscript() {
  local r="$1" idx="${R2I[$1]}" hl=""
  [ "$r" != 0 ] && hl="--headless"
  cat <<EOF
docker rm -f $NAME >/dev/null 2>&1 || true
mkdir -p $TABLE_HOST
docker run -d --name $NAME --network host --ipc host --shm-size 32g --gpus all \\
  --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \\
  --device /dev/infiniband:/dev/infiniband --restart no --init \\
  -v \$HOME/.cache/huggingface:/cache/huggingface \\
  -v $TABLE_HOST:/table \\
  -v \$HOME/cc-scratch/vllm-v41/cache:/cache \\
  -v \$HOME/cc-scratch/vllm-v41/tlcache:/root/.tilelang \\
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/vllm-cache \\
  -e VLLM_HOST_IP=${NODE_LAN[$idx]} -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \\
  $NCCL_MODE $ARCH_ENV $GRAPH_ENV $VERIFY_ENV \\
  -e VLLM_USE_FLASHINFER_SAMPLER=0 -e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 \\
  -e TRITON_CACHE_DIR=/cache/triton \\
  --entrypoint /bin/bash $IMAGE -lc '
    exec vllm serve $SNAP \\
      --served-model-name $SERVED \\
      --host 0.0.0.0 --port $PORT \\
      --tensor-parallel-size $TP \\
      --gpu-memory-utilization $GPU_UTIL \\
      --max-model-len $CTX --max-num-seqs $MAXSEQS --max-num-batched-tokens $MAXBATCH \\
      --engram-config "{\\"table_path\\":\\"/table\\",\\"disk_read_threads\\":$DISK_THREADS,\\"disk_direct_io\\":$DISK_ODIRECT}" \\
      $SPEC_ARG \\
      --tokenizer-mode deepseek_v41 \\
      $EAGER_ARG $GRAPH_ARG --enable-chunked-prefill \\
      --distributed-executor-backend mp \\
      --nnodes 4 --node-rank $r --master-addr $HEAD_DIST --master-port $MASTER_PORT $hl $EXTRA
  '
EOF
}

if [ "${DRYRUN:-0}" = 1 ]; then
  echo "### DRYRUN IMAGE=$IMAGE TP=$TP CTX=$CTX UTIL=$GPU_UTIL EAGER=$EAGER CG=$CGMODE[$CG_SIZES] DSPARK=$DSPARK"
  for r in 0 1 2 3; do echo "===== rank $r -> ${NODE_TS[${R2I[$r]}]} ====="; runscript "$r"; echo; done
  exit 0
fi

echo ">>> [1/3] clearing $NAME on all 4 boxes (image=$IMAGE eager=$EAGER cg=$CGMODE[$CG_SIZES] dspark=$DSPARK ctx=$CTX seqs=$MAXSEQS)"
for i in 0 1 2 3; do sshto "$i" "docker rm -f $NAME >/dev/null 2>&1 || true"; done
sleep 2
echo ">>> [2/3] starting workers (3,2,1) then head (0 -> ${NODE_TS[${R2I[0]}]})"
for r in 3 2 1; do echo "    rank $r -> ${NODE_TS[${R2I[$r]}]}"; runscript "$r" | sshpipe "${R2I[$r]}"; sleep 4; done
echo "    rank 0 (head) -> ${NODE_TS[${R2I[0]}]}"; runscript 0 | sshpipe "${R2I[0]}"
echo ">>> [3/3] polling head:$PORT"
for i in $(seq 1 120); do
  sleep 15
  if sshto 0 "curl -s --max-time 4 localhost:$PORT/v1/models 2>/dev/null" | grep -q "$SERVED"; then
    echo ">>> OK: $SERVED UP after ~$((i*15))s"; exit 0
  fi
  hs=$(sshto 0 "docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null" 2>/dev/null || true)
  [ "$hs" = exited ] && { echo ">>> FAIL: head exited at ~$((i*15))s"; exit 1; }
done
echo ">>> FAIL: timed out (~30min)"; exit 2
