#!/usr/bin/env bash
# Re-apply the joeai sparkrun patches after `sparkrun update` / a reinstall
# overwrites the package scripts + the eugr cache. Run from spark-1.
#
# WHY: sparkrun derives each ray node's IP from the DEFAULT-ROUTE interface
# (`ip route get 8.8.8.8`), which on the Sparks is the OOB `enP7s7`
# (192.168.0.x) — and that network does NOT route between the two boxes. Only the
# CX7 direct link (`enp1s0f0np0`, 192.168.1.x) does. So ray binds its GCS to
# 192.168.0.74, the worker can't reach it, and the placement group hangs forever.
# These patches pin ray's node IP to the CX7 interface. See README.md.
#
# The cluster must use the CX7 host IPs:
#   sparkrun cluster update my-cluster --hosts 192.168.1.74,192.168.1.49
# Idempotent: patch --forward skips an already-applied hunk.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Set these to your head node (where sparkrun is installed) and SSH key.
SPARK1_HOST="${SPARK1_HOST:-USER@192.168.1.11}"
SPARK_SSH_KEY="${SPARK_SSH_KEY:-$HOME/.ssh/id_ed25519}"
K=(-i "$SPARK_SSH_KEY" -o IdentitiesOnly=yes -o ConnectTimeout=10
   -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

apply() {  # $1 = remote dir (relative to $HOME), $2 = patch file
  local dir="$1" pf="$2" base; base="$(basename "$pf" .cx7.patch)"
  scp "${K[@]}" "$HERE/$pf" "$SPARK1_HOST:/tmp/$pf" >/dev/null
  ssh "${K[@]}" "$SPARK1_HOST" "cd \"\$HOME/$dir\" && patch --forward -p1 < /tmp/$pf >/dev/null 2>&1; \
    grep -q 'joeai patch' '$base' && echo '  $base OK' || echo '  $base FAILED'"
}

PKG=".local/share/uv/tools/sparkrun/lib/python3.12/site-packages/sparkrun/scripts"
ORCH=".local/share/uv/tools/sparkrun/lib/python3.12/site-packages/sparkrun/orchestration"
EUGR=".config/sparkrun/cache/eugr-spark-vllm-docker"

echo ">>> ray node-IP scripts (CX7 for ray GCS):"
apply "$PKG" ray_head.sh.cx7.patch
apply "$PKG" ray_worker.sh.cx7.patch
echo ">>> IB detect DEFAULT_IF (CX7 for socket iface AND the container NODE_IP vLLM advertises):"
apply "$PKG" ib_detect.sh.cx7.patch
echo ">>> NCCL/GLOO/TP socket + IB HCA pin + VLLM_HOST_IP (the deadlock fix):"
apply "$ORCH" infiniband.py.cx7.patch
echo ">>> eugr launcher (HEAD_IP-in-nodes check passes with CX7 host IPs):"
apply "$EUGR" launch-cluster.sh.cx7.patch
echo ">>> done. Verify the cluster uses CX7 host IPs (192.168.1.74,192.168.1.49)."
