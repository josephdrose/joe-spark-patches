#!/usr/bin/env bash
# vlpage-build-image.sh — build vlpage-eng:3 on all four Sparks.
#
# vlpage-eng:3 = vl41-eng:2 (see vl41-build-image.sh) + vlpage-page64.py, which
# puts every DeepSeek-V4.1 KV page on 64 states so the FlashInfer sm120
# sparse-MLA kernels and DeepGEMM's paged MQA logits both accept it.
# Python only; no kernel is compiled here. Build is about 1 second per box.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WORK="${WORK:-$HOME/cc-scratch/vl41-page}"
BOXES=(spark-1 spark-2 spark-3 spark-4)
SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes"

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cp "$HERE/vlpage-page64.py" "$tmp/"
cat > "$tmp/Dockerfile" <<'DOCK'
FROM vl41-eng:2
COPY vlpage-page64.py /opt/vl41/vlpage-page64.py
RUN python3 /opt/vl41/vlpage-page64.py
DOCK
tar czf "$tmp/ctx.tgz" -C "$tmp" vlpage-page64.py Dockerfile

for h in "${BOXES[@]}"; do
  echo ">>> $h"
  ssh $SSH_OPTS "$h" "mkdir -p $WORK/ctx"
  scp -q $SSH_OPTS "$tmp/ctx.tgz" "$h:$WORK/"
  ssh $SSH_OPTS "$h" "rm -rf $WORK/ctx && mkdir -p $WORK/ctx && cd $WORK/ctx && tar xzf ../ctx.tgz && docker build -t vlpage-eng:3 . | tail -2"
done
