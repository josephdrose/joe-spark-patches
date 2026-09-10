#!/usr/bin/env bash
# vlspeed-build-image.sh: build vlspeed-eng:4 on all four Sparks.
#
# vlspeed-eng:4 = vlpage-eng:3 (see vlpage-build-image.sh)
#   + vlspeed-topk.py      -> sm12x indexer decode top-k off persistent_topk
#   + vlspeed-prestage.py  -> disk Engram rows staged before the forward
# Together those are what CUDA graphs need: the top-k kernel that does not die
# on GB10, and a forward with no host call in it. Python only; nothing is
# compiled here, so the build is about a second per box.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WORK="${WORK:-$HOME/cc-scratch/vlspeed}"
BOXES=(spark-1 spark-2 spark-3 spark-4)
SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes"

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cp "$HERE/vlspeed-topk.py" "$HERE/vlspeed-prestage.py" "$tmp/"
cat > "$tmp/Dockerfile" <<'DOCK'
FROM vlpage-eng:3
COPY vlspeed-topk.py vlspeed-prestage.py /opt/vl41/
RUN python3 /opt/vl41/vlspeed-topk.py && python3 /opt/vl41/vlspeed-prestage.py
DOCK
tar czf "$tmp/ctx.tgz" -C "$tmp" vlspeed-topk.py vlspeed-prestage.py Dockerfile

for h in "${BOXES[@]}"; do
  echo ">>> $h"
  ssh $SSH_OPTS "$h" "mkdir -p $WORK"
  scp -q $SSH_OPTS "$tmp/ctx.tgz" "$h:$WORK/"
  ssh $SSH_OPTS "$h" "rm -rf $WORK/b && mkdir -p $WORK/b && cd $WORK/b && tar xzf ../ctx.tgz && docker build -t vlspeed-eng:4 . | tail -3"
done
