#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Validate engram-disk-table.patch from scratch on one GB10 box.
#
# Fetches the three PR files at the pinned sha, applies the patch, drops the
# harness in beside them, and runs everything in a vLLM image on the GPU.
# The image predates PR #56214, so conftest.py registers the patched files
# under their real package names; see its docstring.
#
#   ./check_engram_disk.sh
#
# The container needs a real filesystem for the O_DIRECT case, so the work
# tree goes on the target box's NVMe. tmpfs rejects O_DIRECT at open.
#
# Set HOST and SPARK. The defaults are placeholders.
set -euo pipefail

SHA=e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba
HOST="${HOST:-${USER}@192.0.2.10}"   # the box you launch from; reaches $SPARK by ssh alias
SPARK="${SPARK:-spark-1}"
REMOTE=cc-scratch/engram-disk/check
IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
HERE=$(cd "$(dirname "$0")" && pwd)
SSH="ssh -o StrictHostKeyChecking=no"

TREE=$(mktemp -d)
trap 'rm -rf "$TREE"' EXIT
mkdir -p "$TREE"/vllm/config "$TREE"/vllm/models/deepseek_v4_1/common "$TREE"/tests/kernels

for f in tests/kernels/test_engram.py \
         vllm/models/deepseek_v4_1/common/engram.py \
         vllm/config/engram.py; do
  curl -fsS -o "$TREE/$f" "https://raw.githubusercontent.com/vllm-project/vllm/$SHA/$f"
done

git -C "$TREE" init -q .
git -C "$TREE" apply "$HERE/engram-disk-table.patch"
echo "patch applied to $SHA"
cp "$HERE"/conftest.py "$HERE"/engram_disk_harness.py "$HERE"/negative_control.py \
   "$HERE"/config_checks.py "$TREE/"

BUNDLE=$(base64 -w0 < <(tar czf - --exclude=.git -C "$TREE" .))
$SSH "$HOST" "$SSH -o BatchMode=yes $SPARK \"sudo rm -rf ~/$REMOTE && mkdir -p ~/$REMOTE && echo $BUNDLE | base64 -d | tar xzf - -C ~/$REMOTE\""

# Runs as the calling uid so the container leaves no root-owned files behind.
# No `set -e`: every step has to run, and the first pytest is expected to fail
# on one unrelated test. Each step echoes its exit code instead, because a
# script that dies half way still printed PASS lines above its traceback and
# the run reads as clean without them.
read -r -d '' SCRIPT <<'EOF' || true
set -x
cd /w && mkdir -p pytmp scratch
python3 -m pytest tests/kernels/test_engram.py --basetemp=/w/pytmp -q -p no:randomly -rf; echo "### rc=$? pytest all"
python3 -m pytest tests/kernels/test_engram.py --basetemp=/w/pytmp -p no:randomly -k disk -v; echo "### rc=$? pytest -k disk"
python3 engram_disk_harness.py /w/scratch; echo "### rc=$? engram_disk_harness"
python3 negative_control.py /w/scratch; echo "### rc=$? negative_control"
python3 config_checks.py; echo "### rc=$? config_checks"
EOF
CMD=$(echo "$SCRIPT" | base64 -w0)
$SSH "$HOST" "$SSH -o BatchMode=yes $SPARK \"echo $CMD | base64 -d > /tmp/engram_check.sh && chmod 644 /tmp/engram_check.sh && docker run --rm --gpus all --network host --user \\\$(id -u):\\\$(id -g) -e HOME=/w -e TRITON_CACHE_DIR=/w/.triton --entrypoint bash -v \\\$HOME/$REMOTE:/w -v /tmp/engram_check.sh:/cmd.sh -w /w $IMAGE /cmd.sh\""
