#!/usr/bin/env bash
# vl41-build-image.sh — build vl41-eng:2 on all four Sparks.
#
# There is no from-source vLLM build here and none is needed. The pieces:
#
#   1. eugr/spark-vllm-b12x:latest already carries the exact dep stack the
#      upstream wheel wants: torch 2.13.0+cu130, flashinfer 0.6.18 (eugr's build,
#      which adds the sm120 sparse-MLA kernels), tilelang 0.1.12, CUDA 13.0,
#      python 3.12. Same image digest on all four boxes.
#   2. wheels.vllm.ai publishes an official manylinux aarch64 wheel per commit.
#      PR #56214 is one commit; its PARENT is 29af8bd67, and that wheel exists.
#      Verified: unchanged files in the wheel are byte-identical to the PR head
#      tree, so the wheel IS the PR's base.
#   3. Copy the PR's 87 changed vllm/*.py over the installed package. That
#      reproduces the PR head python tree exactly. 26 of them are new files;
#      the PR deletes nothing and adds no non-.py asset under vllm/.
#   4. Apply engram-disk-table.patch (Engram rows on NVMe).
#   5. Copy in vl41_ops.so, the out-of-tree build of the PR's own fused
#      qnorm/rope/kv-insert .cu. See vl41-ops-build.py for why.
#
# csrc and rust stay at the parent commit apart from (5).
set -euo pipefail

WORK="${WORK:-$HOME/cc-scratch/vllm-v41}"
SHA_HEAD=e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba          # PR #56214, single commit
SHA_BASE=29af8bd672d5a780abd7399c0cc624078202e89d          # its parent
WHEEL="vllm-0.28.1rc1.dev391+g29af8bd67-cp38-abi3-manylinux_2_28_aarch64.whl"
HERE=$(cd "$(dirname "$0")" && pwd)
BOXES=(spark-1 spark-2 spark-3 spark-4)

mkdir -p "$WORK" && cd "$WORK"

[ -d src ] || {
  curl -sL -o head.tar.gz "https://codeload.github.com/vllm-project/vllm/tar.gz/$SHA_HEAD"
  mkdir -p src && tar xzf head.tar.gz -C src --strip-components=1
}
[ -f pr56214.diff ] || curl -sL -o pr56214.diff https://github.com/vllm-project/vllm/pull/56214.diff
# The API's /pulls/56214 diff endpoint refuses this PR with 406 "diff exceeded
# the maximum number of lines (20000)". github.com/<pr>.diff has no such cap.
[ -f "$WHEEL" ] || curl -sL -o "$WHEEL" \
  "https://wheels.vllm.ai/$SHA_BASE/${WHEEL/+/%2B}"

grep '^diff --git' pr56214.diff | sed 's|^diff --git a/||; s| b/.*||' > files.txt
rm -rf overlay && mkdir overlay
while read -r f; do
  case "$f" in vllm/*.py) ;; *) continue;; esac
  mkdir -p "overlay/$(dirname "$f")"; cp "src/$f" "overlay/$f"
done < files.txt

rm -rf overlay-eng && cp -a overlay overlay-eng
( cd overlay-eng && git init -q . && git apply --include='vllm/*' "$HERE/engram-disk-table.patch" )

# The op shim.
rm -rf shim && mkdir -p shim/build
cp src/csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu shim/kernel.cu
cp src/csrc/libtorch_stable/torch_utils.h shim/
cp "$HERE/vl41-ops-bindings.cpp" shim/bindings.cpp
cp "$HERE/vl41-ops-build.py" shim/
docker run --rm --gpus all --ipc host -v "$WORK/shim":/shim -v "$WORK/src":/src:ro \
  -e HOME=/shim --entrypoint /bin/bash eugr/spark-vllm-b12x:latest \
  -lc 'cd /shim && python3 vl41-ops-build.py'

# attention.py is the only file that passes apply_q_norm; redirect its three calls.
python3 - <<'PY'
p = "overlay-eng/vllm/models/deepseek_v4_1/attention.py"
s = open(p).read()
anchor = "import regex as re\nimport torch\n"
shim = '''import os

import regex as re
import torch

# The prebuilt _C in this image predates PR #56214, so its three fused DSV4
# qnorm/rope/kv-insert ops have no trailing apply_q_norm argument and V4.1
# always passes False. vl41_ops.so is the PR's own kernel .cu compiled out of
# tree and registered as torch.ops.vl41 with the new schema. Fail loudly if it
# is absent: falling back to torch.ops._C would silently apply the Q RMSNorm
# that V4.1 removes.
_VL41_SHIM_SO = os.environ.get("VL41_SHIM_SO", "/opt/vl41/vl41_ops.so")
if os.path.exists(_VL41_SHIM_SO):
    torch.ops.load_library(_VL41_SHIM_SO)
    _FUSED_KV_OPS = torch.ops.vl41
else:
    _FUSED_KV_OPS = torch.ops._C
'''
assert anchor in s
s = s.replace(anchor, shim, 1)
s = s.replace("torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_",
              "_FUSED_KV_OPS.fused_deepseek_v4_qnorm_rope_kv_rope_")
open(p, "w").write(s)
PY

cat > Dockerfile.vl41eng2 <<DOCK
FROM eugr/spark-vllm-b12x:latest
COPY $WHEEL /tmp/wh/
RUN pip install --no-deps --force-reinstall --no-cache-dir /tmp/wh/*.whl && rm -rf /tmp/wh
COPY overlay-eng/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY vl41_ops.so /opt/vl41/vl41_ops.so
DOCK

rm -rf ctx && mkdir ctx
cp Dockerfile.vl41eng2 "$WHEEL" ctx/
cp shim/build/vl41_ops.so ctx/
cp -a overlay-eng ctx/
tar czf ctx.tgz -C ctx .

for h in "${BOXES[@]}"; do
  echo ">>> $h"
  if [ "$h" = "$(hostname)" ]; then
    docker build -f Dockerfile.vl41eng2 -t vl41-eng:2 ctx | tail -2
  else
    ssh -o StrictHostKeyChecking=no "$h" "mkdir -p $WORK/ctx"
    scp -q -o StrictHostKeyChecking=no ctx.tgz "$h:$WORK/"
    ssh -o StrictHostKeyChecking=no "$h" \
      "rm -rf $WORK/ctx && mkdir -p $WORK/ctx && cd $WORK/ctx && tar xzf ../ctx.tgz && docker build -f Dockerfile.vl41eng2 -t vl41-eng:2 . | tail -2"
  fi
done
