#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fetch DeepSeek's DeepSeek-V4.1-Flash reference inference tree onto a DGX Spark
# and run the engram-enabled self-test in it.
#
# Copy this file, engram_selftest.py and engram_graph_capture.py to the box
# together, then:
#     ./engram-selftest.sh [scratch_dir] [extra args for the python]
#
# SCRIPT selects which test runs. engram_graph_capture.py wants a file for
# --cache, so pass it again:
#     SCRIPT=engram_graph_capture.py ./engram-selftest.sh /w \
#         --cache /w/cache/token_map.pt
#
# First run downloads ~6 MB of tokenizer plus the .py files and compiles 15
# tilelang kernels; later runs reuse both caches and take under 10 s.
#
# The image only has to carry torch >= 2.10 and tilelang. Its ENTRYPOINT is
# `vllm`, hence --entrypoint python3.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DIR=${1:-$HOME/cc-scratch/dsv41eng}
shift || true
IMAGE=${IMAGE:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2}
SCRIPT=${SCRIPT:-engram_selftest.py}
BASE=https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main

mkdir -p "$DIR/inference" "$DIR/tok" "$DIR/cache"
# only what model.py imports, plus config.json for the released engram shapes
for f in model.py engram.py kernel.py vision.py image_processor.py config.json; do
  [ -s "$DIR/inference/$f" ] || curl -sSfL -o "$DIR/inference/$f" "$BASE/inference/$f"
done
for f in tokenizer.json tokenizer_config.json; do
  [ -s "$DIR/tok/$f" ] || curl -sSfL -o "$DIR/tok/$f" "$BASE/$f"
done
cp "$HERE/engram_selftest.py" "$HERE/engram_graph_capture.py" "$DIR/inference/"

# WARNING: --memory does NOT protect a co-resident serve. GB10 memory is one
# unified pool and CUDA allocations are not charged to the cgroup, so this caps
# host anon memory only. Verified 2026-09-10: a 10 GiB run on a box with 17.5 GiB
# free produced 29 NVRM NV_ERR_NO_MEMORY events and killed the glm53-zaifp8 TP
# rank 47 s later, with OOMKilled=false and ExitCode=0. Stop the serve first.
exec docker run --rm --gpus all --network host --memory "${MEM:-12g}" \
  --entrypoint python3 \
  -v "$DIR:/w" -w /w/inference \
  -e TILELANG_CACHE_DIR=/w/tlcache -e TRITON_CACHE_DIR=/w/trcache -e HF_HUB_OFFLINE=1 \
  "$IMAGE" "$SCRIPT" --tokenizer /w/tok --cache /w/cache "$@"
