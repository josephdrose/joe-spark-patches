# SPDX-License-Identifier: Apache-2.0
"""Compile PR #56214's fused DSV4 qnorm/rope/kv-insert kernel out of tree.

The prebuilt vLLM wheel we run is at the PR's PARENT commit, so its
_C_stable_libtorch.abi3.so registers those three ops without the trailing
`apply_q_norm` bool that the PR adds and that V4.1 always passes as False.
This builds the PR's own .cu unchanged and registers the same three ops under
torch.ops.vl41 with the new schema. 1.1 MB .so, about a minute on a GB10.

Run inside the vLLM image with the PR source tree mounted at /src:

  docker run --rm --gpus all --ipc host \
    -v <workdir>/shim:/shim -v <workdir>/src:/src:ro -e HOME=/shim \
    --entrypoint /bin/bash vl41-eng:1 -lc 'cd /shim && python3 vl41-ops-build.py'

Needs, beside this file, in the same directory:
  bindings.cpp  = vl41-ops-bindings.cpp
  kernel.cu     = src/csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu
  torch_utils.h = src/csrc/libtorch_stable/torch_utils.h

Two include-path gotchas, both cost a build each:
  - /src/csrc/libtorch_stable must come BEFORE /src/csrc, or csrc/dispatch_utils.h
    shadows the stable one and VLLM_STABLE_DISPATCH_HALF_TYPES is undefined.
  - -DUSE_CUDA, or torch's shim.h hides aoti_torch_get_current_cuda_stream.
"""

import os

import torch
from torch.utils.cpp_extension import load

here = os.path.dirname(os.path.abspath(__file__))
load(
    name="vl41_ops",
    sources=[os.path.join(here, "bindings.cpp"), os.path.join(here, "kernel.cu")],
    extra_include_paths=[here, "/src/csrc/libtorch_stable", "/src/csrc"],
    extra_cflags=["-O2", "-std=c++17", "-DUSE_CUDA"],
    extra_cuda_cflags=[
        "-O3", "-std=c++17", "--expt-relaxed-constexpr", "-DUSE_CUDA",
        "-gencode", "arch=compute_121,code=sm_121",
    ],
    build_directory=os.path.join(here, "build"),
    # No PyInit_; it is a torch library, not a python module.
    is_python_module=False,
    verbose=True,
)
op = torch.ops.vl41.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
print("BUILT", os.path.join(here, "build", "vl41_ops.so"))
print("schema:", op.default._schema)
