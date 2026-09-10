# Fix 3: the `apply_q_norm` argument, and why Python cannot cancel it

## The failure

No exception. Wrong numbers.

The image installs the vLLM wheel for PR #56214's **parent** commit. That
wheel's `_C_stable_libtorch.abi3.so` predates the PR.

PR #56214 adds a trailing `bool apply_q_norm=True` to three op schemas:

```
fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert
fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert
```

`vllm/models/deepseek_v4_1/attention.py` passes the literal `False` at all three
call sites. V4.1 removes the Q RMSNorm that V4 applies.

The parent commit's `_C` has no such argument. Calling it applies the Q RMSNorm.

## Why a Python workaround does not exist

**RMSNorm is scale invariant.** Multiplying `q` by any constant leaves the
normalized output unchanged. No pre-scaling or post-scaling in Python cancels a
norm that the kernel already applied.

There is no eager fallback path for these three ops.

Measured, by scaling `q` by a factor of 10 and taking the output ratio:

| `apply_q_norm` | Output ratio |
|---|--:|
| `True` | 0.9999 |
| `False` | 9.9992 |

The flag changes the result. The wrong value produces coherent but wrong output.

## The change

Compile the PR's own `.cu` unchanged for sm_121 as an out-of-tree extension.
Register the three ops under `torch.ops.vl41` with the new schema.

The result is a 1.1 MB `.so`. Build time is about one minute on a GB10.

This works because `csrc/libtorch_stable` builds into its own `.so`, and
`STABLE_TORCH_LIBRARY_FRAGMENT` allows a second library beside it. The kernel
body is not modified.

Files:

| File | Role |
|---|---|
| `patch/vl41-ops-bindings.cpp` | The `STABLE_TORCH_LIBRARY(vl41, ops)` declarations |
| `patch/vl41-ops-build.py` | `torch.utils.cpp_extension.load` with the sm_121 gencode |

`build/vl41-build-image.sh` then rewrites `attention.py` to load the library:

```python
_VL41_SHIM_SO = os.environ.get("VL41_SHIM_SO", "/opt/vl41/vl41_ops.so")
if os.path.exists(_VL41_SHIM_SO):
    torch.ops.load_library(_VL41_SHIM_SO)
    _FUSED_KV_OPS = torch.ops.vl41
else:
    _FUSED_KV_OPS = torch.ops._C
```

The fallback to `torch.ops._C` exists so a tree with the PR's own `csrc` built in
still works. On this image the `.so` is always present.

## Two build traps

Each one costs a build.

1. **Include order.** `/src/csrc/libtorch_stable` must precede `/src/csrc`.
   Otherwise `csrc/dispatch_utils.h` shadows the stable header and
   `VLLM_STABLE_DISPATCH_HALF_TYPES` is undefined.
2. **`-DUSE_CUDA` is required.** Without it torch's `shim.h` hides
   `aoti_torch_get_current_cuda_stream`.

## What this does not cover

`per_token_group_quant`'s group-32 widening is the PR's other csrc change. It is
unbuilt here. Nothing exercised it in these runs.
