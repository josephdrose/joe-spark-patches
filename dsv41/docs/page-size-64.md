# Fix 4: every KV page holds 64 states

## The failure

The model loads. The first decode step raises:

```
ValueError: SM120 sparse-MLA has no decode kernel for this shape:
num_tokens=2, num_heads=16, topk=1152, d_qk=512, page_block_size=32,
model_type=dsv4, extra_topk=0. Mismatch: page_block_size=32 is unsupported;
decode kernels are instantiated only for page_block_size=64.
```

Everything else on that boot was correct. 476 GiB loaded at TP=4. Weights 78.95
GiB per rank. The Engram tables were on NVMe. The backend was
`FLASHINFER_MLA_SPARSE_DSV41`. The MoE backend was `DEEPGEMM_MXFP4`. TileLang MHC
kernels JIT compiled for sm_121.

Page size was the only mismatch. A probe with `page=64, topk=1152` returned
`True`.

## The cause

`vllm/models/deepseek_v4_1/attention.py:456` hard-codes `block_size=32` for the
**sliding-window cache**.

The SWA cache is the **primary** tensor in every sm120 sparse-MLA call. An
instrumented probe shows it:

```
VL41PROBE kv.shape=(443259, 32, 1, 584) pbs=32 | extra.shape=None topk=1152
```

The page size does not come from `block_size // compress_ratio`. Two earlier
write-ups in this work said it did. Both were wrong.

## Why 64 is the only value

Three consumers constrain it. Each rejects a different set.

| Consumer | Accepts | Source |
|---|---|---|
| FlashInfer sm120 sparse-MLA decode | 64 only | Template parameter, instantiated at 64 |
| FlashInfer sm120 sparse-MLA prefill, extra cache | 64 and 2 | `sparse_mla_sm120_prefill.cu:449` |
| DeepGEMM paged MQA logits, used by the indexer | 32 and 64 | `assert block_kv == 32 or 64` |

Two other values were measured on the way to this conclusion.

- **Block 128.** The indexer gets `num_states=128`. DeepGEMM asserts
  `block_kv == 32 or 64` and fails.
- **Block 64 applied globally.** Ratio-2 layers then get an extra page of 32.
  Prefill refuses it.

V4.1's per-layer `compress_ratios` codes are 0, 1 and 2. Code 0 means sliding
window. A compressed cache holds `block_size / compress_ratio` states per block.
One global block size therefore cannot put both ratios on 64.

## The change

Three KV-cache spec changes in Python. About five edits. No kernel build.

| Spec | Before | After |
|---|---|---|
| Sliding-window cache | `block_size=32` | `block_size=64` |
| Compressed KV | `cache_config.block_size` | `64 * compress_ratio` |
| Indexer k_cache | `cache_config.block_size` | `64 * compress_ratio` |

The compressed and indexer specs land in separate KV-cache groups. Both backends
must then accept either group's manager block size, or
`select_common_block_size` rejects the ratio-1 group:

| File | Change |
|---|---|
| `nvidia/flashinfer_sparse.py` | `get_supported_kernel_block_sizes` returns `[64, 128]` |
| `v1/attention/backends/mla/indexer.py` | Returns `[64, 128]` on device capability family 120 |

Script: `patch/vlpage-page64.py`. It edits an installed vLLM tree in place. Every
substitution asserts its occurrence count, so a version drift fails loudly.

## Result

The endpoint served after this change. See the numbers in the
[README](../README.md).

## Difference from the prior art

`tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark` reached the same three KV-spec
changes independently. That repository was found after this one was derived.

Their route needs FlashInfer 0.7.0rc1 and `--block-size 128`. This route needs
neither. The eugr 0.6.18 build serves `topk=1152` at page 64, and the compressed
specs here are made independent of `cache_config.block_size`.
