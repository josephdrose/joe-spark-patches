#!/usr/bin/env python3
"""vl41-page64: put every DeepSeek-V4.1 KV page on 64 states for SM120/SM121.

FlashInfer's sm120 sparse-MLA kernels take the PRIMARY cache page size as a C++
template parameter instantiated only at 64 (_PAGE_BLOCK_SIZE in
flashinfer/mla/_sparse_mla_sm120_plan.py). The EXTRA (compressed) cache page is
runtime in the decode kernel but template in the prefill dual kernel, where only
64 and 2 are instantiated. DeepGEMM's paged MQA logits, used by the indexer,
asserts block_kv in {32, 64}. The only value satisfying all three is 64.

V4.1's per-layer compress_ratios are 0 (sliding window only), 1 and 2, and a
compressed cache holds block_size/compress_ratio states per block. One global
block size therefore cannot put both ratios on 64, so the compressed and indexer
specs take 64*compress_ratio and land in separate KV-cache groups.

Applies to an installed vLLM tree; pass the dist-packages/vllm path.
"""
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages/vllm"


def sub(path, old, new, count=1):
    p = f"{ROOT}/{path}"
    s = open(p).read()
    n = s.count(old)
    assert n == count, f"{path}: expected {count} occurrence(s) of {old!r}, found {n}"
    open(p, "w").write(s.replace(old, new))
    print(f"  {path}: {old.strip()!r} -> {new.strip()!r}")


print("vl41-page64: patching", ROOT)

# 1. SWA cache. The default is 64; V4.1 asked for 32 on the (FlashMLA-true,
#    FlashInfer-false) assumption that the decode kernels take the page at
#    runtime. This is the primary cache in every sm120 sparse-MLA call.
sub("models/deepseek_v4_1/attention.py",
    "            block_size=32,\n",
    "            block_size=64,  # vl41-page64: sm120 primary page is compiled at 64\n")

# 2. Compressed KV spec: 64 states per block for either ratio.
sub("models/deepseek_v4_1/attention.py",
    "        return MLAAttentionSpec(\n            block_size=vllm_config.cache_config.block_size,\n",
    "        return MLAAttentionSpec(\n"
    "            # vl41-page64: block/compress_ratio states per block must be 64.\n"
    "            block_size=64 * max(1, self.compress_ratio),\n")

# 3. Indexer spec: same, so its block table matches the compressed cache's and
#    DeepGEMM sees block_kv=64.
sub("models/deepseek_v4_1/attention.py",
    "        return MLAAttentionSpec(\n            block_size=self.cache_config.block_size,\n",
    "        return MLAAttentionSpec(\n"
    "            # vl41-page64: matches the compressed spec above.\n"
    "            block_size=64 * max(1, self.compress_ratio),\n")

# 4/5. Both backends must accept either group's manager block size, or
#      select_common_block_size rejects the ratio-1 group.
sub("models/deepseek_v4_1/nvidia/flashinfer_sparse.py",
    "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n        return [128]\n",
    "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
    "        # vl41-page64: 64 for compress_ratio 1, 128 for compress_ratio 2.\n"
    "        return [64, 128]\n")

s = open(f"{ROOT}/v1/attention/backends/mla/indexer.py").read()
old = ("    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
       "        return [64 if current_platform.is_device_capability_family(90) else 128]\n")
assert s.count(old) == 1, "indexer.py: DeepseekV4IndexerBackend block sizes not found"
new = ("    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
       "        if current_platform.is_device_capability_family(120):\n"
       "            # vl41-page64: 64 for compress_ratio 1, 128 for compress_ratio 2.\n"
       "            return [64, 128]\n"
       "        return [64 if current_platform.is_device_capability_family(90) else 128]\n")
open(f"{ROOT}/v1/attention/backends/mla/indexer.py", "w").write(s.replace(old, new))
print("  v1/attention/backends/mla/indexer.py: DeepseekV4IndexerBackend -> [64, 128] on sm12x")
print("vl41-page64: done")
