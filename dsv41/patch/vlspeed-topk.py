#!/usr/bin/env python3
"""vlspeed-topk: route the SM12x indexer decode top-k off persistent_topk.

`persistent_topk` sizes its grid from the logits buffer width, so a long row
asks for more thread blocks than GB10's 48 SMs, and the FilteredTopK fallback
it then takes wants 128 KB of shared memory per block against GB10's 99 KB.
The launch fails and the engine dies. `top_k_per_row_decode` has neither
limit and is 1.6-3.6x faster on GB10 besides.

That failure is theirs, not ours. `persistent_topk` never failed on this
fleet. It was clean at every width and row count tested here, up to 48 rows by
300,000 wide. At a 16,384-wide buffer `persistent_topk` is the faster of the
two. This edit is carried as insurance for long context. No crash here required
it. See docs/topk-swap.md.

Taken verbatim from tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark fix 7
(patch/sm12x-indexer-topk). Our installed sparse_attn_indexer.py is
byte-identical to the file that diff was cut against, so this is their edit,
not a port of it.

Applies to an installed vLLM tree; pass the dist-packages/vllm path.
"""
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages/vllm"
PATH = f"{ROOT}/model_executor/layers/sparse_attn_indexer.py"

OLD = """        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
            512,
            1024,
            2048,
        )
"""
NEW = """        # vlspeed-topk (SM12x): GB10 has 48 SMs and 99 KB smem per block. On long
        # rows persistent_topk oversubscribes and its FilteredTopK fallback needs
        # >=128 KB smem, so the launch fails. Use top_k_per_row_decode.
        use_persistent_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and not current_platform.is_device_capability_family(120)
        )
"""

s = open(PATH).read()
assert s.count(OLD) == 1, f"sparse_attn_indexer.py: expected 1 use_persistent_topk block, found {s.count(OLD)}"
open(PATH, "w").write(s.replace(OLD, NEW))
print("vlspeed-topk: sparse_attn_indexer.py -> top_k_per_row_decode on sm12x")
