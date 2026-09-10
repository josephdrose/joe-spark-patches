# Fix 7: route the sm12x indexer top-k off persistent_topk

## The failure

None on this fleet. `persistent_topk` never failed here. This change is
insurance for long context. It fixed no failure seen in this work.

The reported failure belongs to `tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark`,
fix 7. Their crash needed a 1M-wide logits buffer. The measurements below were
taken at 16,384, where the buffer is far from that width.

## The cause

`persistent_topk` sizes its grid from the logits buffer width. A long row asks
for more thread blocks than GB10's 48 SMs. The `FilteredTopK` fallback it then
takes wants 128 KB of shared memory per block. A GB10 block can hold 99 KB. The
launch fails and the engine dies.

`top_k_per_row_decode` has neither limit.

## The change

`patch/vlspeed-topk.py`. One edit, in
`model_executor/layers/sparse_attn_indexer.py`:

```python
use_persistent_topk = (
    current_platform.is_cuda()
    and topk_tokens in (512, 1024, 2048)
    and not current_platform.is_device_capability_family(120)
)
```

The edit is taken verbatim from the prior art. The installed
`sparse_attn_indexer.py` here is byte-identical to the file their diff was cut
against, so it applies unchanged.

## Measured on this GB10

Two findings.

`persistent_topk` was clean at every width and row count tested here, up to 48
rows by 300,000 wide.

At the operating point of this serve the swap is slower. 16,384-wide buffer,
6 to 24 rows:

| Kernel | Time |
|---|---|
| `persistent_topk` | 10.3 to 15.8 us |
| `top_k_per_row_decode` | 14.1 to 18.5 us |

`top_k_per_row_decode` wins only at large buffers. It is 1.4x to 3.0x faster at
300,000 wide.

Correctness was checked at widths 600 through 300,000. Its index sets are
identical to `torch.topk`.

## Why it is left on

38 of 40 layers run the indexer. The cost is roughly 40 to 80 calls at about
4 us. That is under 0.5% of a 70 ms step.

The measured penalty is smaller than the cost of hitting the failure at long
context.

## Not covered

The failure mode itself. This edit was installed for every long-context run
here, including the 262,144-token needle, so `persistent_topk` never ran at
those widths. Whether it would have failed there is untested.
