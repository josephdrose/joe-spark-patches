# Speculative decoding: use DSpark

## The failure

Setting `"method": "mtp"` is refused by source:

```
DeepSeek V4.1 has no classic-MTP draft. Use speculative method 'dspark'
instead of 'mtp'.
```

The message comes from `vllm/config/speculative.py`.

## The cause

The model has no classic-MTP draft head. The checkpoint's 7.4 GiB of MTP weights
serve the DSpark cascade instead.

## The change

Use `method: dspark`. `num_speculative_tokens` must be a multiple of
`dspark_block_size`, which is 5 in this checkpoint.

```
--speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

`launch/vlspeed-tp4-4node-up.sh` exposes this as `DSPARK=k`.

## Measured

Same prompt, same boot, `--enforce-eager`, 16,384 context:

| Configuration | tok/s |
|---|--:|
| No speculation | 14.94 |
| DSpark k=5 | 61.90 |

Acceptance rate was not recorded on this boot.

## Which k

Later boot, CUDA graphs on, same 16,384 context. C1 is the mean over 8
categories with counting excluded, in tok/s.

| k | C1 agg | Counting C1 | Acceptance |
|--:|--:|--:|---|
| 5 | 46.31 | 85.19 | 3.87 over the full bench. Counting 6.00, code 4.09, prose 2.49 |
| 10 | 34.09 | 96.23 | 2.03 to 7.12 across intervals |

k=10 wins 13% on pure counting and loses 26% on the mixed set. Draft cost
doubles, and prose accepts about 2.

Per-position acceptance at k=5 on counting was 1.000, 1.000, 0.987, 0.987,
0.974.

k=5 is the keeper. See [cuda-graphs.md](cuda-graphs.md).
