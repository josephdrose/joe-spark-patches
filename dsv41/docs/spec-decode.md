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

`launch/vlpage-tp4-4node-up.sh` exposes this as `DSPARK=k`.

## Measured

Same prompt, same boot, `--enforce-eager`, 16,384 context:

| Configuration | tok/s |
|---|--:|
| No speculation | 14.94 |
| DSpark k=5 | 61.90 |

Acceptance rate was not recorded on this boot.
