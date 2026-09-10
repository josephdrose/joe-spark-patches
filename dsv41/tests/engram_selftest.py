# SPDX-License-Identifier: Apache-2.0
"""Engram-enabled self-test for the DeepSeek-V4.1-Flash reference implementation.

DeepSeek's `inference/model.py` has a `__main__` self-test, but its `ModelArgs`
defaults leave Engram off (`engram_layer_ids=()`), so nothing in the shipped
tree ever runs the n-gram path. This script turns Engram on with a small table
and exercises it on real hardware.

Run it from inside the fetched `inference/` directory, in any image that has
torch >= 2.10 and tilelang:

    python3 engram_selftest.py --tokenizer /w/tok --cache /w/cache

Five checks, in order:

  layout    `EngramLayout.from_args` draws `(max_ngram_size-1) * n_heads`
            primes per engram layer, upward from `engram_vocab_size - 1` and
            never reused. `engram_num_embeddings` is not derived by the code:
            it comes from the config JSON and must equal the sum of that
            layer's primes. Checked against the released config, whose
            16M-bucket layers must reproduce [384006168, 384016682].

  hashes    `NgramHashState` on the real tokenizer's compressed token map.
            Asserts the ids land inside the table and are deterministic.

  gemm      `Engram.wkv` is an fp8 `Linear`, so the engram path goes through
            tilelang's `fp8_gemm`. Weights and activations are set to small
            integers, which fp8 e4m3 holds exactly, so an fp32 reference
            rounded to bf16 should match the kernel bit for bit.

  engram    The full `Engram.forward` gate, checked for finiteness and for
            actually moving the residual stream.

  model     The stock `__main__` self-test (128-token prefill, 22 decode steps,
            DSpark head) with Engram enabled on two layers.

The compressed token map costs ~130k Rust decode+normalize calls, so it is
built once and cached; `--rebuild-map` forces a rebuild and compares.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engram as engram_mod  # noqa: E402
from engram import EngramLayout  # noqa: E402

FP8_BLOCK = 32  # model.fp8_block_size; also the engram table's scale block


class Args:
    """Only the fields EngramLayout.from_args reads. Avoids needing a device."""

    def __init__(self, layer_ids, ngram, heads, head_dim, vocab, num_embeddings=()):
        self.engram_layer_ids = layer_ids
        self.engram_max_ngram_size = ngram
        self.engram_n_heads = heads
        self.engram_head_dim = head_dim
        self.engram_vocab_size = vocab
        self.engram_num_embeddings = num_embeddings


def num_embeddings_for(layout: EngramLayout) -> tuple[int, ...]:
    return tuple(int(sum(p for per_ngram in layer for p in per_ngram)) for layer in layout.primes)


def check_layout(layer_ids, ngram, heads, head_dim, vocab):
    t0 = time.time()
    layout = EngramLayout.from_args(Args(layer_ids, ngram, heads, head_dim, vocab))
    rows = num_embeddings_for(layout)
    flat = [p for layer in layout.primes for per_ngram in layer for p in per_ngram]
    assert len(flat) == len(set(flat)), "primes must be disjoint across layers"
    print(f"[layout] vocab={vocab} layers={layer_ids} rows/layer={rows}")
    print(f"[layout] {len(flat)} primes, first={flat[0]} last={flat[-1]}, {time.time() - t0:.2f}s")

    # the released config is the only ground truth for the num_embeddings recipe
    t0 = time.time()
    real = num_embeddings_for(EngramLayout.from_args(Args((1, 14), 4, 8, 256, 16000000)))
    expect = (384006168, 384016682)
    print(f"[layout] released-config rows {real} expected {expect} match={real == expect} "
          f"({time.time() - t0:.2f}s)")
    assert real == expect, "prime-sum recipe does not reproduce the shipped config"
    return layout, rows


def load_token_map(tokenizer_dir, cache_path, rebuild):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    cached = None
    if os.path.exists(cache_path) and not rebuild:
        blob = torch.load(cache_path, weights_only=False)
        cached = (blob["lookup"], blob["size"])
    if cached is None:
        t0 = time.time()
        lookup, size = engram_mod.build_compressed_token_map(tok)
        print(f"[hashes] built compressed token map in {time.time() - t0:.1f}s")
        if os.path.exists(cache_path):
            old = torch.load(cache_path, weights_only=False)
            assert old["lookup"] == lookup and old["size"] == size, "cached token map is stale"
            print("[hashes] rebuild matches cache")
        torch.save({"lookup": lookup, "size": size}, cache_path)
        cached = (lookup, size)
    lookup, size = cached
    print(f"[hashes] tokenizer={len(tok)} tokens, compressed vocab={size}")
    # NgramHashState looks this up by module global, so patching here reaches it
    engram_mod.build_compressed_token_map = lambda _tok: (lookup, size)
    return tok, len(lookup), size


def build_model_args(model, layer_ids, ngram, heads, head_dim, vocab, rows, compressed):
    return model.ModelArgs(
        dspark_block_size=6,
        dspark_target_layer_ids=(3, 4),
        engram_layer_ids=layer_ids,
        engram_num_embeddings=rows,
        engram_max_ngram_size=ngram,
        engram_n_heads=heads,
        engram_head_dim=head_dim,
        engram_vocab_size=vocab,
        engram_compressed_vocab_size=compressed,
    )


CHUNK = 1 << 22


def fill_small_ints(t, gen, device):
    """Write -4..4 into a float8 tensor a chunk at a time.

    fp8 e4m3 holds those exactly. Staging the whole thing as float32 first would
    cost four times the table, which the big engram tables cannot afford.
    """
    flat = t.view(-1)
    for i in range(0, flat.numel(), CHUNK):
        n = min(CHUNK, flat.numel() - i)
        flat[i : i + n] = torch.randint(
            -4, 5, (n,), generator=gen, device=device, dtype=torch.float32
        ).to(t.dtype)
    return t


def fill_exact(eng, gen, device):
    """Small integers and 2^0 block scales throughout, so the fp32 reference in
    the gemm check is exact and the kernel has nothing to round until it writes
    bf16."""
    e = eng.embed
    fill_small_ints(e.weight.data, gen, device)
    e.scale.data = torch.ones(e.part_num_embeddings, e.dim // FP8_BLOCK, device=device).to(torch.float8_e8m0fnu)
    lin = eng.wkv
    fill_small_ints(lin.weight.data, gen, device)
    lin.scale.data = torch.ones_like(lin.scale.float()).to(torch.float8_e8m0fnu)
    lin.weight.scale = lin.scale
    return lin.weight.data.float()


def sane_init(module, gen, device):
    """Give every parameter a finite value.

    The stock self-test runs on `torch.empty`, so an fp8 e4m3 weight can hold a
    NaN bit pattern and the README only claims shapes and plumbing. e4m3 and
    e2m1 both hold small integers exactly and e2m1 has no NaN code at all, so
    filling them this way keeps the whole forward finite and lets the engram
    A/B below mean something.
    """
    kept = 0
    for _, p in module.named_parameters():
        d = p.dtype
        if d == torch.float8_e4m3fn:
            fill_small_ints(p.data, gen, device)
        elif d == torch.float8_e8m0fnu:
            p.data = torch.ones(p.shape, device=device).to(d)
        elif d == torch.float4_e2m1fn_x2:
            raw = p.data.view(torch.uint8).view(-1)
            for i in range(0, raw.numel(), CHUNK):
                n = min(CHUNK, raw.numel() - i)
                raw[i : i + n] = torch.randint(0, 256, (n,), generator=gen, device=device, dtype=torch.uint8)
        elif torch.equal(p.float(), torch.ones_like(p.float())):
            kept += 1  # RMSNorm and the engram gate weights are already ones
        else:
            p.data = torch.empty(p.shape, device=device, dtype=d).normal_(0, 0.02, generator=gen)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/w/tok")
    ap.add_argument("--cache", default="/w/cache")
    ap.add_argument("--vocab", type=int, default=4096, help="engram_vocab_size, the prime search start")
    ap.add_argument("--rebuild-map", action="store_true")
    ap.add_argument("--skip-model", action="store_true")
    a = ap.parse_args()

    layer_ids, ngram, heads, head_dim = (1, 3), 4, 8, 256
    os.makedirs(a.cache, exist_ok=True)

    print(f"[env] torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"cap={torch.cuda.get_device_capability()} {torch.cuda.get_device_name(0)}")

    layout, rows = check_layout(layer_ids, ngram, heads, head_dim, a.vocab)
    n_cols = (ngram - 1) * heads
    print(f"[layout] {n_cols} hash cols x {head_dim} dim, "
          f"{sum(rows) * (head_dim + head_dim // FP8_BLOCK) / 2**20:.1f} MiB of table")

    tok, map_len, compressed = load_token_map(
        a.tokenizer, os.path.join(a.cache, "token_map.pt"), a.rebuild_map
    )

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    import model  # noqa: E402  imports tilelang, so keep it after the cheap checks

    args = build_model_args(model, layer_ids, ngram, heads, head_dim, a.vocab, rows, compressed)
    dev = torch.device("cuda")
    # rebuild from the real args: the probe layout above carried no num_embeddings
    layout = EngramLayout.from_args(args)
    assert layout.num_embeddings == rows

    hash_state = engram_mod.NgramHashState(args, layout, tok)
    ids = torch.randint(0, min(args.vocab_size, map_len), (2, 150), device=dev)
    t0 = time.time()
    hashes = hash_state(ids, 0)
    torch.cuda.synchronize()
    print(f"[hashes] shape={tuple(hashes.shape)} dtype={hashes.dtype} {time.time() - t0:.3f}s")
    assert hashes.shape == (2, 150, len(layer_ids), n_cols)
    for i, r in enumerate(rows):
        lo, hi = int(hashes[:, :, i].min()), int(hashes[:, :, i].max())
        print(f"[hashes] layer {layer_ids[i]}: ids in [{lo}, {hi}], table rows {r}")
        assert 0 <= lo and hi < r, "hash id outside the table"
    hash_state.cache.zero_()
    assert torch.equal(hash_state(ids, 0), hashes), "hashing is not deterministic"
    # position 0 has no history, so every column there is the all-pad hash
    assert not torch.equal(hashes[:, 0], hashes[:, 5]), "hashes ignore context"
    print("[hashes] deterministic, context-sensitive, in range")

    eng = model.Engram(args, layer_ids[0], layout).to(dev)
    print(f"[gemm] wkv {eng.wkv.in_features}->{eng.wkv.out_features} weight dtype={eng.wkv.weight.dtype}")
    gen = torch.Generator(device=dev).manual_seed(1)
    lw = fill_exact(eng, gen, dev)

    flat = eng.embed(hashes[:, :, 0, :]).flatten(-2)
    assert flat.dtype == torch.bfloat16 and torch.isfinite(flat.float()).all()
    t0 = time.time()
    kv = eng.wkv(flat)
    torch.cuda.synchronize()
    ref = F.linear(flat.float().view(-1, flat.shape[-1]), lw).view(*flat.shape[:-1], -1)
    exact = torch.equal(kv, ref.to(torch.bfloat16))
    # the fp32 gap is just bf16 output rounding; at |c|~2048 one bf16 ulp is 16
    print(f"[gemm] fp8_gemm {tuple(kv.shape)} in {time.time() - t0:.3f}s")
    print(f"[gemm] bit-exact vs fp32 reference rounded to bf16: {exact}")
    print(f"[gemm] max diff vs unrounded fp32 {(kv.float() - ref).abs().max().item()}, "
          f"reference range [{ref.min().item():.0f}, {ref.max().item():.0f}]")

    x = torch.randn(2, 150, args.hc_mult, args.dim, generator=gen, device=dev, dtype=torch.bfloat16)
    t0 = time.time()
    out = eng(x, hashes[:, :, 0, :])
    torch.cuda.synchronize()
    dt = time.time() - t0
    moved = (out.float() - x.float()).abs()
    print(f"[engram] forward {tuple(out.shape)} in {dt:.3f}s, finite={bool(torch.isfinite(out.float()).all())}, "
          f"mean|delta|={moved.mean().item():.4f} max|delta|={moved.max().item():.4f}")
    assert torch.isfinite(out.float()).all(), "engram forward produced non-finite values"
    assert moved.max().item() > 0, "engram forward did not touch the residual stream"

    mask = torch.ones(2, 150, dtype=torch.bool, device=dev)
    mask[:, :10] = False
    out_masked = eng(x, hashes[:, :, 0, :], mask)
    assert torch.equal(out_masked[:, :10], x[:, :10]), "masked positions must pass through"
    print("[engram] token_mask closes the gate on masked positions")

    # decode-step cost: one position pulls n_cols rows, which is the unit an
    # NVMe-backed table would have to serve
    with torch.inference_mode():
        x1, h1 = x[:1, :1].clone(), hashes[:1, :1, 0, :].clone()
        for _ in range(5):
            eng(x1, h1)
        torch.cuda.synchronize()
        t0 = time.time()
        iters = 200
        for _ in range(iters):
            eng(x1, h1)
        torch.cuda.synchronize()
        per = (time.time() - t0) / iters
    print(f"[engram] decode step (1 position, {n_cols} rows): {per * 1e3:.3f} ms, "
          f"{n_cols / per:,.0f} rows/s")

    del eng, out, out_masked, x, x1, h1, flat, kv, ref
    torch.cuda.empty_cache()

    if a.skip_model:
        return
    t0 = time.time()
    m = model.Transformer(args, tok)
    build = time.time() - t0
    print(f"[model] built in {build:.1f}s, {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated")
    engram_layers = [i for i, l in enumerate(m.layers) if l.engram is not None]
    print(f"[model] engram attached to layers {engram_layers}")
    assert engram_layers == list(layer_ids)

    kept = sane_init(m, gen, dev)
    print(f"[model] parameters filled with finite values, {kept} left as ones")

    x = torch.randint(0, min(args.vocab_size, map_len), (2, 150), device=dev)
    t0 = time.time()
    output_ids, logits, main_hidden = m(x[:, :128])
    m.forward_spec(output_ids, main_hidden)
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(logits.float()).all())
    print(f"[model] prefill 128 tok in {time.time() - t0:.1f}s, logits {tuple(logits.shape)} finite={finite}")
    assert finite, "prefill logits are not finite"

    # A/B on the same weights and the same input: detaching the engram modules is
    # the only change, so any difference in the logits came through the engram path
    saved = [(l, l.engram) for l in m.layers if l.engram is not None]
    for layer, _ in saved:
        layer.engram = None
    logits_off = m(x[:, :128])[1]
    for layer, mod in saved:
        layer.engram = mod
    logits_on = m(x[:, :128])[1]
    assert torch.equal(logits_on, logits), "engram-on rerun is not reproducible"
    delta = (logits_on.float() - logits_off.float()).abs()
    rel = delta.max().item() / logits_off.float().abs().max().item()
    print(f"[model] engram on vs off: max|dlogit|={delta.max().item():.4f} "
          f"mean|dlogit|={delta.mean().item():.4f} relative={rel:.3%}")
    assert delta.max().item() > 0, "engram made no difference to the logits"
    t0 = time.time()
    for i in range(128, 150):
        output_ids, logits, main_hidden = m(x[:, i : i + 1], i)
        result = m.forward_spec(output_ids, main_hidden, i)
        assert result is not None
        output_ids, logits, confidence = result
    torch.cuda.synchronize()
    print(f"[model] 22 decode steps in {time.time() - t0:.1f}s")
    print(f"[model] peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("OK")


if __name__ == "__main__":
    main()
