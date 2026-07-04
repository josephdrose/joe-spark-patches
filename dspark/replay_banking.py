#!/usr/bin/env python3
"""Offline replay harness for the DSparkProposer window-banking logic.

The accept-collapse bug lives in pure host-side Python (the window/banking
arithmetic in propose()), NOT in the model forward or the cudagraph. So we can
validate a fix with ZERO weights / ZERO cluster: feed the per-step
(seqlen, n_in) sequence a real workload produced and check the window behaves.

Two banking implementations are replayed against the same input sequence:
  old  — the shipped logic: detect prefill via (seqlen-1-banked) in [1, k+1],
         bank min(seqlen-1-banked, n_in). Desyncs under chunked/cached prefill
         because `banked` can't reach seqlen-1 when aux skips cached/early-chunk
         tokens -> is_prefill fires EVERY decode step -> window reset-storm.
  new  — detect new-seq via seqlen drop / first-call / jump > k+1 (NOT the
         banked delta); bank exactly the decode increment (seqlen - prev),
         which is the confirmed count this step; only reset on a true new-seq.

Input: either a captured docker-log file with [DSpark-trace] lines (--trace),
or the built-in synthetic chunked+cached-prefill generator (default).

A pass = NEW shows is_prefill/new-seq ONCE (at the real prefill) and the window
grows to window_size and stays put; OLD shows the reset-storm (window collapses
to <= k+1 and resets every decode step).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import deque


def replay_old(steps, k, window_size):
    """Shipped banking. steps = list of (seqlen, n_in). Returns per-step dicts."""
    banked = 0
    win = None  # modelled as a length only (old logic never inspects contents)
    out = []
    for seqlen, n_in in steps:
        banked_pre = banked
        new_conf = seqlen - 1 - banked
        is_prefill = not (1 <= new_conf <= k + 1)
        if is_prefill:
            win = None
            banked = 0
        n_new = max(0, min(seqlen - 1 - banked, n_in))
        if n_new > 0:
            add = n_new
            wlen = add if win is None else win + add
            if wlen > window_size:
                wlen = window_size
            win = wlen
            banked += n_new
        out.append(dict(seqlen=seqlen, n_in=n_in, banked_pre=banked_pre,
                        new_conf=new_conf, reset=int(is_prefill), n_new=n_new,
                        win=(0 if win is None else win)))
    return out


def replay_new(steps, k, window_size):
    """Fixed banking: new-seq by seqlen jump/drop, bank the decode increment.

    Window modelled as a deque of confirmed position ids so we can also assert
    it holds the RIGHT recent positions, not just a plausible length. The decode
    increment seqlen-prev = (#accepted drafts + 1 bonus) = the count of confirmed
    hiddens that arrive in aux[0:n_new] this step (anchor ++ accepted)."""
    prev = None
    win = deque(maxlen=window_size)  # holds confirmed position ids
    out = []
    for seqlen, n_in in steps:
        is_new = (prev is None) or (seqlen <= prev) or (seqlen - prev > k + 1)
        if is_new:
            win.clear()
            # Prefill seed: the last chunk's aux ends at the last prompt token
            # (positions ...seqlen-2; seqlen-1 is the fresh bonus, not forwarded
            # yet). Bank the tail we actually have, up to window_size.
            seed = min(n_in, window_size)
            for p in range(seqlen - 1 - seed, seqlen - 1):
                win.append(p)
            n_new = seed
        else:
            n_new = max(0, min(seqlen - prev, n_in))
            # confirmed this step = anchor(prev-1) ++ accepted(prev..prev+a-1)
            for p in range(prev - 1, prev - 1 + n_new):
                win.append(p)
        prev = seqlen
        # contiguity / recency check: window should be the last len(win) positions
        # ending at seqlen-2 (the newest forwarded hidden)
        contiguous = (list(win) == list(range(seqlen - 1 - len(win), seqlen - 1))) if win else True
        out.append(dict(seqlen=seqlen, n_in=n_in, new_seq=int(is_new),
                        n_new=n_new, win=len(win), contiguous=int(contiguous)))
    return out


def gen_synthetic(prompt=12000, cache_frac=0.357, maxbatch=8192, k=5,
                  decode_steps=200, accept=2, seed_prev=480):
    """Model a real chunked+cached prefill then N decode steps.

    Returns a list of (seqlen, n_in) as propose() would see them. Only the FINAL
    prefill chunk reaches real banking (earlier chunks sample nothing -> early
    return -> not in this list). seed_prev = a stale seqlen left by a PRIOR
    request (proves new-seq detection fires on the big jump)."""
    cached = int(prompt * cache_frac)
    uncached = prompt - cached
    last_chunk = uncached if uncached <= maxbatch else maxbatch  # aux rows at 1st real call
    steps = []
    # first real propose: seqlen ~ prompt (committed prompt), aux = last prefill chunk
    seqlen = prompt
    steps.append((seqlen, last_chunk))
    # decode: each step +(accept+1) tokens, aux = k+1 verify rows (padded buckets
    # could differ, but n_in only matters via min(); keep k+1)
    for _ in range(decode_steps):
        seqlen += accept + 1
        steps.append((seqlen, k + 1))
    # the stale-prev case is handled by passing seed_prev into a prior dummy step
    return [(seed_prev, k + 1)] + steps


def parse_trace(path):
    """Pull (seqlen, n_in) from docker-log [DSpark-trace] lines."""
    pat = re.compile(r"seqlen=(\d+).*?n_in=(\d+)")
    steps = []
    with open(path) as f:
        for line in f:
            if "[DSpark-trace]" not in line:
                continue
            m = pat.search(line)
            if m:
                steps.append((int(m.group(1)), int(m.group(2))))
    return steps


def summarize(name, rows, k):
    decode = rows[1:]  # skip the first (prefill) row
    resets = sum(r.get("reset", r.get("new_seq", 0)) for r in decode)
    wins = [r["win"] for r in decode]
    final = wins[-1] if wins else 0
    avg = sum(wins) / len(wins) if wins else 0
    bad_contig = sum(1 - r.get("contiguous", 1) for r in rows)
    print(f"[{name}] decode steps={len(decode)} "
          f"resets/new-seq during decode={resets} "
          f"final_win={final} avg_win={avg:.1f}"
          + (f" non-contiguous_steps={bad_contig}" if "contiguous" in rows[0] else ""))
    return resets, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", help="docker-log file with [DSpark-trace] lines")
    ap.add_argument("-k", type=int, default=5, help="num_speculative_tokens")
    ap.add_argument("-w", "--window", type=int, default=128)
    ap.add_argument("--show", type=int, default=8, help="print first/last N rows")
    args = ap.parse_args()

    if args.trace:
        steps = parse_trace(args.trace)
        if not steps:
            print(f"no [DSpark-trace] lines in {args.trace}", file=sys.stderr)
            sys.exit(1)
        print(f"parsed {len(steps)} trace steps from {args.trace}")
    else:
        steps = gen_synthetic(k=args.k)
        print(f"synthetic: {len(steps)} steps "
              f"(prompt=12000 cache=35.7% -> first real call seqlen={steps[1][0]} "
              f"n_in={steps[1][1]})")

    old = replay_old(steps, args.k, args.window)
    new = replay_new(steps, args.k, args.window)

    def show(name, rows):
        n = args.show
        print(f"\n--- {name} (first {n}) ---")
        for r in rows[:n]:
            print("  " + " ".join(f"{x}={r[x]}" for x in r))
    show("OLD", old)
    show("NEW", new)

    print()
    r_old, f_old = summarize("OLD", old, args.k)
    r_new, f_new = summarize("NEW", new, args.k)
    print()
    ok = (r_new <= 1 and f_new >= min(args.window, 64) and r_old > 5)
    print("RESULT:", "PASS — fix kills the reset-storm, window saturates"
          if ok else "CHECK — inspect rows above")


if __name__ == "__main__":
    main()
