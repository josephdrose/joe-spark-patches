# DSparkProposer — custom_class speculative decoding for DeepSeek-V4-Flash.
#
# Path B: vLLM serves the 284B target (fast); this proposer runs the eager
# 3-stage DSpark draft (model.py). custom_class only requires a `propose`
# method; the runner calls it with the contract mirrored from
# llm_base_proposer.propose (target_hidden_states = the aux capture from layers
# 40/41/42, next_token_ids = the per-seq bonus/anchor) and consumes the returned
# [batch, num_speculative_tokens] draft ids (+ _last_draft_probs for rejection).
#
# Registered via:
#   --speculative-config '{"method":"custom_class",
#                          "model":"dspark.proposer.DSparkProposer",
#                          "num_speculative_tokens":5}'
#
# STATUS: piece 3, FIRST PASS — UNVALIDATED, needs live serve-debug. The big
# open item is the SLIDING-WINDOW CONTEXT: DSpark's draft attends to the last
# `window_size` context vectors, which accumulate across decode steps, so this
# must keep per-request context buffers (add on accept, evict on finish). The
# v1 below recomputes from the hidden handed in this step only — correct for the
# prefill/first block, a TODO for multi-step decode. Batching is per-sequence.
from __future__ import annotations

import inspect
import os
import sys
import traceback

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class DSparkProposer:
    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        spec = vllm_config.speculative_config
        assert spec is not None
        self.num_speculative_tokens = spec.num_speculative_tokens
        hf = spec.draft_model_config.hf_config
        self.block_size = int(getattr(hf, "dspark_block_size", 5))
        self.window_size = int(getattr(hf, "sliding_window", 128) or 128)
        self.confidence_threshold = float(
            getattr(spec, "dspark_confidence_threshold", 0.0))
        # Batched draft graph: capture at a FIXED B = max_num_seqs and replay at
        # that batch every step (unused lanes padded + masked), so the draft
        # latency stays ~flat as concurrency grows — the draft is tiny, so batching
        # B of its matmuls into one graph is near-free vs looping batch=1 B times.
        self.max_batch = int(getattr(
            vllm_config.scheduler_config, "max_num_seqs", 4) or 4)
        self.noise_token_id = int(getattr(hf, "dspark_noise_token_id", 128799))

        # Recover the runner + device off the call stack (create_custom_proposer
        # instantiates us inside GPUModelRunner.__init__ with only vllm_config).
        runner = None
        for fi in inspect.stack():
            s = fi.frame.f_locals.get("self")
            if s is not None and "ModelRunner" in type(s).__name__ and hasattr(s, "device"):
                runner = s
                break
        self.runner = runner
        self.device = runner.device if runner is not None else torch.device(
            current_platform.device_type)
        self.model = None  # set in load_model
        # Raw-aux sliding-window context (batch=1 path). TODO: per-request buffers
        # keyed by req_id + reset on sequence boundary for batched serving.
        # Per-request draft-context state, keyed by req_id (batch>1 safe). Each
        # entry: {"window": [<=window_size, n*dim] aux hiddens, "prev_seqlen": int}.
        # New-seq / prefill is detected from each request's seqlen trajectory (first
        # call, drop, or jump > k+1) — NOT a banked-vs-seqlen delta, which desyncs
        # under chunked/cached prefill (the target aux only covers FORWARDED tokens)
        # and reset-storms the window. Evicted when a req leaves the batch.
        self._win: dict = {}

        # --- cudagraph state (graph the eager 3-stage draft) -----------------
        # The draft forward has ONE variable dim: the context length (<=
        # window_size). We pad context to a STATIC window_size + an additive mask,
        # giving fully static shapes -> capturable. Positions are FIXED (rope is
        # relative here — YaRN off — so absolute offsets don't matter): context
        # [0..N-1], block [N-1..N-1+block_size) (block-first == context-last, gap
        # 0, matching the reference). One graph is captured on the first decode
        # step and replayed every step thereafter.
        self._cap_n = self.window_size
        self._graph = None
        self._graph_disabled = (os.environ.get("DSPARK_GRAPH", "1") == "0")
        # DSPARK_TIMING=1: cuda-event time the draft forward, print rolling avg ms
        # to stderr every 100 calls (tells us the draft's share of the step).
        self._timing = (os.environ.get("DSPARK_TIMING", "0") == "1")
        self._t_events = []
        # DSPARK_TRACE=1: dump per-step banking state (seqlen, banked, aux rows,
        # is_prefill, window len) to stderr for the first N calls. Confirms the
        # banked-vs-seqlen divergence + is_prefill reset-storm under chunked
        # prefill + prefix caching (the real-workload accept-collapse bug).
        self._trace = (os.environ.get("DSPARK_TRACE", "0") == "1")
        self._trace_n = 0
        self._trace_max = int(os.environ.get("DSPARK_TRACE_MAX", "600"))
        # Step decomposition: entry->entry cuda-event interval = one full step
        # (verify + draft + sample + bubbles); entry->exit = the propose() GPU
        # span (draft + sample). step - propose = verify + vLLM rejection/bubble
        # (NOT in our code); propose - draft = sample_block. Tells us whether the
        # ~8ms non-draft/non-verify overhead is ours to cut or vLLM-internal.
        self._prev_entry = None
        self._ev_step = []
        self._ev_propose = []
        self._g_w = None
        self._g_anchor = None
        self._g_ctx_pos = None
        self._g_blk_pos = None
        self._g_mask = None
        self._g_out = None   # captured block ids [anchor, d1..d_block]

    # --- model load (mirrors llm_base_proposer.load_model essentials) --------

    def get_aux_layers(self):
        # We register on the target DIRECTLY (custom_class, not the EAGLE3 runner
        # path), so there is NO +1 shift — capture the exact target_layer_ids.
        return list(self.model.target_layer_ids)

    def load_model(self, target_model) -> None:
        from vllm.model_executor.model_loader import get_model
        cfg = self.vllm_config.speculative_config.draft_model_config
        cfg.hf_config.architectures = ["DeepseekV4DSparkForCausalLM"]
        self.model = get_model(vllm_config=self.vllm_config,
                               model_config=cfg)
        # Register aux capture on the target (piece 1 patch).
        if hasattr(target_model, "model") and hasattr(
                target_model.model, "set_aux_hidden_state_layers"):
            target_model.model.set_aux_hidden_state_layers(self.get_aux_layers())
        # Share the target embed/head into the draft (DSpark draft uses the
        # top-level embed/head, already loaded from the checkpoint here).

    # --- draft graphing ------------------------------------------------------

    def _ensure_buffers(self, w):
        if self._g_w is not None:
            return
        dev, dt, n = w.device, w.dtype, self._cap_n
        bs, Bc = self.block_size, self.max_batch
        # Batched static buffers: B_cap request lanes. _g_w [Bc, n, ctx_dim] holds
        # each lane's window (right-aligned, zero-padded); _g_mask [Bc, n] is the
        # per-lane additive -inf pad mask; _g_anchor [Bc] the per-lane bonus id.
        self._g_w = torch.zeros(Bc, n, w.shape[1], dtype=dt, device=dev)
        self._g_anchor = torch.full((Bc,), self.noise_token_id,
                                    dtype=torch.long, device=dev)
        # FIXED positions, SHARED across the batch: context [0..N-1]; block at
        # [N..N+bs) (block-first = context-last + 1, gap 1, per the reference;
        # rope is relative so only the offset matters).
        self._g_ctx_pos = torch.arange(n, device=dev)
        self._g_blk_pos = torch.arange(n, n + bs, device=dev)
        self._g_mask = torch.zeros(Bc, n, dtype=dt, device=dev)

    def _draft_and_sample(self):
        # forward (3-stage backbone) + sample_block (markov-biased autoregressive
        # block sampling). Both are static-shape — the markov gather has a data-
        # dependent INDEX but a fixed shape — so the whole thing is capturable;
        # folding sample_block in collapses its ~7.5ms eager launch overhead (5
        # sequential dependent vocab-wide ops) to the replay. Returns block ids.
        logits, h = self.model(self._g_w, self._g_anchor, self._g_ctx_pos,
                               self._g_blk_pos, self._g_mask)
        out, _, _ = self.model.sample_block(logits, h, self._g_anchor, 0.0)
        return out

    def _capture(self):
        from vllm.forward_context import set_forward_context
        try:
            with set_forward_context(None, self.vllm_config,
                                     num_tokens=self.max_batch * self.block_size):
                # Warm up on a side stream (force rope-table build, cublas/cudnn
                # autotune, mega_moe workspace alloc) before capturing.
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self._draft_and_sample()
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out = self._draft_and_sample()
                self._g_out, self._graph = out, g
            # stderr (not logger — the dspark.* logger is suppressed under vLLM's
            # logging config) so capture status is unmistakable in docker logs.
            print(f"[DSpark] draft+sample cudagraph CAPTURED ctx_pad={self._cap_n}",
                  file=sys.stderr, flush=True)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            print("[DSpark] draft cudagraph CAPTURE FAILED -> eager-draft fallback",
                  file=sys.stderr, flush=True)
            self._graph = None
            self._graph_disabled = True
            torch.cuda.synchronize()

    def _run_draft_batched(self, windows, anchors):
        # Assemble the B_cap static lanes from the active requests' windows
        # (right-aligned + -inf pad mask), pad the unused lanes, then capture-once /
        # replay ONE batched graph. windows[i]/anchors[i] for lane i (None = pad).
        # Returns [B_cap, block_size+1] block ids — proposer reads active lanes.
        sample = next((w for w in windows if w is not None), None)
        if sample is None:
            return None
        self._ensure_buffers(sample)
        n, Bc = self._cap_n, self.max_batch
        self._g_w.zero_()
        self._g_mask.fill_(float("-inf"))
        self._g_anchor.fill_(self.noise_token_id)
        for i in range(min(len(windows), Bc)):
            w, anc = windows[i], anchors[i]
            if w is None:
                continue
            n_ctx = min(w.shape[0], n)
            self._g_w[i, n - n_ctx:].copy_(w[-n_ctx:])
            self._g_mask[i, n - n_ctx:].zero_()
            self._g_anchor[i] = int(anc)
        if self._graph is None and not self._graph_disabled:
            self._capture()
        ev0 = ev1 = None
        if self._timing:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
        if self._graph is not None:
            self._graph.replay()
            out = self._g_out
        else:
            from vllm.forward_context import set_forward_context
            with set_forward_context(None, self.vllm_config,
                                     num_tokens=self.max_batch * self.block_size):
                out = self._draft_and_sample()
        if self._timing:
            ev1.record()
            self._t_events.append((ev0, ev1))
            if len(self._t_events) >= 100:
                torch.cuda.synchronize()
                ms = sum(a.elapsed_time(b) for a, b in self._t_events) / len(
                    self._t_events)
                mode = "replay" if self._graph is not None else "eager"
                print(f"[DSpark] draft+sample {mode} avg {ms:.2f} ms "
                      f"over {len(self._t_events)} calls", file=sys.stderr,
                      flush=True)
                self._t_events = []
        return out

    # --- propose -------------------------------------------------------------

    @torch.inference_mode()
    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu,
                slot_mappings=None, *args, **kwargs):
        # custom_class RAW interface (gpu_model_runner ~:4780): per-req sampled
        # (bonus) tokens + batch token state; we fetch the target's aux hidden
        # ourselves (v4_aux_patch stashed it during the target forward).
        #
        # BATCH>1: vLLM continuous-batching merges concurrent decodes into one
        # forward, so propose() gets `batch` requests. We keep PER-REQUEST window
        # state keyed by req_id and split the flattened aux per request via
        # query_start_loc. The draft is run per request (looped batch=1 graph —
        # Phase 1; Phase 2 batches it). Read the RAW _dspark_aux_buffer (not
        # get_dspark_aux_hidden, whose _dspark_aux_T is stale under cudagraph
        # replay) — the decode forward writes EVERY req's block to buf[0:], so
        # query_start_loc[i]..[i+1] indexes req i's hiddens directly.
        target = getattr(self.runner, "model", None)
        inner = getattr(target, "model", None) if target is not None else None
        aux_buf = getattr(inner, "_dspark_aux_buffer", None) if inner is not None else None
        batch = len(sampled_token_ids)
        k = self.num_speculative_tokens
        out_drafts = torch.full((batch, k), -1, dtype=torch.long, device=self.device)
        # Global no-op: no aux yet (profiling) or all-empty sampled (prefill-only /
        # memory-profiling dummy run).
        if (aux_buf is None or num_tokens_no_spec is None
                or not any(sampled_token_ids)):
            self._last_draft_probs = None
            return out_drafts

        qsl = self.runner.query_start_loc.np    # cumulative; [qsl[i], qsl[i+1]) = req i
        req_ids = self.runner.input_batch.req_ids
        window_size, k1 = self.window_size, k + 1

        entry = None
        if self._timing:
            entry = torch.cuda.Event(enable_timing=True)
            entry.record()
            if self._prev_entry is not None:
                self._ev_step.append((self._prev_entry, entry))
            self._prev_entry = entry

        seen = set()
        windows: list = [None] * batch
        anchors: list = [None] * batch
        for i in range(batch):
            si = sampled_token_ids[i]
            if not si:                          # still prefilling this req -> no draft
                continue
            rid = req_ids[i]
            seen.add(rid)
            seqlen = int(num_tokens_no_spec[i])
            start, end = int(qsl[i]), int(qsl[i + 1])
            n_in = end - start
            st = self._win.get(rid)
            prev = st["prev_seqlen"] if st is not None else None
            # New-seq / prefill boundary from THIS req's seqlen trajectory (first
            # call, drop, or jump > k+1) — robust to chunked/cached prefill where
            # the aux only covers forwarded tokens (see the 2026-06-28 banking fix).
            is_new_seq = (prev is None) or (seqlen <= prev) or (seqlen - prev > k1)
            if is_new_seq:
                # Prefill seed: tail of this req's just-forwarded block = its most-
                # recent context before the first generated token.
                window = (aux_buf[max(start, end - window_size):end].clone()
                          if n_in > 0 else None)
                n_new = 0 if n_in <= 0 else min(n_in, window_size)
            else:
                # Decode: bank this req's confirmed hiddens = the FRONT
                # aux[start : start+(seqlen-prev)] (anchor = last bonus ++ accepted),
                # which the graph-replayed forward refreshes each step.
                n_new = max(0, min(seqlen - prev, n_in))
                window = st["window"] if st is not None else None
                if n_new > 0:
                    add = aux_buf[start:start + n_new]
                    w = (add.clone() if window is None
                         else torch.cat([window, add], 0))
                    if w.shape[0] > window_size:
                        w = w[-window_size:]
                    window = w
            self._win[rid] = {"window": window, "prev_seqlen": seqlen}
            if self._trace and self._trace_n < self._trace_max:
                self._trace_n += 1
                print(f"[DSpark-trace] req={i} seqlen={seqlen} prev={prev} "
                      f"n_in={n_in} is_new_seq={int(is_new_seq)} n_new={n_new} "
                      f"win={0 if window is None else int(window.shape[0])}",
                      file=sys.stderr, flush=True)
            if window is None:
                continue
            # Collect this lane's window + anchor; the draft for ALL active lanes
            # runs in ONE batched graph replay after the loop (flat latency).
            windows[i] = window
            anchors[i] = int(si[-1])

        # Evict state for requests no longer in the batch (finished / preempted).
        if len(self._win) > len(seen):
            for rid in [r for r in self._win if r not in seen]:
                del self._win[rid]

        # Batched draft: one graph replay for every active lane; read lane i's
        # block ids [anchor, d1..d_k] -> the k drafts for request i.
        out = self._run_draft_batched(windows, anchors)
        if out is not None:
            for i in range(batch):
                if windows[i] is not None:
                    out_drafts[i] = out[i, 1:1 + k]

        if self._timing and entry is not None:
            exitev = torch.cuda.Event(enable_timing=True)
            exitev.record()
            self._ev_propose.append((entry, exitev))
            if len(self._ev_propose) >= 100:
                torch.cuda.synchronize()
                sp = sum(a.elapsed_time(b) for a, b in self._ev_propose) / len(
                    self._ev_propose)
                stp = (sum(a.elapsed_time(b) for a, b in self._ev_step)
                       / max(len(self._ev_step), 1))
                print(f"[DSpark] step avg {stp:.2f} ms, propose avg {sp:.2f} ms "
                      f"(draft+sample) over {len(self._ev_propose)}",
                      file=sys.stderr, flush=True)
                self._ev_propose = []
                self._ev_step = []
        # No draft probs -> greedy exact-match rejection.
        self._last_draft_probs = None
        return out_drafts

    def take_last_draft_probs(self):
        p, self._last_draft_probs = getattr(self, "_last_draft_probs", None), None
        return p
