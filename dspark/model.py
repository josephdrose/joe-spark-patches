# DSpark draft model for DeepSeek-V4-Flash, vLLM-native (Path B).
#
# The DSpark draft is a 3-stage cascade stored under mtp.0/mtp.1/mtp.2:
#   stage 0 (mtp.0): main_proj (3*dim->dim) + main_norm, ingests the target's
#                    [40,41,42] mean-over-hc_mult hidden; then a DSpark block.
#   stage 1 (mtp.1): a DSpark block.
#   stage 2 (mtp.2): a DSpark block + hc_head + markov_head + confidence_head + norm.
# A "DSpark block" is a DeepSeek-V4 decoder block whose attention is a NON-CAUSAL
# BLOCK CROSS-ATTENTION (query = the noise/bonus block; K/V = the projected
# target context ++ the block) rather than V4's causal MLA+indexer self-attn.
#
# Path B: we run the draft EAGER and reuse vLLM's V4 MoE (TP-correct experts) +
# mHC ops for the FFN, and reuse the V4 attention's projection layout
# (fused_wqa_wkv / wq_b / q_norm / kv_norm / wo_a / wo_b / attn_sink) so the
# mtp.* weights load unchanged. Only the attention MATH is hand-written here,
# dense, following DeepSeek's reference DSparkAttention.forward (sparse_attn ->
# dense SDPA; the draft block+window is tiny so sparsity buys nothing).
#
# STATUS: piece 2/4 of the vLLM port. UNVALIDATED — the eager MLA cross-attention
# (rope split, qk-norm, attn_sink, wo_a bmm) and the weight remap must be checked
# numerically against the reference; a wrong tensor here silently kills accept.
from __future__ import annotations

import typing
from collections.abc import Callable, Iterable

import regex as re
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform

from .heads import AcceptRatePredictor, VanillaMarkovHead

logger = init_logger(__name__)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


import math
import os
import sys
from functools import lru_cache


@lru_cache(4)
def _freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow,
               device):
    # Exact port of DeepSeek reference precompute_freqs_cis (YaRN).
    def corr_dim(nr):
        return dim * math.log(original_seq_len / (nr * 2 * math.pi)) / (2 * math.log(base))

    def corr_range(lo, hi):
        return max(math.floor(corr_dim(lo)), 0), min(math.ceil(corr_dim(hi)), dim - 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        lo, hi = corr_range(beta_fast, beta_slow)
        if lo == hi:
            hi += 0.001
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - lo) / (hi - lo), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs).to(device)


def _apply_rope(x, freqs_cis, inverse=False):
    # Exact port of DeepSeek reference apply_rotary_emb (in-place on x).
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    fc = freqs_cis.conj() if inverse else freqs_cis
    if xc.ndim == 3:
        fc = fc.view(1, xc.size(1), xc.size(-1))
    else:
        fc = fc.view(1, xc.size(1), 1, xc.size(-1))
    y.copy_(torch.view_as_real(xc * fc).flatten(-2))
    return y


class DSparkCrossAttention(nn.Module):
    """Eager dense MLA block cross-attention (V4 projection layout).

    Query = the block (noise/bonus) tokens; K/V latent = kv_norm(wkv(.)) of the
    projected target context concatenated with the block. Mirrors DeepSeek's
    reference DSparkAttention.forward but dense (no sparse_attn / no paged cache).
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        qc = vllm_config.quant_config
        self._prefix = prefix
        self.eps = cfg.rms_norm_eps
        self.n_heads = cfg.num_attention_heads
        tp = get_tensor_model_parallel_world_size()
        self.n_local_heads = self.n_heads // tp
        self.q_lora_rank = cfg.q_lora_rank
        self.o_lora_rank = cfg.o_lora_rank
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.qk_rope_head_dim
        self.n_groups = cfg.o_groups
        self.n_local_groups = self.n_groups // tp
        self.softmax_scale = self.head_dim ** -0.5

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            cfg.hidden_size, [self.q_lora_rank, self.head_dim], bias=False,
            quant_config=qc, prefix=f"{prefix}.fused_wqa_wkv", disable_tp=True)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False,
            quant_config=qc, return_bias=False, prefix=f"{prefix}.wq_b")
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank, bias=False, quant_config=qc,
            return_bias=False, prefix=f"{prefix}.wo_a")
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank, cfg.hidden_size, bias=False,
            quant_config=qc, return_bias=False, prefix=f"{prefix}.wo_b")
        self.attn_sink = nn.Parameter(
            torch.full((self.n_local_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False)

        # Rope: DSpark stages are compress_ratio==0, which in the DeepSeek
        # reference (Attention.__init__) DISABLES YaRN (original_seq_len=0) and
        # uses the plain base rope_theta. YaRN's frequency interpolation damps
        # short-range rotation, which would make the closely-spaced block
        # positions (distances 1..block_size) nearly indistinguishable and
        # collapse every draft position to the same prediction. So orig=0 here.
        self.rope_base = float(getattr(cfg, "rope_theta", 10000.0))
        self.rope_orig = 0          # YaRN OFF for the sliding-window draft attn
        self.rope_factor = 1.0
        self.beta_fast = 32.0
        self.beta_slow = 1.0
        # For cudagraph capture: fixed position layout + a precomputed rope table.
        self.window_size = int(getattr(cfg, "sliding_window", 128) or 128)
        self.block_size = int(getattr(cfg, "dspark_block_size", 5))
        self._rope_full = None

    def _rope_table(self, max_len, device):
        return _freqs_cis(self.rope_head_dim, max_len, self.rope_orig, self.rope_base,
                          self.rope_factor, self.beta_fast, self.beta_slow, device)

    def _wo_a_weight(self):
        # vLLM keeps wo_a as block-fp8; DeepSeek's reference wo_a is bf16 (convert
        # dequantizes it). Block(128)-dequantize once: w_real = w_fp8 * scale.
        cached = getattr(self, "_wo_a_bf16", None)
        if cached is not None:
            return cached
        w = self.wo_a.weight                                   # may be 3D (is_bmm)
        scale = getattr(self.wo_a, "weight_scale_inv", None)
        if w.dtype == torch.float8_e4m3fn and scale is not None:
            orig = w.shape
            w2 = w.reshape(-1, w.shape[-1]).float()            # [out, in]
            out, inn = w2.shape

            def deq(s):  # s: [rows, cols] block scales; derive block sizes
                if s.dim() != 2 or out % s.shape[0] or inn % s.shape[1]:
                    return None
                bo, bi = out // s.shape[0], inn // s.shape[1]
                wf = w2.unflatten(0, (s.shape[0], bo)).unflatten(-1, (s.shape[1], bi))
                return (wf * s[:, None, :, None]).flatten(2, 3).flatten(0, 1)

            s2 = scale.float().reshape(-1, scale.shape[-1])
            res = deq(s2)
            if res is None:
                res = deq(s2.t())
            if res is None:
                logger.warning("DSpark wo_a: can't align scale %s to w %s",
                               tuple(scale.shape), tuple(w2.shape))
                res = w2
            w = res.to(torch.bfloat16).reshape(orig)
        else:
            w = w.to(torch.bfloat16)
        self._wo_a_bf16 = w
        return w

    # Cross-attention context, threaded by the model before each stage runs
    # (the V4 decoder layer's forward only passes (positions, x, scaling)).
    _main_x: torch.Tensor | None = None
    _ctx_positions: torch.Tensor | None = None
    _ctx_mask: torch.Tensor | None = None

    def forward(self, positions, x, _scaling=None):
        # V4-decoder-layer-compatible signature. x:[B*Tb, dim] block stream (post
        # attn_norm), flattened over the request batch B; per-request context is
        # self._main_x [B, Tc, dim] (each req's window) + self._ctx_mask [B, Tc].
        # rope positions (ctx/blk) are SHARED across the batch (relative rope,
        # fixed layout). Everything below is already batched over the leading `b`.
        Tb = self.block_size
        B = x.shape[0] // Tb
        x_block = x.view(B, Tb, -1)                     # [B, Tb, dim]
        main_x = self._main_x                          # [B, Tc, dim]
        ctx_positions = self._ctx_positions
        blk_positions = positions
        rd = self.rope_head_dim
        dev = x_block.device
        # Precompute the rope table ONCE during warmup (before cudagraph capture)
        # for the FIXED position layout; indexing it does no host sync, so the
        # draft forward is graph-capturable. The old int(max(...)) did a .item()
        # host sync — illegal mid-capture — and its lru_cache key changed every
        # decode step (positions advance), reallocating the table each call.
        if self._rope_full is None or self._rope_full.device != dev:
            self._rope_full = self._rope_table(
                self.window_size + self.block_size + 1, dev)
        tbl = self._rope_full
        fc_ctx, fc_blk = tbl[ctx_positions], tbl[blk_positions]

        main_lat = self.fused_wqa_wkv(main_x)[0][..., self.q_lora_rank:]
        main_kv = self.kv_norm(main_lat)                       # [B,Tc,head_dim]
        _apply_rope(main_kv[..., -rd:], fc_ctx)

        fused = self.fused_wqa_wkv(x_block)[0]
        qa = fused[..., :self.q_lora_rank]
        q = self.wq_b(self.q_norm(qa))
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))   # [B,Tb,H,hd]
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        _apply_rope(q[..., -rd:], fc_blk)
        blk_kv = self.kv_norm(fused[..., self.q_lora_rank:])    # [B,Tb,head_dim]
        _apply_rope(blk_kv[..., -rd:], fc_blk)

        kv = torch.cat([main_kv, blk_kv], dim=1)               # [B,Tc+Tb,head_dim]
        Tc = main_kv.shape[1]
        scores = torch.einsum("bqhd,bkd->bhqk", q, kv) * self.softmax_scale
        if self._ctx_mask is not None:
            # Additive mask over the Tc context columns (0 valid / -inf pad), PER
            # REQUEST [B, Tc]; the block columns stay unmasked. Padding rows of
            # main_x are zero so their scores are 0 -> 0 + -inf = -inf cleanly (no
            # nan from inf-inf). A fully-masked slot (a padded batch lane) still
            # attends to its own block columns + sink, so it's non-nan and ignored.
            m = scores.new_zeros(B, scores.shape[-1])          # [B, Tc+Tb]
            m[:, :self._ctx_mask.shape[1]] = self._ctx_mask
            scores = scores + m.view(B, 1, 1, -1)
        sink = self.attn_sink.view(1, -1, 1, 1).expand(scores.size(0), -1, scores.size(2), 1)
        attn = torch.softmax(torch.cat([scores, sink], dim=-1).float(), dim=-1).to(q.dtype)
        attn = attn[..., :-1]                                  # drop sink col (value 0)
        o = torch.einsum("bhqk,bkd->bqhd", attn, kv)
        _apply_rope(o[..., -rd:], fc_blk, inverse=True)

        o = o.reshape(B, Tb, self.n_local_groups, -1)
        woa = self._wo_a_weight().view(self.n_local_groups, self.o_lora_rank, -1)
        oa = torch.einsum("bsgd,grd->bsgr", o, woa.to(o.dtype))
        ret = self.wo_b(oa.flatten(2))                 # [B, Tb, dim]
        return ret.reshape(B * Tb, -1)                 # [B*Tb, dim]


def _hc_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps):
    # Plain-torch port of DeepSeek's hc_split_sinkhorn (kernel.py). mixes:[T,(2+hc)*hc]
    # fp32, hc_scale:[3], hc_base:[(2+hc)*hc]. Returns pre[T,hc], post[T,hc], comb[T,hc,hc].
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps                  # row softmax
    comb = comb / (comb.sum(-2, keepdim=True) + eps)          # col norm
    for _ in range(iters - 1):                                # sinkhorn iterations
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _hc_pre(x, hc_fn, hc_scale, hc_base, norm_eps, hc, iters, hc_eps):
    # DeepSeek Block.hc_pre (reduce hc_mult streams -> 1). x:[T, hc, dim].
    xf = x.flatten(1).float()                                 # [T, hc*dim]
    rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = torch.nn.functional.linear(xf, hc_fn) * rs        # [T, (2+hc)*hc]
    pre, post, comb = _hc_sinkhorn(mixes, hc_scale, hc_base, hc, iters, hc_eps)
    y = (pre.unsqueeze(-1) * x.float()).sum(1)                # [T, dim]
    return y.to(x.dtype), post, comb


def _hc_post(x, residual, post, comb):
    # DeepSeek Block.hc_post (expand 1 -> hc_mult). x:[T,dim], residual:[T,hc,dim].
    new = post.unsqueeze(-1) * x.float().unsqueeze(-2)               # [T,hc,dim]
    mixed = (comb.unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(1)  # [T,hc,dim]
    return (new + mixed).to(x.dtype)


def _block_div(t):
    # DSpark differentiation diagnostic (mirrors the reference forward_spec _div):
    # max abs spread of each block position's features around the block mean, and
    # the overall norm. rel = spread/norm; ~0 means the block positions collapsed
    # to the same prediction (kills accept beyond position 0). t: [block, hc, dim].
    f = t.float().flatten(1)
    return float((f - f.mean(0, keepdim=True)).abs().max()), float(f.norm())


class DeepseekV4DSparkDraftModel(nn.Module):
    """3-stage DSpark draft. Reuses V4 MoE per stage; eager cross-attention.

    The mHC (hyper-connection residual mixing) is done in PLAIN TORCH here, NOT
    via the vLLM decoder layer's tilelang mhc kernel: that kernel produces a
    garbage ~1e15 residual for the draft's tiny 5-token eager invocation (it's
    tuned for the target's full-sequence forward), which destroys the
    per-position differentiation. We reuse only the layer's attn (swapped) +
    MoE + norms + loaded hc params, and orchestrate the stage like DeepSeek's
    reference Block.forward."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        self.config = cfg
        self.quant_config = vllm_config.quant_config
        self.rms_norm_eps = cfg.rms_norm_eps
        self.hc_mult = cfg.hc_mult
        hidden = cfg.hidden_size
        vocab = cfg.vocab_size
        self.block_size = int(getattr(cfg, "dspark_block_size", 5))
        self.target_layer_ids = list(
            getattr(cfg, "dspark_target_layer_ids", [40, 41, 42]))
        self.markov_rank = int(getattr(cfg, "dspark_markov_rank", 256))
        self.noise_token_id = int(getattr(cfg, "dspark_noise_token_id", 128799))
        # DSpark draft = 3 stages (checkpoint mtp.0/mtp.1/mtp.2). The HF config's
        # num_nextn_predict_layers is the BASE model's MTP count (1), NOT the
        # DSpark stage count, so it can't be trusted here — DSpark is 3.
        self.n_stages = int(getattr(cfg, "dspark_n_mtp_layers", 3))

        self.embed_tokens = VocabParallelEmbedding(
            vocab, hidden, prefix=maybe_prefix(prefix, "embed_tokens"))
        self.lm_head = ParallelLMHead(
            vocab, hidden, quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(vocab)
        self.norm = RMSNorm(hidden, eps=cfg.rms_norm_eps)

        # stage 0: target-context projection (DeepSeek main_proj/main_norm).
        self.main_proj = ReplicatedLinear(
            hidden * len(self.target_layer_ids), hidden, bias=False,
            quant_config=self.quant_config, prefix=maybe_prefix(prefix, "main_proj"),
            return_bias=False)
        self.main_norm = RMSNorm(hidden, eps=cfg.rms_norm_eps)

        # hc_head (stage 2): hypercompressed vocab projection before lm_head.
        from vllm.model_executor.layers.mhc import HCHeadOp
        self.hc_eps = cfg.hc_eps
        hc_dim = self.hc_mult * hidden
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False)
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False)
        self.hc_head_op = HCHeadOp()

        # The 3 DSpark stages. Each is a real V4 decoder layer (exact mHC + MoE +
        # weight layout + hc_attn/hc_ffn params) with its attention swapped for the
        # eager DSpark cross-attention. Keyed by ABSOLUTE layer index so both
        # extract_layer_index(prefix) and the mtp.{s}->stages.{N} remap line up.
        from vllm.models.deepseek_v4.nvidia.model import (  # noqa: E501
            DeepseekV4DecoderLayer, make_deepseek_v4_expert_params_mapping,
        )
        self._make_expert_mapping = make_deepseek_v4_expert_params_mapping
        topk_buf = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            int(cfg.index_topk), dtype=torch.int32,
            device=current_platform.device_type)
        aux_streams = (None if current_platform.is_rocm()
                       else [torch.cuda.Stream() for _ in range(3)])
        self.stage_ids = [cfg.num_hidden_layers + s for s in range(self.n_stages)]
        self.stages = nn.ModuleDict()
        for lid in self.stage_ids:
            sp = maybe_prefix(prefix, f"stages.{lid}")
            stage = DeepseekV4DecoderLayer(
                vllm_config, prefix=sp, topk_indices_buffer=topk_buf,
                aux_stream_list=aux_streams)
            # Swap the paged MLA+indexer attention for the eager DSpark cross-attn.
            stage.attn = DSparkCrossAttention(vllm_config, prefix=f"{sp}.attn")
            self.stages[str(lid)] = stage

        self.markov_head = VanillaMarkovHead(vocab_size=vocab, markov_rank=self.markov_rank)
        self.confidence_head = AcceptRatePredictor(input_dim=hidden + self.markov_rank)

        # DSPARK_DIV=1: one-shot block-differentiation trace (eager only — the
        # _block_div .item() host syncs would break cudagraph capture).
        self._div_dbg = os.environ.get("DSPARK_DIV", "0") == "1"
        self._div_done = False

    # --- proposer interface --------------------------------------------------

    def combine_hidden_states(self, target_hidden: torch.Tensor) -> torch.Tensor:
        # [T, 3*dim] -> [T, dim]; DeepSeek main_norm(main_proj(.)).
        return self.main_norm(self.main_proj(target_hidden))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, target_hidden, anchor_id, ctx_positions, blk_positions,
                ctx_mask=None):
        """Batched backbone: project the per-request target context, build the
        noise block, run the 3 DSpark stages (V4 mHC + MoE, eager cross-attention),
        hc_head + norm.

        target_hidden:[B, Tc, 3*dim] per-request aux (layers 40/41/42, mean over
        hc_mult). anchor_id:[B] bonus token ids (block position 0). ctx/blk_positions:
        [Tc]/[block_size] positions for rope, SHARED across the batch (FIXED layout
        for graphing). ctx_mask:[B, Tc] additive mask (0 valid / -inf pad) per
        request. Returns (block_logits [B, block, vocab], block_hidden [B, block,
        dim]). The mHC/MoE/head ops run per-token over the flattened [B*block]
        stream; the cross-attention reshapes to [B, block, dim] internally to attend
        each request to its own window. Sampling is the proposer's job. Mirrors
        reference forward_spec + forward_embed + forward_head(backbone part).
        """
        B = target_hidden.shape[0]
        bs = self.block_size
        main_x = self.combine_hidden_states(target_hidden)        # [B, Tc, dim]
        for lid in self.stage_ids:
            attn = self.stages[str(lid)].attn
            attn._main_x = main_x
            attn._ctx_positions = ctx_positions
            attn._ctx_mask = ctx_mask

        draft_ids = torch.full((B, bs), self.noise_token_id,
                               dtype=torch.long, device=main_x.device)
        draft_ids[:, 0] = anchor_id
        x = self.embed_tokens(draft_ids)                          # [B, block, dim]
        x = x.unsqueeze(-2).repeat(1, 1, self.hc_mult, 1)        # [B, block, hc_mult, dim]
        x = x.reshape(B * bs, self.hc_mult, -1)                  # [B*block, hc_mult, dim]

        dbg = self._div_dbg and not self._div_done
        if dbg:
            d, n = _block_div(x[:bs])                             # first req's block
            print(f"[DSpark] div embed: spread={d:.3f} norm={n:.3g} "
                  f"rel={d / max(n, 1e-6):.4f}", file=sys.stderr, flush=True)

        # Run the 3 DSpark stages with PLAIN-TORCH mHC (see class docstring): each
        # stage mirrors DeepSeek's Block.forward — hc_pre -> attn_norm -> eager
        # cross-attn -> hc_post -> hc_pre -> ffn_norm -> MoE -> hc_post. x carries
        # the full [block, hc_mult, dim] h between stages (no vLLM split threading).
        for si, lid in enumerate(self.stage_ids):
            stg = self.stages[str(lid)]
            it = stg.hc_sinkhorn_iters
            residual = x
            xp, post, comb = _hc_pre(x, stg.hc_attn_fn, stg.hc_attn_scale,
                                     stg.hc_attn_base, self.rms_norm_eps,
                                     self.hc_mult, it, self.hc_eps)
            xp = stg.attn_norm(xp)
            xa = stg.attn(blk_positions, xp, None)               # DSparkCrossAttention
            x = _hc_post(xa, residual, post, comb)
            residual = x
            xp, post, comb = _hc_pre(x, stg.hc_ffn_fn, stg.hc_ffn_scale,
                                     stg.hc_ffn_base, self.rms_norm_eps,
                                     self.hc_mult, it, self.hc_eps)
            xp = stg.ffn_norm(xp)
            xf = stg.ffn(xp, None)                               # MoE
            x = _hc_post(xf, residual, post, comb)
            if dbg:
                d, n = _block_div(x[:bs])
                print(f"[DSpark] div stage {si}: spread={d:.3f} norm={n:.3g} "
                      f"rel={d / max(n, 1e-6):.4f}", file=sys.stderr, flush=True)
        if self._div_dbg:
            self._div_done = True
        # x is now the full [B*block, hc_mult, dim] h.
        h = self.hc_head_op(x, self.hc_head_fn, self.hc_head_scale,
                            self.hc_head_base, self.rms_norm_eps, self.hc_eps)  # [B*block,dim]
        h = self.norm(h)
        logits = self.logits_processor(self.lm_head, h)           # [B*block, vocab]
        return logits.view(B, bs, -1), h.view(B, bs, -1)

    @staticmethod
    def _sample(logits, temperature):
        if temperature == 0:
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits.float() / max(temperature, 1e-5), dim=-1)
        return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)

    def sample_block(self, logits, hidden, anchor_id, temperature=0.0):
        """Batched autoregressive markov-biased block sampling + confidence, per
        the reference forward_head. logits/hidden:[B,block,vocab]/[B,block,dim];
        anchor_id:[B]. Returns (out_ids [B,block+1] = [anchor, d1..d_block],
        biased_logits [B,block,vocab], confidence [B,block]). The block axis is
        sequential (markov bias chains position i -> i+1); the request axis B
        batches — fixed-shape, so the whole thing stays cudagraph-capturable."""
        B = logits.shape[0]
        out = logits.new_empty(B, self.block_size + 1, dtype=torch.long)
        out[:, 0] = anchor_id
        biased = logits.clone()
        embeds = []
        for i in range(self.block_size):
            emb = self.markov_head.markov_w1(out[:, i:i + 1].long())  # [B, 1, rank]
            bias = self.markov_head.markov_w2(emb).squeeze(1)         # [B, vocab]
            biased[:, i] = logits[:, i] + bias
            embeds.append(emb.squeeze(1))                             # [B, rank]
            out[:, i + 1] = self._sample(biased[:, i], temperature)   # [B]
        me = torch.stack(embeds, dim=1).to(hidden.dtype)             # [B, block, rank]
        conf = self.confidence_head(torch.cat([hidden, me], dim=-1))  # [B, block]
        return out, biased, conf

    # --- weight loading ------------------------------------------------------

    def _rewrite(self, name: str) -> str | None:
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m:
            s, tail = int(m.group(1)), m.group(2)
            if tail.startswith("main_proj"):
                return "main_proj." + tail.split(".", 1)[1]
            if tail.startswith("main_norm"):
                return "main_norm." + tail.split(".", 1)[1]
            if tail.startswith(("markov_head.", "confidence_head.")):
                return tail
            if tail == "norm.weight":
                return "norm.weight"
            if tail.startswith("hc_head_"):
                return tail
            return f"stages.{self.config.num_hidden_layers + s}.{tail}"
        if name.startswith("embed."):
            return "embed_tokens." + name[len("embed."):]
        if name.startswith("head."):
            return "lm_head." + name[len("head."):]
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        stacked = [("gate_up_proj", "w1", 0),
                   ("gate_up_proj", "w3", 1),
                   ("attn.fused_wqa_wkv", "attn.wq_a", 0),
                   ("attn.fused_wqa_wkv", "attn.wkv", 1)]
        tp = get_tensor_model_parallel_world_size()
        rk = get_tensor_model_parallel_rank()
        nlh = self.config.num_attention_heads // tp
        h0, h1 = nlh * rk, nlh * (rk + 1)
        emap = self._make_expert_mapping(self.config.n_routed_experts)
        esuf = ".weight_scale" if getattr(self.config, "expert_dtype", "fp4") == "fp4" \
            else ".weight_scale_inv"
        for raw, w in weights:
            name = self._rewrite(raw)
            if name is None:
                continue
            if name.endswith(".scale"):
                suf = esuf if _EXPERT_SCALE_RE.search(raw) else ".weight_scale_inv"
                name = name.removesuffix(".scale") + suf
            for pn, wn, sid in stacked:
                # "w1"/"w3" are gate_up_proj shards for shared_experts only;
                # exclude routed experts (own mapping) and markov_head (markov_w1
                # would mis-match "w1" -> markov_gate_up_proj).
                if ".experts." in name or "markov" in name or wn not in name:
                    continue
                name = name.replace(wn, pn)
                params[name].weight_loader(params[name], w, sid)
                loaded.add(name)
                break
            else:
                if ".experts." in name:
                    if "weight_scale" in name and w.dtype == torch.float8_e8m0fnu:
                        w = w.view(torch.uint8)
                    for pn, wn, eid, esid in emap:
                        if wn not in name:
                            continue
                        nm = name.replace(wn, pn)
                        wl = typing.cast(Callable[..., bool], params[nm].weight_loader)
                        if wl(params[nm], w, nm, shard_id=esid, expert_id=eid,
                              return_success=True):
                            loaded.add(nm)
                            break
                    continue
                if "attn_sink" in name:
                    if name not in params:
                        logger.warning("DSpark MISS attn_sink %s -> %s; attn keys: %s",
                                       raw, name,
                                       [k for k in params if name.rsplit(".", 1)[0] in k][:12])
                        continue
                    nw = w[h0:h1]
                    params[name][:nw.shape[0]].copy_(nw)
                    loaded.add(name)
                    continue
                if ".shared_experts.w2" in name:
                    name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
                if name not in params:
                    logger.warning("DSpark: unmapped %s -> %s", raw, name)
                    continue
                ld = getattr(params[name], "weight_loader", default_weight_loader)
                ld(params[name], w)
                loaded.add(name)
        for st in self.stages.values():
            st.ffn.finalize_mega_moe_weights()
            # Dequantize wo_a to bf16 from the RAW checkpoint fp8 NOW and cache it.
            # vLLM's process_weights_after_loading (runs AFTER this) reformats
            # self.wo_a.weight/weight_scale_inv, after which the in-forward dequant
            # produces ~1e14 garbage (-> 1e15 attention output -> collapsed draft).
            st.attn._wo_a_weight()
        logger.info("DSpark draft loaded: %d params", len(loaded))
        return loaded
