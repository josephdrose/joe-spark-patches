# Aux-hidden-state capture patch for vLLM's DeepSeek-V4 target model, for DSpark.
#
# DSpark's draft (stage 0 main_proj) consumes the target's hidden from the
# dspark_target_layer_ids (40,41,42): specifically the MEAN over the hc_mult(=4)
# hyper-connection streams of each layer's post-layer residual, concatenated ->
# [num_tokens, len(layers)*dim]. (DeepSeek reference Transformer.forward:
# `main_hiddens.append(h.mean(dim=2))` collected at the 3 target layer indices.)
#
# vLLM's V4 model (vllm.models.deepseek_v4.nvidia.model:DeepseekV4Model) has NO
# aux-capture hook and carries the inter-layer residual SPLIT across the mHC
# tensors (hidden_states + residual + post_mix + res_mix); the full [T,hc_mult,dim]
# residual is only reconstructed via layer.hc_post(...). So to reproduce the
# reference `h` at a target layer we run hc_post at that layer, mean over the
# hc_mult axis, and stash. This module monkeypatches DeepseekV4Model to add:
#   - set_aux_hidden_state_layers(layers): register the capture indices
#   - get_dspark_aux_hidden(): return the concatenated [T, n*dim] aux tensor
#   - a wrapped forward that captures at the registered layers
#
# NOTE: the runner +1's draft target_layer_ids (dsk gpu_model_runner ~:5247), so
# to capture absolute layers [40,41,42] the draft config carries [39,40,41].
# Imported for side effects (applies the patch) by the dspark package __init__.
from __future__ import annotations

from itertools import islice

import torch
from vllm.distributed import get_pp_group
from vllm.platforms import current_platform

_PATCHED = False


def apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4Model

    def set_aux_hidden_state_layers(self, layers) -> None:
        self._dspark_aux_layers = tuple(sorted(set(int(x) for x in layers)))
        self._dspark_aux_slot = {idx: i
                                 for i, idx in enumerate(self._dspark_aux_layers)}
        # Persistent aux buffer with a STABLE address OUTSIDE the cudagraph pool,
        # allocated here (registration runs in load_model, before graph capture).
        # The per-layer copy_ in forward() then refreshes it correctly under graph
        # replay -- exactly the trick the model uses for _mtp_hidden_buffer. A
        # fresh torch.cat is a graph-pool intermediate that gets recycled before
        # the proposer reads it, which collapses the draft accept rate (~1.0)
        # under EAGER=0 (cudagraph) serving.
        mtp = getattr(self, "_mtp_hidden_buffer", None)
        if mtp is not None and getattr(self, "_dspark_aux_buffer", None) is None:
            dim = mtp.shape[1] // self.hc_mult          # per-layer aux width
            self._dspark_aux_dim = dim
            self._dspark_aux_buffer = torch.empty(
                mtp.shape[0], len(self._dspark_aux_layers) * dim,
                dtype=mtp.dtype, device=mtp.device)
            self._dspark_aux_T = 0

    def get_dspark_aux_hidden(self):
        buf = getattr(self, "_dspark_aux_buffer", None)
        if buf is None:
            return getattr(self, "_dspark_aux_hidden", None)
        return buf[:getattr(self, "_dspark_aux_T", 0)]

    orig_forward = DeepseekV4Model.forward

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None):
        aux_layers = getattr(self, "_dspark_aux_layers", ())
        if not aux_layers:
            return orig_forward(self, input_ids, positions, intermediate_tensors,
                                inputs_embeds)

        # Reimplements the V4 forward (model.py:1507-1549) with capture at the
        # registered layer indices. Kept in lockstep with the base forward.
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)

        buf = getattr(self, "_dspark_aux_buffer", None)
        dim = getattr(self, "_dspark_aux_dim", 0)
        slot = getattr(self, "_dspark_aux_slot", {})
        last_idx = self.end_layer - 1
        residual, post_mix, res_mix = None, None, None
        layer = None
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states, positions, input_ids, post_mix, res_mix, residual,
            )
            # Reconstruct the full [T, hc_mult, dim] residual via hc_post + mean
            # and copy_ it into this layer's slot of the persistent aux buffer.
            # EXCEPT the last layer: its hc_post runs once below, and calling it a
            # second time corrupts the b12x mHC op (IndexError).
            if (idx in aux_layers and idx != last_idx
                    and current_platform.is_cuda() and buf is not None):
                full = layer.hc_post(hidden_states, residual, post_mix, res_mix)
                T = hidden_states.shape[0]
                s = slot[idx]
                buf[:T, s * dim:(s + 1) * dim].copy_(full.mean(dim=1))

        if not get_pp_group().is_last_rank:
            return {"hidden_states": hidden_states} if not isinstance(
                hidden_states, dict) else hidden_states

        if layer is not None and current_platform.is_cuda():
            hidden_states = layer.hc_post(hidden_states, residual, post_mix, res_mix)
            if last_idx in aux_layers and buf is not None:  # capture from the
                T = hidden_states.shape[0]                  # single final hc_post
                s = slot[last_idx]
                buf[:T, s * dim:(s + 1) * dim].copy_(hidden_states.mean(dim=1))
        if buf is not None:
            self._dspark_aux_T = hidden_states.shape[0]
        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
        hidden_states = self.hc_head_op(
            hidden_states, self.hc_head_fn, self.hc_head_scale,
            self.hc_head_base, self.rms_norm_eps, self.hc_eps,
        )
        return self.norm(hidden_states)

    DeepseekV4Model.set_aux_hidden_state_layers = set_aux_hidden_state_layers
    DeepseekV4Model.get_dspark_aux_hidden = get_dspark_aux_hidden
    DeepseekV4Model.forward = forward
    _PATCHED = True
