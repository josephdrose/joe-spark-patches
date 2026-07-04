# DSpark draft heads for serving DeepSeek-V4-Flash-DSpark in vLLM.
#
# Ported from DeepSeek's DeepSpec (github.com/deepseek-ai/DeepSpec):
#   - VanillaMarkov            <- deepspec/modeling/dspark/markov_head.py
#   - AcceptRatePredictor      <- deepspec/modeling/dspark/common.py
#   - confident_prefix_length  <- deepspec/eval/dspark/draft_ops.py (_confident_prefix_length)
#
# The DeepSeek-V4-Flash-DSpark checkpoint carries the *vanilla* Markov head
# (only mtp.2.markov_head.markov_w1/markov_w2 are present — no gate_proj /
# joint_proj), so the gated/RNN variants in DeepSpec are intentionally omitted.
from __future__ import annotations

import torch
from torch import nn


class VanillaMarkovHead(nn.Module):
    """Low-rank Markov bias over the vocabulary, conditioned on the previous token.

    Each draft position's logits get a bias  W2 @ W1[prev_token]  where
    W1: Embedding(vocab, rank) and W2: Linear(rank, vocab). This is what makes
    block drafting cheap: one embedding lookup + one (rank x vocab) matmul per
    step instead of a full LM head pass.
    """

    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        assert markov_rank > 0, f"markov_rank must be > 0, got {markov_rank}"
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def step_bias(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        """Markov logit bias for the tokens that FOLLOW prev_token_ids.

        prev_token_ids: [...]   ->   bias: [..., vocab_size]
        """
        return self.markov_w2(self.markov_w1(prev_token_ids.long()))

    def apply_step_logits(
        self, base_logits: torch.Tensor, prev_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """base_logits[..., vocab] + Markov bias for the next position."""
        return base_logits + self.step_bias(prev_token_ids)

    @torch.no_grad()
    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        sample_step,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference autoregressive block sampler (parity with DeepSpec).

        Drives the block one position at a time, feeding each sampled token back
        as the Markov condition for the next. vLLM's proposer will instead call
        ``step_bias`` inside its own captured draft loop, but this mirrors
        deepspec sample_block_tokens for correctness testing.

        base_logits:          [batch, block, vocab]  (target backbone logits)
        first_prev_token_ids: [batch]                 (token before the block)
        sample_step(logits)   -> [batch] token ids    (caller's sampler)

        returns (sampled_tokens [batch, block], corrected_logits [batch, block, vocab])
        """
        batch, block = base_logits.shape[:2]
        if block == 0:
            empty = base_logits.new_empty((batch, 0), dtype=torch.long)
            return empty, base_logits

        sampled, corrected = [], []
        prev = first_prev_token_ids.long()
        for k in range(block):
            step_logits = self.apply_step_logits(base_logits[:, k, :], prev)
            corrected.append(step_logits.unsqueeze(1))
            prev = sample_step(step_logits)
            sampled.append(prev)
        return torch.stack(sampled, dim=1), torch.cat(corrected, dim=1)


class AcceptRatePredictor(nn.Module):
    """Confidence head: per-position accept-probability logit (DSpark adaptive block).

    A single Linear over the draft features -> one scalar logit per draft
    position. sigmoid(logit) estimates whether that position will be accepted by
    the target; the block is truncated to the confident prefix (see
    confident_prefix_length). Matches mtp.2.confidence_head.proj.weight.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [..., input_dim]  ->  logit: [...]
        return self.proj(features).squeeze(-1)


def confident_prefix_length(
    confidence_logits: torch.Tensor, *, block_size: int, threshold: float
) -> int:
    """Length of the leading prefix whose confidence stays >= threshold.

    confidence_logits: [block] (or [1, block]) for a single sequence.
    Returns block_size if every position clears the bar, else the index of the
    first position that drops below it (so that position and the rest are
    dropped from the proposal). threshold == 0 disables truncation.
    """
    if threshold <= 0.0:
        return int(block_size)
    logits = confidence_logits.reshape(-1)
    below = logits.sigmoid() < threshold
    if not bool(below.any()):
        return int(block_size)
    return int(torch.nonzero(below, as_tuple=False)[0].item())
