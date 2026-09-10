#!/usr/bin/env python3
"""vlspeed-prestage: stage the disk-backed Engram rows before the forward.

An on-disk Engram table is read by the CPU, so `prepare_embeddings` puts a
host round trip inside the model forward and the forward cannot be captured
in a CUDA graph. Our engram.py already refuses capture with that message and
already carries the seam its docstring describes: `ParallelEngramEmbedding
.prefetch(ids)` starts a read that a later `lookup` for exactly those ids
takes without reading again. It says no caller exists. This adds one.

`EngramDiskStager.stage()` runs from the V2 model state's `prepare_inputs`,
outside the forward. It runs the same `NgramHashState` the forward runs, on
the same step inputs, submits both engram layers' reads (each layer has its
own table and its own reader pool, so the two overlap), then lands the rows
in each layer's persistent `staged_rows`. `prepare_embeddings` then only has
to not undo it, so the forward holds no host call and captures.

Design taken from tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark fix 8
(patch/cudagraph-prestage). The model_state.py wiring is theirs. The stager
body is ours, because our disk reader is not theirs: ours is one interleaved
row file per layer per rank with O_DIRECT and its own worker set, addressed
by global row id, so `gather_dequant_many`, `disk_rel_owned` and the pinned
staging buffers their version needs have no counterpart here.

VL41_ENGRAM_PRESTAGE_VERIFY=N makes `prepare_embeddings` redo the lookup and
compare bitwise against the staged rows for the first N calls, then turn
itself off and say so, so one eager boot gives both the proof and a clean
benchmark. Eager only: under capture this would be the host call the whole
patch exists to remove, and it raises there.

Applies to an installed vLLM tree; pass the dist-packages/vllm path.
"""
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages/vllm"


def sub(path, old, new):
    p = f"{ROOT}/{path}"
    s = open(p).read()
    n = s.count(old)
    assert n == 1, f"{path}: expected 1 occurrence of {old.splitlines()[0]!r}, found {n}"
    open(p, "w").write(s.replace(old, new))
    print(f"  {path}: patched at {old.strip().splitlines()[0]!r}")


print("vlspeed-prestage: patching", ROOT)

ENGRAM = "models/deepseek_v4_1/common/engram.py"

# 1. staged_rows must be finite for tokens the stager never covers: cudagraph
#    padding rows, and the tail of a padded prefill batch. torch.empty is not.
sub(ENGRAM,
    """        max_tokens = get_current_vllm_config().scheduler_config.max_num_batched_tokens
        # Keep lookup results alive across breakable graph segments.
        self.staged_rows = torch.empty(
""",
    """        # vlspeed-prestage: a disk table under the V2 runner is staged before
        # the forward, so the forward holds no host call and can be captured.
        self.prestage = (
            self.embed_tokens.table_path is not None
            and bool(getattr(get_current_vllm_config(), "use_v2_model_runner", False))
            # Off-switch, so an A/B does not need a second image.
            and os.environ.get("VL41_ENGRAM_PRESTAGE", "1") == "1"
        )
        max_tokens = get_current_vllm_config().scheduler_config.max_num_batched_tokens
        # Keep lookup results alive across breakable graph segments. Zeroed
        # because padding tokens are never staged and their rows still get
        # multiplied by wkv.
        self.staged_rows = torch.zeros(
""")

# 2. The forward's hook. Under prestage it must not touch the host.
sub(ENGRAM,
    """    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        \"\"\"Gather this layer's rows on the main stream before decoder layers.

        Consumes a matching `prefetch` when one is outstanding.
        \"\"\"
        self.embed_tokens.lookup(hash_ids, self.staged_rows[: hash_ids.shape[0]])
""",
    """    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        \"\"\"Gather this layer's rows on the main stream before decoder layers.

        Consumes a matching `prefetch` when one is outstanding. Inert when the
        rows were staged before the forward, which is the only way this can be
        reached from inside a CUDA graph capture.
        \"\"\"
        if self.prestage:
            if _VERIFY_LEFT > 0:
                self._verify_staged(hash_ids)
            return
        self.embed_tokens.lookup(hash_ids, self.staged_rows[: hash_ids.shape[0]])

    def _verify_staged(self, hash_ids: torch.Tensor) -> None:
        \"\"\"Read the rows again here and compare against the staged ones.

        The staged rows arrived through `prefetch` on ids the stager hashed
        in prepare_inputs; this is a fresh gather on the ids the forward
        hashed. Equal rows mean the two hashes agree and the answer landed in
        the right buffer. It says nothing about whether either hash is the
        RIGHT hash, or about what is in the row file. Eager mode only.
        \"\"\"
        global _VERIFY_LEFT, _VERIFY_TOKENS
        n = hash_ids.shape[0]
        ref = torch.empty_like(self.staged_rows[:n])
        self.embed_tokens.lookup(hash_ids, ref)
        got = self.staged_rows[:n]
        # bf16 through int16: NaN compares equal to itself, which is what is
        # wanted here. A staged NaN is a bug whether or not it matches.
        if not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
            bad = int((ref.view(torch.int16) != got.view(torch.int16)).any(-1).sum())
            raise RuntimeError(
                f"engram prestage mismatch: layer_hash_index="
                f"{self.layer_hash_index}, {bad} of {n * ref.shape[1]} rows "
                "differ from an in-forward lookup of the same ids"
            )
        _VERIFY_TOKENS += n
        _VERIFY_LEFT -= 1
        if _VERIFY_LEFT == 0:
            logger.info(
                "ENGRAM PRESTAGE VERIFY DONE: %d lookups, %d token-rows, all "
                "bit-exact against an in-forward lookup of the same ids. "
                "Verification is now off for the rest of this process.",
                _VERIFY_CALLS,
                _VERIFY_TOKENS,
            )
""")

# 3. The stager itself.
with open(f"{ROOT}/{ENGRAM}", "a") as f:
    f.write('''

_VERIFY_CALLS = int(os.environ.get("VL41_ENGRAM_PRESTAGE_VERIFY", "0"))
_VERIFY_LEFT = _VERIFY_CALLS
_VERIFY_TOKENS = 0


class EngramDiskStager:
    """Reads a step's disk-backed Engram rows before the model forward runs.

    Built by the V2 model state, which calls `stage` from `prepare_inputs`
    once per step, after the runner has written this step's input ids,
    positions and query_start_loc and gathered the lookback window.

    The hash is recomputed here rather than handed over from the forward,
    because the forward runs after this and is the thing being taken off the
    host. Both call the same `NgramHashState` on the same tensors, so both
    get the same ids; `VL41_ENGRAM_PRESTAGE_VERIFY=N` checks that claim
    against the rows themselves.
    """

    def __init__(self, hash_state: NgramHashState, engrams: list["Engram"]) -> None:
        assert engrams and all(e.embed_tokens.table_path is not None for e in engrams)
        self.hash_state = hash_state
        self.engrams = sorted(engrams, key=lambda e: e.layer_hash_index)
        self.max_tokens = self.engrams[0].staged_rows.shape[0]
        self.num_staged = 0
        logger.info(
            "Engram rows staged before the forward (graph-safe): %d layers, "
            "%d local heads, up to %d tokens per step",
            len(self.engrams),
            self.engrams[0].embed_tokens.part_n_hash_cols,
            self.max_tokens,
        )

    @torch.inference_mode()
    def stage(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        num_tokens: int,
    ) -> int:
        """Stage rows for this step's first `num_tokens` unpadded tokens.

        Returns the number staged, 0 while the KV cache is unbound, which is
        when the forward skips engram too.
        """
        n = min(int(num_tokens), self.max_tokens)
        if n <= 0 or not self.hash_state.ensure_cache():
            return 0
        from .mm_preprocess import image_sentinel_mask

        ids = input_ids[:n]
        hashes = self.hash_state(
            ids,
            positions[:n],
            query_start_loc,
            image_sentinel_mask(ids),
            lookback_token_ids,
            image_sentinel_mask(lookback_token_ids),
            None,
            None,
        )
        # Submit every layer before waiting on any: each layer has its own
        # table, fd and reader pool, so the reads overlap. Issued in one pass
        # and consumed in a second, not prefetch-then-lookup per layer.
        for engram in self.engrams:
            engram.embed_tokens.prefetch(hashes[:, engram.layer_hash_index])
        for engram in self.engrams:
            engram.embed_tokens.lookup(
                hashes[:, engram.layer_hash_index], engram.staged_rows[:n]
            )
        self.num_staged = n
        return n
''')
print(f"  {ENGRAM}: appended EngramDiskStager")

# 4. Wiring, from tonyd2wild patch/cudagraph-prestage/model_state-prestage.diff.
STATE = "models/deepseek_v4_1/nvidia/model_state.py"

sub(STATE,
    """from vllm.config import VllmConfig
from vllm.triton_utils import tl, triton
""",
    """from vllm.config import VllmConfig
from vllm.models.deepseek_v4_1.common.engram import (
    Engram,
    EngramDiskStager,
    NgramHashState,
)
from vllm.triton_utils import tl, triton
""")

sub(STATE,
    """                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )

    def prepare_inputs(
""",
    """                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )

        # vlspeed-prestage: disk-backed Engram rows are read in prepare_inputs,
        # outside the (possibly captured) forward.
        self.engram_stager: EngramDiskStager | None = None
        engrams = [m for m in model.modules() if isinstance(m, Engram) and m.prestage]
        if engrams:
            hash_states = [m for m in model.modules() if isinstance(m, NgramHashState)]
            assert len(hash_states) == 1, (
                f"expected one NgramHashState, found {len(hash_states)}"
            )
            self.engram_stager = EngramDiskStager(hash_states[0], engrams)

    def prepare_inputs(
""")

sub(STATE,
    """        model_inputs["lookback_token_ids"] = window
        return model_inputs
""",
    """        model_inputs["lookback_token_ids"] = window
        if self.engram_stager is not None and input_batch.input_ids is not None:
            # After the runner wrote this step's ids, positions and
            # query_start_loc and after the lookback window above; before the
            # forward or the graph replay.
            positions = model_inputs.get("positions")
            self.engram_stager.stage(
                input_batch.input_ids,
                positions if positions is not None else input_batch.positions,
                input_batch.query_start_loc[: input_batch.num_reqs + 1],
                window,
                input_batch.num_tokens,
            )
        return model_inputs
""")

print("vlspeed-prestage: done")
