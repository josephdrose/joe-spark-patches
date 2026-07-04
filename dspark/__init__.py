# DSpark speculative decoding for DeepSeek-V4-Flash, out-of-tree for vLLM.
#
# Importing this package (PYTHONPATH=/opt/dspark in the serving image):
#   1. applies the V4 target aux-hidden-capture patch (piece 1),
#   2. registers the DSpark draft model under a synthetic arch name,
#   3. exposes DSparkProposer for the custom_class spec-decode path.
#
#   --speculative-config '{"method":"custom_class",
#                          "model":"dspark.proposer.DSparkProposer",
#                          "num_speculative_tokens":5}'
from __future__ import annotations

from vllm import ModelRegistry

from .proposer import DSparkProposer
from .v4_aux_patch import apply_patch

# Patch the V4 target to capture aux hidden at the dspark_target_layer_ids. Must
# run before the target model is built; package import (image start) is early
# enough. Loud on failure — a missing patch silently disables DSpark.
apply_patch()

ModelRegistry.register_model(
    "DeepseekV4DSparkForCausalLM", "dspark.model:DeepseekV4DSparkDraftModel"
)

__all__ = ["DSparkProposer"]
