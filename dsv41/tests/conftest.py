# SPDX-License-Identifier: Apache-2.0
"""Make the unmerged PR's modules importable under their real package names.

The container's vLLM predates PR #56214, so `vllm.models.deepseek_v4_1` and
`vllm.config.engram` do not exist. The patched files sitting next to this
conftest are loaded and registered under those names, so
`tests/kernels/test_engram.py` imports exactly what it would import in a
checkout of the PR. Everything else (triton, distributed, platforms) comes
from the installed vLLM.
"""

import importlib.util
import pathlib
import sys
import types

import vllm.config
import vllm.models

HERE = pathlib.Path(__file__).parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    parent, _, leaf = name.rpartition(".")
    setattr(sys.modules[parent], leaf, mod)
    return mod


def _package(name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod
    parent, _, leaf = name.rpartition(".")
    setattr(sys.modules[parent], leaf, mod)
    return mod


_load("vllm.config.engram", HERE / "vllm/config/engram.py")
_package("vllm.models.deepseek_v4_1", HERE / "vllm/models/deepseek_v4_1")
_package(
    "vllm.models.deepseek_v4_1.common", HERE / "vllm/models/deepseek_v4_1/common"
)
_load(
    "vllm.models.deepseek_v4_1.common.engram_disk",
    HERE / "vllm/models/deepseek_v4_1/common/engram_disk.py",
)
_load(
    "vllm.models.deepseek_v4_1.common.engram",
    HERE / "vllm/models/deepseek_v4_1/common/engram.py",
)
