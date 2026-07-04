# Tool-aware DeepSeek V3/V4 reasoning parser. DROPS IN OVER
# /opt/env/lib/python3.12/site-packages/vllm/reasoning/deepseek_v3_reasoning_parser.py
# via dashboard/sparkrun/dsv4-vllm-patched.Dockerfile.
#
# WHY: DSV4's chat template opens <think> implicitly (the prompt ends inside
# the reasoning region), and the model goes straight from thinking into
# <｜DSML｜tool_calls>... without ever emitting </think>. The upstream V3
# wrapper's "implicit-start" branch assumes everything is reasoning until
# </think> arrives — so it swallows the DSML tool tags before the tool-call
# parser can see them. vLLM tracks this as #36654 (open since 2026-03-10,
# unfixed as of 2026-06-06).
#
# FIX: in both extract_reasoning paths, treat <｜DSML｜tool_calls> (V4) and
# <｜DSML｜function_calls> (V3.2) as alternate reasoning-end markers.
# Whichever terminator appears first in the stream splits the output —
# everything before goes to reasoning, everything from the tag onwards goes
# to content where the tool-call parser extracts structured tool_calls.
#
# Note: the tool-call parser operates on the cumulative `current_text`, not
# the reasoning/content channel split. So even when a tag arrives split
# across SSE chunks and a few chars of the prefix leak into reasoning, the
# tool parser still sees the full tag and fires structured tool_calls
# correctly. The leak is cosmetic only.
#
# Scope is intentionally narrow: ONLY this V3 wrapper changes. The base
# BaseThinkingReasoningParser is untouched (other model families safe).
# The DSR1 parser is untouched (still inherits the original behavior).

import os
import sys
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase

from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

from .identity_reasoning_parser import IdentityReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

logger = init_logger(__name__)

# DSML_TRACE=1: dump per-delta reasoning-split decisions to stderr (interleaves
# with the tool parser's [DSML-t] trace) to debug the reasoning->tool handoff.
_DSML_TRACE = os.environ.get("DSML_TRACE", "0") == "1"
_dsml_trace_n = [0]


def _dlog(msg: str) -> None:
    if _DSML_TRACE and _dsml_trace_n[0] < 400:
        _dsml_trace_n[0] += 1
        print(msg, file=sys.stderr, flush=True)

# DSML tool-call boundary logic lives in a vllm-free sibling module so it is
# unit-testable without vLLM/torch/CUDA (tests/test_dsv4_dsml_split.py). The
# Dockerfile copies dsv4_dsml_split.py in next to this file. Relative import
# works in the image (vllm.reasoning package); flat import is the standalone
# fallback (e.g. when the module is loaded outside the package).
try:
    from .dsv4_dsml_split import (
        earliest_tool_tag_pos as _earliest_tool_tag_pos,
        filter_think_close as _filter_think_close,
        stream_deltas as _stream_deltas,
        tag_present_or_forming as _tag_present_or_forming,
    )
except ImportError:  # pragma: no cover - standalone import fallback
    from dsv4_dsml_split import (  # type: ignore[no-redef]
        earliest_tool_tag_pos as _earliest_tool_tag_pos,
        filter_think_close as _filter_think_close,
        stream_deltas as _stream_deltas,
        tag_present_or_forming as _tag_present_or_forming,
    )


class DeepSeekV3ReasoningParser(ReasoningParser):
    """
    V3 parser that delegates to either DeepSeekR1ReasoningParser or
    IdentityReasoningParser based on `thinking`, with an ADDITIONAL
    DSML tool-tag terminator on top of </think> (see module docstring).
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = bool(chat_kwargs.get("thinking", False))
        enable_thinking = bool(chat_kwargs.get("enable_thinking", False))
        thinking = thinking or enable_thinking

        self._parser: ReasoningParser
        if thinking:
            self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
            self._thinking = True
        else:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)
            self._thinking = False

        # Set once a DSML tool-call start tag has appeared in this stream. Used
        # to force reasoning-end (see is_reasoning_end_streaming) so the serving
        # orchestrator hands the marker to the tool parser even when the model
        # never emits </think> (DSV4 jumps straight from thinking to the tool
        # call). Fresh per request — this parser is instantiated per request.
        self._tool_tag_seen = False

        # Carries a withheld trailing partial </think> across content deltas, so a
        # stray close tag split over chunks (or one truncated by EOS) is stripped
        # from the content channel instead of leaking. See _filter_think_close.
        self._content_pending = ""

    @property
    def reasoning_start_str(self) -> str | None:
        return self._parser.reasoning_start_str

    @property
    def reasoning_end_str(self) -> str | None:
        return self._parser.reasoning_end_str

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._parser.is_reasoning_end(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        # A DSML tool tag IS the end of reasoning for DSV4: the model jumps
        # straight from thinking into <｜DSML｜tool_calls> without ever emitting
        # </think>, so the upstream R1 end-token check never fires. Without this,
        # the serving orchestrator (vllm/parser/abstract_parser.parse_delta) stays
        # in the reasoning phase forever and NEVER hands the marker to the tool
        # parser — so the whole tool call leaks into the content channel. The flag
        # is set by extract_reasoning_streaming the instant the full tag appears,
        # which is called immediately before this in parse_delta. vLLM #36654/#41132.
        if self._tool_tag_seen:
            return True
        return self._parser.is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._parser.extract_content_ids(input_ids)

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        reasoning, content = self._parser.extract_reasoning(model_output, request)

        # Drop a stray </think> the model emitted after reasoning already ended
        # (degeneration — see _filter_think_close). Non-streaming sees the whole
        # output, so a plain strip suffices; </think> is never real content.
        if self._thinking and content:
            content = content.replace("</think>", "") or None

        # Tool-tag awareness only matters in thinking mode AND only when the
        # upstream parser put everything in reasoning (because no </think>
        # was seen). If content is already non-None, </think> fired
        # normally and we don't second-guess.
        if not self._thinking or content is not None or reasoning is None:
            return reasoning, content

        tag_pos = _earliest_tool_tag_pos(reasoning)
        if tag_pos < 0:
            return reasoning, content

        new_reasoning = reasoning[:tag_pos] or None
        new_content = reasoning[tag_pos:] or None
        return new_reasoning, new_content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> "DeltaMessage | None":
        from vllm.entrypoints.openai.engine.protocol import DeltaMessage

        if not self._thinking:
            return self._parser.extract_reasoning_streaming(
                previous_text, current_text, delta_text,
                previous_token_ids, current_token_ids, delta_token_ids,
            )

        # Once a full DSML tool tag has appeared, latch it so
        # is_reasoning_end_streaming (called right after this in parse_delta)
        # forces the serving orchestrator out of the reasoning phase and into
        # tool parsing — DSV4 never emits </think> before a tool call, so the
        # upstream end-token check never fires on its own.
        if _earliest_tool_tag_pos(current_text) >= 0:
            self._tool_tag_seen = True

        result: "DeltaMessage"; case: str
        # Case 1: tool tag already fully visible in previous_text — past the
        # reasoning boundary; everything from here streams as content.
        if _earliest_tool_tag_pos(previous_text) >= 0:
            case = "1"
            result = DeltaMessage(content=delta_text or None)
        # Case 2: no DSML tool tag present OR forming at the tail — ordinary
        # reasoning; defer to upstream (which handles </think> correctly).
        elif not _tag_present_or_forming(current_text):
            case = "2"
            result = self._parser.extract_reasoning_streaming(
                previous_text, current_text, delta_text,
                previous_token_ids, current_token_ids, delta_token_ids,
            )
        else:
            # Case 3: a DSML tool tag is present or still arriving across chunks.
            # Withholding split (dsv4_dsml_split.stream_deltas): any trailing
            # partial of the start tag is held OUT of the reasoning channel, so
            # when the tag completes the WHOLE marker streams to content in one
            # delta and the streaming tool parser fires structured tool_calls —
            # instead of the block leaking as content because the parser only saw
            # the tail (e.g. ``_calls>``). vLLM #36654 / #41132.
            case = "3"
            reasoning_delta, content_delta = _stream_deltas(previous_text, current_text)
            result = DeltaMessage(
                reasoning=reasoning_delta or None,
                content=content_delta or None,
            )

        # Strip a stray </think> that the model emitted AFTER reasoning already
        # ended (a degeneration surfaced by DSpark multi-token deltas — see
        # _filter_think_close). Applies to whatever content this delta produced,
        # in every case. Reasoning channel + tool-call parsing are untouched.
        if getattr(result, "content", None):
            emit, self._content_pending = _filter_think_close(
                self._content_pending, result.content
            )
            result.content = emit or None
        if _DSML_TRACE:
            _dlog(f"[DSML-r c{case}] prev=...{previous_text[-16:]!r} "
                  f"delta={delta_text!r} -> R={getattr(result, 'reasoning', None)!r} "
                  f"C={getattr(result, 'content', None)!r} ttseen={self._tool_tag_seen}")
        return result


class DeepSeekV3ReasoningWithThinkingParser(DeepSeekV3ReasoningParser):
    """
    DeepSeekV3ReasoningParser that defaults to thinking mode.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        enable_thinking = chat_kwargs.get("enable_thinking", None)
        if thinking is None and enable_thinking is None:
            chat_kwargs["thinking"] = True
            chat_kwargs["enable_thinking"] = True
            kwargs["chat_template_kwargs"] = chat_kwargs
        super().__init__(tokenizer, *args, **kwargs)
