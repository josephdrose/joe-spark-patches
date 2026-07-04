# Pure (vllm-free) DSML tool-call boundary logic for the DeepSeek V3/V4
# tool-aware reasoning parser. Kept as a standalone, dependency-free module so
# it is unit-testable WITHOUT vLLM/torch/CUDA, and imported by BOTH:
#   - dsv4-reasoning-tool-aware.py  (-> vllm/reasoning/deepseek_v3_reasoning_parser.py)
#   - tests/test_dsv4_dsml_split.py
# The Dockerfile (dsv4-vllm-patched.Dockerfile) copies this in next to the
# parser so the relative import resolves in the image. NO third-party imports —
# keep it that way or the test (and the standalone import fallback) breaks.

# DeepSeek tool-call start markers. V4 uses tool_calls; V3.2 uses
# function_calls. Closing tags are NOT terminators — once a start tag fires,
# everything from there onward is content.
TOOL_START_TAGS: tuple[str, ...] = (
    "<｜DSML｜tool_calls>",
    "<｜DSML｜function_calls>",
)

# MTP (speculative decoding) intermittently drops the leading '<' of the start
# tag at a draft-rejection boundary when the model jumps STRAIGHT to a tool call
# with no reasoning — leaving a "headless" ｜DSML｜tool_calls> in the stream, so
# the tool parser (which matches "<｜DSML｜tool_calls>") never fires and the call
# leaks into content. The headless remainder is unambiguous — the model never
# emits ｜DSML｜…> except to open a tool call — so we recognize it and REPAIR the
# dropped '<' (see dsml_split) so the downstream tool parser still matches.
# Live A/B 2026-06-14: MTP-on leaked ~25% of no-thinking tool calls, all this
# exact headless format; MTP-off was 0. This repair keeps MTP on (~41 tok/s)
# without the leak. vLLM MTP + #36654 / #41132.
_HEADLESS_TAGS: tuple[str, ...] = tuple(t[1:] for t in TOOL_START_TAGS)
_ALL_TAGS: tuple[str, ...] = TOOL_START_TAGS + _HEADLESS_TAGS


def earliest_tool_tag_pos(text: str) -> int:
    """Index of the earliest tool-call start tag (full or MTP-headless) in
    ``text``, or -1 if none."""
    positions = [p for p in (text.find(t) for t in _ALL_TAGS) if p >= 0]
    return min(positions) if positions else -1


def trailing_partial_tag_len(text: str) -> int:
    """Longest suffix of ``text`` that is a strict prefix of a tool-call start
    tag (full OR MTP-headless) — i.e. a start tag that may still be mid-arrival
    across stream chunks. Returns 0 if the tail cannot be the beginning of one."""
    best = 0
    for tag in _ALL_TAGS:
        limit = min(len(text), len(tag) - 1)
        for k in range(limit, 0, -1):
            if text.endswith(tag[:k]):
                best = max(best, k)
                break
    return best


def tag_present_or_forming(text: str) -> bool:
    """True if a full tool start tag is present OR a partial one is still
    arriving at the tail. When False, the text is ordinary reasoning/content
    and the caller can defer to the upstream reasoning parser untouched."""
    return earliest_tool_tag_pos(text) >= 0 or trailing_partial_tag_len(text) > 0


def dsml_split(text: str) -> tuple[str, str]:
    """Split cumulative ``text`` into (reasoning, content). Once a tool start tag
    appears, everything from it onward is content. A full tag is preferred; a
    headless (MTP-dropped-'<') tag is repaired by prepending the '<' so content
    carries the canonical start marker for the tool parser. Before any tag, a
    trailing partial-tag prefix is WITHHELD from reasoning (lands in neither
    half) until it completes or is ruled out — the tag is never torn across the
    reasoning/content boundary mid-stream."""
    # Prefer the full tag (with '<') — its '<' sits one char before the headless
    # match, so checking full first avoids mis-flagging an intact tag as headless.
    for tag in TOOL_START_TAGS:
        p = text.find(tag)
        if p >= 0:
            return text[:p], text[p:]
    for tag in _HEADLESS_TAGS:
        p = text.find(tag)
        if p >= 0:
            return text[:p], "<" + text[p:]  # repair MTP's dropped '<'
    hold = trailing_partial_tag_len(text)
    return (text[: len(text) - hold] if hold else text), ""


def stream_deltas(previous_text: str, current_text: str) -> tuple[str, str]:
    """Per-delta ``(reasoning_delta, content_delta)`` for the DSML path,
    computed by diffing the withholding split of previous_text vs current_text.

    The point: when the start tag finishes arriving, the WHOLE marker — including
    any prefix withheld on earlier deltas — lands in the content delta in one
    piece, so the streaming tool-call parser sees the full ``<｜DSML｜tool_calls>``
    start tag and fires structured tool_calls. The old code emitted the partial
    prefix as reasoning, leaving the tool parser only the tail (e.g. ``_calls>``)
    which never matched the start token → the block leaked as content.
    vLLM #36654 / #41132."""
    prev_reasoning, prev_content = dsml_split(previous_text)
    curr_reasoning, curr_content = dsml_split(current_text)
    return (
        curr_reasoning[len(prev_reasoning):],
        curr_content[len(prev_content):],
    )


# Reasoning-end close tag. Once reasoning has ended (upstream consumed the FIRST
# </think>), a further </think> in the CONTENT channel is never legitimate output.
THINK_CLOSE = "</think>"


def filter_think_close(pending: str, new_content: str) -> tuple[str, str]:
    """Strip stray ``</think>`` close tags from a CONTENT delta, withholding a
    trailing partial across stream chunks. Returns ``(emit, new_pending)``.

    WHY: the model intermittently emits a SECOND/stray ``</think>`` after reasoning
    has already ended — a degeneration at high context, made VISIBLE by DSpark
    spec-decode: at one-token-per-delta the stray close token (id 128822) is
    dropped, but when DSpark accepts a multi-token chunk the literal ``</think>``
    rides into the content channel and leaks to the user (e.g. ``\\n\\n</think>\\n\\n``
    sitting in front of a tool call, or a truncated ``\\n\\n</`` at EOS). Reasoning
    is over, so ``</think>`` here is never real content — drop complete tags and
    hold a trailing partial (``</``, ``</th``…) until it completes or EOS discards
    it. The tool-call parser reads cumulative ``current_text``, NOT this channel,
    so stripping here cannot affect tool_calls. vLLM #36654 / #41132."""
    buf = pending + new_content
    buf = buf.replace(THINK_CLOSE, "")
    hold = 0
    for k in range(min(len(buf), len(THINK_CLOSE) - 1), 0, -1):
        if buf.endswith(THINK_CLOSE[:k]):
            hold = k
            break
    if hold:
        return buf[: len(buf) - hold], buf[len(buf) - hold:]
    return buf, ""
