"""AI-generated text detector.

Two-agent pipeline that reviews the text of a web page (or pasted text) for
hints of LLM-generated content and returns an inline markdown report: flagged
passages are tagged in the text and explained in a findings list.

Agents:
  1. detector -- per-segment forensic analysis (structured output)
  2. critic   -- reviews and adjusts the detector's findings, writes a summary

All numbers (labels, overall score) are derived in Python from one threshold
function, so the report is deterministic given the LLM output.

Callable from Vibe chat: the input model is tagged @le_chat_input and the
workflow returns ChatAssistantWorkflowOutput whose text renders inline in chat.
"""

import html
import json
import re
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, Field

import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as wm
from mistralai.workflows.plugins.mistralai import (
    ChatCompletionRequest,
    SystemMessage,
    UserMessage,
    chat_parse_to_model,
)

# Bounds that keep every activity payload well under the 2MB platform cap.
MAX_TEXT_CHARS = 120_000
MAX_SEGMENTS = 60
MAX_HTML_BYTES = 2_000_000
MIN_TEXT_CHARS = 400

Label = Literal["ai_likely", "uncertain", "human_likely"]


def label_for(p: float) -> Label:
    """Single source of truth mapping a score to a label."""
    if p >= 0.7:
        return "ai_likely"
    if p >= 0.4:
        return "uncertain"
    return "human_likely"


# ---------------------------------------------------------------------------
# Input / output models
# ---------------------------------------------------------------------------


@wm.le_chat_input
class AIDetectionInput(BaseModel):
    """Input for the AI text detector, shown as a form in Vibe chat."""

    url: str | None = Field(default=None, description="URL of the web page to analyze")
    text: str | None = Field(default=None, description="Or paste the raw text directly")


class Segment(BaseModel):
    index: int
    text: str


class SegmentAnalysis(BaseModel):
    index: int
    ai_probability: float
    label: Label
    reasons: list[str] = Field(default_factory=list)
    flagged_phrases: list[str] = Field(default_factory=list)


class DetectionBatch(BaseModel):
    segments: list[SegmentAnalysis]


class SegmentAdjustment(BaseModel):
    """Critic's change to one segment (label is derived from the score in code)."""

    index: int
    ai_probability: float
    reasons: list[str] = Field(default_factory=list)
    flagged_phrases: list[str] = Field(default_factory=list)


class _CriticOutput(BaseModel):
    """LLM-facing schema: only the segments the critic changes, plus a summary."""

    adjustments: list[SegmentAdjustment]
    summary: str


class CritiqueResult(BaseModel):
    segments: list[SegmentAnalysis]
    summary: str


# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------

DETECTOR_SYSTEM = """You are a forensic linguist detecting LLM-generated text. Analyze each segment of the
input JSON array for markers of AI generation: formulaic openings/closings ("In today's fast-paced
world", "In conclusion", "It is important to note"), AI-typical vocabulary ("delve", "tapestry",
"landscape", "elevate", "showcase", "plethora"), uniform sentence rhythm, hedged neutral tone with
no personal voice, flawless grammar with no idiosyncrasies, formulaic 3/5-item lists, and vague
claims lacking concrete details, numbers, or named sources.

Bias guard: formal, technical, legal, and non-native-English writing often looks "polished and
neutral" -- treat those traits as evidence of nothing unless corroborated by several other markers.

For EACH segment return one entry: index (copied unchanged), ai_probability (0.0 = human,
1.0 = AI), label ("ai_likely" if p>=0.7, "uncertain" if 0.4<=p<0.7, else "human_likely"),
reasons (1-3 short explanations), flagged_phrases (exact phrases from the segment, quoted verbatim).
Respond with {"segments": [ ... one entry per input segment, same order ... ]}. Do not omit any segment.
"""

CRITIC_SYSTEM = """You are a skeptical reviewing editor. A detector agent produced per-segment findings (index,
ai_probability, reasons, flagged_phrases) for AI-generated text. Review them: lower ai_probability for
"ai_likely" segments that lack strong evidence, raise it for missed AI markers.
Return ONLY the segments you change as "adjustments" (index, ai_probability, reasons, flagged_phrases);
omit unchanged segments entirely. Do NOT include label (it is derived from the score in code).
Write a 2-4 sentence "summary" of the overall verdict.
Respond with {"adjustments": [...], "summary": "..."}.
"""


def _llm_request(system: str, user: str, max_tokens: int) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="mistral-medium-latest",
        messages=[SystemMessage(content=system), UserMessage(content=user)],
        temperature=0.0,
        max_tokens=max_tokens,
    )


def _normalize(seg: SegmentAnalysis) -> SegmentAnalysis:
    """Clamp the score and derive the label from it (single source of truth)."""
    seg.ai_probability = min(1.0, max(0.0, seg.ai_probability))
    seg.label = label_for(seg.ai_probability)
    return seg


# ---------------------------------------------------------------------------
# Activities (all I/O and LLM calls live here, never in the workflow body)
# ---------------------------------------------------------------------------

_BLOCKED_HOSTS = (
    "localhost",
    "127.",
    "0.0.0.0",
    "169.254.",
    "[::1]",
    "metadata.google",
)


@workflows.activity(
    start_to_close_timeout=timedelta(seconds=60), retry_policy_max_attempts=2
)
async def fetch_and_extract(url: str) -> str:
    """Download a page (http/https only, public hosts) and return clean text."""
    import httpx
    from html.parser import HTMLParser

    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Only http/https URLs are allowed: {url!r}")
    host = url.split("://", 1)[1].split("/", 1)[0].lower()
    if any(host.startswith(b) or b in host for b in _BLOCKED_HOSTS):
        raise ValueError(f"Refusing to fetch internal/loopback host: {host!r}")

    class _TextExtractor(HTMLParser):
        _SKIP = {"script", "style", "noscript", "head"}
        _BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "section", "article"}

        def __init__(self) -> None:
            super().__init__()
            self.parts: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag: str, attrs: list) -> None:
            if tag in self._SKIP:
                self._skip += 1
            elif tag in self._BLOCK:
                self.parts.append("\n")

        def handle_endtag(self, tag: str) -> None:
            if tag in self._SKIP and self._skip > 0:
                self._skip -= 1
            elif tag in self._BLOCK:
                self.parts.append("\n")

        def handle_data(self, data: str) -> None:
            if self._skip == 0:
                self.parts.append(data)

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; AIDetector/1.0)"},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        raw = resp.text[:MAX_HTML_BYTES]

    parser = _TextExtractor()
    parser.feed(raw)
    text = html.unescape("".join(parser.parts))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:MAX_TEXT_CHARS]


@workflows.activity()
async def split_segments(text: str, min_length: int = 80) -> list[Segment]:
    """Split text into paragraph segments; fall back to sentence grouping if too few."""
    text = text[:MAX_TEXT_CHARS]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    segments = [
        Segment(index=i, text=p)
        for i, p in enumerate(paragraphs)
        if len(p) >= min_length
    ]
    if len(segments) < 2:
        segments, buf = [], ""
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            buf = f"{buf} {sentence}".strip()
            if len(buf) >= min_length:
                segments.append(Segment(index=len(segments), text=buf))
                buf = ""
        if buf:
            segments.append(Segment(index=len(segments), text=buf))
    return segments[:MAX_SEGMENTS]


async def _detect_batch(batch: list[Segment]) -> list[SegmentAnalysis]:
    payload = json.dumps(
        [{"index": s.index, "text": s.text} for s in batch], ensure_ascii=False
    )
    result = await chat_parse_to_model(
        DetectionBatch,
        _llm_request(DETECTOR_SYSTEM, f"Segments to analyze:\n{payload}", 8192),
    )
    by_index = {s.index: _normalize(s) for s in result.segments}
    out = []
    for s in batch:  # backfill any segment the model omitted
        out.append(
            by_index.get(s.index)
            or SegmentAnalysis(
                index=s.index,
                ai_probability=0.5,
                label="uncertain",
                reasons=["analysis unavailable for this segment"],
            )
        )
    return out


@workflows.activity(
    start_to_close_timeout=timedelta(seconds=120), retry_policy_max_attempts=2
)
async def detect_segments(segments: list[Segment]) -> DetectionBatch:
    """Detector agent: per-segment AI analysis, batches in parallel."""
    import asyncio

    batches = [segments[i : i + 12] for i in range(0, len(segments), 12)]
    results = await asyncio.gather(
        *[_detect_batch(b) for b in batches], return_exceptions=True
    )
    analyses: list[SegmentAnalysis] = []
    for batch, res in zip(batches, results):
        if isinstance(res, Exception):
            analyses.extend(
                SegmentAnalysis(
                    index=s.index,
                    ai_probability=0.5,
                    label="uncertain",
                    reasons=[f"analysis failed: {type(res).__name__}"],
                )
                for s in batch
            )
        else:
            analyses.extend(res)
    return DetectionBatch(segments=analyses)


@workflows.activity(
    start_to_close_timeout=timedelta(seconds=120), retry_policy_max_attempts=2
)
async def critique_segments(
    segments: list[Segment], detections: DetectionBatch
) -> CritiqueResult:
    """Critic agent: review and adjust the detector's findings.

    The critic returns only the segments it changes; on any parse/API failure we
    keep the detector's findings unchanged instead of failing the run.
    """
    payload = json.dumps(
        {
            "segments": [{"index": s.index, "text": s.text} for s in segments],
            "detections": detections.model_dump()["segments"],
        },
        ensure_ascii=False,
    )
    try:
        parsed = await chat_parse_to_model(
            _CriticOutput, _llm_request(CRITIC_SYSTEM, payload, 8192)
        )
        adjustments = {a.index: a for a in parsed.adjustments}
        summary = parsed.summary
    except Exception:
        adjustments, summary = (
            {},
            "Critic review unavailable; showing detector findings unchanged.",
        )

    final: list[SegmentAnalysis] = []
    for d in detections.segments:
        adj = adjustments.get(d.index)
        if adj is not None:
            d.ai_probability = adj.ai_probability
            if adj.reasons:
                d.reasons = adj.reasons
            if adj.flagged_phrases:
                d.flagged_phrases = adj.flagged_phrases
        final.append(_normalize(d))
    return CritiqueResult(segments=final, summary=summary)


_TAG = {"ai_likely": "[AI {p:.0%}]", "uncertain": "[? {p:.0%}]"}
_LABEL_TEXT = {
    "ai_likely": "AI-likely",
    "uncertain": "Uncertain",
    "human_likely": "Likely human",
}


@workflows.activity()
async def build_report_markdown(
    text: str, segments: list[Segment], critique: CritiqueResult, overall: float
) -> str:
    """Inline markdown report: text with flagged passages tagged + a findings list."""
    analysis_by_index = {s.index: s for s in critique.segments}
    seg_text_by_index = {s.index: s.text for s in segments}

    parts: list[str] = []
    cursor = 0
    remaining = text
    unmatched = 0
    for idx in sorted(seg_text_by_index):
        seg_text = seg_text_by_index[idx]
        pat = r"\s+".join(
            map(re.escape, seg_text[:80].split())
        )  # whitespace-flexible match
        m = re.search(pat, remaining) if pat.strip() else None
        if m is None:
            unmatched += 1
            continue
        abs_start = cursor + m.start()
        parts.append(text[cursor:abs_start])
        analysis = analysis_by_index.get(idx)
        if analysis is not None and analysis.label in _TAG:
            parts.append(
                f"**{_TAG[analysis.label].format(p=analysis.ai_probability)}** {seg_text}"
            )
        else:
            parts.append(seg_text)  # human / unanalyzed: plain
        cursor = abs_start + len(seg_text)
        remaining = text[cursor:]
    parts.append(text[cursor:])
    annotated = "".join(parts)

    findings = []
    for seg in sorted(critique.segments, key=lambda s: s.index):
        if seg.label == "human_likely" and not seg.reasons:
            continue
        reasons = "; ".join(seg.reasons) or "none"
        phrases = (
            ", ".join(
                f"`{p}`"
                for p in seg.flagged_phrases
                if p and p in seg_text_by_index.get(seg.index, "")
            )
            or "none"
        )
        findings.append(
            f"- **Segment {seg.index}** — {_LABEL_TEXT[seg.label]} ({seg.ai_probability:.0%}): {reasons}. Flagged: {phrases}"
        )
    findings_md = "\n".join(findings) or "- No flagged segments."

    verdict_word = "High" if overall >= 0.7 else "Moderate" if overall >= 0.4 else "Low"
    n_ai = sum(1 for s in critique.segments if s.label == "ai_likely")
    n_total = len(critique.segments)
    unmatched_note = (
        f"\n_{unmatched} segment(s) could not be located in the source text._\n"
        if unmatched
        else ""
    )

    return f"""## AI Text Detection Report

**AI-likeness: {verdict_word} ({overall:.0%})** — {n_ai}/{n_total} segments AI-likely · heuristic estimate, not proof (false positives are common on short, formal, or non-native-English text).

{critique.summary}

**Legend:** `**[AI x%]**` = AI-likely · `**[? x%]**` = uncertain · untagged = assessed likely human

### Annotated text

{annotated}
{unmatched_note}
### Findings

{findings_md}
"""


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


@workflows.workflow.define(
    name="ai-text-detector",
    workflow_display_name="AI Text Detector",
    workflow_description=(
        "Analyzes a web page URL or pasted text for signs of AI-generated content and "
        "returns an inline annotated markdown report tagging AI-likely passages with the "
        "reasons why. Heuristic triage, not proof."
    ),
    execution_timeout=timedelta(minutes=30),
)
class AITextDetectorWorkflow(workflows.InteractiveWorkflow):
    @workflows.workflow.entrypoint
    async def run(self, inp: AIDetectionInput) -> wm.ChatAssistantWorkflowOutput:
        if bool(inp.url) == bool(inp.text):
            return wm.ChatAssistantWorkflowOutput(
                content=[
                    wm.TextOutput(text="Provide exactly one of: a URL or pasted text.")
                ],
                isError=True,
            )

        fetch_item = wm.TodoListItem(
            title="Fetch content", description="Download and extract text"
        )
        segment_item = wm.TodoListItem(
            title="Split into segments", description="Segmentation"
        )
        detect_item = wm.TodoListItem(
            title="Detector agent", description="Per-segment AI-marker analysis"
        )
        critique_item = wm.TodoListItem(
            title="Critic agent", description="Review and adjust findings"
        )
        viz_item = wm.TodoListItem(
            title="Build report", description="Inline markdown report"
        )

        async with wm.TodoList(
            items=[fetch_item, segment_item, detect_item, critique_item, viz_item]
        ):
            async with fetch_item:
                try:
                    text = (
                        await fetch_and_extract(inp.url)
                        if inp.url
                        else (inp.text or "")
                    )
                except Exception as exc:
                    return wm.ChatAssistantWorkflowOutput(
                        content=[
                            wm.TextOutput(
                                text=f"Could not fetch/analyze the input: {exc}"
                            )
                        ],
                        isError=True,
                    )

            async with segment_item:
                segments = await split_segments(text)
            if len(segments) < 2 or len(text) < MIN_TEXT_CHARS:
                return wm.ChatAssistantWorkflowOutput(
                    content=[
                        wm.TextOutput(
                            text="Not enough text to analyze reliably (need at least a few paragraphs). "
                            "AI detection on very short text is unreliable."
                        )
                    ],
                    isError=True,
                )

            async with detect_item:
                detections = await detect_segments(segments)
            async with critique_item:
                critique = await critique_segments(segments, detections)
            async with viz_item:
                # overall = length-weighted mean score, computed here (deterministic)
                seg_len = {s.index: len(s.text) for s in segments}
                total = sum(seg_len.values()) or 1
                overall = (
                    sum(
                        a.ai_probability * seg_len.get(a.index, 0)
                        for a in critique.segments
                    )
                    / total
                )
                report_md = await build_report_markdown(
                    text, segments, critique, overall
                )

        return wm.ChatAssistantWorkflowOutput(content=[wm.TextOutput(text=report_md)])
