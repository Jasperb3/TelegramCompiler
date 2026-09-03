from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import LengthFinishReasonError, OpenAI
from pydantic import BaseModel, Field, field_validator

from tg_compiler.config import AppConfig, ChannelConfig, LMStudioConfig
from tg_compiler.db import AnalysisRecord, Database, PostRecord
from tg_compiler.utils import _ENTITY_GARBAGE, clean_entities, escape_html

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an intelligence analyst processing raw Telegram posts for a daily geopolitical "
    "briefing. For each post:\n"
    "1. title: a concise factual headline (5-10 words, no punctuation at end, no quotes). "
    "State the event, not the post ('Iran closes Strait of Hormuz', never 'Post about Iran').\n"
    "2. summary: 1-2 sentences stating what happened — actor, action, location, and why it "
    "matters. Describe the event itself, never the post or the image ('The post shows…' and "
    "'The image features…' are wrong). Attribute unverified claims ('according to…', "
    "'reportedly'). No meta-commentary, no repeated words or phrases. Use people's names, "
    "titles, and offices exactly as the post gives them — never 'correct' them from your own "
    "knowledge, which may be outdated (if the post says 'President X', write 'President X', "
    "not 'former President X').\n"
    "3. Score each 1-5:\n"
    "   importance — how consequential the development is (5 = major geopolitical impact; "
    "1 = trivia, memes, channel promotion).\n"
    "   urgency — how time-critical (5 = unfolding right now; 1 = background or historical).\n"
    "   credibility — how reliable the claim appears (5 = official statement or multi-source "
    "confirmation; 1 = anonymous, sensational, or internally inconsistent).\n"
    "   relevance — pertinence to geopolitical/security monitoring (5 = conflict, diplomacy, "
    "strategic industry; 1 = sport, entertainment, advertising).\n"
    "4. Category, exactly one of: Breaking News (new event reported as happening) | Analysis "
    "(interpretation or commentary) | Official Statement (attributed government/military/agency "
    "communication) | Rumor (unverified or anonymous claim) | Media (post whose substance is a "
    "photo/video/meme) | Other.\n"
    "5. key_entities: up to 5 named actors, organisations, or places, each in its canonical "
    "form ('United States' not 'U.S.', 'Israel Defense Forces' not 'IDF'). Entities must be "
    "subjects of the event — never the news agency, photographer, or platform credited as the "
    "source (AFP, Reuters, Telegram, X), and never generic terms ('military', 'officials').\n"
    "6. Set image_substantive=true only if the image contains information absent from the "
    "text; if so, image_description must state that extra information in one sentence. If the "
    "image contains non-English text (signs, banners, documents, captions), include an English "
    "translation of it in image_description.\n"
    "7. threat_level, exactly one of: CRITICAL, HIGH, MODERATE, LOW.\n"
    "   CRITICAL — imminent risk of mass casualties, confirmed state-level military action underway, "
    "nuclear/chemical/biological threat, or active attack on critical infrastructure.\n"
    "   HIGH — confirmed armed conflict development, significant political crisis, major terror attack, "
    "or credible escalation warning from a named senior state official.\n"
    "   MODERATE — ongoing conflict updates, diplomatic developments, significant arrests or detentions, "
    "or unverified but plausible escalation claims.\n"
    "   LOW — background context, routine troop movement reports, unverified rumours, "
    "social media content, statistical or historical reports.\n"
    "Respond with valid JSON matching the PostAnalysis schema."
)


_VALID_THREAT_LEVELS = {"CRITICAL", "HIGH", "MODERATE", "LOW"}

# Posts with less text than this and no media are skipped before analysis (B1).
MIN_CONTENT_CHARS = 30
MAX_PROMPT_TEXT_CHARS = 3000  # post text is truncated to this many chars before being sent to the LLM
MAX_IMAGES_PER_POST = 3       # at most this many images are attached per post's analysis prompt
ANALYSIS_MAX_ATTEMPTS = 3     # analyze_post's structured-output retry ladder
RETRY_BACKOFF_BASE_SECS = 10  # backoff = RETRY_BACKOFF_BASE_SECS * (attempt + 1)
PREFLIGHT_TIMEOUT_SECS = 10   # LM Studio reachability probe timeout before process_unanalysed runs
BATCH_MAX_ATTEMPTS = 2        # a failed batch falls back to per-post calls, so don't retry it for long


class PostAnalysis(BaseModel):
    title: str = ""
    summary: str
    importance_score: int = Field(..., ge=1, le=5)
    urgency_score: int = Field(..., ge=1, le=5)
    credibility_score: int = Field(..., ge=1, le=5)
    relevance_score: int = Field(..., ge=1, le=5)
    category: str
    key_entities: list[str] = Field(default_factory=list)
    image_substantive: bool = False
    image_description: Optional[str] = None
    reasoning: str = ""
    threat_level: str = "MODERATE"

    @field_validator("threat_level")
    @classmethod
    def validate_threat_level(cls, v: str) -> str:
        normalised = v.upper().strip() if v else "MODERATE"
        return normalised if normalised in _VALID_THREAT_LEVELS else "MODERATE"



_TITLE_GARBAGE = re.compile(r'<\||\`\`\`|\{|\}|<image>|thought', re.IGNORECASE)

def _clean_title(title: str) -> str:
    if not title or not title.strip():
        return ""
    if len(title) > 120:
        return ""
    if _TITLE_GARBAGE.search(title):
        return ""
    return title


_NO_EXTRA_INFO_RE = re.compile(
    r"no (?:additional|new|further|extra) (?:substantive )?(?:information|details|detail|context)"
    r"|adds? nothing|nothing (?:new|substantive|further)"
    r"|no (?:substantive )?information beyond"
)


def _clean_image_insights(text: str | None) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    if stripped.lower() in ('n/a', 'none', 'no image provided', 'no image provided.',
                             'no image.', 'no image', 'na', '', 'none provided',
                             'none provided.', 'no video provided', 'no video provided.'):
        return None
    low = stripped.lower()
    if (
        low.startswith('a telegram post from')
        or low.startswith('a screenshot of a post')
        or 'text-only announcement' in low
        or 'text-based report' in low
        or 'featuring a text' in low
        # Descriptions whose content is "there is nothing extra here" — the model
        # answering the image_substantive question in prose instead of the flag.
        # They render an "Image" line in the briefing that tells the reader nothing.
        or _NO_EXTRA_INFO_RE.search(low)
    ):
        return None
    if len(stripped) < 10:
        return None
    if _ENTITY_GARBAGE.search(stripped) or '{' in stripped or '}' in stripped:
        return None
    return stripped


def parse_analysis_fallback(raw: str) -> PostAnalysis:
    score_match = re.search(r"importance[^\d]*(\d)", raw, re.IGNORECASE)
    importance = int(score_match.group(1)) if score_match else 3
    importance = max(1, min(5, importance))

    summary_match = re.search(r"[Ss]ummary[:\s]+([^.\n]+)", raw)
    summary = summary_match.group(1).strip() if summary_match else raw[:120].strip()

    cat_match = re.search(
        r"(Breaking News|Analysis|Official Statement|Rumor|Media|Other)", raw
    )
    category = cat_match.group(1) if cat_match else "Other"

    return PostAnalysis(
        summary=summary,
        importance_score=importance,
        urgency_score=3,
        credibility_score=3,
        relevance_score=3,
        category=category,
        threat_level="MODERATE",
        reasoning="Extracted via fallback parser",
    )


def _encode_image(path: str) -> str | None:
    try:
        data = Path(path).read_bytes()
        return base64.b64encode(data).decode()
    except Exception as e:
        log.warning("Could not read image %s: %s", path, e)
        return None


def build_messages(post: PostRecord, system_prompt: str) -> list[dict]:
    text = post.text[:MAX_PROMPT_TEXT_CHARS] if len(post.text) > MAX_PROMPT_TEXT_CHARS else post.text
    header = f"Post from {post.channel_name} at {post.timestamp.isoformat()}:\n\n{text}"

    content: list[dict] = [{"type": "text", "text": header}]
    for path in post.media_paths[:MAX_IMAGES_PER_POST]:
        b64 = _encode_image(path)
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def compute_token_budget(post: PostRecord, cfg: LMStudioConfig) -> int:
    """Per-post max_tokens for an analysis call.

    LM Studio's OpenAI-compatible endpoint exposes no reasoning-token controls
    (only max_tokens/temperature/sampling params), and reasoning models spend a
    large, mostly length-independent chunk of the completion budget deliberating
    before emitting JSON (~800 reasoning tokens observed even on a short post),
    so the budget is a flat base — reasoning allowance plus the structured JSON
    output — plus increments for the text and images actually sent in the prompt
    (measured after build_messages' truncation/cap, so unsent content never
    inflates the budget), clamped at analysis_max_tokens:

        min(base + text_chars * per_char + images * per_image, max)
    """
    text_chars = min(len(post.text), MAX_PROMPT_TEXT_CHARS)
    n_images = min(len(post.media_paths), MAX_IMAGES_PER_POST)
    budget = (
        cfg.analysis_base_tokens
        + math.ceil(text_chars * cfg.analysis_tokens_per_char)
        + n_images * cfg.analysis_tokens_per_image
    )
    return min(budget, cfg.analysis_max_tokens)


BATCH_INSTRUCTIONS = (
    "\n\nYou will be given several numbered posts in a single message. The posts are unrelated "
    "to each other: analyse each one strictly on its own content, and never carry facts, "
    "entities, figures, or wording from one post into another post's analysis. Return one "
    "object per post in \"analyses\", each carrying that post's 1-based \"index\" exactly as "
    "given, and \"opening\": the first six words of that post's Text line copied verbatim "
    "(the words after 'Text:', not the Channel or Time lines). Return an object for every "
    "post and nothing else."
)


class BatchPostAnalysis(PostAnalysis):
    """A PostAnalysis carrying the 1-based position of the post it describes.

    `opening` is an integrity anchor, not content: the model copies the first few
    words of the post it actually analysed, which is the only way to tell that an
    item is attached to the right post. See map_batch_results.
    """

    index: int
    opening: str = ""


class BatchAnalysis(BaseModel):
    analyses: list[BatchPostAnalysis] = Field(default_factory=list)


def build_batch_messages(
    posts: list[PostRecord], system_prompt: str, cfg: LMStudioConfig
) -> list[dict]:
    """One multimodal user message holding every post in the batch.

    Each post is a "### POST n" text part followed immediately by its images, so
    the model can tell which image belongs to which post. Text is truncated with
    the same MAX_PROMPT_TEXT_CHARS cap build_messages uses.
    """
    content: list[dict] = [
        {"type": "text", "text": f"Analyse each of the following {len(posts)} posts."}
    ]
    for i, post in enumerate(posts, 1):
        text = post.text[:MAX_PROMPT_TEXT_CHARS]
        content.append({
            "type": "text",
            "text": (
                f"### POST {i}\nChannel: {post.channel_name}\n"
                f"Time: {post.timestamp.isoformat()}\nText: {text}"
            ),
        })
        for path in post.media_paths[:cfg.batch_images_per_post]:
            b64 = _encode_image(path)
            if b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

    return [
        {"role": "system", "content": system_prompt + BATCH_INSTRUCTIONS},
        {"role": "user", "content": content},
    ]


def _chunk(posts: list[PostRecord], size: int, max_chars: int) -> list[list[PostRecord]]:
    """Split posts into runs of at most `size`, closing a run early if its prompt
    text would exceed `max_chars` — a run of long posts must not overflow context."""
    batches: list[list[PostRecord]] = []
    current: list[PostRecord] = []
    chars = 0
    for post in posts:
        n = min(len(post.text), MAX_PROMPT_TEXT_CHARS)
        if current and (len(current) >= size or chars + n > max_chars):
            batches.append(current)
            current, chars = [], 0
        current.append(post)
        chars += n
    if current:
        batches.append(current)
    return batches


def plan_batches(posts: list[PostRecord], cfg: LMStudioConfig) -> list[list[PostRecord]]:
    """Group posts into LLM calls.

    Media-bearing posts are grouped separately and at their own (smaller) size:
    images dominate the prompt, so they amortise far less per extra post than
    text-only ones do. With both sizes at their default of 1 every batch is a
    single post, which process_unanalysed routes down the per-post path.
    """
    text_posts = [p for p in posts if not p.media_paths]
    media_posts = [p for p in posts if p.media_paths]
    return (
        _chunk(text_posts, cfg.batch_size, cfg.batch_max_prompt_chars)
        + _chunk(media_posts, cfg.batch_size_with_images, cfg.batch_max_prompt_chars)
    )


def compute_batch_token_budget(posts: list[PostRecord], cfg: LMStudioConfig) -> int:
    """max_tokens for one batch call.

    Same shape as compute_token_budget, but the reasoning allowance is a single
    flat base for the whole call rather than per post — that shared deliberation
    is exactly what batching amortises:

        min(base + posts * per_post + text_chars * per_char + images * per_image, max)
    """
    text_chars = sum(min(len(p.text), MAX_PROMPT_TEXT_CHARS) for p in posts)
    n_images = sum(min(len(p.media_paths), cfg.batch_images_per_post) for p in posts)
    budget = (
        cfg.batch_base_tokens
        + len(posts) * cfg.batch_tokens_per_post
        + math.ceil(text_chars * cfg.batch_tokens_per_char)
        + n_images * cfg.batch_tokens_per_image
    )
    return min(budget, cfg.batch_max_tokens)


BATCH_OPENING_WORDS = 6           # words of each post the model echoes back as an integrity anchor
BATCH_OPENING_MATCH_RATIO = 0.5   # fraction of those words that must appear in the post's own opening
_WORD_RE = re.compile(r"[a-z0-9]+")
# Telegram posts routinely carry a promo footer — @handle plus Socials/Donate/
# Advertising links — which is pure noise for anchoring and can be most of a
# short caption's tokens.
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+")
_HANDLE_RE = re.compile(r"@\w+")


def _opening_words(text: str, limit: int) -> list[str]:
    text = _MD_LINK_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _HANDLE_RE.sub(" ", text)
    return _WORD_RE.findall(text.lower())[:limit]


def check_opening(opening: str, post: PostRecord) -> str:
    """Classify a batch item's echoed opening against the post it claims.

    Returns "match", "mismatch", or "absent". The three are genuinely different:
    a mismatch means the model attributed this analysis to the wrong post, while
    an absent opening only means the claim could not be checked — conflating them
    would report a model that ignores the field as one that corrupts data.

    Observed on google/gemma-3-4b: a model can drop one post from a batch and
    renumber the rest densely, so every reported index is unique and in range
    while the analyses all describe different posts. Indices alone cannot detect
    that; only something derived from the post's own text can.
    """
    claimed = _opening_words(opening, BATCH_OPENING_WORDS)
    if not claimed:
        return "absent"
    actual_words = _opening_words(post.text, BATCH_OPENING_WORDS * 3)
    if len(actual_words) < BATCH_OPENING_WORDS:
        # Nothing to match against. Image posts often carry a two-word caption with
        # the substance inside the picture, and the model then echoes what it read
        # from the image — a correct analysis the anchor cannot confirm. Claiming
        # misattribution here would assert corruption we have no evidence for.
        return "absent"
    actual = set(actual_words)
    hits = sum(1 for w in claimed if w in actual)
    if hits >= max(1, round(len(claimed) * BATCH_OPENING_MATCH_RATIO)):
        return "match"
    return "mismatch"


def _opening_matches(opening: str, post: PostRecord) -> bool:
    return check_opening(opening, post) == "match"


def map_batch_results(batch: BatchAnalysis, posts: list[PostRecord]) -> dict[int, PostAnalysis]:
    """Map returned analyses onto batch positions by their reported `index`.

    Never by list position — a model that drops or reorders an item would
    otherwise silently attach an analysis to the wrong post. Out-of-range and
    duplicate indices are dropped and logged, and each item must echo an opening
    that actually belongs to the post it claims (see _opening_matches), which is
    what catches a model that renumbers. A position with no surviving result is
    simply absent from the returned dict, leaving that post unanalysed for the
    caller to retry singly or requeue.
    """
    mapped: dict[int, PostAnalysis] = {}
    for item in batch.analyses:
        pos = item.index - 1
        if not 0 <= pos < len(posts):
            log.warning(
                "Batch result index %s outside 1..%d — dropped", item.index, len(posts)
            )
            continue
        if pos in mapped:
            log.warning("Duplicate batch result index %s — keeping the first", item.index)
            continue
        verdict = check_opening(item.opening, posts[pos])
        if verdict == "mismatch":
            log.warning(
                "Batch result %d echoed opening %r, which belongs to a different post than "
                "%s — dropped; the model renumbered its results and cannot be batched safely",
                item.index, item.opening[:60], posts[pos].message_id,
            )
            continue
        if verdict == "absent":
            log.warning(
                "Batch result %d for post %s carried no opening, so it could not be verified "
                "— dropped and requeued; this model may be ignoring the field",
                item.index, posts[pos].message_id,
            )
            continue
        mapped[pos] = _sanitize(item)
    return mapped


def salvage_batch_items(raw: str) -> BatchAnalysis:
    """Recover the complete analysis objects from a truncated batch response.

    A response cut off at max_tokens is invalid JSON as a whole, but the objects
    emitted before the cut are intact and worth keeping — the posts they cover
    don't need re-analysing.
    """
    start = raw.find('"analyses"')
    if start == -1:
        return BatchAnalysis()
    start = raw.find("[", start)
    if start == -1:
        return BatchAnalysis()

    decoder = json.JSONDecoder()
    items: list[BatchPostAnalysis] = []
    pos = start + 1
    while pos < len(raw):
        while pos < len(raw) and raw[pos] in ", \n\r\t":
            pos += 1
        if pos >= len(raw) or raw[pos] != "{":
            break
        try:
            obj, pos = decoder.raw_decode(raw, pos)
        except ValueError:
            break
        try:
            items.append(BatchPostAnalysis.model_validate(obj))
        except Exception:
            continue
    return BatchAnalysis(analyses=items)


_NUM_RE = re.compile(r'\b(\d+(?:\.\d+)?)\b')
_TIME_TOKEN_RE = re.compile(r'\b\d{1,2}:\d{2}\b')  # e.g. "14:30"
_YEAR_TOKEN_RE = re.compile(r'\b(?:19|20)\d{2}\b')  # e.g. "2026"


def _strip_non_quantity_numbers(text: str) -> str:
    """Remove time-of-day and year tokens before numeric comparison — they
    collide with casualty/quantity figures of similar magnitude (e.g. "14:30"
    vs "30 dead") and are not the kind of number the consistency check is for."""
    text = _TIME_TOKEN_RE.sub(" ", text)
    text = _YEAR_TOKEN_RE.sub(" ", text)
    return text


def _check_numeric_consistency(summary: str, image_desc: str) -> bool:
    """Return False if a number in image_desc contradicts a comparable number in summary.

    Two numbers are 'comparable' if they are within the same order of magnitude
    (ratio ≤ 10x).  They 'contradict' if they differ by more than 5% relative to
    the smaller value.  When either text has no numbers, we assume consistent.
    """
    if not summary or not image_desc:
        return True
    summary = _strip_non_quantity_numbers(summary)
    image_desc = _strip_non_quantity_numbers(image_desc)
    s_nums = [float(x) for x in _NUM_RE.findall(summary) if float(x) > 0]
    i_nums = [float(x) for x in _NUM_RE.findall(image_desc) if float(x) > 0]
    if not s_nums or not i_nums:
        return True
    for img_n in i_nums:
        for sum_n in s_nums:
            if max(img_n, sum_n) / min(img_n, sum_n) > 10.0:
                continue  # different orders of magnitude — unrelated quantities
            if abs(img_n - sum_n) / min(img_n, sum_n) > 0.05:
                return False
    return True


_REFUSAL_RE = re.compile(
    r"(?i)\b(the user provided|no content (?:was )?provided|cannot analy[sz]e"
    r"|unable to analy[sz]e|the user|i cannot|as an ai)\b"
)


def _sanitize(analysis: PostAnalysis) -> PostAnalysis:
    analysis.title = _clean_title(analysis.title)
    if _REFUSAL_RE.search(analysis.title):
        analysis.title = ""
    if _REFUSAL_RE.search(analysis.summary):
        analysis.summary = ""
    analysis.key_entities = clean_entities(analysis.key_entities)
    analysis.image_description = _clean_image_insights(analysis.image_description)
    if analysis.image_description and not _check_numeric_consistency(
        analysis.summary, analysis.image_description
    ):
        log.info(
            "Image description dropped for post %r: numeric mismatch with summary",
            analysis.title or analysis.summary[:60],
        )
        analysis.image_description = None

    analysis.title = escape_html(analysis.title)
    analysis.summary = escape_html(analysis.summary)
    analysis.key_entities = [escape_html(e) for e in analysis.key_entities]
    if analysis.image_description:
        analysis.image_description = escape_html(analysis.image_description)
    return analysis


def analysis_to_record(post_id: int, analysis: PostAnalysis, model_used: str) -> AnalysisRecord:
    return AnalysisRecord(
        post_id=post_id,
        title=analysis.title,
        summary=analysis.summary,
        importance_score=analysis.importance_score,
        urgency_score=analysis.urgency_score,
        credibility_score=analysis.credibility_score,
        relevance_score=analysis.relevance_score,
        category=analysis.category,
        key_entities=analysis.key_entities,
        image_insights=analysis.image_description,
        model_used=model_used,
        threat_level=analysis.threat_level,
    )


class Analyzer:
    def __init__(self, config: AppConfig, db: Database):
        self._cfg = config
        self._db = db
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            cfg = self._cfg.lmstudio
            api_key = cfg.api_token or "lm-studio"
            self._client = OpenAI(
                base_url=f"http://{cfg.server_host}:{cfg.server_port}/v1",
                api_key=api_key,
                timeout=120,
                max_retries=0,  # analyze_post has its own retry loop; SDK retries would multiply it
            )
        return self._client

    def _call_llm(
        self, messages: list[dict], structured: bool, max_tokens: int
    ) -> PostAnalysis | str:
        cfg = self._cfg.lmstudio
        client = self._get_client()
        if structured:
            completion = client.beta.chat.completions.parse(
                model=cfg.model,
                messages=messages,
                response_format=PostAnalysis,
                temperature=cfg.temperature,
                max_tokens=max_tokens,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is not None:
                return parsed
            # structured output returned None — fall through to text path
            raw = completion.choices[0].message.content or ""
        else:
            completion = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=max_tokens,
            )
            raw = completion.choices[0].message.content or ""

        # Strip markdown fences then try to parse JSON
        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return PostAnalysis.model_validate_json(stripped)
        except Exception:
            return raw

    async def analyze_post(
        self, post: PostRecord, channel_cfg: ChannelConfig | None = None
    ) -> PostAnalysis:
        system = (
            channel_cfg.custom_prompt
            if (channel_cfg and channel_cfg.custom_prompt)
            else SYSTEM_PROMPT
        )
        messages = build_messages(post, system)
        budget = compute_token_budget(post, self._cfg.lmstudio)

        for attempt in range(ANALYSIS_MAX_ATTEMPTS):
            try:
                result = await asyncio.to_thread(self._call_llm, messages, True, budget)
                if isinstance(result, PostAnalysis):
                    return _sanitize(result)
                return _sanitize(parse_analysis_fallback(result if isinstance(result, str) else ""))
            except Exception as e:
                if attempt == ANALYSIS_MAX_ATTEMPTS - 1:
                    log.warning(
                        "Analysis failed for post %s after %d attempts, trying plain-text fallback: %s",
                        post.message_id, ANALYSIS_MAX_ATTEMPTS, e,
                    )
                    # Last resort: plain-text call. If this also fails, propagate —
                    # writing a fabricated empty analysis would permanently mark the
                    # post as analysed and it would never be retried.
                    result = await asyncio.to_thread(self._call_llm, messages, False, budget)
                    if isinstance(result, PostAnalysis):
                        return _sanitize(result)
                    return _sanitize(parse_analysis_fallback(result if isinstance(result, str) else ""))
                await asyncio.sleep(RETRY_BACKOFF_BASE_SECS * (attempt + 1))

    def _call_batch_llm(self, messages: list[dict], max_tokens: int, expected: int) -> BatchAnalysis:
        cfg = self._cfg.lmstudio
        try:
            completion = self._get_client().beta.chat.completions.parse(
                model=cfg.model,
                messages=messages,
                response_format=BatchAnalysis,
                temperature=cfg.temperature,
                max_tokens=max_tokens,
            )
        except LengthFinishReasonError as e:
            # .parse() refuses a response cut off at max_tokens instead of
            # returning it, but the objects emitted before the cut are intact and
            # the batch's remaining posts simply get requeued. Without this the
            # whole call's work — thousands of tokens — would be thrown away.
            completion = e.completion
        choice = completion.choices[0]
        # LengthFinishReasonError carries a plain ChatCompletion, which has no
        # `parsed` attribute at all — not merely a None one.
        parsed = getattr(choice.message, "parsed", None)
        if parsed is not None:
            return parsed

        raw = (choice.message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return BatchAnalysis.model_validate_json(raw)
        except Exception:
            # A response cut off at max_tokens is unparseable as a whole; keep the
            # objects that made it out and let the caller requeue the rest.
            salvaged = salvage_batch_items(raw)
            log.warning(
                "Batch response was not valid JSON (finish_reason=%s) — salvaged %d of %d items."
                " If finish_reason is 'length', raise lmstudio.batch_base_tokens or lower"
                " batch_size; the unreturned posts are requeued for the next run.",
                choice.finish_reason, len(salvaged.analyses), expected,
            )
            return salvaged

    async def analyze_batch(self, posts: list[PostRecord]) -> dict[int, PostAnalysis]:
        """Analyse several posts in one LLM call.

        Returns {position in `posts`: PostAnalysis} for the posts the model
        actually returned. Positions absent from the result were not analysed —
        the caller decides whether to retry them individually or requeue them.
        Returns {} if the call fails outright.
        """
        messages = build_batch_messages(posts, SYSTEM_PROMPT, self._cfg.lmstudio)
        budget = compute_batch_token_budget(posts, self._cfg.lmstudio)

        last_error: Exception | None = None
        for attempt in range(BATCH_MAX_ATTEMPTS):
            try:
                result = await asyncio.to_thread(
                    self._call_batch_llm, messages, budget, len(posts)
                )
                return map_batch_results(result, posts)
            except Exception as e:
                last_error = e
                if attempt < BATCH_MAX_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_BACKOFF_BASE_SECS * (attempt + 1))
        log.warning(
            "Batch of %d posts failed after %d attempts: %s",
            len(posts), BATCH_MAX_ATTEMPTS, last_error,
        )
        return {}

    def _server_reachable(self) -> bool:
        """Quick preflight probe so a dead LM Studio aborts in seconds, not hours."""
        try:
            self._get_client().with_options(timeout=PREFLIGHT_TIMEOUT_SECS).models.list()
            return True
        except Exception as e:
            cfg = self._cfg.lmstudio
            log.error(
                "LM Studio unreachable at %s:%s — %s",
                cfg.server_host, cfg.server_port, e,
            )
            return False

    async def process_unanalysed(
        self,
        channel_map: dict[int, ChannelConfig] | None = None,
        since: datetime | None = None,
    ) -> tuple[int, int]:
        """Analyse every unanalysed post up to lmstudio.max_concurrent_analyses in
        parallel, after a preflight reachability probe (aborts with everything left
        queued if LM Studio is down). Posts under MIN_CONTENT_CHARS with no media are
        skipped (recorded with category="Skipped") rather than sent to the LLM.
        If `since` is given, only posts with timestamp >= since are analysed — older
        stuck-unanalysed posts are left queued for a future unscoped run.
        When lmstudio.batch_size / batch_size_with_images exceed 1 the surviving
        posts are grouped by plan_batches() and analysed several per LLM call;
        at their default of 1 every post takes the original one-call-per-post path.
        Returns (analysed_count, skipped_count)."""
        posts = self._db.get_unanalysed_posts(since=since)
        if since is not None:
            excluded = len(self._db.get_unanalysed_posts()) - len(posts)
            if excluded:
                log.info(
                    "--since filter: %d older unanalysed posts excluded from this run "
                    "(predate %s), still queued for a future unscoped run",
                    excluded,
                    since.isoformat(),
                )
        if not posts:
            return 0, 0

        if not await asyncio.to_thread(self._server_reachable):
            log.error("Aborting analysis — %d posts remain queued for the next run", len(posts))
            return 0, 0

        cfg = self._cfg.lmstudio
        sem = asyncio.Semaphore(cfg.max_concurrent_analyses)
        skipped = 0
        analysed = 0

        from tqdm import tqdm
        from tqdm.contrib.logging import logging_redirect_tqdm

        bar = tqdm(total=len(posts), desc="Analysing posts", unit="post")

        def _channel_cfg(post: PostRecord) -> ChannelConfig | None:
            return channel_map.get(post.channel_id) if channel_map else None

        def _save(post: PostRecord, analysis: PostAnalysis) -> None:
            nonlocal analysed
            self._db.insert_analysis(analysis_to_record(post.id, analysis, cfg.model))
            analysed += 1

        # Content gate first, so short no-media posts never reach the LLM — batched
        # or not — and are recorded as "Skipped" exactly as before.
        to_analyse: list[PostRecord] = []
        for post in posts:
            if (
                len(post.text.strip()) < MIN_CONTENT_CHARS
                and not post.media_paths
                and not post.has_video
                and not post.has_images
            ):
                self._db.insert_analysis(AnalysisRecord(
                    post_id=post.id,
                    summary="",
                    importance_score=None,
                    urgency_score=None,
                    credibility_score=None,
                    relevance_score=None,
                    category="Skipped",
                    key_entities=[],
                    model_used=cfg.model,
                ))
                skipped += 1
                bar.update(1)
            else:
                to_analyse.append(post)

        async def _analyse_one(post: PostRecord) -> None:
            try:
                async with sem:
                    analysis = await self.analyze_post(post, _channel_cfg(post))
            except Exception as e:
                log.error(
                    "Analysis failed for post %s — left unanalysed for the next run: %s",
                    post.message_id, e,
                )
                bar.update(1)
                return
            _save(post, analysis)
            bar.update(1)

        async def _analyse_batch(batch: list[PostRecord]) -> None:
            if len(batch) == 1:
                await _analyse_one(batch[0])
                return
            async with sem:
                results = await self.analyze_batch(batch)
            for pos in sorted(results):
                _save(batch[pos], results[pos])
            bar.update(len(results))

            missing = [p for i, p in enumerate(batch) if i not in results]
            if not missing:
                return
            if len(results) < cfg.batch_min_yield_ratio * len(batch):
                # A mostly-failed batch signals something systematic (context
                # overflow, a malformed response) — fall back to the per-post path.
                log.warning(
                    "Batch of %d returned only %d analyses — retrying the other %d individually",
                    len(batch), len(results), len(missing),
                )
                await asyncio.gather(*(_analyse_one(p) for p in missing))
            else:
                log.info(
                    "%d of %d batched posts were not returned — left queued for the next run",
                    len(missing), len(batch),
                )
                bar.update(len(missing))

        # A channel with a custom_prompt needs its own system prompt, which a shared
        # batch call cannot provide — those posts stay on the per-post path.
        custom_ids = {
            p.id for p in to_analyse
            if (_channel_cfg(p) is not None and _channel_cfg(p).custom_prompt)
        }
        custom_posts = [p for p in to_analyse if p.id in custom_ids]
        batchable = [p for p in to_analyse if p.id not in custom_ids]

        with logging_redirect_tqdm():
            try:
                await asyncio.gather(
                    *(_analyse_batch(b) for b in plan_batches(batchable, cfg)),
                    *(_analyse_one(p) for p in custom_posts),
                )
            finally:
                bar.close()

        failed = len(to_analyse) - analysed
        if failed:
            log.warning("%d posts failed analysis and remain queued for the next run", failed)
        return analysed, skipped
