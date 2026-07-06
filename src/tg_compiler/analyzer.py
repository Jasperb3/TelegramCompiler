from __future__ import annotations
import asyncio
import base64
import logging
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from tg_compiler.config import AppConfig, ChannelConfig
from tg_compiler.db import Database, PostRecord, AnalysisRecord
from tg_compiler.utils import clean_entities, escape_html, _ENTITY_GARBAGE

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
    text = post.text[:3000] if len(post.text) > 3000 else post.text
    header = f"Post from {post.channel_name} at {post.timestamp.isoformat()}:\n\n{text}"

    content: list[dict] = [{"type": "text", "text": header}]
    for path in post.media_paths[:3]:
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

    def _call_llm(self, messages: list[dict], structured: bool) -> PostAnalysis | str:
        cfg = self._cfg.lmstudio
        client = self._get_client()
        if structured:
            completion = client.beta.chat.completions.parse(
                model=cfg.model,
                messages=messages,
                response_format=PostAnalysis,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
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
                max_tokens=cfg.max_tokens,
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

        for attempt in range(3):
            try:
                result = await asyncio.to_thread(self._call_llm, messages, True)
                if isinstance(result, PostAnalysis):
                    return _sanitize(result)
                return _sanitize(parse_analysis_fallback(result if isinstance(result, str) else ""))
            except Exception as e:
                if attempt == 2:
                    log.warning(
                        "Analysis failed for post %s after 3 attempts, trying plain-text fallback: %s",
                        post.message_id, e,
                    )
                    # Last resort: plain-text call. If this also fails, propagate —
                    # writing a fabricated empty analysis would permanently mark the
                    # post as analysed and it would never be retried.
                    result = await asyncio.to_thread(self._call_llm, messages, False)
                    if isinstance(result, PostAnalysis):
                        return _sanitize(result)
                    return _sanitize(parse_analysis_fallback(result if isinstance(result, str) else ""))
                await asyncio.sleep(10 * (attempt + 1))

    def _server_reachable(self) -> bool:
        """Quick preflight probe so a dead LM Studio aborts in seconds, not hours."""
        try:
            self._get_client().with_options(timeout=10).models.list()
            return True
        except Exception as e:
            cfg = self._cfg.lmstudio
            log.error(
                "LM Studio unreachable at %s:%s — %s",
                cfg.server_host, cfg.server_port, e,
            )
            return False

    async def process_unanalysed(
        self, channel_map: dict[int, ChannelConfig] | None = None
    ) -> tuple[int, int]:
        posts = self._db.get_unanalysed_posts()
        if not posts:
            return 0, 0

        if not await asyncio.to_thread(self._server_reachable):
            log.error("Aborting analysis — %d posts remain queued for the next run", len(posts))
            return 0, 0

        sem = asyncio.Semaphore(self._cfg.lmstudio.max_concurrent_analyses)
        skipped = 0
        failed = 0

        from tqdm import tqdm
        from tqdm.contrib.logging import logging_redirect_tqdm

        bar = tqdm(total=len(posts), desc="Analysing posts", unit="post")

        async def _analyse_and_save(post: PostRecord) -> None:
            nonlocal skipped, failed
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
                    model_used=self._cfg.lmstudio.model,
                ))
                skipped += 1
                bar.update(1)
                return
            channel_cfg = channel_map.get(post.channel_id) if channel_map else None
            try:
                async with sem:
                    analysis = await self.analyze_post(post, channel_cfg)
            except Exception as e:
                log.error(
                    "Analysis failed for post %s — left unanalysed for the next run: %s",
                    post.message_id, e,
                )
                failed += 1
                bar.update(1)
                return
            self._db.insert_analysis(analysis_to_record(post.id, analysis, self._cfg.lmstudio.model))
            bar.update(1)

        with logging_redirect_tqdm():
            try:
                await asyncio.gather(*(_analyse_and_save(p) for p in posts))
            finally:
                bar.close()
        if failed:
            log.warning("%d posts failed analysis and remain queued for the next run", failed)
        return len(posts) - skipped - failed, skipped
