from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI

from tg_compiler.config import AppConfig
from tg_compiler.db import Database
from tg_compiler.trends import TREND_WINDOW_DAYS, compute_trends
from tg_compiler.utils import escape_html

log = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"

_SYNTHESIS_SYSTEM = (
    "You are a senior intelligence analyst producing a daily assessment from the "
    "highest-priority intelligence reports of a 24-hour monitoring period. "
    "Ground every statement strictly in the supplied reports — never invent events, numbers, "
    "quotes, or actors — and mark single-source or Rumor-category claims as unconfirmed "
    "('reportedly'). Use names, titles, and offices exactly as the reports give them; do not "
    "'correct' them from your own knowledge, which may be outdated. Respond immediately with "
    "ONLY a valid JSON object: no reasoning preamble, no markdown fences, no commentary "
    "outside the JSON."
)

_SYNTHESIS_INSTRUCTIONS = """
Each report above has an "index" field. When you cite supporting evidence below, reference these indices; cite only indices that genuinely support the point.

Based on these intelligence reports, produce a JSON object with exactly these four keys:

"situation_summary": A 3-5 sentence executive overview of the overall geopolitical situation, opening with the single most consequential development of the day. Direct and specific — name actors, locations, and developments. If a 7-DAY MENTION TRENDS table is provided, incorporate notable trend shifts.

"key_themes": An array of 3-5 objects. A theme is a pattern connecting two or more reports, not a restatement of one headline. Each object: {"theme": "<short title>", "detail": "<2-3 sentences on what events are connected and why the pattern matters>", "sources": [<indices of ALL reports supporting this theme>], "continuity": "new"|"confirmed"|"escalating"|"retired" — if PREVIOUS ASSESSMENT THEMES are provided, set this relative to whether this theme matches one of them (confirmed if it continues, escalating if it intensified, retired if a previous theme no longer applies); otherwise use "new"}

"signals_and_warnings": An array of 3-5 objects. Each object: {"signal": "<short title>", "assessment": "<what could plausibly develop next, with concrete observable indicators that would confirm or refute it>", "sources": [<indices of reports supporting this signal>]}

"named_actors": An array of the 4-6 most significant actors (people, states, organisations) from today's reporting, in canonical form — never news agencies or platforms credited as sources. Each object: {"actor": "<name>", "role": "<one short phrase>", "activity": "<1-2 sentences on what they did today and its significance>"}
"""


def _format_trends(trends: dict | None) -> str:
    if not trends or not (trends.get("entity_deltas") or trends.get("category_deltas")):
        return ""
    lines = [f"\n{TREND_WINDOW_DAYS}-DAY MENTION TRENDS (prior-days total -> today's count):"]
    if trends.get("entity_deltas"):
        lines.append("Entities:")
        for d in trends["entity_deltas"]:
            lines.append(f'- {d["entity"]}: {d["prior_count"]} -> {d["today_count"]}')
    if trends.get("category_deltas"):
        lines.append("Categories:")
        for d in trends["category_deltas"]:
            lines.append(f'- {d["category"]}: {d["prior_count"]} -> {d["today_count"]}')
    return "\n".join(lines)


def _format_previous_themes(previous_intel: dict | None) -> str:
    if not previous_intel or not previous_intel.get("key_themes"):
        return ""
    lines = ["\nPREVIOUS ASSESSMENT THEMES (assess whether each is new, confirmed, escalating, or retired today):"]
    for theme in previous_intel["key_themes"]:
        lines.append(f'- "{theme.get("theme", "")}": {theme.get("detail", "")}')
    return "\n".join(lines)


def _validate_intel(data: dict) -> str | None:
    """Return an error string if validation fails, or None if valid."""
    required_keys = {"situation_summary", "key_themes", "signals_and_warnings", "named_actors"}
    missing = required_keys - data.keys()
    if missing:
        return f"missing keys: {missing}"
    if not isinstance(data["situation_summary"], str) or not data["situation_summary"].strip():
        return "situation_summary is empty"
    for key in ("key_themes", "signals_and_warnings", "named_actors"):
        val = data[key]
        if not isinstance(val, list) or not val:
            return f"{key} is empty or not a list"
    for item in data["key_themes"]:
        if "theme" not in item or "detail" not in item:
            return "key_themes item missing 'theme' or 'detail'"
        if "sources" not in item or not isinstance(item["sources"], list):
            return "key_themes item missing 'sources' list"
    for item in data["signals_and_warnings"]:
        if "signal" not in item or "assessment" not in item:
            return "signals_and_warnings item missing 'signal' or 'assessment'"
        if "sources" not in item or not isinstance(item["sources"], list):
            return "signals_and_warnings item missing 'sources' list"
    for item in data["named_actors"]:
        if "actor" not in item or "role" not in item or "activity" not in item:
            return "named_actors item missing required sub-key"
    return None


async def synthesise(config: AppConfig, posts: list[dict], trends: dict | None = None, previous_intel: dict | None = None) -> dict | None:
    cfg = config.lmstudio
    try:
        client = OpenAI(
            base_url=f"http://{cfg.server_host}:{cfg.server_port}/v1",
            api_key=cfg.api_token or "lm-studio",
            timeout=300,
        )
    except Exception as e:
        log.error("LM Studio not reachable — cannot generate intelligence front page: %s", e)
        return None

    # Keep only the fields the synthesis LLM needs; drop scoring/routing metadata.
    synthesis_posts = [
        {
            "index": i + 1,
            "title": p.get("title", ""),
            "summary": p.get("summary", ""),
            "category": p.get("category", ""),
            "threat_level": p.get("threat_level", ""),
            "entities": p.get("entities", []),
        }
        for i, p in enumerate(posts)
    ]
    posts_json = json.dumps(synthesis_posts, ensure_ascii=False, default=str)
    trend_block = _format_trends(trends)
    continuity_block = _format_previous_themes(previous_intel)
    user_message = f"{posts_json}\n{trend_block}\n{continuity_block}\n{_SYNTHESIS_INSTRUCTIONS}"

    try:
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": _SYNTHESIS_SYSTEM},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                max_tokens=cfg.synthesis_max_tokens,
            )
        )
    except Exception as e:
        log.error("LM Studio not reachable — cannot generate intelligence front page: %s", e)
        return None

    choice = response.choices[0]
    raw = choice.message.content or ""
    log.debug("Synthesis raw response (finish_reason=%s): %s", choice.finish_reason, raw)

    if not raw.strip():
        log.error(
            "Intelligence synthesis failed: LM Studio returned empty content "
            "(finish_reason=%r, prompt_tokens=%s, completion_tokens=%s)",
            choice.finish_reason,
            getattr(response.usage, "prompt_tokens", "?"),
            getattr(response.usage, "completion_tokens", "?"),
        )
        return None

    # Strip markdown fences if the model wrapped the JSON despite instructions
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    if not stripped:
        log.error(
            "Intelligence synthesis failed: content was only markdown fences "
            "(finish_reason=%r, raw=%r)",
            choice.finish_reason,
            raw[:200],
        )
        return None

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        log.error(
            "Intelligence synthesis failed: JSON parse error: %s — raw content: %r",
            e,
            raw[:500],
        )
        return None

    error = _validate_intel(data)
    if error:
        log.error("Intelligence synthesis failed: %s", error)
        return None

    return data


_VALID_CONTINUITY = {"new", "confirmed", "escalating", "retired"}


def _sanitize_intel(intel: dict) -> dict:
    s = escape_html

    def sources(item: dict) -> list[int]:
        return [x for x in item.get("sources", []) if isinstance(x, int)]

    return {
        "situation_summary": s(intel["situation_summary"]),
        "key_themes": [
            {
                "theme": s(i["theme"]),
                "detail": s(i["detail"]),
                "sources": sources(i),
                "continuity": i.get("continuity") if i.get("continuity") in _VALID_CONTINUITY else "new",
            }
            for i in intel["key_themes"]
        ],
        "signals_and_warnings": [
            {"signal": s(i["signal"]), "assessment": s(i["assessment"]), "sources": sources(i)}
            for i in intel["signals_and_warnings"]
        ],
        "named_actors": [
            {"actor": s(i["actor"]), "role": s(i["role"]), "activity": s(i["activity"])}
            for i in intel["named_actors"]
        ],
    }


def _resolve_sources(items: list[dict], posts: list[dict], channel_links: dict[str, str]) -> list[dict]:
    resolved = []
    for item in items:
        source_links = []
        for idx in item.get("sources", []):
            if not (1 <= idx <= len(posts)):
                continue
            post = posts[idx - 1]
            slug = post["channel_slug"]
            ts = datetime.fromisoformat(post["timestamp"])
            entry = {"channel_slug": slug, "time_str": ts.strftime("%H:%M UTC")}
            username = channel_links.get(slug)
            if username:
                entry["link"] = f"https://t.me/{username}/{post['message_id']}"
            source_links.append(entry)
        resolved.append({**item, "source_links": source_links})
    return resolved


def _triaged_to_dicts(main_items: list) -> list[dict]:
    return [
        {
            "title": item.analysis.title or "",
            "summary": item.analysis.summary or "",
            "category": item.analysis.category or "Other",
            "threat_level": item.analysis.threat_level,
            "composite_score": item.composite_score,
            "channel_slug": item.post.channel_name,
            "message_id": item.post.message_id,
            "timestamp": item.post.timestamp.isoformat(),
            "entities": item.analysis.key_entities,
        }
        for item in main_items
    ]


def _render_front_page_md(
    intel: dict,
    target_date: date,
    posts: list[dict] | None = None,
    channel_links: dict[str, str] | None = None,
    emerging_entities: list[str] | None = None,
) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)
    tmpl = env.get_template("intel_front_page.md.j2")
    clean = _sanitize_intel(intel)
    posts = posts or []
    channel_links = channel_links or {}
    return tmpl.render(
        date=target_date.strftime("%A, %-d %B %Y"),
        situation_summary=clean["situation_summary"],
        key_themes=_resolve_sources(clean["key_themes"], posts, channel_links),
        signals_and_warnings=_resolve_sources(clean["signals_and_warnings"], posts, channel_links),
        named_actors=clean["named_actors"],
        emerging_entities=emerging_entities or [],
    )


def _md_to_pdf(md_text: str, date_str: str, date_dir: Path) -> Path:
    from markdown_pdf import MarkdownPdf, Section

    css_path = TEMPLATES_DIR / "briefing.css"
    user_css = css_path.read_text() if css_path.exists() else None

    pdf_obj = MarkdownPdf(toc_level=0)
    pdf_obj.meta["title"] = f"The Daily Telegram — Intelligence Assessment {date_str}"
    pdf_obj.add_section(Section(md_text), user_css=user_css)
    pdf_path = date_dir / f"intel_front_{date_str}.pdf"
    pdf_obj.save(str(pdf_path))
    return pdf_path


def _prepend_pdf(front_page_path: Path, briefing_path: Path) -> None:
    from pypdf import PdfWriter, PdfReader

    front_reader = PdfReader(str(front_page_path))
    briefing_reader = PdfReader(str(briefing_path))

    skip = 0
    existing_meta = briefing_reader.metadata or {}
    if existing_meta.get("/IntelPages"):
        try:
            skip = int(existing_meta["/IntelPages"])
        except ValueError:
            skip = 0

    writer = PdfWriter()
    for page in front_reader.pages:
        writer.add_page(page)
    for page in briefing_reader.pages[skip:]:
        writer.add_page(page)
    writer.add_metadata({"/IntelPages": str(len(front_reader.pages))})

    fd, tmp_path = tempfile.mkstemp(dir=briefing_path.parent, suffix=".tmp.pdf")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            writer.write(f)
        os.replace(tmp_path, briefing_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


async def run_analysis(config: AppConfig, target_date: date, main_items=None) -> None:
    date_str = target_date.isoformat()
    date_dir = Path(config.generation.output_dir) / date_str

    pdfs = sorted(date_dir.glob("TheDailyTelegram_*.pdf")) if date_dir.exists() else []
    if not pdfs:
        log.error("No briefing found for %s. Run --batch first.", date_str)
        return
    briefing_path = pdfs[-1]

    db = Database(config.storage.db_path)
    try:
        db.init_schema()

        if main_items is None:
            from tg_compiler.triage import triage as do_triage
            pairs = db.get_days_posts_with_analyses(date_str)
            if not pairs:
                log.error(
                    "No analysed posts found for %s — cannot generate intelligence front page",
                    date_str,
                )
                return
            channel_priorities = {ch.slug: ch.priority for ch in config.telegram.channels}
            channel_credibilities = {ch.slug: ch.credibility for ch in config.telegram.channels}
            content = do_triage(pairs, config.triage, today=target_date,
                                 channel_priorities=channel_priorities,
                                 channel_credibilities=channel_credibilities)
            posts = _triaged_to_dicts(content.main_items)
        else:
            posts = _triaged_to_dicts(main_items)

        if not posts:
            log.error(
                "No posts to synthesise for %s — cannot generate intelligence front page",
                date_str,
            )
            return

        history_start = (target_date - timedelta(days=TREND_WINDOW_DAYS - 1)).isoformat()
        history = db.get_posts_with_analyses_in_range(history_start, date_str)
        trends = compute_trends(history, target_date)

        previous_intel = db.get_intel_assessment((target_date - timedelta(days=1)).isoformat())

        log.info("Synthesising intelligence assessment from %d posts…", len(posts))
        intel = await synthesise(config, posts, trends=trends, previous_intel=previous_intel)
        if intel is None:
            return

        db.save_intel_assessment(date_str, intel)
    finally:
        db.close()

    channel_links = {
        ch.slug: ch.username.lstrip("@")
        for ch in config.telegram.channels
        if ch.username
    }
    md = _render_front_page_md(intel, target_date, posts=posts, channel_links=channel_links, emerging_entities=trends["emerging_entities"])
    front_page_pdf = _md_to_pdf(md, date_str, date_dir)

    try:
        _prepend_pdf(front_page_pdf, briefing_path)
    finally:
        front_page_pdf.unlink(missing_ok=True)

    log.info("Intelligence front page prepended → %s", briefing_path)
