from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from tg_compiler.triage import BriefingContent
from tg_compiler.utils import clean_entities

log = logging.getLogger(__name__)

# TODO(packaging): resolves relative to the source tree, not package data — an
# editable install (pip install -e) finds it, but a wheel/non-editable install won't.
TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"


_THREAT_BADGES = {
    "CRITICAL": '<b><span style="color:#c0392b;">&#9632; CRITICAL</span></b>',
    "HIGH":     '<b><span style="color:#d35400;">&#9632; HIGH</span></b>',
    "MODERATE": '<b><span style="color:#b7950b;">&#9632; MODERATE</span></b>',
    "LOW":      '<b><span style="color:#1e8449;">&#9632; LOW</span></b>',
}


def _threat_badge(threat_level: str) -> str:
    return _THREAT_BADGES.get(threat_level, "🟡 MODERATE")


def render_markdown(content: BriefingContent) -> str:
    """Render the Jinja2 briefing template to markdown. Resolves each executive
    (lead) item's post.media_paths to absolute paths in place before rendering —
    fitz.Story (used by _render_pdf) needs absolute image srcs to embed them."""
    # Lead (executive) items are the only ones rendered with embedded images.
    for item in content.executive_items:
        item.post.media_paths = [str(Path(p).resolve()) for p in item.post.media_paths]

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)
    env.globals["threat_badge"] = _threat_badge
    env.filters["clean_entities"] = clean_entities
    tmpl = env.get_template("briefing.md.j2")
    return tmpl.render(content=content)


def generate_briefing(
    content: BriefingContent,
    output_dir: str,
    pdf: bool = False,
    layout: str = "desktop",
) -> Path:
    """Render and save the markdown briefing to output_dir/YYYY-MM-DD/, optionally
    also rendering a PDF via _render_pdf. Returns the PDF path if pdf=True, else
    the markdown path."""
    date_str = content.date.isoformat()
    date_dir = Path(output_dir) / date_str
    date_dir.mkdir(parents=True, exist_ok=True)
    md_text = render_markdown(content)

    md_path = date_dir / f"briefing_{date_str}.md"
    md_path.write_text(md_text)
    log.info("Markdown briefing saved to %s", md_path)

    if pdf:
        return _render_pdf(md_text, date_dir, date_str, layout)
    return md_path


def _render_pdf(md_text: str, out: Path, date_str: str, layout: str = "desktop") -> Path:
    from markdown_pdf import MarkdownPdf, Section

    css_filename = "briefing.css" if layout == "desktop" else "briefing_mobile.css"
    css_path = TEMPLATES_DIR / css_filename
    user_css = css_path.read_text() if css_path.exists() else None

    ts = datetime.now().strftime("%H%M%S")
    pdf_obj = MarkdownPdf(toc_level=0)
    pdf_obj.meta["title"] = f"The Daily Telegram {date_str}"
    # root="/" — image srcs are absolute paths; fitz.Story resolves them against
    # the section root, and the default "." silently drops them ([image] placeholder).
    pdf_obj.add_section(Section(md_text, root="/"), user_css=user_css)
    pdf_path = out / f"TheDailyTelegram_{date_str}_{ts}.pdf"
    pdf_obj.save(str(pdf_path))
    log.info("PDF briefing saved to %s", pdf_path)
    return pdf_path
