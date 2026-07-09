from datetime import date, datetime, timezone

from tg_compiler.db import AnalysisRecord, PostRecord
from tg_compiler.generator import render_markdown
from tg_compiler.triage import BriefingContent, CorroborationRef, TriagedPost


def make_triaged(importance=4, category="Breaking News",
                 summary="Key event occurred.", msg_id=1, channel="news"):
    post = PostRecord(
        id=msg_id, channel_id=1, channel_name=channel, message_id=msg_id,
        timestamp=datetime(2026, 6, 7, 14, 30, tzinfo=timezone.utc),
        text="Original text.", media_paths=[], has_images=False, raw_json="{}",
    )
    analysis = AnalysisRecord(
        id=msg_id, post_id=msg_id, summary=summary,
        importance_score=importance, urgency_score=3, credibility_score=4, relevance_score=3,
        category=category, key_entities=["Alice"], model_used="test",
    )
    return TriagedPost(post=post, analysis=analysis, composite_score=float(importance))


def make_content(n_main=2, n_appendix=1):
    return BriefingContent(
        date=date(2026, 6, 7),
        main_items=[make_triaged(msg_id=i) for i in range(n_main)],
        appendix_items=[make_triaged(importance=1, msg_id=n_main + i) for i in range(n_appendix)],
        channel_names=["news"],
    )


def test_markdown_contains_date():
    md = render_markdown(make_content())
    # Date appears in post timestamps within the briefing body
    assert "14:30 UTC" in md


def test_markdown_has_lead_reports_section():
    md = render_markdown(make_content(n_main=3))
    assert "Lead Reports" in md


def test_markdown_has_channel_section():
    md = render_markdown(make_content())
    assert "news" in md


def test_markdown_shows_threat_badge():
    md = render_markdown(make_content())
    # default threat_level is MODERATE
    assert "MODERATE" in md or "HIGH" in md or "CRITICAL" in md or "LOW" in md


def test_markdown_includes_post_summary():
    md = render_markdown(make_content())
    assert "Key event occurred." in md


def test_in_brief_present_when_low_score_posts_exist():
    md = render_markdown(make_content(n_appendix=1))
    assert "In Brief" in md
    assert "Appendix" not in md


def test_empty_main_items_still_renders():
    content = BriefingContent(
        date=date(2026, 6, 7), main_items=[], appendix_items=[], channel_names=[]
    )
    md = render_markdown(content)
    assert "Lead Reports" in md
    assert "No lead reports today" in md


from tg_compiler.generator import generate_briefing


def test_pdf_file_is_created(tmp_path):
    content = make_content(n_main=2)
    path = generate_briefing(content, output_dir=str(tmp_path), pdf=True)
    assert path.suffix == ".pdf"
    assert path.exists()
    assert path.stat().st_size > 0


def test_markdown_file_is_always_created(tmp_path):
    content = make_content()
    path = generate_briefing(content, output_dir=str(tmp_path), pdf=False)
    assert path.suffix == ".md"
    assert path.exists()


from tg_compiler.main import purge_old_media


def test_purge_removes_old_directories(tmp_path):
    from datetime import datetime, timedelta, timezone

    today = datetime.now(timezone.utc).date()
    old_dir = tmp_path / "news" / "2020-01-01"
    recent_dir = tmp_path / "news" / (today - timedelta(days=5)).isoformat()
    old_dir.mkdir(parents=True)
    recent_dir.mkdir(parents=True)
    (old_dir / "1.jpg").write_bytes(b"x")
    (recent_dir / "2.jpg").write_bytes(b"x")

    removed = purge_old_media(str(tmp_path), retention_days=30)
    assert removed == 1
    assert not old_dir.exists()
    assert recent_dir.exists()


def test_purge_nonexistent_dir_returns_zero():
    assert purge_old_media("/nonexistent/path/abc123", retention_days=30) == 0


def test_corroboration_line_rendered_when_present():
    content = make_content(n_main=1, n_appendix=0)
    content.main_items[0].corroborations = [
        CorroborationRef(channel_slug="other_chan", message_id=42, timestamp=datetime(2026, 6, 7, 14, 0, tzinfo=timezone.utc))
    ]
    content.executive_items = content.main_items[:1]  # corroborations render in Lead Reports
    content.channel_links = {"other_chan": "other_chan"}
    md = render_markdown(content)
    assert "Corroborated by 1 other channel" in md
    assert "other_chan" in md


def test_corroboration_line_absent_when_none():
    md = render_markdown(make_content(n_main=1, n_appendix=0))
    # the static reader's key mentions the phrase, so check the per-item bold marker
    assert "**Corroborated by" not in md


def test_pipeline_stats_rendered_when_scraped_set():
    content = make_content(n_main=1, n_appendix=0)
    content.posts_scraped = 100
    content.posts_analysed = 90
    content.posts_skipped = 10
    content.posts_clustered = 5
    md = render_markdown(content)
    assert "100 scraped · 90 analysed · 10 skipped (low-content)" in md
    assert "5 duplicates merged" in md


def test_pipeline_stats_absent_when_not_set():
    md = render_markdown(make_content(n_main=1, n_appendix=0))
    assert "scraped" not in md


def test_score_display_clamped_at_five():
    content = make_content(n_main=2, n_appendix=0)
    content.main_items[0].composite_score = 7.5  # corroboration boost can exceed 5
    content.main_items[1].composite_score = 6.5
    content.executive_items = content.main_items[:1]  # one lead, one compact line
    md = render_markdown(content)
    assert "Score 5.0/5" in md   # lead byline
    assert "· 5.0/5" in md       # compact line
    assert "7.5/5" not in md and "6.5/5" not in md


def test_main_items_rendered_in_score_order_not_by_channel():
    content = make_content(n_main=0, n_appendix=0)
    a = make_triaged(msg_id=1, channel="zeta", summary="Highest priority story.")
    a.composite_score = 5.0
    b = make_triaged(msg_id=2, channel="alpha", summary="Middle priority story.")
    b.composite_score = 4.0
    c = make_triaged(msg_id=3, channel="zeta", summary="Lowest priority story.")
    c.composite_score = 3.0
    content.main_items = [a, b, c]
    content.channel_names = ["alpha", "zeta"]
    md = render_markdown(content)
    assert "Other Developments" in md
    body = md[md.index("Other Developments"):]
    assert (
        body.index("Highest priority story.")
        < body.index("Middle priority story.")
        < body.index("Lowest priority story.")
    )
    # channel-by-channel sections are gone
    assert "## Channel:" not in md
    assert "No posts above threshold" not in md


def test_smallprint_readers_key_always_present():
    md = render_markdown(make_content())
    assert "READER'S KEY" in md
    assert "0.4×importance + 0.3×urgency + 0.2×credibility + 0.1×relevance" in md
    # also present when the briefing is empty
    empty = BriefingContent(date=date(2026, 6, 7), main_items=[], appendix_items=[], channel_names=[])
    assert "READER'S KEY" in render_markdown(empty)


def test_pdf_embeds_images_from_absolute_paths(tmp_path):
    import fitz

    img_path = tmp_path / "photo.png"
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 12, 12))
    pix.clear_with(120)
    pix.save(str(img_path))

    content = make_content(n_main=1, n_appendix=0)
    content.main_items[0].post.media_paths = [str(img_path)]
    content.executive_items = content.main_items[:1]  # images render in Lead Reports
    pdf_path = generate_briefing(content, output_dir=str(tmp_path), pdf=True)

    doc = fitz.open(str(pdf_path))
    n_images = sum(len(page.get_images()) for page in doc)
    doc.close()
    assert n_images >= 1

def test_same_channel_corroborations_render_as_related_posts():
    content = make_content(n_main=1, n_appendix=0)
    content.main_items[0].corroborations = [
        CorroborationRef(channel_slug="news", message_id=43,
                         timestamp=datetime(2026, 6, 7, 15, 0, tzinfo=timezone.utc))
    ]
    content.executive_items = content.main_items[:1]
    content.channel_links = {"news": "news"}
    md = render_markdown(content)
    assert "**Corroborated by" not in md
    assert "Related posts from this channel" in md
    assert "https://t.me/news/43" in md


def test_corroboration_count_uses_distinct_other_channels():
    content = make_content(n_main=1, n_appendix=0)
    ts = datetime(2026, 6, 7, 15, 0, tzinfo=timezone.utc)
    content.main_items[0].corroborations = [
        CorroborationRef(channel_slug="chan_b", message_id=1, timestamp=ts),
        CorroborationRef(channel_slug="chan_b", message_id=2, timestamp=ts),
        CorroborationRef(channel_slug="news", message_id=3, timestamp=ts),
    ]
    content.executive_items = content.main_items[:1]
    md = render_markdown(content)
    assert "Corroborated by 1 other channel:" in md
    assert "Related posts from this channel" in md


def test_lead_items_not_repeated_in_lower_sections():
    content = make_content(n_main=2, n_appendix=1)
    content.main_items[0].analysis.title = "Lead headline"
    content.main_items[0].analysis.summary = "Unique lead story summary."
    content.main_items[1].analysis.summary = "Unique other development summary."
    content.appendix_items[0].analysis.summary = "Unique in-brief summary."
    content.executive_items = content.main_items[:1]
    md = render_markdown(content)
    assert md.count("Unique lead story summary.") == 1
    assert md.count("Unique other development summary.") == 1
    assert md.count("Unique in-brief summary.") == 1
    # the lead story renders in full, the others compactly
    lead_block = md[md.index("Lead Reports"):md.index("Other Developments")]
    assert "Unique lead story summary." in lead_block


def test_critical_lead_item_from_appendix_not_repeated_in_brief():
    content = make_content(n_main=1, n_appendix=1)
    content.appendix_items[0].analysis.title = "Critical alert headline"
    content.appendix_items[0].analysis.summary = "Low-scored critical alert summary."
    content.executive_items = [content.appendix_items[0]]
    md = render_markdown(content)
    assert md.count("Low-scored critical alert summary.") == 1
    assert "## In Brief" not in md  # its only appendix item was promoted to lead


def test_lead_images_use_width_attribute_not_md_image():
    # fitz.Story ignores CSS max-width; a natural-size image at a page break makes it
    # silently drop the element and following blocks. Explicit width attr avoids this.
    content = make_content(n_main=1, n_appendix=0)
    content.main_items[0].post.media_paths = ["/abs/path/img.jpg"]
    content.executive_items = content.main_items[:1]
    md = render_markdown(content)
    assert '<img src="/abs/path/img.jpg" width="300">' in md
    assert "![](" not in md


def test_other_developments_item_shows_summary_and_corroboration():
    content = make_content(n_main=1, n_appendix=0)
    item = content.main_items[0]
    item.analysis.title = "Concise headline"
    item.analysis.summary = "Full summary of the development."
    item.corroborations = [
        CorroborationRef(channel_slug="chan_b", message_id=7,
                         timestamp=datetime(2026, 6, 7, 15, 0, tzinfo=timezone.utc))
    ]
    content.channel_links = {"chan_b": "chan_b"}
    md = render_markdown(content)
    dev_block = md[md.index("Other Developments"):md.index("## Statistics")]
    assert "Concise headline" in dev_block
    assert "Full summary of the development." in dev_block
    assert "Corroborated by 1 other channel" in dev_block
    assert "https://t.me/chan_b/7" in dev_block


def test_in_brief_shows_corroboration_marker():
    content = make_content(n_main=0, n_appendix=1)
    content.appendix_items[0].corroborations = [
        CorroborationRef(channel_slug="chan_b", message_id=7,
                         timestamp=datetime(2026, 6, 7, 15, 0, tzinfo=timezone.utc)),
        CorroborationRef(channel_slug="chan_c", message_id=8,
                         timestamp=datetime(2026, 6, 7, 15, 0, tzinfo=timezone.utc)),
    ]
    md = render_markdown(content)
    assert "+2 corrob" in md


def test_readers_key_states_threshold():
    content = make_content(n_main=1, n_appendix=0)
    content.min_composite_score = 3.5
    md = render_markdown(content)
    assert "at least 3.5" in md
