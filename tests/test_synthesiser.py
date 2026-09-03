"""Tests for synthesiser JSON validation and PDF merge."""
import pytest

from tg_compiler.synthesiser import _prepend_pdf, _resolve_sources, _validate_intel

# ---------------------------------------------------------------------------
# _validate_intel
# ---------------------------------------------------------------------------

def _good_intel() -> dict:
    return {
        "situation_summary": "Tensions remain elevated across the region.",
        "key_themes": [{"theme": "Escalation", "detail": "Fighting has intensified.", "sources": [1], "continuity": "new"}],
        "signals_and_warnings": [{"signal": "Airspace closure", "assessment": "Watch for further closures.", "sources": [1]}],
        "named_actors": [{"actor": "PM Netanyahu", "role": "Israeli PM", "activity": "Ordered strikes."}],
    }


def test_validate_intel_valid_passes():
    assert _validate_intel(_good_intel()) is None


def test_validate_intel_missing_key_fails():
    data = _good_intel()
    del data["situation_summary"]
    assert _validate_intel(data) is not None


def test_validate_intel_empty_situation_summary_fails():
    data = _good_intel()
    data["situation_summary"] = "   "
    assert _validate_intel(data) is not None


@pytest.mark.parametrize("key", ["key_themes", "signals_and_warnings", "named_actors"])
def test_validate_intel_empty_list_fails(key):
    data = _good_intel()
    data[key] = []
    assert _validate_intel(data) is not None


def test_validate_intel_key_themes_missing_subkey():
    data = _good_intel()
    data["key_themes"] = [{"theme": "X"}]  # missing 'detail'
    assert _validate_intel(data) is not None


def test_validate_intel_signals_missing_subkey():
    data = _good_intel()
    data["signals_and_warnings"] = [{"signal": "X"}]  # missing 'assessment'
    assert _validate_intel(data) is not None


def test_validate_intel_named_actors_missing_subkey():
    data = _good_intel()
    data["named_actors"] = [{"actor": "X", "role": "Y"}]  # missing 'activity'
    assert _validate_intel(data) is not None


def test_validate_intel_key_themes_missing_sources():
    data = _good_intel()
    data["key_themes"] = [{"theme": "X", "detail": "Y"}]  # missing 'sources'
    assert _validate_intel(data) is not None


def test_validate_intel_signals_missing_sources():
    data = _good_intel()
    data["signals_and_warnings"] = [{"signal": "X", "assessment": "Y"}]  # missing 'sources'
    assert _validate_intel(data) is not None


# ---------------------------------------------------------------------------
# _sanitize_intel
# ---------------------------------------------------------------------------

def test_sanitize_intel_filters_non_int_sources_and_defaults_continuity():
    from tg_compiler.synthesiser import _sanitize_intel

    data = _good_intel()
    data["key_themes"][0]["sources"] = [1, "2", 3]
    data["key_themes"][0]["continuity"] = "bogus"
    data["signals_and_warnings"][0]["sources"] = ["x", 2]

    clean = _sanitize_intel(data)
    assert clean["key_themes"][0]["sources"] == [1, 3]
    assert clean["key_themes"][0]["continuity"] == "new"
    assert clean["signals_and_warnings"][0]["sources"] == [2]


def test_sanitize_intel_keeps_valid_continuity():
    from tg_compiler.synthesiser import _sanitize_intel

    data = _good_intel()
    data["key_themes"][0]["continuity"] = "escalating"
    clean = _sanitize_intel(data)
    assert clean["key_themes"][0]["continuity"] == "escalating"


# ---------------------------------------------------------------------------
# _resolve_sources
# ---------------------------------------------------------------------------

def _post_dict(channel_slug, message_id, timestamp):
    return {
        "title": "T", "summary": "S", "category": "Other", "threat_level": "MODERATE",
        "composite_score": 1.0, "channel_slug": channel_slug, "message_id": message_id,
        "timestamp": timestamp, "entities": [],
    }


def test_resolve_sources_with_channel_link():
    posts = [_post_dict("chan_a", 42, "2026-06-09T14:30:00+00:00")]
    items = [{"theme": "X", "detail": "Y", "sources": [1], "continuity": "new"}]
    resolved = _resolve_sources(items, posts, {"chan_a": "chan_a"})
    assert resolved[0]["source_links"] == [
        {"channel_slug": "chan_a", "time_str": "14:30 UTC", "link": "https://t.me/chan_a/42"}
    ]


def test_resolve_sources_without_channel_link():
    posts = [_post_dict("chan_b", 7, "2026-06-09T08:00:00+00:00")]
    items = [{"signal": "X", "assessment": "Y", "sources": [1]}]
    resolved = _resolve_sources(items, posts, {})
    assert resolved[0]["source_links"] == [{"channel_slug": "chan_b", "time_str": "08:00 UTC"}]


def test_resolve_sources_out_of_range_index_ignored():
    posts = [_post_dict("chan_a", 1, "2026-06-09T08:00:00+00:00")]
    items = [{"theme": "X", "detail": "Y", "sources": [99], "continuity": "new"}]
    resolved = _resolve_sources(items, posts, {})
    assert resolved[0]["source_links"] == []


# ---------------------------------------------------------------------------
# _format_trends / _format_previous_themes
# ---------------------------------------------------------------------------

def test_format_trends_empty_returns_empty_string():
    from tg_compiler.synthesiser import _format_trends
    assert _format_trends(None) == ""
    assert _format_trends({"entity_deltas": [], "category_deltas": [], "emerging_entities": []}) == ""


def test_format_trends_includes_entity_and_category_lines():
    from tg_compiler.synthesiser import _format_trends
    trends = {
        "entity_deltas": [{"entity": "bahrain", "today_count": 14, "prior_count": 0}],
        "category_deltas": [{"category": "Military", "today_count": 5, "prior_count": 2}],
        "emerging_entities": ["bahrain"],
    }
    text = _format_trends(trends)
    assert "bahrain: 0 -> 14" in text
    assert "Military: 2 -> 5" in text


def test_format_previous_themes_empty_returns_empty_string():
    from tg_compiler.synthesiser import _format_previous_themes
    assert _format_previous_themes(None) == ""
    assert _format_previous_themes({"key_themes": []}) == ""


def test_format_previous_themes_includes_theme_text():
    from tg_compiler.synthesiser import _format_previous_themes
    previous = {"key_themes": [{"theme": "Escalation", "detail": "Fighting intensified."}]}
    text = _format_previous_themes(previous)
    assert "Escalation" in text
    assert "Fighting intensified." in text


# ---------------------------------------------------------------------------
# _render_front_page_md
# ---------------------------------------------------------------------------

def test_render_front_page_md_includes_sources_and_continuity():
    from datetime import date

    from tg_compiler.synthesiser import _render_front_page_md

    intel = _good_intel()
    intel["key_themes"][0]["continuity"] = "escalating"
    posts = [_post_dict("chan_a", 42, "2026-06-09T14:30:00+00:00")]
    md = _render_front_page_md(intel, date(2026, 6, 9), posts=posts, channel_links={"chan_a": "chan_a"})

    assert "(escalating)" in md
    assert "Sources: chan_a (14:30 UTC)" in md
    assert "https://t.me/chan_a/42" in md


def test_render_front_page_md_emerging_section_present_when_nonempty():
    from datetime import date

    from tg_compiler.synthesiser import _render_front_page_md

    md = _render_front_page_md(_good_intel(), date(2026, 6, 9), emerging_entities=["bahrain"])
    assert "Emerging Actors / Topics" in md
    # normalised lowercase entities are title-cased for display
    assert "Bahrain" in md


def test_render_front_page_md_emerging_section_absent_when_empty():
    from datetime import date

    from tg_compiler.synthesiser import _render_front_page_md

    md = _render_front_page_md(_good_intel(), date(2026, 6, 9))
    assert "Emerging Actors / Topics" not in md


# ---------------------------------------------------------------------------
# _prepend_pdf
# ---------------------------------------------------------------------------

def test_prepend_pdf_merges_pages(tmp_path):
    from pypdf import PdfReader, PdfWriter

    def make_pdf(path, n_pages=1):
        writer = PdfWriter()
        for _ in range(n_pages):
            writer.add_blank_page(width=612, height=792)
        with open(path, "wb") as f:
            writer.write(f)

    front = tmp_path / "front.pdf"
    briefing = tmp_path / "briefing.pdf"
    make_pdf(front, n_pages=1)
    make_pdf(briefing, n_pages=2)

    _prepend_pdf(front, briefing)

    result = PdfReader(str(briefing))
    assert len(result.pages) == 3  # 1 front + 2 briefing


def test_prepend_pdf_idempotent_on_rerun(tmp_path):
    from pypdf import PdfReader, PdfWriter

    def make_pdf(path, n_pages=1):
        writer = PdfWriter()
        for _ in range(n_pages):
            writer.add_blank_page(width=612, height=792)
        with open(path, "wb") as f:
            writer.write(f)

    front = tmp_path / "front.pdf"
    briefing = tmp_path / "briefing.pdf"
    make_pdf(front, n_pages=1)
    make_pdf(briefing, n_pages=2)

    _prepend_pdf(front, briefing)
    _prepend_pdf(front, briefing)

    result = PdfReader(str(briefing))
    assert len(result.pages) == 3  # 1 front + 2 briefing, not stacked twice


def test_prepend_pdf_preserves_briefing_title_metadata(tmp_path):
    from pypdf import PdfReader, PdfWriter

    def make_pdf(path, n_pages=1, metadata=None):
        writer = PdfWriter()
        for _ in range(n_pages):
            writer.add_blank_page(width=612, height=792)
        if metadata:
            writer.add_metadata(metadata)
        with open(path, "wb") as f:
            writer.write(f)

    front = tmp_path / "front.pdf"
    briefing = tmp_path / "briefing.pdf"
    make_pdf(front, n_pages=1)
    make_pdf(briefing, n_pages=2, metadata={"/Title": "The Daily Telegram 2026-06-09"})

    _prepend_pdf(front, briefing)

    result = PdfReader(str(briefing))
    assert result.metadata.get("/Title") == "The Daily Telegram 2026-06-09"

    # Idempotent re-run must also keep the title.
    _prepend_pdf(front, briefing)
    result = PdfReader(str(briefing))
    assert result.metadata.get("/Title") == "The Daily Telegram 2026-06-09"


def test_prepend_pdf_no_tmp_file_left(tmp_path):
    from pypdf import PdfWriter

    def make_pdf(path):
        w = PdfWriter()
        w.add_blank_page(width=612, height=792)
        with open(path, "wb") as f:
            w.write(f)

    front = tmp_path / "front.pdf"
    briefing = tmp_path / "briefing.pdf"
    make_pdf(front)
    make_pdf(briefing)

    _prepend_pdf(front, briefing)

    tmp_files = list(tmp_path.glob("*.tmp.pdf"))
    assert tmp_files == [], f"Unexpected temp files: {tmp_files}"


# ---------------------------------------------------------------------------
# _triaged_to_dicts
# ---------------------------------------------------------------------------

def test_triaged_to_dicts_converts_correctly():
    from datetime import datetime, timezone

    from tg_compiler.db import AnalysisRecord, PostRecord
    from tg_compiler.synthesiser import _triaged_to_dicts
    from tg_compiler.triage import TriagedPost

    post = PostRecord(
        channel_id=1,
        channel_name="chan_a",
        message_id=10,
        timestamp=datetime(2026, 6, 9, 10, 30, tzinfo=timezone.utc),
        text="original text",
        media_paths=[],
        has_images=False,
        raw_json="{}",
    )
    analysis = AnalysisRecord(
        post_id=1,
        title="Big Event",
        summary="A notable development occurred.",
        importance_score=4,
        urgency_score=3,
        credibility_score=5,
        relevance_score=4,
        category="Military",
        key_entities=["Russia", "Ukraine"],
        model_used="test",
        threat_level="HIGH",
    )
    item = TriagedPost(post=post, analysis=analysis, composite_score=3.9)
    result = _triaged_to_dicts([item])

    assert len(result) == 1
    d = result[0]
    assert d["title"] == "Big Event"
    assert d["summary"] == "A notable development occurred."
    assert d["category"] == "Military"
    assert d["threat_level"] == "HIGH"
    assert d["composite_score"] == 3.9
    assert d["channel_slug"] == "chan_a"
    assert d["message_id"] == 10
    assert d["timestamp"] == "2026-06-09T10:30:00+00:00"
    assert d["entities"] == ["Russia", "Ukraine"]


def test_triaged_to_dicts_empty_list():
    from tg_compiler.synthesiser import _triaged_to_dicts
    assert _triaged_to_dicts([]) == []


def test_triaged_to_dicts_missing_title_defaults_empty():
    from datetime import datetime, timezone

    from tg_compiler.db import AnalysisRecord, PostRecord
    from tg_compiler.synthesiser import _triaged_to_dicts
    from tg_compiler.triage import TriagedPost

    post = PostRecord(
        channel_id=2, channel_name="chan_b", message_id=99,
        timestamp=datetime(2026, 6, 9, tzinfo=timezone.utc),
        text="", media_paths=[], has_images=False, raw_json="{}",
    )
    analysis = AnalysisRecord(
        post_id=2, title="", summary="Some summary.",
        importance_score=2, urgency_score=2, credibility_score=2, relevance_score=2,
        category="Other", key_entities=[], model_used="test",
    )
    item = TriagedPost(post=post, analysis=analysis, composite_score=2.0)
    result = _triaged_to_dicts([item])
    assert result[0]["title"] == ""


# ---------------------------------------------------------------------------
# run_analysis signature
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_analysis_no_briefing_returns_without_error(tmp_path, caplog):
    import logging
    from datetime import date

    from tg_compiler.config import AppConfig
    from tg_compiler.synthesiser import run_analysis

    cfg = AppConfig.model_validate({
        "telegram": {"api_id": 1, "api_hash": "x", "channels": []},
        "lmstudio": {"model": "test"},
        "generation": {"output_dir": str(tmp_path)},
    })
    with caplog.at_level(logging.ERROR):
        await run_analysis(cfg, date(2026, 6, 9))

    assert any("Run --batch first" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_run_analysis_accepts_main_items_kwarg():
    """run_analysis must accept main_items=[] without error (empty → early exit)."""
    from datetime import date

    from tg_compiler.config import AppConfig
    from tg_compiler.synthesiser import run_analysis

    cfg = AppConfig.model_validate({
        "telegram": {"api_id": 1, "api_hash": "x", "channels": []},
        "lmstudio": {"model": "test"},
    })
    # Empty main_items — should log error and return without crashing
    await run_analysis(cfg, date(2026, 6, 9), main_items=[])


async def test_synthesise_addresses_the_configured_synthesis_model(monkeypatch):
    import json
    from types import SimpleNamespace

    from tg_compiler import synthesiser as synth_mod
    from tg_compiler.config import AppConfig, LMStudioConfig, TelegramConfig

    intel = {
        "situation_summary": "A quiet day with little movement across the theatres.",
        "key_themes": [{"theme": "T", "detail": "D", "sources": [1]}],
        "signals_and_warnings": [{"signal": "S", "assessment": "A", "sources": [1]}],
        "named_actors": [{"actor": "A", "role": "R", "activity": "X"}],
    }
    calls = []

    def fake_openai(**_kwargs):
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(intel)), finish_reason="stop",
            )])

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(synth_mod, "OpenAI", fake_openai)

    config = AppConfig(
        telegram=TelegramConfig(api_id=1, api_hash="x", channels=[]),
        lmstudio=LMStudioConfig(
            model="fallback-model", analysis_model="small-model",
            synthesis_model="reasoning-model",
        ),
    )
    result = await synth_mod.synthesise(config, [{"index": 1, "summary": "s"}])

    assert result is not None
    assert calls[0]["model"] == "reasoning-model"
