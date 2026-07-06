import pytest
from pydantic import ValidationError
from tg_compiler.analyzer import (
    PostAnalysis, parse_analysis_fallback, build_messages, _clean_image_insights,
    _check_numeric_consistency, _sanitize, analysis_to_record,
)
from tg_compiler.utils import clean_entities


def test_post_analysis_parses_valid_json():
    data = {
        "summary": "A test post about something.",
        "importance_score": 3,
        "urgency_score": 2,
        "credibility_score": 4,
        "relevance_score": 3,
        "category": "Analysis",
        "key_entities": ["Alice", "ACME Corp"],
        "image_substantive": False,
        "image_description": None,
        "reasoning": "Moderate importance.",
    }
    pa = PostAnalysis.model_validate(data)
    assert pa.importance_score == 3
    assert pa.category == "Analysis"
    assert "Alice" in pa.key_entities


def test_importance_score_out_of_range_raises():
    data = {
        "summary": "x", "importance_score": 6, "urgency_score": 1,
        "credibility_score": 1, "relevance_score": 1,
        "category": "Other", "reasoning": "r",
    }
    with pytest.raises(ValidationError):
        PostAnalysis.model_validate(data)


def test_fallback_parser_extracts_score_from_prose():
    raw = "The importance score is 4. Summary: Breaking development in the region. Category: Breaking News."
    pa = parse_analysis_fallback(raw)
    assert pa.importance_score == 4
    assert "Breaking" in pa.summary


def test_fallback_parser_returns_defaults_when_nothing_found():
    pa = parse_analysis_fallback("completely unstructured text with no useful fields")
    assert 1 <= pa.importance_score <= 5
    assert pa.summary != ""
    assert pa.category in ["Breaking News", "Analysis", "Official Statement", "Rumor", "Media", "Other"]


def test_build_messages_text_only(sample_post):
    sample_post.media_paths = []
    messages = build_messages(sample_post, system_prompt="You are an analyst.")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_entity_containing_key_entities_is_filtered():
    entities = ['key_entities,["IRGC Aerospace Force","Hamas"]', "IRGC Aerospace Force", "Hamas"]
    result = clean_entities(entities)
    assert "IRGC Aerospace Force" in result
    assert "Hamas" in result
    assert not any("key_entities" in e for e in result)


def test_entity_containing_bare_false_is_filtered():
    entities = ["false", "Israel", "true", "Pentagon"]
    result = clean_entities(entities)
    assert "Israel" in result
    assert "Pentagon" in result
    assert "false" not in result
    assert "true" not in result


def test_image_description_with_json_artifact_returns_none():
    assert _clean_image_insights(".json(post_analysis){image_description: null}") is None


def test_numeric_consistency_same_numbers_consistent():
    assert _check_numeric_consistency("A 7.8 magnitude earthquake hit Mindanao", "M7.8 earthquake near Mindanao coast") is True


def test_numeric_consistency_contradicting_numbers_inconsistent():
    # 7.8 vs 8.4 — same order of magnitude, >5% difference → inconsistent
    assert _check_numeric_consistency("Major 7.8 Magnitude Earthquake Hits Philippines", "A map showing the epicenter of an M8.4 earthquake in Mindanao") is False


def test_numeric_consistency_no_numbers_consistent():
    assert _check_numeric_consistency("Airstrikes reported near the border", "Smoke rising from a building near a road") is True


def test_numeric_consistency_different_magnitude_not_compared():
    # 7.8 (magnitude) vs 1000 (casualties) — >10x apart, should not be compared → consistent
    assert _check_numeric_consistency("7.8 magnitude earthquake struck the region", "Around 1000 people visible in the image") is True


def test_numeric_consistency_ignores_time_of_day_tokens():
    # "14:30" (time) vs "30 dead" (casualty count) previously collided as 14 vs 30.
    assert _check_numeric_consistency("Strike at 14:30 near Kharkiv", "Image shows 30 vehicles destroyed") is True


def test_numeric_consistency_ignores_year_tokens():
    # 203 (casualty count) vs 2026 (year) has ratio just under 10x, so without
    # stripping the year token this would be wrongly flagged as a contradiction.
    assert _check_numeric_consistency("Official toll of 203 confirmed dead", "Photo taken in 2026 shows the aftermath") is True


def test_numeric_consistency_genuine_contradiction_still_detected():
    assert _check_numeric_consistency("Officials say 12 killed in the blast", "Image caption reads 45 killed") is False


def test_build_messages_truncates_long_text(sample_post):
    sample_post.text = "x" * 5000
    sample_post.media_paths = []
    messages = build_messages(sample_post, system_prompt="You are an analyst.")
    user_content = messages[1]["content"]
    text_part = next(p for p in user_content if p["type"] == "text")
    assert text_part["text"].count("x") == 3000


def _analysis(**overrides):
    base = dict(
        title="A normal title",
        summary="A normal summary about events.",
        importance_score=3, urgency_score=3, credibility_score=3, relevance_score=3,
        category="Analysis", key_entities=[], reasoning="",
    )
    base.update(overrides)
    return PostAnalysis.model_validate(base)


@pytest.mark.parametrize("refusal_summary", [
    "The user provided a post from RerumNovarum at a future date, but no content was provided for analysis.",
    "I cannot analyze this post as no content was provided.",
    "Unable to analyse the image as it was not included.",
    "As an AI, I am unable to analyse this content.",
    "No content provided for this post.",
])
def test_sanitize_strips_refusal_summary(refusal_summary):
    pa = _analysis(summary=refusal_summary)
    result = _sanitize(pa)
    assert result.summary == ""


def test_sanitize_strips_refusal_title():
    pa = _analysis(title="Türkiye's commitment to the user to establish peace")
    result = _sanitize(pa)
    assert result.title == ""


def test_sanitize_keeps_normal_summary_and_title():
    pa = _analysis(title="Iran launches missiles at US targets", summary="Multiple missiles fired overnight.")
    result = _sanitize(pa)
    assert result.title == "Iran launches missiles at US targets"
    assert result.summary == "Multiple missiles fired overnight."


def test_analysis_to_record_includes_title():
    pa = _analysis(title="Headline here", key_entities=["Iran"])
    record = analysis_to_record(post_id=7, analysis=pa, model_used="test-model")
    assert record.post_id == 7
    assert record.title == "Headline here"
    assert record.model_used == "test-model"
    assert record.threat_level == pa.threat_level
    assert record.key_entities == ["Iran"]


@pytest.fixture
def app_config():
    from tg_compiler.config import AppConfig, TelegramConfig, LMStudioConfig

    return AppConfig(
        telegram=TelegramConfig(api_id=1, api_hash="x", channels=[]),
        lmstudio=LMStudioConfig(model="test-model"),
    )


async def test_process_unanalysed_skips_short_textonly_post(db, app_config, monkeypatch):
    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord
    from datetime import datetime, timezone

    short_post = PostRecord(
        channel_id=1, channel_name="chan", message_id=1,
        timestamp=datetime(2026, 6, 7, tzinfo=timezone.utc),
        text="ok", media_paths=[], has_images=False, raw_json="{}",
    )
    long_post = PostRecord(
        channel_id=1, channel_name="chan", message_id=2,
        timestamp=datetime(2026, 6, 7, tzinfo=timezone.utc),
        text="x" * 50, media_paths=[], has_images=False, raw_json="{}",
    )
    db.insert_post(short_post)
    db.insert_post(long_post)

    analyzer = Analyzer(app_config, db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)

    async def fake_analyze_post(post, channel_cfg=None):
        return _analysis(summary="Real analysis output for a long post.")

    monkeypatch.setattr(analyzer, "analyze_post", fake_analyze_post)

    analysed_count, skipped_count = await analyzer.process_unanalysed()
    assert analysed_count == 1
    assert skipped_count == 1

    pairs = db.get_days_posts_with_analyses("2026-06-07")
    by_id = {p.message_id: a for p, a in pairs}
    assert by_id[1].category == "Skipped"
    assert by_id[1].importance_score is None
    assert by_id[2].category == "Analysis"


async def test_process_unanalysed_analyses_short_caption_video_post(db, app_config, monkeypatch):
    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord
    from datetime import datetime, timezone

    video_post = PostRecord(
        channel_id=1, channel_name="chan", message_id=1,
        timestamp=datetime(2026, 6, 7, tzinfo=timezone.utc),
        text="Strike footage", media_paths=[], has_images=False, has_video=True, raw_json="{}",
    )
    db.insert_post(video_post)

    analyzer = Analyzer(app_config, db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)

    async def fake_analyze_post(post, channel_cfg=None):
        return _analysis(summary="Video shows strike footage from the area.")

    monkeypatch.setattr(analyzer, "analyze_post", fake_analyze_post)

    analysed_count, skipped_count = await analyzer.process_unanalysed()
    assert analysed_count == 1
    assert skipped_count == 0

    pairs = db.get_days_posts_with_analyses("2026-06-07")
    assert pairs[0][1].category == "Analysis"


def test_clean_image_insights_rejects_none_provided():
    assert _clean_image_insights("None provided") is None
    assert _clean_image_insights("none provided.") is None


def test_sanitize_escapes_html_in_summary_and_title():
    pa = _analysis(
        title="Breaking <b>news</b> & updates",
        summary="A report mentions <script>alert(1)</script> & other things.",
        key_entities=["<img onerror=alert(1)>"],
    )
    result = _sanitize(pa)
    assert "<" not in result.title and ">" not in result.title
    assert "&lt;" in result.title
    assert "<script>" not in result.summary
    assert "&lt;script&gt;" in result.summary
    assert "&amp;" in result.summary
    assert all("<" not in e and ">" not in e for e in result.key_entities)


def test_sanitize_escapes_image_description():
    pa = _analysis(image_description="A photo shows troops & vehicles moving near the border.")
    result = _sanitize(pa)
    assert "&amp;" in result.image_description


async def test_process_unanalysed_aborts_when_server_unreachable(db, app_config, monkeypatch):
    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord
    from datetime import datetime, timezone

    post = PostRecord(
        channel_id=1, channel_name="chan", message_id=1,
        timestamp=datetime(2026, 6, 7, tzinfo=timezone.utc),
        text="x" * 50, media_paths=[], has_images=False, raw_json="{}",
    )
    db.insert_post(post)

    analyzer = Analyzer(app_config, db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: False)

    called = []
    monkeypatch.setattr(analyzer, "analyze_post", lambda *a, **k: called.append(1))

    analysed_count, skipped_count = await analyzer.process_unanalysed()
    assert (analysed_count, skipped_count) == (0, 0)
    assert called == []
    # post must remain queued — no sentinel or garbage analysis written
    assert len(db.get_unanalysed_posts()) == 1


async def test_process_unanalysed_leaves_post_queued_on_analysis_failure(db, app_config, monkeypatch):
    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord
    from datetime import datetime, timezone

    post = PostRecord(
        channel_id=1, channel_name="chan", message_id=7,
        timestamp=datetime(2026, 6, 7, tzinfo=timezone.utc),
        text="x" * 50, media_paths=[], has_images=False, raw_json="{}",
    )
    db.insert_post(post)

    analyzer = Analyzer(app_config, db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)

    async def failing_analyze_post(post, channel_cfg=None):
        raise ConnectionError("LM Studio died mid-run")

    monkeypatch.setattr(analyzer, "analyze_post", failing_analyze_post)

    analysed_count, skipped_count = await analyzer.process_unanalysed()
    assert (analysed_count, skipped_count) == (0, 0)
    # the failed post stays unanalysed so the next run retries it
    assert len(db.get_unanalysed_posts()) == 1
