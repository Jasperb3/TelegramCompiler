import pytest
from pydantic import ValidationError

from tg_compiler.analyzer import (
    MAX_IMAGES_PER_POST,
    MAX_PROMPT_TEXT_CHARS,
    PostAnalysis,
    _check_numeric_consistency,
    _clean_image_insights,
    _sanitize,
    analysis_to_record,
    build_messages,
    compute_token_budget,
    parse_analysis_fallback,
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
    from tg_compiler.config import AppConfig, LMStudioConfig, TelegramConfig

    return AppConfig(
        telegram=TelegramConfig(api_id=1, api_hash="x", channels=[]),
        lmstudio=LMStudioConfig(model="test-model"),
    )


async def test_process_unanalysed_skips_short_textonly_post(db, app_config, monkeypatch):
    from datetime import datetime, timezone

    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord

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
    from datetime import datetime, timezone

    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord

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


def test_sanitize_escapes_markdown_link_syntax_in_summary():
    pa = _analysis(summary="See details [click here](https://evil.example) for more.")
    result = _sanitize(pa)
    assert "](" not in result.summary
    assert "&#91;click here&#93;" in result.summary


def test_sanitize_escapes_image_description():
    pa = _analysis(image_description="A photo shows troops & vehicles moving near the border.")
    result = _sanitize(pa)
    assert "&amp;" in result.image_description


async def test_process_unanalysed_aborts_when_server_unreachable(db, app_config, monkeypatch):
    from datetime import datetime, timezone

    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord

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


# --- dynamic per-post token budget ---


def _budget_post(text="", media_paths=None):
    from datetime import datetime, timezone

    from tg_compiler.db import PostRecord

    return PostRecord(
        channel_id=1, channel_name="chan", message_id=1,
        timestamp=datetime(2026, 6, 7, tzinfo=timezone.utc),
        text=text, media_paths=media_paths or [], has_images=bool(media_paths),
        raw_json="{}",
    )


@pytest.fixture
def lm_cfg():
    from tg_compiler.config import LMStudioConfig

    return LMStudioConfig(model="test-model")


def test_token_budget_empty_post_gets_base_budget(lm_cfg):
    assert compute_token_budget(_budget_post(text=""), lm_cfg) == lm_cfg.analysis_base_tokens


def test_token_budget_scales_with_text_length(lm_cfg):
    import math

    short = compute_token_budget(_budget_post(text="x" * 50), lm_cfg)
    long = compute_token_budget(_budget_post(text="x" * 2000), lm_cfg)
    assert short < long
    assert long == lm_cfg.analysis_base_tokens + math.ceil(2000 * lm_cfg.analysis_tokens_per_char)


def test_token_budget_counts_images(lm_cfg):
    no_img = compute_token_budget(_budget_post(text="x" * 100), lm_cfg)
    two_img = compute_token_budget(
        _budget_post(text="x" * 100, media_paths=["a.jpg", "b.jpg"]), lm_cfg
    )
    assert two_img == no_img + 2 * lm_cfg.analysis_tokens_per_image


def test_token_budget_clamped_at_max():
    from tg_compiler.config import LMStudioConfig

    cfg = LMStudioConfig(model="test-model", analysis_max_tokens=2000)
    budget = compute_token_budget(
        _budget_post(text="x" * 10000, media_paths=["a.jpg", "b.jpg", "c.jpg"]), cfg
    )
    assert budget == 2000


def test_token_budget_ignores_text_beyond_prompt_truncation(lm_cfg):
    # text past MAX_PROMPT_TEXT_CHARS is never sent to the LLM, so it must not
    # inflate the budget: a post at the limit and one far past it budget the same
    from tg_compiler.config import LMStudioConfig

    cfg = LMStudioConfig(model="test-model", analysis_max_tokens=100000)
    at_limit = compute_token_budget(_budget_post(text="x" * MAX_PROMPT_TEXT_CHARS), cfg)
    past_limit = compute_token_budget(_budget_post(text="x" * (MAX_PROMPT_TEXT_CHARS * 3)), cfg)
    assert at_limit == past_limit


def test_token_budget_images_capped_at_max_images_per_post(lm_cfg):
    at_cap = compute_token_budget(
        _budget_post(media_paths=[f"{i}.jpg" for i in range(MAX_IMAGES_PER_POST)]), lm_cfg
    )
    past_cap = compute_token_budget(
        _budget_post(media_paths=[f"{i}.jpg" for i in range(MAX_IMAGES_PER_POST + 3)]), lm_cfg
    )
    assert at_cap == past_cap


def _fake_openai_client(calls, parsed=None, content=""):
    """Minimal stand-in for the OpenAI client: records every parse/create call's
    kwargs into `calls` and returns a canned completion."""
    from types import SimpleNamespace

    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content=content))]
    )

    def parse(**kwargs):
        calls.append(("parse", kwargs))
        return completion

    def create(**kwargs):
        calls.append(("create", kwargs))
        return completion

    return SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


def test_call_llm_passes_budget_to_structured_call(db, app_config):
    from tg_compiler.analyzer import Analyzer

    analyzer = Analyzer(app_config, db)
    calls = []
    analyzer._client = _fake_openai_client(calls, parsed=_analysis())
    result = analyzer._call_llm([{"role": "user", "content": "hi"}], structured=True, max_tokens=1234)
    assert isinstance(result, PostAnalysis)
    assert calls == [("parse", calls[0][1])]
    assert calls[0][1]["max_tokens"] == 1234


def test_call_llm_passes_budget_to_fallback_call(db, app_config):
    from tg_compiler.analyzer import Analyzer

    analyzer = Analyzer(app_config, db)
    calls = []
    analyzer._client = _fake_openai_client(calls, content="not json")
    analyzer._call_llm([{"role": "user", "content": "hi"}], structured=False, max_tokens=2345)
    assert calls == [("create", calls[0][1])]
    assert calls[0][1]["max_tokens"] == 2345


async def test_analyze_post_uses_computed_budget(db, app_config):
    from tg_compiler.analyzer import Analyzer

    post = _budget_post(text="x" * 500, media_paths=["a.jpg"])
    analyzer = Analyzer(app_config, db)
    calls = []
    analyzer._client = _fake_openai_client(calls, parsed=_analysis())
    await analyzer.analyze_post(post)
    expected = compute_token_budget(post, app_config.lmstudio)
    assert calls[0][1]["max_tokens"] == expected


async def test_analyze_post_fallback_call_uses_same_budget(db, app_config, monkeypatch):
    import tg_compiler.analyzer as analyzer_mod
    from tg_compiler.analyzer import Analyzer

    monkeypatch.setattr(analyzer_mod, "RETRY_BACKOFF_BASE_SECS", 0)

    post = _budget_post(text="x" * 500)
    analyzer = Analyzer(app_config, db)
    calls = []
    client = _fake_openai_client(calls, content="not json")

    def failing_parse(**kwargs):
        calls.append(("parse", kwargs))
        raise RuntimeError("length limit reached")

    client.beta.chat.completions.parse = failing_parse
    analyzer._client = client

    await analyzer.analyze_post(post)
    expected = compute_token_budget(post, app_config.lmstudio)
    assert calls[-1][0] == "create"
    assert all(kwargs["max_tokens"] == expected for _, kwargs in calls)


async def test_process_unanalysed_leaves_post_queued_on_analysis_failure(db, app_config, monkeypatch):
    from datetime import datetime, timezone

    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord

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
