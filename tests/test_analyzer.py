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


def _touch_images(tmp_path, n, start=0):
    """n real image files, so _existing_media() sees them as still on disk."""
    out = []
    for i in range(start, start + n):
        f = tmp_path / f"img{i}.jpg"
        f.write_bytes(b"jpeg")
        out.append(str(f))
    return out


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


def test_numeric_consistency_ignores_hyphenated_designators():
    # "T-72" is a tank, not 72 of anything; against "80 killed" it is 11% apart.
    assert _check_numeric_consistency("Officials say 80 killed in the strike", "Image shows a destroyed T-72 tank") is True
    # "Kh-101" vs "95 drones" is 6.3% apart.
    assert _check_numeric_consistency("Russia launched 95 drones overnight", "The debris is from a Kh-101 cruise missile") is True


def test_numeric_consistency_ignores_thousands_separators():
    # "1,200" previously yielded a phantom 1 AND a 200, so the 200 matched but
    # the 1 collided with every small integer in the other text.
    assert _check_numeric_consistency("Around 1,200 troops were redeployed", "The image shows 3 armoured columns") is True


def test_numeric_consistency_thousands_separator_still_compared_as_one_number():
    assert _check_numeric_consistency("Around 1,200 troops were redeployed", "A caption reading 4,500 troops") is False


def test_numeric_consistency_aircraft_designator_false_positive():
    # From the live DB: 737 vs 524 and 522 vs 737 are two halves of the same
    # designator compared against each other across the two texts.
    summary = "A Boeing 737-524 aircraft from Caspian Airlines was observed returning to the tarmac in Tehran."
    image = "The image shows a flight tracking map for flight CP972, the aircraft type (Boeing 737-522) and registration number (EP1048)."
    assert _check_numeric_consistency(summary, image) is True


def test_numeric_consistency_notam_serial_false_positive():
    # From the live DB: the "26" of NOTAM serial A0821/26 against "June 14".
    summary = "Qatar established alternative flight paths through its airspace between June 7 and June 14."
    image = "The image shows a formal NOTAM notice (OTDF A0821/26) regarding alternate routes within the Doha FIR."
    assert _check_numeric_consistency(summary, image) is True


def test_numeric_consistency_ignores_brand_and_suffix_designators():
    # Reading "M8.4" as a quantity also exposed "Flightradar24" and the "M3" of
    # "Tu-22M3" — mixed letter/digit tokens that are names, not counts.
    assert _check_numeric_consistency(
        "An Airbus A300B4-60SR aircraft operated by Iran Air made 3 approaches",
        "A tracking map from Flightradar24 showing the path of an aircraft",
    ) is True
    assert _check_numeric_consistency(
        "A Tu-22M3 crashed in Irkutsk on the evening of June 15",
        "A video of the Russian Tu-22M3 heavy bomber nose diving into a field",
    ) is True


def test_numeric_consistency_ignores_ordinals():
    assert _check_numeric_consistency("Officials say 80 killed in the strike", "Insignia of the 72nd Mechanised Brigade is visible") is True


def test_numeric_consistency_unrelated_quantities_not_compared():
    # 8 dead and 3 vehicles are both small integers 2.7x apart, but they count
    # different things — the summary describes the event, the image describes
    # what the picture adds, so most number pairs across them are unrelated.
    assert _check_numeric_consistency("Officials say 8 killed in the blast", "The image shows 3 armoured vehicles") is True


def test_numeric_consistency_shared_noun_still_compared():
    assert _check_numeric_consistency("Officials say 8 killed in the blast", "The caption reads 3 killed") is False


def test_extract_numbers_reads_a_letter_prefixed_magnitude():
    from tg_compiler.analyzer import _extract_numbers

    assert [n.value for n in _extract_numbers("An M8.4 earthquake struck")] == [8.4]


def test_extract_numbers_does_not_split_a_plain_decimal():
    from tg_compiler.analyzer import _extract_numbers

    assert [n.value for n in _extract_numbers("A 7.8 magnitude quake")] == [7.8]


def test_sanitize_keeps_the_description_on_a_numeric_mismatch(caplog):
    """Measured over a live 500-post run, every numeric drop was a false
    positive; the rule's only true positive across 13,446 stored descriptions is
    one earthquake caption. It warns now, it does not destroy the field."""
    analysis = _analysis(summary="Officials say 12 killed in the blast.")
    analysis.image_description = "A caption reads 45 killed."

    with caplog.at_level("WARNING"):
        cleaned = _sanitize(analysis)

    assert "45 killed" in cleaned.image_description
    assert "Numeric mismatch" in caplog.text


def test_images_were_sent_is_false_for_a_video_post_and_a_purged_path(tmp_path):
    from datetime import datetime, timezone

    from tg_compiler.analyzer import _images_were_sent
    from tg_compiler.db import PostRecord

    def post(**kw):
        base = dict(channel_id=1, channel_name="chan", message_id=1,
                    timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
                    text="x" * 50, media_paths=[], has_images=False, raw_json="{}")
        return PostRecord(**{**base, **kw})

    assert _images_were_sent(post(has_video=True), 3) is False
    assert _images_were_sent(
        post(media_paths=[str(tmp_path / "gone.jpg")], has_images=True), 3
    ) is False

    present = tmp_path / "there.jpg"
    present.write_bytes(b"jpeg")
    assert _images_were_sent(post(media_paths=[str(present)], has_images=True), 3) is True


def test_images_were_sent_respects_the_image_cap(tmp_path):
    """The cap counts files that still exist: missing paths no longer consume a
    slot (they are filtered before the cap), but a real file past the cap is not
    sent."""
    from datetime import datetime, timezone

    from tg_compiler.analyzer import _existing_media, _images_were_sent
    from tg_compiler.db import PostRecord

    present = _touch_images(tmp_path, 4)

    def post(paths):
        return PostRecord(
            channel_id=1, channel_name="chan", message_id=1,
            timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc), text="x" * 50,
            media_paths=paths, has_images=True, raw_json="{}",
        )

    purged = [str(tmp_path / f"missing{i}.jpg") for i in range(3)]
    # Three purged paths ahead of a real one: the real image is still sent.
    assert _images_were_sent(post(purged + present[:1]), 3) is True
    assert _images_were_sent(post(purged), 3) is False
    # Four real images, cap 3: nothing past the cap is sent.
    assert len(_existing_media(post(present), 3)) == 3


def test_sanitize_discards_a_description_when_no_image_was_sent(caplog):
    """Videos are never downloaded, so a "the video shows..." description is
    invented from the post text and would render as an observed Image line."""
    analysis = _analysis(summary="An explosion was reported in Khan Yunis.")
    analysis.image_substantive = True
    analysis.image_description = "The image shows a large plume of smoke rising over Khan Yunis."

    with caplog.at_level("INFO"):
        cleaned = _sanitize(analysis, images_sent=False)

    assert cleaned.image_description is None
    assert cleaned.image_substantive is False
    assert "no image was sent" in caplog.text


def test_sanitize_keeps_the_description_when_an_image_was_sent():
    analysis = _analysis(summary="An explosion was reported in Khan Yunis.")
    analysis.image_description = "The image shows a large plume of smoke rising over Khan Yunis."

    assert _sanitize(analysis, images_sent=True).image_description is not None


async def test_analyze_post_discards_the_description_for_a_video_post(db, app_config, monkeypatch):
    from datetime import datetime, timezone

    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord

    post = PostRecord(
        channel_id=1, channel_name="chan", message_id=1,
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
        text="An explosion was reported in Khan Yunis this morning.",
        media_paths=[], has_images=False, has_video=True, raw_json="{}",
    )

    analyzer = Analyzer(app_config, db)
    fabricated = _analysis(summary="An explosion was reported in Khan Yunis.")
    fabricated.image_substantive = True
    fabricated.image_description = "The video shows a plume of smoke over Khan Yunis."
    monkeypatch.setattr(analyzer, "_call_llm", lambda *a, **k: fabricated)

    assert (await analyzer.analyze_post(post)).image_description is None


def test_sanitize_logs_the_rejected_image_text_and_the_substantive_flag(caplog):
    """A rejection has to be checkable: 'JSON artefact or garbage' fires on
    _ENTITY_GARBAGE, which was written for entity names, so the text it threw
    away is the only way to tell a real artefact from a false positive."""
    analysis = _analysis(summary="A summary long enough to be real.")
    analysis.image_substantive = True
    analysis.image_description = "A map of the LOW-lying floodplain near Kherson."

    with caplog.at_level("INFO"):
        cleaned = _sanitize(analysis)

    assert cleaned.image_description is None
    assert "LOW-lying floodplain" in caplog.text
    assert "image_substantive=True" in caplog.text


def test_numeric_consistency_shared_unit_alone_is_not_comparable():
    # Two route legs, matched only by "km" — 1,608 pairs with 1,600, but 409 is
    # a different segment entirely. From the 2026-09-05 drain.
    assert _check_numeric_consistency(
        "Ukraine to Tyumen in Russia (~1,600 km), illustrating logistical routes.",
        "Marked distances (~1,608 km and ~409 km segments) and colored zones.",
    ) is True
    # A percentage and a dollar amount, matched only by "USD".
    assert _check_numeric_consistency(
        "Brent crude rose, reflecting a rise of about $3.587 USD in a single day.",
        "Brent crude oil, indicating a 4.33% increase to $86.417 USD over the session.",
    ) is True


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


async def test_process_unanalysed_since_excludes_older_posts(db, app_config, monkeypatch):
    from datetime import datetime, timezone

    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord

    old_post = PostRecord(
        channel_id=1, channel_name="chan", message_id=1,
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
        text="x" * 50, media_paths=[], has_images=False, raw_json="{}",
    )
    new_post = PostRecord(
        channel_id=1, channel_name="chan", message_id=2,
        timestamp=datetime(2026, 6, 10, tzinfo=timezone.utc),
        text="x" * 50, media_paths=[], has_images=False, raw_json="{}",
    )
    db.insert_post(old_post)
    db.insert_post(new_post)

    analyzer = Analyzer(app_config, db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)

    async def fake_analyze_post(post, channel_cfg=None):
        return _analysis(summary="Real analysis output for a long post.")

    monkeypatch.setattr(analyzer, "analyze_post", fake_analyze_post)

    cutoff = datetime(2026, 6, 5, tzinfo=timezone.utc)
    analysed_count, skipped_count = await analyzer.process_unanalysed(since=cutoff)
    assert (analysed_count, skipped_count) == (1, 0)

    remaining = db.get_unanalysed_posts()
    assert [p.message_id for p in remaining] == [1]


async def test_process_unanalysed_since_logs_excluded_count(db, app_config, monkeypatch, caplog):
    from datetime import datetime, timezone

    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord

    old_post = PostRecord(
        channel_id=1, channel_name="chan", message_id=1,
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
        text="x" * 50, media_paths=[], has_images=False, raw_json="{}",
    )
    db.insert_post(old_post)

    analyzer = Analyzer(app_config, db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)
    monkeypatch.setattr(analyzer, "analyze_post", lambda *a, **k: None)

    cutoff = datetime(2026, 6, 5, tzinfo=timezone.utc)
    with caplog.at_level("INFO"):
        await analyzer.process_unanalysed(since=cutoff)

    assert "1 older unanalysed posts excluded" in caplog.text


async def test_process_unanalysed_fetches_the_queue_only_once(db, app_config, monkeypatch):
    """The exclusion count must come from a COUNT(*), not a second unbounded
    fetch — the daemon sweep runs this every DAEMON_ANALYSIS_INTERVAL_SECS."""
    from datetime import datetime, timezone

    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord

    db.insert_post(PostRecord(
        channel_id=1, channel_name="chan", message_id=1,
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
        text="x" * 50, media_paths=[], has_images=False, raw_json="{}",
    ))

    fetches = []
    real_fetch = db.get_unanalysed_posts

    def counting_fetch(*args, **kwargs):
        fetches.append((args, kwargs))
        return real_fetch(*args, **kwargs)

    monkeypatch.setattr(db, "get_unanalysed_posts", counting_fetch)

    analyzer = Analyzer(app_config, db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)
    monkeypatch.setattr(analyzer, "analyze_post", lambda *a, **k: None)

    await analyzer.process_unanalysed(since=datetime(2026, 6, 5, tzinfo=timezone.utc))

    assert len(fetches) == 1


async def test_process_unanalysed_tombstones_short_post_whose_media_is_purged(
    db, app_config, monkeypatch, tmp_path
):
    """has_images is a claim about scrape time; purge_old_media() may since have
    deleted the file, leaving a short post with nothing analysable in it."""
    from datetime import datetime, timezone

    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord

    db.insert_post(PostRecord(
        channel_id=1, channel_name="chan", message_id=1,
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
        text="short", media_paths=[str(tmp_path / "gone.jpg")],
        has_images=True, raw_json="{}",
    ))

    analyzer = Analyzer(app_config, db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)

    async def fail(*a, **k):
        raise AssertionError("a post with no usable content must not reach the LLM")

    monkeypatch.setattr(analyzer, "analyze_post", fail)

    analysed_count, skipped_count = await analyzer.process_unanalysed()

    assert (analysed_count, skipped_count) == (0, 1)
    assert db.get_unanalysed_posts() == []


async def test_process_unanalysed_analyses_short_post_whose_media_is_present(
    db, app_config, monkeypatch, tmp_path
):
    from datetime import datetime, timezone

    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord

    image = tmp_path / "present.jpg"
    image.write_bytes(b"jpeg")
    db.insert_post(PostRecord(
        channel_id=1, channel_name="chan", message_id=1,
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
        text="short", media_paths=[str(image)], has_images=True, raw_json="{}",
    ))

    analyzer = Analyzer(app_config, db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)

    async def fake_analyze_post(post, channel_cfg=None):
        return _analysis(summary="Real analysis output for a long post.")

    monkeypatch.setattr(analyzer, "analyze_post", fake_analyze_post)

    assert await analyzer.process_unanalysed() == (1, 0)


async def test_process_unanalysed_analyses_short_video_post_with_no_files(
    db, app_config, monkeypatch
):
    """Videos are never downloaded, so has_video alone still means usable media."""
    from datetime import datetime, timezone

    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord

    db.insert_post(PostRecord(
        channel_id=1, channel_name="chan", message_id=1,
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
        text="short", media_paths=[], has_images=False, has_video=True, raw_json="{}",
    ))

    analyzer = Analyzer(app_config, db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)

    async def fake_analyze_post(post, channel_cfg=None):
        return _analysis(summary="Real analysis output for a long post.")

    monkeypatch.setattr(analyzer, "analyze_post", fake_analyze_post)

    assert await analyzer.process_unanalysed() == (1, 0)


async def test_process_unanalysed_limit_bounds_the_queue_oldest_first(
    db, app_config, monkeypatch
):
    from datetime import datetime, timezone

    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord

    for mid, day in ((1, 10), (2, 1)):
        db.insert_post(PostRecord(
            channel_id=1, channel_name="chan", message_id=mid,
            timestamp=datetime(2026, 6, day, tzinfo=timezone.utc),
            text="x" * 50, media_paths=[], has_images=False, raw_json="{}",
        ))

    analyzer = Analyzer(app_config, db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)

    seen = []

    async def fake_analyze_post(post, channel_cfg=None):
        seen.append(post.message_id)
        return _analysis(summary="Real analysis output for a long post.")

    monkeypatch.setattr(analyzer, "analyze_post", fake_analyze_post)

    assert await analyzer.process_unanalysed(limit=1) == (1, 0)
    assert seen == [2]


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


def test_token_budget_counts_images(lm_cfg, tmp_path):
    # Real files: the budget counts the images actually sent, and _existing_media()
    # drops paths that are no longer on disk.
    imgs = _touch_images(tmp_path, 2)
    no_img = compute_token_budget(_budget_post(text="x" * 100), lm_cfg)
    two_img = compute_token_budget(
        _budget_post(text="x" * 100, media_paths=imgs), lm_cfg
    )
    assert two_img == no_img + 2 * lm_cfg.analysis_tokens_per_image


def test_token_budget_clamped_at_max(tmp_path):
    from tg_compiler.config import LMStudioConfig

    cfg = LMStudioConfig(model="test-model", analysis_max_tokens=2000)
    budget = compute_token_budget(
        _budget_post(text="x" * 10000, media_paths=_touch_images(tmp_path, 3)), cfg
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


# --------------------------------------------------------------------------
# Batched analysis
# --------------------------------------------------------------------------


def _batch_config(**lm_overrides):
    from tg_compiler.config import AppConfig, LMStudioConfig, TelegramConfig

    lm = dict(model="test-model")
    lm.update(lm_overrides)
    return AppConfig(
        telegram=TelegramConfig(api_id=1, api_hash="x", channels=[]),
        lmstudio=LMStudioConfig(**lm),
    )


def _batch_post(message_id, text="A post with enough text to clear the content gate.",
                media_paths=None, channel_id=1):
    from datetime import datetime, timezone

    from tg_compiler.db import PostRecord

    return PostRecord(
        channel_id=channel_id, channel_name="chan", message_id=message_id,
        timestamp=datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc),
        text=text, media_paths=media_paths or [], has_images=bool(media_paths),
        raw_json="{}",
    )


_DEFAULT_POST_TEXT = "A post with enough text to clear the content gate."


def _batch_item(index, **overrides):
    from tg_compiler.analyzer import BatchPostAnalysis

    base = dict(
        index=index,
        opening=" ".join(_DEFAULT_POST_TEXT.split()[:6]),
        title=f"Title {index}",
        summary=f"Summary for post {index}.",
        importance_score=3, urgency_score=3, credibility_score=3, relevance_score=3,
        category="Analysis", key_entities=[], reasoning="",
    )
    base.update(overrides)
    return BatchPostAnalysis.model_validate(base)


def test_build_batch_messages_numbers_posts_and_interleaves_images(monkeypatch, tmp_path):
    from tg_compiler.analyzer import SYSTEM_PROMPT, build_batch_messages

    monkeypatch.setattr("tg_compiler.analyzer._encode_image", lambda p: f"B64<{p}>")
    cfg = _batch_config(batch_images_per_post=1).lmstudio
    posts = [
        _batch_post(1, text="first post"),
        _batch_post(2, text="second post", media_paths=_touch_images(tmp_path, 2)),
    ]

    messages = build_batch_messages(posts, SYSTEM_PROMPT, cfg)

    assert messages[0]["role"] == "system"
    assert "analyses" in messages[0]["content"]
    assert SYSTEM_PROMPT in messages[0]["content"]

    parts = messages[1]["content"]
    assert "### POST 1" in parts[1]["text"]
    assert "first post" in parts[1]["text"]
    assert "### POST 2" in parts[2]["text"]
    # only one image per post, and it follows its own post's text part
    assert parts[3]["type"] == "image_url"
    assert "B64<" in parts[3]["image_url"]["url"]
    assert "img0.jpg>" in parts[3]["image_url"]["url"]
    assert len(parts) == 4


def test_build_batch_messages_truncates_long_text():
    from tg_compiler.analyzer import SYSTEM_PROMPT, build_batch_messages

    cfg = _batch_config().lmstudio
    post = _batch_post(1, text="y" * (MAX_PROMPT_TEXT_CHARS + 500))
    parts = build_batch_messages([post], SYSTEM_PROMPT, cfg)[1]["content"]
    assert parts[1]["text"].count("y") == MAX_PROMPT_TEXT_CHARS


def test_plan_batches_separates_media_posts_and_honours_sizes():
    from tg_compiler.analyzer import plan_batches

    cfg = _batch_config(batch_size=3, batch_size_with_images=2).lmstudio
    text_posts = [_batch_post(i) for i in range(1, 6)]
    media_posts = [_batch_post(i, media_paths=["x.jpg"]) for i in range(6, 9)]

    batches = plan_batches(text_posts + media_posts, cfg)

    sizes = [len(b) for b in batches]
    assert sizes == [3, 2, 2, 1]
    # no batch mixes media and text-only posts
    for batch in batches:
        assert len({bool(p.media_paths) for p in batch}) == 1


def test_plan_batches_size_one_yields_single_post_batches():
    from tg_compiler.analyzer import plan_batches

    cfg = _batch_config().lmstudio  # batch_size defaults to 1
    posts = [_batch_post(i) for i in range(1, 4)]
    assert [len(b) for b in plan_batches(posts, cfg)] == [1, 1, 1]


def test_plan_batches_splits_batch_exceeding_prompt_char_cap():
    from tg_compiler.analyzer import plan_batches

    cfg = _batch_config(batch_size=10, batch_max_prompt_chars=1000).lmstudio
    posts = [_batch_post(i, text="z" * 400) for i in range(1, 6)]
    assert [len(b) for b in plan_batches(posts, cfg)] == [2, 2, 1]


def test_batch_token_budget_scales_with_posts_and_clamps():
    from tg_compiler.analyzer import compute_batch_token_budget

    cfg = _batch_config(
        batch_base_tokens=1000, batch_tokens_per_post=100,
        batch_tokens_per_char=0.0, batch_tokens_per_image=50,
        batch_max_tokens=1250,
    ).lmstudio
    assert compute_batch_token_budget([_batch_post(1)], cfg) == 1100
    assert compute_batch_token_budget([_batch_post(i) for i in (1, 2)], cfg) == 1200
    # clamped at batch_max_tokens
    assert compute_batch_token_budget([_batch_post(i) for i in range(10)], cfg) == 1250


def test_batch_token_budget_counts_images_up_to_the_per_post_cap(tmp_path):
    from tg_compiler.analyzer import compute_batch_token_budget

    cfg = _batch_config(
        batch_base_tokens=1000, batch_tokens_per_post=0,
        batch_tokens_per_char=0.0, batch_tokens_per_image=50,
        batch_images_per_post=2, batch_max_tokens=99999,
    ).lmstudio
    post = _batch_post(1, media_paths=_touch_images(tmp_path, 3))
    assert compute_batch_token_budget([post], cfg) == 1100


def test_map_batch_results_maps_by_index_not_position():
    from tg_compiler.analyzer import BatchAnalysis, map_batch_results

    batch = BatchAnalysis(analyses=[_batch_item(3), _batch_item(1), _batch_item(2)])
    mapped = map_batch_results(batch, [_batch_post(i) for i in range(1, 4)])
    assert set(mapped) == {0, 1, 2}
    assert mapped[0].summary == "Summary for post 1."
    assert mapped[2].summary == "Summary for post 3."


def test_map_batch_results_drops_out_of_range_index():
    from tg_compiler.analyzer import BatchAnalysis, map_batch_results

    batch = BatchAnalysis(analyses=[_batch_item(1), _batch_item(9), _batch_item(0)])
    mapped = map_batch_results(batch, [_batch_post(i) for i in range(1, 3)])
    assert set(mapped) == {0}


def test_map_batch_results_keeps_first_of_duplicate_indices():
    from tg_compiler.analyzer import BatchAnalysis, map_batch_results

    batch = BatchAnalysis(analyses=[
        _batch_item(1, summary="First one wins here."),
        _batch_item(1, summary="Second one is discarded."),
    ])
    mapped = map_batch_results(batch, [_batch_post(i) for i in range(1, 3)])
    assert mapped[0].summary == "First one wins here."


def test_map_batch_results_omits_positions_the_model_did_not_return():
    from tg_compiler.analyzer import BatchAnalysis, map_batch_results

    mapped = map_batch_results(BatchAnalysis(analyses=[_batch_item(2)]), [_batch_post(i) for i in range(1, 4)])
    assert set(mapped) == {1}


def test_map_batch_results_sanitizes_every_item():
    from tg_compiler.analyzer import BatchAnalysis, map_batch_results

    batch = BatchAnalysis(analyses=[
        _batch_item(1, summary="Tanks <b>rolled</b> into the city & held it."),
        _batch_item(2, summary="The user provided no content for analysis."),
    ])
    mapped = map_batch_results(batch, [_batch_post(i) for i in range(1, 3)])
    assert mapped[0].summary == "Tanks &lt;b&gt;rolled&lt;/b&gt; into the city &amp; held it."
    assert mapped[1].summary == ""


def test_salvage_batch_items_recovers_prefix_of_truncated_response():
    from tg_compiler.analyzer import salvage_batch_items

    complete = (
        '{"index": 1, "opening": "A post with enough text", "title": "T1", '
        '"summary": "S1", "importance_score": 3, "urgency_score": 3, '
        '"credibility_score": 3, "relevance_score": 3, '
        '"category": "Analysis", "key_entities": []}'
    )
    raw = '{"analyses": [' + complete + ', {"index": 2, "title": "T2", "sum'
    salvaged = salvage_batch_items(raw)
    assert [i.index for i in salvaged.analyses] == [1]
    assert salvaged.analyses[0].summary == "S1"


def test_salvage_batch_items_returns_empty_when_key_absent():
    from tg_compiler.analyzer import salvage_batch_items

    assert salvage_batch_items("total garbage, no json here").analyses == []


def _fake_batch_client(calls, parsed=None, content="", finish_reason="stop"):
    from types import SimpleNamespace

    completion = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(parsed=parsed, content=content),
        finish_reason=finish_reason,
    )])

    def parse(**kwargs):
        calls.append(kwargs)
        return completion

    return SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    )


async def test_analyze_batch_returns_mapping_and_uses_batch_budget(db):
    from tg_compiler.analyzer import Analyzer, BatchAnalysis

    config = _batch_config(
        batch_size=2, batch_base_tokens=1000, batch_tokens_per_post=100,
        batch_tokens_per_char=0.0, batch_tokens_per_image=0, batch_max_tokens=99999,
    )
    analyzer = Analyzer(config, db)
    calls = []
    analyzer._client = _fake_batch_client(
        calls, parsed=BatchAnalysis(analyses=[_batch_item(1), _batch_item(2)])
    )

    results = await analyzer.analyze_batch([_batch_post(1), _batch_post(2)])
    assert set(results) == {0, 1}
    assert calls[0]["max_tokens"] == 1200


async def test_analyze_batch_salvages_truncated_response(db):
    from tg_compiler.analyzer import Analyzer

    raw = (
        '{"analyses": [{"index": 1, "opening": "A post with enough text", '
        '"title": "T1", "summary": "S1", "importance_score": 3, '
        '"urgency_score": 3, "credibility_score": 3, "relevance_score": 3, '
        '"category": "Analysis", "key_entities": []}, '
        '{"index": 2, "title": "T2"'
    )
    analyzer = Analyzer(_batch_config(batch_size=2), db)
    analyzer._client = _fake_batch_client([], content=raw, finish_reason="length")

    results = await analyzer.analyze_batch([_batch_post(1), _batch_post(2)])
    assert set(results) == {0}


async def test_analyze_batch_returns_empty_when_call_fails(db, monkeypatch):
    from tg_compiler import analyzer as analyzer_mod
    from tg_compiler.analyzer import Analyzer

    monkeypatch.setattr(analyzer_mod, "RETRY_BACKOFF_BASE_SECS", 0)
    analyzer = Analyzer(_batch_config(batch_size=2), db)

    def boom(*a, **kw):
        raise RuntimeError("server exploded")

    monkeypatch.setattr(analyzer, "_call_batch_llm", boom)
    assert await analyzer.analyze_batch([_batch_post(1), _batch_post(2)]) == {}


async def test_process_unanalysed_batches_posts_into_one_call(db, monkeypatch):
    from tg_compiler.analyzer import Analyzer

    for i in range(1, 5):
        db.insert_post(_batch_post(i))

    analyzer = Analyzer(_batch_config(batch_size=4), db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)
    batches = []

    async def fake_analyze_batch(posts):
        batches.append(list(posts))
        return {i: _analysis(summary=f"Batched summary {i}.") for i in range(len(posts))}

    monkeypatch.setattr(analyzer, "analyze_batch", fake_analyze_batch)

    analysed, skipped = await analyzer.process_unanalysed()
    assert (analysed, skipped) == (4, 0)
    assert [len(b) for b in batches] == [4]
    assert db.get_unanalysed_posts() == []


async def test_process_unanalysed_batch_size_one_uses_single_post_path(db, monkeypatch):
    from tg_compiler.analyzer import Analyzer

    db.insert_post(_batch_post(1))
    analyzer = Analyzer(_batch_config(), db)  # batch_size defaults to 1
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)

    async def fake_analyze_batch(posts):  # pragma: no cover - must never run
        raise AssertionError("batch path used despite batch_size == 1")

    seen = []

    async def fake_analyze_post(post, channel_cfg=None):
        seen.append(post.message_id)
        return _analysis()

    monkeypatch.setattr(analyzer, "analyze_batch", fake_analyze_batch)
    monkeypatch.setattr(analyzer, "analyze_post", fake_analyze_post)

    assert await analyzer.process_unanalysed() == (1, 0)
    assert seen == [1]


async def test_process_unanalysed_low_yield_batch_falls_back_to_single_calls(db, monkeypatch):
    from tg_compiler.analyzer import Analyzer

    for i in range(1, 5):
        db.insert_post(_batch_post(i))

    analyzer = Analyzer(_batch_config(batch_size=4, batch_min_yield_ratio=0.6), db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)

    async def fake_analyze_batch(posts):
        return {0: _analysis(summary="Only the first came back.")}

    retried = []

    async def fake_analyze_post(post, channel_cfg=None):
        retried.append(post.message_id)
        return _analysis(summary="Recovered by the single-post path.")

    monkeypatch.setattr(analyzer, "analyze_batch", fake_analyze_batch)
    monkeypatch.setattr(analyzer, "analyze_post", fake_analyze_post)

    analysed, _ = await analyzer.process_unanalysed()
    assert sorted(retried) == [2, 3, 4]
    assert analysed == 4
    assert db.get_unanalysed_posts() == []


async def test_process_unanalysed_high_yield_batch_leaves_stragglers_queued(db, monkeypatch):
    from tg_compiler.analyzer import Analyzer

    for i in range(1, 5):
        db.insert_post(_batch_post(i))

    analyzer = Analyzer(_batch_config(batch_size=4, batch_min_yield_ratio=0.6), db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)

    async def fake_analyze_batch(posts):
        return {i: _analysis() for i in range(3)}  # post 4 missing

    async def fake_analyze_post(post, channel_cfg=None):  # pragma: no cover
        raise AssertionError("single-post fallback ran on a high-yield batch")

    monkeypatch.setattr(analyzer, "analyze_batch", fake_analyze_batch)
    monkeypatch.setattr(analyzer, "analyze_post", fake_analyze_post)

    analysed, _ = await analyzer.process_unanalysed()
    assert analysed == 3
    assert [p.message_id for p in db.get_unanalysed_posts()] == [4]


async def test_process_unanalysed_skips_short_posts_before_batching(db, monkeypatch):
    from tg_compiler.analyzer import Analyzer

    db.insert_post(_batch_post(1, text="ok"))          # below MIN_CONTENT_CHARS
    db.insert_post(_batch_post(2))
    db.insert_post(_batch_post(3))

    analyzer = Analyzer(_batch_config(batch_size=4), db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)
    batched = []

    async def fake_analyze_batch(posts):
        batched.extend(p.message_id for p in posts)
        return {i: _analysis() for i in range(len(posts))}

    monkeypatch.setattr(analyzer, "analyze_batch", fake_analyze_batch)

    analysed, skipped = await analyzer.process_unanalysed()
    assert (analysed, skipped) == (2, 1)
    assert batched == [2, 3]
    pairs = db.get_days_posts_with_analyses("2026-06-07")
    assert {p.message_id: a.category for p, a in pairs}[1] == "Skipped"


async def test_process_unanalysed_keeps_custom_prompt_posts_out_of_batches(db, monkeypatch):
    from tg_compiler.analyzer import Analyzer
    from tg_compiler.config import ChannelConfig

    db.insert_post(_batch_post(1, channel_id=1))
    db.insert_post(_batch_post(2, channel_id=2))
    db.insert_post(_batch_post(3, channel_id=2))

    analyzer = Analyzer(_batch_config(batch_size=4), db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)
    channel_map = {
        1: ChannelConfig(slug="special", custom_prompt="Be different"),
        2: ChannelConfig(slug="normal"),
    }
    batched, singled = [], []

    async def fake_analyze_batch(posts):
        batched.extend(p.message_id for p in posts)
        return {i: _analysis() for i in range(len(posts))}

    async def fake_analyze_post(post, channel_cfg=None):
        singled.append((post.message_id, channel_cfg.slug if channel_cfg else None))
        return _analysis()

    monkeypatch.setattr(analyzer, "analyze_batch", fake_analyze_batch)
    monkeypatch.setattr(analyzer, "analyze_post", fake_analyze_post)

    analysed, _ = await analyzer.process_unanalysed(channel_map)
    assert analysed == 3
    assert batched == [2, 3]
    assert singled == [(1, "special")]


async def test_process_unanalysed_since_filter_applies_under_batching(db, monkeypatch):
    from datetime import datetime, timezone

    from tg_compiler.analyzer import Analyzer
    from tg_compiler.db import PostRecord

    old = PostRecord(
        channel_id=1, channel_name="chan", message_id=1,
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
        text="An older post with plenty of text in it.", media_paths=[],
        has_images=False, raw_json="{}",
    )
    db.insert_post(old)
    db.insert_post(_batch_post(2))
    db.insert_post(_batch_post(3))

    analyzer = Analyzer(_batch_config(batch_size=4), db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)
    batched = []

    async def fake_analyze_batch(posts):
        batched.extend(p.message_id for p in posts)
        return {i: _analysis() for i in range(len(posts))}

    monkeypatch.setattr(analyzer, "analyze_batch", fake_analyze_batch)

    analysed, _ = await analyzer.process_unanalysed(
        since=datetime(2026, 6, 5, tzinfo=timezone.utc)
    )
    assert analysed == 2
    assert batched == [2, 3]
    assert [p.message_id for p in db.get_unanalysed_posts()] == [1]


# --------------------------------------------------------------------------
# Per-stage models
# --------------------------------------------------------------------------


def test_call_llm_addresses_the_configured_analysis_model(db):
    from tg_compiler.analyzer import Analyzer

    config = _batch_config(analysis_model="small-model", synthesis_model="reasoning-model")
    analyzer = Analyzer(config, db)
    calls = []
    analyzer._client = _fake_openai_client(calls, parsed=_analysis())

    analyzer._call_llm([{"role": "user", "content": "hi"}], structured=True, max_tokens=100)
    assert calls[0][1]["model"] == "small-model"


async def test_batch_call_addresses_the_configured_analysis_model(db):
    from tg_compiler.analyzer import Analyzer, BatchAnalysis

    config = _batch_config(batch_size=2, analysis_model="small-model")
    analyzer = Analyzer(config, db)
    calls = []
    analyzer._client = _fake_batch_client(
        calls, parsed=BatchAnalysis(analyses=[_batch_item(1), _batch_item(2)])
    )

    await analyzer.analyze_batch([_batch_post(1), _batch_post(2)])
    assert calls[0]["model"] == "small-model"


async def test_process_unanalysed_makes_the_analysis_model_resident(db, monkeypatch):
    from tg_compiler import analyzer as analyzer_mod
    from tg_compiler.analyzer import Analyzer

    db.insert_post(_batch_post(1))
    ensured = []

    class FakeManager:
        def __init__(self, cfg):
            self.cfg = cfg

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def ensure(self, model_key):
            ensured.append(model_key)

    monkeypatch.setattr(analyzer_mod, "ModelManager", FakeManager)

    analyzer = Analyzer(_batch_config(analysis_model="small-model"), db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)

    async def fake_analyze_post(post, channel_cfg=None):
        return _analysis()

    monkeypatch.setattr(analyzer, "analyze_post", fake_analyze_post)

    await analyzer.process_unanalysed()
    assert ensured == ["small-model"]


async def test_analysis_row_records_the_analysis_model(db, monkeypatch):
    from tg_compiler.analyzer import Analyzer

    db.insert_post(_batch_post(1))
    analyzer = Analyzer(_batch_config(analysis_model="small-model"), db)
    monkeypatch.setattr(analyzer, "_server_reachable", lambda: True)

    async def fake_analyze_post(post, channel_cfg=None):
        return _analysis()

    monkeypatch.setattr(analyzer, "analyze_post", fake_analyze_post)
    await analyzer.process_unanalysed()

    pairs = db.get_days_posts_with_analyses("2026-06-07")
    assert pairs[0][1].model_used == "small-model"
async def test_analyze_batch_salvages_when_parse_raises_on_length_limit(db):
    """A response cut off at max_tokens reaches us as an exception, not a value.

    client.beta.chat.completions.parse() raises LengthFinishReasonError rather than
    returning the truncated completion, so without unwrapping it the whole batch's
    work — thousands of generated tokens — is discarded.
    """
    from types import SimpleNamespace

    from openai import LengthFinishReasonError

    from tg_compiler.analyzer import Analyzer

    raw = (
        '{"analyses": [{"index": 1, "opening": "A post with enough text", '
        '"title": "T1", "summary": "S1", "importance_score": 3, '
        '"urgency_score": 3, "credibility_score": 3, "relevance_score": 3, '
        '"category": "Analysis", "key_entities": []}, '
        '{"index": 2, "title": "T2", "summ'
    )
    # The real LengthFinishReasonError carries a plain ChatCompletion whose message
    # has no `parsed` attribute at all, so the fake must not define one either —
    # a fake with parsed=None hides an AttributeError in the salvage path.
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    truncated = ChatCompletion(
        id="c1", object="chat.completion", created=0, model="test-model",
        choices=[Choice(
            index=0, finish_reason="length",
            message=ChatCompletionMessage(role="assistant", content=raw),
        )],
    )
    assert not hasattr(truncated.choices[0].message, "parsed")

    def parse(**_kwargs):
        raise LengthFinishReasonError(completion=truncated)

    analyzer = Analyzer(_batch_config(batch_size=2), db)
    analyzer._client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    )

    results = await analyzer.analyze_batch([_batch_post(1), _batch_post(2)])
    assert set(results) == {0}
    assert results[0].summary == "S1"


def test_map_batch_results_rejects_a_renumbered_batch():
    """Observed on google/gemma-3-4b: it dropped one post from a batch of 10 and
    renumbered the rest densely 1..9, so every index was unique and in range while
    each analysis described the *next* post. Indices alone cannot detect that."""
    from tg_compiler.analyzer import BatchAnalysis, map_batch_results

    posts = [
        _batch_post(1, text="Trump says the United States will resume bombing Iran tonight."),
        _batch_post(2, text="Zelensky names Belarusian factories supplying Russian weapons."),
        _batch_post(3, text="Prince Sultan Airbase in Al-Kharj is under ballistic missile attack."),
    ]
    # Each item echoes the opening of the post *after* the one it claims.
    shifted = BatchAnalysis(analyses=[
        _batch_item(1, opening="Zelensky names Belarusian factories supplying Russian"),
        _batch_item(2, opening="Prince Sultan Airbase in Al-Kharj is"),
    ])

    assert map_batch_results(shifted, posts) == {}


def test_map_batch_results_accepts_correctly_anchored_items():
    from tg_compiler.analyzer import BatchAnalysis, map_batch_results

    posts = [
        _batch_post(1, text="Trump says the United States will resume bombing Iran tonight."),
        _batch_post(2, text="Zelensky names Belarusian factories supplying Russian weapons."),
    ]
    aligned = BatchAnalysis(analyses=[
        _batch_item(1, opening="Trump says the United States will"),
        _batch_item(2, opening="Zelensky names Belarusian factories supplying Russian"),
    ])

    assert set(map_batch_results(aligned, posts)) == {0, 1}


def test_map_batch_results_drops_an_item_with_no_opening():
    from tg_compiler.analyzer import BatchAnalysis, map_batch_results

    posts = [_batch_post(1, text="Trump says the United States will resume bombing Iran.")]
    assert map_batch_results(BatchAnalysis(analyses=[_batch_item(1, opening="")]), posts) == {}


def test_opening_match_tolerates_minor_paraphrase_and_punctuation():
    from tg_compiler.analyzer import _opening_matches

    post = _batch_post(1, text="**BREAKING:** Prince Sultan Airbase, Al-Kharj, is under attack.")
    # punctuation and markdown stripped, most words still present
    assert _opening_matches("BREAKING Prince Sultan Airbase Al-Kharj is", post) is True
    assert _opening_matches("Zelensky names Belarusian factories supplying weapons", post) is False


def test_check_opening_distinguishes_mismatch_from_absent():
    """A missing anchor means 'could not verify'; a wrong anchor means the model
    attributed this analysis to another post. Conflating them would report a model
    that ignores the field as one that corrupts data."""
    from tg_compiler.analyzer import check_opening

    post = _batch_post(1, text="Trump says the United States will resume bombing Iran tonight.")
    assert check_opening("Trump says the United States will", post) == "match"
    assert check_opening("Zelensky names Belarusian factories supplying weapons", post) == "mismatch"
    assert check_opening("", post) == "absent"
    assert check_opening("   ", post) == "absent"


def test_map_batch_results_drops_unverifiable_items_without_calling_them_mismatches(caplog):
    import logging

    from tg_compiler.analyzer import BatchAnalysis, map_batch_results

    posts = [_batch_post(1, text="Trump says the United States will resume bombing Iran.")]
    with caplog.at_level(logging.WARNING):
        assert map_batch_results(BatchAnalysis(analyses=[_batch_item(1, opening="")]), posts) == {}
    assert "could not be verified" in caplog.text
    assert "renumbered" not in caplog.text


def test_build_batch_messages_labels_the_body_so_the_anchor_is_unambiguous():
    """The anchor asks for the post's first six words. Observed on
    mistralai/ministral-3-3b: with an unlabelled body it echoed "Channel
    GeoPWatch" — the first words of the *block* — which the matcher then read as
    misattribution. The body must be labelled and the instruction must say so."""
    from tg_compiler.analyzer import SYSTEM_PROMPT, build_batch_messages

    cfg = _batch_config().lmstudio
    post = _batch_post(1, text="China extends the commercial truce agreed last October.")
    messages = build_batch_messages([post], SYSTEM_PROMPT, cfg)

    block = messages[1]["content"][1]["text"]
    assert "Text: China extends the commercial truce" in block
    assert "Text:" in messages[0]["content"]


def test_check_opening_cannot_verify_a_post_with_almost_no_text():
    """Observed on an image post whose whole body was "Estonian Leviathan": the
    substance was in the picture, so the model echoed text it read from the image.
    The analysis was correct; a text anchor simply cannot confirm it, and calling
    that a mismatch would assert corruption on no evidence."""
    from tg_compiler.analyzer import check_opening

    caption_only = _batch_post(1, text="Estonian Leviathan")
    assert check_opening("Russia hates Estonia more than any", caption_only) == "absent"

    # a post with a real body is still checked strictly
    full = _batch_post(2, text="Russia says Ukrainian strikes hit grain infrastructure in Novorossiysk.")
    assert check_opening("Belgium blocks frozen Russian assets today", full) == "mismatch"


def test_opening_words_ignores_channel_promo_boilerplate():
    """A DDGeopolitics image post's whole content was "Estonian Leviathan"; the
    other sixteen tokens were the channel's @handle and Socials/Donate/Advertising
    links. Counting those as body text made the post look verifiable when it was
    not, and produced a false misattribution flag."""
    from tg_compiler.analyzer import _opening_words, check_opening

    text = (
        "🇪🇪🇷🇺🤣 Estonian Leviathan\n\n🔴@DDGeopolitics | "
        "[Socials](https://telegra.ph/Explore-DD-Geopolitics) | "
        "[Donate](https://ko-fi.com/ddgeo)"
    )
    assert _opening_words(text, 18) == ["estonian", "leviathan"]
    assert check_opening("Russia hates Estonia more than any", _batch_post(1, text=text)) == "absent"


@pytest.mark.parametrize("description", [
    "The image is a screenshot of the tweet, but no additional substantive information beyond the text is provided.",
    "The image shows a screenshot next to a profile picture, confirming attribution but no new substantive detail.",
    "A photo that adds nothing to the reported text.",
    "The picture provides no further context.",
])
def test_clean_image_insights_rejects_descriptions_that_say_there_is_nothing_extra(description):
    """mistralai/ministral-3-3b answers the image_substantive question in prose
    rather than only via the flag. Rendered, these produce an "Image" line in the
    briefing that tells the reader nothing."""
    assert _clean_image_insights(description) is None


@pytest.mark.parametrize("description", [
    "The image shows live demolition explosions destroying buildings in Bint Jbeil.",
    "The map depicts warmer temperatures over northern latitudes and wetter zones on the west coast.",
    "Protest signs read 'Make Russia Pay for Their Terrorism' and 'Euroclear We Are Watching You'.",
])
def test_clean_image_insights_keeps_substantive_descriptions(description):
    assert _clean_image_insights(description) == description


def test_batch_schema_requires_the_opening_anchor():
    """A pydantic default would keep `opening` out of the JSON schema's "required"
    list, so LM Studio's grammar-constrained decoding would never force the model
    to emit it. Observed live: ministral-3-3b then omitted it on ~12% of items,
    each dropped as unverifiable and retried on the slow per-post path."""
    from tg_compiler.analyzer import BatchPostAnalysis

    required = BatchPostAnalysis.model_json_schema()["required"]
    assert "opening" in required
    assert "index" in required


def _media_post(tmp_path, paths, **kw):
    from datetime import datetime, timezone

    from tg_compiler.db import PostRecord

    base = dict(channel_id=1, channel_name="chan", message_id=1,
                timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
                text="x" * 50, media_paths=paths, has_images=bool(paths),
                has_video=False, raw_json="{}")
    return PostRecord(**{**base, **kw})


def test_build_messages_skips_a_purged_path_and_still_fills_the_cap(tmp_path, caplog):
    import logging

    from tg_compiler.analyzer import MAX_IMAGES_PER_POST, build_messages

    present = _touch_images(tmp_path, MAX_IMAGES_PER_POST)
    post = _media_post(tmp_path, [str(tmp_path / "gone.jpg"), *present])

    with caplog.at_level(logging.WARNING):
        msgs = build_messages(post, "sys")

    images = [p for p in msgs[1]["content"] if p["type"] == "image_url"]
    assert len(images) == MAX_IMAGES_PER_POST
    assert "Could not read image" not in caplog.text


def test_build_batch_messages_skips_purged_paths(tmp_path, caplog):
    import logging

    from tg_compiler.analyzer import build_batch_messages
    from tg_compiler.config import LMStudioConfig

    cfg = LMStudioConfig(model="m", batch_images_per_post=2)
    post = _media_post(tmp_path, [str(tmp_path / "gone.jpg"), *_touch_images(tmp_path, 1)])

    with caplog.at_level(logging.WARNING):
        msgs = build_batch_messages([post], "sys", cfg)

    images = [p for p in msgs[1]["content"] if p["type"] == "image_url"]
    assert len(images) == 1
    assert "Could not read image" not in caplog.text


def test_token_budgets_count_only_images_that_still_exist(tmp_path):
    from tg_compiler.analyzer import compute_batch_token_budget, compute_token_budget
    from tg_compiler.config import LMStudioConfig

    cfg = LMStudioConfig(model="m")
    purged = _media_post(tmp_path, [str(tmp_path / "gone.jpg")])
    text_only = _media_post(tmp_path, [])

    assert compute_token_budget(purged, cfg) == compute_token_budget(text_only, cfg)
    assert compute_batch_token_budget([purged], cfg) == compute_batch_token_budget([text_only], cfg)
