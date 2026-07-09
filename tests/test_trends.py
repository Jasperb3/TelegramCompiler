from datetime import date, datetime, timezone

from tg_compiler import utils as utils_module
from tg_compiler.db import AnalysisRecord, PostRecord
from tg_compiler.trends import compute_trends


def _pair(msg_id, day, category, entities):
    post = PostRecord(
        channel_id=1, channel_name="chan", message_id=msg_id,
        timestamp=datetime(2026, 6, day, 12, 0, tzinfo=timezone.utc),
        text="x", media_paths=[], has_images=False, raw_json="{}",
    )
    analysis = AnalysisRecord(
        post_id=msg_id, summary="s", importance_score=3, urgency_score=2,
        credibility_score=4, relevance_score=3, category=category,
        key_entities=entities, model_used="test",
    )
    return post, analysis


def test_entity_and_category_deltas():
    history = [
        _pair(1, 9, "Military", ["Bahrain", "Iran"]),
        _pair(2, 9, "Military", ["Bahrain"]),
        _pair(3, 8, "Military", ["Iran"]),
        _pair(4, 7, "Analysis", ["Iran"]),
    ]
    trends = compute_trends(history, date(2026, 6, 9))

    by_entity = {d["entity"]: d for d in trends["entity_deltas"]}
    assert by_entity["bahrain"] == {"entity": "bahrain", "today_count": 2, "prior_count": 0}
    assert by_entity["iran"] == {"entity": "iran", "today_count": 1, "prior_count": 2}

    by_category = {d["category"]: d for d in trends["category_deltas"]}
    assert by_category["military".title()] == {"category": "Military", "today_count": 2, "prior_count": 1}


def test_emerging_entities_unseen_in_prior_days():
    history = [
        _pair(1, 9, "Military", ["Bahrain"]),
        _pair(2, 8, "Military", ["Iran"]),
    ]
    trends = compute_trends(history, date(2026, 6, 9))
    assert trends["emerging_entities"] == ["bahrain"]


def test_entity_alias_normalization_merges_history(tmp_path, monkeypatch):
    alias_file = tmp_path / "entity_aliases.yaml"
    alias_file.write_text("us: united states\n")
    monkeypatch.setattr(utils_module, "_ENTITY_ALIASES_PATH", alias_file)
    utils_module.entity_aliases.cache_clear()

    history = [
        _pair(1, 9, "Military", ["U.S."]),
        _pair(2, 8, "Military", ["United States"]),
    ]
    try:
        trends = compute_trends(history, date(2026, 6, 9))
    finally:
        utils_module.entity_aliases.cache_clear()

    assert trends["emerging_entities"] == []
    by_entity = {d["entity"]: d for d in trends["entity_deltas"]}
    assert by_entity["united states"] == {"entity": "united states", "today_count": 1, "prior_count": 1}


def test_no_history_returns_empty_results():
    trends = compute_trends([], date(2026, 6, 9))
    assert trends == {"entity_deltas": [], "category_deltas": [], "emerging_entities": []}


def test_skipped_category_excluded_from_trends():
    history = [
        _pair(1, 9, "Military", ["Bahrain"]),
        _pair(2, 9, "Skipped", []),
        _pair(3, 8, "Skipped", []),
        _pair(4, 8, "Military", ["Iran"]),
    ]
    trends = compute_trends(history, date(2026, 6, 9))
    categories = {d["category"] for d in trends["category_deltas"]}
    assert "Skipped" not in categories


def test_emerging_entities_empty_without_prior_baseline():
    # All posts on the target day (fresh DB / day one): no baseline, so
    # nothing should be flagged as emerging.
    history = [
        _pair(1, 9, "Military", ["Bahrain"]),
        _pair(2, 9, "Military", ["Iran"]),
    ]
    trends = compute_trends(history, date(2026, 6, 9))
    assert trends["emerging_entities"] == []


def test_emerging_entities_capped_and_sorted_by_frequency():
    from tg_compiler.trends import MAX_EMERGING_ENTITIES

    history = [_pair(1, 8, "Military", ["OldActor"])]
    # 20 new entities today; "hotspot" mentioned in three posts
    for i in range(20):
        history.append(_pair(10 + i, 9, "Military", [f"new_entity_{i:02d}"]))
    for j in range(3):
        history.append(_pair(50 + j, 9, "Military", ["hotspot"]))

    trends = compute_trends(history, date(2026, 6, 9))
    emerging = trends["emerging_entities"]
    assert len(emerging) == MAX_EMERGING_ENTITIES
    assert emerging[0] == "hotspot"

def test_emerging_excludes_newswire_credits_and_numeric_designators():
    history = [
        _pair(1, 8, "Military", ["OldActor"]),
        _pair(2, 9, "Military", ["AFP", "Tasnim", "100th Mechanized Brigade",
                                 "2026 FIFA World Cup", "New Real Actor"]),
    ]
    trends = compute_trends(history, date(2026, 6, 9))
    assert trends["emerging_entities"] == ["new real actor"]


def test_trend_counting_filters_garbage_entities():
    history = [
        _pair(1, 8, "Military", ["OldActor"]),
        _pair(2, 9, "Military", ["key_entities", "{json}", "Bahrain"]),
    ]
    trends = compute_trends(history, date(2026, 6, 9))
    assert trends["emerging_entities"] == ["bahrain"]
    entities = {d["entity"] for d in trends["entity_deltas"]}
    assert "key_entities" not in entities
