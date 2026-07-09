from datetime import datetime, date, timezone, timedelta
from tg_compiler.db import PostRecord, AnalysisRecord
from tg_compiler.triage import triage, TriagedPost, BriefingContent, _jaccard, CorroborationRef, _normalize_entity
from tg_compiler import triage as triage_module
from tg_compiler.config import TriageConfig


def make_pair(
    importance=3, urgency=3, credibility=3, relevance=3,
    category="Analysis", text="test post", msg_id=1,
    summary="A test post summary.", title="", channel_name="test",
    timestamp=None,
):
    ts = timestamp or datetime.now(timezone.utc)
    post = PostRecord(
        id=msg_id, channel_id=1, channel_name=channel_name, message_id=msg_id,
        timestamp=ts,
        text=text, media_paths=[], has_images=False, raw_json="{}",
    )
    analysis = AnalysisRecord(
        id=msg_id, post_id=msg_id, summary=summary, title=title,
        importance_score=importance, urgency_score=urgency,
        credibility_score=credibility, relevance_score=relevance,
        category=category, key_entities=[], model_used="test",
    )
    return post, analysis


def test_composite_score_formula():
    post, analysis = make_pair(importance=5, urgency=4, credibility=3, relevance=2)
    config = TriageConfig(keywords=[], keyword_boost=0.5, min_composite_score=0.0)
    result = triage([(post, analysis)], config)
    expected = 0.4 * 5 + 0.3 * 4 + 0.2 * 3 + 0.1 * 2
    assert abs(result.main_items[0].composite_score - expected) < 0.001


def test_keyword_boost_applied():
    post, analysis = make_pair(importance=2, text="breaking news: urgent launch event")
    config = TriageConfig(keywords=["breaking", "urgent"], keyword_boost=1.0, min_composite_score=0.0)
    result = triage([(post, analysis)], config)
    base = 0.4 * 2 + 0.3 * 3 + 0.2 * 3 + 0.1 * 3
    assert result.main_items[0].composite_score > base


def test_keyword_boost_capped_at_five():
    post, analysis = make_pair(importance=5, urgency=5, credibility=5, relevance=5, text="urgent")
    config = TriageConfig(keywords=["urgent"], keyword_boost=2.0, min_composite_score=0.0)
    result = triage([(post, analysis)], config)
    assert result.main_items[0].composite_score <= 5.0


def test_below_threshold_goes_to_appendix():
    post, analysis = make_pair(importance=1, urgency=1, credibility=1, relevance=1)
    config = TriageConfig(min_composite_score=3.0, min_main_items=0)
    result = triage([(post, analysis)], config)
    assert len(result.main_items) == 0
    assert len(result.appendix_items) == 1


def test_min_main_items_promotes_top_appendix_items():
    # 5 dissimilar low-scoring posts, none clear the threshold; min_main_items=3
    # should promote the 3 highest-scored into main, leaving 2 in the appendix.
    summaries = [
        "Airstrikes hit Beirut overnight causing significant damage",
        "Ceasefire negotiations collapsed after delegates walked out",
        "President signed emergency decree expanding military powers",
        "Floods displaced thousands across southern provinces",
        "Opposition leader arrested on espionage charges",
    ]
    pairs = [
        make_pair(importance=i + 1, urgency=1, credibility=1, relevance=1, msg_id=i,
                  summary=summaries[i])
        for i in range(5)
    ]
    config = TriageConfig(min_composite_score=4.5, min_main_items=3)
    result = triage(pairs, config)
    assert len(result.main_items) == 3
    assert len(result.appendix_items) == 2
    main_scores = [t.composite_score for t in result.main_items]
    assert min(main_scores) >= max(t.composite_score for t in result.appendix_items)


def test_sorted_by_score_descending():
    pairs = [make_pair(importance=i, msg_id=i) for i in range(1, 5)]
    config = TriageConfig(min_composite_score=0.0)
    result = triage(pairs, config)
    scores = [t.composite_score for t in result.main_items]
    assert scores == sorted(scores, reverse=True)


def test_briefing_content_collects_channel_names():
    pairs = [make_pair()]
    config = TriageConfig(min_composite_score=0.0)
    result = triage(pairs, config)
    assert isinstance(result, BriefingContent)
    assert "test" in result.channel_names


def test_keyword_boost_applies_only_once():
    # Text matches both keywords; only one boost should be applied
    post, analysis = make_pair(importance=2, text="breaking urgent event")
    config = TriageConfig(keywords=["breaking", "urgent"], keyword_boost=1.0, min_composite_score=0.0)
    result = triage([(post, analysis)], config)
    base = 0.4 * 2 + 0.3 * 3 + 0.2 * 3 + 0.1 * 3
    # Exactly one boost: base + 1.0
    assert abs(result.main_items[0].composite_score - (base + 1.0)) < 0.001


def test_jaccard_similarity():
    assert _jaccard("Israeli airstrikes hit Dahiyeh Beirut", "Israeli airstrikes struck Dahiyeh Beirut Lebanon") > 0.3
    assert _jaccard("the weather is fine today", "completely unrelated topic about rockets") < 0.1


def test_dedup_removes_similar_post():
    # Two nearly-identical summaries within 1 hour → only the higher-scored one kept
    base_ts = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)
    summary = "Israeli airstrikes were reported in the Dahiyeh suburb of Beirut Lebanon"
    p1, a1 = make_pair(importance=5, msg_id=1, summary=summary, timestamp=base_ts)
    p2, a2 = make_pair(importance=3, msg_id=2,
                       summary="Israeli airstrikes reported in Dahiyeh suburb Beirut",
                       timestamp=base_ts + timedelta(minutes=30))
    config = TriageConfig(min_composite_score=0.0)
    result = triage([(p1, a1), (p2, a2)], config)
    total = len(result.main_items) + len(result.appendix_items)
    assert total == 1
    # The higher-scored post wins
    kept = (result.main_items + result.appendix_items)[0]
    assert kept.post.message_id == 1


def test_dedup_keeps_dissimilar_posts():
    base_ts = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)
    p1, a1 = make_pair(msg_id=1,
                       summary="Israeli airstrikes hit Dahiyeh suburb of Beirut",
                       timestamp=base_ts)
    p2, a2 = make_pair(msg_id=2,
                       summary="Ceasefire negotiations resumed in Qatar between warring parties",
                       timestamp=base_ts + timedelta(minutes=30))
    config = TriageConfig(min_composite_score=0.0)
    result = triage([(p1, a1), (p2, a2)], config)
    total = len(result.main_items) + len(result.appendix_items)
    assert total == 2


def test_dedup_ignores_outside_time_window():
    # Same summary text but beyond the 6h summary-similarity window → both kept (different events)
    summary = "Israeli airstrikes hit Dahiyeh suburb of Beirut Lebanon"
    p1, a1 = make_pair(msg_id=1, summary=summary,
                       timestamp=datetime(2026, 6, 7, 2, 0, tzinfo=timezone.utc))
    p2, a2 = make_pair(msg_id=2, summary=summary,
                       timestamp=datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc))
    config = TriageConfig(min_composite_score=0.0)
    result = triage([(p1, a1), (p2, a2)], config)
    total = len(result.main_items) + len(result.appendix_items)
    assert total == 2


def test_max_main_items_cap():
    # 10 high-score posts with dissimilar summaries, max_main_items=3 → 3 in main, 7 in appendix
    summaries = [
        "Airstrikes hit Beirut overnight causing significant damage",
        "Ceasefire negotiations collapsed after delegates walked out",
        "President signed emergency decree expanding military powers",
        "Floods displaced thousands across southern provinces",
        "Opposition leader arrested on espionage charges",
        "Oil pipeline sabotaged near border crossing",
        "Evacuation ordered for coastal settlements ahead of storm",
        "Diplomatic envoy expelled following spy scandal",
        "Rebel forces captured strategic bridge over river",
        "Parliament dissolved after vote of no confidence passed",
    ]
    pairs = [make_pair(importance=5, msg_id=i, summary=summaries[i]) for i in range(10)]
    config = TriageConfig(min_composite_score=0.0, min_main_items=0, max_main_items=3)
    result = triage(pairs, config)
    assert len(result.main_items) == 3
    assert len(result.appendix_items) == 7


def test_category_counts_populated():
    pairs = [
        make_pair(msg_id=1, summary="Breaking event in the northern region", category="Breaking News"),
        make_pair(msg_id=2, summary="Analysis of the diplomatic situation", category="Analysis"),
        make_pair(msg_id=3, summary="Official statement from government sources", category="Breaking News"),
    ]
    config = TriageConfig(min_composite_score=0.0)
    result = triage(pairs, config)
    assert result.category_counts["Breaking News"] == 2
    assert result.category_counts["Analysis"] == 1


def test_severity_tiebreaker_critical_over_high():
    # Both posts have the same composite score (5.0 via keyword boost); CRITICAL should rank first.
    base_ts = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)
    p1, a1 = make_pair(importance=5, urgency=5, credibility=5, relevance=5,
                       msg_id=1, summary="Critical event requiring immediate attention", timestamp=base_ts)
    a1.threat_level = "HIGH"
    p2, a2 = make_pair(importance=5, urgency=5, credibility=5, relevance=5,
                       msg_id=2, summary="Another event of critical severity level found", timestamp=base_ts)
    a2.threat_level = "CRITICAL"
    config = TriageConfig(min_composite_score=0.0)
    result = triage([(p1, a1), (p2, a2)], config)
    assert len(result.main_items) == 2  # both posts survive dedup
    assert result.main_items[0].post.message_id == 2  # CRITICAL outranks HIGH


def test_severity_tiebreaker_high_over_moderate():
    # Both posts have the same composite score; HIGH should rank above MODERATE.
    base_ts = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)
    p1, a1 = make_pair(importance=5, urgency=5, credibility=5, relevance=5,
                       msg_id=1, summary="Moderate situation developing in the region", timestamp=base_ts)
    a1.threat_level = "MODERATE"
    p2, a2 = make_pair(importance=5, urgency=5, credibility=5, relevance=5,
                       msg_id=2, summary="High severity incident reported at border crossing", timestamp=base_ts)
    a2.threat_level = "HIGH"
    config = TriageConfig(min_composite_score=0.0)
    result = triage([(p1, a1), (p2, a2)], config)
    assert len(result.main_items) == 2  # both posts survive dedup
    assert result.main_items[0].post.message_id == 2  # HIGH outranks MODERATE


def test_dedup_entity_overlap():
    # Same entities, same time window → duplicate
    base_ts = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)
    p1, a1 = make_pair(importance=5, msg_id=1,
                       summary="Unrelated text here with no word overlap whatsoever",
                       timestamp=base_ts)
    a1.key_entities = ["Israel", "Beirut", "Hezbollah", "Lebanon", "IDF"]
    p2, a2 = make_pair(importance=3, msg_id=2,
                       summary="Completely different phrasing and words in this one",
                       timestamp=base_ts + timedelta(minutes=45))
    a2.key_entities = ["Israel", "Beirut", "Hezbollah", "Lebanon", "strikes"]
    config = TriageConfig(min_composite_score=0.0)
    result = triage([(p1, a1), (p2, a2)], config)
    total = len(result.main_items) + len(result.appendix_items)
    assert total == 1


def test_dedup_24h_entity_cluster_collapses_duplicate():
    # Two posts sharing 4+ entities with a 4-hour gap → deduplicated by 24h rule
    base_ts = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)
    p1, a1 = make_pair(importance=5, msg_id=1,
                       summary="Unrelated wording that shares no words with the next post",
                       timestamp=base_ts)
    a1.key_entities = ["Israel", "Iran", "IRGC", "Tel Aviv", "ballistic missile"]
    p2, a2 = make_pair(importance=3, msg_id=2,
                       summary="Completely different phrasing about an entirely other matter",
                       timestamp=base_ts + timedelta(hours=4))
    a2.key_entities = ["Israel", "Iran", "IRGC", "Tel Aviv", "drone strike"]
    # dedup_window_secs=3600 (1h) ensures the 4h gap is outside the primary window,
    # so only the entity-cluster pass can trigger the dedup.
    config = TriageConfig(min_composite_score=0.0, dedup_window_secs=3600)
    result = triage([(p1, a1), (p2, a2)], config)
    total = len(result.main_items) + len(result.appendix_items)
    assert total == 1
    kept = (result.main_items + result.appendix_items)[0]
    assert kept.post.message_id == 1  # higher-scored post wins


def test_null_score_excluded_from_main_and_appendix():
    # A post with a zero importance score (corrupt record) must be excluded entirely.
    post, analysis = make_pair(importance=0, summary="A valid summary that is long enough to pass")
    config = TriageConfig(min_composite_score=0.0)
    result = triage([(post, analysis)], config)
    assert len(result.main_items) == 0
    assert len(result.appendix_items) == 0


def test_none_score_excluded_from_main_and_appendix():
    # A post with a None score (missing column from old DB migration) must be excluded entirely.
    post, analysis = make_pair(summary="A valid summary that is long enough to pass")
    analysis.urgency_score = None
    config = TriageConfig(min_composite_score=0.0)
    result = triage([(post, analysis)], config)
    assert len(result.main_items) == 0
    assert len(result.appendix_items) == 0


def test_dedup_24h_entity_cluster_not_triggered_with_only_3_entities():
    # Two posts sharing only 3 entities with a 3-hour gap → NOT collapsed
    # (gap > 2h so old rule doesn't apply; new rule requires 4 shared entities)
    base_ts = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)
    p1, a1 = make_pair(importance=5, msg_id=1,
                       summary="Unrelated wording that shares no words with the next post",
                       timestamp=base_ts)
    a1.key_entities = ["Israel", "Iran", "IRGC", "Tel Aviv"]
    p2, a2 = make_pair(importance=3, msg_id=2,
                       summary="Completely different phrasing about an entirely other matter",
                       timestamp=base_ts + timedelta(hours=3))
    a2.key_entities = ["Israel", "Iran", "IRGC", "Hezbollah"]
    config = TriageConfig(min_composite_score=0.0)
    result = triage([(p1, a1), (p2, a2)], config)
    total = len(result.main_items) + len(result.appendix_items)
    assert total == 2


def test_dedup_cross_pass_ordering_prefers_summary_pass_over_earlier_cluster_match():
    # Regression test pinning _find_duplicate's two-pass semantics: pass 1 (summary/
    # entity-overlap window) is fully scanned over `kept` before pass 2 (entity-cluster
    # window) ever runs. So a *later* kept item that matches on pass 1 wins over an
    # *earlier* kept item that would only have matched on pass 2 — even though the
    # earlier item is closer in time. If the two passes were ever merged into one
    # loop-per-existing-item, the earlier item (X) would win instead; this test would
    # then need updating rather than silently reporting a different representative.
    base_ts = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)
    # X: kept first (highest score); only qualifies for a match via the entity-cluster
    # pass (4h gap, 4+ shared entities) — its summary is unrelated to the candidate's.
    x_post, x_analysis = make_pair(
        importance=5, msg_id=1,
        summary="Agriculture ministry announces new crop subsidy reforms",
        timestamp=base_ts,
    )
    x_analysis.key_entities = ["Israel", "Iran", "IRGC", "Tel Aviv", "drone"]
    # Y: kept second; matches the candidate via the summary/jaccard pass 1 rule.
    y_post, y_analysis = make_pair(
        importance=4, msg_id=2,
        summary="Israeli airstrikes hit Dahiyeh suburb of Beirut Lebanon",
        timestamp=base_ts + timedelta(minutes=20),
    )
    # Candidate: shares 4 entities with X (4h gap) and near-identical summary with Y.
    c_post, c_analysis = make_pair(
        importance=1, msg_id=3,
        summary="Israeli airstrikes reported in Dahiyeh suburb Beirut",
        timestamp=base_ts + timedelta(hours=4),
    )
    c_analysis.key_entities = ["Israel", "Iran", "IRGC", "Tel Aviv", "strikes"]

    config = TriageConfig(
        min_composite_score=0.0,
        dedup_window_secs=3600,              # 1h: entity-overlap pass window
        dedup_summary_window_secs=21600,     # 6h: summary/jaccard pass window
        entity_cluster_window_secs=86400,    # 24h: entity-cluster pass window
    )
    result = triage([(x_post, x_analysis), (y_post, y_analysis), (c_post, c_analysis)], config)
    kept = result.main_items + result.appendix_items
    assert len(kept) == 2

    y_item = next(t for t in kept if t.post.message_id == 2)
    assert any(c.message_id == 3 for c in y_item.corroborations)
    x_item = next(t for t in kept if t.post.message_id == 1)
    assert x_item.corroborations == []


def test_skipped_category_excluded_from_main_and_appendix():
    post, analysis = make_pair(category="Skipped", summary="", importance=None, urgency=None,
                                credibility=None, relevance=None)
    config = TriageConfig(keywords=[], keyword_boost=0.5, min_composite_score=0.0)
    result = triage([(post, analysis)], config)
    assert result.main_items == []
    assert result.appendix_items == []


def test_dedup_extends_to_6h_summary_window():
    # Real-world regression: 2h19m apart, similar wording, just past the old 2h cutoff.
    p1, a1 = make_pair(
        msg_id=1, importance=5,
        title="Iran launches missiles at U.S. targets",
        summary="Reports indicate Iran launched missiles targeting US bases overnight.",
        timestamp=datetime(2026, 6, 7, 2, 45, tzinfo=timezone.utc),
    )
    p2, a2 = make_pair(
        msg_id=2, importance=3,
        title="Iran launches ballistic missiles at US military sites",
        summary="Iran launched ballistic missiles at US military sites overnight.",
        timestamp=datetime(2026, 6, 7, 5, 4, tzinfo=timezone.utc),
    )
    config = TriageConfig(min_composite_score=0.0)
    result = triage([(p1, a1), (p2, a2)], config)
    total = len(result.main_items) + len(result.appendix_items)
    assert total == 1
    kept = (result.main_items + result.appendix_items)[0]
    assert kept.post.message_id == 1


def test_recency_decay_demotes_older_post():
    now = datetime.now(timezone.utc)
    recent_post, recent_analysis = make_pair(msg_id=1, importance=4, urgency=4, credibility=4, relevance=4,
                                              timestamp=now - timedelta(hours=1),
                                              summary="A recent breaking development overseas")
    old_post, old_analysis = make_pair(msg_id=2, importance=4, urgency=4, credibility=4, relevance=4,
                                        timestamp=now - timedelta(hours=26),
                                        summary="An older unrelated development happened yesterday")
    config = TriageConfig(min_composite_score=0.0)
    result = triage([(recent_post, recent_analysis), (old_post, old_analysis)], config)
    by_id = {t.post.message_id: t.composite_score for t in result.main_items}
    assert by_id[1] > by_id[2]


def test_recency_floor_caps_decay():
    now = datetime.now(timezone.utc)
    post, analysis = make_pair(importance=4, urgency=4, credibility=4, relevance=4,
                                timestamp=now - timedelta(days=10))
    config = TriageConfig(min_composite_score=0.0, recency_half_life_hours=12.0, recency_floor=0.6)
    result = triage([(post, analysis)], config)
    base = 0.4 * 4 + 0.3 * 4 + 0.2 * 4 + 0.1 * 4
    assert abs(result.main_items[0].composite_score - base * 0.6) < 0.001


def test_executive_items_includes_all_critical_regardless_of_score():
    summaries = [
        "Airstrikes hit Beirut overnight causing significant damage",
        "Ceasefire negotiations collapsed after delegates walked out",
        "President signed emergency decree expanding military powers",
        "Floods displaced thousands across southern provinces",
        "Opposition leader arrested on espionage charges",
        "Oil pipeline sabotaged near border crossing",
        "Evacuation ordered for coastal settlements ahead of storm",
        "Diplomatic envoy expelled following spy scandal",
        "Rebel forces captured strategic bridge over river",
        "Parliament dissolved after vote of no confidence passed",
    ]
    pairs = []
    # 10 high-scoring HIGH-severity items fill the executive summary
    for i in range(10):
        post, analysis = make_pair(msg_id=i, importance=5, urgency=5, credibility=5, relevance=5,
                                    summary=summaries[i])
        analysis.threat_level = "HIGH"
        pairs.append((post, analysis))
    # One CRITICAL item with a low composite score (credibility 3: substantiated enough
    # to keep its CRITICAL badge past the badge guard)
    crit_post, crit_analysis = make_pair(msg_id=99, importance=2, urgency=2, credibility=3, relevance=2,
                                          summary="A critical low-scored but urgent unrelated alert")
    crit_analysis.threat_level = "CRITICAL"
    pairs.append((crit_post, crit_analysis))

    config = TriageConfig(min_composite_score=0.0, max_main_items=20)
    result = triage(pairs, config)
    assert len(result.executive_items) == 10
    assert any(t.post.message_id == 99 for t in result.executive_items)


def test_entity_alias_normalization_clusters_duplicates(tmp_path, monkeypatch):
    # Same 3 entities under different naming conventions, distinct summaries
    # so only the alias-aware entity-overlap leg can trigger the match.
    alias_file = tmp_path / "entity_aliases.yaml"
    alias_file.write_text("acme corp: acme corporation\n")
    monkeypatch.setattr(triage_module, "_ENTITY_ALIASES_PATH", alias_file)
    triage_module._entity_aliases.cache_clear()

    base_ts = datetime.now(timezone.utc)
    p1, a1 = make_pair(msg_id=1, importance=5,
                        summary="Forces mass near the northern frontier overnight",
                        timestamp=base_ts)
    a1.key_entities = ["Acme Corp", "Foo", "Bar"]
    p2, a2 = make_pair(msg_id=2, importance=3,
                        summary="Officials describe a tense standoff at the border crossing",
                        timestamp=base_ts + timedelta(minutes=30))
    a2.key_entities = ["Acme Corporation", "Foo", "Bar"]

    try:
        config = TriageConfig(min_composite_score=0.0)
        result = triage([(p1, a1), (p2, a2)], config)
    finally:
        triage_module._entity_aliases.cache_clear()

    total = len(result.main_items) + len(result.appendix_items)
    assert total == 1
    kept = (result.main_items + result.appendix_items)[0]
    assert kept.post.message_id == 1
    assert len(kept.corroborations) == 1
    assert kept.corroborations[0].channel_slug == p2.channel_name


def test_corroboration_recorded_and_score_boosted():
    base_ts = datetime.now(timezone.utc)
    summary = "Israeli airstrikes were reported in the Dahiyeh suburb of Beirut Lebanon"
    p1, a1 = make_pair(importance=5, urgency=5, credibility=5, relevance=5, msg_id=1,
                        summary=summary, channel_name="chan_a", timestamp=base_ts)
    p2, a2 = make_pair(importance=3, msg_id=2,
                        summary="Israeli airstrikes reported in Dahiyeh suburb Beirut",
                        channel_name="chan_b", timestamp=base_ts + timedelta(minutes=30))

    config = TriageConfig(min_composite_score=0.0, corroboration_weight=0.15, corroboration_cap=1.5)
    result = triage([(p1, a1), (p2, a2)], config)
    kept = (result.main_items + result.appendix_items)[0]
    assert kept.post.message_id == 1
    assert len(kept.corroborations) == 1
    assert kept.corroborations[0].channel_slug == "chan_b"
    assert kept.corroborations[0].message_id == 2

    base_score = 0.4 * 5 + 0.3 * 5 + 0.2 * 5 + 0.1 * 5  # 5.0, capped before boost
    expected = base_score * (1 + 0.15 * 1)
    assert abs(kept.composite_score - expected) < 0.01


def test_corroboration_boost_capped():
    base_ts = datetime.now(timezone.utc)
    summary = "Israeli airstrikes were reported in the Dahiyeh suburb of Beirut Lebanon"
    p1, a1 = make_pair(importance=5, urgency=5, credibility=5, relevance=5, msg_id=1,
                        summary=summary, timestamp=base_ts)

    pairs = [(p1, a1)]
    for i in range(2, 7):  # 5 corroborating duplicates
        p, a = make_pair(importance=2, msg_id=i,
                          summary="Israeli airstrikes reported in Dahiyeh suburb Beirut",
                          channel_name=f"chan_{i}",
                          timestamp=base_ts + timedelta(minutes=10 * i))
        pairs.append((p, a))

    config = TriageConfig(min_composite_score=0.0, corroboration_weight=0.15, corroboration_cap=1.5)
    result = triage(pairs, config)
    total = len(result.main_items) + len(result.appendix_items)
    assert total == 1
    kept = (result.main_items + result.appendix_items)[0]
    assert len(kept.corroborations) == 5

    base_score = 0.4 * 5 + 0.3 * 5 + 0.2 * 5 + 0.1 * 5
    expected = base_score * 1.5  # capped at corroboration_cap
    assert abs(kept.composite_score - expected) < 0.01


def test_posts_clustered_count():
    base_ts = datetime.now(timezone.utc)
    summary = "Israeli airstrikes were reported in the Dahiyeh suburb of Beirut Lebanon"
    p1, a1 = make_pair(msg_id=1, importance=5, summary=summary, timestamp=base_ts)
    p2, a2 = make_pair(msg_id=2, importance=3,
                        summary="Israeli airstrikes reported in Dahiyeh suburb Beirut",
                        timestamp=base_ts + timedelta(minutes=30))
    p3, a3 = make_pair(msg_id=3,
                        summary="Ceasefire negotiations resumed in Qatar between warring parties",
                        timestamp=base_ts + timedelta(minutes=45))

    config = TriageConfig(min_composite_score=0.0)
    result = triage([(p1, a1), (p2, a2), (p3, a3)], config)
    assert result.posts_clustered == 1


def test_recency_anchored_to_briefing_day_for_past_dates():
    # Triaging a date in the past must anchor decay to the end of that day,
    # not wall-clock now — otherwise retrospective --analyse runs decay every
    # post to the floor and select a different main set than the original run.
    target = date(2026, 1, 10)
    post, analysis = make_pair(importance=4, urgency=4, credibility=4, relevance=4,
                                timestamp=datetime(2026, 1, 10, 22, 0, tzinfo=timezone.utc))
    config = TriageConfig(min_composite_score=0.0, recency_half_life_hours=12.0, recency_floor=0.6)
    result = triage([(post, analysis)], config, today=target)
    base = 0.4 * 4 + 0.3 * 4 + 0.2 * 4 + 0.1 * 4
    # ~2h old relative to end of the briefing day: 0.5 ** (2/12) ≈ 0.891
    expected = base * (0.5 ** (2 / 12))
    assert abs(result.main_items[0].composite_score - expected) < 0.01


def test_executive_items_includes_critical_from_appendix():
    crit_post, crit_analysis = make_pair(msg_id=99, importance=2, urgency=2, credibility=3, relevance=2,
                                          summary="A critical alert that scored below the main threshold")
    crit_analysis.threat_level = "CRITICAL"
    high_post, high_analysis = make_pair(msg_id=1, importance=5, urgency=5, credibility=5, relevance=5,
                                          summary="A completely different high scoring story from elsewhere")
    high_analysis.threat_level = "HIGH"

    config = TriageConfig(min_composite_score=3.0, min_main_items=0)
    result = triage([(crit_post, crit_analysis), (high_post, high_analysis)], config)
    assert any(t.post.message_id == 99 for t in result.appendix_items)
    assert any(t.post.message_id == 99 for t in result.executive_items)

def test_same_channel_duplicates_recorded_but_not_boosted():
    base_ts = datetime.now(timezone.utc)
    summary = "Israeli airstrikes were reported in the Dahiyeh suburb of Beirut Lebanon"
    p1, a1 = make_pair(importance=5, urgency=5, credibility=5, relevance=5, msg_id=1,
                        summary=summary, channel_name="chan_a", timestamp=base_ts)
    p2, a2 = make_pair(importance=3, msg_id=2,
                        summary="Israeli airstrikes reported in Dahiyeh suburb Beirut",
                        channel_name="chan_a", timestamp=base_ts + timedelta(minutes=30))

    config = TriageConfig(min_composite_score=0.0, corroboration_weight=0.15, corroboration_cap=1.5)
    result = triage([(p1, a1), (p2, a2)], config)
    kept = (result.main_items + result.appendix_items)[0]
    assert len(kept.corroborations) == 1   # still merged as a duplicate
    assert abs(kept.composite_score - 5.0) < 0.01  # but no corroboration boost


def test_corroboration_boost_counts_distinct_other_channels():
    base_ts = datetime.now(timezone.utc)
    summary = "Israeli airstrikes were reported in the Dahiyeh suburb of Beirut Lebanon"
    p1, a1 = make_pair(importance=5, urgency=5, credibility=5, relevance=5, msg_id=1,
                        summary=summary, channel_name="chan_a", timestamp=base_ts)
    # two duplicates from one other channel + one from the representative's own channel
    p2, a2 = make_pair(importance=3, msg_id=2,
                        summary="Israeli airstrikes reported in Dahiyeh suburb Beirut",
                        channel_name="chan_b", timestamp=base_ts + timedelta(minutes=10))
    p3, a3 = make_pair(importance=3, msg_id=3,
                        summary="Israeli airstrikes reported in the Dahiyeh suburb of Beirut",
                        channel_name="chan_b", timestamp=base_ts + timedelta(minutes=20))
    p4, a4 = make_pair(importance=3, msg_id=4,
                        summary="Airstrikes by Israel reported in Dahiyeh Beirut suburb",
                        channel_name="chan_a", timestamp=base_ts + timedelta(minutes=30))

    config = TriageConfig(min_composite_score=0.0, corroboration_weight=0.15, corroboration_cap=1.5)
    result = triage([(p1, a1), (p2, a2), (p3, a3), (p4, a4)], config)
    kept = (result.main_items + result.appendix_items)[0]
    assert len(kept.corroborations) == 3
    expected = 5.0 * (1 + 0.15 * 1)  # chan_b counts once; own-channel dup not at all
    assert abs(kept.composite_score - expected) < 0.01


def test_executive_items_uncapped_when_criticals_exceed_ten():
    summaries = [
        "Airstrikes hit Beirut overnight causing significant damage",
        "Ceasefire negotiations collapsed after delegates walked out",
        "President signed emergency decree expanding military powers",
        "Floods displaced thousands across southern provinces",
        "Opposition leader arrested on espionage charges",
        "Oil pipeline sabotaged near border crossing",
        "Evacuation ordered for coastal settlements ahead of storm",
        "Diplomatic envoy expelled following spy scandal",
        "Rebel forces captured strategic bridge over river",
        "Parliament dissolved after vote of no confidence passed",
        "Border crossing shelled during humanitarian convoy transit",
        "Naval blockade announced around disputed island chain",
    ]
    pairs = []
    for i in range(12):
        post, analysis = make_pair(msg_id=i, importance=5, summary=summaries[i])
        analysis.threat_level = "CRITICAL"
        pairs.append((post, analysis))

    config = TriageConfig(min_composite_score=0.0)
    result = triage(pairs, config)
    assert len(result.executive_items) == 12  # all criticals kept, no cap at 10


def test_content_carries_min_composite_score():
    post, analysis = make_pair()
    config = TriageConfig(min_composite_score=3.5)
    result = triage([(post, analysis)], config)
    assert result.min_composite_score == 3.5


def test_badge_guard_demotes_unsubstantiated_critical():
    # A Rumor-category CRITICAL is demoted to HIGH
    p1, a1 = make_pair(msg_id=1, category="Rumor", credibility=4,
                       summary="An alleged unverified attack on a coastal city region")
    a1.threat_level = "CRITICAL"
    # A low-credibility CRITICAL is demoted to HIGH
    p2, a2 = make_pair(msg_id=2, category="Breaking News", credibility=2,
                       summary="Completely different story about a diplomatic summit meeting")
    a2.threat_level = "CRITICAL"
    # A substantiated CRITICAL keeps its badge
    p3, a3 = make_pair(msg_id=3, category="Breaking News", credibility=4,
                       summary="Another unrelated report on missile defence procurement plans")
    a3.threat_level = "CRITICAL"

    config = TriageConfig(min_composite_score=0.0)
    result = triage([(p1, a1), (p2, a2), (p3, a3)], config)
    levels = {t.post.message_id: t.analysis.threat_level for t in result.main_items}
    assert levels[1] == "HIGH"
    assert levels[2] == "HIGH"
    assert levels[3] == "CRITICAL"


def test_threat_multiplier_ranks_low_below_moderate():
    # identical axis scores, different threat levels: LOW is down-weighted, CRITICAL boosted
    p1, a1 = make_pair(msg_id=1, summary="A sporting event preview with identical scores")
    a1.threat_level = "LOW"
    p2, a2 = make_pair(msg_id=2, summary="A diplomatic development report with same numbers")
    a2.threat_level = "MODERATE"
    p3, a3 = make_pair(msg_id=3, credibility=3, summary="An entirely different confirmed conflict escalation event")
    a3.threat_level = "CRITICAL"

    config = TriageConfig(min_composite_score=0.0)
    result = triage([(p1, a1), (p2, a2), (p3, a3)], config)
    by_id = {t.post.message_id: t.composite_score for t in result.main_items}
    assert by_id[1] < by_id[2] < by_id[3]
    assert abs(by_id[1] / by_id[2] - 0.85) < 0.01
    assert abs(by_id[3] / by_id[2] - 1.15) < 0.01


def test_entity_aliases_loaded_from_custom_file(tmp_path, monkeypatch):
    alias_file = tmp_path / "entity_aliases.yaml"
    alias_file.write_text("foo bar: baz corp\n")
    monkeypatch.setattr(triage_module, "_ENTITY_ALIASES_PATH", alias_file)
    triage_module._entity_aliases.cache_clear()
    try:
        assert _normalize_entity("Foo Bar") == "baz corp"
    finally:
        triage_module._entity_aliases.cache_clear()


def test_entity_aliases_empty_without_custom_file(tmp_path, monkeypatch):
    monkeypatch.setattr(triage_module, "_ENTITY_ALIASES_PATH", tmp_path / "missing.yaml")
    triage_module._entity_aliases.cache_clear()
    try:
        # with no alias file, normalization is just lowercase + strip periods
        assert _normalize_entity("U.S.") == "us"
    finally:
        triage_module._entity_aliases.cache_clear()


def test_entity_aliases_malformed_yaml_ignored_with_warning(tmp_path, monkeypatch, caplog):
    import logging

    alias_file = tmp_path / "entity_aliases.yaml"
    alias_file.write_text("- foo\n- bar\n")  # a list, not a str->str mapping
    monkeypatch.setattr(triage_module, "_ENTITY_ALIASES_PATH", alias_file)
    triage_module._entity_aliases.cache_clear()
    try:
        with caplog.at_level(logging.WARNING):
            assert triage_module._entity_aliases() == {}
        assert any("not a flat string" in r.message for r in caplog.records)
    finally:
        triage_module._entity_aliases.cache_clear()
