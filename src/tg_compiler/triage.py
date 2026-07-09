from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone

from tg_compiler.config import TriageConfig
from tg_compiler.db import AnalysisRecord, PostRecord
from tg_compiler.utils import normalize_entity

log = logging.getLogger(__name__)

_NON_WORD = re.compile(r'[^\w\s]')

_SEVERITY_RANK: dict[str, int] = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}

# Guaranteed size of the Lead Reports section before criticals-overflow, topped
# up with the highest-scoring non-critical main items (see triage()).
LEAD_SECTION_SIZE = 10


@dataclass
class CorroborationRef:
    channel_slug: str
    message_id: int
    timestamp: datetime


@dataclass
class TriagedPost:
    post: PostRecord
    analysis: AnalysisRecord
    composite_score: float
    corroborations: list[CorroborationRef] = field(default_factory=list)
    # Precomputed normalized entity set, filled in by triage() before dedup so
    # _find_duplicate doesn't re-normalize the same post's entities on every comparison.
    norm_entities: frozenset[str] | None = None


@dataclass
class BriefingContent:
    date: date
    main_items: list[TriagedPost]
    appendix_items: list[TriagedPost]
    channel_names: list[str] = field(default_factory=list)
    # slug → bare username (no @) for building t.me deep links
    channel_links: dict[str, str] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)
    # Executive Summary items: all CRITICAL items, then highest-scoring others, capped at 10
    executive_items: list[TriagedPost] = field(default_factory=list)
    # Pipeline funnel stats (set externally by main.py / inside triage())
    posts_scraped: int = 0
    posts_analysed: int = 0
    posts_skipped: int = 0
    posts_clustered: int = 0
    # The configured score threshold, carried so the reader's key can state it
    min_composite_score: float = 0.0


def _composite(a: AnalysisRecord) -> float:
    return (
        0.4 * a.importance_score
        + 0.3 * a.urgency_score
        + 0.2 * a.credibility_score
        + 0.1 * a.relevance_score
    )


def _recency_multiplier(post_time: datetime, now: datetime, half_life_hours: float, floor: float) -> float:
    age_hours = max(0.0, (now - post_time).total_seconds() / 3600)
    multiplier = 0.5 ** (age_hours / half_life_hours)
    return max(floor, multiplier)


def _jaccard(a: str, b: str, min_len: int = 3) -> float:
    words_a = {w for w in _NON_WORD.sub('', a.lower()).split() if len(w) >= min_len}
    words_b = {w for w in _NON_WORD.sub('', b.lower()).split() if len(w) >= min_len}
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _find_duplicate(
    candidate: TriagedPost,
    kept: list[TriagedPost],
    time_window_secs: float,
    entity_cluster_window_secs: float = 86400,
    threshold: float = 0.28,
    summary_window_secs: float | None = None,
    entity_overlap_count: int = 3,
    entity_cluster_overlap_count: int = 4,
) -> TriagedPost | None:
    if summary_window_secs is None:
        summary_window_secs = time_window_secs

    for existing in kept:
        delta = abs(
            (candidate.post.timestamp - existing.post.timestamp).total_seconds()
        )
        if delta <= summary_window_secs:
            if _jaccard(candidate.analysis.summary, existing.analysis.summary) >= threshold:
                return existing
            if candidate.analysis.title and existing.analysis.title:
                if _jaccard(candidate.analysis.title, existing.analysis.title) >= threshold:
                    return existing
        if delta > time_window_secs:
            continue
        # Entity overlap: ≥entity_overlap_count shared named entities within time window
        cand_entities = candidate.norm_entities
        exist_entities = existing.norm_entities
        if len(cand_entities) >= entity_overlap_count and len(exist_entities) >= entity_overlap_count:
            if len(cand_entities & exist_entities) >= entity_overlap_count:
                return existing

    # Extended entity-cluster pass: ≥entity_cluster_overlap_count shared entities within
    # the cluster window (default 24h)
    for existing in kept:
        delta = abs(
            (candidate.post.timestamp - existing.post.timestamp).total_seconds()
        )
        if delta > entity_cluster_window_secs:
            continue
        cand_entities = candidate.norm_entities
        exist_entities = existing.norm_entities
        if len(cand_entities) >= entity_cluster_overlap_count and len(exist_entities) >= entity_cluster_overlap_count:
            if len(cand_entities & exist_entities) >= entity_cluster_overlap_count:
                return existing

    return None


def triage(
    pairs: list[tuple[PostRecord, AnalysisRecord]],
    config: TriageConfig,
    today: date | None = None,
    channel_priorities: dict[str, float] | None = None,
    channel_credibilities: dict[str, float] | None = None,
) -> BriefingContent:
    """Score, badge-guard, dedup-cluster, and split (pairs of post, analysis) into a
    BriefingContent's main/appendix/executive item lists. Mutates analysis.threat_level
    in place when the badge guard demotes an unsubstantiated CRITICAL to HIGH — this is
    by design, so an already-stored analysis is corrected the next time it's triaged."""
    today = today or date.today()
    # Anchor recency decay to the end of the briefing day when triaging a past date,
    # so retrospective runs (--analyse --since) reproduce the same ranking the daemon
    # would have produced instead of decaying every post to the floor.
    now = min(
        datetime.now(timezone.utc),
        datetime.combine(today, time.max, tzinfo=timezone.utc),
    )
    scored: list[TriagedPost] = []

    for post, analysis in pairs:
        if analysis.category == "Skipped":
            continue
        if not analysis.summary or len(analysis.summary.strip()) < 10:
            continue
        if not all([analysis.importance_score, analysis.urgency_score,
                    analysis.credibility_score, analysis.relevance_score]):
            continue
        # Badge guard: CRITICAL means "confirmed ... underway" — a rumor or a
        # low-credibility claim can't be CRITICAL, so demote to HIGH. This stops
        # unsubstantiated posts hijacking the guaranteed lead-section slots.
        if analysis.threat_level == "CRITICAL" and (
            analysis.category == "Rumor" or analysis.credibility_score <= 2
        ):
            analysis.threat_level = "HIGH"
        priority = (channel_priorities or {}).get(post.channel_name, 1.0)
        credibility = (channel_credibilities or {}).get(post.channel_name, 1.0)
        score = _composite(analysis) * priority * credibility
        text_lower = (post.text or "").lower()
        for kw in config.keywords:
            if kw.lower() in text_lower:
                score = min(5.0, score + config.keyword_boost)
                break
        # Second cap: priority/credibility multipliers above 1.0 can push an
        # un-boosted score over 5.0 too, not just the keyword-boost branch above.
        score = min(5.0, score)
        if analysis.category == "Rumor":
            score *= config.rumor_penalty
        score *= config.threat_multipliers.get(analysis.threat_level, 1.0)
        score *= _recency_multiplier(post.timestamp, now, config.recency_half_life_hours, config.recency_floor)
        norm_entities = frozenset(normalize_entity(e) for e in analysis.key_entities)
        scored.append(TriagedPost(post=post, analysis=analysis, composite_score=score, norm_entities=norm_entities))

    scored.sort(key=lambda t: (
        -t.composite_score,
        -_SEVERITY_RANK.get(t.analysis.threat_level, 0),
        -t.post.timestamp.timestamp(),
    ))

    # Cluster duplicates: keep the highest-scoring report per story as the representative,
    # and record lower-scored cross-channel reposts as corroborations rather than dropping
    # them. Two posts are duplicates if they share >= dedup threshold words in summary/title
    # within dedup_summary_window_secs, or share enough named entities within the other
    # configured windows (see _find_duplicate).
    kept: list[TriagedPost] = []
    for item in scored:
        match = _find_duplicate(item, kept, time_window_secs=config.dedup_window_secs,
                                 entity_cluster_window_secs=config.entity_cluster_window_secs,
                                 threshold=config.dedup_jaccard_threshold,
                                 summary_window_secs=config.dedup_summary_window_secs,
                                 entity_overlap_count=config.dedup_entity_overlap_count,
                                 entity_cluster_overlap_count=config.dedup_entity_cluster_overlap_count)
        if match is not None:
            match.corroborations.append(CorroborationRef(
                channel_slug=item.post.channel_name,
                message_id=item.post.message_id,
                timestamp=item.post.timestamp,
            ))
        else:
            kept.append(item)

    posts_clustered = len(scored) - len(kept)

    # Apply a multiplicative corroboration boost to representatives with corroborations,
    # then re-sort since boosting can change relative ranking. Only distinct *other*
    # channels count — repeat posts from the representative's own channel are merged
    # but are not independent confirmation, so they earn no boost.
    for item in kept:
        other_channels = {c.channel_slug for c in item.corroborations} - {item.post.channel_name}
        if other_channels:
            boost = min(
                1 + config.corroboration_weight * len(other_channels),
                config.corroboration_cap,
            )
            item.composite_score *= boost

    kept.sort(key=lambda t: (
        -t.composite_score,
        -_SEVERITY_RANK.get(t.analysis.threat_level, 0),
        -t.post.timestamp.timestamp(),
    ))

    main_scored = [t for t in kept if t.composite_score >= config.min_composite_score]
    appendix_items = [t for t in kept if t.composite_score < config.min_composite_score]
    # Adaptive floor: when too few items clear min_composite_score, promote the
    # highest-scoring appendix items so the main briefing never runs thin.
    # appendix_items is already sorted descending, so order is preserved.
    if len(main_scored) < config.min_main_items:
        shortfall = config.min_main_items - len(main_scored)
        main_scored += appendix_items[:shortfall]
        appendix_items = appendix_items[shortfall:]
    if config.max_main_items > 0:
        appendix_items = main_scored[config.max_main_items:] + appendix_items
        main_items = main_scored[:config.max_main_items]
    else:
        main_items = main_scored

    all_kept = main_items + appendix_items
    category_counts = dict(Counter(t.analysis.category for t in all_kept))
    channel_names = sorted({p.channel_name for p, _ in pairs})

    # Lead reports: every CRITICAL item is guaranteed a slot, regardless of score
    # (including those relegated to the appendix), topped up with the highest-scoring
    # remaining main items to 10 total — never truncating criticals, so the lead
    # section grows beyond 10 when criticals alone exceed it.
    critical_items = [t for t in all_kept if t.analysis.threat_level == "CRITICAL"]
    other_items = [t for t in main_items if t.analysis.threat_level != "CRITICAL"]
    executive_items = critical_items + other_items[: max(0, LEAD_SECTION_SIZE - len(critical_items))]

    return BriefingContent(
        date=today,
        main_items=main_items,
        appendix_items=appendix_items,
        channel_names=channel_names,
        category_counts=category_counts,
        executive_items=executive_items,
        posts_clustered=posts_clustered,
        min_composite_score=config.min_composite_score,
    )
