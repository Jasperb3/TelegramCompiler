from __future__ import annotations
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from functools import lru_cache
from pathlib import Path

import yaml

from tg_compiler.config import TriageConfig
from tg_compiler.db import PostRecord, AnalysisRecord

log = logging.getLogger(__name__)

_NON_WORD = re.compile(r'[^\w\s]')

_SEVERITY_RANK: dict[str, int] = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}

# Aliases for entity-overlap dedup so naming variants of the same actor match
# (e.g. "U.S." and "United States" both normalize to "united states"). Keys are
# compared with periods stripped. Defined entirely via entity_aliases.yaml
# (gitignored, project root) — see entity_aliases.yaml.example. Empty if absent.
_ENTITY_ALIASES_PATH = Path("entity_aliases.yaml")


@lru_cache(maxsize=1)
def _entity_aliases() -> dict[str, str]:
    if not _ENTITY_ALIASES_PATH.exists():
        log.info("No entity_aliases.yaml found at %s — alias normalisation disabled", _ENTITY_ALIASES_PATH.resolve())
        return {}
    with open(_ENTITY_ALIASES_PATH) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        log.warning("entity_aliases.yaml at %s is not a flat string→string mapping — ignoring it", _ENTITY_ALIASES_PATH.resolve())
        return {}
    log.info("Loaded %d entity aliases from %s", len(data), _ENTITY_ALIASES_PATH.resolve())
    return data


def _normalize_entity(e: str) -> str:
    stripped = e.strip().lower().replace(".", "")
    return _entity_aliases().get(stripped, stripped)


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
        cand_entities = {_normalize_entity(e) for e in candidate.analysis.key_entities}
        exist_entities = {_normalize_entity(e) for e in existing.analysis.key_entities}
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
        cand_entities = {_normalize_entity(e) for e in candidate.analysis.key_entities}
        exist_entities = {_normalize_entity(e) for e in existing.analysis.key_entities}
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
        score = min(5.0, score)
        if analysis.category == "Rumor":
            score *= config.rumor_penalty
        score *= config.threat_multipliers.get(analysis.threat_level, 1.0)
        score *= _recency_multiplier(post.timestamp, now, config.recency_half_life_hours, config.recency_floor)
        scored.append(TriagedPost(post=post, analysis=analysis, composite_score=score))

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
    executive_items = critical_items + other_items[: max(0, 10 - len(critical_items))]

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
