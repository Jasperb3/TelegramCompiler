from __future__ import annotations

import re
from collections import Counter
from datetime import date

from tg_compiler.db import AnalysisRecord, PostRecord
from tg_compiler.utils import clean_entities, normalize_entity

TREND_WINDOW_DAYS = 7
TOP_N_ENTITIES = 10
MAX_EMERGING_ENTITIES = 15

# Newswire/photo credits and platforms the LLM extracts as "entities" — they are
# sources, not actors, so they never qualify as emerging. Compared after
# normalize_entity (lowercased, periods stripped).
_EMERGING_STOPLIST = {
    "afp", "ap", "associated press", "reuters", "getty", "getty images",
    "anadolu", "anadolu agency", "dpa", "efe", "tass", "ria novosti", "interfax",
    "fars", "farsna", "fars news agency", "tasnim", "tasnim news agency",
    "mehr", "mehr news", "mehr news agency", "irna", "isna", "sana", "kcna", "xinhua",
    "telegram", "twitter", "twitter/x", "x", "instagram", "facebook", "youtube",
    "tiktok", "social media",
}

# Entities starting with a digit ("100th Mechanized Brigade", "2026 Fifa World Cup",
# "1998 Peace Agreement") are date/designator churn, not emerging actors.
_LEADING_DIGIT = re.compile(r"^\d")


def compute_trends(history: list[tuple[PostRecord, AnalysisRecord]], target_date: date) -> dict:
    """Split a multi-day history into today vs prior days and compute
    entity/category mention deltas plus entities first seen today."""
    today_entities: Counter[str] = Counter()
    today_categories: Counter[str] = Counter()
    prior_entities: Counter[str] = Counter()
    prior_categories: Counter[str] = Counter()

    for post, analysis in history:
        if analysis.category == "Skipped":
            continue
        is_today = post.timestamp.date() == target_date
        entity_bucket = today_entities if is_today else prior_entities
        category_bucket = today_categories if is_today else prior_categories

        for entity in {normalize_entity(e) for e in clean_entities(analysis.key_entities)}:
            entity_bucket[entity] += 1
        if analysis.category:
            category_bucket[analysis.category] += 1

    entity_deltas = [
        {
            "entity": entity,
            "today_count": today_count,
            "prior_count": prior_entities.get(entity, 0),
        }
        for entity, today_count in today_entities.items()
    ]
    entity_deltas.sort(key=lambda d: d["today_count"] - d["prior_count"], reverse=True)
    entity_deltas = entity_deltas[:TOP_N_ENTITIES]

    category_deltas = [
        {
            "category": category,
            "today_count": today_count,
            "prior_count": prior_categories.get(category, 0),
        }
        for category, today_count in today_categories.items()
    ]
    category_deltas.sort(key=lambda d: d["today_count"] - d["prior_count"], reverse=True)

    # "Emerging" is only meaningful against a baseline: with no prior-day data at all
    # (first run / fresh DB) every entity would be flagged, so emit nothing instead.
    if prior_entities:
        emerging_entities = sorted(
            (
                e for e, c in today_entities.items()
                if c > 0 and prior_entities.get(e, 0) == 0
                and e not in _EMERGING_STOPLIST
                and not _LEADING_DIGIT.match(e)
            ),
            key=lambda e: (-today_entities[e], e),
        )[:MAX_EMERGING_ENTITIES]
    else:
        emerging_entities = []

    return {
        "entity_deltas": entity_deltas,
        "category_deltas": category_deltas,
        "emerging_entities": emerging_entities,
    }
