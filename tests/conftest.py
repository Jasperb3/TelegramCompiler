from datetime import datetime, timezone

import pytest

from tg_compiler.db import Database, PostRecord


@pytest.fixture
def db():
    database = Database(":memory:")
    database.init_schema()
    return database


@pytest.fixture
def sample_post():
    return PostRecord(
        channel_id=100,
        channel_name="test_chan",
        message_id=42,
        timestamp=datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc),
        text="Hello world",
        media_paths=[],
        has_images=False,
        raw_json="{}",
    )


from tg_compiler.db import AnalysisRecord


@pytest.fixture
def sample_analysis_record(sample_post, db):
    post_id = db.insert_post(sample_post)
    return AnalysisRecord(
        post_id=post_id,
        summary="Test summary",
        importance_score=3,
        urgency_score=2,
        credibility_score=4,
        relevance_score=3,
        category="Analysis",
        key_entities=["Alice"],
        model_used="test-model",
    )
