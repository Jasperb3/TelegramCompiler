from datetime import datetime, timezone
import pytest
from tg_compiler.db import AnalysisRecord, Database, PostRecord


def test_insert_and_fetch_post(db, sample_post):
    post_id = db.insert_post(sample_post)
    assert post_id > 0
    fetched = db.get_post(post_id)
    assert fetched.message_id == 42
    assert fetched.text == "Hello world"
    assert fetched.channel_id == 100


def test_duplicate_post_is_ignored(db, sample_post):
    db.insert_post(sample_post)
    second_id = db.insert_post(sample_post)
    assert second_id is None


def test_insert_post_reraises_non_unique_integrity_error(db):
    import sqlite3
    from dataclasses import replace

    bad_post = replace(sample_post_factory(), channel_id=None)
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_post(bad_post)


def sample_post_factory():
    return PostRecord(
        channel_id=100,
        channel_name="test_chan",
        message_id=999,
        timestamp=datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc),
        text="Hello world",
        media_paths=[],
        has_images=False,
        raw_json="{}",
    )


def test_get_last_seen_id_defaults_to_zero(db):
    assert db.get_last_seen_id(channel_id=999) == 0


def test_set_and_get_last_seen_id(db):
    db.set_last_seen_id(channel_id=100, message_id=500)
    assert db.get_last_seen_id(channel_id=100) == 500


def test_get_unanalysed_posts(db, sample_post):
    post_id = db.insert_post(sample_post)
    unanalysed = db.get_unanalysed_posts()
    assert any(p.id == post_id for p in unanalysed)


def test_insert_analysis_and_joined_query(db, sample_post):
    post_id = db.insert_post(sample_post)
    rec = AnalysisRecord(
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
    a_id = db.insert_analysis(rec)
    assert a_id > 0

    results = db.get_days_posts_with_analyses("2026-06-07")
    assert len(results) == 1
    post, analysis = results[0]
    assert post.message_id == 42
    assert analysis.importance_score == 3
    assert "Alice" in analysis.key_entities


def test_get_intel_assessment_returns_none_when_missing(db):
    assert db.get_intel_assessment("2026-06-07") is None


def test_save_and_get_intel_assessment_round_trip(db):
    intel = {"situation_summary": "Calm.", "key_themes": []}
    db.save_intel_assessment("2026-06-07", intel)
    assert db.get_intel_assessment("2026-06-07") == intel


def test_save_intel_assessment_overwrites_on_conflict(db):
    db.save_intel_assessment("2026-06-07", {"situation_summary": "First."})
    db.save_intel_assessment("2026-06-07", {"situation_summary": "Second."})
    assert db.get_intel_assessment("2026-06-07")["situation_summary"] == "Second."


def _post(message_id, day):
    return PostRecord(
        channel_id=100, channel_name="test_chan", message_id=message_id,
        timestamp=datetime(2026, 6, day, 12, 0, tzinfo=timezone.utc),
        text="x", media_paths=[], has_images=False, raw_json="{}",
    )


def _record(post_id):
    return AnalysisRecord(
        post_id=post_id, summary="s", importance_score=3, urgency_score=2,
        credibility_score=4, relevance_score=3, category="Analysis",
        key_entities=[], model_used="test-model",
    )


def test_get_posts_with_analyses_in_range(db):
    for day, msg_id in [(5, 1), (7, 2), (9, 3)]:
        post_id = db.insert_post(_post(msg_id, day))
        db.insert_analysis(_record(post_id))

    results = db.get_posts_with_analyses_in_range("2026-06-06", "2026-06-08")
    assert [p.message_id for p, _ in results] == [2]


def test_wal_and_busy_timeout_enabled_for_file_db(tmp_path):
    db_path = tmp_path / "test.db"
    database = Database(str(db_path))
    database.init_schema()
    journal_mode = database._conn.execute("PRAGMA journal_mode").fetchone()[0]
    busy_timeout = database._conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert journal_mode == "wal"
    assert busy_timeout == 10000
    database.close()


def test_busy_timeout_enabled_for_memory_db(db):
    busy_timeout = db._conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert busy_timeout == 10000


def test_insert_analysis_enforces_one_per_post(db, sample_post):
    post_id = db.insert_post(sample_post)
    first_id = db.insert_analysis(_record(post_id))
    assert first_id is not None
    second_id = db.insert_analysis(_record(post_id))
    assert second_id is None
    results = db.get_days_posts_with_analyses("2026-06-07")
    assert len(results) == 1


def test_init_schema_dedupes_pre_existing_duplicate_analyses(tmp_path):
    db_path = tmp_path / "dupe.db"
    database = Database(str(db_path))
    # Build the schema without the unique index to simulate a pre-existing DB.
    database._conn.executescript("""
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id BIGINT NOT NULL,
            channel_name TEXT,
            message_id BIGINT NOT NULL,
            timestamp DATETIME NOT NULL,
            text TEXT,
            media_paths TEXT,
            has_images BOOLEAN,
            has_video BOOLEAN DEFAULT 0,
            raw_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(channel_id, message_id)
        );
        CREATE TABLE analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER REFERENCES posts(id),
            title TEXT,
            summary TEXT NOT NULL,
            importance_score INTEGER,
            urgency_score INTEGER,
            credibility_score INTEGER,
            relevance_score INTEGER,
            category TEXT,
            key_entities TEXT,
            image_insights TEXT,
            model_used TEXT,
            threat_level TEXT DEFAULT 'MODERATE',
            processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    database._conn.execute(
        """INSERT INTO posts (channel_id, channel_name, message_id, timestamp, text,
           media_paths, has_images, has_video, raw_json)
           VALUES (100, 'test_chan', 1, '2026-06-07T12:00:00+00:00', 'x', '[]', 0, 0, '{}')"""
    )
    post_id = database._conn.execute("SELECT id FROM posts").fetchone()[0]
    for summary in ("first", "second"):
        database._conn.execute(
            """INSERT INTO analyses (post_id, summary, category, key_entities, model_used)
               VALUES (?, ?, 'Analysis', '[]', 'test-model')""",
            (post_id, summary),
        )
    database._conn.commit()

    database.init_schema()

    rows = database._conn.execute("SELECT summary FROM analyses WHERE post_id=?", (post_id,)).fetchall()
    assert len(rows) == 1
    assert rows[0]["summary"] == "first"
    database.close()
