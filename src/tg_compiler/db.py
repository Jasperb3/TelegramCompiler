from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from tg_compiler.utils import secure_file


@dataclass
class PostRecord:
    channel_id: int
    channel_name: str
    message_id: int
    timestamp: datetime
    text: str
    media_paths: list[str]
    has_images: bool
    raw_json: str
    id: int | None = None
    has_video: bool = False


@dataclass
class AnalysisRecord:
    post_id: int
    summary: str
    importance_score: int | None
    urgency_score: int | None
    credibility_score: int | None
    relevance_score: int | None
    category: str
    key_entities: list[str]
    model_used: str
    image_insights: str | None = None
    title: str = ""
    threat_level: str = "MODERATE"
    id: int | None = None


class Database:
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 10000")
        if db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            secure_file(db_path)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS posts (
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
            CREATE TABLE IF NOT EXISTS analyses (
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
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS channel_cursors (
                channel_id BIGINT PRIMARY KEY,
                last_seen_id BIGINT NOT NULL DEFAULT 0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS intel_assessments (
                date TEXT PRIMARY KEY,
                intel_json TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_analyses_post_id ON analyses(post_id);
            CREATE INDEX IF NOT EXISTS idx_posts_timestamp ON posts(timestamp);
        """)
        self._conn.commit()
        # Existing DBs may already have duplicate analyses per post from
        # overlapping batch/daemon runs before the UNIQUE index existed.
        self._conn.execute(
            "DELETE FROM analyses WHERE id NOT IN (SELECT MIN(id) FROM analyses GROUP BY post_id)"
        )
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_analyses_post_unique ON analyses(post_id)"
        )
        self._conn.commit()
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(analyses)").fetchall()
        }
        for col, defn in [
            ("title", "TEXT"),
            ("threat_level", "TEXT DEFAULT 'MODERATE'"),
        ]:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE analyses ADD COLUMN {col} {defn}")
                self._conn.commit()

        post_cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(posts)").fetchall()
        }
        if "has_video" not in post_cols:
            self._conn.execute("ALTER TABLE posts ADD COLUMN has_video BOOLEAN DEFAULT 0")
            self._conn.commit()

    def insert_post(self, post: PostRecord, commit: bool = True) -> int | None:
        try:
            cur = self._conn.execute(
                """INSERT INTO posts
                   (channel_id, channel_name, message_id, timestamp, text,
                    media_paths, has_images, has_video, raw_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (post.channel_id, post.channel_name, post.message_id,
                 post.timestamp.isoformat(), post.text,
                 json.dumps(post.media_paths), post.has_images, post.has_video, post.raw_json),
            )
            if commit:
                self._conn.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError as e:
            if "UNIQUE" not in str(e):
                raise
            return None

    def commit(self) -> None:
        self._conn.commit()

    def get_post(self, post_id: int) -> PostRecord:
        row = self._conn.execute(
            "SELECT * FROM posts WHERE id=?", (post_id,)
        ).fetchone()
        return _row_to_post(row)

    def get_last_seen_id(self, channel_id: int) -> int:
        row = self._conn.execute(
            "SELECT last_seen_id FROM channel_cursors WHERE channel_id=?",
            (channel_id,),
        ).fetchone()
        return row["last_seen_id"] if row else 0

    def reset_all_cursors(self) -> None:
        self._conn.execute("UPDATE channel_cursors SET last_seen_id=0")
        self._conn.commit()

    def set_last_seen_id(self, channel_id: int, message_id: int) -> None:
        self._conn.execute(
            """INSERT INTO channel_cursors(channel_id, last_seen_id)
               VALUES(?,?)
               ON CONFLICT(channel_id) DO UPDATE SET
                 last_seen_id=excluded.last_seen_id,
                 updated_at=CURRENT_TIMESTAMP""",
            (channel_id, message_id),
        )
        self._conn.commit()

    def get_unanalysed_posts(self) -> list[PostRecord]:
        rows = self._conn.execute(
            """SELECT p.* FROM posts p
               LEFT JOIN analyses a ON a.post_id = p.id
               WHERE a.id IS NULL"""
        ).fetchall()
        return [_row_to_post(r) for r in rows]

    def insert_analysis(self, rec: AnalysisRecord) -> int | None:
        cur = self._conn.execute(
            """INSERT OR IGNORE INTO analyses
               (post_id, title, summary, importance_score, urgency_score, credibility_score,
                relevance_score, category, key_entities, image_insights, model_used, threat_level)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rec.post_id, rec.title, rec.summary, rec.importance_score, rec.urgency_score,
             rec.credibility_score, rec.relevance_score, rec.category,
             json.dumps(rec.key_entities), rec.image_insights, rec.model_used, rec.threat_level),
        )
        self._conn.commit()
        return cur.lastrowid if cur.rowcount else None

    def get_days_posts_with_analyses(self, date_str: str) -> list[tuple[PostRecord, AnalysisRecord]]:
        start = date_str
        end = (date.fromisoformat(date_str) + timedelta(days=1)).isoformat()
        # Timestamps are uniform ISO-8601 UTC strings, so lexicographic
        # comparison against date bounds is exact and index-friendly.
        rows = self._conn.execute(
            """SELECT p.*, a.id as a_id, a.title, a.summary, a.importance_score, a.urgency_score,
                      a.credibility_score, a.relevance_score, a.category,
                      a.key_entities, a.image_insights, a.model_used, a.threat_level
               FROM posts p
               JOIN analyses a ON a.post_id = p.id
               WHERE p.timestamp >= ? AND p.timestamp < ?""",
            (start, end),
        ).fetchall()
        return _rows_to_pairs(rows)

    def get_posts_with_analyses_in_range(self, start_date_str: str, end_date_str: str) -> list[tuple[PostRecord, AnalysisRecord]]:
        start = start_date_str
        end = (date.fromisoformat(end_date_str) + timedelta(days=1)).isoformat()
        rows = self._conn.execute(
            """SELECT p.*, a.id as a_id, a.title, a.summary, a.importance_score, a.urgency_score,
                      a.credibility_score, a.relevance_score, a.category,
                      a.key_entities, a.image_insights, a.model_used, a.threat_level
               FROM posts p
               JOIN analyses a ON a.post_id = p.id
               WHERE p.timestamp >= ? AND p.timestamp < ?""",
            (start, end),
        ).fetchall()
        return _rows_to_pairs(rows)

    def save_intel_assessment(self, date_str: str, intel: dict) -> None:
        self._conn.execute(
            """INSERT INTO intel_assessments(date, intel_json) VALUES (?,?)
               ON CONFLICT(date) DO UPDATE SET
                 intel_json=excluded.intel_json,
                 created_at=CURRENT_TIMESTAMP""",
            (date_str, json.dumps(intel)),
        )
        self._conn.commit()

    def get_intel_assessment(self, date_str: str) -> dict | None:
        row = self._conn.execute(
            "SELECT intel_json FROM intel_assessments WHERE date=?", (date_str,)
        ).fetchone()
        return json.loads(row["intel_json"]) if row else None



def _rows_to_pairs(rows: list[sqlite3.Row]) -> list[tuple[PostRecord, AnalysisRecord]]:
    result = []
    for row in rows:
        post = _row_to_post(row)
        analysis = AnalysisRecord(
            id=row["a_id"],
            post_id=post.id,
            title=row["title"] or "",
            summary=row["summary"],
            importance_score=row["importance_score"],
            urgency_score=row["urgency_score"],
            credibility_score=row["credibility_score"],
            relevance_score=row["relevance_score"],
            category=row["category"],
            key_entities=json.loads(row["key_entities"] or "[]"),
            image_insights=row["image_insights"],
            model_used=row["model_used"],
            threat_level=row["threat_level"] or "MODERATE",
        )
        result.append((post, analysis))
    return result


def _row_to_post(row: sqlite3.Row) -> PostRecord:
    return PostRecord(
        id=row["id"],
        channel_id=row["channel_id"],
        channel_name=row["channel_name"],
        message_id=row["message_id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        text=row["text"] or "",
        media_paths=json.loads(row["media_paths"] or "[]"),
        has_images=bool(row["has_images"]),
        has_video=bool(row["has_video"]),
        raw_json=row["raw_json"] or "{}",
    )
