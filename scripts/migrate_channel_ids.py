#!/usr/bin/env python
"""One-off migration: canonicalise posts.channel_id to the marked peer id.

Background
----------
The batch scraper used to store channel_id as the *bare* Telegram id
(e.g. 1410429120) while the daemon stored the *marked* peer id
(e.g. -1001410429120). UNIQUE(channel_id, message_id) never collided across the
two, so the same message was inserted — and analysed — twice. The scraper now
uses the marked id everywhere (telethon.utils.get_peer_id); this script repairs
databases written before that fix.

What it does
------------
1. Backs up the DB to <db>.bak-<timestamp>.
2. For every post stored under a bare (positive) channel_id:
   - if the marked-id twin already exists, deletes the bare post + its analyses;
   - otherwise relabels the bare post's channel_id to the marked id.
3. Rebuilds channel_cursors keyed by the marked id, last_seen_id = MAX(message_id).

Idempotent: a second run finds no positive channel_ids and only refreshes cursors.

Usage
-----
    python scripts/migrate_channel_ids.py [path/to/briefing.db]   # default ./data/briefing.db
    python scripts/migrate_channel_ids.py --dry-run               # report only, no writes
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from telethon import utils as telethon_utils
from telethon.tl.types import PeerChannel


def marked_id(bare: int) -> int:
    """Marked peer id for a broadcast channel/supergroup bare id."""
    return telethon_utils.get_peer_id(PeerChannel(channel_id=bare))


def migrate(db_path: str, dry_run: bool = False) -> None:
    path = Path(db_path)
    if not path.exists():
        sys.exit(f"Database not found: {path}")

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    bare_rows = conn.execute(
        "SELECT id, channel_id, message_id FROM posts WHERE channel_id > 0"
    ).fetchall()
    print(f"posts total            : {conn.execute('SELECT COUNT(*) FROM posts').fetchone()[0]}")
    print(f"posts with bare id     : {len(bare_rows)}")

    if not bare_rows:
        print("No bare-id posts; only refreshing cursors.")

    if dry_run:
        # Predict collisions vs relabels without writing.
        relabel = collide = 0
        for row in bare_rows:
            target = marked_id(row["channel_id"])
            exists = conn.execute(
                "SELECT 1 FROM posts WHERE channel_id=? AND message_id=?",
                (target, row["message_id"]),
            ).fetchone()
            if exists:
                collide += 1
            else:
                relabel += 1
        print(f"[dry-run] would relabel: {relabel}")
        print(f"[dry-run] would delete (duplicate of marked twin): {collide}")
        conn.close()
        return

    backup = path.with_suffix(path.suffix + f".bak-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(path, backup)
    print(f"Backup written         : {backup}")

    relabel = collide = 0
    try:
        for row in bare_rows:
            target = marked_id(row["channel_id"])
            twin = conn.execute(
                "SELECT id FROM posts WHERE channel_id=? AND message_id=?",
                (target, row["message_id"]),
            ).fetchone()
            if twin:
                # Marked twin already present — drop the bare duplicate + its analyses.
                conn.execute("DELETE FROM analyses WHERE post_id=?", (row["id"],))
                conn.execute("DELETE FROM posts WHERE id=?", (row["id"],))
                collide += 1
            else:
                conn.execute(
                    "UPDATE posts SET channel_id=? WHERE id=?", (target, row["id"])
                )
                relabel += 1

        # Rebuild cursors from the now-canonical post ids.
        conn.execute("DELETE FROM channel_cursors")
        conn.execute(
            """INSERT INTO channel_cursors(channel_id, last_seen_id)
               SELECT channel_id, MAX(message_id) FROM posts GROUP BY channel_id"""
        )
        conn.commit()
    except Exception:
        conn.rollback()
        print("Migration failed — rolled back. DB unchanged (backup retained).")
        raise
    finally:
        conn.close()

    print(f"relabelled             : {relabel}")
    print(f"deleted duplicates     : {collide}")
    print("Cursors rebuilt. Done.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("db_path", nargs="?", default="./data/briefing.db")
    ap.add_argument("--dry-run", action="store_true", help="report only, no writes")
    args = ap.parse_args()
    migrate(args.db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
