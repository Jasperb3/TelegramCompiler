#!/usr/bin/env python
"""Report on — and optionally drain — the unanalysed-post backlog.

A development tool: never imported by the package, never exercised by pytest.
The default mode only reads the database.

    python scripts/drain_backlog.py                    # report the three buckets
    python scripts/drain_backlog.py --drain 500        # analyse 500 posts, media present only

The backlog splits three ways, because `main.purge_old_media()` deletes media
older than `storage.retention_days` while the posts themselves live forever:

  A  media still on disk         -> analysable in full, what --drain works on
  B  media purged, text >= 30    -> analysable text-only, degraded; left queued
  C  media purged, text < 30     -> nothing to analyse; process_unanalysed()
                                    tombstones these as category="Skipped"

--drain is scoped with --since (default: the earliest date under data/media/) so
it can only ever touch bucket A, and bounded by its post count so the run is a
known length. It needs a live LM Studio.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tg_compiler.analyzer import MIN_CONTENT_CHARS, Analyzer, _has_usable_media  # noqa: E402
from tg_compiler.config import load_config  # noqa: E402
from tg_compiler.db import Database, PostRecord  # noqa: E402

MEDIA_ROOT = Path("data/media")


def earliest_media_date() -> str | None:
    """The oldest date directory surviving purge_old_media(), as YYYY-MM-DD."""
    days = {d.name for channel in MEDIA_ROOT.glob("*") for d in channel.glob("*")}
    return min(days) if days else None


def bucket(post: PostRecord) -> str:
    if _has_usable_media(post):
        return "A"
    return "B" if len(post.text.strip()) >= MIN_CONTENT_CHARS else "C"


def report(posts: list[PostRecord]) -> None:
    buckets: dict[str, list[PostRecord]] = {"A": [], "B": [], "C": []}
    for post in posts:
        buckets[bucket(post)].append(post)

    labels = {
        "A": "media on disk         — drain (full analysis)",
        "B": "media purged, text    — queued (text-only would be degraded)",
        "C": "media purged, no text — tombstoned on the next unscoped run",
    }
    print(f"{len(posts)} unanalysed posts\n")
    for key in ("A", "B", "C"):
        group = buckets[key]
        span = ""
        if group:
            stamps = [p.timestamp for p in group]
            span = f"  {min(stamps).date()} .. {max(stamps).date()}"
        print(f"  {key}  {len(group):>7}  {labels[key]}{span}")


async def drain(cfg, db: Database, since: datetime, limit: int) -> None:
    channel_map = {c.channel_id: c for c in cfg.channels if c.channel_id}
    analysed, skipped = await Analyzer(cfg, db).process_unanalysed(
        channel_map, since=since, limit=limit
    )
    print(f"\n{analysed} analysed, {skipped} skipped")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--profile", default="batch", help="analysis profile to drain under")
    ap.add_argument("--drain", type=int, metavar="N",
                    help="analyse up to N posts (needs a live LM Studio)")
    ap.add_argument("--since", metavar="YYYY-MM-DD",
                    help="drain cutoff; defaults to the earliest date under data/media/")
    args = ap.parse_args()

    cfg = load_config(args.config).with_analysis_profile(args.profile)
    with Database(cfg.storage.db_path) as db:
        report(db.get_unanalysed_posts())
        if args.drain is None:
            return

        cutoff = args.since or earliest_media_date()
        if cutoff is None:
            sys.exit(f"No media under {MEDIA_ROOT} — pass --since explicitly to drain anyway.")
        since = datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc)
        print(f"\nDraining up to {args.drain} posts from {cutoff} "
              f"with profile {args.profile!r}...")
        asyncio.run(drain(cfg, db, since, args.drain))


if __name__ == "__main__":
    main()
