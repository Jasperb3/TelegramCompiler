#!/usr/bin/env python
"""Replay the image/summary numeric-consistency check over the live database.

A development tool: never imported by the package, never exercised by pytest.
Read-only — it opens the database with mode=ro, so it is safe alongside a
running --batch or --daemon.

    python scripts/audit_numeric_drops.py              # summary counts
    python scripts/audit_numeric_drops.py --show 40    # print flagged pairs

Only rows that still HAVE an image description can be replayed: a description
the rule dropped at analysis time was never written. The sample is therefore
biased towards survivors, and the count here is a floor on the true drop rate,
not a measurement of it. Its purpose is to expose the SHAPE of the collisions
so the false positives can be hand-classified.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tg_compiler.analyzer import _find_numeric_conflict  # noqa: E402

QUERY = """SELECT a.id, a.title, a.summary, a.image_insights, a.model_used
           FROM analyses a
           WHERE a.image_insights IS NOT NULL AND a.image_insights != ''"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/briefing.db")
    ap.add_argument("--show", type=int, default=20, help="how many flagged pairs to print")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = conn.execute(QUERY).fetchall()

    flagged = []
    by_model: dict[str, list[int]] = {}
    for row_id, title, summary, image_desc, model in rows:
        counts = by_model.setdefault(model or "(none)", [0, 0])
        counts[0] += 1
        conflict = _find_numeric_conflict(summary or "", image_desc)
        if conflict is not None:
            counts[1] += 1
            flagged.append((row_id, title, conflict))

    print(f"{len(flagged)} of {len(rows)} surviving image descriptions would be dropped\n")
    for model in sorted(by_model):
        total, hits = by_model[model]
        print(f"  {hits:>5} / {total:<6} {model}")

    if flagged and args.show:
        print()
        for row_id, title, (img_n, sum_n) in flagged[: args.show]:
            print(f"analyses.id={row_id}  {title!r}")
            print(f"  image   {img_n.value:g}  ...{img_n.context}...")
            print(f"  summary {sum_n.value:g}  ...{sum_n.context}...\n")


if __name__ == "__main__":
    main()
