#!/usr/bin/env python
"""Compare batched analysis against per-post analysis on the same posts.

Speed alone must not decide whether batching ships: the risk is that the model
blends adjacent posts together. This runs the identical sample twice — once at
batch size 1, once batched — and reports agreement plus a direct contamination
signal. Needs a live LM Studio; a development tool, never run by pytest.

    python scripts/eval_batch.py --batch-size 10 --n-text 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bench_analysis import compare_analyses, load_sample, run_cell  # noqa: E402
from openai import OpenAI  # noqa: E402

from tg_compiler import analyzer as A  # noqa: E402
from tg_compiler.config import load_config  # noqa: E402
from tg_compiler.db import Database  # noqa: E402
from tg_compiler.utils import normalize_entity  # noqa: E402


def count_leaked_entities(analyses: list[dict], posts: list, cfg) -> tuple[int, int]:
    """Entities credited to a post that appear only in one of its batch siblings.

    That is the shape contamination takes: an actor named in POST 3 turning up in
    POST 4's key_entities. Returns (leaked, total entities examined).
    """
    by_id = {p.message_id: p for p in posts}
    siblings: dict[int, list] = {}
    for batch in A.plan_batches(posts, cfg):
        for post in batch:
            siblings[post.message_id] = [p for p in batch if p.message_id != post.message_id]

    leaked = total = 0
    for row in analyses:
        post = by_id.get(row["message_id"])
        if post is None:
            continue
        own = post.text.lower()
        others = " ".join(p.text.lower() for p in siblings.get(post.message_id, []))
        for entity in row["key_entities"]:
            name = normalize_entity(entity)
            if not name:
                continue
            total += 1
            if name not in own and name in others:
                leaked += 1
    return leaked, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--n-text", type=int, default=20)
    ap.add_argument("--with-images", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    config = load_config(args.config)
    db = Database(config.storage.db_path)
    try:
        sample = load_sample(db, args.n_text, 10 if args.with_images else 0)
    finally:
        db.close()
    posts = sample["images"] if args.with_images else sample["text"]
    if not posts:
        sys.exit("Sample is empty — delete scripts/bench_sample.json and retry.")

    lm = config.lmstudio
    client = OpenAI(base_url=f"http://{lm.server_host}:{lm.server_port}/v1",
                    api_key=lm.api_token or "lm-studio", timeout=3600, max_retries=0)

    print(f"single-call baseline over {len(posts)} posts…", file=sys.stderr)
    single = run_cell(client, lm, lm.model, posts, 1)
    print(f"batched at {args.batch_size}…", file=sys.stderr)
    batched = run_cell(client, lm, lm.model, posts, args.batch_size)

    agreement = compare_analyses(single["analyses"], batched["analyses"])
    if agreement is None:
        sys.exit("No posts were analysed by both runs — nothing to compare.")

    batch_cfg = lm.model_copy(update={"batch_size": args.batch_size,
                                      "batch_size_with_images": args.batch_size})
    leak_b, total_b = count_leaked_entities(batched["analyses"], posts, batch_cfg)
    leak_a, total_a = count_leaked_entities(single["analyses"], posts, batch_cfg)

    empty = agreement["empty_summaries"]
    print(f"\nsingle vs batch({args.batch_size}) over {agreement['posts']} posts analysed by both")
    print(f"  coverage        : single {len(single['analyses'])}/{len(posts)}, "
          f"batched {len(batched['analyses'])}/{len(posts)}")
    print(f"  speed           : {single['s_per_post']:.1f} → {batched['s_per_post']:.1f} s/post "
          f"({single['s_per_post'] / max(batched['s_per_post'], 1e-9):.1f}x)")
    print(f"  category agree  : {agreement['category_agreement']:.0%}")
    print(f"  threat agree    : {agreement['threat_agreement']:.0%}")
    for name, delta in agreement["score_deltas"].items():
        print(f"  |Δ {name:<12}: {delta:.2f}")
    print(f"  empty summaries : single {empty['reference']}, batched {empty['candidate']}")
    print(f"  entity leakage  : batched {leak_b}/{total_b}, single (control) {leak_a}/{total_a}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"single": single, "batched": batched, "agreement": agreement,
             "leakage": {"batched": [leak_b, total_b], "single": [leak_a, total_a]}},
            indent=1,
        ))
        print(f"\nraw results → {args.out}")


if __name__ == "__main__":
    main()
