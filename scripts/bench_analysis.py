#!/usr/bin/env python
"""Benchmark analysis throughput across models and batch sizes.

Needs a live LM Studio and a populated database; it is a development tool and is
never imported by the package or exercised by pytest. It only reads the DB.

    python scripts/bench_analysis.py --models prism-ml/bonsai-27b --batch-sizes 1,5,10
    python scripts/bench_analysis.py --models a,b --batch-sizes 10 --with-images

The post sample is drawn once with a fixed seed and cached in
scripts/bench_sample.json, so every model and batch size is scored on identical
input and runs stay comparable across sessions.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openai import LengthFinishReasonError, OpenAI  # noqa: E402

from tg_compiler import analyzer as A  # noqa: E402
from tg_compiler.config import load_config  # noqa: E402
from tg_compiler.db import Database  # noqa: E402

SAMPLE_FILE = Path(__file__).with_name("bench_sample.json")
SAMPLE_SEED = 20260903


def build_sample(db: Database, n_text: int, n_images: int) -> dict:
    """Draw a reproducible stratified sample of message ids."""
    posts = db.get_unanalysed_posts()
    text = [
        p for p in posts
        if not p.media_paths and 200 <= len(p.text.strip()) <= 1200
    ]
    media = [
        p for p in posts
        if p.media_paths and all(Path(x).exists() for x in p.media_paths)
        and len(p.text.strip()) >= A.MIN_CONTENT_CHARS
    ]
    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(text)
    rng.shuffle(media)
    return {
        "seed": SAMPLE_SEED,
        "text": [[p.channel_id, p.message_id] for p in text[:n_text]],
        "images": [[p.channel_id, p.message_id] for p in media[:n_images]],
    }


def load_sample(db: Database, n_text: int, n_images: int) -> dict:
    if not SAMPLE_FILE.exists():
        SAMPLE_FILE.write_text(json.dumps(build_sample(db, n_text, n_images), indent=1))
        print(f"Wrote a new sample to {SAMPLE_FILE}", file=sys.stderr)
    sample = json.loads(SAMPLE_FILE.read_text())
    by_key = {(p.channel_id, p.message_id): p for p in db.get_unanalysed_posts()}
    resolve = lambda keys: [by_key[tuple(k)] for k in keys if tuple(k) in by_key]  # noqa: E731
    return {"text": resolve(sample["text"]), "images": resolve(sample["images"])}


def _usage(completion) -> tuple[int, int, int]:
    u = completion.usage
    reasoning = getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0
    return u.prompt_tokens, u.completion_tokens, reasoning


def run_cell(client, cfg, model: str, posts: list, batch_size: int) -> dict:
    """Analyse `posts` at the given batch size and report timing and tokens."""
    cfg = cfg.model_copy(update={"model": model, "batch_size": batch_size,
                                 "batch_size_with_images": batch_size})
    batches = A.plan_batches(posts, cfg)
    wall = prompt_t = completion_t = reasoning_t = 0.0
    returned = aligned = 0
    finishes: dict[str, int] = {}
    analyses: list[dict] = []

    for batch in batches:
        single = len(batch) == 1
        if single:
            messages = A.build_messages(batch[0], A.SYSTEM_PROMPT)
            budget = A.compute_token_budget(batch[0], cfg)
            schema = A.PostAnalysis
        else:
            messages = A.build_batch_messages(batch, A.SYSTEM_PROMPT, cfg)
            budget = A.compute_batch_token_budget(batch, cfg)
            schema = A.BatchAnalysis

        started = time.time()
        try:
            completion = client.beta.chat.completions.parse(
                model=model, messages=messages, response_format=schema,
                temperature=cfg.temperature, max_tokens=budget,
            )
        except LengthFinishReasonError as e:
            # .parse() raises rather than returning a response cut off at
            # max_tokens; keep it so the cell reports what actually came back.
            completion = e.completion
            print(f"  batch of {len(batch)} hit the token limit — salvaging",
                  file=sys.stderr)
        except Exception as e:
            print(f"  batch of {len(batch)} failed: {e}", file=sys.stderr)
            continue
        wall += time.time() - started

        p_t, c_t, r_t = _usage(completion)
        prompt_t += p_t
        completion_t += c_t
        reasoning_t += r_t
        choice = completion.choices[0]
        finishes[choice.finish_reason] = finishes.get(choice.finish_reason, 0) + 1

        parsed = choice.message.parsed
        if parsed is None and not single:
            parsed = A.salvage_batch_items(choice.message.content or "")
        if parsed is None:
            continue
        items = [parsed] if single else list(parsed.analyses)
        returned += len(items)
        for pos, item in enumerate(items):
            post = batch[0] if single else batch[min(getattr(item, "index", pos + 1) - 1,
                                                     len(batch) - 1)]
            if single or A._opening_matches(getattr(item, "opening", ""), post):
                aligned += 1
            analyses.append({
                "message_id": post.message_id,
                "title": item.title, "summary": item.summary,
                "category": item.category, "threat_level": item.threat_level,
                "importance": item.importance_score, "urgency": item.urgency_score,
                "credibility": item.credibility_score, "relevance": item.relevance_score,
                "key_entities": item.key_entities,
            })

    n = len(posts)
    return {
        "model": model, "batch_size": batch_size, "posts": n, "returned": returned,
        "aligned": aligned,
        "wall": wall, "s_per_post": wall / n if n else 0.0,
        "prompt_tokens": prompt_t, "completion_tokens": completion_t,
        "reasoning_per_post": reasoning_t / n if n else 0.0,
        "finish_reasons": finishes, "analyses": analyses,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", required=True, help="comma-separated LM Studio model ids")
    ap.add_argument("--batch-sizes", default="1,5,10", help="comma-separated batch sizes")
    ap.add_argument("--n-text", type=int, default=10)
    ap.add_argument("--n-images", type=int, default=0)
    ap.add_argument("--with-images", action="store_true",
                    help="benchmark the image sample instead of the text sample")
    ap.add_argument("--out", default="", help="write the raw per-cell JSON here")
    args = ap.parse_args()

    config = load_config(args.config)
    db = Database(config.storage.db_path)
    try:
        sample = load_sample(db, args.n_text, max(args.n_images, 10 if args.with_images else 0))
    finally:
        db.close()
    posts = sample["images"] if args.with_images else sample["text"]
    if not posts:
        sys.exit("Sample is empty — delete scripts/bench_sample.json and retry.")

    lm = config.lmstudio
    client = OpenAI(base_url=f"http://{lm.server_host}:{lm.server_port}/v1",
                    api_key=lm.api_token or "lm-studio", timeout=3600, max_retries=0)

    cells = []
    for model in args.models.split(","):
        for size in (int(s) for s in args.batch_sizes.split(",")):
            print(f"running {model} @ batch {size} over {len(posts)} posts…", file=sys.stderr)
            cells.append(run_cell(client, lm, model.strip(), posts, size))

    print(f"\n{len(posts)} posts, {'with images' if args.with_images else 'text-only'}\n")
    print("| model | batch | s/post | returned | reasoning tok/post | completion tok | finish |")
    print("|---|---|---|---|---|---|---|")
    for c in cells:
        print(f"| {c['model']} | {c['batch_size']} | {c['s_per_post']:.1f} | "
              f"{c['returned']}/{c['posts']} | {c['reasoning_per_post']:.0f} | "
              f"{c['completion_tokens']:.0f} | {c['finish_reasons']} |")

    if len(cells) > 1:
        best = min(cells, key=lambda c: c["s_per_post"])
        worst = max(cells, key=lambda c: c["s_per_post"])
        print(f"\nfastest: {best['model']} @ batch {best['batch_size']} — "
              f"{worst['s_per_post'] / best['s_per_post']:.1f}x over the slowest cell")
        print(f"median s/post across cells: "
              f"{statistics.median(c['s_per_post'] for c in cells):.1f}")

    if args.out:
        Path(args.out).write_text(json.dumps(cells, indent=1))
        print(f"\nraw results → {args.out}")


if __name__ == "__main__":
    main()
