from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

load_dotenv()

from tg_compiler.config import AppConfig, ChannelConfig, load_config
from tg_compiler.db import Database
from tg_compiler.scraper import Scraper
from tg_compiler.utils import connect_telegram_client, secure_file

if TYPE_CHECKING:
    from tg_compiler.triage import BriefingContent

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HEARTBEAT_INTERVAL_SECS = 3600  # daemon logs stored/analysed counts on this cadence


def _share_pdf(config: AppConfig, pdf_path: Path) -> None:
    if not config.generation.share_to_directory:
        return
    dest_dir = Path(config.generation.share_to_directory)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / pdf_path.name
    shutil.copy2(pdf_path, dest)
    log.info("Copied briefing to %s", dest)


async def generate_daily_briefing(
    config: AppConfig,
    target_date: date,
    db: Database,
    posts_scraped: int = 0,
    posts_analysed: int = 0,
    posts_skipped: int = 0,
    layout: str | None = None,
) -> tuple[Path, "BriefingContent"]:
    """Triage the day's analysed posts, render the PDF/markdown briefing off the
    event loop, and return its path plus the BriefingContent used to render it."""
    from tg_compiler.generator import generate_briefing
    from tg_compiler.triage import triage as do_triage

    pairs = db.get_days_posts_with_analyses(target_date.isoformat())
    content = do_triage(pairs, config.triage, today=target_date,
                         channel_priorities=config.channel_priority_map(),
                         channel_credibilities=config.channel_credibility_map())
    content.channel_links = config.channel_link_map()
    content.posts_scraped = posts_scraped
    content.posts_analysed = posts_analysed
    content.posts_skipped = posts_skipped
    # Render off the event loop: markdown-pdf/PyMuPDF is synchronous and CPU-bound,
    # so running it inline would stall the daemon (live message handling + Telethon
    # keepalive) for the duration of the render.
    path = await asyncio.to_thread(
        generate_briefing, content, config.generation.output_dir, True,
        layout or config.generation.pdf_layout,
    )
    log.info("Briefing generated: %s", path)
    return path, content


async def run_batch(config: AppConfig, since_dt: datetime | None = None) -> None:
    from tg_compiler.analyzer import Analyzer
    from tg_compiler.synthesiser import run_analysis

    db = Database(config.storage.db_path)
    db.init_schema()
    today = datetime.now(timezone.utc).date()

    total_scraped = 0
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
    async with Scraper(config, db) as scraper:
        with logging_redirect_tqdm():
            for channel_cfg in tqdm(config.telegram.channels, desc="Scraping channels", unit="channel"):
                try:
                    posts = await scraper.scrape_channel(channel_cfg)
                    log.info("Scraped %d new posts from %s", len(posts), channel_cfg.slug)
                    total_scraped += len(posts)
                except Exception as e:
                    log.error("Scraping channel %s failed: %s", channel_cfg.slug, e)
        channel_map = scraper.channel_map

    analyzer = Analyzer(config, db)
    analysed_count, skipped_count = await analyzer.process_unanalysed(channel_map, since=since_dt)
    log.info("Analysed %d posts (skipped %d)", analysed_count, skipped_count)

    path, content = await generate_daily_briefing(
        config, today, db,
        posts_scraped=total_scraped, posts_analysed=analysed_count, posts_skipped=skipped_count,
    )
    await run_analysis(config, today, main_items=content.main_items)
    _share_pdf(config, path)
    removed = purge_old_media(config.storage.media_dir, config.storage.retention_days)
    log.info("Purged %d old media directories", removed)


def purge_old_media(media_dir: str, retention_days: int) -> int:
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=retention_days)).date()
    base = Path(media_dir)
    if not base.exists():
        return 0
    removed = 0
    for date_dir in base.rglob("????-??-??"):
        if date_dir.is_dir():
            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
                if dir_date < cutoff_date:
                    shutil.rmtree(date_dir)
                    removed += 1
            except ValueError:
                pass
    return removed


async def _run_daily_generation(config: AppConfig) -> None:
    """Run one scheduled generation cycle. Never raises — a transient failure
    (LM Studio down at the trigger time, a render error, a DB lock) is logged and
    swallowed so the scheduler loop survives and tries again the next day."""
    from tg_compiler.synthesiser import run_analysis

    today = datetime.now(timezone.utc).date()
    log.info("Scheduled daily generation starting for %s", today)
    db = Database(config.storage.db_path)
    try:
        db.init_schema()
        path, content = await generate_daily_briefing(config, today, db)
        await run_analysis(config, today, main_items=content.main_items)
        _share_pdf(config, path)
        removed = purge_old_media(config.storage.media_dir, config.storage.retention_days)
        log.info("Scheduled daily generation complete: %s (purged %d old media dirs)", path, removed)
    except Exception:
        log.exception("Scheduled daily generation failed for %s — will retry next cycle", today)
    finally:
        db.close()


async def schedule_daily_generation(config: AppConfig) -> None:
    import zoneinfo
    h, m = map(int, config.generation.generate_at.split(":"))
    # generate_at/timezone are validated at config load; this fallback is now
    # defensive only.
    try:
        tz = zoneinfo.ZoneInfo(config.generation.timezone)
    except Exception:
        log.warning("Unknown timezone %r — falling back to UTC", config.generation.timezone)
        tz = zoneinfo.ZoneInfo("UTC")

    utc_hour = datetime.now(tz).replace(hour=h, minute=m).astimezone(timezone.utc).hour
    if utc_hour < 3:
        log.warning(
            "generate_at %s (%s) converts to %02d:00 UTC — the briefing date is always "
            "the UTC calendar date, so this may generate a near-empty briefing for the "
            "new UTC day instead of the day you meant. See README's Daemon mode section.",
            config.generation.generate_at, tz.key, utc_hour,
        )

    while True:
        now = datetime.now(tz)
        target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        log.info("Next daily generation scheduled for %s (%s)", target.isoformat(), tz.key)
        await asyncio.sleep((target - now).total_seconds())
        # Defensive belt-and-braces: _run_daily_generation already swallows its
        # own errors, but never let an unexpected failure kill the scheduler.
        try:
            await _run_daily_generation(config)
        except Exception:
            log.exception("Daily generation cycle raised unexpectedly — scheduler continuing")


async def run_daemon(config: AppConfig) -> None:
    from openai import APIConnectionError
    from telethon import TelegramClient, events
    from telethon import utils as telethon_utils

    from tg_compiler.analyzer import Analyzer, analysis_to_record
    from tg_compiler.scraper import build_post_record

    db = Database(config.storage.db_path)
    db.init_schema()
    analyzer = Analyzer(config, db)
    analysis_sem = asyncio.Semaphore(config.lmstudio.max_concurrent_analyses)
    last_probe_failure: float | None = None
    PROBE_BACKOFF_SECS = 60
    stored_count = 0
    analysed_count = 0

    client = TelegramClient(
        config.telegram.session_name,
        config.telegram.api_id,
        config.telegram.api_hash,
    )
    await connect_telegram_client(client, config.telegram.session_name)
    secure_file(f"{config.telegram.session_name}.session")
    try:
        channel_entities = []
        channel_cfg_by_id: dict[int, ChannelConfig] = {}
        for ch in config.telegram.channels:
            identifier = ch.username or ch.id
            if not identifier:
                raise ValueError(f"Channel config has neither username nor id: {ch!r}")
            try:
                entity = await client.get_entity(identifier)
            except Exception as e:
                log.error("Cannot resolve channel %s — skipping for this daemon run: %s", ch.slug, e)
                continue
            channel_entities.append(entity)
            channel_cfg_by_id[telethon_utils.get_peer_id(entity)] = ch

        if not channel_entities:
            raise SystemExit("No configured channels could be resolved — daemon cannot start")

        @client.on(events.NewMessage(chats=channel_entities))
        async def handle_new_message(event):
            nonlocal last_probe_failure, stored_count, analysed_count
            msg = event.message
            channel_id = event.chat_id
            channel_cfg = channel_cfg_by_id.get(channel_id)
            if channel_cfg is None:
                log.warning("Received message from unmapped channel %s — skipping", channel_id)
                return
            record = await build_post_record(client, msg, channel_id, channel_cfg, config.storage)
            post_id = db.insert_post(record)
            # Advance the per-channel cursor so a later --batch resumes from here
            # instead of re-walking everything the daemon already captured. Guard
            # with max() so out-of-order live events never rewind it.
            if msg.id > db.get_last_seen_id(channel_id):
                db.set_last_seen_id(channel_id, msg.id)
            analysed = False
            if post_id is not None:
                record.id = post_id
                if (
                    last_probe_failure is not None
                    and time.monotonic() - last_probe_failure < PROBE_BACKOFF_SECS
                ):
                    log.debug("LM Studio recently unreachable — post %s left queued", msg.id)
                else:
                    try:
                        async with analysis_sem:
                            analysis = await analyzer.analyze_post(record, channel_cfg)
                        db.insert_analysis(
                            analysis_to_record(
                                post_id, analysis, config.lmstudio.model_for("analysis")
                            )
                        )
                        last_probe_failure = None
                        analysed = True
                    except Exception as e:
                        log.error("Analysis failed for post %s: %s", msg.id, e)
                        if isinstance(e, APIConnectionError):
                            last_probe_failure = time.monotonic()
            stored_count += 1
            if analysed:
                analysed_count += 1
            log.debug("Stored post %s from %s%s", msg.id, channel_cfg.slug, " (analysed)" if analysed else "")

        scheduler_task = asyncio.create_task(schedule_daily_generation(config))

        def _on_scheduler_done(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception() is not None:
                log.error("Daily generation scheduler crashed", exc_info=task.exception())

        scheduler_task.add_done_callback(_on_scheduler_done)

        async def _heartbeat() -> None:
            nonlocal stored_count, analysed_count
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECS)
                log.info("Daemon heartbeat: %d posts stored, %d analysed in the last hour",
                          stored_count, analysed_count)
                stored_count = 0
                analysed_count = 0

        heartbeat_task = asyncio.create_task(_heartbeat())

        def _on_heartbeat_done(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception() is not None:
                log.error("Daemon heartbeat task crashed", exc_info=task.exception())

        heartbeat_task.add_done_callback(_on_heartbeat_done)

        log.info("Daemon running on %d channels", len(channel_entities))
        await client.run_until_disconnected()
    finally:
        await client.disconnect()


def _parse_since(since_str: str) -> datetime:
    """Parse --since into a UTC datetime. Accepts HH:MM (today), YYYY-MM-DD, or YYYY-MM-DDTHH:MM."""
    now = datetime.now(timezone.utc)
    for fmt in ("%H:%M", "%Y-%m-%d", "%Y-%m-%dT%H:%M"):
        try:
            parsed = datetime.strptime(since_str, fmt)
            if fmt == "%H:%M":
                return now.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
            if fmt == "%Y-%m-%d":
                return parsed.replace(hour=0, minute=0, second=0, tzinfo=timezone.utc)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise SystemExit(f"Cannot parse --since value: {since_str!r}. Use HH:MM, YYYY-MM-DD, or YYYY-MM-DDTHH:MM")


def main() -> None:
    parser = argparse.ArgumentParser(prog="tg_compiler")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--analyse", action="store_true")
    parser.add_argument(
        "--since",
        metavar="TIME",
        help="Re-scrape from this point (HH:MM, YYYY-MM-DD, or YYYY-MM-DDTHH:MM). "
             "Resets channel cursors and overrides lookback_seconds.",
    )
    parser.add_argument(
        "--layout",
        choices=["desktop", "mobile"],
        default=None,
        help="PDF layout to use (default: config.generation.pdf_layout, falling back to 'desktop').",
    )
    args = parser.parse_args()

    if not (args.batch or args.daemon or args.generate or args.analyse):
        parser.print_help()
        return

    cfg = load_config(args.config, env_override=True)
    os.makedirs(cfg.storage.media_dir, exist_ok=True)

    if args.layout:
        cfg.generation.pdf_layout = args.layout

    since_dt = None
    if args.since:
        if not (args.batch or args.analyse):
            raise SystemExit("--since can only be used with --batch or --analyse")
        since_dt = _parse_since(args.since)
        if args.batch:
            now = datetime.now(timezone.utc)
            cfg.telegram.lookback_seconds = max(1, int((now - since_dt).total_seconds()))
            db = Database(cfg.storage.db_path)
            db.init_schema()
            db.reset_all_cursors()
            log.info("--since %s: lookback set to %ds, all channel cursors reset", args.since, cfg.telegram.lookback_seconds)

    if args.batch:
        asyncio.run(run_batch(cfg, since_dt))
    elif args.daemon:
        asyncio.run(run_daemon(cfg))
    elif args.analyse:
        from tg_compiler.synthesiser import run_analysis
        target_date = since_dt.date() if since_dt else datetime.now(timezone.utc).date()
        asyncio.run(run_analysis(cfg, target_date))
        date_dir = Path(cfg.generation.output_dir) / target_date.isoformat()
        pdfs = sorted(date_dir.glob("TheDailyTelegram_*.pdf")) if date_dir.exists() else []
        if pdfs:
            _share_pdf(cfg, pdfs[-1])
    elif args.generate:
        db = Database(cfg.storage.db_path)
        db.init_schema()
        out, _ = asyncio.run(generate_daily_briefing(cfg, datetime.now(timezone.utc).date(), db))
        _share_pdf(cfg, out)
        print(f"Generated: {out}")


if __name__ == "__main__":
    main()
