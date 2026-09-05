from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from telethon import TelegramClient
from telethon import utils as telethon_utils
from telethon.errors import (  # FloodWaitError: only fires if wait > flood_sleep_threshold
    ChannelPrivateError,
    FloodWaitError,
)
from telethon.tl.types import Message

from tg_compiler.config import AppConfig, ChannelConfig, StorageConfig
from tg_compiler.db import Database, PostRecord
from tg_compiler.utils import connect_telegram_client, secure_file

log = logging.getLogger(__name__)

# Ceiling on re-downloads per --batch run. The repair pass only touches posts that
# are both unanalysed and inside the analysis window, so steady state is a handful;
# the cap exists so the first run after adopting the feature can't turn a long-
# standing backlog of failed downloads into one enormous burst of API calls.
MEDIA_REPAIR_MAX_PER_RUN = 200
_MEDIA_REPAIR_IDS_PER_REQUEST = 100  # Telegram's own cap on get_messages(ids=[...])


def media_path_for(base_dir: str, channel_slug: str, date_str: str, message_id: int, ext: str) -> str:
    p = Path(base_dir) / channel_slug / date_str
    p.mkdir(parents=True, exist_ok=True)
    return str(p / f"{message_id}.{ext}")


async def build_post_record(
    client: TelegramClient,
    msg: Message,
    channel_id: int,
    channel_cfg: ChannelConfig,
    storage_cfg: StorageConfig,
) -> PostRecord:
    """Build a PostRecord from a Telethon message: downloads a photo (one retry,
    cleaning up any partial file on permanent failure), normalises the timestamp
    to aware UTC, and sets has_video/has_images. Shared by the batch scraper loop
    and the daemon's live message handler so the two paths can't diverge."""
    text = msg.text or ""
    has_video = bool(msg.video or msg.gif)
    media_paths: list[str] = []
    media_failed = False

    if msg.photo:
        date_str = msg.date.strftime("%Y-%m-%d")
        dest = media_path_for(storage_cfg.media_dir, channel_cfg.slug, date_str, msg.id, "jpg")
        try:
            await client.download_media(msg, file=dest)
            media_paths.append(dest)
        except Exception as e:
            log.warning("Media download failed for msg %s (attempt 1): %s", msg.id, e)
            try:
                await client.download_media(msg, file=dest)
                media_paths.append(dest)
            except Exception:
                log.error("Media download permanently failed for msg %s", msg.id)
                media_failed = True
                Path(dest).unlink(missing_ok=True)

    ts = msg.date
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    return PostRecord(
        channel_id=channel_id,
        channel_name=channel_cfg.slug,
        message_id=msg.id,
        timestamp=ts,
        text=text,
        media_paths=media_paths,
        # A permanently-failed download still had a photo — keep has_images True
        # so the analyzer's content gate doesn't treat a short-caption photo post
        # as text-only and skip it.
        has_images=bool(media_paths) or media_failed,
        has_video=has_video,
        raw_json=json.dumps({"id": msg.id, "text": text}),
    )


class Scraper:
    def __init__(self, config: AppConfig, db: Database):
        self._cfg = config
        self._db = db
        self._client = TelegramClient(
            config.telegram.session_name,
            config.telegram.api_id,
            config.telegram.api_hash,
            flood_sleep_threshold=600,
        )
        self.channel_map: dict[int, ChannelConfig] = {}

    async def __aenter__(self):
        await connect_telegram_client(self._client, self._cfg.telegram.session_name)
        secure_file(f"{self._cfg.telegram.session_name}.session")
        return self

    async def __aexit__(self, *_):
        await self._client.disconnect()

    async def repair_missing_media(self, posts: list[PostRecord]) -> int:
        """Re-download media for posts whose files are no longer on disk.

        Two things leave a post referencing media the analyzer cannot see: a
        download that permanently failed at scrape time (build_post_record keeps
        has_images True with no path), and purge_old_media() deleting a date
        directory the post still points at. Analysing such a post means analysing
        it blind on its text, so inside the analysis window it is worth one more
        fetch before falling back to text-only.

        Returns the number of posts whose media_paths were repaired. Failures are
        logged and skipped — a batch run must never abort here."""
        candidates = [
            p for p in posts
            # An empty media_paths with has_images set is the failed-download case,
            # so `not all(...)` on an empty list would wrongly exclude it.
            if p.has_images
            and (not p.media_paths or not all(Path(m).exists() for m in p.media_paths))
        ]
        if not candidates:
            return 0

        by_channel: dict[int, list[PostRecord]] = {}
        for post in candidates[:MEDIA_REPAIR_MAX_PER_RUN]:
            by_channel.setdefault(post.channel_id, []).append(post)

        repaired = 0
        for channel_id, channel_posts in by_channel.items():
            channel_cfg = self.channel_map.get(channel_id)
            if channel_cfg is None:
                # Only channels scraped this run are resolvable; a post from a
                # channel since removed from config has nothing to fetch against.
                continue
            try:
                entity = await self._client.get_entity(channel_cfg.username or channel_cfg.id)
                by_message_id = {p.message_id: p for p in channel_posts}
                ids = list(by_message_id)
                for start in range(0, len(ids), _MEDIA_REPAIR_IDS_PER_REQUEST):
                    chunk = ids[start:start + _MEDIA_REPAIR_IDS_PER_REQUEST]
                    for msg in await self._client.get_messages(entity, ids=chunk):
                        # None means Telegram no longer has the message.
                        if msg is None or not getattr(msg, "photo", None):
                            continue
                        post = by_message_id[msg.id]
                        dest = media_path_for(
                            self._cfg.storage.media_dir, channel_cfg.slug,
                            msg.date.strftime("%Y-%m-%d"), msg.id, "jpg",
                        )
                        try:
                            await self._client.download_media(msg, file=dest)
                        except Exception as e:
                            log.warning("Media repair failed for msg %s: %s", msg.id, e)
                            Path(dest).unlink(missing_ok=True)
                            continue
                        self._db.update_post_media_paths(post.id, [dest])
                        post.media_paths = [dest]
                        repaired += 1
                await asyncio.sleep(self._cfg.telegram.rate_limit_delay_ms / 1000)
            except Exception as e:
                log.warning("Media repair for channel %s failed: %s", channel_cfg.slug, e)

        if len(candidates) > MEDIA_REPAIR_MAX_PER_RUN:
            log.info(
                "Media repair capped at %d of %d candidates this run",
                MEDIA_REPAIR_MAX_PER_RUN, len(candidates),
            )
        return repaired

    async def scrape_channel(self, channel_cfg: ChannelConfig) -> list[PostRecord]:
        """Fetch and store new messages for one channel since its cursor, resolving
        the entity, downloading media, and advancing channel_cursors. Isolates its
        own failures — a bad channel returns whatever was collected so far instead
        of propagating, so one broken channel never aborts a batch run."""
        entity = channel_cfg.username or channel_cfg.id
        collected: list[PostRecord] = []
        channel_id: int | None = None
        last_seen: int | None = None
        max_id_seen: int | None = None

        try:
            channel_entity = await self._client.get_entity(entity)
            # Use the marked peer id (-100… form) so posts and cursors share the
            # daemon's representation (event.chat_id); a bare id would split the
            # same channel into two UNIQUE(channel_id, message_id) namespaces and
            # duplicate every message.
            channel_id = telethon_utils.get_peer_id(channel_entity)
            self.channel_map[channel_id] = channel_cfg
            last_seen = self._db.get_last_seen_id(channel_id)
            max_id_seen = last_seen

            lookback_date = (
                datetime.now(timezone.utc) - timedelta(seconds=self._cfg.telegram.lookback_seconds)
                if last_seen == 0 else None
            )
            # reverse=True walks oldest-to-newest from the cursor, so max_id_seen only
            # ever advances monotonically — a mid-iteration crash's cursor write covers
            # exactly the messages already inserted (no gap, no re-fetch of committed ones).
            iter_kwargs: dict = {"reverse": True, "limit": None}
            if last_seen > 0:
                iter_kwargs["offset_id"] = last_seen
            else:
                iter_kwargs["offset_date"] = lookback_date

            async for msg in self._client.iter_messages(channel_entity, **iter_kwargs):
                if not isinstance(msg, Message):
                    if msg.id > max_id_seen:
                        max_id_seen = msg.id
                    continue
                record = await build_post_record(
                    self._client, msg, channel_id, channel_cfg, self._cfg.storage
                )
                post_id = self._db.insert_post(record, commit=False)
                if post_id is not None:
                    record.id = post_id
                    collected.append(record)
                if msg.id > max_id_seen:
                    max_id_seen = msg.id

            await asyncio.sleep(self._cfg.telegram.rate_limit_delay_ms / 1000)

        except ChannelPrivateError:
            log.error("Channel %s is private or inaccessible", entity)
        except FloodWaitError as e:
            log.warning("FloodWait for %s exceeded threshold (%ds) — partial results saved", entity, e.seconds)
        except Exception as e:
            log.error("Scraping channel %s failed: %s", entity, e)
        finally:
            if max_id_seen is not None and max_id_seen > last_seen:
                self._db.set_last_seen_id(channel_id, max_id_seen)
            elif channel_id is not None:
                # set_last_seen_id above also commits pending post inserts on the
                # same connection; when the cursor didn't move, commit explicitly.
                self._db.commit()

        return collected
