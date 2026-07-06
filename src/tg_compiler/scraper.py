from __future__ import annotations
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

from telethon import TelegramClient
from telethon import utils as telethon_utils
from telethon.errors import ChannelPrivateError, FloodWaitError  # FloodWaitError: only fires if wait > flood_sleep_threshold
from telethon.tl.types import Message

from tg_compiler.config import AppConfig, ChannelConfig
from tg_compiler.db import Database, PostRecord
from tg_compiler.utils import connect_telegram_client, secure_file

log = logging.getLogger(__name__)


def media_path_for(base_dir: str, channel_slug: str, date_str: str, message_id: int, ext: str) -> str:
    p = Path(base_dir) / channel_slug / date_str
    p.mkdir(parents=True, exist_ok=True)
    return str(p / f"{message_id}.{ext}")


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

    async def scrape_channel(self, channel_cfg: ChannelConfig) -> list[PostRecord]:
        entity = channel_cfg.username or channel_cfg.id
        collected: list[PostRecord] = []

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
                text = msg.text or ""
                media_paths: list[str] = []
                has_video = bool(msg.video or msg.gif)

                media_failed = False
                if msg.photo:
                    date_str = msg.date.strftime("%Y-%m-%d")
                    dest = media_path_for(
                        self._cfg.storage.media_dir,
                        channel_cfg.slug,
                        date_str,
                        msg.id,
                        "jpg",
                    )
                    try:
                        await self._client.download_media(msg, file=dest)
                        media_paths.append(dest)
                    except Exception as e:
                        log.warning("Media download failed for msg %s (attempt 1): %s", msg.id, e)
                        try:
                            await self._client.download_media(msg, file=dest)
                            media_paths.append(dest)
                        except Exception:
                            log.error("Media download permanently failed for msg %s", msg.id)
                            media_failed = True
                            Path(dest).unlink(missing_ok=True)

                ts = msg.date
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                record = PostRecord(
                    channel_id=channel_id,
                    channel_name=channel_cfg.slug,
                    message_id=msg.id,
                    timestamp=ts,
                    text=text,
                    media_paths=media_paths,
                    # A permanently-failed download still had a photo — keep
                    # has_images True so the analyzer's content gate doesn't
                    # treat a short-caption photo post as text-only and skip it.
                    has_images=bool(media_paths) or media_failed,
                    has_video=has_video,
                    raw_json=json.dumps({"id": msg.id, "text": text}),
                )
                post_id = self._db.insert_post(record)
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
            if "max_id_seen" in locals() and max_id_seen > last_seen:
                self._db.set_last_seen_id(channel_id, max_id_seen)

        return collected
