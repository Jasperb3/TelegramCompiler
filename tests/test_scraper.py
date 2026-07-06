from pathlib import Path
from tg_compiler.scraper import media_path_for


def test_media_path_structure(tmp_path):
    path = media_path_for(
        base_dir=str(tmp_path),
        channel_slug="news",
        date_str="2026-06-07",
        message_id=42,
        ext="jpg",
    )
    assert path == str(tmp_path / "news" / "2026-06-07" / "42.jpg")


def test_media_path_creates_directories(tmp_path):
    path = media_path_for(
        base_dir=str(tmp_path),
        channel_slug="intel",
        date_str="2026-06-07",
        message_id=99,
        ext="png",
    )
    assert Path(path).parent.exists()


def test_different_channels_dont_collide(tmp_path):
    p1 = media_path_for(str(tmp_path), "chan_a", "2026-06-07", 1, "jpg")
    p2 = media_path_for(str(tmp_path), "chan_b", "2026-06-07", 1, "jpg")
    assert p1 != p2


import pytest
from tg_compiler.scraper import Scraper
from tg_compiler.config import AppConfig, TelegramConfig, LMStudioConfig, ChannelConfig


@pytest.fixture
def scraper_config(tmp_path):
    return AppConfig(
        telegram=TelegramConfig(
            api_id=1, api_hash="x", session_name=str(tmp_path / "session"),
            channels=[ChannelConfig(slug="bad_chan", username="@bad_chan")],
        ),
        lmstudio=LMStudioConfig(model="m"),
    )


async def test_scrape_channel_returns_empty_on_get_entity_failure(db, scraper_config, monkeypatch):
    channel_cfg = scraper_config.telegram.channels[0]
    scraper = Scraper(scraper_config, db)

    async def fake_get_entity(entity):
        raise ValueError("UsernameNotOccupiedError")

    monkeypatch.setattr(scraper._client, "get_entity", fake_get_entity)

    posts = await scraper.scrape_channel(channel_cfg)
    assert posts == []


async def test_scrape_channel_does_not_cap_iter_messages_limit(db, scraper_config, monkeypatch):
    from telethon.tl.types import PeerChannel

    channel_cfg = scraper_config.telegram.channels[0]
    scraper = Scraper(scraper_config, db)

    async def fake_get_entity(entity):
        return PeerChannel(channel_id=12345)

    captured_kwargs = {}

    def fake_iter_messages(entity, **kwargs):
        captured_kwargs.update(kwargs)

        async def _empty_gen():
            return
            yield

        return _empty_gen()

    monkeypatch.setattr(scraper._client, "get_entity", fake_get_entity)
    monkeypatch.setattr(scraper._client, "iter_messages", fake_iter_messages)

    await scraper.scrape_channel(channel_cfg)

    assert captured_kwargs["limit"] is None


async def test_scrape_channel_uses_marked_peer_id_for_post_and_cursor(db, scraper_config, monkeypatch):
    from datetime import datetime, timezone
    from telethon import utils as telethon_utils
    from telethon.tl.types import Message, PeerChannel

    channel_cfg = scraper_config.telegram.channels[0]
    scraper = Scraper(scraper_config, db)

    entity = PeerChannel(channel_id=12345)
    marked_id = telethon_utils.get_peer_id(entity)
    assert marked_id < 0  # marked channel ids carry the -100 prefix

    async def fake_get_entity(_):
        return entity

    def fake_iter_messages(_, **kwargs):
        async def _gen():
            yield Message(
                id=7,
                peer_id=PeerChannel(12345),
                date=datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc),
                message="hello",
            )

        return _gen()

    monkeypatch.setattr(scraper._client, "get_entity", fake_get_entity)
    monkeypatch.setattr(scraper._client, "iter_messages", fake_iter_messages)

    posts = await scraper.scrape_channel(channel_cfg)

    # Post is stored under the marked peer id (same representation the daemon uses),
    # so a later daemon insert of the same message collides on UNIQUE instead of duplicating.
    assert len(posts) == 1
    assert posts[0].channel_id == marked_id
    # Cursor is keyed by the same marked id and advanced to the highest message id.
    assert db.get_last_seen_id(marked_id) == 7
    # channel_map (consumed by the analyzer) is keyed by the same id.
    assert scraper.channel_map[marked_id] is channel_cfg


async def test_permanent_media_download_failure_keeps_has_images_and_cleans_partial_file(
    db, scraper_config, monkeypatch, tmp_path
):
    from datetime import datetime, timezone
    from telethon.tl.types import Message, PeerChannel

    class FakeMessage(Message):
        @property
        def photo(self):
            return True

    channel_cfg = scraper_config.telegram.channels[0]
    scraper_config.storage.media_dir = str(tmp_path / "media")
    scraper = Scraper(scraper_config, db)

    async def fake_get_entity(_):
        return PeerChannel(12345)

    def fake_iter_messages(_, **kwargs):
        async def _gen():
            yield FakeMessage(
                id=7,
                peer_id=PeerChannel(12345),
                date=datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc),
                message="short caption",
            )
        return _gen()

    async def failing_download_media(msg, file):
        # Simulate a partial write before the failure.
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        Path(file).write_bytes(b"partial")
        raise ConnectionError("network blip")

    monkeypatch.setattr(scraper._client, "get_entity", fake_get_entity)
    monkeypatch.setattr(scraper._client, "iter_messages", fake_iter_messages)
    monkeypatch.setattr(scraper._client, "download_media", failing_download_media)

    posts = await scraper.scrape_channel(channel_cfg)

    assert len(posts) == 1
    assert posts[0].media_paths == []
    assert posts[0].has_images is True
    dest = media_path_for(scraper_config.storage.media_dir, channel_cfg.slug, "2026-06-15", 7, "jpg")
    assert not Path(dest).exists()


async def test_daemon_and_batch_inserts_collide_on_unique(db):
    """A message scraped by batch (now marked id) and the same message from the
    daemon (event.chat_id, marked id) must be the same row, not two."""
    from datetime import datetime, timezone
    from tg_compiler.db import PostRecord

    marked_id = -1000000012345
    record = PostRecord(
        channel_id=marked_id, channel_name="chan", message_id=7,
        timestamp=datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc),
        text="hello", media_paths=[], has_images=False, raw_json="{}",
    )
    first = db.insert_post(record)
    second = db.insert_post(record)
    assert first is not None
    assert second is None  # UNIQUE(channel_id, message_id) blocks the duplicate
