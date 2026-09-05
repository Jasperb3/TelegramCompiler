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

from tg_compiler.config import AppConfig, ChannelConfig, LMStudioConfig, TelegramConfig
from tg_compiler.scraper import Scraper


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


async def test_build_post_record_stub_message(tmp_path):
    from datetime import datetime, timezone

    from telethon.tl.types import Message, PeerChannel

    from tg_compiler.config import StorageConfig
    from tg_compiler.scraper import build_post_record

    class StubClient:
        parse_mode = None

        async def download_media(self, msg, file):
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            Path(file).write_bytes(b"jpg-bytes")

    channel_cfg = ChannelConfig(slug="test_chan", username="@test_chan")
    storage_cfg = StorageConfig(media_dir=str(tmp_path / "media"))
    msg = Message(
        id=42,
        peer_id=PeerChannel(999),
        date=datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc),
        message="hello world",
    )
    stub_client = StubClient()
    msg._client = stub_client

    record = await build_post_record(stub_client, msg, channel_id=-1000000000999,
                                      channel_cfg=channel_cfg, storage_cfg=storage_cfg)

    assert record.channel_id == -1000000000999
    assert record.channel_name == "test_chan"
    assert record.message_id == 42
    assert record.text == "hello world"
    assert record.media_paths == []
    assert record.has_images is False
    assert record.has_video is False
    assert record.raw_json == '{"id": 42, "text": "hello world"}'


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


async def test_scrape_channel_commits_posts_visible_to_other_connection(scraper_config, monkeypatch, tmp_path):
    from datetime import datetime, timezone

    from telethon.tl.types import Message, PeerChannel

    from tg_compiler.db import Database

    db_path = tmp_path / "scrape.db"
    db = Database(str(db_path))
    db.init_schema()
    channel_cfg = scraper_config.telegram.channels[0]
    scraper = Scraper(scraper_config, db)

    async def fake_get_entity(_):
        return PeerChannel(12345)

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

    await scraper.scrape_channel(channel_cfg)
    db.close()

    other_conn = Database(str(db_path))
    rows = other_conn._conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    other_conn.close()
    assert rows == 1


async def test_scrape_channel_commits_prior_posts_on_mid_iteration_exception(
    scraper_config, monkeypatch, tmp_path
):
    from datetime import datetime, timezone

    from telethon.tl.types import Message, PeerChannel

    from tg_compiler.db import Database

    db_path = tmp_path / "scrape_partial.db"
    db = Database(str(db_path))
    db.init_schema()
    channel_cfg = scraper_config.telegram.channels[0]
    scraper = Scraper(scraper_config, db)

    async def fake_get_entity(_):
        return PeerChannel(12345)

    def fake_iter_messages(_, **kwargs):
        async def _gen():
            yield Message(
                id=7,
                peer_id=PeerChannel(12345),
                date=datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc),
                message="first message",
            )
            raise ConnectionError("simulated mid-iteration failure")

        return _gen()

    monkeypatch.setattr(scraper._client, "get_entity", fake_get_entity)
    monkeypatch.setattr(scraper._client, "iter_messages", fake_iter_messages)

    await scraper.scrape_channel(channel_cfg)
    db.close()

    other_conn = Database(str(db_path))
    rows = other_conn._conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    other_conn.close()
    assert rows == 1


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


# --- media repair -------------------------------------------------------------

def _repair_post(db, scraper_config, tmp_path, message_id, media_paths, has_images=True):
    from datetime import datetime, timezone

    from tg_compiler.db import PostRecord

    rec = PostRecord(
        channel_id=-100123, channel_name="bad_chan", message_id=message_id,
        timestamp=datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc),
        text="a post with a picture", media_paths=media_paths,
        has_images=has_images, has_video=False, raw_json="{}",
    )
    rec.id = db.insert_post(rec)
    return rec


def _repair_scraper(db, scraper_config, monkeypatch, downloaded, messages=None, get_messages=None):
    from telethon.tl.types import PeerChannel

    scraper = Scraper(scraper_config, db)
    scraper.channel_map[-100123] = scraper_config.telegram.channels[0]

    async def fake_get_entity(_):
        return PeerChannel(channel_id=123)

    async def default_get_messages(entity, ids=None):
        return [messages.get(i) for i in ids]

    async def fake_download(msg, file=None):
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        Path(file).write_bytes(b"jpeg")
        downloaded.append(file)

    monkeypatch.setattr(scraper._client, "get_entity", fake_get_entity)
    monkeypatch.setattr(scraper._client, "get_messages", get_messages or default_get_messages)
    monkeypatch.setattr(scraper._client, "download_media", fake_download)
    return scraper


class _FakeMsg:
    def __init__(self, id):
        from datetime import datetime, timezone

        self.id = id
        self.date = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
        self.photo = object()


async def test_repair_missing_media_redownloads_and_updates_paths(
    db, scraper_config, tmp_path, monkeypatch
):
    scraper_config.storage.media_dir = str(tmp_path / "media")
    post = _repair_post(db, scraper_config, tmp_path, 7, [str(tmp_path / "gone.jpg")])
    downloaded = []
    scraper = _repair_scraper(db, scraper_config, monkeypatch, downloaded, {7: _FakeMsg(7)})

    assert await scraper.repair_missing_media([post]) == 1
    assert len(downloaded) == 1
    stored = db.get_post(post.id).media_paths
    assert stored == downloaded
    assert Path(stored[0]).exists()


async def test_repair_missing_media_retries_a_post_whose_download_never_succeeded(
    db, scraper_config, tmp_path, monkeypatch
):
    """build_post_record leaves has_images=True with no paths when a download
    permanently fails — that post is worth retrying."""
    scraper_config.storage.media_dir = str(tmp_path / "media")
    post = _repair_post(db, scraper_config, tmp_path, 8, [])
    downloaded = []
    scraper = _repair_scraper(db, scraper_config, monkeypatch, downloaded, {8: _FakeMsg(8)})

    assert await scraper.repair_missing_media([post]) == 1
    assert db.get_post(post.id).media_paths == downloaded


async def test_repair_missing_media_skips_posts_whose_files_are_present(
    db, scraper_config, tmp_path, monkeypatch
):
    present = tmp_path / "there.jpg"
    present.write_bytes(b"jpeg")
    post = _repair_post(db, scraper_config, tmp_path, 9, [str(present)])
    downloaded = []
    scraper = _repair_scraper(db, scraper_config, monkeypatch, downloaded, {9: _FakeMsg(9)})

    assert await scraper.repair_missing_media([post]) == 0
    assert downloaded == []


async def test_repair_missing_media_skips_a_deleted_message(
    db, scraper_config, tmp_path, monkeypatch
):
    scraper_config.storage.media_dir = str(tmp_path / "media")
    post = _repair_post(db, scraper_config, tmp_path, 10, [str(tmp_path / "gone.jpg")])
    downloaded = []
    scraper = _repair_scraper(db, scraper_config, monkeypatch, downloaded, {10: None})

    assert await scraper.repair_missing_media([post]) == 0
    assert downloaded == []
    assert db.get_post(post.id).media_paths == [str(tmp_path / "gone.jpg")]


async def test_repair_missing_media_honours_the_per_run_cap(
    db, scraper_config, tmp_path, monkeypatch
):
    import tg_compiler.scraper as scraper_mod

    scraper_config.storage.media_dir = str(tmp_path / "media")
    monkeypatch.setattr(scraper_mod, "MEDIA_REPAIR_MAX_PER_RUN", 2)
    posts = [
        _repair_post(db, scraper_config, tmp_path, i, [str(tmp_path / f"gone{i}.jpg")])
        for i in range(20, 25)
    ]
    downloaded = []
    scraper = _repair_scraper(
        db, scraper_config, monkeypatch, downloaded, {p.message_id: _FakeMsg(p.message_id) for p in posts}
    )

    assert await scraper.repair_missing_media(posts) == 2
    assert len(downloaded) == 2


async def test_repair_missing_media_survives_a_failing_channel(
    db, scraper_config, tmp_path, monkeypatch
):
    scraper_config.storage.media_dir = str(tmp_path / "media")
    post = _repair_post(db, scraper_config, tmp_path, 11, [str(tmp_path / "gone.jpg")])
    downloaded = []

    async def boom(entity, ids=None):
        raise RuntimeError("flood")

    scraper = _repair_scraper(db, scraper_config, monkeypatch, downloaded, get_messages=boom)

    assert await scraper.repair_missing_media([post]) == 0


async def test_repair_missing_media_ignores_posts_from_unknown_channels(
    db, scraper_config, tmp_path, monkeypatch
):
    post = _repair_post(db, scraper_config, tmp_path, 12, [str(tmp_path / "gone.jpg")])
    downloaded = []
    scraper = _repair_scraper(db, scraper_config, monkeypatch, downloaded, {12: _FakeMsg(12)})
    scraper.channel_map.clear()

    assert await scraper.repair_missing_media([post]) == 0
    assert downloaded == []
