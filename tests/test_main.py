from datetime import datetime, timezone

import pytest

from tg_compiler import main as main_module
from tg_compiler.config import (
    AppConfig,
    ChannelConfig,
    LMStudioConfig,
    StorageConfig,
    TelegramConfig,
)
from tg_compiler.main import _parse_since, _share_pdf, purge_old_media

# ---------------------------------------------------------------------------
# _parse_since
# ---------------------------------------------------------------------------

def test_parse_since_time_of_day_today():
    result = _parse_since("00:00")
    now = datetime.now(timezone.utc)
    assert (result.year, result.month, result.day) == (now.year, now.month, now.day)
    assert result.hour == 0
    assert result.minute == 0
    assert result.tzinfo == timezone.utc


def test_parse_since_date_only():
    result = _parse_since("2026-06-01")
    assert result == datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)


def test_parse_since_date_and_time():
    result = _parse_since("2026-06-01T08:30")
    assert result == datetime(2026, 6, 1, 8, 30, tzinfo=timezone.utc)


def test_parse_since_invalid_raises_system_exit():
    with pytest.raises(SystemExit):
        _parse_since("not-a-date")


# ---------------------------------------------------------------------------
# purge_old_media
# ---------------------------------------------------------------------------

def test_purge_old_media_removes_old_dirs(tmp_path):
    old_dir = tmp_path / "chan" / "2020-01-01"
    new_dir = tmp_path / "chan" / "2099-01-01"
    other_dir = tmp_path / "chan" / "not-a-date"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)

    removed = purge_old_media(str(tmp_path), retention_days=30)

    assert removed == 1
    assert not old_dir.exists()
    assert new_dir.exists()
    assert other_dir.exists()


def test_purge_old_media_missing_base_dir_returns_zero(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert purge_old_media(str(missing), retention_days=30) == 0


def test_purge_old_media_uses_utc_date_boundary(tmp_path, monkeypatch):
    """Cutoff must be computed from the UTC date, not naive local time, so the
    retention boundary doesn't drift by the host's UTC offset."""
    import tg_compiler.main as main_module

    old_dir = tmp_path / "chan" / "2026-06-05"
    boundary_dir = tmp_path / "chan" / "2026-06-06"
    old_dir.mkdir(parents=True)
    boundary_dir.mkdir(parents=True)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 6, 7, 1, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(main_module, "datetime", FixedDatetime)

    removed = purge_old_media(str(tmp_path), retention_days=1)

    assert removed == 1
    assert not old_dir.exists()
    assert boundary_dir.exists()  # exactly at the cutoff date, not yet older than it


# ---------------------------------------------------------------------------
# _share_pdf
# ---------------------------------------------------------------------------

def test_share_pdf_copies_when_configured(tmp_path):
    config = AppConfig(
        telegram=TelegramConfig(api_id=1, api_hash="x", session_name=str(tmp_path / "session"), channels=[]),
        lmstudio=LMStudioConfig(model="m"),
    )
    config.generation.share_to_directory = str(tmp_path / "shared")

    pdf_path = tmp_path / "TheDailyTelegram_2026-06-13_120000.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    _share_pdf(config, pdf_path)

    dest = tmp_path / "shared" / pdf_path.name
    assert dest.exists()
    assert dest.read_bytes() == pdf_path.read_bytes()


def test_share_pdf_noop_when_not_configured(tmp_path):
    config = AppConfig(
        telegram=TelegramConfig(api_id=1, api_hash="x", session_name=str(tmp_path / "session"), channels=[]),
        lmstudio=LMStudioConfig(model="m"),
    )
    pdf_path = tmp_path / "TheDailyTelegram_2026-06-13_120000.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    _share_pdf(config, pdf_path)

    assert list(tmp_path.iterdir()) == [pdf_path]


@pytest.fixture
def batch_config(tmp_path):
    return AppConfig(
        telegram=TelegramConfig(
            api_id=1, api_hash="x", session_name=str(tmp_path / "session"),
            channels=[
                ChannelConfig(slug="chan_a", username="@chan_a"),
                ChannelConfig(slug="chan_b", username="@chan_b"),
                ChannelConfig(slug="chan_c", username="@chan_c"),
            ],
        ),
        lmstudio=LMStudioConfig(model="m"),
    )


async def test_run_batch_continues_after_one_channel_fails(tmp_path, batch_config, monkeypatch):
    batch_config.storage.db_path = str(tmp_path / "db.sqlite")

    scraped_channels = []

    class FakeScraper:
        def __init__(self, config, db):
            self.channel_map = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def scrape_channel(self, channel_cfg):
            if channel_cfg.slug == "chan_b":
                raise RuntimeError("boom")
            scraped_channels.append(channel_cfg.slug)
            return []

        async def repair_missing_media(self, posts):
            return 0

    class FakeAnalyzer:
        def __init__(self, config, db):
            pass

        async def process_unanalysed(self, channel_map=None, since=None):
            return 0, 0

    async def fake_generate_daily_briefing(config, today, db, **kwargs):
        from tg_compiler.triage import BriefingContent
        return "fake.pdf", BriefingContent(date=today, main_items=[], appendix_items=[])

    async def fake_run_analysis(config, today, main_items=None):
        return None

    monkeypatch.setattr(main_module, "Scraper", FakeScraper)
    monkeypatch.setattr("tg_compiler.analyzer.Analyzer", FakeAnalyzer)
    monkeypatch.setattr(main_module, "generate_daily_briefing", fake_generate_daily_briefing)
    monkeypatch.setattr("tg_compiler.synthesiser.run_analysis", fake_run_analysis)

    await main_module.run_batch(batch_config)

    assert scraped_channels == ["chan_a", "chan_c"]


async def test_run_batch_passes_since_dt_to_process_unanalysed(tmp_path, batch_config, monkeypatch):
    from datetime import datetime, timezone

    batch_config.storage.db_path = str(tmp_path / "db.sqlite")

    class FakeScraper:
        def __init__(self, config, db):
            self.channel_map = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def scrape_channel(self, channel_cfg):
            return []

        async def repair_missing_media(self, posts):
            return 0

    received_since = []

    class FakeAnalyzer:
        def __init__(self, config, db):
            pass

        async def process_unanalysed(self, channel_map=None, since=None):
            received_since.append(since)
            return 0, 0

    async def fake_generate_daily_briefing(config, today, db, **kwargs):
        from tg_compiler.triage import BriefingContent
        return "fake.pdf", BriefingContent(date=today, main_items=[], appendix_items=[])

    async def fake_run_analysis(config, today, main_items=None):
        return None

    monkeypatch.setattr(main_module, "Scraper", FakeScraper)
    monkeypatch.setattr("tg_compiler.analyzer.Analyzer", FakeAnalyzer)
    monkeypatch.setattr(main_module, "generate_daily_briefing", fake_generate_daily_briefing)
    monkeypatch.setattr("tg_compiler.synthesiser.run_analysis", fake_run_analysis)

    since_dt = datetime(2026, 6, 1, tzinfo=timezone.utc)
    await main_module.run_batch(batch_config, since_dt)

    assert received_since == [since_dt]


async def test_run_batch_purges_old_media(tmp_path, batch_config, monkeypatch):
    batch_config.storage.db_path = str(tmp_path / "db.sqlite")
    batch_config.storage.media_dir = str(tmp_path / "media")

    class FakeScraper:
        def __init__(self, config, db):
            self.channel_map = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def scrape_channel(self, channel_cfg):
            return []

        async def repair_missing_media(self, posts):
            return 0

    class FakeAnalyzer:
        def __init__(self, config, db):
            pass

        async def process_unanalysed(self, channel_map=None, since=None):
            return 0, 0

    async def fake_generate_daily_briefing(config, today, db, **kwargs):
        from tg_compiler.triage import BriefingContent
        return "fake.pdf", BriefingContent(date=today, main_items=[], appendix_items=[])

    async def fake_run_analysis(config, today, main_items=None):
        return None

    purge_calls = []

    def fake_purge(media_dir, retention_days):
        purge_calls.append((media_dir, retention_days))
        return 0

    monkeypatch.setattr(main_module, "Scraper", FakeScraper)
    monkeypatch.setattr("tg_compiler.analyzer.Analyzer", FakeAnalyzer)
    monkeypatch.setattr(main_module, "generate_daily_briefing", fake_generate_daily_briefing)
    monkeypatch.setattr("tg_compiler.synthesiser.run_analysis", fake_run_analysis)
    monkeypatch.setattr(main_module, "purge_old_media", fake_purge)

    await main_module.run_batch(batch_config)

    assert purge_calls == [(batch_config.storage.media_dir, batch_config.storage.retention_days)]


@pytest.fixture
def daemon_config(tmp_path):
    return AppConfig(
        telegram=TelegramConfig(
            api_id=1, api_hash="x", session_name=str(tmp_path / "session"),
            channels=[ChannelConfig(slug="chan_a", username="@chan_a")],
        ),
        lmstudio=LMStudioConfig(model="m"),
    )


async def test_run_daemon_logs_scheduler_crash(tmp_path, daemon_config, monkeypatch, caplog):
    import asyncio
    import logging

    import telethon
    from telethon.tl.types import PeerChannel

    daemon_config.storage.db_path = str(tmp_path / "db.sqlite")

    FakeEntity = PeerChannel

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            return None

        async def connect(self):
            return None

        async def is_user_authorized(self):
            return True

        async def get_entity(self, identifier):
            return FakeEntity(channel_id=1)

        def on(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

        async def run_until_disconnected(self):
            await asyncio.sleep(0.05)

        async def disconnect(self):
            return None

    async def fake_schedule_daily_generation(config):
        raise RuntimeError("scheduler boom")

    monkeypatch.setattr(telethon, "TelegramClient", FakeClient)
    monkeypatch.setattr(main_module, "schedule_daily_generation", fake_schedule_daily_generation)

    with caplog.at_level(logging.ERROR):
        await main_module.run_daemon(daemon_config)

    assert any("Daily generation scheduler crashed" in r.message for r in caplog.records)


async def test_run_daemon_maps_channel_by_marked_peer_id(tmp_path, daemon_config, monkeypatch, caplog):
    import asyncio
    import logging

    import telethon
    from telethon import utils as telethon_utils
    from telethon.tl.types import PeerChannel

    daemon_config.storage.db_path = str(tmp_path / "db.sqlite")

    entity = PeerChannel(channel_id=12345)
    marked_id = telethon_utils.get_peer_id(entity)

    handlers = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            return None

        async def connect(self):
            return None

        async def is_user_authorized(self):
            return True

        async def get_entity(self, identifier):
            return entity

        def on(self, *args, **kwargs):
            def decorator(fn):
                handlers.append(fn)
                return fn
            return decorator

        async def run_until_disconnected(self):
            await asyncio.sleep(0.05)

        async def disconnect(self):
            return None

    async def fake_schedule_daily_generation(config):
        return None

    class FakeAnalyzer:
        def __init__(self, config, db):
            pass

        async def analyze_post(self, record, channel_cfg):
            from tg_compiler.analyzer import PostAnalysis
            return PostAnalysis(
                title="Title", summary="Summary",
                importance_score=1, urgency_score=1, credibility_score=1, relevance_score=1,
                category="Other", key_entities=[], image_description=None,
                threat_level="LOW",
            )

    monkeypatch.setattr(telethon, "TelegramClient", FakeClient)
    monkeypatch.setattr(main_module, "schedule_daily_generation", fake_schedule_daily_generation)
    monkeypatch.setattr("tg_compiler.analyzer.Analyzer", FakeAnalyzer)

    with caplog.at_level(logging.WARNING):
        await main_module.run_daemon(daemon_config)

    assert handlers, "handle_new_message was not registered"
    handler = handlers[0]

    class FakeMessage:
        id = 1
        text = "hello"
        date = datetime.now(timezone.utc)
        photo = None
        video = None
        gif = None

    class FakeEvent:
        chat_id = marked_id
        message = FakeMessage()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        await handler(FakeEvent())

    assert not any("unmapped channel" in r.message for r in caplog.records)


async def test_run_daemon_logs_stored_post_debug_line(tmp_path, daemon_config, monkeypatch, caplog):
    import asyncio
    import logging

    import telethon
    from telethon import utils as telethon_utils
    from telethon.tl.types import PeerChannel

    daemon_config.storage.db_path = str(tmp_path / "db.sqlite")

    entity = PeerChannel(channel_id=12345)
    marked_id = telethon_utils.get_peer_id(entity)

    handlers = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            return None

        async def connect(self):
            return None

        async def is_user_authorized(self):
            return True

        async def get_entity(self, identifier):
            return entity

        def on(self, *args, **kwargs):
            def decorator(fn):
                handlers.append(fn)
                return fn
            return decorator

        async def run_until_disconnected(self):
            await asyncio.sleep(0.05)

        async def disconnect(self):
            return None

    async def fake_schedule_daily_generation(config):
        return None

    class FakeAnalyzer:
        def __init__(self, config, db):
            pass

        async def process_unanalysed(self, channel_map=None, since=None):
            return 0, 0

    monkeypatch.setattr(telethon, "TelegramClient", FakeClient)
    monkeypatch.setattr(main_module, "schedule_daily_generation", fake_schedule_daily_generation)
    monkeypatch.setattr("tg_compiler.analyzer.Analyzer", FakeAnalyzer)

    await main_module.run_daemon(daemon_config)
    handler = handlers[0]

    class FakeMessage:
        id = 1
        text = "hello"
        date = datetime.now(timezone.utc)
        photo = None
        video = None
        gif = None

    class FakeEvent:
        chat_id = marked_id
        message = FakeMessage()

    with caplog.at_level(logging.DEBUG):
        await handler(FakeEvent())

    # The daemon stores on arrival and analyses in the periodic sweep, so the
    # arrival line no longer claims the post was analysed.
    assert any("Stored post 1 from chan_a" in r.message for r in caplog.records)
    assert not any("(analysed)" in r.message for r in caplog.records)

    from tg_compiler.db import Database
    db = Database(daemon_config.storage.db_path)
    db.init_schema()
    assert [p.message_id for p in db.get_unanalysed_posts()] == [1]
    db.close()


async def test_run_daemon_heartbeat_counts_stored_and_analysed(tmp_path, daemon_config, monkeypatch, caplog):
    import asyncio
    import logging

    import telethon
    from telethon import utils as telethon_utils
    from telethon.tl.types import PeerChannel

    daemon_config.storage.db_path = str(tmp_path / "db.sqlite")

    entity = PeerChannel(channel_id=12345)
    marked_id = telethon_utils.get_peer_id(entity)

    handlers = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            return None

        async def connect(self):
            return None

        async def is_user_authorized(self):
            return True

        async def get_entity(self, identifier):
            return entity

        def on(self, *args, **kwargs):
            def decorator(fn):
                handlers.append(fn)
                return fn
            return decorator

        async def run_until_disconnected(self):
            await asyncio.sleep(0.02)

        async def disconnect(self):
            return None

    async def fake_schedule_daily_generation(config):
        return None

    class FakeAnalyzer:
        def __init__(self, config, db):
            pass

        swept = {"done": False}

        async def process_unanalysed(self, channel_map=None, since=None):
            # the first sweep clears the three stored posts; later sweeps find none
            if self.swept["done"]:
                return 0, 0
            self.swept["done"] = True
            return 3, 0

    monkeypatch.setattr(telethon, "TelegramClient", FakeClient)
    monkeypatch.setattr(main_module, "schedule_daily_generation", fake_schedule_daily_generation)
    monkeypatch.setattr("tg_compiler.analyzer.Analyzer", FakeAnalyzer)
    monkeypatch.setattr(main_module, "HEARTBEAT_INTERVAL_SECS", 0.05)
    monkeypatch.setattr(main_module, "DAEMON_ANALYSIS_INTERVAL_SECS", 0.01)

    await main_module.run_daemon(daemon_config)
    handler = handlers[0]

    for i in range(1, 4):
        class FakeMessage:
            id = i
            text = "hello"
            date = datetime.now(timezone.utc)
            photo = None
            video = None
            gif = None

        class FakeEvent:
            chat_id = marked_id
            message = FakeMessage()

        await handler(FakeEvent())

    with caplog.at_level(logging.INFO):
        await asyncio.sleep(0.05)

    assert any(
        "Daemon heartbeat: 3 posts stored, 3 analysed in the last hour" in r.message
        for r in caplog.records
    )


async def test_run_daemon_advances_cursor(tmp_path, daemon_config, monkeypatch):
    """Daemon must advance channel_cursors so a later --batch doesn't re-walk the
    window the daemon already captured."""
    import asyncio

    import telethon
    from telethon import utils as telethon_utils
    from telethon.tl.types import PeerChannel

    from tg_compiler.db import Database

    daemon_config.storage.db_path = str(tmp_path / "db.sqlite")
    entity = PeerChannel(channel_id=12345)
    marked_id = telethon_utils.get_peer_id(entity)

    handlers = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            return None

        async def connect(self):
            return None

        async def is_user_authorized(self):
            return True

        async def get_entity(self, identifier):
            return entity

        def on(self, *args, **kwargs):
            def decorator(fn):
                handlers.append(fn)
                return fn
            return decorator

        async def run_until_disconnected(self):
            await asyncio.sleep(0.05)

        async def disconnect(self):
            return None

    async def fake_schedule_daily_generation(config):
        return None

    class FakeAnalyzer:
        def __init__(self, config, db):
            pass

        async def analyze_post(self, record, channel_cfg):
            from tg_compiler.analyzer import PostAnalysis
            return PostAnalysis(
                title="Title", summary="Summary",
                importance_score=1, urgency_score=1, credibility_score=1, relevance_score=1,
                category="Other", key_entities=[], image_description=None,
                threat_level="LOW",
            )

    monkeypatch.setattr(telethon, "TelegramClient", FakeClient)
    monkeypatch.setattr(main_module, "schedule_daily_generation", fake_schedule_daily_generation)
    monkeypatch.setattr("tg_compiler.analyzer.Analyzer", FakeAnalyzer)

    await main_module.run_daemon(daemon_config)
    handler = handlers[0]

    class FakeMessage:
        def __init__(self, mid):
            self.id = mid
            self.text = "hello"
            self.date = datetime.now(timezone.utc)
            self.photo = None
            self.video = None
            self.gif = None

    class FakeEvent:
        def __init__(self, mid):
            self.chat_id = marked_id
            self.message = FakeMessage(mid)

    await handler(FakeEvent(10))
    await handler(FakeEvent(8))   # out-of-order: must not rewind the cursor

    db = Database(daemon_config.storage.db_path)
    db.init_schema()
    assert db.get_last_seen_id(marked_id) == 10
    db.close()


# ---------------------------------------------------------------------------
# schedule_daily_generation — resilience (Issue 1)
# ---------------------------------------------------------------------------

async def test_run_daily_generation_swallows_errors(tmp_path, daemon_config, monkeypatch, caplog):
    """A failed generation cycle must be logged and swallowed, never raised —
    otherwise the scheduler loop dies and no report is ever produced again."""
    import logging

    daemon_config.storage.db_path = str(tmp_path / "db.sqlite")

    async def boom(config, today, db, **kwargs):
        raise RuntimeError("LM Studio down / db locked")

    monkeypatch.setattr(main_module, "generate_daily_briefing", boom)

    with caplog.at_level(logging.ERROR):
        await main_module._run_daily_generation(daemon_config)  # must not raise

    assert any("generation" in r.message.lower() and r.levelno >= logging.ERROR
               for r in caplog.records)


async def test_scheduler_loop_survives_failed_cycle(daemon_config, monkeypatch, caplog):
    """One failing cycle must not terminate the loop; it logs and proceeds to
    schedule the next day."""
    import asyncio
    import logging

    daemon_config.generation.generate_at = "00:00"
    daemon_config.generation.timezone = "UTC"

    cycles = []

    class _StopLoop(Exception):
        pass

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise _StopLoop  # break out after the loop has gone round twice

    async def failing_cycle(config):
        cycles.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(main_module, "_run_daily_generation", failing_cycle)

    with caplog.at_level(logging.INFO):
        with pytest.raises(_StopLoop):
            await main_module.schedule_daily_generation(daemon_config)

    assert len(cycles) == 1            # the cycle ran and raised
    assert len(sleeps) == 2            # loop survived and scheduled the next day
    assert any("next" in r.message.lower() for r in caplog.records)  # arm-time visibility


async def test_generate_daily_briefing_offloads_pdf_render(tmp_path, daemon_config, monkeypatch):
    """PDF render must run via asyncio.to_thread so the daemon event loop keeps
    servicing Telegram messages during generation."""
    import asyncio

    from tg_compiler.db import Database

    daemon_config.storage.db_path = str(tmp_path / "db.sqlite")
    db = Database(daemon_config.storage.db_path)
    db.init_schema()

    def fake_generate_briefing(content, output_dir, pdf=False, layout="desktop"):
        return f"{output_dir}/out.pdf"

    to_thread_calls = []
    real_to_thread = asyncio.to_thread

    async def spy_to_thread(fn, *args, **kwargs):
        to_thread_calls.append(fn)
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr("tg_compiler.generator.generate_briefing", fake_generate_briefing)
    monkeypatch.setattr(asyncio, "to_thread", spy_to_thread)

    path, content = await main_module.generate_daily_briefing(
        daemon_config, datetime.now(timezone.utc).date(), db
    )
    db.close()

    assert str(path).endswith("out.pdf")
    assert fake_generate_briefing in to_thread_calls  # rendered off the event loop


async def test_run_daemon_skips_unresolvable_channel_and_continues(tmp_path, monkeypatch, caplog):
    import asyncio
    import logging

    import telethon
    from telethon.tl.types import PeerChannel

    config = AppConfig(
        telegram=TelegramConfig(
            api_id=1, api_hash="x", session_name=str(tmp_path / "session"),
            channels=[
                ChannelConfig(slug="chan_bad", username="@chan_bad"),
                ChannelConfig(slug="chan_good", username="@chan_good"),
            ],
        ),
        lmstudio=LMStudioConfig(model="m"),
    )
    config.storage.db_path = str(tmp_path / "db.sqlite")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            return None

        async def connect(self):
            return None

        async def is_user_authorized(self):
            return True

        async def get_entity(self, identifier):
            if identifier == "@chan_bad":
                raise ValueError("channel not found")
            return PeerChannel(channel_id=1)

        def on(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

        async def run_until_disconnected(self):
            await asyncio.sleep(0.05)

        async def disconnect(self):
            return None

    async def fake_schedule_daily_generation(cfg):
        return None

    monkeypatch.setattr(telethon, "TelegramClient", FakeClient)
    monkeypatch.setattr(main_module, "schedule_daily_generation", fake_schedule_daily_generation)

    with caplog.at_level(logging.ERROR):
        await main_module.run_daemon(config)

    assert any("Cannot resolve channel chan_bad" in r.message for r in caplog.records)


async def test_run_daemon_exits_when_all_channels_unresolvable(tmp_path, monkeypatch):
    import telethon

    config = AppConfig(
        telegram=TelegramConfig(
            api_id=1, api_hash="x", session_name=str(tmp_path / "session"),
            channels=[ChannelConfig(slug="chan_bad", username="@chan_bad")],
        ),
        lmstudio=LMStudioConfig(model="m"),
    )
    config.storage.db_path = str(tmp_path / "db.sqlite")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            return None

        async def connect(self):
            return None

        async def is_user_authorized(self):
            return True

        async def get_entity(self, identifier):
            raise ValueError("channel not found")

        def on(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

        async def disconnect(self):
            return None

    monkeypatch.setattr(telethon, "TelegramClient", FakeClient)

    with pytest.raises(SystemExit):
        await main_module.run_daemon(config)


async def test_run_daemon_sweep_scopes_analysis_to_daemon_uptime(tmp_path, daemon_config, monkeypatch):
    """The sweep must pass `since`, or its first tick would try to analyse every
    unanalysed post ever stored (~92k in the live database) with the daemon's
    slower model. --batch stays the tool for historical catch-up."""
    import asyncio
    from datetime import datetime, timedelta, timezone

    import telethon
    from telethon import utils as telethon_utils
    from telethon.tl.types import PeerChannel

    daemon_config.storage.db_path = str(tmp_path / "db.sqlite")
    entity = PeerChannel(channel_id=12345)
    marked_id = telethon_utils.get_peer_id(entity)
    handlers = []
    calls = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            return None

        async def connect(self):
            return None

        async def is_user_authorized(self):
            return True

        async def get_entity(self, identifier):
            return entity

        def on(self, *args, **kwargs):
            def decorator(fn):
                handlers.append(fn)
                return fn
            return decorator

        async def run_until_disconnected(self):
            await asyncio.sleep(0.03)

        async def disconnect(self):
            return None

    async def fake_schedule_daily_generation(cfg):
        return None

    class FakeAnalyzer:
        def __init__(self, config, db):
            pass

        async def process_unanalysed(self, channel_map=None, since=None):
            calls.append({"channel_map": channel_map, "since": since})
            return 0, 0

    monkeypatch.setattr(telethon, "TelegramClient", FakeClient)
    monkeypatch.setattr(main_module, "schedule_daily_generation", fake_schedule_daily_generation)
    monkeypatch.setattr("tg_compiler.analyzer.Analyzer", FakeAnalyzer)
    monkeypatch.setattr(main_module, "DAEMON_ANALYSIS_INTERVAL_SECS", 0.01)

    before = datetime.now(timezone.utc) - timedelta(seconds=1)
    await main_module.run_daemon(daemon_config)
    await asyncio.sleep(0.03)

    assert calls, "the analysis sweep never ran"
    since = calls[0]["since"]
    assert since is not None, "sweep ran unscoped — it would attack the whole backlog"
    assert since >= before
    # the channel map is passed so per-channel custom prompts still apply
    assert marked_id in calls[0]["channel_map"]


async def test_run_daemon_survives_a_failing_sweep(tmp_path, daemon_config, monkeypatch, caplog):
    """A sweep that raises must be logged and the loop must continue — the daemon
    runs for days and one bad cycle (LM Studio restarting, a DB lock) must not
    silently end analysis for the rest of its life."""
    import asyncio
    import logging

    import telethon
    from telethon.tl.types import PeerChannel

    daemon_config.storage.db_path = str(tmp_path / "db.sqlite")
    entity = PeerChannel(channel_id=12345)
    handlers = []
    attempts = {"n": 0}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            return None

        async def connect(self):
            return None

        async def is_user_authorized(self):
            return True

        async def get_entity(self, identifier):
            return entity

        def on(self, *args, **kwargs):
            def decorator(fn):
                handlers.append(fn)
                return fn
            return decorator

        async def run_until_disconnected(self):
            await asyncio.sleep(0.03)

        async def disconnect(self):
            return None

    async def fake_schedule_daily_generation(cfg):
        return None

    class FakeAnalyzer:
        def __init__(self, config, db):
            pass

        async def process_unanalysed(self, channel_map=None, since=None):
            attempts["n"] += 1
            raise RuntimeError("LM Studio went away")

    monkeypatch.setattr(telethon, "TelegramClient", FakeClient)
    monkeypatch.setattr(main_module, "schedule_daily_generation", fake_schedule_daily_generation)
    monkeypatch.setattr("tg_compiler.analyzer.Analyzer", FakeAnalyzer)
    monkeypatch.setattr(main_module, "DAEMON_ANALYSIS_INTERVAL_SECS", 0.01)

    with caplog.at_level(logging.ERROR):
        await main_module.run_daemon(daemon_config)
        await asyncio.sleep(0.05)

    assert attempts["n"] >= 2, "the sweep loop stopped after the first failure"
    assert any("Analysis sweep failed" in r.message for r in caplog.records)


async def test_scheduler_warns_when_generate_at_converts_to_near_midnight_utc(daemon_config, monkeypatch, caplog):
    import asyncio
    import logging

    daemon_config.generation.generate_at = "01:30"
    daemon_config.generation.timezone = "Europe/London"  # BST in summer -> 00:30 UTC

    class _StopLoop(Exception):
        pass

    async def fake_sleep(seconds):
        raise _StopLoop

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(_StopLoop):
            await main_module.schedule_daily_generation(daemon_config)

    assert any("converts to" in r.message for r in caplog.records)


async def test_scheduler_no_warning_for_safe_generate_at(daemon_config, monkeypatch, caplog):
    import asyncio
    import logging

    daemon_config.generation.generate_at = "23:59"
    daemon_config.generation.timezone = "UTC"

    class _StopLoop(Exception):
        pass

    async def fake_sleep(seconds):
        raise _StopLoop

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(_StopLoop):
            await main_module.schedule_daily_generation(daemon_config)

    assert not any("converts to" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# CLI analysis-profile selection
# --------------------------------------------------------------------------


def _profile_config(tmp_path):
    """A config with distinct batch/daemon analysis profiles."""
    return AppConfig(
        telegram=TelegramConfig(
            api_id=1, api_hash="x", session_name=str(tmp_path / "session"),
            channels=[ChannelConfig(slug="chan_a", username="@chan_a")],
        ),
        lmstudio=LMStudioConfig(
            model="shared-model",
            analysis_model="fallback-model",
            analysis_profiles={
                "batch": {"model": "small-model", "max_concurrent_analyses": 4},
                "daemon": {"model": "big-model", "max_concurrent_analyses": 1},
            },
        ),
        storage=StorageConfig(db_path=str(tmp_path / "db.sqlite"),
                              media_dir=str(tmp_path / "media")),
    )


def _run_cli(monkeypatch, tmp_path, argv, seen):
    """Invoke main() with the run functions stubbed, capturing the resolved config."""
    import sys

    cfg = _profile_config(tmp_path)
    monkeypatch.setattr(main_module, "load_config", lambda *a, **kw: cfg)
    monkeypatch.setattr(main_module.asyncio, "run", lambda coro: coro.close())
    monkeypatch.setattr(main_module, "run_batch",
                        lambda config, since=None: seen.update(config=config) or _noop())
    monkeypatch.setattr(main_module, "run_daemon",
                        lambda config: seen.update(config=config) or _noop())
    monkeypatch.setattr(sys, "argv", ["tg_compiler", *argv])
    main_module.main()


def _noop():
    async def _c():
        return None
    return _c()


def test_cli_batch_selects_the_batch_profile(tmp_path, monkeypatch):
    seen = {}
    _run_cli(monkeypatch, tmp_path, ["--batch"], seen)
    lm = seen["config"].lmstudio
    assert lm.model_for("analysis") == "small-model"
    assert lm.max_concurrent_analyses == 4


def test_cli_daemon_selects_the_daemon_profile(tmp_path, monkeypatch):
    seen = {}
    _run_cli(monkeypatch, tmp_path, ["--daemon"], seen)
    lm = seen["config"].lmstudio
    assert lm.model_for("analysis") == "big-model"
    assert lm.max_concurrent_analyses == 1


def test_cli_since_uses_the_batch_profile(tmp_path, monkeypatch):
    """--since is a --batch modifier, and shares its time-critical profile."""
    seen = {}
    _run_cli(monkeypatch, tmp_path, ["--batch", "--since", "2026-09-02"], seen)
    assert seen["config"].lmstudio.model_for("analysis") == "small-model"


def test_cli_analysis_profile_flag_overrides_the_mode_default(tmp_path, monkeypatch):
    seen = {}
    _run_cli(monkeypatch, tmp_path, ["--daemon", "--analysis-profile", "batch"], seen)
    assert seen["config"].lmstudio.model_for("analysis") == "small-model"


def test_cli_unknown_explicit_profile_exits(tmp_path, monkeypatch):
    import pytest

    with pytest.raises(SystemExit, match="Unknown analysis profile"):
        _run_cli(monkeypatch, tmp_path, ["--batch", "--analysis-profile", "nope"], {})


def test_cli_without_profiles_configured_is_unchanged(tmp_path, monkeypatch):
    """A config predating profiles must behave exactly as before."""
    import sys

    cfg = AppConfig(
        telegram=TelegramConfig(
            api_id=1, api_hash="x", session_name=str(tmp_path / "session"),
            channels=[ChannelConfig(slug="chan_a", username="@chan_a")],
        ),
        lmstudio=LMStudioConfig(model="only-model"),
        storage=StorageConfig(db_path=str(tmp_path / "db.sqlite"),
                              media_dir=str(tmp_path / "media")),
    )
    seen = {}
    monkeypatch.setattr(main_module, "load_config", lambda *a, **kw: cfg)
    monkeypatch.setattr(main_module.asyncio, "run", lambda coro: coro.close())
    monkeypatch.setattr(main_module, "run_batch",
                        lambda config, since=None: seen.update(config=config) or _noop())
    monkeypatch.setattr(sys, "argv", ["tg_compiler", "--batch"])
    main_module.main()

    assert seen["config"] is cfg
    assert seen["config"].lmstudio.model_for("analysis") == "only-model"


# --- analysis window ----------------------------------------------------------

def _window_fakes(monkeypatch, received_since, repaired):
    class FakeScraper:
        def __init__(self, config, db):
            self.channel_map = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def scrape_channel(self, channel_cfg):
            return []

        async def repair_missing_media(self, posts):
            repaired.append(posts)
            return 0

    class FakeAnalyzer:
        def __init__(self, config, db):
            pass

        async def process_unanalysed(self, channel_map=None, since=None):
            received_since.append(since)
            return 0, 0

    async def fake_generate_daily_briefing(config, today, db, **kwargs):
        from tg_compiler.triage import BriefingContent
        return "fake.pdf", BriefingContent(date=today, main_items=[], appendix_items=[])

    async def fake_run_analysis(config, today, main_items=None):
        return None

    monkeypatch.setattr(main_module, "Scraper", FakeScraper)
    monkeypatch.setattr("tg_compiler.analyzer.Analyzer", FakeAnalyzer)
    monkeypatch.setattr(main_module, "generate_daily_briefing", fake_generate_daily_briefing)
    monkeypatch.setattr("tg_compiler.synthesiser.run_analysis", fake_run_analysis)


async def test_run_batch_without_since_scopes_analysis_to_the_window(
    tmp_path, batch_config, monkeypatch
):
    from datetime import datetime, timedelta, timezone

    batch_config.storage.db_path = str(tmp_path / "db.sqlite")
    batch_config.storage.retention_days = 30
    received_since, repaired = [], []
    _window_fakes(monkeypatch, received_since, repaired)

    before = datetime.now(timezone.utc)
    await main_module.run_batch(batch_config)
    after = datetime.now(timezone.utc)

    assert len(received_since) == 1
    cutoff = received_since[0]
    assert before - timedelta(days=30) <= cutoff <= after - timedelta(days=30)


async def test_run_batch_since_overrides_the_window(tmp_path, batch_config, monkeypatch):
    from datetime import datetime, timezone

    batch_config.storage.db_path = str(tmp_path / "db.sqlite")
    received_since, repaired = [], []
    _window_fakes(monkeypatch, received_since, repaired)

    since_dt = datetime(2026, 6, 1, tzinfo=timezone.utc)
    await main_module.run_batch(batch_config, since_dt)

    assert received_since == [since_dt]


async def test_run_batch_repairs_missing_media_before_analysing(
    tmp_path, batch_config, monkeypatch
):
    batch_config.storage.db_path = str(tmp_path / "db.sqlite")
    received_since, repaired = [], []
    _window_fakes(monkeypatch, received_since, repaired)

    await main_module.run_batch(batch_config)

    assert repaired == [[]]  # called once, with the (empty) in-window queue


def _run_cli_with_config(monkeypatch, cfg, argv):
    """main() with the run functions stubbed, returning the config run_batch saw."""
    import sys

    seen = {}
    monkeypatch.setattr(main_module, "load_config", lambda *a, **kw: cfg)
    monkeypatch.setattr(main_module.asyncio, "run", lambda coro: coro.close())
    monkeypatch.setattr(main_module, "run_batch",
                        lambda config, since=None: seen.update(config=config) or _noop())
    monkeypatch.setattr(sys, "argv", ["tg_compiler", *argv])
    main_module.main()
    return seen["config"]


def _window_cli_config(tmp_path, lookback_seconds, retention_days):
    return AppConfig(
        telegram=TelegramConfig(
            api_id=1, api_hash="x", session_name=str(tmp_path / "session"),
            channels=[ChannelConfig(slug="chan_a", username="@chan_a")],
            lookback_seconds=lookback_seconds,
        ),
        lmstudio=LMStudioConfig(model="m"),
        storage=StorageConfig(db_path=str(tmp_path / "db.sqlite"),
                              media_dir=str(tmp_path / "media"),
                              retention_days=retention_days),
    )


def test_cli_caps_lookback_seconds_at_the_analysis_window(tmp_path, monkeypatch):
    cfg = _window_cli_config(tmp_path, lookback_seconds=99999999, retention_days=30)
    assert _run_cli_with_config(monkeypatch, cfg, ["--batch"]).telegram.lookback_seconds == 30 * 86400


def test_cli_since_still_overrides_the_lookback_cap(tmp_path, monkeypatch):
    cfg = _window_cli_config(tmp_path, lookback_seconds=600, retention_days=1)
    resolved = _run_cli_with_config(monkeypatch, cfg, ["--batch", "--since", "2026-06-01"])
    # --since is explicit intent and deliberately reaches past the window.
    assert resolved.telegram.lookback_seconds > 30 * 86400


def test_cli_leaves_a_short_lookback_alone(tmp_path, monkeypatch):
    cfg = _window_cli_config(tmp_path, lookback_seconds=600, retention_days=30)
    assert _run_cli_with_config(monkeypatch, cfg, ["--batch"]).telegram.lookback_seconds == 600
