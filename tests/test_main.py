from datetime import datetime, timezone

import pytest
from tg_compiler.config import AppConfig, TelegramConfig, LMStudioConfig, ChannelConfig
from tg_compiler import main as main_module
from tg_compiler.main import _parse_since, purge_old_media, _share_pdf


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

    class FakeAnalyzer:
        def __init__(self, config, db):
            pass

        async def process_unanalysed(self, channel_map=None):
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
                importance=1, urgency=1, credibility=1, relevance=1,
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
                importance=1, urgency=1, credibility=1, relevance=1,
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
    from tg_compiler.triage import BriefingContent

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
