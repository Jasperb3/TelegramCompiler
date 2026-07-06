import io
import sqlite3

import pytest

from tg_compiler.utils import connect_telegram_client, escape_html


def test_escape_html_neutralises_markdown_link_syntax():
    result = escape_html("[click here](https://evil.example)")
    assert "](" not in result
    assert "&#91;click here&#93;" in result


def test_escape_html_neutralises_backticks():
    assert "`" not in escape_html("run `rm -rf /`")


def test_escape_html_still_escapes_angle_brackets_and_amp():
    result = escape_html("<script>alert(1)</script> & co")
    assert "<" not in result and ">" not in result
    assert "&amp;" in result


class FakeClient:
    def __init__(self, authorized=True, connect_error=None):
        self._authorized = authorized
        self._connect_error = connect_error
        self.start_called = False

    async def connect(self):
        if self._connect_error:
            raise self._connect_error

    async def is_user_authorized(self):
        return self._authorized

    async def start(self):
        self.start_called = True


async def test_connect_telegram_client_skips_start_when_already_authorized():
    client = FakeClient(authorized=True)
    await connect_telegram_client(client, "session")
    assert client.start_called is False


async def test_connect_telegram_client_starts_interactively_when_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO())
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    client = FakeClient(authorized=False)
    await connect_telegram_client(client, "session")
    assert client.start_called is True


async def test_connect_telegram_client_exits_when_unauthorized_and_no_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO())
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    client = FakeClient(authorized=False)
    with pytest.raises(SystemExit, match="no TTY"):
        await connect_telegram_client(client, "session")
    assert client.start_called is False


async def test_connect_telegram_client_raises_actionable_error_on_locked_session():
    client = FakeClient(connect_error=sqlite3.OperationalError("database is locked"))
    with pytest.raises(SystemExit, match="locked"):
        await connect_telegram_client(client, "my_session")
