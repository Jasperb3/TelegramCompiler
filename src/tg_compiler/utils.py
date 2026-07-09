from __future__ import annotations

import html
import logging
import os
import re
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def escape_html(text: str) -> str:
    """Escape &, < and > in LLM-derived text so it can't inject HTML markup into
    rendered output, plus [, ] and backtick so it can't form a markdown link/image
    ([text](url), ![alt](url)) or code span in the rendered markdown document."""
    if not text:
        return text
    text = html.escape(text, quote=False)
    return text.replace("[", "&#91;").replace("]", "&#93;").replace("`", "&#96;")


async def connect_telegram_client(client, session_name: str) -> None:
    """Connect a Telethon client without blindly falling into an interactive
    login prompt. Two callers (batch scraper, daemon) share this so a locked
    session file or a missing/expired session under a TTY-less process
    (nohup/systemd) produces an actionable error instead of a silent hang or
    a cryptic sqlite3 lock error."""
    try:
        await client.connect()
    except sqlite3.OperationalError as e:
        raise SystemExit(
            f"Session file '{session_name}.session' is locked — is the daemon "
            f"(or another batch run) already using it? Stop it or configure a "
            f"different session_name."
        ) from e
    if not await client.is_user_authorized():
        if sys.stdin.isatty():
            await client.start()
        else:
            raise SystemExit(
                "Telegram session missing/expired and no TTY for interactive login — "
                "run 'python -m tg_compiler.main --batch' in a terminal once to authenticate."
            )


def secure_file(path: str) -> None:
    """Restrict a sensitive file to owner read/write only, if it exists."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass

_ENTITY_GARBAGE = re.compile(
    r'[`{}<>\[\]]|json|PostAnalysis|importance_score|urgency_score'
    r'|title|summary|category|image_substantive|post_id|credibility|relevance|reasoning'
    r'|key_entities|threat_level|image_description|\.json\('
    r'|\bfalse\b|\btrue\b|\bnull\b'
    r'|\bCRITICAL\b|\bHIGH\b|\bMODERATE\b|\bLOW\b'
    r'|Breaking News|Official Statement|Breaking news|Official statement'
)


def clean_entities(entities: list[str]) -> list[str]:
    return [
        e.strip() for e in entities
        if e
        and len(e.strip()) >= 2
        and len(e.strip()) <= 80
        and not re.fullmatch(r'\d+', e.strip())
        and not _ENTITY_GARBAGE.search(e)
    ]


# Aliases for entity-overlap dedup so naming variants of the same actor match
# (e.g. "U.S." and "United States" both normalize to "united states"). Keys are
# compared with periods stripped. Defined entirely via entity_aliases.yaml
# (gitignored, project root) — see entity_aliases.yaml.example. Empty if absent.
_ENTITY_ALIASES_PATH = Path("entity_aliases.yaml")


@lru_cache(maxsize=1)
def entity_aliases() -> dict[str, str]:
    if not _ENTITY_ALIASES_PATH.exists():
        log.info("No entity_aliases.yaml found at %s — alias normalisation disabled", _ENTITY_ALIASES_PATH.resolve())
        return {}
    with open(_ENTITY_ALIASES_PATH) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        log.warning("entity_aliases.yaml at %s is not a flat string→string mapping — ignoring it", _ENTITY_ALIASES_PATH.resolve())
        return {}
    log.info("Loaded %d entity aliases from %s", len(data), _ENTITY_ALIASES_PATH.resolve())
    return data


def normalize_entity(e: str) -> str:
    stripped = e.strip().lower().replace(".", "")
    return entity_aliases().get(stripped, stripped)
