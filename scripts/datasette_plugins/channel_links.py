"""Registers a `tme_link(slug, message_id)` SQL function for the inspection UI.

`posts.channel_name` stores the channel *slug*, not its Telegram username, and for
7 of the 16 configured channels those differ (RerumNovarum -> rnintel,
WarFrontWitness -> wfwitness, ...). Building a deep link as
`t.me/<channel_name>/<message_id>` in SQL would therefore emit dead links for
those channels.

The slug -> username mapping lives in `config.yaml`, not in the database, and
`AppConfig.channel_link_map()` is already the single source of truth for it
(`main.py` and `synthesiser.py` both go through it). This reuses that rather than
duplicating the mapping as a CASE expression in the metadata YAML, so the two
cannot drift apart.

The viewer must still start when config is unavailable — the database is the thing
being inspected — so a missing or invalid `config.yaml` degrades to `tme_link()`
returning NULL rather than refusing to boot.
"""

import logging
import sys
from pathlib import Path

from datasette import hookimpl

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

_link_map: dict[str, str] | None = None


def _load_link_map() -> dict[str, str]:
    """slug -> bare username, loaded once and cached (empty dict if unavailable)."""
    global _link_map
    if _link_map is not None:
        return _link_map
    src = str(_PROJECT_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from tg_compiler.config import load_config

        _link_map = load_config(str(_CONFIG_PATH)).channel_link_map()
        logger.info("tme_link: loaded %d channel links from %s", len(_link_map), _CONFIG_PATH)
    except Exception as exc:  # missing config, bad YAML, import failure
        logger.warning("tme_link: no channel links (%s: %s) — links will be NULL", type(exc).__name__, exc)
        _link_map = {}
    return _link_map


def tme_link(slug, message_id):
    """Telegram deep link for a post, or None when the slug has no configured username."""
    if slug is None or message_id is None:
        return None
    username = _load_link_map().get(str(slug))
    if not username:
        return None
    return f"https://t.me/{username}/{message_id}"


@hookimpl
def prepare_connection(conn):
    conn.create_function("tme_link", 2, tme_link)
