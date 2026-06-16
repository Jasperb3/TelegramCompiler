from __future__ import annotations
import os
import yaml
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChannelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slug: str
    username: str | None = None
    id: int | None = None
    priority: float = 1.0          # multiplier applied to composite score (0.1–2.0)
    credibility: float = 1.0       # channel credibility prior, multiplier applied to composite score (0.1–2.0)
    custom_prompt: str | None = None  # override the default LLM system prompt for this channel


class TelegramConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_id: int
    api_hash: str
    session_name: str = "briefing_session"
    channels: list[ChannelConfig]
    rate_limit_delay_ms: int = 500
    lookback_seconds: int = 604800  # how far back to fetch on first run (default: 1 week)


class LMStudioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    server_host: str = "localhost"
    server_port: int = 1234
    api_token: str | None = None
    temperature: float = 0.3
    max_tokens: int = 800
    synthesis_max_tokens: int = 24000  # token budget for the intel front-page synthesis (large: reasoning models deliberate before emitting JSON)
    max_concurrent_analyses: int = 1  # parallel LLM calls; increase if LM Studio can handle it


class TriageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keywords: list[str] = Field(default_factory=list)
    keyword_boost: float = 0.5
    min_composite_score: float = 2.5
    min_main_items: int = 15    # if fewer items clear min_composite_score, promote top appendix items to fill
    max_main_items: int = 50
    dedup_window_secs: int = 7200       # max age gap (seconds) for cross-channel dedup
    dedup_summary_window_secs: int = 21600  # window (6h) for summary/title Jaccard similarity dedup
    entity_cluster_window_secs: int = 86400  # wider window for entity-cluster dedup (24h)
    recency_half_life_hours: float = 24.0  # composite score halves every this many hours of post age
    recency_floor: float = 0.7          # minimum recency multiplier, however old the post
    # ranking multiplier per threat level, applied to the composite score
    threat_multipliers: dict[str, float] = Field(
        default_factory=lambda: {"CRITICAL": 1.15, "HIGH": 1.05, "MODERATE": 1.0, "LOW": 0.85}
    )
    corroboration_weight: float = 0.15  # composite-score multiplier per corroborating channel
    corroboration_cap: float = 1.5      # max multiplier from corroboration boost
    rumor_penalty: float = 0.7          # composite-score multiplier applied to category == "Rumor"
    dedup_jaccard_threshold: float = 0.28           # min word-overlap ratio for summary/title dedup
    dedup_entity_overlap_count: int = 3             # shared entities required within dedup_window_secs
    dedup_entity_cluster_overlap_count: int = 4     # shared entities required within entity_cluster_window_secs


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_dir: str = "./briefings"
    generate_at: str = "23:59"          # time for daily auto-generation in daemon mode (HH:MM, interpreted in timezone below)
    timezone: str = "UTC"               # IANA timezone name for generate_at scheduling (e.g. "Europe/London")
    pdf_layout: Literal["desktop", "mobile"] = "desktop"  # CSS layout for the generated PDF; override with --layout
    share_to_directory: str | None = None  # if set, copy the final generated PDF to this directory


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    db_path: str = "./data/briefing.db"
    media_dir: str = "./data/media"
    retention_days: int = 30


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    telegram: TelegramConfig
    lmstudio: LMStudioConfig
    triage: TriageConfig = Field(default_factory=TriageConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


def load_config(path: str, env_override: bool = False) -> AppConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    if env_override:
        if api_id := os.getenv("TG_API_ID"):
            try:
                data.setdefault("telegram", {})["api_id"] = int(api_id)
            except ValueError:
                raise ValueError(f"TG_API_ID env var must be an integer, got: {api_id!r}")
        if api_hash := os.getenv("TG_API_HASH"):
            data.setdefault("telegram", {})["api_hash"] = api_hash
        if lm_token := os.getenv("LM_API_TOKEN"):
            data.setdefault("lmstudio", {})["api_token"] = lm_token
    return AppConfig.model_validate(data)
