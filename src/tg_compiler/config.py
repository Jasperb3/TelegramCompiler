from __future__ import annotations

import os
import re
import zoneinfo
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_GENERATE_AT_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_THREAT_LEVELS = {"CRITICAL", "HIGH", "MODERATE", "LOW"}


class ChannelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slug: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    username: str | None = None
    id: int | None = None
    priority: float = Field(default=1.0, gt=0, le=5)       # multiplier applied to composite score (0.1–2.0)
    credibility: float = Field(default=1.0, gt=0, le=5)    # channel credibility prior, multiplier applied to composite score (0.1–2.0)
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
    # Dynamic per-post analysis token budget (replaces the old fixed max_tokens; see
    # analyzer.compute_token_budget):
    #   budget = min(analysis_base_tokens + text_chars * analysis_tokens_per_char
    #                + images * analysis_tokens_per_image, analysis_max_tokens)
    analysis_base_tokens: int = Field(default=1500, gt=0)        # flat floor: reasoning deliberation + structured JSON output
    analysis_tokens_per_char: float = Field(default=0.3, ge=0)   # extra budget per prompt-text character
    analysis_tokens_per_image: int = Field(default=250, ge=0)    # extra budget per attached image
    analysis_max_tokens: int = Field(default=4000, gt=0)         # hard ceiling on any single analysis call
    synthesis_max_tokens: int = 24000  # token budget for the intel front-page synthesis (large: reasoning models deliberate before emitting JSON)
    max_concurrent_analyses: int = 1  # parallel LLM calls; increase if LM Studio can handle it
    # Batched analysis (see analyzer.plan_batches / analyze_batch). Several posts go
    # into one LLM call, which amortises the mostly per-call reasoning burn across
    # them. Both sizes default to 1 = one post per call = the pre-batching path.
    batch_size: int = Field(default=1, ge=1)              # text-only posts per LLM call
    batch_size_with_images: int = Field(default=1, ge=1)  # media-bearing posts per LLM call
    batch_images_per_post: int = Field(default=1, ge=1)   # images attached per post in a batch
    #   budget = min(batch_base_tokens + posts * batch_tokens_per_post
    #                + text_chars * batch_tokens_per_char
    #                + images * batch_tokens_per_image, batch_max_tokens)
    # Defaults sized from measurement so a reasoning model doesn't truncate: a batch
    # of 10 text posts spent 9,364 completion tokens on prism-ml/bonsai-27b, a batch
    # of 3 image posts 7,668. max_tokens is a cap, not an allocation, so headroom
    # costs a non-reasoning model nothing.
    batch_base_tokens: int = Field(default=4000, gt=0)       # flat shared reasoning allowance per batch call
    batch_tokens_per_post: int = Field(default=600, ge=0)    # per-post JSON output allowance
    batch_tokens_per_char: float = Field(default=0.3, ge=0)  # per character of batched prompt text
    batch_tokens_per_image: int = Field(default=1200, ge=0)  # per image attached in a batch
    batch_max_tokens: int = Field(default=32000, gt=0)      # hard ceiling on any batch call
    batch_max_prompt_chars: int = Field(default=24000, gt=0)  # split a batch whose prompt text exceeds this
    batch_min_yield_ratio: float = Field(default=0.6, ge=0.0, le=1.0)  # below this, retry the batch's posts singly

    @model_validator(mode="after")
    def _validate_analysis_budget_bounds(self) -> "LMStudioConfig":
        if self.analysis_base_tokens > self.analysis_max_tokens:
            raise ValueError(
                f"analysis_base_tokens ({self.analysis_base_tokens}) must be <= "
                f"analysis_max_tokens ({self.analysis_max_tokens})"
            )
        if self.batch_base_tokens > self.batch_max_tokens:
            raise ValueError(
                f"batch_base_tokens ({self.batch_base_tokens}) must be <= "
                f"batch_max_tokens ({self.batch_max_tokens})"
            )
        return self


class TriageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keywords: list[str] = Field(default_factory=list)
    keyword_boost: float = 0.5
    min_composite_score: float = 3.5
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

    @field_validator("threat_multipliers")
    @classmethod
    def _validate_threat_multiplier_keys(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v) - _THREAT_LEVELS
        if unknown:
            raise ValueError(
                f"threat_multipliers has unknown key(s) {sorted(unknown)}; "
                f"must be a subset of {sorted(_THREAT_LEVELS)}"
            )
        return v

    @model_validator(mode="after")
    def _validate_main_items_bounds(self) -> "TriageConfig":
        if self.max_main_items > 0 and self.min_main_items > self.max_main_items:
            raise ValueError(
                f"min_main_items ({self.min_main_items}) must be <= "
                f"max_main_items ({self.max_main_items})"
            )
        return self


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_dir: str = "./briefings"
    generate_at: str = "23:59"          # time for daily auto-generation in daemon mode (HH:MM, interpreted in timezone below)
    timezone: str = "UTC"               # IANA timezone name for generate_at scheduling (e.g. "Europe/London")
    pdf_layout: Literal["desktop", "mobile"] = "desktop"  # CSS layout for the generated PDF; override with --layout
    share_to_directory: str | None = None  # if set, copy the final generated PDF to this directory

    @field_validator("generate_at")
    @classmethod
    def _validate_generate_at(cls, v: str) -> str:
        if not _GENERATE_AT_RE.match(v):
            raise ValueError(f"generate_at must be HH:MM (24-hour), got: {v!r}")
        return v

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, v: str) -> str:
        try:
            zoneinfo.ZoneInfo(v)
        except Exception as e:
            raise ValueError(f"Unknown IANA timezone: {v!r}") from e
        return v


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

    def channel_priority_map(self) -> dict[str, float]:
        return {ch.slug: ch.priority for ch in self.telegram.channels}

    def channel_credibility_map(self) -> dict[str, float]:
        return {ch.slug: ch.credibility for ch in self.telegram.channels}

    def channel_link_map(self) -> dict[str, str]:
        """slug -> bare username (no @) for channels that have one, for t.me deep links."""
        return {
            ch.slug: ch.username.lstrip("@")
            for ch in self.telegram.channels
            if ch.username
        }


def load_config(path: str, env_override: bool = False) -> AppConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
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
