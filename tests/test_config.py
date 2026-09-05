import pytest
from pydantic import ValidationError

from tg_compiler.config import load_config

MINIMAL_YAML = """
telegram:
  api_id: 111
  api_hash: "abc"
  channels:
    - slug: "test"
      username: "@testchan"
lmstudio:
  model: "gemma-3-4b-it"
"""


def test_load_minimal_config(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text(MINIMAL_YAML)
    cfg = load_config(str(f))
    assert cfg.telegram.api_id == 111
    assert cfg.telegram.channels[0].slug == "test"
    assert cfg.lmstudio.model == "gemma-3-4b-it"
    assert cfg.triage.keyword_boost == 0.5
    assert cfg.storage.retention_days == 30


def test_missing_api_id_raises(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("telegram:\n  api_hash: x\n  channels: []\nlmstudio:\n  model: x\n")
    with pytest.raises(ValidationError):
        load_config(str(bad))


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TG_API_ID", "999")
    monkeypatch.setenv("TG_API_HASH", "envhash")
    f = tmp_path / "config.yaml"
    f.write_text(MINIMAL_YAML)
    cfg = load_config(str(f), env_override=True)
    assert cfg.telegram.api_id == 999
    assert cfg.telegram.api_hash == "envhash"


def test_synthesis_post_limit_rejected(tmp_path):
    """synthesis_post_limit was removed; config must reject it."""
    yaml_with_old_field = MINIMAL_YAML + "\ngeneration:\n  synthesis_post_limit: 20\n"
    f = tmp_path / "config.yaml"
    f.write_text(yaml_with_old_field)
    with pytest.raises(ValidationError):
        load_config(str(f))


def test_unknown_triage_key_rejected(tmp_path):
    """Typos inside nested sections (e.g. rumour_penalty) must fail loudly."""
    bad_yaml = MINIMAL_YAML + "\ntriage:\n  rumour_penalty: 0.5\n"
    f = tmp_path / "config.yaml"
    f.write_text(bad_yaml)
    with pytest.raises(ValidationError):
        load_config(str(f))


def test_unknown_channel_key_rejected(tmp_path):
    bad_yaml = MINIMAL_YAML.replace(
        'username: "@testchan"', 'username: "@testchan"\n      credibilty: 1.2'
    )
    f = tmp_path / "config.yaml"
    f.write_text(bad_yaml)
    with pytest.raises(ValidationError):
        load_config(str(f))


def test_empty_config_file_raises_validation_error(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("")
    with pytest.raises(ValidationError):
        load_config(str(f))


def test_bad_generate_at_rejected(tmp_path):
    bad_yaml = MINIMAL_YAML + "\ngeneration:\n  generate_at: \"25:99\"\n"
    f = tmp_path / "config.yaml"
    f.write_text(bad_yaml)
    with pytest.raises(ValidationError):
        load_config(str(f))


def test_bad_timezone_rejected(tmp_path):
    bad_yaml = MINIMAL_YAML + "\ngeneration:\n  timezone: \"Not/AZone\"\n"
    f = tmp_path / "config.yaml"
    f.write_text(bad_yaml)
    with pytest.raises(ValidationError):
        load_config(str(f))


def test_negative_channel_priority_rejected(tmp_path):
    bad_yaml = MINIMAL_YAML.replace(
        'username: "@testchan"', 'username: "@testchan"\n      priority: -1.0'
    )
    f = tmp_path / "config.yaml"
    f.write_text(bad_yaml)
    with pytest.raises(ValidationError):
        load_config(str(f))


def test_bad_channel_slug_rejected(tmp_path):
    bad_yaml = MINIMAL_YAML.replace('slug: "test"', 'slug: "../../etc"')
    f = tmp_path / "config.yaml"
    f.write_text(bad_yaml)
    with pytest.raises(ValidationError):
        load_config(str(f))


def test_lowercase_threat_multiplier_key_rejected(tmp_path):
    bad_yaml = MINIMAL_YAML + "\ntriage:\n  threat_multipliers:\n    critical: 1.5\n"
    f = tmp_path / "config.yaml"
    f.write_text(bad_yaml)
    with pytest.raises(ValidationError):
        load_config(str(f))


def test_min_main_items_exceeding_max_rejected(tmp_path):
    bad_yaml = MINIMAL_YAML + "\ntriage:\n  min_main_items: 60\n  max_main_items: 50\n"
    f = tmp_path / "config.yaml"
    f.write_text(bad_yaml)
    with pytest.raises(ValidationError):
        load_config(str(f))


def test_lmstudio_analysis_budget_defaults(tmp_path):
    """A config without the analysis_* fields gets the documented defaults."""
    f = tmp_path / "config.yaml"
    f.write_text(MINIMAL_YAML)
    cfg = load_config(str(f))
    assert cfg.lmstudio.analysis_base_tokens == 1500
    assert cfg.lmstudio.analysis_tokens_per_char == 0.3
    assert cfg.lmstudio.analysis_tokens_per_image == 250
    assert cfg.lmstudio.analysis_max_tokens == 4000


def test_removed_lmstudio_max_tokens_rejected(tmp_path):
    """max_tokens was replaced by the analysis_* budget fields; a stale config
    still carrying it must fail loudly rather than silently doing nothing."""
    bad_yaml = MINIMAL_YAML.replace(
        'model: "gemma-3-4b-it"', 'model: "gemma-3-4b-it"\n  max_tokens: 800'
    )
    f = tmp_path / "config.yaml"
    f.write_text(bad_yaml)
    with pytest.raises(ValidationError):
        load_config(str(f))


def test_analysis_base_tokens_exceeding_max_rejected(tmp_path):
    bad_yaml = MINIMAL_YAML.replace(
        'model: "gemma-3-4b-it"',
        'model: "gemma-3-4b-it"\n  analysis_base_tokens: 5000\n  analysis_max_tokens: 4000',
    )
    f = tmp_path / "config.yaml"
    f.write_text(bad_yaml)
    with pytest.raises(ValidationError):
        load_config(str(f))


def test_min_composite_score_default_is_3_5(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text(MINIMAL_YAML)
    cfg = load_config(str(f))
    assert cfg.triage.min_composite_score == 3.5


# --------------------------------------------------------------------------
# Per-run-mode analysis profiles
# --------------------------------------------------------------------------


def _profiled_lmstudio(**profiles):
    from tg_compiler.config import LMStudioConfig

    return LMStudioConfig(
        model="shared-model",
        analysis_model="small-model",
        analysis_base_tokens=700,
        analysis_max_tokens=1600,
        max_concurrent_analyses=4,
        analysis_profiles=profiles,
    )


def test_profile_overrides_only_the_fields_it_sets():
    cfg = _profiled_lmstudio(daemon={"analysis_base_tokens": 9500, "analysis_max_tokens": 16000})
    daemon = cfg.with_analysis_profile("daemon")

    assert daemon.analysis_base_tokens == 9500
    assert daemon.analysis_max_tokens == 16000
    # untouched fields are inherited
    assert daemon.max_concurrent_analyses == 4
    assert daemon.analysis_model == "small-model"


def test_profile_model_becomes_the_analysis_model_only():
    """A profile's `model:` is the analysis model. `model` itself stays the global
    fallback that synthesis resolves through, so a profile must not hijack it."""
    cfg = _profiled_lmstudio(daemon={"model": "big-model"})
    daemon = cfg.with_analysis_profile("daemon")

    assert daemon.model_for("analysis") == "big-model"
    assert daemon.model == "shared-model"
    assert daemon.model_for("synthesis") == "shared-model"


def test_profile_values_are_revalidated_after_merging():
    """model_copy(update=...) skips validators, so a profile could otherwise set
    analysis_base_tokens above analysis_max_tokens and truncate every response."""
    import pytest

    cfg = _profiled_lmstudio(bad={"analysis_base_tokens": 20000})
    with pytest.raises(ValidationError, match="analysis_base_tokens"):
        cfg.with_analysis_profile("bad")


def test_unknown_profile_leaves_the_config_untouched():
    cfg = _profiled_lmstudio(daemon={"model": "big-model"})
    assert cfg.with_analysis_profile("nope") is cfg
    assert cfg.with_analysis_profile(None) is cfg


def test_profile_rejects_unknown_keys():
    import pytest

    with pytest.raises(ValidationError):
        _profiled_lmstudio(daemon={"analysis_base_tokns": 9500})  # typo


def test_app_config_with_analysis_profile_replaces_lmstudio():
    from tg_compiler.config import AppConfig, TelegramConfig

    app = AppConfig(
        telegram=TelegramConfig(api_id=1, api_hash="x", channels=[]),
        lmstudio=_profiled_lmstudio(daemon={"model": "big-model", "max_concurrent_analyses": 1}),
    )
    resolved = app.with_analysis_profile("daemon")

    assert resolved.lmstudio.model_for("analysis") == "big-model"
    assert resolved.lmstudio.max_concurrent_analyses == 1
    # the original is not mutated
    assert app.lmstudio.model_for("analysis") == "small-model"
    assert app.lmstudio.max_concurrent_analyses == 4


def test_config_without_profiles_is_unchanged():
    from tg_compiler.config import LMStudioConfig

    cfg = LMStudioConfig(model="m")
    assert cfg.analysis_profiles == {}
    assert cfg.with_analysis_profile("batch") is cfg


def test_analysis_lookback_days_defaults_to_retention_days():
    from tg_compiler.config import StorageConfig

    assert StorageConfig(retention_days=14).analysis_lookback_days_effective() == 14


def test_analysis_lookback_days_overrides_retention_days():
    from tg_compiler.config import StorageConfig

    cfg = StorageConfig(retention_days=30, analysis_lookback_days=7)
    assert cfg.analysis_lookback_days_effective() == 7


def test_analysis_cutoff_is_now_minus_the_window():
    from datetime import datetime, timedelta, timezone

    from tg_compiler.config import StorageConfig

    now = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
    assert StorageConfig(retention_days=30).analysis_cutoff(now) == now - timedelta(days=30)


def test_zero_analysis_lookback_days_rejected():
    from tg_compiler.config import StorageConfig

    with pytest.raises(ValidationError):
        StorageConfig(analysis_lookback_days=0)


def test_unknown_storage_key_rejected(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text(MINIMAL_YAML + '\nstorage:\n  analysis_lookback: 7\n')
    with pytest.raises(ValidationError):
        load_config(str(f))
