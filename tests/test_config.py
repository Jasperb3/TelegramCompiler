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
