from types import SimpleNamespace

import pytest

from tg_compiler.config import LMStudioConfig
from tg_compiler.models import ModelManager


def _cfg(**overrides):
    base = dict(model="big-model", server_host="host", server_port=1234, manage_models=True)
    base.update(overrides)
    return LMStudioConfig(**base)


class FakeModel:
    def __init__(self, identifier, model_key=None):
        self.identifier = identifier
        self._model_key = model_key or identifier
        self.unloaded = False

    def get_info(self):
        return {"identifier": self.identifier, "modelKey": self._model_key}

    def unload(self):
        self.unloaded = True


class FakeClient:
    def __init__(self, loaded=()):
        self.loaded = list(loaded)
        self.loads = []
        self.closed = False
        self.llm = SimpleNamespace(load_new_instance=self._load)

    def _load(self, model_key, config=None, ttl=None):
        self.loads.append({"model_key": model_key, "config": config, "ttl": ttl})
        return FakeModel(model_key)

    def list_loaded_models(self, namespace=None):
        return list(self.loaded)

    def close(self):
        self.closed = True


def test_manager_is_inert_when_management_disabled(monkeypatch):
    def explode(*a, **kw):  # pragma: no cover - must never run
        raise AssertionError("the SDK was contacted despite manage_models being off")

    monkeypatch.setattr("lmstudio.Client", explode)
    with ModelManager(_cfg(manage_models=False)) as manager:
        manager.ensure("some-model")  # no client, nothing to do


def test_ensure_is_a_noop_when_the_model_is_already_loaded():
    client = FakeClient(loaded=[FakeModel("small-model")])
    ModelManager(_cfg(), client=client).ensure("small-model")
    assert client.loads == []
    assert client.loaded[0].unloaded is False


def test_ensure_matches_on_model_key_not_just_identifier():
    # LM Studio lets a loaded instance be renamed; the configured key may be either.
    client = FakeClient(loaded=[FakeModel("my-renamed-instance", model_key="small-model")])
    ModelManager(_cfg(), client=client).ensure("small-model")
    assert client.loads == []


def test_ensure_unloads_others_then_loads_with_ttl_and_context():
    resident = FakeModel("big-model")
    client = FakeClient(loaded=[resident])
    cfg = _cfg(model_ttl_seconds=900, model_context_length=8192)

    ModelManager(cfg, client=client).ensure("small-model")

    assert resident.unloaded is True
    assert len(client.loads) == 1
    load = client.loads[0]
    assert load["model_key"] == "small-model"
    assert load["ttl"] == 900
    assert load["config"].context_length == 8192


def test_ensure_without_context_override_passes_no_load_config():
    client = FakeClient(loaded=[])
    ModelManager(_cfg(), client=client).ensure("small-model")
    assert client.loads[0]["config"] is None


def test_ensure_keeps_other_models_loaded_when_unload_others_is_off():
    resident = FakeModel("big-model")
    client = FakeClient(loaded=[resident])
    ModelManager(_cfg(unload_others=False), client=client).ensure("small-model")
    assert resident.unloaded is False
    assert client.loads[0]["model_key"] == "small-model"


def test_a_failing_unload_does_not_stop_the_load():
    class StubbornModel(FakeModel):
        def unload(self):
            raise RuntimeError("busy")

    client = FakeClient(loaded=[StubbornModel("big-model")])
    ModelManager(_cfg(), client=client).ensure("small-model")
    assert client.loads[0]["model_key"] == "small-model"


def test_unreachable_sdk_raises_naming_the_host(monkeypatch):
    def explode(*a, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr("lmstudio.Client", explode)
    with pytest.raises(RuntimeError, match="host:1234"):
        ModelManager(_cfg())


def test_injected_client_is_not_closed_by_the_manager():
    client = FakeClient()
    with ModelManager(_cfg(), client=client):
        pass
    assert client.closed is False


def test_model_for_falls_back_to_the_shared_model():
    cfg = LMStudioConfig(model="big-model")
    assert cfg.model_for("analysis") == "big-model"
    assert cfg.model_for("synthesis") == "big-model"


def test_model_for_returns_the_stage_model_when_set():
    cfg = LMStudioConfig(model="big-model", analysis_model="small-model")
    assert cfg.model_for("analysis") == "small-model"
    assert cfg.model_for("synthesis") == "big-model"
