from __future__ import annotations

import logging
import time
from typing import Any

from tg_compiler.config import LMStudioConfig

log = logging.getLogger(__name__)


class ModelManager:
    """Makes one model resident in LM Studio at a pipeline stage boundary.

    Inference still goes over the OpenAI-compatible REST endpoint; this only
    controls which weights are loaded, because a 27B model leaves too little VRAM
    for a second model to co-reside (16 GB card, 11.9 GB resident) and "just send
    a different model= " would then thrash or fail.

    Does nothing at all unless `manage_models` is set, so single-model setups are
    untouched. When it is set and LM Studio cannot be reached the constructor
    raises: a run that cannot swap models is going to fail later anyway, and
    failing here says why.
    """

    def __init__(self, cfg: LMStudioConfig, client: Any | None = None):
        self._cfg = cfg
        self._client = client
        self._owned = client is None
        if not cfg.manage_models or client is not None:
            return
        try:
            import lmstudio as lms

            self._client = lms.Client(api_host=f"{cfg.server_host}:{cfg.server_port}")
        except Exception as e:
            raise RuntimeError(
                f"manage_models is enabled but the LM Studio SDK could not reach "
                f"{cfg.server_host}:{cfg.server_port} — {e}"
            ) from e

    def __enter__(self) -> "ModelManager":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owned and self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                log.warning("Closing the LM Studio SDK client failed: %s", e)
            self._client = None

    @staticmethod
    def _keys(model: Any) -> set[str]:
        """Every name a loaded model answers to.

        LM Studio reports both an `identifier` (which the user can rename) and the
        underlying `modelKey`; a stage's configured key may match either.
        """
        keys = {getattr(model, "identifier", None)}
        try:
            info = model.get_info()
            info = info.to_dict() if hasattr(info, "to_dict") else info
            if isinstance(info, dict):
                keys |= {info.get("identifier"), info.get("modelKey"), info.get("model_key")}
        except Exception as e:
            log.debug("Could not read model info: %s", e)
        return {k for k in keys if k}

    def ensure(self, model_key: str) -> None:
        """Make `model_key` the loaded model, unloading whatever else is resident."""
        if self._client is None:
            return

        loaded = list(self._client.list_loaded_models("llm"))
        if any(model_key in self._keys(m) for m in loaded):
            log.debug("Model %s is already loaded", model_key)
            return

        if self._cfg.unload_others:
            for model in loaded:
                name = next(iter(self._keys(model)), "<unknown>")
                log.info("Unloading %s to make room for %s", name, model_key)
                try:
                    model.unload()
                except Exception as e:
                    log.warning("Could not unload %s: %s", name, e)

        config = None
        if self._cfg.model_context_length:
            import lmstudio as lms

            config = lms.LlmLoadModelConfig(context_length=self._cfg.model_context_length)

        started = time.time()
        self._client.llm.load_new_instance(
            model_key,
            config=config,
            ttl=self._cfg.model_ttl_seconds,
        )
        log.info("Loaded %s in %.1fs", model_key, time.time() - started)
