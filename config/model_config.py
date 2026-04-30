import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MODEL_PRESETS_PATH = (
    Path(__file__).resolve().parent
    / "model_presets.json"
)

ModelPreset = dict[str, str | bool | int]


@dataclass(frozen=True)
class ResolvedModelConfig:
    provider: str
    model: str
    base_url: str | None
    api_key: str | None


def _load_model_presets_document() -> dict[str, Any]:
    if not _MODEL_PRESETS_PATH.exists():
        return {}

    try:
        raw = _MODEL_PRESETS_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    return data


def _read_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return value


def load_model_presets() -> dict[str, ModelPreset]:
    """Load model presets from config/model_presets.json."""
    data = _load_model_presets_document()
    if not data:
        return {}

    preset_data = data.get("presets")
    if not isinstance(preset_data, dict):
        return {}

    presets: dict[str, ModelPreset] = {}
    for name, config in preset_data.items():
        if not isinstance(name, str) or not isinstance(config, dict):
            continue

        model = config.get("model")
        if not isinstance(model, str) or not model.strip():
            continue

        cleaned: ModelPreset = {"model": model.strip()}

        provider = config.get("provider")
        if isinstance(provider, str) and provider.strip():
            cleaned["provider"] = provider.strip().lower()

        base_url = config.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            cleaned["base_url"] = base_url.strip()

        if str(cleaned.get("provider", "openai")).lower() == "gateway":
            api_key_env = config.get("api_key_env")
            if isinstance(api_key_env, str) and api_key_env.strip():
                cleaned["api_key_env"] = api_key_env.strip()
        else:
            api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
            if isinstance(api_key_env, str) and api_key_env.strip():
                cleaned["api_key_env"] = api_key_env.strip()
            else:
                cleaned["api_key_env"] = "OPENAI_API_KEY"

        cleaned["vision"] = bool(config.get("vision", False))

        max_input_tokens = _read_positive_int(config.get("max_input_tokens"))
        if max_input_tokens is not None:
            cleaned["max_input_tokens"] = max_input_tokens

        max_output_tokens = _read_positive_int(config.get("max_output_tokens"))
        if max_output_tokens is not None:
            cleaned["max_output_tokens"] = max_output_tokens

        presets[name.strip()] = cleaned

    return presets
def get_default_preset(presets: dict[str, ModelPreset] | None = None) -> str | None:
    """Get the default preset name from model_presets.json."""
    if presets is None:
        presets = load_model_presets()

    data = _load_model_presets_document()
    if not data:
        return None

    default_name = data.get("default")
    if isinstance(default_name, str) and default_name.strip():
        return default_name.strip()

    return None


def get_auto_vision_preset(presets: dict[str, ModelPreset] | None = None) -> str | None:
    """Get the auto vision preset name from model_presets.json."""
    if presets is None:
        presets = load_model_presets()

    data = _load_model_presets_document()
    if not data:
        return None

    preset_name = data.get("auto_vision_preset")
    if isinstance(preset_name, str) and preset_name.strip():
        return preset_name.strip()

    return None


def get_image_summary_preset(presets: dict[str, ModelPreset] | None = None) -> str | None:
    """Get the image summary preset name from model_presets.json."""
    if presets is None:
        presets = load_model_presets()

    data = _load_model_presets_document()
    if not data:
        return None

    preset_name = data.get("image_summary_preset")
    if isinstance(preset_name, str) and preset_name.strip():
        return preset_name.strip()

    return None


def get_model_limits(
    model_name: str | None,
    presets: dict[str, ModelPreset] | None = None,
) -> tuple[int | None, int | None]:
    """Resolve model token limits from presets by model name."""
    if not model_name:
        return None, None

    if presets is None:
        presets = load_model_presets()

    for preset in presets.values():
        preset_model = preset.get("model")
        if preset_model != model_name:
            continue

        max_input_tokens = preset.get("max_input_tokens")
        max_output_tokens = preset.get("max_output_tokens")
        return (
            max_input_tokens if isinstance(max_input_tokens, int) else None,
            max_output_tokens if isinstance(max_output_tokens, int) else None,
        )

    return None, None


def get_model_config(
    model_name: str | None,
    presets: dict[str, ModelPreset] | None = None,
) -> tuple[str, str | None, str | None]:
    """Get model config from presets or environment variables.

    Args:
        model_name: Model preset name (e.g., "glm") or model name
        presets: Optional pre-loaded presets dict

    Returns:
        Tuple of (model, base_url, api_key)
    """
    resolved = resolve_model_config(model_name, presets=presets)
    return resolved.model, resolved.base_url, resolved.api_key


def resolve_model_config(
    model_name: str | None,
    presets: dict[str, ModelPreset] | None = None,
    *,
    fallback_base_url: str | None = None,
    fallback_api_key: str | None = None,
) -> ResolvedModelConfig:
    """Resolve provider-aware runtime config from presets or environment variables."""
    if presets is None:
        presets = load_model_presets()

    if model_name and model_name in presets:
        preset = presets[model_name]
        provider = str(preset.get("provider", "openai")).strip().lower() or "openai"
        model = str(preset["model"]).strip()
        base_url_raw = preset.get("base_url")
        base_url = str(base_url_raw).strip() if isinstance(base_url_raw, str) else None
        if not base_url:
            base_url = fallback_base_url

        api_key: str | None = None
        api_key_env_raw = preset.get("api_key_env")
        api_key_env = (
            str(api_key_env_raw).strip()
            if isinstance(api_key_env_raw, str) and str(api_key_env_raw).strip()
            else None
        )
        if api_key_env:
            api_key = os.getenv(api_key_env)
        if not api_key and provider != "gateway":
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and provider == "gateway":
            api_key = (
                (os.getenv("LLM_GATEWAY_AUTHORIZATION") or "").strip()
                or (os.getenv("CRAB_GATEWAY_API_KEY") or "").strip()
                or None
            )
        if not api_key:
            api_key = fallback_api_key

        return ResolvedModelConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower() or "openai"
    model = model_name or os.getenv("LLM_MODEL", "GLM-4.7")

    if provider == "gateway":
        base_url = (
            (os.getenv("LLM_GATEWAY_BASE_URL") or "").strip()
            or (os.getenv("LLM_BASE_URL") or "").strip()
            or fallback_base_url
        )
        api_key = (
            (os.getenv("LLM_GATEWAY_AUTHORIZATION") or "").strip()
            or (os.getenv("CRAB_GATEWAY_API_KEY") or "").strip()
            or fallback_api_key
        )
    else:
        base_url = (
            (os.getenv("LLM_BASE_URL") or "").strip()
            or fallback_base_url
            or "https://open.bigmodel.cn/api/coding/paas/v4"
        )
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or fallback_api_key

    return ResolvedModelConfig(
        provider=provider,
        model=model,
        base_url=base_url or None,
        api_key=api_key or None,
    )
