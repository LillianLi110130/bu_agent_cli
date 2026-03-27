import json
from pathlib import Path
from uuid import uuid4

import pytest

from agent_core.agent.compaction.service import CompactionService
from agent_core.llm.openai.chat import ChatOpenAI
from config import model_config


def _make_temp_file() -> Path:
    return Path.cwd() / f"test_model_presets_{uuid4().hex}.json"


def _write_model_presets(config_path: Path, payload: dict) -> None:
    config_path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_model_presets_reads_token_limits(monkeypatch):
    config_path = _make_temp_file()
    _write_model_presets(
        config_path,
        {
            "default": "GLM-4.7",
            "auto_vision_preset": "GLM-4.6V",
            "image_summary_preset": "GLM-4.6V",
            "presets": {
                "GLM-4.7": {
                    "model": "GLM-4.7",
                    "base_url": "https://example.invalid/v1",
                    "api_key_env": "OPENAI_API_KEY",
                    "vision": False,
                    "max_input_tokens": 128000,
                    "max_output_tokens": 8192,
                },
                "bad": {
                    "model": "bad-model",
                    "max_input_tokens": -1,
                    "max_output_tokens": True,
                },
            },
        },
    )
    monkeypatch.setattr(model_config, "_MODEL_PRESETS_PATH", config_path)

    presets = model_config.load_model_presets()

    assert presets["GLM-4.7"]["max_input_tokens"] == 128000
    assert presets["GLM-4.7"]["max_output_tokens"] == 8192
    assert "max_input_tokens" not in presets["bad"]
    assert "max_output_tokens" not in presets["bad"]
    assert model_config.get_default_preset(presets) == "GLM-4.7"
    assert model_config.get_auto_vision_preset(presets) == "GLM-4.6V"
    assert model_config.get_image_summary_preset(presets) == "GLM-4.6V"
    assert model_config.get_model_limits("GLM-4.7", presets) == (128000, 8192)


def test_chat_openai_uses_model_preset_limits(monkeypatch):
    config_path = _make_temp_file()
    _write_model_presets(
        config_path,
        {
            "presets": {
                "GLM-4.7": {
                    "model": "GLM-4.7",
                    "max_input_tokens": 128000,
                    "max_output_tokens": 8192,
                }
            }
        },
    )
    monkeypatch.setattr(model_config, "_MODEL_PRESETS_PATH", config_path)

    llm = ChatOpenAI(model="GLM-4.7")
    explicit = ChatOpenAI(model="GLM-4.7", max_completion_tokens=2048, max_input_tokens=64000)
    fallback = ChatOpenAI(model="unknown-model")

    assert llm.max_input_tokens == 128000
    assert llm.max_completion_tokens == 8192
    assert explicit.max_input_tokens == 64000
    assert explicit.max_completion_tokens == 2048
    assert fallback.max_input_tokens is None
    assert fallback.max_completion_tokens == 4096


@pytest.mark.asyncio
async def test_compaction_service_uses_model_preset_max_input_tokens(monkeypatch):
    config_path = _make_temp_file()
    _write_model_presets(
        config_path,
        {
            "presets": {
                "GLM-4.7": {
                    "model": "GLM-4.7",
                    "max_input_tokens": 256000,
                }
            }
        },
    )
    monkeypatch.setattr(model_config, "_MODEL_PRESETS_PATH", config_path)

    service = CompactionService()

    context_limit = await service.get_model_context_limit("GLM-4.7")

    assert context_limit == 256000
