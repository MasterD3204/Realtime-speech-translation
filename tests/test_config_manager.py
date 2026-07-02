import os

import pytest

from config_manager import Config, ConfigManager, VadConfig


SAMPLE_YAML = """
asr:
  provider: sherpa_onnx
  model_dir: ./models/asr
  num_threads: 2

llm:
  provider: gemini
  model: models/test-model
  temperature: 0.2
  max_tokens: 100

vad:
  threshold: 0.6
  min_silence_ms: 300
  min_speech_ms: 150

translation:
  frozen_context_window: 5
"""


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(SAMPLE_YAML, encoding="utf-8")
    return str(path)


def test_load_parses_all_sections(config_path):
    manager = ConfigManager.load(config_path)
    config = manager.config

    assert config.asr.model_dir == "./models/asr"
    assert config.asr.num_threads == 2
    assert config.llm.model == "models/test-model"
    assert config.llm.temperature == 0.2
    assert config.vad.threshold == 0.6
    assert config.vad.min_silence_ms == 300
    assert config.translation.frozen_context_window == 5


def test_load_ignores_unknown_fields(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("vad:\n  threshold: 0.7\n  bogus_field: 123\n", encoding="utf-8")
    manager = ConfigManager.load(str(path))
    assert manager.config.vad.threshold == 0.7
    assert not hasattr(manager.config.vad, "bogus_field")


def test_load_falls_back_to_env_var_for_api_key(config_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-from-env")
    manager = ConfigManager.load(config_path)
    assert manager.config.llm.api_key == "secret-from-env"


def test_load_uses_explicit_api_key_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "should-not-be-used")
    path = tmp_path / "config.yaml"
    path.write_text("llm:\n  api_key: explicit-key\n", encoding="utf-8")
    manager = ConfigManager.load(str(path))
    assert manager.config.llm.api_key == "explicit-key"


def test_apply_update_mutates_vad_section_in_place():
    manager = ConfigManager(Config())
    original_vad = manager.config.vad

    manager.apply_update({"vad": {"threshold": 0.9, "min_silence_ms": 500}})

    assert manager.config.vad is original_vad  # mutated in place, not replaced
    assert manager.config.vad.threshold == 0.9
    assert manager.config.vad.min_silence_ms == 500
    assert manager.config.vad.min_speech_ms == 200  # untouched field keeps default


def test_apply_update_ignores_unknown_section():
    manager = ConfigManager(Config())
    manager.apply_update({"not_a_real_section": {"foo": "bar"}})
    # No exception, config untouched
    assert manager.config.vad.threshold == 0.5


def test_apply_update_ignores_unknown_field_within_known_section():
    manager = ConfigManager(Config())
    manager.apply_update({"vad": {"bogus_field": 123}})
    assert not hasattr(manager.config.vad, "bogus_field")


def test_apply_update_ignores_none_values():
    manager = ConfigManager(Config())
    manager.apply_update({"vad": {"threshold": None}})
    assert manager.config.vad.threshold == 0.5  # default untouched
