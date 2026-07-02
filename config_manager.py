"""
Config Manager
- Load config từ config.yaml khi khởi động
- Nhận override runtime qua WebSocket event `config_update`
- Apply ngay vào VAD layer / pipeline không cần restart process
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

import yaml


@dataclass
class ASRConfig:
    provider: str = "sherpa_onnx"
    model_dir: str = "./sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"
    language: str = "vi"
    num_threads: int = 4
    decode_hop_ms: int = 960
    local_agreement_n: int = 3


@dataclass
class LLMConfig:
    provider: str = "gemini"
    model: str = "models/gemini-3.1-flash-lite"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.1
    max_tokens: int = 256


@dataclass
class VadConfig:
    model_path: str = "./silero_vad.onnx"
    threshold: float = 0.5
    min_silence_ms: int = 400
    min_speech_ms: int = 200


@dataclass
class TranslationConfig:
    source_lang: str = "vi"
    target_lang: str = "en"
    frozen_context_window: int = 3


@dataclass
class Config:
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)


_SECTION_TYPES = {
    "asr": ASRConfig,
    "llm": LLMConfig,
    "vad": VadConfig,
    "translation": TranslationConfig,
}


def _build_section(section_cls: type, raw: dict) -> object:
    valid_names = {f.name for f in fields(section_cls)}
    kwargs = {k: v for k, v in raw.items() if k in valid_names and v is not None}
    return section_cls(**kwargs)


class ConfigManager:
    """Load config.yaml, giữ Config instance sống để apply runtime override."""

    def __init__(self, config: Config):
        self.config = config

    @classmethod
    def load(cls, path: str = "config.yaml") -> "ConfigManager":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        sections = {}
        for name, section_cls in _SECTION_TYPES.items():
            sections[name] = _build_section(section_cls, raw.get(name, {}) or {})

        config = Config(**sections)

        if not config.llm.api_key:
            config.llm.api_key = (
                os.environ.get("LLM_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )

        return cls(config)

    def apply_update(self, patch: dict) -> None:
        """
        Merge runtime override vào config đang chạy, mutate in-place.
        patch dạng: {"vad": {"threshold": 0.6, "min_silence_ms": 300}, ...}
        Chỉ field hợp lệ của mỗi section được áp dụng, field lạ bị bỏ qua.
        """
        for section_name, section_patch in patch.items():
            if section_name not in _SECTION_TYPES or not isinstance(section_patch, dict):
                continue
            section_obj = getattr(self.config, section_name)
            valid_names = {f.name for f in fields(type(section_obj))}
            for key, value in section_patch.items():
                if key in valid_names and value is not None:
                    setattr(section_obj, key, value)
