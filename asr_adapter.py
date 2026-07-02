"""
ASR Adapter Interface (thin adapter) — spec §5.2
Cho phép swap provider mà không ảnh hưởng tầng Translation Pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import sherpa_onnx


@dataclass(frozen=True)
class ASRResult:
    text: str
    window_start_ms: int
    utterance_id: int = 0


class ASRAdapter(Protocol):
    def transcribe(self, audio_window: np.ndarray, window_start_ms: int, utterance_id: int = 0) -> ASRResult: ...


class SherpaOnnxAdapter:
    """Offline transducer recognizer (zipformer) — mỗi window decode độc lập, full transcript."""

    def __init__(self, model_dir: str, num_threads: int = 4, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=f"{model_dir}/tokens.txt",
            encoder=f"{model_dir}/encoder-epoch-20-avg-10.onnx",
            decoder=f"{model_dir}/decoder.onnx",
            joiner=f"{model_dir}/joiner-epoch-20-avg-10.onnx",
            num_threads=num_threads,
            decoding_method="greedy_search",
            debug=False,
        )

    def transcribe(self, audio_window: np.ndarray, window_start_ms: int, utterance_id: int = 0) -> ASRResult:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, audio_window)
        self._recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        return ASRResult(text=text, window_start_ms=window_start_ms, utterance_id=utterance_id)
