"""
VAD Layer — spec §4 + §5.1

Manual chunked sliding-window state machine (thay cho sherpa_onnx.VoiceActivityDetector
queue-based cũ). Tách riêng "speech probability model" (Silero ONNX) khỏi state machine
chunk-counting để có thể swap implementation (spec: "Có thể swap implementation (SileroVAD
hoặc khác) mà không ảnh hưởng tầng trên") và để unit test dễ dàng bằng fake model.

Chunk: 32ms (512 samples @ 16kHz).
Window: 320ms (10 chunk), hop 160ms (5 chunk) → overlap 160ms giữa 2 window liên tiếp
(spec §5.1, khớp ví dụ Window1 0-320ms / Window2 160-480ms / Window3 320-640ms).

threshold quyết định speech/silence mỗi chunk.
min_speech_ms: số ms speech liên tiếp tối thiểu để CONFIRM là speech thật (debounce, lọc
blip nhiễu ngắn) — chunk trong giai đoạn chưa confirm được giữ ở hàng đợi `_pending` để
khi confirm thì flush vào buffer chính, tránh mất phần đầu của từ.
min_silence_ms: số ms silence liên tiếp tối thiểu để CONFIRM là hết câu (hangover) — im
lặng ngắn hơn (khoảng nghỉ giữa từ) không làm reset buffer.

`window_start_ms` chỉ có ý nghĩa SO SÁNH trong cùng một utterance (đoạn speech liên tục
giữa hai lần silence-reset) — nó đếm từ 0 lại mỗi khi buffer bị silence-timeout xóa. Vì
vậy mỗi AudioWindow còn mang `utterance_id`: Translation Pipeline chỉ so sánh
`window_start_ms < last_freeze_ms` khi hai window đến từ CÙNG utterance; window của
utterance mới luôn được coi là không overlap với freeze cũ (đúng thực tế — sau một
khoảng lặng đủ dài, không còn gì để strip).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import onnxruntime as ort

from config_manager import VadConfig

CHUNK_MS = 32
WINDOW_CHUNKS = 10  # 320ms
HOP_CHUNKS = 5  # 160ms
WINDOW_MS = WINDOW_CHUNKS * CHUNK_MS


@dataclass(frozen=True)
class AudioWindow:
    samples: np.ndarray
    window_start_ms: int
    utterance_id: int


class SpeechProbabilityModel(Protocol):
    def predict(self, chunk: np.ndarray) -> float: ...
    def reset(self) -> None: ...


class SileroSpeechProbabilityModel:
    """Silero VAD onnx: input x[1,512] + recurrent state h[2,1,64]/c[2,1,64] → prob[1,1]."""

    def __init__(self, model_path: str, sample_rate: int = 16000):
        self._session = ort.InferenceSession(model_path)
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def predict(self, chunk: np.ndarray) -> float:
        x = chunk.reshape(1, -1).astype(np.float32)
        outputs = self._session.run(
            None,
            {"x": x, "h": self._h, "c": self._c},
        )
        prob, self._h, self._c = outputs[0], outputs[1], outputs[2]
        return float(prob[0][0])


class ChunkedSlidingVAD:
    def __init__(
        self,
        model: SpeechProbabilityModel,
        config: VadConfig,
        sample_rate: int = 16000,
        chunk_ms: int = CHUNK_MS,
    ):
        self._model = model
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)

        self.threshold = config.threshold
        self.min_silence_chunks = self._ms_to_chunks(config.min_silence_ms)
        self.min_speech_chunks = self._ms_to_chunks(config.min_speech_ms)

        self._pending: list[np.ndarray] = []
        self._buffer: list[np.ndarray] = []
        self._speech_run = 0
        self._silence_run = 0
        self._speech_confirmed = False
        self._total_speech_chunks = 0
        self._last_forward_count = 0
        self._utterance_id = 0

    def _ms_to_chunks(self, ms: int) -> int:
        return max(1, round(ms / self.chunk_ms))

    def update_config(
        self,
        threshold: float | None = None,
        min_silence_ms: int | None = None,
        min_speech_ms: int | None = None,
    ) -> None:
        if threshold is not None:
            self.threshold = threshold
        if min_silence_ms is not None:
            self.min_silence_chunks = self._ms_to_chunks(min_silence_ms)
        if min_speech_ms is not None:
            self.min_speech_chunks = self._ms_to_chunks(min_speech_ms)

    def push_chunk(self, chunk: np.ndarray) -> AudioWindow | None:
        prob = self._model.predict(chunk)
        if prob >= self.threshold:
            self._on_speech_chunk(chunk)
            return self._maybe_forward()
        self._on_silence_chunk()
        return None

    def _on_speech_chunk(self, chunk: np.ndarray) -> None:
        self._silence_run = 0
        self._speech_run += 1

        if not self._speech_confirmed:
            self._pending.append(chunk)
            if self._speech_run >= self.min_speech_chunks:
                self._speech_confirmed = True
                self._buffer.extend(self._pending)
                self._total_speech_chunks += len(self._pending)
                self._pending = []
        else:
            self._buffer.append(chunk)
            self._total_speech_chunks += 1

    def _on_silence_chunk(self) -> None:
        self._speech_run = 0

        if not self._speech_confirmed:
            # Blip nhiễu chưa từng confirm — silence hủy ngay, không cần hangover.
            self._pending = []
            return

        self._silence_run += 1
        if self._silence_run >= self.min_silence_chunks:
            self._reset_accumulation()

    def _maybe_forward(self) -> AudioWindow | None:
        if not self._speech_confirmed:
            return None

        count = self._total_speech_chunks
        if count < WINDOW_CHUNKS:
            return None
        if count == self._last_forward_count:
            return None
        if (count - WINDOW_CHUNKS) % HOP_CHUNKS != 0:
            return None

        window_chunks = self._buffer[-WINDOW_CHUNKS:]
        window_start_ms = (count - WINDOW_CHUNKS) * self.chunk_ms
        self._last_forward_count = count
        samples = np.concatenate(window_chunks)
        return AudioWindow(samples=samples, window_start_ms=window_start_ms, utterance_id=self._utterance_id)

    def _reset_accumulation(self) -> None:
        self._buffer = []
        self._pending = []
        self._speech_confirmed = False
        self._speech_run = 0
        self._silence_run = 0
        self._total_speech_chunks = 0
        self._last_forward_count = 0
        self._utterance_id += 1

    def reset(self) -> None:
        self._reset_accumulation()
        self._model.reset()
