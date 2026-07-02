"""
Rolling ASR + LocalAgreement for realtime offline recognizers.

The Sherpa offline recognizer has no streaming decoder state, so we periodically
decode the current VAD utterance from the beginning. LocalAgreement commits only
the word prefix shared by the last N hypotheses; the rest remains replaceable
partial text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from asr_adapter import ASRAdapter
from config_manager import VadConfig
from diff_utils import diff_new_suffix, tokenize_words
from vad import CHUNK_MS, SpeechProbabilityModel


@dataclass(frozen=True)
class StreamingASREvent:
    type: str
    text: str
    utterance_id: int
    audio_ms: int


class LocalAgreementBuffer:
    def __init__(self, agreement_n: int = 3):
        self.agreement_n = agreement_n
        self.history: list[str] = []
        self.committed = ""

    def push(self, new_hypothesis: str) -> tuple[str, str]:
        self.history.append(new_hypothesis)
        if len(self.history) < self.agreement_n:
            return "", new_hypothesis

        agreed_prefix = self._longest_common_word_prefix(self.history[-self.agreement_n:])
        new_committed = diff_new_suffix(self.committed, agreed_prefix)
        if new_committed:
            self.committed = agreed_prefix
            return new_committed, diff_new_suffix(agreed_prefix, new_hypothesis)

        return "", diff_new_suffix(self.committed, new_hypothesis)

    def flush(self, final_hypothesis: str) -> tuple[str, str]:
        new_committed = diff_new_suffix(self.committed, final_hypothesis)
        self.committed = final_hypothesis
        return new_committed, ""

    def reset(self) -> None:
        self.history = []
        self.committed = ""

    def _longest_common_word_prefix(self, strings: list[str]) -> str:
        token_lists = [tokenize_words(text) for text in strings]
        if not token_lists:
            return ""

        shortest_len = min(len(tokens) for tokens in token_lists)
        prefix: list[str] = []
        for i in range(shortest_len):
            word = token_lists[0][i]
            if any(tokens[i] != word for tokens in token_lists[1:]):
                break
            prefix.append(word)
        return " ".join(prefix)


@dataclass
class RollingLocalAgreementASR:
    speech_model: SpeechProbabilityModel
    vad_config: VadConfig
    asr_adapter: ASRAdapter
    sample_rate: int = 16000
    chunk_ms: int = CHUNK_MS
    decode_hop_ms: int = 960
    agreement_n: int = 3

    _pending: list[np.ndarray] = field(default_factory=list)
    _buffer: list[np.ndarray] = field(default_factory=list)
    _speech_run: int = 0
    _silence_run: int = 0
    _speech_confirmed: bool = False
    _utterance_id: int = 0
    _utterance_start_chunk: int = 0
    _total_chunks_seen: int = 0
    _last_decode_buffer_chunks: int = 0
    _agreement: LocalAgreementBuffer = field(init=False)

    def __post_init__(self) -> None:
        self.chunk_samples = int(self.sample_rate * self.chunk_ms / 1000)
        self.min_silence_chunks = self._ms_to_chunks(self.vad_config.min_silence_ms)
        self.min_speech_chunks = self._ms_to_chunks(self.vad_config.min_speech_ms)
        self.decode_hop_chunks = self._ms_to_chunks(self.decode_hop_ms)
        self.threshold = self.vad_config.threshold
        self._agreement = LocalAgreementBuffer(self.agreement_n)

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

    def push_chunk(self, chunk: np.ndarray) -> list[StreamingASREvent]:
        events: list[StreamingASREvent] = []
        chunk_index = self._total_chunks_seen
        self._total_chunks_seen += 1

        prob = self.speech_model.predict(chunk)
        if prob >= self.threshold:
            self._on_speech_chunk(chunk, chunk_index)
            events.extend(self._maybe_decode_partial())
            return events

        events.extend(self._on_silence_chunk(chunk_index))
        return events

    def flush(self) -> list[StreamingASREvent]:
        if not self._speech_confirmed or not self._buffer:
            self._reset_utterance()
            return []
        return self._finalize_utterance(self._total_chunks_seen - 1)

    def reset(self) -> None:
        self._pending = []
        self._buffer = []
        self._speech_run = 0
        self._silence_run = 0
        self._speech_confirmed = False
        self._utterance_id = 0
        self._utterance_start_chunk = 0
        self._total_chunks_seen = 0
        self._last_decode_buffer_chunks = 0
        self._agreement.reset()
        self.speech_model.reset()

    def _on_speech_chunk(self, chunk: np.ndarray, chunk_index: int) -> None:
        self._silence_run = 0
        self._speech_run += 1

        if not self._speech_confirmed:
            if not self._pending:
                self._utterance_start_chunk = chunk_index
            self._pending.append(chunk)
            if self._speech_run >= self.min_speech_chunks:
                self._speech_confirmed = True
                self._buffer.extend(self._pending)
                self._pending = []
            return

        self._buffer.append(chunk)

    def _on_silence_chunk(self, chunk_index: int) -> list[StreamingASREvent]:
        self._speech_run = 0
        if not self._speech_confirmed:
            self._pending = []
            return []

        self._silence_run += 1
        if self._silence_run < self.min_silence_chunks:
            return []
        return self._finalize_utterance(chunk_index)

    def _maybe_decode_partial(self) -> list[StreamingASREvent]:
        if not self._speech_confirmed or not self._buffer:
            return []
        if len(self._buffer) - self._last_decode_buffer_chunks < self.decode_hop_chunks:
            return []

        text = self._decode_buffer()
        self._last_decode_buffer_chunks = len(self._buffer)
        committed, partial = self._agreement.push(text)
        audio_ms = self._current_audio_ms()

        events: list[StreamingASREvent] = []
        if committed:
            events.append(StreamingASREvent("commit", committed, self._utterance_id, audio_ms))
        events.append(StreamingASREvent("partial", partial, self._utterance_id, audio_ms))
        return events

    def _finalize_utterance(self, chunk_index: int) -> list[StreamingASREvent]:
        text = self._decode_buffer()
        committed, partial = self._agreement.flush(text)
        audio_ms = (chunk_index + 1) * self.chunk_ms

        events: list[StreamingASREvent] = []
        if committed:
            events.append(StreamingASREvent("commit", committed, self._utterance_id, audio_ms))
        events.append(StreamingASREvent("partial", partial, self._utterance_id, audio_ms))
        self._reset_utterance()
        return events

    def _decode_buffer(self) -> str:
        samples = np.concatenate(self._buffer)
        result = self.asr_adapter.transcribe(samples, self._current_audio_ms(), self._utterance_id)
        return result.text

    def _current_audio_ms(self) -> int:
        return (self._utterance_start_chunk + len(self._buffer)) * self.chunk_ms

    def _reset_utterance(self) -> None:
        self._pending = []
        self._buffer = []
        self._speech_run = 0
        self._silence_run = 0
        self._speech_confirmed = False
        self._last_decode_buffer_chunks = 0
        self._utterance_id += 1
        self._utterance_start_chunk = self._total_chunks_seen
        self._agreement.reset()
