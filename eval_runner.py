"""
Eval runner — chạy audio qua đúng pipeline thật (VAD → ASR → Translation) và tính
điểm so với ground truth. Không dùng WebSocket: audio đã có sẵn toàn bộ trong bộ
nhớ (khác với server nhận từng frame qua network), nên xử lý trực tiếp theo chunk
32ms từ đầu đến cuối mảng — cùng logic chunk/forward window như
asr_translate_server.py nhưng không cần buffer "leftover" giữa các message.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from asr_adapter import ASRAdapter
from eval_audio import load_audio_16k_mono
from eval_metrics import (
    TranslationQualityResult,
    WerResult,
    compute_translation_quality,
    compute_wer,
)
from translation_pipeline import TranslationPipeline
from vad import ChunkedSlidingVAD

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = ("*.wav", "*.flac", "*.mp3", "*.ogg")


@dataclass
class SampleRunResult:
    asr_transcript: str
    translation_text: str
    segment_latencies_s: list[float] = field(default_factory=list)
    pending_leftover_words: int = 0
    dropped_tail_ms: float = 0.0


async def run_pipeline_over_audio(
    audio: np.ndarray,
    sample_rate: int,
    vad: ChunkedSlidingVAD,
    asr_adapter: ASRAdapter,
    pipeline: TranslationPipeline,
) -> SampleRunResult:
    chunk_samples = vad.chunk_samples
    n_chunks = audio.size // chunk_samples

    asr_parts: list[str] = []
    segment_latencies_s: list[float] = []

    for i in range(n_chunks):
        chunk = audio[i * chunk_samples:(i + 1) * chunk_samples]

        try:
            window = vad.push_chunk(chunk)
        except Exception:
            logger.exception("VAD error on chunk %d, skipping", i)
            continue

        if window is None:
            continue

        t0 = time.monotonic()
        try:
            asr_result = asr_adapter.transcribe(window.samples, window.window_start_ms, window.utterance_id)
        except Exception:
            logger.exception("ASR error on window at %dms, skipping", window.window_start_ms)
            continue

        if not asr_result.text:
            continue

        events = await pipeline.process_window(asr_result)
        elapsed = time.monotonic() - t0

        for event in events:
            if event["type"] == "asr_partial":
                asr_parts.append(event["text"])
            elif event["type"] == "translation":
                segment_latencies_s.append(elapsed)

    dropped_tail_samples = audio.size - n_chunks * chunk_samples
    dropped_tail_ms = dropped_tail_samples / sample_rate * 1000

    return SampleRunResult(
        asr_transcript=" ".join(asr_parts),
        translation_text=pipeline.full_en,
        segment_latencies_s=segment_latencies_s,
        pending_leftover_words=len(pipeline.pending_buffer.split()),
        dropped_tail_ms=dropped_tail_ms,
    )


def load_sample(sample_dir: str) -> tuple[np.ndarray, str, str]:
    """Đọc audio.* + transcript.txt + translation.txt từ 1 thư mục mẫu."""
    sample_path = Path(sample_dir)

    audio_candidates = sorted(
        p for ext in AUDIO_EXTENSIONS for p in sample_path.glob(ext)
    )
    if not audio_candidates:
        raise FileNotFoundError(
            f"Không tìm thấy file audio (wav/flac/mp3/ogg) trong {sample_dir}"
        )

    transcript_path = sample_path / "transcript.txt"
    translation_path = sample_path / "translation.txt"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Thiếu transcript.txt trong {sample_dir}")
    if not translation_path.exists():
        raise FileNotFoundError(f"Thiếu translation.txt trong {sample_dir}")

    audio = load_audio_16k_mono(str(audio_candidates[0]))
    reference_transcript = transcript_path.read_text(encoding="utf-8").strip()
    reference_translation = translation_path.read_text(encoding="utf-8").strip()
    return audio, reference_transcript, reference_translation


def summarize_latencies(latencies: list[float]) -> dict:
    if not latencies:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}

    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "count": len(latencies),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


@dataclass
class SampleResult:
    name: str
    reference_transcript: str
    hypothesis_transcript: str
    reference_translation: str
    hypothesis_translation: str
    wer: WerResult
    quality: TranslationQualityResult
    latency_seconds: dict
    raw_latencies_s: list[float]
    pending_leftover_words: int
    dropped_tail_ms: float


@dataclass
class DatasetReport:
    samples: list[SampleResult]
    aggregate_wer: WerResult
    aggregate_quality: TranslationQualityResult
    aggregate_latency_seconds: dict


async def evaluate_sample(
    sample_dir: str,
    asr_adapter: ASRAdapter,
    make_vad: Callable[[], ChunkedSlidingVAD],
    make_pipeline: Callable[[], TranslationPipeline],
    sample_rate: int = 16000,
) -> SampleResult:
    audio, reference_transcript, reference_translation = load_sample(sample_dir)

    vad = make_vad()
    pipeline = make_pipeline()
    run_result = await run_pipeline_over_audio(audio, sample_rate, vad, asr_adapter, pipeline)

    wer = compute_wer([reference_transcript], [run_result.asr_transcript])
    quality = compute_translation_quality([reference_translation], [run_result.translation_text])

    return SampleResult(
        name=Path(sample_dir).name,
        reference_transcript=reference_transcript,
        hypothesis_transcript=run_result.asr_transcript,
        reference_translation=reference_translation,
        hypothesis_translation=run_result.translation_text,
        wer=wer,
        quality=quality,
        latency_seconds=summarize_latencies(run_result.segment_latencies_s),
        raw_latencies_s=run_result.segment_latencies_s,
        pending_leftover_words=run_result.pending_leftover_words,
        dropped_tail_ms=run_result.dropped_tail_ms,
    )


async def evaluate_dataset(
    data_dir: str,
    asr_adapter: ASRAdapter,
    make_vad: Callable[[], ChunkedSlidingVAD],
    make_pipeline: Callable[[], TranslationPipeline],
    sample_rate: int = 16000,
    on_sample_done: Callable[[SampleResult], None] | None = None,
) -> DatasetReport:
    sample_dirs = sorted(p for p in Path(data_dir).iterdir() if p.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(f"Không có thư mục mẫu nào trong {data_dir}")

    results: list[SampleResult] = []
    for sample_dir in sample_dirs:
        result = await evaluate_sample(str(sample_dir), asr_adapter, make_vad, make_pipeline, sample_rate)
        results.append(result)
        if on_sample_done:
            on_sample_done(result)

    aggregate_wer = compute_wer(
        [r.reference_transcript for r in results],
        [r.hypothesis_transcript for r in results],
    )
    aggregate_quality = compute_translation_quality(
        [r.reference_translation for r in results],
        [r.hypothesis_translation for r in results],
    )
    pooled_latencies = [lat for r in results for lat in r.raw_latencies_s]

    return DatasetReport(
        samples=results,
        aggregate_wer=aggregate_wer,
        aggregate_quality=aggregate_quality,
        aggregate_latency_seconds=summarize_latencies(pooled_latencies),
    )
