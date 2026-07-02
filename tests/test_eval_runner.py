import numpy as np
import pytest
import soundfile as sf

from asr_adapter import ASRResult
from config_manager import TranslationConfig, VadConfig
from eval_runner import (
    evaluate_dataset,
    evaluate_sample,
    load_sample,
    run_pipeline_over_audio,
    summarize_latencies,
)
from translation_pipeline import TranslationPipeline
from vad import CHUNK_MS, ChunkedSlidingVAD

SAMPLE_RATE = 16000
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)


class AlwaysSpeechModel:
    """Fake VAD probability model: mọi chunk đều là speech, không cần script cố định
    độ dài — tránh phải tính trước đúng số chunk cho từng audio length khác nhau
    giữa các test/mẫu."""

    def predict(self, chunk: np.ndarray) -> float:
        return 1.0

    def reset(self) -> None:
        pass


class MappedASRAdapter:
    """Fake ASRAdapter: tra cứu text theo (utterance_id, window_start_ms) thay vì
    giữ con trỏ tuần tự — nhờ vậy dùng CHUNG được cho nhiều mẫu trong 1 test
    evaluate_dataset (utterance_id/window_start_ms reset về 0 mỗi VAD mới, nên key
    không đụng nhau giữa các mẫu độc lập) mà không cần lo state bị lẫn."""

    def __init__(self, mapping: dict[tuple[int, int], str]):
        self._mapping = mapping

    def transcribe(self, audio_window, window_start_ms, utterance_id=0):
        text = self._mapping.get((utterance_id, window_start_ms), "")
        return ASRResult(text=text, window_start_ms=window_start_ms, utterance_id=utterance_id)


class ScriptedLLM:
    """Cùng pattern với tests/test_translation_pipeline.py — trả lần lượt các response
    đã định sẵn theo thứ tự gọi."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._i = 0

    async def complete(self, system_prompt, user_message, stream=True):
        response = self._responses[self._i]
        self._i += 1
        yield response


def make_vad(min_silence_ms=32, min_speech_ms=32) -> ChunkedSlidingVAD:
    config = VadConfig(model_path="unused", threshold=0.5, min_silence_ms=min_silence_ms, min_speech_ms=min_speech_ms)
    return ChunkedSlidingVAD(AlwaysSpeechModel(), config, sample_rate=SAMPLE_RATE)


def zeros_audio(n_chunks: int, extra_samples: int = 0) -> np.ndarray:
    return np.zeros(n_chunks * CHUNK_SAMPLES + extra_samples, dtype=np.float32)


# ── run_pipeline_over_audio ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_pipeline_over_audio_basic_flow_two_windows():
    # 15 chunk speech liên tục -> window đầu tại chunk 10 (window_start_ms=0),
    # window thứ hai tại chunk 15 (window_start_ms=160, do HOP_CHUNKS=5*32ms).
    audio = zeros_audio(15)
    vad = make_vad()
    asr_adapter = MappedASRAdapter({
        (0, 0): "hôm nay là",
        (0, 160): "hôm nay là một ngày",
    })
    pipeline = TranslationPipeline(
        llm_adapter=ScriptedLLM(["[WAIT]", "Today is a nice day."]),
        config=TranslationConfig(),
    )

    result = await run_pipeline_over_audio(audio, SAMPLE_RATE, vad, asr_adapter, pipeline)

    assert result.asr_transcript == "hôm nay là một ngày"
    assert result.translation_text == "Today is a nice day."
    assert len(result.segment_latencies_s) == 1
    assert result.segment_latencies_s[0] >= 0
    assert result.pending_leftover_words == 0
    assert result.dropped_tail_ms == 0.0


@pytest.mark.asyncio
async def test_run_pipeline_over_audio_reports_dropped_tail_ms():
    audio = zeros_audio(15, extra_samples=100)
    vad = make_vad()
    asr_adapter = MappedASRAdapter({(0, 0): "a", (0, 160): "a b"})
    pipeline = TranslationPipeline(llm_adapter=ScriptedLLM(["[WAIT]", "[WAIT]"]), config=TranslationConfig())

    result = await run_pipeline_over_audio(audio, SAMPLE_RATE, vad, asr_adapter, pipeline)

    assert result.dropped_tail_ms == pytest.approx(100 / SAMPLE_RATE * 1000)


@pytest.mark.asyncio
async def test_run_pipeline_over_audio_reports_pending_leftover_when_never_frozen():
    audio = zeros_audio(15)
    vad = make_vad()
    asr_adapter = MappedASRAdapter({(0, 0): "hôm nay là", (0, 160): "hôm nay là một ngày"})
    # LLM luôn [WAIT] -> không bao giờ freeze -> pending_buffer giữ nguyên window cuối
    pipeline = TranslationPipeline(llm_adapter=ScriptedLLM(["[WAIT]", "[WAIT]"]), config=TranslationConfig())

    result = await run_pipeline_over_audio(audio, SAMPLE_RATE, vad, asr_adapter, pipeline)

    assert result.translation_text == ""
    assert result.pending_leftover_words == 4  # "hôm nay là một ngày"
    assert result.segment_latencies_s == []


# ── summarize_latencies ──────────────────────────────────────────────────────

def test_summarize_latencies_empty_list():
    summary = summarize_latencies([])
    assert summary == {"count": 0, "mean": None, "median": None, "p95": None, "max": None}


def test_summarize_latencies_basic_stats():
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["mean"] == 2.5
    assert summary["median"] == 2.5
    assert summary["max"] == 4.0


# ── load_sample ──────────────────────────────────────────────────────────────

def _write_sample(tmp_path, name, n_chunks=15, transcript="hôm nay là một ngày", translation="Today is a nice day."):
    sample_dir = tmp_path / name
    sample_dir.mkdir()
    sf.write(str(sample_dir / "audio.wav"), zeros_audio(n_chunks), SAMPLE_RATE)
    (sample_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    (sample_dir / "translation.txt").write_text(translation, encoding="utf-8")
    return sample_dir


def test_load_sample_reads_audio_and_ground_truth(tmp_path):
    sample_dir = _write_sample(tmp_path, "sample_001")

    audio, transcript, translation = load_sample(str(sample_dir))

    assert isinstance(audio, np.ndarray)
    assert transcript == "hôm nay là một ngày"
    assert translation == "Today is a nice day."


def test_load_sample_missing_audio_raises(tmp_path):
    sample_dir = tmp_path / "no_audio"
    sample_dir.mkdir()
    (sample_dir / "transcript.txt").write_text("x", encoding="utf-8")
    (sample_dir / "translation.txt").write_text("x", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_sample(str(sample_dir))


def test_load_sample_missing_transcript_raises(tmp_path):
    sample_dir = tmp_path / "no_transcript"
    sample_dir.mkdir()
    sf.write(str(sample_dir / "audio.wav"), zeros_audio(1), SAMPLE_RATE)
    (sample_dir / "translation.txt").write_text("x", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_sample(str(sample_dir))


def test_load_sample_missing_translation_raises(tmp_path):
    sample_dir = tmp_path / "no_translation"
    sample_dir.mkdir()
    sf.write(str(sample_dir / "audio.wav"), zeros_audio(1), SAMPLE_RATE)
    (sample_dir / "transcript.txt").write_text("x", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_sample(str(sample_dir))


# ── evaluate_sample / evaluate_dataset ──────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluate_sample_end_to_end(tmp_path):
    sample_dir = _write_sample(tmp_path, "sample_001")
    asr_adapter = MappedASRAdapter({(0, 0): "hôm nay là", (0, 160): "hôm nay là một ngày"})

    result = await evaluate_sample(
        str(sample_dir),
        asr_adapter,
        make_vad=make_vad,
        make_pipeline=lambda: TranslationPipeline(
            llm_adapter=ScriptedLLM(["[WAIT]", "Today is a nice day."]),
            config=TranslationConfig(),
        ),
        sample_rate=SAMPLE_RATE,
    )

    assert result.name == "sample_001"
    assert result.hypothesis_transcript == "hôm nay là một ngày"
    assert result.hypothesis_translation == "Today is a nice day."
    assert result.wer.wer == 0.0
    assert result.quality.bleu > 90


@pytest.mark.asyncio
async def test_evaluate_dataset_aggregates_across_samples(tmp_path):
    # 2 mẫu độc lập, cùng audio length (đủ để MappedASRAdapter dùng chung key an toàn
    # vì utterance_id/window_start_ms reset về 0 cho mỗi VAD mới của từng mẫu).
    _write_sample(
        tmp_path, "sample_a",
        transcript="hôm nay là một ngày",  # khớp hệt hypothesis -> WER=0 cho mẫu này
        translation="Today is a nice day.",
    )
    _write_sample(
        tmp_path, "sample_b",
        transcript="hôm nay là một ngày đẹp trời",  # hypothesis thiếu 2 từ cuối -> có deletions
        translation="Today is a nice day and beautiful.",
    )

    asr_adapter = MappedASRAdapter({(0, 0): "hôm nay là", (0, 160): "hôm nay là một ngày"})

    def make_pipeline():
        return TranslationPipeline(
            llm_adapter=ScriptedLLM(["[WAIT]", "Today is a nice day."]),
            config=TranslationConfig(),
        )

    seen_names = []
    report = await evaluate_dataset(
        str(tmp_path),
        asr_adapter,
        make_vad=make_vad,
        make_pipeline=make_pipeline,
        sample_rate=SAMPLE_RATE,
        on_sample_done=lambda r: seen_names.append(r.name),
    )

    assert len(report.samples) == 2
    assert seen_names == ["sample_a", "sample_b"]  # callback invoked incrementally per sample

    # sample_a: ref "hôm nay là một ngày" (5 từ) khớp hệt hypothesis -> 0 lỗi.
    # sample_b: ref "hôm nay là một ngày đẹp trời" (7 từ), hypothesis chỉ có 5 từ đầu
    # -> 2 deletions. Pooled WER = tổng lỗi / tổng từ ref = 2 / (5+7) = 2/12.
    assert report.aggregate_wer.wer == pytest.approx(2 / 12)

    # Mỗi mẫu góp đúng 1 latency (1 translation event mỗi mẫu) -> pooled count=2
    assert report.aggregate_latency_seconds["count"] == 2


@pytest.mark.asyncio
async def test_evaluate_dataset_raises_on_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        await evaluate_dataset(
            str(tmp_path),
            asr_adapter=MappedASRAdapter({}),
            make_vad=make_vad,
            make_pipeline=lambda: TranslationPipeline(llm_adapter=ScriptedLLM([]), config=TranslationConfig()),
        )
