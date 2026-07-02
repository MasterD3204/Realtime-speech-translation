"""
ASR-only comparison for streaming strategies.

This script intentionally avoids the LLM path so the numbers isolate ASR +
chunking/commit behavior:

- full_file: decode the whole audio once.
- sliding_current: current VAD sliding windows + word-diff stitching.
- vad_utterance: collect one VAD utterance, then decode it once.
- rolling_replaceable: decode the growing utterance periodically for preview,
  but commit only the final utterance text at VAD end.
- stable_prefix: decode the growing utterance periodically, commit only words
  that are stable across consecutive decodes, and flush the utterance at VAD end.
- local_agreement_N: decode the growing utterance periodically, commit only the
  prefix agreed by the last N hypotheses, and flush the utterance at VAD end.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from asr_adapter import ASRAdapter, SherpaOnnxAdapter
from config_manager import ConfigManager, TranslationConfig, VadConfig
from diff_utils import diff_new_suffix, tokenize_words
from eval_audio import load_audio_16k_mono
from eval_metrics import WerResult, compute_wer
from eval_runner import AUDIO_EXTENSIONS
from translation_pipeline import TranslationPipeline
from vad import CHUNK_MS, ChunkedSlidingVAD, SileroSpeechProbabilityModel, SpeechProbabilityModel

SAMPLE_RATE = 16000


class WaitLLM:
    async def complete(self, system_prompt: str, user_message: str, stream: bool = True):
        yield "[WAIT]"


@dataclass(frozen=True)
class CommitEvent:
    text: str
    audio_ms: float


@dataclass(frozen=True)
class StrategyRun:
    transcript: str
    commits: list[CommitEvent] = field(default_factory=list)
    utterances: int | None = None
    partial_decodes: int | None = None


@dataclass(frozen=True)
class LatencySummary:
    commit_count: int
    first_commit_ms: float | None
    final_commit_ms: float | None
    mean_commit_gap_ms: float | None
    wall_time_s: float
    rtf: float


@dataclass(frozen=True)
class Utterance:
    samples: np.ndarray
    utterance_id: int
    start_chunk: int
    end_chunk: int


@dataclass
class VADUtteranceSegmenter:
    model: SpeechProbabilityModel
    config: VadConfig
    sample_rate: int = SAMPLE_RATE
    chunk_ms: int = CHUNK_MS

    pending: list[np.ndarray] = field(default_factory=list)
    buffer: list[np.ndarray] = field(default_factory=list)
    speech_run: int = 0
    silence_run: int = 0
    speech_confirmed: bool = False
    utterance_id: int = 0
    utterance_start_chunk: int = 0

    def __post_init__(self) -> None:
        self.chunk_samples = int(self.sample_rate * self.chunk_ms / 1000)
        self.min_silence_chunks = max(1, round(self.config.min_silence_ms / self.chunk_ms))
        self.min_speech_chunks = max(1, round(self.config.min_speech_ms / self.chunk_ms))

    def push_chunk(self, chunk: np.ndarray, chunk_index: int) -> Utterance | None:
        prob = self.model.predict(chunk)
        if prob >= self.config.threshold:
            self._on_speech_chunk(chunk, chunk_index)
            return None
        return self._on_silence_chunk(chunk_index)

    def flush(self, chunk_index: int) -> Utterance | None:
        if not self.speech_confirmed or not self.buffer:
            self._reset(next_start_chunk=chunk_index + 1)
            return None
        return self._finalize(chunk_index)

    def _on_speech_chunk(self, chunk: np.ndarray, chunk_index: int) -> None:
        self.silence_run = 0
        self.speech_run += 1

        if not self.speech_confirmed:
            if not self.pending:
                self.utterance_start_chunk = chunk_index
            self.pending.append(chunk)
            if self.speech_run >= self.min_speech_chunks:
                self.speech_confirmed = True
                self.buffer.extend(self.pending)
                self.pending = []
            return

        self.buffer.append(chunk)

    def _on_silence_chunk(self, chunk_index: int) -> Utterance | None:
        self.speech_run = 0
        if not self.speech_confirmed:
            self.pending = []
            return None

        self.silence_run += 1
        if self.silence_run < self.min_silence_chunks:
            return None
        return self._finalize(chunk_index)

    def _finalize(self, chunk_index: int) -> Utterance:
        samples = np.concatenate(self.buffer)
        utterance = Utterance(
            samples=samples,
            utterance_id=self.utterance_id,
            start_chunk=self.utterance_start_chunk,
            end_chunk=chunk_index,
        )
        self._reset(next_start_chunk=chunk_index + 1)
        return utterance

    def _reset(self, next_start_chunk: int) -> None:
        self.pending = []
        self.buffer = []
        self.speech_run = 0
        self.silence_run = 0
        self.speech_confirmed = False
        self.utterance_start_chunk = next_start_chunk
        self.utterance_id += 1


def make_segmenter(config: VadConfig) -> VADUtteranceSegmenter:
    return VADUtteranceSegmenter(
        model=SileroSpeechProbabilityModel(config.model_path, sample_rate=SAMPLE_RATE),
        config=config,
        sample_rate=SAMPLE_RATE,
    )


def iter_vad_utterances(audio: np.ndarray, config: VadConfig) -> tuple[list[Utterance], float]:
    segmenter = make_segmenter(config)
    n_chunks = audio.size // segmenter.chunk_samples
    utterances: list[Utterance] = []

    for i in range(n_chunks):
        chunk = audio[i * segmenter.chunk_samples:(i + 1) * segmenter.chunk_samples]
        utterance = segmenter.push_chunk(chunk, i)
        if utterance is not None:
            utterances.append(utterance)

    tail = segmenter.flush(n_chunks)
    if tail is not None:
        utterances.append(tail)

    dropped_tail_samples = audio.size - n_chunks * segmenter.chunk_samples
    return utterances, dropped_tail_samples / SAMPLE_RATE * 1000


def chunk_end_ms(chunk_index: int) -> float:
    return (chunk_index + 1) * CHUNK_MS


def transcribe_full_file(audio: np.ndarray, asr: ASRAdapter) -> StrategyRun:
    text = asr.transcribe(audio, window_start_ms=0, utterance_id=0).text
    audio_ms = audio.size / SAMPLE_RATE * 1000
    commits = [CommitEvent(text=text, audio_ms=audio_ms)] if text else []
    return StrategyRun(text, commits=commits)


async def transcribe_current_sliding(audio: np.ndarray, config: VadConfig, asr: ASRAdapter) -> StrategyRun:
    vad = ChunkedSlidingVAD(make_segmenter(config).model, config, sample_rate=SAMPLE_RATE)
    pipeline = TranslationPipeline(WaitLLM(), TranslationConfig())
    n_chunks = audio.size // vad.chunk_samples
    commits: list[CommitEvent] = []

    for i in range(n_chunks):
        chunk = audio[i * vad.chunk_samples:(i + 1) * vad.chunk_samples]
        window = vad.push_chunk(chunk)
        if window is None:
            continue

        asr_result = asr.transcribe(window.samples, window.window_start_ms, window.utterance_id)
        if not asr_result.text:
            continue

        events = await pipeline.process_window(asr_result)
        event_ms = chunk_end_ms(i)
        for event in events:
            if event["type"] == "asr_partial" and event["text"]:
                commits.append(CommitEvent(text=event["text"], audio_ms=event_ms))

    transcript = " ".join(event.text for event in commits)
    return StrategyRun(transcript, commits=commits)


def transcribe_vad_utterances(audio: np.ndarray, config: VadConfig, asr: ASRAdapter) -> StrategyRun:
    utterances, _ = iter_vad_utterances(audio, config)
    commits: list[CommitEvent] = []
    for utterance in utterances:
        text = asr.transcribe(utterance.samples, window_start_ms=0, utterance_id=utterance.utterance_id).text
        if text:
            commits.append(CommitEvent(text=text, audio_ms=chunk_end_ms(utterance.end_chunk)))
    return StrategyRun(" ".join(event.text for event in commits), commits=commits, utterances=len(utterances))


def transcribe_rolling_replaceable(
    audio: np.ndarray,
    config: VadConfig,
    asr: ASRAdapter,
    decode_hop_ms: int = 960,
) -> StrategyRun:
    segmenter = make_segmenter(config)
    n_chunks = audio.size // segmenter.chunk_samples
    decode_hop_chunks = max(1, round(decode_hop_ms / CHUNK_MS))
    commits: list[CommitEvent] = []
    partial_decode_count = 0
    utterance_count = 0

    def flush_utterance(utterance: Utterance) -> None:
        nonlocal utterance_count
        final_text = asr.transcribe(utterance.samples, window_start_ms=0, utterance_id=utterance.utterance_id).text
        if final_text:
            commits.append(CommitEvent(text=final_text, audio_ms=chunk_end_ms(utterance.end_chunk)))
        utterance_count += 1

    for i in range(n_chunks):
        chunk = audio[i * segmenter.chunk_samples:(i + 1) * segmenter.chunk_samples]
        utterance = segmenter.push_chunk(chunk, i)
        if utterance is not None:
            flush_utterance(utterance)
            continue

        if not segmenter.speech_confirmed or not segmenter.buffer:
            continue
        if len(segmenter.buffer) % decode_hop_chunks != 0:
            continue

        current_audio = np.concatenate(segmenter.buffer)
        asr.transcribe(current_audio, window_start_ms=0, utterance_id=segmenter.utterance_id)
        partial_decode_count += 1

    tail = segmenter.flush(n_chunks)
    if tail is not None:
        flush_utterance(tail)

    return StrategyRun(
        " ".join(event.text for event in commits),
        commits=commits,
        utterances=utterance_count,
        partial_decodes=partial_decode_count,
    )


def common_prefix_word_count(a: str, b: str) -> int:
    words_a = tokenize_words(a)
    words_b = tokenize_words(b)
    count = 0
    for word_a, word_b in zip(words_a, words_b):
        if word_a != word_b:
            break
        count += 1
    return count


def transcribe_stable_prefix(
    audio: np.ndarray,
    config: VadConfig,
    asr: ASRAdapter,
    decode_hop_ms: int = 960,
    stable_margin_words: int = 3,
) -> StrategyRun:
    segmenter = make_segmenter(config)
    n_chunks = audio.size // segmenter.chunk_samples
    decode_hop_chunks = max(1, round(decode_hop_ms / CHUNK_MS))

    commits: list[CommitEvent] = []
    utterance_committed = ""
    previous_partial = ""
    partial_decode_count = 0
    utterance_count = 0

    def maybe_commit_stable(current_text: str) -> None:
        nonlocal utterance_committed
        if not previous_partial or not current_text:
            return
        stable_count = common_prefix_word_count(previous_partial, current_text)
        target_count = max(0, stable_count - stable_margin_words)
        already_count = len(tokenize_words(utterance_committed))
        if target_count <= already_count:
            return

        stable_words = tokenize_words(current_text)[:target_count]
        stable_text = " ".join(stable_words)
        new_text = diff_new_suffix(utterance_committed, stable_text)
        if new_text:
            current_audio_ms = len(segmenter.buffer) * CHUNK_MS + segmenter.utterance_start_chunk * CHUNK_MS
            commits.append(CommitEvent(text=new_text, audio_ms=current_audio_ms))
            utterance_committed = stable_text

    def flush_utterance(utterance: Utterance) -> None:
        nonlocal previous_partial, utterance_committed, utterance_count
        final_text = asr.transcribe(utterance.samples, window_start_ms=0, utterance_id=utterance.utterance_id).text
        new_text = diff_new_suffix(utterance_committed, final_text)
        if new_text:
            commits.append(CommitEvent(text=new_text, audio_ms=chunk_end_ms(utterance.end_chunk)))
        previous_partial = ""
        utterance_committed = ""
        utterance_count += 1

    for i in range(n_chunks):
        chunk = audio[i * segmenter.chunk_samples:(i + 1) * segmenter.chunk_samples]
        utterance = segmenter.push_chunk(chunk, i)
        if utterance is not None:
            flush_utterance(utterance)
            continue

        if not segmenter.speech_confirmed or not segmenter.buffer:
            continue
        if len(segmenter.buffer) % decode_hop_chunks != 0:
            continue

        current_audio = np.concatenate(segmenter.buffer)
        current_text = asr.transcribe(current_audio, window_start_ms=0, utterance_id=segmenter.utterance_id).text
        maybe_commit_stable(current_text)
        previous_partial = current_text
        partial_decode_count += 1

    tail = segmenter.flush(n_chunks)
    if tail is not None:
        flush_utterance(tail)

    return StrategyRun(
        " ".join(event.text for event in commits),
        commits=commits,
        utterances=utterance_count,
        partial_decodes=partial_decode_count,
    )


class LocalAgreementBuffer:
    def __init__(self, agreement_n: int = 2):
        self.agreement_n = agreement_n
        self.history: list[str] = []
        self.committed = ""

    def push(self, new_hypothesis: str) -> tuple[str, str]:
        self.history.append(new_hypothesis)
        if len(self.history) < self.agreement_n:
            return "", new_hypothesis

        recent = self.history[-self.agreement_n:]
        agreed_prefix = self._longest_common_word_prefix(recent)
        new_committed = diff_new_suffix(self.committed, agreed_prefix)
        if new_committed:
            self.committed = agreed_prefix
            partial = diff_new_suffix(agreed_prefix, new_hypothesis)
            return new_committed, partial

        partial = diff_new_suffix(self.committed, new_hypothesis)
        return "", partial

    def flush(self, final_hypothesis: str) -> str:
        return diff_new_suffix(self.committed, final_hypothesis)

    def _longest_common_word_prefix(self, strings: list[str]) -> str:
        if not strings:
            return ""
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


def transcribe_local_agreement(
    audio: np.ndarray,
    config: VadConfig,
    asr: ASRAdapter,
    agreement_n: int,
    decode_hop_ms: int = 960,
) -> StrategyRun:
    segmenter = make_segmenter(config)
    n_chunks = audio.size // segmenter.chunk_samples
    decode_hop_chunks = max(1, round(decode_hop_ms / CHUNK_MS))

    commits: list[CommitEvent] = []
    partial_decode_count = 0
    utterance_count = 0
    buffer = LocalAgreementBuffer(agreement_n=agreement_n)

    def flush_utterance(utterance: Utterance) -> None:
        nonlocal buffer, utterance_count
        final_text = asr.transcribe(utterance.samples, window_start_ms=0, utterance_id=utterance.utterance_id).text
        final_commit = buffer.flush(final_text)
        if final_commit:
            commits.append(CommitEvent(text=final_commit, audio_ms=chunk_end_ms(utterance.end_chunk)))
        buffer = LocalAgreementBuffer(agreement_n=agreement_n)
        utterance_count += 1

    for i in range(n_chunks):
        chunk = audio[i * segmenter.chunk_samples:(i + 1) * segmenter.chunk_samples]
        utterance = segmenter.push_chunk(chunk, i)
        if utterance is not None:
            flush_utterance(utterance)
            continue

        if not segmenter.speech_confirmed or not segmenter.buffer:
            continue
        if len(segmenter.buffer) % decode_hop_chunks != 0:
            continue

        current_audio = np.concatenate(segmenter.buffer)
        current_text = asr.transcribe(current_audio, window_start_ms=0, utterance_id=segmenter.utterance_id).text
        new_commit, _partial = buffer.push(current_text)
        if new_commit:
            commits.append(CommitEvent(text=new_commit, audio_ms=chunk_end_ms(i)))
        partial_decode_count += 1

    tail = segmenter.flush(n_chunks)
    if tail is not None:
        flush_utterance(tail)

    return StrategyRun(
        " ".join(event.text for event in commits),
        commits=commits,
        utterances=utterance_count,
        partial_decodes=partial_decode_count,
    )


@dataclass(frozen=True)
class StrategyResult:
    name: str
    hypothesis: str
    wer: WerResult
    latency: LatencySummary
    utterances: int | None = None
    partial_decodes: int | None = None


def summarize_latency(run: StrategyRun, wall_time_s: float, audio: np.ndarray) -> LatencySummary:
    commit_times = [event.audio_ms for event in run.commits]
    gaps = [
        commit_times[i] - commit_times[i - 1]
        for i in range(1, len(commit_times))
    ]
    audio_duration_s = audio.size / SAMPLE_RATE
    return LatencySummary(
        commit_count=len(run.commits),
        first_commit_ms=commit_times[0] if commit_times else None,
        final_commit_ms=commit_times[-1] if commit_times else None,
        mean_commit_gap_ms=float(np.mean(gaps)) if gaps else None,
        wall_time_s=wall_time_s,
        rtf=wall_time_s / audio_duration_s if audio_duration_s else 0.0,
    )


def build_result(name: str, reference: str, run: StrategyRun, wall_time_s: float, audio: np.ndarray) -> StrategyResult:
    return StrategyResult(
        name=name,
        hypothesis=run.transcript,
        wer=compute_wer([reference], [run.transcript]),
        latency=summarize_latency(run, wall_time_s, audio),
        utterances=run.utterances,
        partial_decodes=run.partial_decodes,
    )


async def timed_async(fn, *args) -> tuple[StrategyRun, float]:
    t0 = time.perf_counter()
    run = await fn(*args)
    return run, time.perf_counter() - t0


def timed(fn, *args) -> tuple[StrategyRun, float]:
    t0 = time.perf_counter()
    run = fn(*args)
    return run, time.perf_counter() - t0


async def evaluate_sample(
    sample_dir: Path,
    config: VadConfig,
    asr: ASRAdapter,
    decode_hop_ms: int,
) -> list[StrategyResult]:
    audio_candidates = sorted(p for ext in AUDIO_EXTENSIONS for p in sample_dir.glob(ext))
    if not audio_candidates:
        raise FileNotFoundError(f"Missing audio in {sample_dir}")
    audio = load_audio_16k_mono(str(audio_candidates[0]))
    reference = (sample_dir / "transcript.txt").read_text(encoding="utf-8").strip()

    full, full_s = timed(transcribe_full_file, audio, asr)
    sliding, sliding_s = await timed_async(transcribe_current_sliding, audio, config, asr)
    utterance, utterance_s = timed(transcribe_vad_utterances, audio, config, asr)
    rolling, rolling_s = timed(transcribe_rolling_replaceable, audio, config, asr, decode_hop_ms)
    local_2, local_2_s = timed(transcribe_local_agreement, audio, config, asr, 2, decode_hop_ms)
    local_3, local_3_s = timed(transcribe_local_agreement, audio, config, asr, 3, decode_hop_ms)
    stable, stable_s = timed(transcribe_stable_prefix, audio, config, asr, decode_hop_ms)

    return [
        build_result("full_file", reference, full, full_s, audio),
        build_result("sliding_current", reference, sliding, sliding_s, audio),
        build_result("vad_utterance", reference, utterance, utterance_s, audio),
        build_result("rolling_replace", reference, rolling, rolling_s, audio),
        build_result("local_agree_2", reference, local_2, local_2_s, audio),
        build_result("local_agree_3", reference, local_3, local_3_s, audio),
        build_result("stable_prefix", reference, stable, stable_s, audio),
    ]


def print_result(sample_name: str, result: StrategyResult) -> None:
    extra = ""
    if result.utterances is not None:
        extra += f" utterances={result.utterances}"
    if result.partial_decodes is not None:
        extra += f" partial_decodes={result.partial_decodes}"
    lat = result.latency
    first = f"{lat.first_commit_ms / 1000:.2f}s" if lat.first_commit_ms is not None else "n/a"
    final = f"{lat.final_commit_ms / 1000:.2f}s" if lat.final_commit_ms is not None else "n/a"
    gap = f"{lat.mean_commit_gap_ms / 1000:.2f}s" if lat.mean_commit_gap_ms is not None else "n/a"
    print(
        f"{sample_name:<8} {result.name:<16} "
        f"WER={result.wer.wer * 100:6.1f}% "
        f"S/I/D={result.wer.substitutions}/{result.wer.insertions}/{result.wer.deletions} "
        f"commits={lat.commit_count:<2} first={first:>6} final={final:>6} "
        f"gap={gap:>6} rtf={lat.rtf:.3f}"
        f"{extra}"
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ASR chunking strategies on eval samples.")
    parser.add_argument("--data-dir", default="eval_data")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--samples", nargs="+", default=["sample1", "sample2", "sample3"])
    parser.add_argument("--decode-hop-ms", type=int, default=960)
    args = parser.parse_args()

    config = ConfigManager.load(args.config).config
    asr = SherpaOnnxAdapter(config.asr.model_dir, num_threads=config.asr.num_threads, sample_rate=SAMPLE_RATE)

    aggregate_refs: dict[str, list[str]] = {}
    aggregate_hyps: dict[str, list[str]] = {}
    aggregate_latencies: dict[str, list[LatencySummary]] = {}
    sample_dirs = [Path(args.data_dir) / sample for sample in args.samples]

    for sample_dir in sample_dirs:
        reference = (sample_dir / "transcript.txt").read_text(encoding="utf-8").strip()
        results = await evaluate_sample(sample_dir, config.vad, asr, args.decode_hop_ms)
        for result in results:
            print_result(sample_dir.name, result)
            aggregate_refs.setdefault(result.name, []).append(reference)
            aggregate_hyps.setdefault(result.name, []).append(result.hypothesis)
            aggregate_latencies.setdefault(result.name, []).append(result.latency)
        print()

    print("AGGREGATE")
    for name, refs in aggregate_refs.items():
        wer = compute_wer(refs, aggregate_hyps[name])
        latencies = aggregate_latencies[name]
        first_values = [lat.first_commit_ms for lat in latencies if lat.first_commit_ms is not None]
        final_values = [lat.final_commit_ms for lat in latencies if lat.final_commit_ms is not None]
        commit_counts = [lat.commit_count for lat in latencies]
        rtfs = [lat.rtf for lat in latencies]
        mean_first = float(np.mean(first_values)) / 1000 if first_values else None
        mean_final = float(np.mean(final_values)) / 1000 if final_values else None
        first_str = f"{mean_first:.2f}s" if mean_first is not None else "n/a"
        final_str = f"{mean_final:.2f}s" if mean_final is not None else "n/a"
        print(
            f"{name:<16} WER={wer.wer * 100:6.1f}% "
            f"S/I/D={wer.substitutions}/{wer.insertions}/{wer.deletions} "
            f"avg_commits={np.mean(commit_counts):.1f} "
            f"avg_first={first_str} avg_final={final_str} avg_rtf={np.mean(rtfs):.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
