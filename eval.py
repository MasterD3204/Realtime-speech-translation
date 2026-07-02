"""
CLI đánh giá chất lượng pipeline trên bộ mẫu (audio, transcript VI chuẩn,
bản dịch EN chuẩn) — chạy qua đúng VAD/ASR/Translation pipeline thật với
config từ config.yaml, không phải bản mô phỏng riêng.

Format 1 mẫu — thư mục con của --data-dir:
    sample_001/
        audio.wav (hoặc .flac/.mp3/.ogg)
        transcript.txt   (VI ground truth)
        translation.txt  (EN ground truth)

Dùng:
    python eval.py ./eval_data
    python eval.py ./eval_data --config config.yaml --output report.json
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys

from asr_adapter import SherpaOnnxAdapter
from config_manager import ConfigManager
from eval_runner import DatasetReport, SampleResult, evaluate_dataset
from llm_adapter import build_llm_adapter
from translation_pipeline import TranslationPipeline
from vad import ChunkedSlidingVAD, SileroSpeechProbabilityModel

SAMPLE_RATE = 16000

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("eval")


def format_sample_row(r: SampleResult) -> str:
    lat = r.latency_seconds
    lat_str = f"{lat['mean']:.2f}s" if lat["count"] else "n/a"
    return (
        f"{r.name:<24} WER={r.wer.wer * 100:5.1f}%  BLEU={r.quality.bleu:5.1f}  "
        f"chrF={r.quality.chrf:5.1f}  lat(mean)={lat_str:>8}  "
        f"pending_leftover={r.pending_leftover_words}w  dropped_tail={r.dropped_tail_ms:.0f}ms"
    )


def print_report(report: DatasetReport) -> None:
    print()
    print("=" * 100)
    print(f"{'Sample':<24} {'WER':<10} {'BLEU':<11} {'chrF':<10} {'Latency':<14} Diagnostics")
    print("-" * 100)
    for r in report.samples:
        print(format_sample_row(r))
    print("-" * 100)

    agg_lat = report.aggregate_latency_seconds
    print(f"{'AGGREGATE':<24} WER={report.aggregate_wer.wer * 100:5.1f}%  "
          f"BLEU={report.aggregate_quality.bleu:5.1f}  chrF={report.aggregate_quality.chrf:5.1f}")
    if agg_lat["count"]:
        print(
            f"Latency (n={agg_lat['count']}): mean={agg_lat['mean']:.2f}s  "
            f"median={agg_lat['median']:.2f}s  p95={agg_lat['p95']:.2f}s  max={agg_lat['max']:.2f}s"
        )
    else:
        print("Latency: không có segment nào được dịch xong (mọi mẫu đều [WAIT] hoặc lỗi)")
    print("=" * 100)


def report_to_dict(report: DatasetReport) -> dict:
    return {
        "samples": [dataclasses.asdict(s) for s in report.samples],
        "aggregate_wer": dataclasses.asdict(report.aggregate_wer),
        "aggregate_quality": dataclasses.asdict(report.aggregate_quality),
        "aggregate_latency_seconds": report.aggregate_latency_seconds,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Đánh giá chất lượng ASR + Translation pipeline trên bộ mẫu")
    parser.add_argument("data_dir", help="Thư mục chứa các thư mục mẫu (mỗi mẫu: audio.*, transcript.txt, translation.txt)")
    parser.add_argument("--config", default="config.yaml", help="Đường dẫn config.yaml (mặc định: config.yaml)")
    parser.add_argument("--output", default=None, help="Ghi report chi tiết dạng JSON ra file này (tùy chọn)")
    args = parser.parse_args()

    config_manager = ConfigManager.load(args.config)
    config = config_manager.config

    print(f"[Eval] Loading ASR model from {config.asr.model_dir} ...")
    asr_adapter = SherpaOnnxAdapter(config.asr.model_dir, num_threads=config.asr.num_threads, sample_rate=SAMPLE_RATE)
    print(f"[Eval] LLM provider={config.llm.provider} model={config.llm.model}")

    def make_vad() -> ChunkedSlidingVAD:
        speech_model = SileroSpeechProbabilityModel(config.vad.model_path, sample_rate=SAMPLE_RATE)
        return ChunkedSlidingVAD(speech_model, config.vad, sample_rate=SAMPLE_RATE)

    def make_pipeline() -> TranslationPipeline:
        llm_adapter = build_llm_adapter(config.llm)
        return TranslationPipeline(llm_adapter, config.translation)

    def on_sample_done(result: SampleResult) -> None:
        print(f"[Eval] Done: {format_sample_row(result)}")

    try:
        report = await evaluate_dataset(
            args.data_dir,
            asr_adapter,
            make_vad,
            make_pipeline,
            sample_rate=SAMPLE_RATE,
            on_sample_done=on_sample_done,
        )
    except FileNotFoundError as exc:
        print(f"[Eval] Lỗi: {exc}", file=sys.stderr)
        return 1

    print_report(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report_to_dict(report), f, ensure_ascii=False, indent=2)
        print(f"[Eval] Report chi tiết đã lưu tại {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
