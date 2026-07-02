"""
Metrics cho eval mode:
- WER (Word Error Rate) đánh giá ASR tiếng Việt, dùng jiwer.
- BLEU + chrF đánh giá bản dịch tiếng Anh, dùng sacrebleu.

Tất cả hàm nhận list[str] (1 phần tử = 1 mẫu) thay vì 1 string, để tính đúng
corpus-level: pool toàn bộ token/n-gram qua các mẫu trước khi chia, không phải
trung bình cộng các tỷ lệ phần trăm riêng lẻ (2 cách cho kết quả khác nhau khi
độ dài các mẫu chênh lệch).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import jiwer
import sacrebleu

# Bỏ mọi ký tự không phải chữ cái (kể cả có dấu), chữ số, hoặc khoảng trắng.
# \w trong Python re với UNICODE flag (mặc định từ Python 3) đã bao gồm chữ có
# dấu tiếng Việt, nên không cần bảng chữ cái riêng.
_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_vi_text(text: str) -> str:
    """
    Chuẩn hóa để so sánh WER công bằng: lowercase, bỏ dấu câu, gộp khoảng trắng.
    Cần thiết vì model ASR xuất toàn bộ chữ HOA không dấu câu, còn transcript
    chuẩn do người viết thường ở dạng câu bình thường có dấu câu.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _PUNCTUATION_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


@dataclass(frozen=True)
class WerResult:
    wer: float
    hits: int
    substitutions: int
    insertions: int
    deletions: int


def compute_wer(references: list[str], hypotheses: list[str]) -> WerResult:
    """references/hypotheses: 1 phần tử = 1 mẫu (transcript chuẩn vs ASR ghép lại)."""
    norm_refs = [normalize_vi_text(r) for r in references]
    norm_hyps = [normalize_vi_text(h) for h in hypotheses]

    output = jiwer.process_words(norm_refs, norm_hyps)
    return WerResult(
        wer=output.wer,
        hits=output.hits,
        substitutions=output.substitutions,
        insertions=output.insertions,
        deletions=output.deletions,
    )


@dataclass(frozen=True)
class TranslationQualityResult:
    bleu: float
    chrf: float


def compute_translation_quality(references: list[str], hypotheses: list[str]) -> TranslationQualityResult:
    """references/hypotheses: 1 phần tử = 1 mẫu (bản dịch chuẩn vs bản dịch pipeline)."""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return TranslationQualityResult(bleu=bleu.score, chrf=chrf.score)
