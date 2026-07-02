"""
Word-level diff helpers dùng chung cho:
- Bước 1 (strip overlap): so `last_frozen_source` với asr_output window mới
- Bước 2 (display diff): so `display_buffer` với asr_output để tìm phần text mới

Cả hai là cùng một phép toán: cho "reference" (đoạn cũ) và "text" (đoạn mới, dài hơn
hoặc bằng, chứa reference là tiền tố theo từ) → trả về phần từ mới xuất hiện sau đó.
"""

from __future__ import annotations


def tokenize_words(text: str) -> list[str]:
    return text.split()


def word_common_prefix_len(a: str, b: str) -> int:
    """Số từ liên tiếp giống hệt nhau tính từ đầu giữa a và b."""
    words_a = tokenize_words(a)
    words_b = tokenize_words(b)
    n = 0
    for word_a, word_b in zip(words_a, words_b):
        if word_a != word_b:
            break
        n += 1
    return n


def diff_new_suffix(reference: str, text: str) -> str:
    """
    Trả về phần từ trong `text` xuất hiện sau đoạn tiền tố chung với `reference`.
    Dùng cho cả overlap-strip (reference=last_frozen_source) và display-diff
    (reference=display_buffer).
    """
    words_text = tokenize_words(text)
    prefix_len = word_common_prefix_len(reference, text)
    return " ".join(words_text[prefix_len:])
