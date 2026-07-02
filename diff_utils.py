"""
Word-level diff helpers dùng chung cho:
- Bước 1 (strip overlap): so `last_frozen_source` với asr_output window mới
- Bước 2 (display diff): so `display_buffer` với asr_output để tìm phần text mới

Cả hai đều là bài toán ghép chuỗi chồng lấn (giống stitch 2 đoạn audio/text overlap):
tìm đoạn hậu tố của `reference` khớp với đoạn tiền tố của `text`, rồi trả về phần
`text` nằm SAU điểm khớp đó. Quan trọng: KHÔNG được giả định `reference` luôn vừa
trọn làm tiền tố của `text` (tức "window mới luôn chứa y nguyên toàn bộ reference
rồi thêm chữ") — với sliding window, window mới có thể đã trượt qua và bỏ mất phần
đầu của `reference` ra khỏi audio, nên điểm chồng lấn thực tế nằm ở giữa/cuối
`reference` chứ không phải từ đầu. Giả định sai này gây lặp từ hàng loạt khi window
đủ lớn để `reference` (pending/display buffer tích lũy) dài hơn nội dung 1 window.
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


def find_word_overlap(reference: str, text: str) -> int:
    """
    Tìm K lớn nhất (0 <= K <= min số từ của 2 chuỗi) sao cho K từ CUỐI của
    `reference` khớp chính xác với K từ ĐẦU của `text`. Quét từ K lớn nhất giảm
    dần để ưu tiên đoạn khớp dài nhất (giảm nguy cơ khớp trùng ngẫu nhiên 1 từ
    hư từ phổ biến như "là", "và").
    """
    ref_words = tokenize_words(reference)
    text_words = tokenize_words(text)
    max_k = min(len(ref_words), len(text_words))
    for k in range(max_k, 0, -1):
        if ref_words[-k:] == text_words[:k]:
            return k
    return 0


def diff_new_suffix(reference: str, text: str) -> str:
    """
    Trả về phần từ trong `text` xuất hiện SAU đoạn chồng lấn với `reference`.
    Dùng cho cả overlap-strip (reference=last_frozen_source) và display-diff
    (reference=display_buffer). Nếu không tìm thấy điểm chồng lấn nào (K=0) —
    do ASR bất đồng ngay tại từ cuối cùng của đoạn overlap, hoặc do đã sang
    utterance khác — trả về toàn bộ `text` (không có gì để strip).
    """
    words_text = tokenize_words(text)
    k = find_word_overlap(reference, text)
    return " ".join(words_text[k:])
