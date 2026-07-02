from eval_metrics import compute_translation_quality, compute_wer, normalize_vi_text


def test_normalize_vi_text_lowercases_and_strips_punctuation():
    assert normalize_vi_text("Hôm nay, trời đẹp!") == "hôm nay trời đẹp"


def test_normalize_vi_text_collapses_whitespace():
    assert normalize_vi_text("hôm   nay\ttrời") == "hôm nay trời"


def test_normalize_vi_text_preserves_vietnamese_diacritics():
    # Đảm bảo không bị strip nhầm dấu thanh/dấu mũ thành ASCII gần giống
    assert normalize_vi_text("MƯỜI MỘT THÁNG MƯỜI BA") == "mười một tháng mười ba"


def test_compute_wer_identical_transcript_is_zero():
    result = compute_wer(["hôm nay trời đẹp"], ["hôm nay trời đẹp"])
    assert result.wer == 0.0
    assert result.substitutions == 0
    assert result.insertions == 0
    assert result.deletions == 0


def test_compute_wer_counts_insertion():
    # hyp có thêm 1 từ thừa so với ref (4 từ) -> 1 insertion / 4 ref words
    result = compute_wer(["hôm nay trời đẹp"], ["hôm nay trời rất đẹp"])
    assert result.insertions == 1
    assert result.wer == 1 / 4


def test_compute_wer_ignores_case_and_punctuation_differences():
    # ASR xuất hoa không dấu câu, transcript chuẩn viết thường có dấu câu -> vẫn WER=0
    result = compute_wer(["Hôm nay, trời đẹp."], ["HÔM NAY TRỜI ĐẸP"])
    assert result.wer == 0.0


def test_compute_wer_pools_across_multiple_samples_not_averaged():
    # Mẫu 1: khớp hoàn toàn (0 lỗi / 2 từ). Mẫu 2: 1 substitution / 2 từ.
    # Pooled WER = 1 lỗi / 4 tổng từ ref = 0.25, KHÁC với trung bình cộng (0 + 0.5)/2 = 0.25
    # -> chọn case này để tổng bằng nhau là không đủ phân biệt, kiểm tra bằng case lệch số từ:
    result = compute_wer(
        ["một hai ba bốn năm sáu bảy tám"],  # 8 từ, khớp hết
        ["một hai ba bốn năm sáu bảy tám"],
    )
    assert result.wer == 0.0

    result2 = compute_wer(
        ["một hai ba bốn năm sáu bảy tám", "chín mười"],  # tổng 10 từ ref
        ["một hai ba bốn năm sáu bảy tám", "chín mười một"],  # mẫu 2 có 1 insertion
    )
    # Pooled: 1 lỗi / 10 tổng từ = 0.1. Nếu tính trung bình cộng WER từng mẫu riêng
    # (0 + 0.5)/2 = 0.25 thì sẽ SAI — assert đúng giá trị pooled để chốt hành vi.
    assert result2.wer == 1 / 10


def test_compute_translation_quality_identical_translation_scores_high():
    result = compute_translation_quality(
        ["Today is a beautiful day."],
        ["Today is a beautiful day."],
    )
    assert result.bleu > 90
    assert result.chrf > 90


def test_compute_translation_quality_unrelated_translation_scores_low():
    result = compute_translation_quality(
        ["Today is a beautiful day."],
        ["The stock market crashed yesterday."],
    )
    assert result.bleu < 20


def test_compute_translation_quality_partial_match_scores_between():
    identical = compute_translation_quality(
        ["Today is a beautiful day."], ["Today is a beautiful day."]
    )
    partial = compute_translation_quality(
        ["Today is a beautiful day."], ["Today is a nice day."]
    )
    unrelated = compute_translation_quality(
        ["Today is a beautiful day."], ["The stock market crashed yesterday."]
    )
    assert unrelated.bleu < partial.bleu < identical.bleu
