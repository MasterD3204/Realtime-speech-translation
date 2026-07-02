from diff_utils import diff_new_suffix, find_word_overlap, word_common_prefix_len


def test_word_common_prefix_len_full_overlap():
    assert word_common_prefix_len("hôm nay là", "hôm nay là") == 3


def test_word_common_prefix_len_partial_overlap():
    assert word_common_prefix_len("hôm nay là", "hôm nay là một ngày") == 3


def test_word_common_prefix_len_no_overlap():
    assert word_common_prefix_len("xin chào", "hôm nay") == 0


def test_word_common_prefix_len_empty_reference():
    assert word_common_prefix_len("", "hôm nay") == 0


def test_word_common_prefix_len_diverges_midway():
    assert word_common_prefix_len("hôm nay là thứ hai", "hôm nay là thứ ba") == 4


def test_find_word_overlap_reference_is_full_prefix_of_text():
    # Case đơn giản: reference vừa trọn làm tiền tố (window nhỏ, chưa trượt qua)
    assert find_word_overlap("hôm nay là", "hôm nay là một ngày") == 3


def test_find_word_overlap_when_window_has_slid_past_start_of_reference():
    # Case thật gây bug: window đã trượt qua, "hôm nay" (đầu reference) không còn
    # trong text mới nữa — chỉ phần ĐUÔI reference ("là một ngày") còn xuất hiện,
    # và nó nằm ở ĐẦU text mới, không phải cả reference làm tiền tố.
    reference = "hôm nay là một ngày"
    text = "là một ngày đẹp trời hôm nay"  # đuôi reference trùng đầu text
    assert find_word_overlap(reference, text) == 3  # "là một ngày"


def test_find_word_overlap_no_overlap_at_all():
    assert find_word_overlap("xin chào buổi sáng", "hôm nay là") == 0


def test_find_word_overlap_prefers_longest_match_not_shortest():
    # "là" một mình khớp (từ phổ biến) nhưng "một ngày là" khớp dài hơn -> phải chọn dài nhất
    reference = "hôm nay một ngày là"
    text = "một ngày là đẹp trời"
    assert find_word_overlap(reference, text) == 3  # "một ngày là", không phải 1 ("là")


def test_find_word_overlap_empty_reference_or_text():
    assert find_word_overlap("", "hôm nay") == 0
    assert find_word_overlap("hôm nay", "") == 0


def test_diff_new_suffix_returns_only_new_words():
    assert diff_new_suffix("hôm nay là", "hôm nay là một ngày") == "một ngày"


def test_diff_new_suffix_no_new_words():
    assert diff_new_suffix("hôm nay là", "hôm nay là") == ""


def test_diff_new_suffix_reference_empty_returns_full_text():
    assert diff_new_suffix("", "hôm nay là") == "hôm nay là"


def test_diff_new_suffix_completely_diverged_returns_full_text():
    # Không có overlap nào -> toàn bộ text mới được coi là "mới"
    assert diff_new_suffix("xin chào", "hôm nay là") == "hôm nay là"


def test_diff_new_suffix_handles_window_slid_past_reference_start():
    # Bug tái hiện: window lớn (WINDOW_CHUNKS=40) khiến display_buffer/pending_buffer
    # tích lũy dài hơn nội dung 1 window audio mới -> window mới không còn chứa
    # nguyên vẹn buffer cũ làm tiền tố, chỉ phần đuôi buffer cũ overlap với đầu text mới.
    # Thuật toán tiền tố-từ-đầu-vào-đầu cũ sẽ trả về gần như FULL text mới -> lặp từ
    # hàng loạt (đúng triệu chứng log thực tế: "BAN CHẤP HÀNH BAN CHẤP HÀNH...").
    reference = "hội nghị ban chấp hành trung ương lần thứ mười một"
    text = "trung ương lần thứ mười một tháng mười ba bế mạc"
    assert diff_new_suffix(reference, text) == "tháng mười ba bế mạc"
