from diff_utils import diff_new_suffix, word_common_prefix_len


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


def test_diff_new_suffix_returns_only_new_words():
    assert diff_new_suffix("hôm nay là", "hôm nay là một ngày") == "một ngày"


def test_diff_new_suffix_no_new_words():
    assert diff_new_suffix("hôm nay là", "hôm nay là") == ""


def test_diff_new_suffix_reference_empty_returns_full_text():
    assert diff_new_suffix("", "hôm nay là") == "hôm nay là"


def test_diff_new_suffix_completely_diverged_returns_full_text():
    # Không có tiền tố chung -> toàn bộ text mới được coi là "mới"
    assert diff_new_suffix("xin chào", "hôm nay là") == "hôm nay là"
