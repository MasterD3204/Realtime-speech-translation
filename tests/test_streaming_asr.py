from streaming_asr import LocalAgreementBuffer


def test_local_agreement_waits_for_n_hypotheses():
    buffer = LocalAgreementBuffer(agreement_n=3)

    committed, partial = buffer.push("xin chào các")
    assert committed == ""
    assert partial == "xin chào các"

    committed, partial = buffer.push("xin chào các bạn")
    assert committed == ""
    assert partial == "xin chào các bạn"

    committed, partial = buffer.push("xin chào các bạn hôm nay")
    assert committed == "xin chào các"
    assert partial == "bạn hôm nay"


def test_local_agreement_commits_only_new_suffix():
    buffer = LocalAgreementBuffer(agreement_n=2)

    assert buffer.push("hà nội là")[0] == ""
    committed, partial = buffer.push("hà nội là thủ đô")
    assert committed == "hà nội là"
    assert partial == "thủ đô"

    committed, partial = buffer.push("hà nội là thủ đô của việt nam")
    assert committed == "thủ đô"
    assert partial == "của việt nam"


def test_local_agreement_flush_commits_remaining_text():
    buffer = LocalAgreementBuffer(agreement_n=3)
    buffer.push("hôm nay tôi")
    buffer.push("hôm nay tôi rất")
    buffer.push("hôm nay tôi rất vui")

    committed, partial = buffer.flush("hôm nay tôi rất vui")

    assert committed == "rất vui"
    assert partial == ""
