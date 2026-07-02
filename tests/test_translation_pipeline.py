import pytest

from asr_adapter import ASRResult
from config_manager import TranslationConfig
from translation_pipeline import TranslationPipeline, strip_wrapping_quotes
from vad import WINDOW_MS


class ScriptedLLM:
    """Fake LLMAdapter: trả 1 response cố định (đã 'stream' xong) mỗi lần complete() được gọi.

    Giữ list các response theo thứ tự gọi để mô phỏng chuỗi [WAIT] rồi bản dịch thật,
    và ghi lại toàn bộ prompt đã nhận để assert prompt build đúng (frozen_context_window).
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._i = 0
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system_prompt: str, user_message: str, stream: bool = True):
        self.calls.append((system_prompt, user_message))
        response = self._responses[self._i]
        self._i += 1
        yield response


def make_pipeline(responses: list[str], frozen_context_window=3) -> TranslationPipeline:
    llm = ScriptedLLM(responses)
    config = TranslationConfig(frozen_context_window=frozen_context_window)
    return TranslationPipeline(llm_adapter=llm, config=config)


@pytest.mark.asyncio
async def test_wait_response_emits_wait_event_and_keeps_pending():
    pipeline = make_pipeline(["[WAIT]"])
    events = await pipeline.process_window(ASRResult(text="hôm nay là", window_start_ms=0, utterance_id=0))

    assert {"type": "asr_partial", "text": "hôm nay là"} in events
    assert {"type": "wait", "pending": "hôm nay là"} in events
    assert pipeline.frozen_segments == []
    assert pipeline.pending_buffer == "hôm nay là"
    assert pipeline.display_buffer == "hôm nay là"


@pytest.mark.asyncio
async def test_translation_response_freezes_segment_and_resets_buffers():
    pipeline = make_pipeline(["Today is a beautiful day."])
    events = await pipeline.process_window(
        ASRResult(text="hôm nay là một ngày đẹp trời", window_start_ms=0, utterance_id=0)
    )

    translation_events = [e for e in events if e["type"] == "translation"]
    assert len(translation_events) == 1
    assert translation_events[0]["text"] == "Today is a beautiful day."
    assert translation_events[0]["segment_id"] == 1

    assert pipeline.frozen_segments == ["Today is a beautiful day."]
    assert pipeline.pending_buffer == ""
    assert pipeline.display_buffer == ""
    assert pipeline.last_freeze_ms == WINDOW_MS  # window_start_ms(0) + WINDOW_MS
    assert pipeline.last_freeze_utterance_id == 0
    assert pipeline.last_frozen_source == "hôm nay là một ngày đẹp trời"


@pytest.mark.asyncio
async def test_display_diff_only_sends_new_words_across_windows():
    pipeline = make_pipeline(["[WAIT]", "[WAIT]"])

    await pipeline.process_window(ASRResult(text="hôm nay là", window_start_ms=0, utterance_id=0))
    events2 = await pipeline.process_window(
        ASRResult(text="hôm nay là một ngày", window_start_ms=160, utterance_id=0)
    )

    asr_partial = next(e for e in events2 if e["type"] == "asr_partial")
    assert asr_partial["text"] == "một ngày"
    assert pipeline.display_buffer == "hôm nay là một ngày"


@pytest.mark.asyncio
async def test_overlap_strip_when_window_precedes_last_freeze_same_utterance():
    # Segment 1 dịch xong tại window_start_ms=0 (window 320ms -> last_freeze_ms=320)
    pipeline = make_pipeline(["First segment."])
    await pipeline.process_window(ASRResult(text="hôm nay là", window_start_ms=0, utterance_id=0))

    # Window kế tiếp thuộc CÙNG utterance nhưng window_start_ms=160 < last_freeze_ms=320
    # (do overlap 160ms) và ASR trả lại đầy đủ text cũ + text mới -> phải strip phần cũ.
    pipeline.llm_adapter._responses.append("[WAIT]")
    events = await pipeline.process_window(
        ASRResult(text="hôm nay là một ngày", window_start_ms=160, utterance_id=0)
    )

    wait_event = next(e for e in events if e["type"] == "wait")
    assert wait_event["pending"] == "một ngày"
    assert pipeline.pending_buffer == "một ngày"


@pytest.mark.asyncio
async def test_no_overlap_strip_across_different_utterance():
    pipeline = make_pipeline(["First segment.", "[WAIT]"])
    await pipeline.process_window(ASRResult(text="hôm nay là", window_start_ms=0, utterance_id=0))

    # utterance_id khác (=1, sau một khoảng lặng) dù window_start_ms=0 < last_freeze_ms=320
    # -> KHÔNG được strip, vì trục thời gian không liên quan giữa 2 utterance.
    events = await pipeline.process_window(ASRResult(text="chào buổi sáng", window_start_ms=0, utterance_id=1))

    wait_event = next(e for e in events if e["type"] == "wait")
    assert wait_event["pending"] == "chào buổi sáng"


@pytest.mark.asyncio
async def test_prompt_includes_only_last_frozen_context_window_segments():
    # frozen_context_window=3: cần > 3 segment ĐÃ frozen trước lúc build prompt để
    # kiểm chứng phần cũ nhất bị cắt. _build_system_prompt() chạy TRƯỚC khi response
    # của chính lần gọi đó được append, nên request thứ N chỉ thấy (N-1) segment đã có.
    # Dùng 5 lần gọi: request thứ 5 nhìn thấy 4 segment đã frozen, phải cắt còn 3.
    responses = ["Seg one.", "Seg two.", "Seg three.", "Seg four.", "Seg five."]
    pipeline = make_pipeline(responses, frozen_context_window=3)

    for i, text in enumerate(["một", "hai", "ba", "bốn", "năm"]):
        await pipeline.process_window(ASRResult(text=text, window_start_ms=i * 1000, utterance_id=i))

    last_system_prompt = pipeline.llm_adapter.calls[-1][0]
    assert "Seg one." not in last_system_prompt  # segment cũ nhất bị cắt khỏi context
    assert "Seg two." in last_system_prompt
    assert "Seg three." in last_system_prompt
    assert "Seg four." in last_system_prompt
    # "Seg five." là kết quả của chính request thứ 5, chưa tồn tại lúc build prompt cho nó
    assert "Seg five." not in last_system_prompt


@pytest.mark.asyncio
async def test_llm_error_emits_error_event_and_preserves_pending():
    class RaisingLLM:
        async def complete(self, system_prompt, user_message, stream=True):
            raise RuntimeError("network down")
            yield  # pragma: no cover - unreachable, makes this an async generator

    pipeline = TranslationPipeline(llm_adapter=RaisingLLM(), config=TranslationConfig())
    events = await pipeline.process_window(ASRResult(text="hôm nay", window_start_ms=0, utterance_id=0))

    # Bước 2 (display diff) chạy trước bước 4 (gọi LLM) theo spec §6.2 — asr_partial
    # luôn được emit trước khi LLM có cơ hội lỗi.
    assert events == [
        {"type": "asr_partial", "text": "hôm nay"},
        {"type": "error", "code": "llm_error", "message": "network down"},
    ]
    assert pipeline.pending_buffer == "hôm nay"
    assert pipeline.frozen_segments == []


@pytest.mark.asyncio
async def test_empty_asr_text_after_overlap_strip_produces_no_events():
    pipeline = make_pipeline(["Seg."])
    await pipeline.process_window(ASRResult(text="hôm nay là", window_start_ms=0, utterance_id=0))

    # ASR trả đúng y hệt phần đã freeze, không có gì mới -> sau strip text rỗng
    events = await pipeline.process_window(ASRResult(text="hôm nay là", window_start_ms=100, utterance_id=0))
    assert events == []


@pytest.mark.asyncio
async def test_reset_clears_all_state():
    pipeline = make_pipeline(["Seg."])
    await pipeline.process_window(ASRResult(text="hôm nay là", window_start_ms=0, utterance_id=0))
    assert pipeline.frozen_segments

    pipeline.reset()
    assert pipeline.frozen_segments == []
    assert pipeline.pending_buffer == ""
    assert pipeline.display_buffer == ""
    assert pipeline.last_freeze_ms == -1
    assert pipeline.last_freeze_utterance_id == -1
    assert pipeline.last_frozen_source == ""


def test_strip_wrapping_quotes_removes_matching_pairs():
    assert strip_wrapping_quotes('"Today is nice."') == "Today is nice."
    assert strip_wrapping_quotes("'Today is nice.'") == "Today is nice."
    assert strip_wrapping_quotes("Today is nice.") == "Today is nice."
    assert strip_wrapping_quotes("“Today is nice.”") == "Today is nice."


@pytest.mark.asyncio
async def test_result_that_is_only_quotes_emits_wait_not_empty_translation():
    # Sau khi strip_wrapping_quotes, nếu LLM chỉ trả về đúng cặp ngoặc kép rỗng
    # ('""') thì kết quả rỗng — phải coi như WAIT, không được freeze segment rỗng.
    pipeline = make_pipeline(['""'])
    events = await pipeline.process_window(ASRResult(text="hôm nay", window_start_ms=0, utterance_id=0))

    assert {"type": "wait", "pending": "hôm nay"} in events
    assert pipeline.frozen_segments == []


@pytest.mark.asyncio
async def test_full_en_joins_all_frozen_segments():
    pipeline = make_pipeline(["First.", "Second."])
    await pipeline.process_window(ASRResult(text="một", window_start_ms=0, utterance_id=0))
    await pipeline.process_window(ASRResult(text="một hai", window_start_ms=1000, utterance_id=1))

    assert pipeline.full_en == "First. Second."
