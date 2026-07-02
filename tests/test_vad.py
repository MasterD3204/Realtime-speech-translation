import numpy as np
import pytest

from config_manager import VadConfig
from vad import CHUNK_MS, HOP_CHUNKS, WINDOW_CHUNKS, ChunkedSlidingVAD

SAMPLE_RATE = 16000
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)


class ScriptedModel:
    """Fake speech-probability model: trả prob theo kịch bản list[bool] is_speech cho sẵn.

    Toàn bộ script cho một test được truyền 1 lần lúc khởi tạo — model giữ con trỏ nội
    bộ xuyên suốt mọi lần gọi `feed()`, giống hệt cách VAD thật gọi predict() tuần tự.
    """

    def __init__(self, script: list[bool]):
        self._script = list(script)
        self._i = 0
        self.reset_count = 0

    def predict(self, chunk: np.ndarray) -> float:
        val = self._script[self._i]
        self._i += 1
        return 1.0 if val else 0.0

    def reset(self) -> None:
        self.reset_count += 1


def make_chunk() -> np.ndarray:
    return np.zeros(CHUNK_SAMPLES, dtype=np.float32)


def make_vad(script: list[bool], min_silence_ms=32, min_speech_ms=32) -> ChunkedSlidingVAD:
    config = VadConfig(model_path="unused", threshold=0.5, min_silence_ms=min_silence_ms, min_speech_ms=min_speech_ms)
    model = ScriptedModel(script)
    return ChunkedSlidingVAD(model, config, sample_rate=SAMPLE_RATE)


def feed(vad: ChunkedSlidingVAD, n: int):
    """Push n chunk, đọc giá trị tiếp theo từ script đã gán cho model của vad."""
    return [vad.push_chunk(make_chunk()) for _ in range(n)]


def test_no_window_before_10_speech_chunks():
    vad = make_vad([True] * 9)
    windows = feed(vad, 9)
    assert all(w is None for w in windows)


def test_first_window_forwarded_at_10th_speech_chunk():
    vad = make_vad([True] * 10)
    windows = feed(vad, 10)
    assert windows[:9] == [None] * 9
    assert windows[9] is not None
    assert windows[9].window_start_ms == 0
    assert len(windows[9].samples) == WINDOW_CHUNKS * CHUNK_SAMPLES


def test_second_window_forwarded_after_hop_of_5_more_chunks():
    vad = make_vad([True] * 15)
    windows = feed(vad, 15)
    forwarded = [(i, w) for i, w in enumerate(windows) if w is not None]
    assert len(forwarded) == 2
    assert forwarded[0][0] == 9
    assert forwarded[1][0] == 14
    # spec example: Window1 0-320ms, Window2 160-480ms
    assert forwarded[0][1].window_start_ms == 0
    assert forwarded[1][1].window_start_ms == HOP_CHUNKS * CHUNK_MS


def test_window_advances_every_hop_not_every_chunk():
    vad = make_vad([True] * 20)
    windows = feed(vad, 20)
    forwarded_indices = [i for i, w in enumerate(windows) if w is not None]
    # 10th, 15th, 20th chunk (0-indexed: 9, 14, 19)
    assert forwarded_indices == [9, 14, 19]


def test_speech_run_reset_by_silence_before_confirm_never_forwards():
    # min_speech_ms=96 -> cần 3 chunk speech liên tiếp mới confirm. Pattern
    # True,True,False lặp lại không bao giờ đạt 3 liên tiếp -> không bao giờ confirm.
    script = [True, True, False] * 10
    vad = make_vad(script, min_speech_ms=96)
    windows = feed(vad, len(script))
    assert all(w is None for w in windows)


def test_silence_shorter_than_hangover_does_not_reset_buffer():
    # 10 speech (window forwarded) + 1 silence (chưa đủ hangover 64ms=2 chunk) + 4 speech
    script = [True] * 10 + [False] + [True] * 4
    vad = make_vad(script, min_silence_ms=64, min_speech_ms=32)
    windows = feed(vad, len(script))

    assert windows[9] is not None  # window đầu tiên tại chunk speech thứ 10
    assert windows[10] is None  # 1 chunk silence — chưa đủ hangover để reset

    # 4 speech tiếp theo cộng dồn vào buffer cũ (không bị reset): tổng speech = 14,
    # (14-10) % 5 = 4 != 0 -> chưa đến hop tiếp theo, đúng như kỳ vọng không forward.
    assert all(w is None for w in windows[11:])


def test_silence_run_reaching_hangover_resets_buffer_and_starts_new_utterance():
    # 10 speech (window #1) + 2 silence (đủ hangover 64ms=2 chunk -> reset) + 10 speech mới
    script = [True] * 10 + [False, False] + [True] * 10
    vad = make_vad(script, min_silence_ms=64, min_speech_ms=32)
    windows = feed(vad, len(script))

    first_window = windows[9]
    assert first_window is not None
    assert first_window.window_start_ms == 0

    # Sau reset, utterance mới: window_start_ms lại bắt đầu từ 0, utterance_id tăng thêm 1
    second_window = windows[21]  # chunk cuối cùng (index 21 = 10+2+10-1)
    assert second_window is not None
    assert second_window.window_start_ms == 0
    assert second_window.utterance_id == first_window.utterance_id + 1


def test_update_config_changes_thresholds_at_runtime():
    vad = make_vad([], min_silence_ms=400, min_speech_ms=200)
    assert vad.min_silence_chunks == round(400 / CHUNK_MS)
    assert vad.min_speech_chunks == round(200 / CHUNK_MS)

    vad.update_config(threshold=0.8, min_silence_ms=64, min_speech_ms=32)
    assert vad.threshold == 0.8
    assert vad.min_silence_chunks == 2
    assert vad.min_speech_chunks == 1


def test_reset_clears_state_and_resets_model():
    vad = make_vad([True] * 20)
    windows = feed(vad, 10)
    assert windows[9] is not None

    vad.reset()
    assert vad._model.reset_count == 1

    # Sau reset thủ công, cần đủ 10 chunk speech mới lại để forward window mới,
    # utterance/window_start_ms bắt đầu lại từ 0.
    more_windows = feed(vad, 10)
    assert more_windows[9] is not None
    assert more_windows[9].window_start_ms == 0
