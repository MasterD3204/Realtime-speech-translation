"""
Translation Pipeline — spec §6

Component trung tâm quản lý toàn bộ state của segment đang được dịch. Nhận ASRResult
từ mỗi sliding window, xử lý đúng 4 bước spec §6.2 rồi gọi LLM, xử lý response theo
spec §6.3. Trả về list các event dict để tầng server gửi qua WebSocket — pipeline
không tự biết gì về WebSocket.

Lưu ý về trục thời gian: `window_start_ms` chỉ có ý nghĩa trong cùng một utterance VAD
(xem vad.py) — nó reset về 0 sau mỗi lần silence-timeout. Vì vậy state freeze cũng phải
neo theo utterance: `last_freeze_ms` là vị trí (ms) NGAY SAU window vừa freeze, và
`last_freeze_utterance_id` ghi lại utterance đó. Bước strip-overlap chỉ áp dụng khi
window mới đến từ cùng utterance với lần freeze gần nhất — khác utterance nghĩa là đã
qua một khoảng lặng, không còn gì để chồng lấn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from asr_adapter import ASRResult
from config_manager import TranslationConfig
from diff_utils import diff_new_suffix
from llm_adapter import LLMAdapter, consume_with_wait_detection
from vad import WINDOW_MS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """Bạn là module dịch hội nghị streaming VI→EN.

{history_block}Nhiệm vụ: Dịch ĐOẠN MỚI bên dưới.
Quy tắc:
- Nếu ĐOẠN MỚI chưa đủ nghĩa để dịch tự nhiên → chỉ trả về đúng 1 token: [WAIT]
- Nếu đủ nghĩa → chỉ dịch ĐOẠN MỚI, không lặp lại đoạn cũ, không giải thích.
- Đầu ra phải là một câu hoặc cụm câu tiếng Anh sạch, tự nhiên, sẵn sàng hiển thị trực tiếp.
- Không bao giờ bọc toàn bộ đầu ra trong dấu ngoặc kép như "..." hoặc '...'.
- Không thêm chú thích, không giải thích, không thêm tiền tố như Translation:, English:, Output:.
- Tự chuẩn hóa viết hoa đầu câu, đại từ "I", chữ cái đầu của tên riêng, địa danh, tổ chức.
- Chỉ trả về đúng một trong hai dạng:
  1. [WAIT]
  2. Bản dịch tiếng Anh thuần, không có ngoặc kép bao quanh."""

HISTORY_BLOCK_TEMPLATE = """Đoạn đã dịch trước (KHÔNG dịch lại, chỉ dùng làm ngữ cảnh):
{items}

"""

HISTORY_ITEM_TEMPLATE = "[{i}] {text}"


def strip_wrapping_quotes(text: str) -> str:
    pairs = [
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
    ]
    result = text.strip()

    changed = True
    while changed and len(result) >= 2:
        changed = False
        for left, right in pairs:
            if result.startswith(left) and result.endswith(right):
                result = result[1:-1].strip()
                changed = True
                break

    return result


@dataclass
class TranslationPipeline:
    llm_adapter: LLMAdapter
    config: TranslationConfig = field(default_factory=TranslationConfig)

    frozen_segments: list[str] = field(default_factory=list)
    pending_buffer: str = ""
    display_buffer: str = ""
    last_freeze_ms: int = -1
    last_freeze_utterance_id: int = -1
    last_frozen_source: str = ""

    def _build_system_prompt(self) -> str:
        recent = self.frozen_segments[-self.config.frozen_context_window:]
        if not recent:
            history_block = ""
        else:
            items = "\n".join(
                HISTORY_ITEM_TEMPLATE.format(i=i + 1, text=text)
                for i, text in enumerate(recent)
            )
            history_block = HISTORY_BLOCK_TEMPLATE.format(items=items)
        return SYSTEM_PROMPT_TEMPLATE.format(history_block=history_block)

    async def process_window(self, asr_result: ASRResult) -> list[dict]:
        events: list[dict] = []
        text = asr_result.text

        # Bước 1 — strip overlap nếu window nằm trước điểm freeze (chỉ có ý nghĩa
        # trong cùng một utterance — window_start_ms reset về 0 giữa các utterance)
        same_utterance = asr_result.utterance_id == self.last_freeze_utterance_id
        if same_utterance and asr_result.window_start_ms < self.last_freeze_ms:
            text = diff_new_suffix(self.last_frozen_source, text)

        if not text:
            return events

        # Bước 2 — update display (ASR side, màn hình trái)
        new_display = diff_new_suffix(self.display_buffer, text)
        self.display_buffer = text
        if new_display:
            events.append({"type": "asr_partial", "text": new_display})

        # Bước 3 — update pending_buffer
        self.pending_buffer = text

        # Bước 4 — gọi LLM
        try:
            system_prompt = self._build_system_prompt()
            user_message = f"ĐOẠN MỚI: {self.pending_buffer}"
            logger.info("LLM >>> %s", user_message)
            stream = self.llm_adapter.complete(system_prompt, user_message, stream=True)
            result = await consume_with_wait_detection(stream)
            logger.info("LLM <<< %s", result if result is not None else "[WAIT]")
        except Exception as exc:
            events.append({"type": "error", "code": "llm_error", "message": str(exc)})
            return events

        if result is None:
            events.append({"type": "wait", "pending": self.pending_buffer})
            return events

        result = strip_wrapping_quotes(result)
        if not result:
            events.append({"type": "wait", "pending": self.pending_buffer})
            return events

        self.frozen_segments.append(result)
        self.last_freeze_ms = asr_result.window_start_ms + WINDOW_MS
        self.last_freeze_utterance_id = asr_result.utterance_id
        self.last_frozen_source = self.pending_buffer
        self.pending_buffer = ""
        self.display_buffer = ""

        events.append({
            "type": "translation",
            "text": result,
            "segment_id": len(self.frozen_segments),
        })
        return events

    def reset(self) -> None:
        self.frozen_segments = []
        self.pending_buffer = ""
        self.display_buffer = ""
        self.last_freeze_ms = -1
        self.last_freeze_utterance_id = -1
        self.last_frozen_source = ""

    @property
    def full_en(self) -> str:
        return " ".join(self.frozen_segments)
