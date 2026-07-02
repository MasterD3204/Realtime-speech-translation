"""
Segment-based Realtime Translation Pipeline
- Nhận ASR chunks liên tục
- Gửi vào Gemini với frozen history context
- [WAIT] → accumulate thêm, không hiển thị
- Đủ nghĩa → append segment mới vào frozen history
"""

import asyncio
from google import genai
from google.genai import types

client = genai.Client()
MODEL = "models/gemini-3.1-flash-lite"  # hoặc gemini-2.5-flash

SYSTEM_PROMPT_TEMPLATE = """Bạn là module dịch hội nghị VI→EN streaming.

{history_block}
Nhiệm vụ: Dịch ĐOẠN MỚI bên dưới.
Quy tắc:
- Nếu đoạn mới chưa đủ nghĩa để dịch tự nhiên → chỉ trả về: [WAIT]
- Nếu đủ nghĩa → trả về đúng bản dịch tiếng Anh của đoạn mới, KHÔNG lặp lại đoạn cũ.
- Đầu ra phải là một câu hoặc cụm câu tiếng Anh sạch, tự nhiên, sẵn sàng hiển thị trực tiếp.
- Không bao giờ bọc toàn bộ đầu ra trong dấu ngoặc kép như "..." hoặc '...'.
- Không thêm chú thích, không giải thích, không thêm tiền tố như Translation:, English:, Output:.
- Tự chuẩn hóa viết hoa đầu câu, đại từ "I", chữ cái đầu của tên riêng, địa danh, tổ chức, và các thực thể cần viết hoa trong tiếng Anh.
- Với tên người Việt hoặc tên riêng tiếng Việt, hãy suy luận ranh giới tên và viết hoa theo dạng tên riêng tự nhiên, ví dụ: "nguyen phu trong" → "Nguyen Phu Trong".
- Được phép chỉnh nhẹ hoa/thường và dấu câu để câu dịch tự nhiên, nhưng không được thêm ý ngoài nội dung gốc.
- Chỉ trả về đúng một trong hai dạng:
  1. [WAIT]
  2. Bản dịch tiếng Anh thuần, không có ngoặc kép bao quanh."""

HISTORY_BLOCK_TEMPLATE = """Các đoạn đã dịch trước đó (KHÔNG được thay đổi, chỉ dùng làm ngữ cảnh):
{items}

"""

HISTORY_ITEM_TEMPLATE = "[{i}] \"{text}\""


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


class TranslationPipeline:
    def __init__(self, on_translation=None, on_wait=None):
        """
        on_translation(vi_segment: str, en_segment: str) → callback khi có bản dịch mới
        on_wait(pending: str) → callback khi LLM trả [WAIT] (optional, để debug)
        """
        self.translated_segments: list[tuple[str, str]] = []  # [(vi, en), ...]
        self.pending_vi = ""       # ASR chunks đang accumulate
        self.on_translation = on_translation
        self.on_wait = on_wait
        self._lock = asyncio.Lock()

    def _build_system_prompt(self) -> str:
        if not self.translated_segments:
            history_block = ""
        else:
            items = "\n".join(
                HISTORY_ITEM_TEMPLATE.format(i=i + 1, text=en)
                for i, (_, en) in enumerate(self.translated_segments)
            )
            history_block = HISTORY_BLOCK_TEMPLATE.format(items=items)
        return SYSTEM_PROMPT_TEMPLATE.format(history_block=history_block)

    async def update_text(self, asr_text: str, finalize: bool = False) -> str | None:
        """
        Cập nhật toàn bộ câu hiện tại từ ASR.
        - finalize=False: dùng cho partial ASR, chỉ trả bản dịch tạm thời nếu đủ nghĩa
        - finalize=True:  dùng cho VAD-final ASR, freeze câu dịch vào history
        Returns: bản dịch nếu LLM confirm đủ nghĩa, None nếu [WAIT].
        """
        async with self._lock:
            self.pending_vi = asr_text.strip()

            if not self.pending_vi:
                return None

            system_prompt = self._build_system_prompt()
            completion_state = "ĐÃ CHỐT bởi VAD" if finalize else "CHƯA CHỐT, đang là bản nháp realtime"
            user_message = (
                f'TRẠNG THÁI ĐOẠN: {completion_state}\n'
                f'ĐOẠN MỚI: "{self.pending_vi}"'
            )

            # Gọi Gemini (non-streaming để lấy full response trước khi quyết định)
            # Dùng to_thread vì generate_content là sync — tránh block asyncio event loop
            try:
                def _call_gemini():
                    return client.models.generate_content(
                        model=MODEL,
                        contents=user_message,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.1,
                            max_output_tokens=256,
                        )
                    )
                response = await asyncio.to_thread(_call_gemini)
                result = response.text.strip() if response.text else "[WAIT]"
            except Exception as e:
                print(f"[LLM Error] {e}")
                return None

            if result == "[WAIT]":
                if self.on_wait:
                    self.on_wait(self.pending_vi)
                return None

            result = strip_wrapping_quotes(result)

            if not finalize:
                return result

            # LLM đã dịch → freeze segment này
            vi_segment = self.pending_vi
            en_segment = result
            self.translated_segments.append((vi_segment, en_segment))
            self.pending_vi = ""  # reset buffer

            if self.on_translation:
                self.on_translation(vi_segment, en_segment)

            return en_segment

    async def push_chunk(self, asr_chunk: str) -> str | None:
        """
        Backward-compatible alias: coi input là đoạn đã chốt.
        """
        return await self.update_text(asr_chunk, finalize=True)

    def reset(self):
        """Reset toàn bộ session (vd: bắt đầu cuộc họp mới)."""
        self.translated_segments.clear()
        self.pending_vi = ""

    @property
    def full_vi(self) -> str:
        """Toàn bộ transcript tiếng Việt đã frozen + pending."""
        parts = [vi for vi, _ in self.translated_segments]
        if self.pending_vi:
            parts.append(f"[{self.pending_vi}]")  # pending được đánh dấu bằng []
        return " ".join(parts)

    @property
    def full_en(self) -> str:
        """Toàn bộ bản dịch tiếng Anh đã frozen."""
        return " ".join(en for _, en in self.translated_segments)


# ── Standalone test ──────────────────────────────────────────────────────────
async def demo():
    pipeline = TranslationPipeline(
        on_translation=lambda vi, en: print(f"\n✅ DỊCH: [{vi}] → [{en}]"),
        on_wait=lambda pending: print(f"   ⏳ WAIT: '{pending}'"),
    )

    asr_chunks = [
        "hôm nay",
        "là một ngày",
        "đẹp trời",
        "tôi đi làm",
        "và gặp một người bạn cũ",
        "chúng tôi đã nói chuyện",
        "rất lâu",
    ]

    print("=== Demo Translation Pipeline ===\n")
    for chunk in asr_chunks:
        print(f"📝 ASR chunk: '{chunk}'")
        result = await pipeline.push_chunk(chunk)
        if result:
            print(f"   → Bản dịch mới: '{result}'")
        await asyncio.sleep(0.3)  # simulate ASR delay

    print("\n=== Kết quả cuối ===")
    print(f"🇻🇳 VI: {pipeline.full_vi}")
    print(f"🇬🇧 EN: {pipeline.full_en}")


if __name__ == "__main__":
    asyncio.run(demo())
