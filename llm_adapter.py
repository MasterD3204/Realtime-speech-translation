"""
LLM Layer — spec §7

LLMAdapter interface (thin adapter) cho phép swap provider (Gemini/OpenAI) mà không
ảnh hưởng Translation Pipeline. `complete()` luôn là async generator token — kể cả khi
provider không stream thật, để pipeline dùng chung 1 luồng xử lý.

`consume_with_wait_detection` implement spec §7.3: phát hiện "[WAIT]" ngay khi token đã
nhận khớp chính xác chuỗi đó → cancel stream (aclose) ngay, không đợi hết generation.
"""

from __future__ import annotations

import asyncio
import threading
from typing import AsyncGenerator, Protocol

from config_manager import LLMConfig

WAIT_TOKEN = "[WAIT]"


class LLMAdapter(Protocol):
    def complete(self, system_prompt: str, user_message: str, stream: bool = True) -> AsyncGenerator[str, None]: ...


async def consume_with_wait_detection(stream: AsyncGenerator[str, None]) -> str | None:
    """
    Đọc từng token từ stream. Nếu phần đã nhận (sau strip) khớp chính xác "[WAIT]"
    → cancel stream ngay (aclose), trả None. Nếu stream kết thúc tự nhiên với nội dung
    khác → trả full text đã tích lũy (bản dịch thật).
    """
    accumulated = ""
    async for delta in stream:
        accumulated += delta
        if accumulated.strip() == WAIT_TOKEN:
            await stream.aclose()
            return None
    return accumulated.strip()


class GeminiAdapter:
    """
    google-genai client là sync. Để không block asyncio event loop khi stream,
    chạy toàn bộ vòng lặp sync generator trong 1 thread nền, bridge từng chunk
    qua asyncio.Queue về coroutine gọi.
    """

    def __init__(self, config: LLMConfig):
        from google import genai

        self._client = genai.Client(api_key=config.api_key) if config.api_key else genai.Client()
        self._model = config.model
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens

    async def complete(self, system_prompt: str, user_message: str, stream: bool = True) -> AsyncGenerator[str, None]:
        from google.genai import types

        gen_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
        )

        if not stream:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model,
                contents=user_message,
                config=gen_config,
            )
            yield response.text or ""
            return

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def _run():
            try:
                response = self._client.models.generate_content_stream(
                    model=self._model,
                    contents=user_message,
                    config=gen_config,
                )
                for chunk in response:
                    if chunk.text:
                        loop.call_soon_threadsafe(queue.put_nowait, chunk.text)
            except Exception as exc:  # pragma: no cover - network/provider error path
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        threading.Thread(target=_run, daemon=True).start()

        while True:
            item = await queue.get()
            if item is sentinel:
                return
            if isinstance(item, Exception):
                raise item
            yield item


class OpenAIAdapter:
    """AsyncOpenAI-compatible provider (vLLM/OpenAI) — native async streaming."""

    def __init__(self, config: LLMConfig):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=config.base_url or None, api_key=config.api_key or "not-needed")
        self._model = config.model
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens

    async def complete(self, system_prompt: str, user_message: str, stream: bool = True) -> AsyncGenerator[str, None]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        if not stream:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stream=False,
            )
            yield response.choices[0].message.content or ""
            return

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def build_llm_adapter(config: LLMConfig) -> LLMAdapter:
    if config.provider == "openai":
        return OpenAIAdapter(config)
    return GeminiAdapter(config)
