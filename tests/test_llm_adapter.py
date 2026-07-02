import pytest

from config_manager import LLMConfig
from llm_adapter import GeminiAdapter, OpenAIAdapter, build_llm_adapter, consume_with_wait_detection


async def gen(chunks: list[str]):
    for c in chunks:
        yield c


@pytest.mark.asyncio
async def test_wait_detected_from_single_token():
    result = await consume_with_wait_detection(gen(["[WAIT]"]))
    assert result is None


@pytest.mark.asyncio
async def test_wait_detected_across_multiple_tokens():
    result = await consume_with_wait_detection(gen(["[WA", "IT]"]))
    assert result is None


@pytest.mark.asyncio
async def test_wait_cancels_stream_early_no_further_tokens_consumed():
    consumed = []

    async def tracking_gen():
        for c in ["[WAIT]", "should", "not", "be", "read"]:
            consumed.append(c)
            yield c

    result = await consume_with_wait_detection(tracking_gen())
    assert result is None
    assert consumed == ["[WAIT]"]


@pytest.mark.asyncio
async def test_full_translation_returned_when_stream_completes_naturally():
    result = await consume_with_wait_detection(gen(["Today ", "is ", "a ", "nice ", "day."]))
    assert result == "Today is a nice day."


@pytest.mark.asyncio
async def test_partial_match_of_wait_prefix_does_not_trigger_cancel():
    # "[WAIT" một mình (chưa đủ "]") không được coi là WAIT — phải đợi thêm token
    result = await consume_with_wait_detection(gen(["[WAIT", " actually a sentence."]))
    assert result == "[WAIT actually a sentence."


def test_build_llm_adapter_defaults_to_gemini():
    config = LLMConfig(provider="gemini", api_key="fake-key-for-init-only")
    adapter = build_llm_adapter(config)
    assert isinstance(adapter, GeminiAdapter)


def test_build_llm_adapter_selects_openai_provider():
    config = LLMConfig(provider="openai", base_url="http://localhost:8000/v1", api_key="not-needed")
    adapter = build_llm_adapter(config)
    assert isinstance(adapter, OpenAIAdapter)


def test_build_llm_adapter_unknown_provider_falls_back_to_gemini():
    config = LLMConfig(provider="something_else", api_key="fake-key-for-init-only")
    adapter = build_llm_adapter(config)
    assert isinstance(adapter, GeminiAdapter)
