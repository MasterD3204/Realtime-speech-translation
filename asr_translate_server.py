"""
ASR + Realtime Translation WebSocket Server
- Port 6006: nhận PCM float32 từ browser → partial ASR mỗi 32ms hoặc final ASR theo VAD
- Gửi về browser 3 loại event:
    {"type": "asr_partial", "text": "hôm nay là một ngày"}   ← transcript VI tạm thời
    {"type": "asr",         "text": "hôm nay là một ngày"}   ← transcript VI đã chốt theo VAD
    {"type": "translation_partial", "en": "..."}             ← bản dịch tạm thời
    {"type": "translation", "vi": "...", "en": "..."}         ← segment đã dịch xong
    {"type": "wait",        "pending": "..."}                 ← đang chờ (debug only)
"""

import asyncio
import json
import numpy as np
import sherpa_onnx
import websockets

from translation_pipeline import TranslationPipeline

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR   = "./sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"
SAMPLE_RATE = 16000
HOST        = "0.0.0.0"
PORT        = 6006
PARTIAL_CHUNK_SAMPLES = int(SAMPLE_RATE * 0.032)  # 32ms @ 16kHz = 512 samples
PARTIAL_MIN_DECODE_SAMPLES = int(SAMPLE_RATE * 0.128)  # cần đủ ngữ cảnh để offline ASR không vỡ shape
PARTIAL_MAX_SAMPLES = SAMPLE_RATE * 8

# ── VAD config (dùng chung, mỗi client tạo instance riêng) ───────────────────
vad_config = sherpa_onnx.VadModelConfig()
vad_config.silero_vad.model          = "./silero_vad.onnx"
vad_config.silero_vad.threshold      = 0.35
vad_config.silero_vad.min_silence_duration = 0.12
vad_config.silero_vad.min_speech_duration  = 0.25
vad_config.sample_rate = SAMPLE_RATE

# ── ASR (shared, thread-safe cho decode) ──────────────────────────────────────
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    tokens  = f"{MODEL_DIR}/tokens.txt",
    encoder = f"{MODEL_DIR}/encoder-epoch-20-avg-10.onnx",
    decoder = f"{MODEL_DIR}/decoder.onnx",
    joiner  = f"{MODEL_DIR}/joiner-epoch-20-avg-10.onnx",
    num_threads      = 4,
    decoding_method  = "greedy_search",
    debug            = False,
)

print(f"[Server] Models loaded. Listening on ws://{HOST}:{PORT}")


def decode_samples(samples: np.ndarray) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


async def handle_client(websocket):
    client_addr = websocket.remote_address
    print(f"[+] {client_addr} connected")

    # Mỗi client có VAD + pipeline riêng
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)

    pipeline = TranslationPipeline(
        on_translation=None,  # handled inline bên dưới
        on_wait=None,
    )
    partial_buffer = np.empty(0, dtype=np.float32)
    partial_progress = 0
    last_partial_text = ""

    async def send(data: dict):
        try:
            await websocket.send(json.dumps(data, ensure_ascii=False))
        except Exception:
            pass

    try:
        async for message in websocket:

            # ── Control messages ───────────────────────────────────────────
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "ping":
                    await send({"type": "pong"})

                elif data.get("type") == "reset":
                    pipeline.reset()
                    partial_buffer = np.empty(0, dtype=np.float32)
                    partial_progress = 0
                    last_partial_text = ""
                    await send({"type": "reset_ack"})
                    print(f"[{client_addr}] Session reset")

                continue

            # ── PCM audio (bytes) ──────────────────────────────────────────
            if not isinstance(message, bytes):
                continue

            samples = np.frombuffer(message, dtype=np.float32)
            vad.accept_waveform(samples)
            partial_buffer = np.concatenate((partial_buffer, samples))
            if partial_buffer.size > PARTIAL_MAX_SAMPLES:
                partial_buffer = partial_buffer[-PARTIAL_MAX_SAMPLES:]
            partial_progress += len(samples)

            finalized_any = False

            while not vad.empty():
                segment = vad.front
                vad.pop()
                finalized_any = True

                # 1. ASR decode theo VAD (ưu tiên cao nhất)
                asr_text = decode_samples(segment.samples)

                if not asr_text:
                    continue

                # 2. Gửi transcript VI đã chốt
                await send({"type": "asr", "text": asr_text})
                print(f"[ASR] {asr_text}")

                # 3. Đưa vào translation pipeline với ưu tiên VAD-final
                en_result = await pipeline.update_text(asr_text, finalize=True)

                if en_result is None:
                    # LLM trả [WAIT] — gửi debug event (UI có thể bỏ qua)
                    await send({"type": "wait", "pending": pipeline.pending_vi})
                    print(f"[LLM] WAIT: '{pipeline.pending_vi}'")
                else:
                    # Có bản dịch — gửi segment đã freeze
                    vi_seg, en_seg = pipeline.translated_segments[-1]
                    await send({"type": "translation", "vi": vi_seg, "en": en_seg})
                    print(f"[LLM] TRANSLATED: [{vi_seg}] → [{en_seg}]")

            if finalized_any:
                partial_buffer = np.empty(0, dtype=np.float32)
                partial_progress = 0
                last_partial_text = ""
                continue

            while partial_progress >= PARTIAL_CHUNK_SAMPLES and partial_buffer.size > 0:
                partial_progress -= PARTIAL_CHUNK_SAMPLES
                if partial_buffer.size < PARTIAL_MIN_DECODE_SAMPLES:
                    continue

                partial_window = partial_buffer[-PARTIAL_MAX_SAMPLES:]

                try:
                    partial_text = decode_samples(partial_window)
                except RuntimeError as exc:
                    print(f"[ASR partial skipped] {exc}")
                    continue

                if partial_text and partial_text != last_partial_text:
                    last_partial_text = partial_text
                    await send({"type": "asr_partial", "text": partial_text})
                    en_preview = await pipeline.update_text(partial_text, finalize=False)
                    if en_preview is None:
                        await send({"type": "wait", "pending": pipeline.pending_vi})
                    else:
                        await send({"type": "translation_partial", "en": en_preview})

    except websockets.exceptions.ConnectionClosed:
        print(f"[-] {client_addr} disconnected")
    except Exception as e:
        print(f"[!] Error from {client_addr}: {e}")
        import traceback; traceback.print_exc()


async def main():
    async with websockets.serve(handle_client, HOST, PORT):
        print("[Server] Ready.")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
