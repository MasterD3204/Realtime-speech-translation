"""
ASR + Realtime Translation WebSocket Server

Luồng: audio (binary, 32ms/512-sample PCM float32 frames) → rolling VAD utterance
buffer → decode lại buffer định kỳ bằng ASR offline → LocalAgreement commit phần
ổn định → TranslationPipeline.process_window → list events → gửi WebSocket.

Event protocol gửi về frontend (spec §2.4):
    {"type": "asr_partial",  "text": "...", "replace": true|false}
    {"type": "translation_delta", "text": "..."}
    {"type": "wait",         "pending": "..."}
    {"type": "translation",  "text": "...", "segment_id": 3, "streamed": true|false}
    {"type": "error",        "code": "...", "message": "..."}
"""

import asyncio
import json
import logging

import numpy as np
import websockets

from asr_adapter import ASRResult, SherpaOnnxAdapter
from config_manager import ConfigManager
from llm_adapter import build_llm_adapter
from streaming_asr import RollingLocalAgreementASR
from translation_pipeline import TranslationPipeline
from vad import SileroSpeechProbabilityModel

HOST = "0.0.0.0"
PORT = 6006
SAMPLE_RATE = 16000
LOG_FILE = "server.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("server")

config_manager = ConfigManager.load("config.yaml")
config = config_manager.config

logger.info("Loading ASR model from %s ...", config.asr.model_dir)
asr_adapter = SherpaOnnxAdapter(config.asr.model_dir, num_threads=config.asr.num_threads, sample_rate=SAMPLE_RATE)
logger.info(
    "LLM provider=%s model=%s base_url=%s",
    config.llm.provider, config.llm.model, config.llm.base_url,
)
logger.info("Models loaded. Listening on ws://%s:%s", HOST, PORT)


async def handle_client(websocket):
    client_addr = websocket.remote_address
    logger.info("[+] %s connected", client_addr)

    # Mỗi client giữ ASR stream + pipeline riêng (VAD có recurrent state không share được).
    # Bọc try/except riêng: nếu adapter khởi tạo lỗi (thiếu key, sai base_url, model
    # không tồn tại...) phải thấy lỗi rõ ràng ngay, không được chết âm thầm trước khi
    # vòng lặp message bên dưới (với try/except riêng của nó) kịp chạy.
    try:
        speech_model = SileroSpeechProbabilityModel(config.vad.model_path, sample_rate=SAMPLE_RATE)
        streaming_asr = RollingLocalAgreementASR(
            speech_model=speech_model,
            vad_config=config.vad,
            asr_adapter=asr_adapter,
            sample_rate=SAMPLE_RATE,
            decode_hop_ms=config.asr.decode_hop_ms,
            agreement_n=config.asr.local_agreement_n,
        )
        llm_adapter = build_llm_adapter(config.llm)
        pipeline = TranslationPipeline(llm_adapter, config.translation)
    except Exception:
        logger.exception("[%s] Failed to initialize session (VAD/LLM adapter)", client_addr)
        await websocket.close(code=1011, reason="session init failed")
        return

    chunk_samples = streaming_asr.chunk_samples
    leftover = np.empty(0, dtype=np.float32)

    async def send(data: dict):
        try:
            await websocket.send(json.dumps(data, ensure_ascii=False))
        except Exception:
            pass

    async def send_all(events: list[dict]):
        for event in events:
            await send(event)
            if event["type"] == "translation":
                logger.info("[Translation #%s] %s", event["segment_id"], event["text"])
            elif event["type"] == "error":
                logger.error("[%s] %s", event["code"], event["message"])

    try:
        async for message in websocket:

            # ── Control messages ───────────────────────────────────────────
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")

                if msg_type == "ping":
                    await send({"type": "pong"})

                elif msg_type == "reset":
                    pipeline.reset()
                    streaming_asr.reset()
                    leftover = np.empty(0, dtype=np.float32)
                    await send({"type": "reset_ack"})
                    logger.info("[%s] Session reset", client_addr)

                elif msg_type == "config_update":
                    patch = {k: v for k, v in data.items() if k != "type"}
                    config_manager.apply_update(patch)
                    if "vad" in patch:
                        streaming_asr.update_config(
                            threshold=patch["vad"].get("threshold"),
                            min_silence_ms=patch["vad"].get("min_silence_ms"),
                            min_speech_ms=patch["vad"].get("min_speech_ms"),
                        )
                    logger.info("[%s] Config updated: %s", client_addr, patch)

                continue

            # ── PCM audio (bytes) ──────────────────────────────────────────
            if not isinstance(message, bytes):
                continue

            samples = np.frombuffer(message, dtype=np.float32)
            buffer = np.concatenate((leftover, samples))

            n_chunks = buffer.size // chunk_samples
            for i in range(n_chunks):
                chunk = buffer[i * chunk_samples:(i + 1) * chunk_samples]

                try:
                    asr_events = streaming_asr.push_chunk(chunk)
                except Exception as exc:
                    await send({"type": "error", "code": "asr_stream_error", "message": str(exc)})
                    continue

                for asr_event in asr_events:
                    if asr_event.type == "partial":
                        await send({
                            "type": "asr_partial",
                            "text": asr_event.text,
                            "replace": True,
                        })
                        continue

                    if asr_event.type != "commit" or not asr_event.text:
                        continue

                    await send({
                        "type": "asr_partial",
                        "text": asr_event.text,
                        "replace": False,
                    })
                    asr_result = ASRResult(
                        text=asr_event.text,
                        window_start_ms=asr_event.audio_ms,
                        utterance_id=asr_event.utterance_id,
                    )
                    async for event in pipeline.process_window_stream(
                        asr_result,
                        emit_asr_partial=False,
                        strip_overlap=False,
                        input_is_delta=True,
                    ):
                        await send(event)
                        if event["type"] == "translation":
                            logger.info("[Translation #%s] %s", event["segment_id"], event["text"])
                        elif event["type"] == "error":
                            logger.error("[%s] %s", event["code"], event["message"])

            leftover = buffer[n_chunks * chunk_samples:]

    except websockets.exceptions.ConnectionClosed:
        logger.info("[-] %s disconnected", client_addr)
    except Exception:
        logger.exception("Error from %s", client_addr)


async def main():
    async with websockets.serve(handle_client, HOST, PORT, max_size=None):
        logger.info("Server ready.")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
