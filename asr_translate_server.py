"""
ASR + Realtime Translation WebSocket Server

Luồng: audio (binary, 32ms/512-sample PCM float32 frames) → VAD (ChunkedSlidingVAD,
spec §4) → mỗi khi đủ window 320ms → ASRAdapter.transcribe (spec §5) → full transcript
window đó → TranslationPipeline.process_window (spec §6) → list events → gửi WebSocket.

Event protocol gửi về frontend (spec §2.4):
    {"type": "asr_partial",  "text": "..."}
    {"type": "wait",         "pending": "..."}
    {"type": "translation",  "text": "...", "segment_id": 3}
    {"type": "error",        "code": "...", "message": "..."}
"""

import asyncio
import json
import logging

import numpy as np
import websockets

from asr_adapter import SherpaOnnxAdapter
from config_manager import ConfigManager
from llm_adapter import build_llm_adapter
from translation_pipeline import TranslationPipeline
from vad import ChunkedSlidingVAD, SileroSpeechProbabilityModel

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
logger.info("Models loaded. Listening on ws://%s:%s", HOST, PORT)


async def handle_client(websocket):
    client_addr = websocket.remote_address
    logger.info("[+] %s connected", client_addr)

    # Mỗi client giữ VAD + pipeline riêng (VAD có recurrent state không share được)
    speech_model = SileroSpeechProbabilityModel(config.vad.model_path, sample_rate=SAMPLE_RATE)
    vad = ChunkedSlidingVAD(speech_model, config.vad, sample_rate=SAMPLE_RATE)
    llm_adapter = build_llm_adapter(config.llm)
    pipeline = TranslationPipeline(llm_adapter, config.translation)

    chunk_samples = vad.chunk_samples
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
                    vad.reset()
                    leftover = np.empty(0, dtype=np.float32)
                    await send({"type": "reset_ack"})
                    logger.info("[%s] Session reset", client_addr)

                elif msg_type == "config_update":
                    patch = {k: v for k, v in data.items() if k != "type"}
                    config_manager.apply_update(patch)
                    if "vad" in patch:
                        vad.update_config(
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
                    window = vad.push_chunk(chunk)
                except Exception as exc:
                    await send({"type": "error", "code": "vad_error", "message": str(exc)})
                    continue

                if window is None:
                    continue

                try:
                    asr_result = asr_adapter.transcribe(window.samples, window.window_start_ms, window.utterance_id)
                except Exception as exc:
                    await send({"type": "error", "code": "asr_error", "message": str(exc)})
                    continue

                if not asr_result.text:
                    continue

                events = await pipeline.process_window(asr_result)
                await send_all(events)

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
