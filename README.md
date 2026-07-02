# Realtime Speech Translation

Hệ thống dịch song song **thời gian thực** từ tiếng Việt (giọng nói) sang tiếng Anh (văn bản), hiển thị chia đôi màn hình: bên trái là transcript ASR đang chạy, bên phải là bản dịch đã chốt. Mục tiêu latency end-to-end dưới 1 giây.

Kiến trúc và toàn bộ core logic implement theo [`spec.md`](./spec.md).

```
Microphone → WebSocket → VAD → ASR (sliding window) → Translation Pipeline (LLM) → Display
```

## Tính năng chính

- **VAD chunk-based**: đánh giá từng chunk 32ms bằng Silero VAD (ONNX), tự điều chỉnh `threshold` / `min_silence_ms` / `min_speech_ms` runtime không cần restart server.
- **ASR sliding window**: cửa sổ 320ms, hop 160ms (overlap 160ms giữa 2 window liên tiếp) chạy trên sherpa-onnx (Zipformer tiếng Việt), tránh cắt ngang từ.
- **Translation Pipeline có state**: quản lý `frozen_segments` / `pending_buffer` / `display_buffer`, tự động strip phần overlap trùng lặp giữa các window, chỉ gửi phần transcript mới (diff) ra frontend.
- **LLM streaming + early-cancel**: phát hiện token `[WAIT]` ngay từ đầu stream để hủy sớm, không đợi hết generation khi câu chưa đủ nghĩa để dịch.
- **Provider swap được**: ASR và LLM đều qua interface adapter mỏng — đổi provider (vd. Gemini ↔ OpenAI/vLLM) không đụng vào pipeline.
- **Frontend AudioWorklet**: capture PCM trên audio thread (không phải `ScriptProcessorNode` đã deprecated), render append-only (không rebuild lại toàn bộ DOM mỗi lần nhận text mới).

## Kiến trúc & cấu trúc file

```
config.yaml                  # Cấu hình ASR / LLM / VAD / Translation, load lúc khởi động
config_manager.py            # Load config.yaml, apply runtime override (config_update)
vad.py                       # VAD chunk-based state machine + Silero speech-probability model
asr_adapter.py                # ASRAdapter interface + SherpaOnnxAdapter
llm_adapter.py                 # LLMAdapter interface + GeminiAdapter, OpenAIAdapter, early-cancel [WAIT]
diff_utils.py                 # Word-level diff dùng chung cho overlap-strip và display-diff
translation_pipeline.py      # Component trung tâm: state machine dịch theo spec §6
asr_translate_server.py      # WebSocket server, orchestrate VAD → ASR → Pipeline
index.html                   # Frontend: split-screen UI, VAD config panel, visualizer
audio-worklet.js              # AudioWorkletProcessor: gom PCM thành frame 32ms
tests/                        # Unit test cho toàn bộ core logic (không cần model/network thật)
```

### Luồng dữ liệu

1. Frontend capture audio qua `AudioWorklet`, gửi từng frame 32ms (512 sample @16kHz, PCM float32) qua WebSocket dạng binary.
2. `ChunkedSlidingVAD` (`vad.py`) đánh giá speech/silence từng chunk. Đủ 10 chunk speech liên tiếp (320ms) → forward window đầu tiên; sau đó cứ mỗi 5 chunk mới (160ms) → forward window kế tiếp (overlap 160ms).
3. `SherpaOnnxAdapter` (`asr_adapter.py`) decode window thành transcript đầy đủ (không phải diff).
4. `TranslationPipeline` (`translation_pipeline.py`) xử lý transcript đó:
   - Strip phần overlap nếu window đến sau điểm đã freeze gần nhất (trong cùng một đoạn speech liên tục).
   - Diff so với `display_buffer` → gửi event `asr_partial` (chỉ phần text mới) về pane bên trái.
   - Gọi LLM (`llm_adapter.py`) với `pending_buffer` + ngữ cảnh các câu đã dịch gần nhất.
   - `[WAIT]` → gửi event `wait`, tiếp tục tích lũy. Có bản dịch → freeze segment, gửi event `translation` về pane bên phải.

## Yêu cầu hệ thống

- Python 3.11+
- Model ASR: thư mục `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/` (đã có sẵn trong repo — `encoder`, `decoder`, `joiner`, `tokens.txt`)
- Model VAD: `silero_vad.onnx` (đã có sẵn trong repo)
- API key Gemini (mặc định) hoặc OpenAI-compatible endpoint (vLLM, v.v.)

### Dependencies

```bash
pip install websockets numpy sherpa_onnx onnxruntime google-genai openai pyyaml
```

Cho testing:

```bash
pip install pytest pytest-asyncio pytest-cov
```

## Cài đặt & chạy

### 1. Cấu hình API key

Server đọc `GEMINI_API_KEY` (hoặc `OPENAI_API_KEY` nếu dùng provider `openai`) từ biến môi trường — không hardcode key trong `config.yaml`.

```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 2. Chỉnh `config.yaml` nếu cần

```yaml
asr:
  provider: sherpa_onnx
  model_dir: ./sherpa-onnx-zipformer-vi-30M-int8-2026-02-09
  num_threads: 4

llm:
  provider: gemini              # hoặc "openai" (vLLM / OpenAI-compatible endpoint)
  model: models/gemini-3.1-flash-lite
  base_url:                     # cần cho provider openai (vd. http://localhost:8000/v1)
  temperature: 0.1
  max_tokens: 256

vad:
  model_path: ./silero_vad.onnx
  threshold: 0.5
  min_silence_ms: 400
  min_speech_ms: 200

translation:
  frozen_context_window: 3      # số câu đã dịch gần nhất dùng làm ngữ cảnh cho LLM
```

### 3. Chạy server

```bash
python asr_translate_server.py
```

Server lắng nghe WebSocket tại `ws://0.0.0.0:6006`.

### 4. Mở frontend

```bash
python -m http.server 8080
```

Mở `http://127.0.0.1:8080/index.html` (không dùng `0.0.0.0` — trình duyệt không kết nối được tới địa chỉ đó). Nếu mở từ máy khác, nhập đúng IP máy chạy server vào ô WebSocket URL, ví dụ `ws://192.168.x.x:6006`.

Bấm **Kết nối** → **▶ Ghi** để bắt đầu nói. Có thể chỉnh 3 tham số VAD (threshold, min silence, min speech) trực tiếp trên UI và bấm **Áp dụng VAD** — thay đổi có hiệu lực ngay, không cần reconnect.

## WebSocket Event Protocol

Downstream (server → client):

```jsonc
{ "type": "asr_partial",  "text": "một ngày đẹp trời" }        // phần transcript VI mới xuất hiện
{ "type": "wait",         "pending": "hôm nay là một ngày" }    // LLM chưa đủ nghĩa để dịch
{ "type": "translation",  "text": "Today is a beautiful day.", "segment_id": 3 }
{ "type": "error",        "code": "asr_error", "message": "..." }
```

Upstream (client → server):

```jsonc
{ "type": "config_update", "vad": { "threshold": 0.6, "min_silence_ms": 300 } }
{ "type": "reset" }   // xóa toàn bộ state phiên hiện tại (VAD + pipeline)
{ "type": "ping" }    // server trả { "type": "pong" }
```

## Testing

Toàn bộ core logic (VAD state machine, word-diff, translation pipeline state transitions, config manager, LLM early-cancel) có unit test — không cần model ASR hay network thật, dùng fake model / scripted LLM adapter.

```bash
pytest tests/ -v --cov=vad --cov=diff_utils --cov=translation_pipeline --cov=llm_adapter --cov=asr_adapter --cov=config_manager --cov-report=term-missing
```

Phần **không** unit test (cần model/network thật, verify bằng chạy tay theo hướng dẫn ở trên): decode thật của `SherpaOnnxAdapter`, gọi API thật của `GeminiAdapter`/`OpenAIAdapter`, và `SileroSpeechProbabilityModel` (ONNX session thật).

## Provider khác

Đổi sang OpenAI-compatible endpoint (vd. tự host vLLM) chỉ cần sửa `config.yaml`:

```yaml
llm:
  provider: openai
  model: qwen2.5-7b-instruct
  base_url: http://localhost:8000/v1
```

`OpenAIAdapter` cùng interface với `GeminiAdapter` (`llm_adapter.py`) — không cần đổi gì ở `translation_pipeline.py`.
