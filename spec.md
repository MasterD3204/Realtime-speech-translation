# Spec Pipeline: Realtime Streaming Translation

---

## 1. Tổng quan

Hệ thống dịch realtime từ speech nguồn (VI) sang text đích (EN), hiển thị song song hai bên màn hình. Mục tiêu latency end-to-end < 1 giây.

Luồng chính:

```
Microphone → WebSocket → VAD → ASR (sliding window) → Text Buffer → LLM → Display
```

---

## 2. Frontend

### 2.1 Audio Capture
- Dùng `getUserMedia` + `AudioWorklet` capture raw PCM liên tục
- Gửi audio frames lên backend qua WebSocket dưới dạng binary, mỗi frame 32ms

### 2.2 VAD Config Panel
- UI cho phép chỉnh các tham số VAD trực tiếp không cần restart:
  - `threshold` (0.0 – 1.0): ngưỡng energy để phân biệt speech / silence
  - `min_silence_ms`: thời gian im lặng tối thiểu để drop chunk
  - `min_speech_ms`: độ dài tối thiểu để tính là speech hợp lệ
- Khi user chỉnh → gửi event `config_update` qua WebSocket → backend apply ngay

### 2.3 Split-screen Display
- Màn hình chia đôi:
  - **Trái (nguồn VI)**: hiển thị ASR output, chỉ append phần text mới (diff), không rerender lại toàn bộ
  - **Phải (đích EN)**: hiển thị bản dịch từ LLM, chỉ append khi LLM confirm đủ nghĩa, không bao giờ rewrite đoạn đã freeze

### 2.4 WebSocket Event Protocol (downstream từ backend)
```json
{ "type": "asr_partial",  "text": "Hôm nay là" }
{ "type": "wait",         "pending": "Hôm nay là một ngày" }
{ "type": "translation",  "text": "Today is a beautiful day.", "segment_id": 3 }
{ "type": "error",        "code": "asr_timeout", "message": "..." }
```

---

## 3. Backend

### 3.1 WebSocket Server
- Nhận binary audio frames từ frontend
- Nhận `config_update` JSON để update VAD config runtime
- Điều phối luồng: audio → VAD → ASR → Translation Pipeline
- Gửi events về frontend

### 3.2 Config Manager
- Load config từ `config.yaml` khi khởi động
- Nhận override runtime qua WebSocket event `config_update`
- Apply ngay vào VAD layer không cần restart process

```yaml
asr:
  provider: sherpa_onnx
  model: zipformer-30m
  language: vi

llm:
  provider: openai
  model: qwen2.5-7b-instruct
  base_url: http://localhost:8000
  api_key: 
  temperature: 0.1
  max_tokens: 150

vad:
  threshold: 0.5
  min_silence_ms: 400
  min_speech_ms: 200

translation:
  source_lang: vi
  target_lang: en
  frozen_context_window: 3
```

---

## 4. VAD Layer

- Nhận audio stream liên tục dạng chunk 32ms
- Đánh giá từng chunk: speech hay silence dựa trên threshold từ Config Manager
- Chunk silence → DROP, reset bộ đếm
- Chunk speech → tích lũy, tăng bộ đếm
- Đủ 10 chunk liên tiếp (320ms) → forward audio window xuống ASR
- Có thể swap implementation (SileroVAD hoặc khác) mà không ảnh hưởng tầng trên

---

## 5. ASR Layer

### 5.1 Sliding Window
- Window size: 320ms (10 chunk × 32ms)
- Overlap: 160ms — mỗi window mới bao gồm 160ms cuối của window trước
- Mục đích overlap: tránh cắt ngang từ, giúp ASR có context để transcript chính xác hơn

### 5.2 ASR Adapter Interface (thin adapter)
```
interface ASRAdapter:
    transcribe(audio_window: bytes) → ASRResult

ASRResult:
    text: str
    window_start_ms: int
```

Các provider implement: `SherpaOnnxAdapter`

### 5.3 Output
- Mỗi window → ASR trả ra full transcript của window đó (không chỉ phần mới)
- Ví dụ:
  ```
  Window 1 (0–320ms):   "Hôm nay là"
  Window 2 (160–480ms): "Hôm nay là một ngày"
  Window 3 (320–640ms): "Hôm nay là một ngày đẹp trời"
  ```

---

## 6. Translation Pipeline

Đây là component trung tâm, quản lý toàn bộ state của một segment đang được dịch.

### 6.1 State

```python
frozen_segments: list[str]   # các đoạn đã dịch xong, bất biến, không bao giờ rewrite
pending_buffer: str          # text đang tích lũy, chưa đủ nghĩa để dịch
display_buffer: str          # dùng để diff cho ASR display layer
last_freeze_ms: int          # timestamp của lần freeze gần nhất
```

### 6.2 Xử lý ASR output

Mỗi khi nhận ASR output từ một window:

**Bước 1 — Strip overlap nếu window nằm trước điểm freeze:**
```
Nếu window_start_ms < last_freeze_ms:
    Bỏ phần text trùng với frozen text gần nhất
    Chỉ giữ lại phần text sau điểm freeze
```
Mục đích: tránh đoạn đã freeze bị lẫn vào pending_buffer mới do overlap.

**Bước 2 — Update display (ASR side, màn hình trái):**
```
new_display = diff(display_buffer, asr_output)
display_buffer = asr_output
Gửi event asr_partial với new_display về frontend → append vào màn hình trái
```
Chỉ phần text mới xuất hiện so với window trước được hiển thị, không rerender lại.

**Bước 3 — Update pending_buffer:**
```
pending_buffer = asr_output (toàn bộ, không phải diff)
```

**Bước 4 — Gọi LLM:**
```
Gửi pending_buffer vào LLM Adapter
```

### 6.3 Xử lý LLM response

```
Nếu response == "[WAIT]":
    Gửi event wait về frontend
    Tiếp tục accumulate, không làm gì thêm

Nếu response là bản dịch:
    frozen_segments.append(response)
    last_freeze_ms = current_time_ms
    pending_buffer = ""
    display_buffer = ""
    Gửi event translation về frontend → append vào màn hình phải
```

---

## 7. LLM Layer

### 7.1 Prompt Design
Cần làm chi tiết hơn, prompt này chỉ là ví dụ.
```
System:
  Bạn là module dịch hội nghị streaming VI→EN.

  Đoạn đã dịch trước (KHÔNG dịch lại, chỉ dùng làm ngữ cảnh):
  [1] {frozen_segments[-3]}
  [2] {frozen_segments[-2]}
  [3] {frozen_segments[-1]}

  Quy tắc:
  - Nếu ĐOẠN MỚI chưa đủ nghĩa → trả về đúng 1 token: [WAIT]
  - Nếu đủ nghĩa → chỉ dịch ĐOẠN MỚI, không lặp lại đoạn cũ, không giải thích

User:
  ĐOẠN MỚI: {pending_buffer}
```

### 7.2 LLM Adapter Interface (thin adapter)

```
interface LLMAdapter:
    complete(prompt: str, stream: bool) → AsyncGenerator[str]

LLMConfig:
    provider: openai | gemini
    model: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int
```

Các provider implement: `OpenAIAdapter`, `GeminiAdapter`

### 7.3 Streaming output
- Bật `stream=True` với vLLM để token đầu tiên xuất hiện ngay khi LLM bắt đầu generate
- Phát hiện `[WAIT]` ngay từ token đầu → cancel stream sớm, không chờ hết generation
- Bản dịch thật → stream từng token về frontend để hiển thị dần

---

## 8. Tóm tắt luồng dữ liệu hoàn chỉnh

```
32ms audio chunk
    │
    ▼
VAD: speech? → No  → DROP, reset counter
             → Yes → counter++, đủ 10? → forward 320ms window
    │
    ▼
ASR (sliding window, overlap 160ms)
    → trả ra full transcript của window
    │
    ▼
Translation Pipeline:
    1. window_start < last_freeze_ms? → strip overlap
    2. diff(display_buffer, asr_output) → asr_partial event → màn hình trái
    3. pending_buffer = asr_output (full)
    4. gọi LLM(pending_buffer, frozen_context)
    │
    ├── [WAIT] → wait event, tiếp tục
    └── dịch  → frozen_segments.append()
               → last_freeze_ms = now
               → pending_buffer = ""
               → display_buffer = ""
               → translation event → màn hình phải (append only)
```