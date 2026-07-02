
from google import genai
from google.genai import types
client = genai.Client()

prompt = "Xin chào hôm nay là một ngày đẹp trời nhỉ"

systemPropmt = """
Bạn là module dịch hội nghị thời gian thực.

Đầu vào là một đoạn transcript ASR chưa chắc đã kết thúc câu.

Quy tắc:
- Nếu thông tin chưa đủ để dịch tự nhiên, chỉ trả về: [WAIT]
- Nếu đủ để dịch, chỉ trả về bản dịch tiếng Anh.
- Không giải thích.
- Không thêm bất kỳ ký tự nào ngoài bản dịch hoặc [WAIT].
"""

response = client.models.generate_content_stream(
    model="models/gemini-3.1-flash-lite",
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction=(systemPropmt
        )
    )
)

for chunk in response:
    if chunk.text:
        print(chunk.text, end="", flush=True)

print()

# models/gemini-2.5-flash
# models/gemini-2.5-pro
# models/gemini-2.0-flash
# models/gemini-2.0-flash-001
# models/gemini-2.0-flash-lite-001
# models/gemini-2.0-flash-lite
# models/gemini-2.5-flash-preview-tts
# models/gemini-2.5-pro-preview-tts
# models/gemma-4-26b-a4b-it
# models/gemma-4-31b-it
# models/gemini-flash-latest
# models/gemini-flash-lite-latest
# models/gemini-pro-latest
# models/gemini-2.5-flash-lite
# models/gemini-2.5-flash-image
# models/gemini-3-pro-preview
# models/gemini-3-flash-preview
# models/gemini-3.1-pro-preview
# models/gemini-3.1-pro-preview-customtools
# models/gemini-3.1-flash-lite-preview
# models/gemini-3.1-flash-lite
# models/gemini-3-pro-image-preview
# models/gemini-3-pro-image
# models/nano-banana-pro-preview
# models/gemini-3.1-flash-image-preview
# models/gemini-3.1-flash-image
# models/gemini-3.5-flash
# models/lyria-3-clip-preview
# models/lyria-3-pro-preview
# models/gemini-3.1-flash-tts-preview
# models/gemini-robotics-er-1.5-preview
# models/gemini-robotics-er-1.6-preview
# models/gemini-2.5-computer-use-preview-10-2025
# models/antigravity-preview-05-2026
# models/deep-research-max-preview-04-2026
# models/deep-research-preview-04-2026
# models/deep-research-pro-preview-12-2025
# models/gemini-embedding-001
# models/gemini-embedding-2-preview
# models/gemini-embedding-2
# models/aqa
# models/imagen-4.0-generate-001
# models/imagen-4.0-ultra-generate-001
# models/imagen-4.0-fast-generate-001
# models/veo-2.0-generate-001
# models/veo-3.0-generate-001
# models/veo-3.0-fast-generate-001
# models/veo-3.1-generate-preview
# models/veo-3.1-fast-generate-preview
# models/veo-3.1-lite-generate-preview
# models/gemini-2.5-flash-native-audio-latest
# models/gemini-2.5-flash-native-audio-preview-09-2025
# models/gemini-2.5-flash-native-audio-preview-12-2025
# models/gemini-3.1-flash-live-preview
# models/gemini-3.5-live-translate-preview