"""
Audio loading cho eval mode.

Chấp nhận file audio bất kỳ format/sample rate/số kênh (wav/flac/mp3/ogg qua
soundfile), luôn trả về mono float32 tại đúng sample rate pipeline yêu cầu
(16kHz) — cùng dtype/format mà AudioWorklet gửi lên qua WebSocket, để audio
đi qua VAD/ASR y hệt như audio thật từ mic.
"""

from __future__ import annotations

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio_16k_mono(path: str, target_sr: int = 16000) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)

    # Downmix multi-channel bằng trung bình kênh (giữ nguyên năng lượng tương đối
    # giữa các kênh, không thiên vị kênh nào).
    mono = data.mean(axis=1).astype(np.float32)

    if sr == target_sr:
        return mono

    # resample_poly cần tỉ lệ nguyên up/down đã rút gọn — dùng gcd để tránh
    # upsample/downsample quá mức không cần thiết (vd. 44100/16000 rút về 160/441).
    g = gcd(target_sr, sr)
    up, down = target_sr // g, sr // g
    resampled = resample_poly(mono, up, down).astype(np.float32)
    return resampled
