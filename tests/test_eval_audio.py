import numpy as np
import soundfile as sf

from eval_audio import load_audio_16k_mono


def make_sine(duration_s: float, sr: int, freq: float = 440.0, channels: int = 1) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    tone = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    if channels == 1:
        return tone
    return np.stack([tone] * channels, axis=1)


def test_load_audio_already_at_target_sample_rate(tmp_path):
    path = tmp_path / "audio.wav"
    sf.write(str(path), make_sine(1.0, 16000), 16000)

    audio = load_audio_16k_mono(str(path))

    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(len(audio) - 16000) <= 1  # tolerate off-by-one from encode/decode


def test_load_audio_resamples_from_higher_sample_rate(tmp_path):
    path = tmp_path / "audio.wav"
    sf.write(str(path), make_sine(1.0, 44100), 44100)

    audio = load_audio_16k_mono(str(path))

    assert audio.dtype == np.float32
    # 1s of audio at 44100Hz resampled to 16000Hz should be ~16000 samples
    assert abs(len(audio) - 16000) <= 5


def test_load_audio_resamples_from_lower_sample_rate(tmp_path):
    path = tmp_path / "audio.wav"
    sf.write(str(path), make_sine(1.0, 8000), 8000)

    audio = load_audio_16k_mono(str(path))

    assert abs(len(audio) - 16000) <= 5


def test_load_audio_downmixes_stereo_to_mono(tmp_path):
    path = tmp_path / "audio.wav"
    sf.write(str(path), make_sine(0.5, 16000, channels=2), 16000)

    audio = load_audio_16k_mono(str(path))

    assert audio.ndim == 1
    assert abs(len(audio) - 8000) <= 1


def test_load_audio_flac_format(tmp_path):
    path = tmp_path / "audio.flac"
    sf.write(str(path), make_sine(0.5, 16000), 16000, format="FLAC")

    audio = load_audio_16k_mono(str(path))

    assert audio.dtype == np.float32
    assert abs(len(audio) - 8000) <= 1
