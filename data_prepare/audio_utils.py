"""Shared audio I/O and noise mixing for feature extraction."""
from __future__ import annotations

import os
import random
import wave

import numpy as np


def read_audio(wav_path: str) -> np.ndarray:
    with wave.open(wav_path, "rb") as wf:
        sw = wf.getsampwidth()
        audio_data = wf.readframes(wf.getnframes())
    dtype = np.int16 if sw == 2 else np.int32
    return np.frombuffer(audio_data, dtype=dtype)


def write_audio(audio_f32: np.ndarray, path: str) -> None:
    import os

    i16 = (audio_f32 * 32768.0).clip(-32768, 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setparams((1, 2, 16000, len(i16), "NONE", "NONE"))
        wf.writeframes(i16.tobytes())


def audio_add_noise(
    clean_i16: np.ndarray,
    noise_i16: np.ndarray,
    snr: float,
    rng: random.Random | None = None,
) -> np.ndarray:
    clean = clean_i16.astype(np.float32) / 32768.0
    noise = noise_i16.astype(np.float32) / 32768.0
    cp = np.mean(clean ** 2)
    if len(noise) > len(clean):
        if rng is None:
            start = random.randint(0, len(noise) - len(clean))
        else:
            start = rng.randint(0, len(noise) - len(clean))
        noise = noise[start : start + len(clean)]
    else:
        noise = np.pad(noise, (0, len(clean) - len(noise)), "wrap")
    npow = np.mean(noise ** 2)
    if npow == 0:
        return clean
    scale = np.sqrt(cp / (10 ** (snr / 10) * npow))
    return clean + noise * scale


def noise_rng_for_utterance(wav_path: str, snr: int, base_seed: int = 42) -> random.Random:
    """Deterministic noise choice per (utterance, SNR) for reproducible utterance caches."""
    key = hash((os.path.normpath(wav_path), snr, base_seed)) & 0xFFFFFFFF
    return random.Random(key)
