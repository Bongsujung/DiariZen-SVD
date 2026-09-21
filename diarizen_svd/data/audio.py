#!/usr/bin/env python3
"""Waveform loading for the calibration sweep."""
import numpy as np
import torch


def load_wav(path: str, sr: int = 16000, max_s: float = 16.0):
    """Load a recording (first channel, resampled to 16 kHz) and cut it into 16 s windows (>= 4 s kept)."""
    try:
        import soundfile as sf
        wav, fs = sf.read(path, dtype="float32")
    except Exception:
        import torchaudio
        t, fs = torchaudio.load(path)
        wav = t[0].numpy()
    if wav.ndim > 1:
        wav = wav[:, 0]
    wav = torch.from_numpy(np.ascontiguousarray(wav)).float()
    if fs != sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav, fs, sr)
    n = int(max_s * sr)
    return [wav[i:i + n] for i in range(0, max(1, len(wav) - sr), n) if len(wav[i:i + n]) >= 4 * sr]
