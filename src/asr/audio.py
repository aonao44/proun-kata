"""音声の読み込みと前処理ユーティリティ。"""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

TARGET_SAMPLE_RATE = 16_000


def load_waveform(audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    """WAVバイト列をモノラルTensorへ変換する。"""

    buffer = io.BytesIO(audio_bytes)
    data, sample_rate = sf.read(buffer, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    waveform = torch.from_numpy(data)
    return waveform, sample_rate


def ensure_sample_rate(waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
    """ターゲットサンプリング周波数への変換を行う。"""

    if sample_rate == TARGET_SAMPLE_RATE:
        return waveform, sample_rate
    resampled = AF.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
    return resampled, TARGET_SAMPLE_RATE
