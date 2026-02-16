"""Precompute mel spectrograms from reference wav files for ONNX inference."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import librosa
from scipy.io.wavfile import read
from mel_processing import mel_spectrogram_torch

# ---- Config (must match training config) ----
PATH_TO_CONFIG = "/mnt/d/mbank/config.json"
SAMPLING_RATE = 22050
MAX_WAV_VALUE = 32768.0
FILTER_LENGTH = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MEL_CHANNELS = 80
MEL_FMIN = 0.0
MEL_FMAX = None

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # inference/

# ---- Reference wav files ----
WAV_FILES = [
    "/mnt/d/mbank-audio/studio_audio_flat/wavs/00044_003_sales_mislamic_names_addresses_001_5_Aiganysh_friendly_kg.wav",
    "/mnt/d/mbank-audio/studio_audio_flat/wavs/00000_000_inter_sounds_neutral_035_1_Aiganysh_strict_kg.wav",
    "/mnt/d/mbank-audio/studio_audio_flat/wavs/00002_002_news_01-05_018_1_Aiganysh_neutral_kg.wav",
    "/mnt/d/mbank-audio/studio_audio_flat/wavs/00002_002_inter_sounds_neutral_048_1_Timur_strict_kg.wav",
    "/mnt/d/mbank-audio/studio_audio_flat/wavs/00005_005_inter_sounds_neutral_048_1_Timur_friendly_kg.wav",
    "/mnt/d/mbank-audio/studio_audio_flat/wavs/00001_001_news_01-05_001_1_Timur_neutral_kg.wav",
]


def load_wav(path):
    """Load and resample wav to target sampling rate (same as training pipeline)."""
    sr_orig, data = read(path)
    data = data.astype(np.float32)
    if sr_orig != SAMPLING_RATE:
        if data.ndim > 1:
            data = data.T
        data = librosa.resample(y=data, orig_sr=sr_orig, target_sr=SAMPLING_RATE)
    return torch.FloatTensor(data)


def compute_mel(audio):
    """Compute mel spectrogram exactly as in training."""
    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = audio_norm.unsqueeze(0)  # [1, T]
    mel = mel_spectrogram_torch(
        audio_norm, FILTER_LENGTH, N_MEL_CHANNELS, SAMPLING_RATE,
        HOP_LENGTH, WIN_LENGTH, MEL_FMIN, MEL_FMAX, center=False
    )
    return mel.squeeze(0)  # [n_mels, frames]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for wav_path in WAV_FILES:
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{basename}.mel.pt")

        if not os.path.exists(wav_path):
            print(f"SKIP (not found): {wav_path}")
            continue

        audio = load_wav(wav_path)
        mel = compute_mel(audio)
        torch.save(mel, out_path)
        print(f"OK: {basename}.mel.pt  shape={list(mel.shape)}")

    print(f"\nDone! Saved {len(WAV_FILES)} mel specs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
