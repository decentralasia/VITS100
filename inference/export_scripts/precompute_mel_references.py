#!/usr/bin/env python3

import os
import sys

import torch

# Add project root so we can import mel_processing directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from mel_processing import mel_spectrogram_torch
import utils
from utils import load_wav_to_torch_2

# Path to config used during training — must exist for mel params
PATH_TO_CONFIG = os.path.join(PROJECT_ROOT, "configs", "mbank.json")

# Hardcode list of WAV files to process (no CLI required).
# Example: set absolute paths to WAV files you want converted.
REAL_WAV_DIR = "/mnt/d/studio_audio_flat/wavs/"
SYNTHETIC_WAV_DIR = "/mnt/d/elevenlabs_generate_kyrgyz_voices/ru_tts/output"
SYNTHETIC_WAV_NAME = "000000.wav"

HARD_CODED_WAVS = [
    # lid=0 (ky) - sid=0 (Timur)
    os.path.join(REAL_WAV_DIR, "00003_003_inter_sounds_neutral_050_1_Timur_neutral_kg.wav"),
    os.path.join(REAL_WAV_DIR, "00000_000_news_quest_1-1.0_010_1_Timur_strict_kg.wav"),
    os.path.join(REAL_WAV_DIR, "00133_002_inter_sounds_neutral_069_14_Timur_friendly_kg.wav"),

    # sid=1 (Aiganysh)
    os.path.join(REAL_WAV_DIR, "00004_004_support_000-2_1_Aiganysh_neutral_kg.wav"),
    os.path.join(REAL_WAV_DIR, "00001_001_quest_tv_show_004_1_Aiganysh_strict_kg.wav"),
    os.path.join(REAL_WAV_DIR, "00000_000_sales_010_1_Aiganysh_friendly_kg.wav"),

    # ── Russian voices (lid=1)
    # sid=0 (Timur)
    os.path.join(REAL_WAV_DIR, "00003_003_ru_support_038_1_Timur_neutral_ru.wav"),
    os.path.join(REAL_WAV_DIR, "00002_002_ru_sales_004_1_Timur_strict_ru.wav"),
    os.path.join(REAL_WAV_DIR, "Timur_friendly_ru.wav"),

    # sid=1 (Aiganysh)
    os.path.join(REAL_WAV_DIR, "00003_003_ru_support_012_1_Aiganysh_neutral_ru.wav"),
    os.path.join(REAL_WAV_DIR, "00006_006_ru_support_050_1_Aiganysh_strict_ru.wav"),
    os.path.join(REAL_WAV_DIR, "Aiganysh_friendly_ru.wav"),

    # synthetic voices (ElevenLabs outputs)
    os.path.join(SYNTHETIC_WAV_DIR, "alexander_vlasov", SYNTHETIC_WAV_NAME),
    os.path.join(SYNTHETIC_WAV_DIR, "artem_lebedev", SYNTHETIC_WAV_NAME),
    os.path.join(SYNTHETIC_WAV_DIR, "kari", SYNTHETIC_WAV_NAME),
    os.path.join(SYNTHETIC_WAV_DIR, "larisa_actrisa", SYNTHETIC_WAV_NAME),
    os.path.join(SYNTHETIC_WAV_DIR, "nikolay", SYNTHETIC_WAV_NAME),
    os.path.join(SYNTHETIC_WAV_DIR, "rina", SYNTHETIC_WAV_NAME),
    os.path.join(SYNTHETIC_WAV_DIR, "victoria", SYNTHETIC_WAV_NAME),
]
# Output directory to use when HARD_CODED_WAVS is non-empty
HARD_CODED_OUTPUT_DIR = "../voices"


def process_wav(wav_path: str, output_path: str, hps):
    """Mirror of data_utils_speaker_tone_lang.get_audio() — identical mel computation to training."""
    audio, sampling_rate = load_wav_to_torch_2(wav_path, target_sampling_rate=hps.data.sampling_rate)
    if sampling_rate != hps.data.sampling_rate:
        raise ValueError(f"{wav_path}: SR {sampling_rate} doesn't match target {hps.data.sampling_rate}")
    audio_norm = audio / hps.data.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)

    mel = mel_spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
        center=False,
    )
    mel = torch.squeeze(mel, 0)  # [n_mels, T] — matches training saved format
    torch.save(mel, output_path)

    duration_s = audio.shape[0] / hps.data.sampling_rate
    print(f"  ✅ {os.path.basename(output_path)}: "
          f"shape={list(mel.shape)}, "
          f"duration={duration_s:.2f}s, "
          f"mel_frames={mel.shape[1]}")
    return mel


def main():
    # Only hardcoded list mode is supported. Populate HARD_CODED_WAVS at the top of the file.
    if 'HARD_CODED_WAVS' not in globals() or not HARD_CODED_WAVS:
        print("HARD_CODED_WAVS is empty — populate HARD_CODED_WAVS list in the script and re-run.")
        sys.exit(1)

    # Load training hyperparameters — required for exact mel parity with training
    hps = utils.get_hparams_from_file(PATH_TO_CONFIG)
    print(f"Loaded config from {PATH_TO_CONFIG}")

    output_dir = HARD_CODED_OUTPUT_DIR if 'HARD_CODED_OUTPUT_DIR' in globals() and HARD_CODED_OUTPUT_DIR else './voices'
    os.makedirs(output_dir, exist_ok=True)
    success = 0
    for wav_path in HARD_CODED_WAVS:
        if not os.path.isabs(wav_path):
            # Interpret relative paths as relative to this script file
            wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), wav_path)
        if not os.path.isfile(wav_path):
            print(f"  ⚠️  {wav_path} — not found, skipping")
            continue
        basename = os.path.basename(wav_path)
        if basename == SYNTHETIC_WAV_NAME:
            name = os.path.basename(os.path.dirname(wav_path))
        else:
            name = os.path.splitext(basename)[0]
        output_path = os.path.join(output_dir, f"{name}.mel.pt")
        process_wav(wav_path, output_path, hps)
        success += 1

    print(f"\nDone: {success}/{len(HARD_CODED_WAVS)} voices converted")
    if success == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
