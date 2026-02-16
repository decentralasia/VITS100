"""Quick sanity check for the exported ONNX model.

Replicates the eval logic from train.py:
  - Converts test sentences to token IDs (Kyrgyz + Russian)
  - Runs inference for each (speaker, tone) combo
  - Saves wav files to inference/test_outputs/
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import numpy as np
import onnxruntime as ort
from scipy.io.wavfile import write as wav_write

import commons

# ---- Text processing (mirrors data_utils_speaker_tone_lang.py) ----

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SYMBOLS_ENCODING_PATH = os.path.join(_PROJECT_ROOT, "AIF/model_artifacts/symbols_encoding_train_v7.json")
_TRANSLITERATION_MAPPING_PATH = os.path.join(_PROJECT_ROOT, "AIF/phonimization_artifacts/transliteration_mapping.json")
_FST_PATH = os.path.join(_PROJECT_ROOT, "AIF/phonimization_artifacts/ipa.ohfst")

with open(_SYMBOLS_ENCODING_PATH, "r", encoding="utf-8") as f:
    SYMBOLS_MAPPING = json.load(f)

BLANK_TOKEN_ID = max(SYMBOLS_MAPPING.values()) + 2

with open(_TRANSLITERATION_MAPPING_PATH, "r", encoding="utf-8") as f:
    TRANSLITERATION_MAPPING = json.load(f)

sys.path.append(os.path.join(_PROJECT_ROOT, "AIF"))
from preprocessing_utils.ky_phonemizer import KyrgyzToIpaV2

KYRGYZ_PHONEMIZER = KyrgyzToIpaV2(
    symbols_mapping=SYMBOLS_MAPPING,
    transliteratetion_mapping=TRANSLITERATION_MAPPING,
    fst_path=_FST_PATH,
)

MAPPING_SOUND = {
    "<inhale/>": "ω", "<yawn/>": "ξ", "<cough/>": "σ",
    "<inhale>": "ω",  "<yawn>": "ξ",  "<cough>": "σ",
    "<exhale>": "τ",
}

_whitespace_re = re.compile(r"\s+")


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def clean_spaces(text):
    return re.sub(r"\s*([@,.?!])\s*", r"\1", text)


def _process_phonemized_with_highlights(text):
    text_norm, is_highlighted = [], []
    in_highlight = False
    for ch in text:
        if ch == '*':
            in_highlight = not in_highlight
            continue
        if ch in SYMBOLS_MAPPING:
            text_norm.append(SYMBOLS_MAPPING[ch])
            if ch == '^' and is_highlighted and is_highlighted[-1] == 1:
                is_highlighted.append(1)
            else:
                is_highlighted.append(1 if in_highlight else 0)
    return text_norm, is_highlighted


def _process_russian_with_highlights(text):
    highlight_positions = set()
    i = 0
    while i < len(text):
        if text[i].isalpha() or text[i] == '^':
            start = i
            while i < len(text) and (text[i].isalpha() or text[i] == '^'):
                i += 1
            word = text[start:i]
            letters = [c for c in word if c.isalpha()]
            if letters and all(c.isupper() for c in letters):
                for j in range(start, i):
                    highlight_positions.add(j)
        else:
            i += 1

    text_norm, is_highlighted = [], []
    for i, ch in enumerate(text):
        ch_lower = ch.lower()
        if ch_lower in SYMBOLS_MAPPING:
            text_norm.append(SYMBOLS_MAPPING[ch_lower])
            is_highlighted.append(1 if i in highlight_positions else 0)
    return text_norm, is_highlighted


def _intersperse_emphasis(is_highlighted):
    if not is_highlighted:
        return [0]
    result = []
    for i, h in enumerate(is_highlighted):
        blank_emph = h if i == 0 else (1 if is_highlighted[i - 1] or h else 0)
        result.append(blank_emph)
        result.append(h)
    result.append(is_highlighted[-1])
    return result


def get_text(text, lid, add_blank=True):
    """Convert raw text to (token_ids, emphasis) numpy arrays."""
    is_kyrgyz = lid in ("kg", "ky")
    if not text.startswith("@"):
        text = "@" + text
    for tag, sym in MAPPING_SOUND.items():
        text = text.replace(tag, sym)

    if is_kyrgyz:
        phonemized = KYRGYZ_PHONEMIZER.phonemize(text)
        phonemized = clean_spaces(collapse_whitespace(phonemized)).strip()
        text_norm, is_highlighted = _process_phonemized_with_highlights(phonemized)
    else:
        text_clean = clean_spaces(collapse_whitespace(text)).strip()
        text_norm, is_highlighted = _process_russian_with_highlights(text_clean)

    if add_blank:
        text_norm = commons.intersperse(text_norm, BLANK_TOKEN_ID)
        is_highlighted = _intersperse_emphasis(is_highlighted)

    return np.array(text_norm, dtype=np.int32), np.array(is_highlighted, dtype=np.int32)


# ---- Config ----

ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.onnx")
SAMPLING_RATE = 22050
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_outputs")

# Same test sentences as train.py evaluate()
TEST_CASES = [
    # (text, lid_str, lid_int, sid, tid, description)
    ("рыноктук шартка ылайыкташкан ушул ишканалар өнөр жай, курулуш, транспорт, соода же тейлөөнүн башка тармактарына таандык.", "ky", 0, 0, 0, "timur_neutral_ky"),
    ("рыноктук шартка ылайыкташкан ушул ишканалар өнөр жай, курулуш, транспорт, соода же тейлөөнүн башка тармактарына таандык.", "ky", 0, 0, 1, "timur_strict_ky"),
    ("рыноктук шартка ылайыкташкан ушул ишканалар өнөр жай, курулуш, транспорт, соода же тейлөөнүн башка тармактарына таандык.", "ky", 0, 0, 2, "timur_friendly_ky"),
    ("рыноктук шартка ылайыкташкан ушул ишканалар өнөр жай, курулуш, транспорт, соода же тейлөөнүн башка тармактарына таандык.", "ky", 0, 1, 0, "aiganysh_neutral_ky"),
    ("рыноктук шартка ылайыкташкан ушул ишканалар өнөр жай, курулуш, транспорт, соода же тейлөөнүн башка тармактарына таандык.", "ky", 0, 1, 1, "aiganysh_strict_ky"),
    ("рыноктук шартка ылайыкташкан ушул ишканалар өнөр жай, курулуш, транспорт, соода же тейлөөнүн башка тармактарына таандык.", "ky", 0, 1, 2, "aiganysh_friendly_ky"),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading ONNX model: {ONNX_PATH}")
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    print("Model inputs:")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: {inp.type} {inp.shape}")
    print("Model outputs:")
    for out in sess.get_outputs():
        print(f"  {out.name}: {out.type} {out.shape}")
    print()

    for text, lid_str, lid_int, sid, tid, desc in TEST_CASES:
        tokens, emphasis = get_text(text, lid_str, add_blank=True)

        # Add batch dim: [1, seq_len]
        tokens_inp = tokens[np.newaxis, :]
        emphasis_inp = emphasis[np.newaxis, :]
        sid_inp = np.array([sid], dtype=np.int32)
        tid_inp = np.array([tid], dtype=np.int32)
        lid_inp = np.array([lid_int], dtype=np.int32)

        feed = {
            "input": tokens_inp,
            "emphasis": emphasis_inp,
            "sid": sid_inp,
            "tid": tid_inp,
        }
        # lid is pruned by ONNX tracing (unused in computation), only add if present
        model_input_names = {inp.name for inp in sess.get_inputs()}
        if "lid" in model_input_names:
            feed["lid"] = lid_inp

        raw_waveform, y_length = sess.run(None, feed)

        print(f"  DEBUG {desc}: waveform shape={raw_waveform.shape}, y_length={y_length}, "
              f"min={raw_waveform.min():.6f}, max={raw_waveform.max():.6f}, "
              f"abs_mean={np.abs(raw_waveform).mean():.6f}")

        # y_length is in mel frames — convert to audio samples
        HOP_LENGTH = 256
        y_len_samples = int(y_length[0]) * HOP_LENGTH
        audio_full = raw_waveform[0]
        audio = audio_full[:min(y_len_samples, len(audio_full))]
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        out_path = os.path.join(OUTPUT_DIR, f"{desc}.wav")
        wav_write(out_path, SAMPLING_RATE, audio_int16)
        print(f"  OK  {desc:30s}  tokens={len(tokens):4d}  audio={len(audio):6d} samples ({len(audio)/SAMPLING_RATE:.2f}s)")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
