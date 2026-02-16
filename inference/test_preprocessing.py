#!/usr/bin/env python3
"""
Standalone test for the preprocessing pipeline (no Triton/Docker).
Runs normalization → accentuation → phonemization → tokenization locally.

IMPORTANT: hfst (phonemizer) must be loaded BEFORE pynini (normalizer)
because both link against libfst.so and the normalizer's version breaks
hfst's ability to read .ohfst files.

Usage:
    python test_preprocessing.py --text "Салам дүйнө"
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_REPO = os.path.join(SCRIPT_DIR, "triton_model_repo/vits-ky-6-preprocessing")

# ---- Load phonemizer FIRST (before normalizer to avoid libfst.so conflict) ----

sys.path.insert(0, MODEL_REPO)
from preprocessing_utils.ky_phonemizer import KyrgyzToIpaV2

symbols_path = os.path.join(MODEL_REPO, "model_artifacts/symbols_encoding_train_v7.json")
translit_path = os.path.join(MODEL_REPO, "phonimization_artifacts/transliteration_mapping.json")
fst_path = os.path.join(MODEL_REPO, "phonimization_artifacts/ipa.ohfst")

with open(symbols_path, "r", encoding="utf-8") as f:
    symbols_mapping = json.load(f)
with open(translit_path, "r", encoding="utf-8") as f:
    translit_mapping = json.load(f)

BLANK_TOKEN_ID = max(symbols_mapping.values()) + 2

phonemizer = KyrgyzToIpaV2(
    symbols_mapping=symbols_mapping,
    transliteratetion_mapping=translit_mapping,
    fst_path=fst_path,
)

# ---- Load normalizer + accentor AFTER phonemizer ----

tts_pkg_src = os.path.join(PROJECT_ROOT, "triton_model_repo/vits-ky-6-preprocessing/tts-text-processing-kg/src")
sys.path.insert(0, tts_pkg_src)

from tts_text_processing_kg.normalizer.kg import Normalizer, NormalizerConfig
from tts_text_processing_kg.accentor.kg.accentor import Accentor, AccentorConfig
from tts_text_processing_kg.accentor.kg.vowel import Vowel
from tts_text_processing_kg.accentor.kg.rules import Rules

cache_dir = Path(tts_pkg_src).parent / "cache"
normalizer = Normalizer(config=NormalizerConfig(
    cache_dir=cache_dir,
    cache_name="tts_normalizer_kg",
    cache_overwrite=False,
    data_dir=tts_pkg_src + "/tts_text_processing_kg/normalizer/kg/data/",
))

dict_path = tts_pkg_src + "/tts_text_processing_kg/accentor/kg/data/accentor_kg.tsv"
rules_path = tts_pkg_src + "/tts_text_processing_kg/accentor/kg/configs/rules_kg.yaml"
with open(rules_path, "r") as f:
    rules = yaml.safe_load(f)

vowels_list = [Vowel(unaccented=v, accented=v + "^") for v in rules["vowels"]]
accentor = Accentor(
    config=AccentorConfig(
        dictionary_path=dict_path,
        accent_single_syllable_words=False,
        accent_last_syllable=True,
        vowels=vowels_list,
    ),
    rules=Rules(config=rules),
)

# ---- Helpers ----

_ws_re = re.compile(r"\s+")

SOUND_MAPPING = {
    "<inhale/>": "\u03c9", "<yawn/>": "\u03be", "<cough/>": "\u03c3",
    "<inhale>": "\u03c9",  "<yawn>": "\u03be",  "<cough>": "\u03c3",
    "<exhale>": "\u03c4",
}


def collapse_ws(text):
    return re.sub(_ws_re, " ", text)


def clean_spaces(text):
    return re.sub(r"\s*([@,.?!])\s*", r"\1", text)


def process_phonemized(text):
    text_norm, is_highlighted = [], []
    in_hl = False
    for ch in text:
        if ch == '*':
            in_hl = not in_hl
            continue
        if ch in symbols_mapping:
            text_norm.append(symbols_mapping[ch])
            if ch == '^' and is_highlighted and is_highlighted[-1] == 1:
                is_highlighted.append(1)
            else:
                is_highlighted.append(1 if in_hl else 0)
    return text_norm, is_highlighted


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def intersperse_emphasis(hl):
    if not hl:
        return [0]
    result = []
    for i, h in enumerate(hl):
        blank = h if i == 0 else (1 if hl[i - 1] or h else 0)
        result.append(blank)
        result.append(h)
    result.append(hl[-1])
    return result


def process_text(text):
    text = text.lower()
    if text and text[-1] not in ".!?;":
        text += "."

    # 1. Normalize
    normalized = clean_spaces(collapse_ws(normalizer.normalize(text)))

    # 2. Accentuate
    accentuated = clean_spaces(collapse_ws(accentor.accent_text(normalized))).strip()

    # 3. Prepare for phonemizer
    phonemizer_input = accentuated
    if not phonemizer_input.startswith("@"):
        phonemizer_input = "@" + phonemizer_input
    for tag, sym in SOUND_MAPPING.items():
        phonemizer_input = phonemizer_input.replace(tag, sym)

    # 4. Phonemize
    phonemized = clean_spaces(collapse_ws(phonemizer.phonemize(phonemizer_input))).strip()

    # 5. Tokenize + emphasis
    tokens, emphasis = process_phonemized(phonemized)

    # 6. Intersperse blanks
    tokens = intersperse(tokens, BLANK_TOKEN_ID)
    emphasis = intersperse_emphasis(emphasis)

    return (
        np.array(tokens, dtype=np.int32),
        np.array(emphasis, dtype=np.int32),
        normalized,
        accentuated,
        phonemized,
    )


def main():
    parser = argparse.ArgumentParser(description="Test preprocessing pipeline locally")
    parser.add_argument("--text", type=str, default="Салам дүйнө, бул сыноо текст.", help="Text to process")
    args = parser.parse_args()

    print(f"Input:        {args.text}")
    tokens, emphasis, normalized, accentuated, phonemized = process_text(args.text)

    print(f"Normalized:   {normalized}")
    print(f"Accentuated:  {accentuated}")
    print(f"Phonemized:   {phonemized}")
    print(f"Tokens:       {tokens[:20]}... (len={len(tokens)})")
    print(f"Emphasis:     {emphasis[:20]}... (len={len(emphasis)})")
    print(f"Blank token:  {BLANK_TOKEN_ID}")


if __name__ == "__main__":
    main()
