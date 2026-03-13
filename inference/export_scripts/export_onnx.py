#!/usr/bin/env python3

import argparse
import json
import os
import sys
import warnings

import torch
from torch.nn import functional as F
from torch.onnx import TrainingMode

# Add project root for imports (models, utils, text)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore", message="Exporting a model to ONNX with a batch_size other than 1")
warnings.filterwarnings("ignore", message="Constant folding - Only steps=1 can be constant folded")
warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean")
warnings.filterwarnings("ignore", message="Converting a tensor to a Python integer")
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

import utils
from models import SynthesizerTrn
from text.symbols import symbols


# ════════════════════════════════════════════════════════════════════════
# VOICE REGISTRY
#
# Each entry: (display_name, mel_pt_filename, sid, tid, lid)
#
# mel_pt_filename is relative to --voices-dir.
# Set to None or "PLACEHOLDER" for voices without a reference yet —
# the script will use a silent dummy mel for those slots.
#
# Index into baked ref_specs:
#   idx = lid * n_speakers * n_tones + sid * n_tones + tid
# ════════════════════════════════════════════════════════════════════════
WAV_DIR = "../voices/"

VOICE_REGISTRY = [
    # ── Kyrgyz voices (lid=0) ──────────────────────────────────────────
    # sid=0 (Timur)
    ("Timur-neutral-kg",    "00003_003_inter_sounds_neutral_050_1_Timur_neutral_kg.mel.pt",  0, 0, 0),
    ("Timur-strict-kg",     "00000_000_news_quest_1-1.0_010_1_Timur_strict_kg.mel.pt",       0, 1, 0),
    ("Timur-friendly-kg",   "00133_002_inter_sounds_neutral_069_14_Timur_friendly_kg.mel.pt", 0, 2, 0),

    # sid=1 (Aiganysh)
    ("Aiganysh-neutral-kg", "00004_004_support_000-2_1_Aiganysh_neutral_kg.mel.pt",  1, 0, 0),
    ("Aiganysh-strict-kg",  "00001_001_quest_tv_show_004_1_Aiganysh_strict_kg.mel.pt",  1, 1, 0),
    ("Aiganysh-friendly-kg", "00000_000_sales_010_1_Aiganysh_friendly_kg.mel.pt",  1, 2, 0),

    # ── Russian voices (lid=1) ─────────────────────────────────────────
    # sid=0 (Timur)
    ("Timur-neutral-ru",    "00003_003_ru_support_038_1_Timur_neutral_ru.mel.pt",  0, 0, 1),
    ("Timur-strict-ru",     "00002_002_ru_sales_004_1_Timur_strict_ru.mel.pt",  0, 1, 1),
    ("Timur-friendly-ru",   "Timur_friendly_ru.mel.pt",  0, 2, 1),

    # sid=1 (Aiganysh)
    ("Aiganysh-neutral-ru", "00003_003_ru_support_012_1_Aiganysh_neutral_ru.mel.pt",  1, 0, 1),
    ("Aiganysh-strict-ru",  "00006_006_ru_support_050_1_Aiganysh_strict_ru.mel.pt",  1, 1, 1),
    ("Aiganysh-friendly-ru", "00006_006_ru_support_050_1_Aiganysh_strict_ru.mel.pt",  1, 2, 1),

    # ── ElevenLabs cloned voices (lid=1, tid=3) ───────────────────────
    ("alexander_vlasov",    "alexander_vlasov.mel.pt",  2, 3, 1),
    ("artem_lebedev",       "artem_lebedev.mel.pt",     3, 3, 1),
    ("kari",                "kari.mel.pt",              4, 3, 1),
    ("larisa_actrisa",      "larisa_actrisa.mel.pt",    5, 3, 1),
    ("nikolay",             "nikolay.mel.pt",           6, 3, 1),
    ("rina",                "rina.mel.pt",              7, 3, 1),
    ("victoria",            "victoria.mel.pt",          8, 3, 1),
]


def load_reference(filepath, voices_dir):
    """Load a .mel.pt reference. Returns [80, T] tensor."""
    if filepath is None or filepath == "PLACEHOLDER":
        return None

    full_path = os.path.join(voices_dir, filepath)
    if not os.path.isfile(full_path):
        print(f"  ⚠️  {filepath} not found in {voices_dir}, using dummy")
        return None

    spec = torch.load(full_path, weights_only=True)
    # Normalize shape: [1, 80, T] → [80, T] or [80, T] stays as is
    if spec.dim() == 3:
        spec = spec.squeeze(0)
    return spec


def build_ref_specs(n_speakers, n_tones, n_languages, voices_dir):
    """
    Build padded reference spec tensor [N, 80, max_T] and lengths [N].
    N = n_languages * n_speakers * n_tones
    """
    total_slots = n_languages * n_speakers * n_tones
    specs = [None] * total_slots
    names = ["(empty)"] * total_slots

    # Filter and place registry entries that fit this model's dimensions
    placed = 0
    skipped = 0
    for display_name, mel_file, sid, tid, lid in VOICE_REGISTRY:
        if sid >= n_speakers or tid >= n_tones or lid >= n_languages:
            skipped += 1
            continue

        idx = lid * n_speakers * n_tones + sid * n_tones + tid
        ref = load_reference(mel_file, voices_dir)
        if ref is not None:
            specs[idx] = ref
            names[idx] = f"{display_name} ← {mel_file}"
            placed += 1
        else:
            names[idx] = f"{display_name} ← PLACEHOLDER (dummy)"

    # Fill empty slots with a short silent dummy mel
    dummy = torch.zeros(80, 50)
    for i in range(total_slots):
        if specs[i] is None:
            specs[i] = dummy

    # Pad all to same length and stack
    lengths = torch.tensor([s.shape[-1] for s in specs], dtype=torch.long)
    max_len = max(s.shape[-1] for s in specs)
    padded = torch.stack([F.pad(s, (0, max_len - s.shape[-1])) for s in specs])

    # Print summary
    print(f"\nVoice reference table ({total_slots} slots, {placed} with real refs, {skipped} skipped):")
    print(f"  n_speakers={n_speakers}, n_tones={n_tones}, n_languages={n_languages}")
    print(f"  Max mel length: {max_len} frames")
    print()
    for i, name in enumerate(names):
        lid_i = i // (n_speakers * n_tones)
        remainder = i % (n_speakers * n_tones)
        sid_i = remainder // n_tones
        tid_i = remainder % n_tones
        print(f"  [{i:3d}] sid={sid_i} tid={tid_i} lid={lid_i}  {name}")
    print()

    return padded, lengths


def export(config, checkpoint, voices_dir, output):
    print(f"Config:     {config}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Voices dir: {voices_dir}")
    print(f"Output:     {output}")

    # Load config
    hps = utils.get_hparams_from_file(config)

    n_speakers = hps.data.n_speakers
    n_tones = hps.data.n_tones
    n_languages = hps.data.n_languages
    posterior_channels = 80

    # Build model
    net_g = SynthesizerTrn(
        len(symbols),
        spec_channels=posterior_channels,
        segment_size=hps.train.segment_size // hps.data.hop_length,
        is_onnx=True,
        **hps.model,
    )

    # Load weights
    if os.path.isfile(checkpoint):
        print(f"Loading checkpoint: {checkpoint}")
        _ = utils.load_checkpoint(checkpoint, net_g, None)
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint}")
        print("   Exporting with random weights (for testing structure only)")

    num_symbols = net_g.n_vocab

    # Build and bake reference specs
    ref_specs, ref_lengths = build_ref_specs(
        n_speakers, n_tones, n_languages, voices_dir,
    )
    net_g.register_buffer("ref_specs", ref_specs)
    net_g.register_buffer("ref_spec_lengths", ref_lengths)

    N_SPEAKERS = n_speakers
    N_TONES = n_tones

    # 20 seconds of audio at 22050 Hz with hop_length=256 → 1724 mel frames
    # Fixed size ensures TensorRT has deterministic intermediate/output shapes
    MAX_MEL_LENGTH = int(20 * hps.data.sampling_rate / hps.data.hop_length)
    HOP_LENGTH = hps.data.hop_length

    # ONNX-compatible forward wrapper
    def infer_forward(text, emphasis, sid, tid, lid,
                      input_ids_length, duration_perc, duration_force_ms, scales):
        """
        Inputs:
            text:              [B, T]  INT64  phoneme token IDs
            emphasis:          [B, T]  INT64  emphasis/highlight mask
            sid:               [B]     INT64  speaker ID
            tid:               [B]     INT64  tone ID
            lid:               [B]     INT64  language ID
            input_ids_length:  [B]     INT64  actual token count per sample
            duration_perc:     [B, T]  FP32   (unused, ensemble compatibility)
            duration_force_ms: [B, T]  INT64  (unused, ensemble compatibility)
            scales:            [B, 3]  FP32   (unused, ensemble compatibility)

        Outputs:
            raw_waveform: [B, 1, T_audio]  FP32  (fixed size = MAX_MEL_LENGTH * hop_length)
            y_length:     [B]              INT64  (actual valid length in mel frames)
        """
        idx = lid.long() * N_SPEAKERS * N_TONES + sid.long() * N_TONES + tid.long()
        spec = net_g.ref_specs[idx]
        spec_len = net_g.ref_spec_lengths[idx]

        result = net_g.infer(
            text,
            spec,
            emphasis=emphasis,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0,
            spec_lengths=spec_len,
            max_y_length=MAX_MEL_LENGTH,
        )
        audio = result[0]                      # [B, 1, T] — keep channel dim to match OLD vocoder output
        y_lengths = result[5].to(torch.int64)  # mel frame count (NOT audio samples)

        # Keep unused inputs in the ONNX graph so TensorRT doesn't prune them.
        # These are required by the Triton ensemble wiring (OLD reference compat).
        _unused = (input_ids_length.sum()
                   + duration_perc.sum()
                   + duration_force_ms.float().sum()
                   + scales.sum()) * 0
        audio = audio + _unused

        return audio, y_lengths

    # Prepare for export
    with torch.no_grad():
        net_g.dec.remove_weight_norm()
        net_g.flow.remove_weight_norm()
        net_g.forward = infer_forward

    net_g.eval()

    # Dummy inputs for ONNX tracing
    B = 1
    T = 50
    dummy_inputs = (
        torch.randint(0, num_symbols, (B, T), dtype=torch.int64),  # text
        torch.zeros(B, T, dtype=torch.int64),                       # emphasis
        torch.zeros(B, dtype=torch.int64),                           # sid
        torch.zeros(B, dtype=torch.int64),                           # tid
        torch.zeros(B, dtype=torch.int64),                           # lid
        torch.tensor([T], dtype=torch.int64),                        # input_ids_length
        torch.zeros(B, T, dtype=torch.float32),                      # duration_perc
        torch.zeros(B, T, dtype=torch.int64),                        # duration_force_ms
        torch.zeros(B, 3, dtype=torch.float32),                      # scales
    )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    torch.onnx.export(
        model=net_g,
        args=dummy_inputs,
        f=output,
        verbose=False,
        opset_version=17,
        training=TrainingMode.EVAL,
        input_names=["input", "emphasis", "sid", "tid", "lid",
                     "input_ids_length", "duration_perc", "duration_force_ms", "scales"],
        output_names=["raw_waveform", "y_length"],
        dynamic_axes={
            "input":            {0: "batch_size", 1: "phonemes"},
            "emphasis":         {0: "batch_size", 1: "phonemes"},
            "sid":              {0: "batch_size"},
            "tid":              {0: "batch_size"},
            "lid":              {0: "batch_size"},
            "input_ids_length": {0: "batch_size"},
            "duration_perc":    {0: "batch_size", 1: "phonemes"},
            "duration_force_ms":{0: "batch_size", 1: "phonemes"},
            "scales":           {0: "batch_size"},
            "raw_waveform":     {0: "batch_size"},
            "y_length":         {0: "batch_size"},
        },
    )

    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"✅ Exported: {output} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Export VITS2 model to ONNX with baked voice references",
    )
    parser.add_argument("--lang", required=True, help="language of the model")

    args = parser.parse_args()

    if args.lang == "ky":
        config = "/home/fudo/kenenbek/m/kg/config.json"
        checkpoint = "/home/fudo/kenenbek/m/kg/G_69000.pth"
        output = "./model_ky.onnx"
    elif args.lang == "ru":
        config = "/home/fudo/kenenbek/m/ru/config.json"
        checkpoint = "/home/fudo/kenenbek/m/ru/G_89000.pth"
        output = "./model_ru.onnx"
    else:
        raise NotImplementedError

    export(config, checkpoint, WAV_DIR, output)


if __name__ == "__main__":
    main()
