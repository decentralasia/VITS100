#!/usr/bin/env python3
"""
prepare_speaker_embeddings.py

Read a list file with lines like:
wav_path|speaker|... and compute speaker embedding for each wav using pyannote Inference,
convert the embedding to a PyTorch tensor and save it as a .pt file next to the wav.

Usage:
    python prepare_speaker_embeddings.py path/to/list.txt [--overwrite]

The list file must contain one record per line with fields separated by '|'. The wav path is taken
from the first field (index 0).
"""

from pathlib import Path
from typing import List
import argparse
import sys


def parse_args():
    # Hardcode input files: train and val manifests
    class Args:
        overwrite = False
        model = "pyannote/wespeaker-voxceleb-resnet34-LM"
        # listfile is unused when hardcoding; kept for compatibility
        listfile = None
    return Args()


def read_wav_paths(listfile: Path) -> List[Path]:
    # listfile parameter is ignored; hardcode two manifest files
    manifests = [Path("DUMMY1/train_filtered_manifest.txt"), Path("DUMMY1/val_filtered_manifest.txt")]
    paths: List[Path] = []
    for m in manifests:
        if not m.exists():
            print(f"Manifest not found: {m}", file=sys.stderr)
            continue
        with m.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                if not parts:
                    continue
                wav = parts[0].strip()
                if not wav:
                    continue
                paths.append(Path(wav))
    return paths


def main():
    args = parse_args()
    # listfile arg no longer used when hardcoding

    wav_paths = read_wav_paths(None)
    if not wav_paths:
        print("No wav paths found in the manifest files.", file=sys.stderr)
        sys.exit(0)

    try:
        from pyannote.audio import Model, Inference
    except Exception as e:
        print("pyannote.audio not available: " + str(e), file=sys.stderr)
        sys.exit(1)

    try:
        import numpy as np
        import torch
    except Exception as e:
        print("Missing numpy/torch: " + str(e), file=sys.stderr)
        sys.exit(1)

    # Load model and create inference instance
    model = Model.from_pretrained(args.model)
    inference = Inference(model, window="whole")

    # Progress bar
    try:
        from tqdm import tqdm
    except Exception:
        # Fallback to no progress bar if tqdm is not installed
        def tqdm(x, **kw):
            return x

    for wav in tqdm(wav_paths, desc="Embedding"):
        wav = wav.expanduser()
        if not wav.exists():
            print(f"Skipping missing file: {wav}", file=sys.stderr)
            continue

        out_path = wav.with_suffix('.pt')
        if out_path.exists() and not args.overwrite:
            # skip existing
            continue

        # Compute embedding
        try:
            emb = inference(str(wav))
        except TypeError:
            # Try the (uri, audio) dict form accepted by some pyannote Inference versions
            try:
                emb = inference({"uri": wav.stem, "audio": str(wav)})
            except Exception as e:
                print(f"Failed to compute embedding for {wav}: {e}", file=sys.stderr)
                continue
        except Exception as e:
            print(f"Failed to compute embedding for {wav}: {e}", file=sys.stderr)
            continue

        # Convert to torch tensor if needed
        try:
            if isinstance(emb, torch.Tensor):
                emb_t = emb.detach().cpu()
            else:
                emb_np = np.asarray(emb)
                if emb_np.dtype != np.float32:
                    emb_np = emb_np.astype(np.float32)
                emb_t = torch.from_numpy(emb_np)
        except Exception as e:
            print(f"Failed to convert embedding for {wav} to tensor: {e}", file=sys.stderr)
            continue

        # Save tensor
        try:
            torch.save(emb_t, out_path)
        except Exception as e:
            print(f"Failed to save embedding for {wav} to {out_path}: {e}", file=sys.stderr)
            continue

    print("Done.")


if __name__ == '__main__':
    main()
