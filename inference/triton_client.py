#!/usr/bin/env python3
"""
Triton Inference Client for VITS TTS Ensemble.

Sends raw text to the ensemble which handles normalization, accentuation,
phonemization, and synthesis end-to-end.

Usage:
    python triton_client.py --text "Салам дүйнө" --output output.wav
    python triton_client.py --text "Бишкек шаары" --speaker 1 --tone 2 --output output.wav

Speaker IDs: 0=Timur, 1=Aiganysh
Tone IDs:    0=neutral, 1=strict, 2=friendly
"""

import argparse

import numpy as np
import tritonclient.grpc as grpcclient
from scipy.io import wavfile

SAMPLE_RATE = 22050
HOP_LENGTH = 256


def run_inference(triton_url, text, speaker_id, tone_id, model_name="vits-ky-6-ensemble"):
    """Send raw text to Triton ensemble and get audio back."""
    client = grpcclient.InferenceServerClient(url=triton_url)

    text_np = np.array([text.encode("utf-8")], dtype=object).reshape(1)
    sid = np.array([speaker_id], dtype=np.int32)
    tid = np.array([tone_id], dtype=np.int32)

    inputs = [
        grpcclient.InferInput("INPUT_TEXT", [1], "BYTES"),
        grpcclient.InferInput("sid", [1], "INT32"),
        grpcclient.InferInput("tid", [1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(text_np)
    inputs[1].set_data_from_numpy(sid)
    inputs[2].set_data_from_numpy(tid)

    outputs = [
        grpcclient.InferRequestedOutput("raw_waveform"),
        grpcclient.InferRequestedOutput("y_length"),
        grpcclient.InferRequestedOutput("normalized_text"),
        grpcclient.InferRequestedOutput("accentuated_text"),
        grpcclient.InferRequestedOutput("phonemized_text"),
    ]

    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    audio = response.as_numpy("raw_waveform")
    y_length = response.as_numpy("y_length")

    # Print preprocessing stages
    normalized = response.as_numpy("normalized_text")[0].decode("utf-8")
    accentuated = response.as_numpy("accentuated_text")[0].decode("utf-8")
    phonemized = response.as_numpy("phonemized_text")[0].decode("utf-8")
    print(f"Normalized:  {normalized}")
    print(f"Accentuated: {accentuated}")
    print(f"Phonemized:  {phonemized}")

    actual_length = int(y_length[0]) * HOP_LENGTH
    audio = audio.squeeze()[:actual_length]

    return audio


def save_wav(audio, output_path, sample_rate=SAMPLE_RATE):
    """Save audio as WAV file."""
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, audio_int16)
    print(f"Saved audio to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VITS TTS Triton Ensemble Client")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize (Kyrgyz)")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file path")
    parser.add_argument("--triton-url", type=str, default="localhost:8001", help="Triton gRPC URL")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID (0=Timur, 1=Aiganysh)")
    parser.add_argument("--tone", type=int, default=0, help="Tone ID (0=neutral, 1=strict, 2=friendly)")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help="Output sample rate")
    parser.add_argument("--model-name", type=str, default="vits-ky-6-ensemble", help="Triton model name")

    args = parser.parse_args()

    print(f"Text:     {args.text}")
    print(f"Speaker:  {args.speaker} ({'Timur' if args.speaker == 0 else 'Aiganysh'})")
    print(f"Tone:     {args.tone} ({['neutral', 'strict', 'friendly'][args.tone]})")

    print(f"Connecting to Triton at {args.triton_url}...")
    audio = run_inference(
        triton_url=args.triton_url,
        text=args.text,
        speaker_id=args.speaker,
        tone_id=args.tone,
        model_name=args.model_name,
    )

    print(f"Generated {len(audio)} samples ({len(audio)/args.sample_rate:.2f}s)")
    save_wav(audio, args.output, args.sample_rate)


if __name__ == "__main__":
    main()
