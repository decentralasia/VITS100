#!/usr/bin/env python3
"""
Triton Inference Client for VITS TTS Model

Usage:
    python triton_client.py --text "Салам дүйнө" --lang ky --output output.wav
    python triton_client.py --text "Привет мир" --lang ru --speaker 0 --output output.wav
"""

import argparse
import numpy as np
import tritonclient.grpc as grpcclient
from scipy.io import wavfile
import sys
sys.path.insert(0, '.')

from text import text_to_sequence


def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result


def get_text(text, lid):
    lang_code = "ky" if lid == 0 else "ru"
    text_norm = text_to_sequence(text, cleaner_names=[], lang_code=lang_code)
    text_norm = intersperse(text_norm, 0)
    return np.array(text_norm, dtype=np.int32)


# def text_to_sequence(text: str, language: str = 'ky') -> np.ndarray:
#     """Convert text to phoneme sequence with language prefix."""
#     symbol_to_id = {s: i for i, s in enumerate(symbols)}
#
#     # Punctuation (no language prefix)
#     punctuation = '!? ^,;.-'
#
#     sequence = []
#     lang_prefix = f'<{language}>'
#
#     for char in text:
#         if char in punctuation:
#             # Punctuation without prefix
#             if char in symbol_to_id:
#                 sequence.append(symbol_to_id[char])
#         else:
#             # Letters with language prefix
#             prefixed_char = lang_prefix + char
#             if prefixed_char in symbol_to_id:
#                 sequence.append(symbol_to_id[prefixed_char])
#             elif char in symbol_to_id:
#                 # Fallback to unprefixed
#                 sequence.append(symbol_to_id[char])
#
#     return np.array(sequence, dtype=np.int32)


def load_reference_mel(audio_path: str) -> np.ndarray:
    """Load and convert reference audio to mel spectrogram."""
    ref = np.load(audio_path)
    return ref


def create_dummy_reference_mel(length: int = 200, n_mels: int = 80) -> np.ndarray:
    """Create dummy reference mel spectrogram for testing."""
    return np.random.randn(n_mels, length).astype(np.float32) * 0.1


def run_inference(
    triton_url: str,
    text_sequence: np.ndarray,
    reference_mel: np.ndarray,
    speaker_id: int = 0,
    tone_id: int = 0,
    language_id: int = 0,
    model_name: str = "vits_tts"
) -> np.ndarray:
    """Run TTS inference on Triton server."""
    
    client = grpcclient.InferenceServerClient(url=triton_url)
    
    # Prepare inputs with batch dimension
    input_ids = text_sequence.reshape(1, -1).astype(np.int32)  # [1, seq_len]
    spec_ref = reference_mel.reshape(1, reference_mel.shape[0], -1)  # [1, 80, spec_len]
    sid = np.array([speaker_id], dtype=np.int32)  # [1]
    tid = np.array([tone_id], dtype=np.int32)  # [1]
    lid = np.array([language_id], dtype=np.int32)  # [1]
    
    # Create Triton inputs
    inputs = [
        grpcclient.InferInput("input", input_ids.shape, "INT32"),
        grpcclient.InferInput("spec_ref", spec_ref.shape, "FP32"),
        grpcclient.InferInput("sid", sid.shape, "INT32"),
        grpcclient.InferInput("tid", tid.shape, "INT32"),
        grpcclient.InferInput("lid", lid.shape, "INT32"),
    ]
    
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(spec_ref)
    inputs[2].set_data_from_numpy(sid)
    inputs[3].set_data_from_numpy(tid)
    inputs[4].set_data_from_numpy(lid)
    
    # Create output placeholders
    outputs = [
        grpcclient.InferRequestedOutput("raw_waveform"),
        grpcclient.InferRequestedOutput("y_length"),
    ]
    
    # Run inference
    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    
    # Get audio output and length
    audio = response.as_numpy("raw_waveform")
    y_length = response.as_numpy("y_length")
    
    # Trim audio to actual length to remove metallic noise at the end
    # y_length is in frames, audio_samples = y_length * hop_length (256)
    hop_length = 256
    actual_length = int(y_length[0]) * hop_length
    audio = audio.squeeze()[:actual_length]  # Remove batch dim and trim
    
    return audio


def save_wav(audio: np.ndarray, output_path: str, sample_rate: int = 22050):
    """Save audio as WAV file."""
    # Normalize to int16 range
    audio = audio / np.abs(audio).max() * 0.95
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, audio_int16)
    print(f"Saved audio to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VITS TTS Triton Client")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file path")
    parser.add_argument("--triton-url", type=str, default="localhost:8001", help="Triton server URL")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID")
    parser.add_argument("--tone", type=int, default=0, help="Tone ID")
    parser.add_argument("--language", type=int, default=1, help="Language ID (embedding index)")
    parser.add_argument("--lang", type=str, default="ru", choices=["ky", "ru"], help="Text language for tokenization")
    parser.add_argument("--reference-audio", type=str, default="/mnt/d/timur_ref.npy", help="Reference audio for style")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Output sample rate")
    parser.add_argument("--model-name", type=str, default="vits_tts", help="Triton model name")
    
    args = parser.parse_args()
    
    # Convert text to sequence
    text_sequence = get_text(args.text, args.lang)
    if len(text_sequence) == 0:
        print("Error: No valid characters in input text")
        return
    
    print(f"Input text: {args.text}")
    print(f"Language: {args.lang}")
    print(f"Sequence length: {len(text_sequence)}")
    
    # Load or create reference mel
    if args.reference_audio:
        reference_mel = load_reference_mel(args.reference_audio)
        print(f"Loaded reference mel from: {args.reference_audio}")
    else:
        reference_mel = create_dummy_reference_mel()
        print("Using dummy reference mel spectrogram")
    
    print(f"Reference mel shape: {reference_mel.shape}")
    
    # Run inference
    print(f"Connecting to Triton server at {args.triton_url}...")
    audio = run_inference(
        triton_url=args.triton_url,
        text_sequence=text_sequence,
        reference_mel=reference_mel,
        speaker_id=args.speaker,
        tone_id=args.tone,
        language_id=args.language,
        model_name=args.model_name
    )
    
    print(f"Generated audio shape: {audio.shape}")
    
    # Save output
    save_wav(audio, args.output, args.sample_rate)


if __name__ == "__main__":
    main()
