#!/usr/bin/env python3
"""
Triton client to test vits-ky-6-preprocessing model only.

Usage:
    python triton_preprocessing_client.py --text "Салам дүйнө"
    python triton_preprocessing_client.py --text "Бишкек шаары" --triton-url localhost:8001
"""

import argparse

import numpy as np
import tritonclient.grpc as grpcclient


def run_preprocessing(triton_url, text, model_name="vits-ky-6-preprocessing"):
    client = grpcclient.InferenceServerClient(url=triton_url)

    # Check model is ready
    if not client.is_model_ready(model_name):
        print(f"ERROR: Model '{model_name}' is not ready")
        return

    text_np = np.array([text.encode("utf-8")], dtype=object).reshape(1)

    inputs = [
        grpcclient.InferInput("INPUT_TEXT", [1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(text_np)

    outputs = [
        grpcclient.InferRequestedOutput("input"),
        grpcclient.InferRequestedOutput("emphasis"),
        grpcclient.InferRequestedOutput("normalized_text"),
        grpcclient.InferRequestedOutput("accentuated_text"),
        grpcclient.InferRequestedOutput("phonemized_text"),
    ]

    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    tokens = response.as_numpy("input")
    emphasis = response.as_numpy("emphasis")
    normalized = response.as_numpy("normalized_text")[0].decode("utf-8")
    accentuated = response.as_numpy("accentuated_text")[0].decode("utf-8")
    phonemized = response.as_numpy("phonemized_text")[0].decode("utf-8")

    print(f"Normalized:   {normalized}")
    print(f"Accentuated:  {accentuated}")
    print(f"Phonemized:   {phonemized}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Tokens:       {tokens[0][:20]}...")
    print(f"Emphasis:     {emphasis[0][:20]}...")


def main():
    parser = argparse.ArgumentParser(description="Test vits-ky-6-preprocessing via Triton")
    parser.add_argument("--text", type=str, default="Салам дүйнө, бул сыноо текст.", help="Text to process")
    parser.add_argument("--triton-url", type=str, default="localhost:8001", help="Triton gRPC URL")
    parser.add_argument("--model-name", type=str, default="vits-ky-6-preprocessing", help="Model name")
    args = parser.parse_args()

    print(f"Input: {args.text}")
    print(f"Triton: {args.triton_url}")
    print()
    run_preprocessing(args.triton_url, args.text, args.model_name)


if __name__ == "__main__":
    main()
