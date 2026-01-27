#!/bin/bash
# Start Triton Inference Server with VITS TTS model

MODEL_REPO="$(cd "$(dirname "$0")" && pwd)/triton_model_repo"

# Check if model.plan exists and copy it
if [ -f "$(dirname "$0")/model.plan" ]; then
    mkdir -p "$MODEL_REPO/vits_tts/1"
    cp "$(dirname "$0")/model.plan" "$MODEL_REPO/vits_tts/1/model.plan"
    echo "Copied model.plan to Triton model repository"
else
    echo "ERROR: model.plan not found. Please run trtexec first."
    exit 1
fi

echo "Starting Triton Inference Server..."
echo "Model repository: $MODEL_REPO"

# Run Triton server with Docker
# Use --runtime=nvidia instead of --gpus=all for better compatibility
docker run --rm -it \
    --runtime=nvidia \
    --shm-size=1g \
    --ulimit memlock=-1 \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v "$MODEL_REPO:/models" \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models --log-verbose=1
