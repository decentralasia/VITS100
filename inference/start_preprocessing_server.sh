#!/bin/bash
# Start Triton with ONLY the preprocessing model for testing.
# Uses --exit-on-error=false so partial failures don't kill the server.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_REPO="$SCRIPT_DIR/triton_model_repo"

echo "Starting Triton (preprocessing only)..."
echo "Model repository: $MODEL_REPO"

docker run --rm -it \
    --shm-size=1g \
    --ulimit memlock=-1 \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v "$MODEL_REPO:/models" \
    -v "$PROJECT_ROOT:/workspace" \
    kingoftheflies/aiph-triton-all \
    bash -c "pip install hfst && tritonserver \
        --model-repository=/models \
        --load-model=vits-ky-6-preprocessing \
        --model-control-mode=explicit \
        --log-verbose=1 \
        --exit-on-error=false"
