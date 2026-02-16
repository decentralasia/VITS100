#!/bin/bash
# Start Triton Inference Server with the full VITS TTS ensemble pipeline:
#   vits-ky-6-preprocessing  (Python: normalize → accentuate → phonemize → tokenize)
#   vits-ky-6-acoustic-vocoder (TensorRT: tokens → audio)
#   vits-ky-6-ensemble       (chains the above two)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_REPO="$SCRIPT_DIR/triton_model_repo"
PLAN_SRC="$SCRIPT_DIR/model.plan"
PLAN_DST="$MODEL_REPO/vits-ky-6-acoustic-vocoder/1/model.plan"

# Copy model.plan if it exists
if [ -f "$PLAN_SRC" ]; then
    mkdir -p "$MODEL_REPO/vits-ky-6-acoustic-vocoder/1"
    cp "$PLAN_SRC" "$PLAN_DST"
    echo "Copied model.plan to vits-ky-6-acoustic-vocoder"
else
    echo "ERROR: model.plan not found. Run build_tensorrt.sh first."
    exit 1
fi

echo "Starting Triton Inference Server..."
echo "Model repository: $MODEL_REPO"
echo ""
echo "Models:"
echo "  vits-ky-6-preprocessing     (Python)    INPUT_TEXT → tokens, emphasis"
echo "  vits-ky-6-acoustic-vocoder  (TensorRT)  tokens, emphasis, sid, tid → audio"
echo "  vits-ky-6-ensemble          (Ensemble)  INPUT_TEXT, sid, tid → audio"

docker run --rm -it \
    --runtime=nvidia \
    --shm-size=1g \
    --ulimit memlock=-1 \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v "$MODEL_REPO:/models" \
    -v "$PROJECT_ROOT:/workspace" \
    kingoftheflies/aiph-triton-all \
    bash -c "pip install hfst && tritonserver --model-repository=/models --log-verbose=1"
