#!/bin/bash
# Start Triton Inference Server with all 4 VITS2 models.
# Mounts: model repo, packages, preprocessing artifacts, synthesis artifacts, voices.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_REPO="$SCRIPT_DIR/triton_repo"

echo "Starting Triton Inference Server..."
echo "Model repository: $MODEL_REPO"

docker run --rm -d \
    --runtime=nvidia \
    --shm-size=1g \
    --ulimit memlock=-1 \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v "$MODEL_REPO:/models" \
    -v "$SCRIPT_DIR/packages/tts-text-processing-kg:/packages/tts-text-processing-kg" \
    -v "$SCRIPT_DIR/packages/tts-text-processing-ru:/packages/tts-text-processing-ru" \
    -v "$SCRIPT_DIR/packages/vits-preprocessing:/packages/vits-preprocessing" \
    -v "$SCRIPT_DIR/preprocessing_artifacts/kg:/artifacts/ky" \
    -v "$SCRIPT_DIR/preprocessing_artifacts/ru:/artifacts/ru" \
    -v "$SCRIPT_DIR/synthesis_artifacts/kg:/artifacts/ky_synth" \
    -v "$SCRIPT_DIR/synthesis_artifacts/ru:/artifacts/ru_synth" \
    -v "$SCRIPT_DIR/voices:/voices" \
    -e PACKAGES_DIR=/packages \
    -e ARTIFACTS_KY=/artifacts/ky \
    -e ARTIFACTS_RU=/artifacts/ru \
    -e SYNTH_KY=/artifacts/ky_synth \
    -e SYNTH_RU=/artifacts/ru_synth \
    -e VOICES_DIR=/voices \
    kingoftheflies/aiph-triton-all \
    bash -c "pip install hfst ctranslate2 ruaccent && tritonserver --model-repository=/models --log-verbose=1"
