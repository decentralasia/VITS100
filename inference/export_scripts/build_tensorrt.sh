#!/bin/bash
# Convert ONNX models to TensorRT .plan files for Triton.
#
# Prerequisites:
#   - ONNX models already exported to triton_repo/vits-{ky,ru}-synthesis/1/model.onnx
#   - Docker with NVIDIA runtime
#
# Usage:
#   cd inference/export_scripts
#   bash build_tensorrt.sh           # build both
#   bash build_tensorrt.sh ky        # Kyrgyz only
#   bash build_tensorrt.sh ru        # Russian only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

KY_MODEL_DIR="$INFERENCE_DIR/triton_repo/vits-ky-synthesis/1"
RU_MODEL_DIR="$INFERENCE_DIR/triton_repo/vits-ru-synthesis/1"

# TensorRT 10.9 container — must match the Triton server version
TRT_IMAGE="nvcr.io/nvidia/tensorrt:25.04-py3"

build_plan() {
    local MODEL_DIR="$1"
    local LABEL="$2"

    if [ ! -f "$MODEL_DIR/model.onnx" ]; then
        echo "❌ ONNX not found: $MODEL_DIR/model.onnx"
        return 1
    fi

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "Building TensorRT plan: $LABEL"
    echo "  ONNX:   $MODEL_DIR/model.onnx"
    echo "  Output: $MODEL_DIR/model.plan"
    echo "════════════════════════════════════════════════════"

    # Dynamic shape profiles for baked ONNX inputs:
    #   input:    [1, T]  INT32  phoneme token IDs    (T is dynamic)
    #   emphasis: [1, T]  INT32  emphasis/highlight    (T is dynamic)
    #   sid:      [1]     INT32  speaker ID            (scalar, batch=1)
    #   tid:      [1]     INT32  tone ID               (scalar, batch=1)
    #   lid:      [1]     INT32  language ID            (scalar, batch=1)

    docker run --rm \
        --privileged \
        --runtime=nvidia \
        -v "$MODEL_DIR:/workspace" \
        "$TRT_IMAGE" \
        bash -c "
            cd /workspace && \
            trtexec --onnx=model.onnx \
                --saveEngine=model.plan \
                --minShapes=input:1x1,emphasis:1x1,sid:1,tid:1,lid:1 \
                --optShapes=input:1x250,emphasis:1x250,sid:1,tid:1,lid:1 \
                --maxShapes=input:1x500,emphasis:1x500,sid:1,tid:1,lid:1 \
                --fp16 \
                --verbose
        "

    if [ $? -eq 0 ] && [ -f "$MODEL_DIR/model.plan" ]; then
        local SIZE
        SIZE=$(du -h "$MODEL_DIR/model.plan" | cut -f1)
        echo "✅ $LABEL: model.plan ($SIZE)"
    else
        echo "❌ $LABEL: model.plan was NOT created"
        return 1
    fi
}

# Parse argument
TARGET="${1:-all}"

case "$TARGET" in
    ky)
        build_plan "$KY_MODEL_DIR" "Kyrgyz"
        ;;
    ru)
        build_plan "$RU_MODEL_DIR" "Russian"
        ;;
    all)
        build_plan "$KY_MODEL_DIR" "Kyrgyz"
        build_plan "$RU_MODEL_DIR" "Russian"
        ;;
    *)
        echo "Usage: $0 [ky|ru|all]"
        exit 1
        ;;
esac

echo ""
echo "Done. Place model.plan files in triton_repo/vits-{ky,ru}-synthesis/1/"
echo "Then start Triton: bash ../start_triton_server.sh"