#!/bin/bash
# Convert ONNX models to TensorRT .plan files for Triton.
#
# Prerequisites:
#   - ONNX models model_ky.onnx and model_ru.onnx in the current directory
#   - Docker with NVIDIA runtime
#
# Usage:
#   cd inference/export_scripts
#   bash build_tensorrt.sh           # build both
#   bash build_tensorrt.sh ky        # Kyrgyz only
#   bash build_tensorrt.sh ru        # Russian only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# TensorRT 10.9 container — must match the Triton server version
TRT_IMAGE="nvcr.io/nvidia/tensorrt:25.04-py3"

build_plan() {
    local ONNX_FILE="$1"
    local PLAN_FILE="$2"
    local LABEL="$3"

    if [ ! -f "$SCRIPT_DIR/$ONNX_FILE" ]; then
        echo "❌ ONNX not found: $SCRIPT_DIR/$ONNX_FILE"
        return 1
    fi

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "Building TensorRT plan: $LABEL"
    echo "  ONNX:   $SCRIPT_DIR/$ONNX_FILE"
    echo "  Output: $SCRIPT_DIR/$PLAN_FILE"
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
        -v "$SCRIPT_DIR:/workspace" \
        "$TRT_IMAGE" \
        bash -c "
            cd /workspace && \
            trtexec --onnx=$ONNX_FILE \
                --saveEngine=$PLAN_FILE \
                --minShapes=input:1x1,emphasis:1x1,sid:1,tid:1,lid:1 \
                --optShapes=input:1x250,emphasis:1x250,sid:1,tid:1,lid:1 \
                --maxShapes=input:1x500,emphasis:1x500,sid:1,tid:1,lid:1 \
                --fp16 \
                --verbose
        "

    if [ $? -eq 0 ] && [ -f "$SCRIPT_DIR/$PLAN_FILE" ]; then
        local SIZE
        SIZE=$(du -h "$SCRIPT_DIR/$PLAN_FILE" | cut -f1)
        echo "✅ $LABEL: $PLAN_FILE ($SIZE)"
    else
        echo "❌ $LABEL: $PLAN_FILE was NOT created"
        return 1
    fi
}

# Parse argument
TARGET="${1:-all}"

case "$TARGET" in
    ky)
        build_plan "model_ky.onnx" "model_ky.plan" "Kyrgyz"
        ;;
    ru)
        build_plan "model_ru.onnx" "model_ru.plan" "Russian"
        ;;
    all)
        build_plan "model_ky.onnx" "model_ky.plan" "Kyrgyz"
        build_plan "model_ru.onnx" "model_ru.plan" "Russian"
        ;;
    *)
        echo "Usage: $0 [ky|ru|all]"
        exit 1
        ;;
esac

echo ""
echo "Done. You can now move model_ky.plan and model_ru.plan to their respective triton_repo directories."
