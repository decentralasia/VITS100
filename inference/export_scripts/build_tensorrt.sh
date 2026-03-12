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
    #   input:            [B, T]  INT64  phoneme token IDs    (B, T are dynamic)
    #   emphasis:         [B, T]  INT64  emphasis/highlight    (B, T are dynamic)
    #   sid:              [B]     INT64  speaker ID            (B is dynamic)
    #   tid:              [B]     INT64  tone ID               (B is dynamic)
    #   lid:              [B]     INT64  language ID            (B is dynamic)
    #   input_ids_length: [B]     INT64  actual token count    (B is dynamic)
    #   duration_perc:    [B, T]  FP32   duration pct          (unused, ensemble compat)
    #   duration_force_ms:[B, T]  INT64  forced durations      (unused, ensemble compat)
    #   scales:           [B, 3]  FP32   inference scales      (unused, ensemble compat)

    docker run --rm \
        --privileged \
        --runtime=nvidia \
        -v "$SCRIPT_DIR:/workspace" \
        "$TRT_IMAGE" \
        bash -c "
            cd /workspace && \
            trtexec --onnx=$ONNX_FILE \
                --saveEngine=$PLAN_FILE \
                --minShapes=input:1x1,emphasis:1x1,sid:1,tid:1,lid:1,input_ids_length:1,duration_perc:1x1,duration_force_ms:1x1,scales:1x3 \
                --optShapes=input:16x250,emphasis:16x250,sid:16,tid:16,lid:16,input_ids_length:16,duration_perc:16x250,duration_force_ms:16x250,scales:16x3 \
                --maxShapes=input:32x500,emphasis:32x500,sid:32,tid:32,lid:32,input_ids_length:32,duration_perc:32x500,duration_force_ms:32x500,scales:32x3 \
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
