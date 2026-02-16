#!/bin/bash
# Build TensorRT plan inside the TensorRT container to ensure version compatibility

MODEL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Building TensorRT plan inside TensorRT container..."
echo "This ensures TensorRT version compatibility with Triton 25.04."

# Use TensorRT 10.9.0.34-1+cuda12.8 container which matches Triton 25.04
docker run --rm \
    --runtime=nvidia \
    -v "$MODEL_DIR:/workspace" \
    nvcr.io/nvidia/tensorrt:25.04-py3 \
    bash -c "
        cd /workspace && \
        trtexec --onnx=model.onnx \
            --saveEngine=model.plan \
            --minShapes=input:1x1,emphasis:1x1,sid:1,tid:1 \
            --optShapes=input:1x250,emphasis:1x250,sid:1,tid:1 \
            --maxShapes=input:1x500,emphasis:1x500,sid:1,tid:1 \
            --fp16 \
            --verbose
    "

if [ $? -eq 0 ]; then
    echo ""
    echo "SUCCESS: model.plan created successfully!"
    echo "Now run: ./start_triton_server.sh"
else
    echo ""
    echo "FAILED: TensorRT conversion failed."
    exit 1
fi
