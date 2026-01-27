# VITS TTS with Triton Inference Server

## Quick Start

```bash
# 1. Export ONNX model (requires PyTorch environment)
python onnx_export.py

# 2. Build TensorRT plan (inside Docker for version compatibility)
./build_tensorrt.sh

# 3. Start Triton server
./start_triton_server.sh

# 4. In another terminal, run inference
pip install tritonclient[grpc] scipy numpy
python triton_client.py --text "Салам дүйнө" --lang ky --output output.wav
```

## Detailed Setup Instructions

### 1. Export ONNX Model
```bash
python onnx_export.py
```
This creates `model.onnx` from your trained PyTorch model.

### 2. Build TensorRT Plan

**Important:** The TensorRT plan must be built with the same TensorRT version as the Triton server. Use the provided script which builds inside Docker:

```bash
chmod +x build_tensorrt.sh
./build_tensorrt.sh
```

This uses `nvcr.io/nvidia/tensorrt:23.12-py3` container (compatible with Triton 24.01) and creates `model.plan`.

**Manual build (if needed):**
```bash
docker run --rm --runtime=nvidia \
    -v $(pwd):/workspace \
    nvcr.io/nvidia/tensorrt:23.12-py3 \
    bash -c "cd /workspace && trtexec --onnx=model.onnx \
        --saveEngine=model.plan \
        --minShapes=input:1x1,spec_ref:1x80x128,sid:1,tid:1,lid:1 \
        --optShapes=input:1x50,spec_ref:1x80x200,sid:1,tid:1,lid:1 \
        --maxShapes=input:1x300,spec_ref:1x80x1000,sid:1,tid:1,lid:1 \
        --fp16"
```

### 3. Start Triton Server

```bash
chmod +x start_triton_server.sh
./start_triton_server.sh
```

The script automatically copies `model.plan` to the model repository and starts the server.

**Manual Docker command:**
```bash
cp model.plan triton_model_repo/vits_tts/1/model.plan

docker run --rm -it \
    --runtime=nvidia \
    --shm-size=1g \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v $(pwd)/triton_model_repo:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models
```

**Note:** Use `--runtime=nvidia` instead of `--gpus all` if you encounter GPU access issues.

### 4. Install Client Dependencies
```bash
pip install tritonclient[grpc] scipy numpy librosa
```

### 5. Run Inference

**Kyrgyz text:**
```bash
python triton_client.py --text "Салам дүйнө" --lang ky --output output.wav
```

**Russian text:**
```bash
python triton_client.py --text "Привет мир" --lang ru --output output.wav
```

**With all options:**
```bash
python triton_client.py \
    --text "Салам дүйнө" \
    --lang ky \
    --speaker 0 \
    --tone 0 \
    --language 0 \
    --reference-audio reference.wav \
    --output output.wav
```

## Client Options

| Option | Default | Description |
|--------|---------|-------------|
| `--text` | (required) | Text to synthesize |
| `--lang` | `ky` | Text language for tokenization (`ky` or `ru`) |
| `--speaker` | `0` | Speaker ID |
| `--tone` | `0` | Tone ID |
| `--language` | `0` | Language embedding ID |
| `--reference-audio` | None | Reference audio for style transfer |
| `--output` | `output.wav` | Output WAV file path |
| `--triton-url` | `localhost:8001` | Triton gRPC server URL |
| `--sample-rate` | `22050` | Output sample rate |

## Files Structure
```
VITS100/
├── model.onnx                    # Exported ONNX model
├── model.plan                    # TensorRT engine
├── build_tensorrt.sh             # Script to build TensorRT plan
├── start_triton_server.sh        # Script to start Triton server
├── triton_client.py              # Python inference client
└── triton_model_repo/
    └── vits_tts/
        ├── config.pbtxt          # Triton model configuration
        └── 1/
            └── model.plan        # TensorRT engine (copied by script)
```

## API Endpoints

| Service | URL |
|---------|-----|
| HTTP | http://localhost:8000 |
| gRPC | localhost:8001 |
| Metrics | http://localhost:8002/metrics |

## Troubleshooting

### GPU not accessible in Docker
```
ERROR: The NVIDIA Driver is present, but CUDA failed to initialize
```
**Solution:** Use `--runtime=nvidia` instead of `--gpus all`:
```bash
docker run --runtime=nvidia ...
```

If that doesn't work, reinstall nvidia-container-toolkit:
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### TensorRT version mismatch
```
Error Code 1: Serialization (Version tag does not match)
```
**Solution:** Rebuild the plan using `./build_tensorrt.sh` which ensures version compatibility.

### Model not loading
- Check that `model.plan` exists in `triton_model_repo/vits_tts/1/`
- Verify the plan was built with matching TensorRT version

### Dynamic batching error
```
dynamic batching preferred size must be <= max batch size
```
**Solution:** The config uses `max_batch_size: 0` (no batching). Don't add `dynamic_batching` config.

## Performance

With RTX 4070:
- **Throughput:** ~164 QPS
- **Latency:** ~6ms average (5-17ms range)
- **GPU Memory:** ~80 MiB
