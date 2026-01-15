import tensorrt as trt
import os

# 1. Setup Logger
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

# 2. Define Builder
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()

# Set workspace memory limit (4GB)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

parser = trt.OnnxParser(network, TRT_LOGGER)

# 3. Parse ONNX
with open("model.onnx", "rb") as model:
    if not parser.parse(model.read()):
        print("ERROR: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()

# 4. Define Optimization Profile
# Note: The model uses fixed max_y_length = phonemes * 30 internally
# So we need to ensure input shapes are reasonable
profile = builder.create_optimization_profile()

# input shape: [batch_size, phonemes]
# min/opt/max for (batch_size, phonemes)
profile.set_shape("input",
    min=(1, 1),        # minimum: 1 batch, 1 phoneme
    opt=(1, 128),      # optimal: 1 batch, 128 phonemes
    max=(4, 512)       # maximum: 4 batch, 512 phonemes
)

# spec_ref shape: [batch_size, 80, spec_length]
# min/opt/max for (batch_size, mel_channels, spec_length)
profile.set_shape("spec_ref",
    min=(1, 80, 1),    # minimum: 1 batch, 80 mels, 1 frame
    opt=(1, 80, 128),  # optimal: 1 batch, 80 mels, 128 frames
    max=(1, 80, 512)   # maximum: 4 batch, 80 mels, 512 frames
)

config.add_optimization_profile(profile)

# 5. Build Engine
print("Building engine... this might take a while...")
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine:
    with open("model.plan", "wb") as f:
        f.write(serialized_engine)
    print("Success! Engine saved to model.plan")
else:
    print("Failed to build engine.")