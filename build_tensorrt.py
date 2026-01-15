import tensorrt as trt
import os

# 1. Setup Logger
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

# 2. Define Builder
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
parser = trt.OnnxParser(network, TRT_LOGGER)

# 3. Parse ONNX
with open("model.onnx", "rb") as model:
    if not parser.parse(model.read()):
        print("ERROR: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()

# 4. Define Optimization Profile
profile = builder.create_optimization_profile()
profile.set_shape("input",         (1, 4),       (1, 128),       (4, 512))
profile.set_shape("spec_ref",      (1, 80, 8),  (1, 80, 128),  (1, 80, 512))
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