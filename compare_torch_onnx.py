import torch
import numpy as np
from scipy.io.wavfile import write
import onnxruntime as ort
import librosa

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

# Configuration
PATH_TO_CONFIG = "/mnt/d/super_last/config.json"
PATH_TO_MODEL = "/mnt/d/super_last/G_690000.pth"
PATH_TO_ONNX = "model.onnx"
INPUT_TEXT = "бишкек"
INPUT_LANG = "ky"  # Language code: 'ky' for Kyrgyz, 'ru' for Russian
OUTPUT_DIR = "comparison_outputs"
posterior_channels = 80  # Must match the ONNX export configuration

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load config
hps = utils.get_hparams_from_file(PATH_TO_CONFIG)

if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:
    posterior_channels = 80
    hps.data.use_mel_posterior_encoder = True
else:
    posterior_channels = hps.data.filter_length // 2 + 1
    hps.data.use_mel_posterior_encoder = False

# Load PyTorch model
net_g = SynthesizerTrn(
    len(symbols),
    posterior_channels,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).to(device)
net_g.eval()
utils.load_checkpoint(PATH_TO_MODEL, net_g, None)

# Load ONNX model
ort_session = ort.InferenceSession(PATH_TO_ONNX)

# Print ONNX input names to verify
print("ONNX model inputs:")
for inp in ort_session.get_inputs():
    print(f"  {inp.name}: {inp.shape}, {inp.type}")
print()

# Prepare text
def get_text(text, hps, lang_code=None):
    text_norm = text_to_sequence(text, hps.data.text_cleaners, lang_code=lang_code)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

stn_tst = get_text(INPUT_TEXT, hps, lang_code=INPUT_LANG)
x_tst = stn_tst.to(device).unsqueeze(0)
x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)

# Create a dummy reference spectrogram (spec_ref)
# In practice, this should be a real mel spectrogram from a reference audio
spec_ref_length = 77  # Same as used in onnx_export.py
spec_ref = torch.rand(1, posterior_channels, spec_ref_length, dtype=torch.float32).to(device)

# Test configurations: (sid, tid, lid)
configs = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 0),
]

scales = torch.FloatTensor([0.667, 1.0, 0.8])

print(f"Generating audio for text: '{INPUT_TEXT}'")
print(f"Sample rate: {hps.data.sampling_rate}")
print()

# Generate with PyTorch
print("=== PyTorch Generation ===")
for sid, tid, lid in configs:
    sid_tensor = torch.LongTensor([sid]).to(device)
    tid_tensor = torch.LongTensor([tid]).to(device)
    lid_tensor = torch.LongTensor([lid]).to(device)

    with torch.no_grad():
        audio = net_g.infer(
            x_tst, 
            x_tst_lengths,
            y=spec_ref,  # Reference spectrogram
            sid=sid_tensor,
            tid=tid_tensor,
            lid=lid_tensor,
            noise_scale=0.667,
            noise_scale_w=0.8, 
            length_scale=1.0
        )[0][0, 0].data.cpu().float().numpy()
    
    filename = f"{OUTPUT_DIR}/torch_sid{sid}_tid{tid}_lid{lid}.wav"
    write(filename, hps.data.sampling_rate, audio)
    print(f"Generated: {filename} (shape: {audio.shape})")

print()

# Generate with ONNX
print("=== ONNX Generation ===")
x_tst_np = x_tst.cpu().numpy()
x_tst_lengths_np = x_tst_lengths.cpu().numpy()
spec_ref_np = spec_ref.cpu().numpy()
scales_np = scales.numpy()

for sid, tid, lid in configs:
    sid_np = np.array([sid], dtype=np.int64)
    tid_np = np.array([tid], dtype=np.int64)
    lid_np = np.array([lid], dtype=np.int64)

    ort_inputs = {
        "input": x_tst_np,
        "input_lengths": x_tst_lengths_np,
        "spec_ref": spec_ref_np,
        "scales": scales_np,
        "sid": sid_np,
        "tid": tid_np,
        "lid": lid_np,
    }

    audio = ort_session.run(None, ort_inputs)[0]
    audio = audio.squeeze()
    
    filename = f"{OUTPUT_DIR}/onnx_sid{sid}_tid{tid}_lid{lid}.wav"
    write(filename, hps.data.sampling_rate, audio)
    print(f"Generated: {filename} (shape: {audio.shape})")

print()

# Compare outputs
print("=== Comparison ===")
for sid, tid, lid in configs:
    torch_audio, _ = librosa.load(f"{OUTPUT_DIR}/torch_sid{sid}_tid{tid}_lid{lid}.wav", sr=hps.data.sampling_rate)
    onnx_audio, _ = librosa.load(f"{OUTPUT_DIR}/onnx_sid{sid}_tid{tid}_lid{lid}.wav", sr=hps.data.sampling_rate)

    # Ensure same length
    min_len = min(len(torch_audio), len(onnx_audio))
    torch_audio = torch_audio[:min_len]
    onnx_audio = onnx_audio[:min_len]
    
    # Calculate metrics
    mse = np.mean((torch_audio - onnx_audio) ** 2)
    mae = np.mean(np.abs(torch_audio - onnx_audio))
    max_diff = np.max(np.abs(torch_audio - onnx_audio))
    
    print(f"sid={sid}, tid={tid}, lid={lid}:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Max diff: {max_diff:.6f}")
    print()

print(f"All outputs saved to: {OUTPUT_DIR}/")
