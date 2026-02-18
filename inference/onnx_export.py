### from @nshmyrev's fork :  https://github.com/alphacep/MB-iSTFT-VITS2/blob/main/export.py

import warnings
import torch
from torch.onnx import TrainingMode
from torch.nn import functional as F
import sys
import os

# Add parent directory to path for project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress ONNX export warnings that don't affect model correctness
warnings.filterwarnings("ignore", message="Exporting a model to ONNX with a batch_size other than 1")
warnings.filterwarnings("ignore", message="Constant folding - Only steps=1 can be constant folded")
warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean")
warnings.filterwarnings("ignore", message="Converting a tensor to a Python integer")
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

import utils
from models import SynthesizerTrn
from text.symbols import symbols


#- Variable section
PATH_TO_CONFIG = "../logs/mbank/config.json"
PATH_TO_MODEL = "../logs/mbank/G_194000.pth"
OUTPUT_ONNX = "model.onnx"
OPSET_VERSION = 17
posterior_channels = 80

# Reference mel specs — ordered by [lid * N_SPEAKERS * N_TONES + sid * N_TONES + tid]
#
# lid=0 (ky):
#   sid=0 (Timur):    tid=0 neutral, tid=1 strict, tid=2 friendly
#   sid=1 (Aiganysh): tid=0 neutral, tid=1 strict, tid=2 friendly
# lid=1 (ru):
#   sid=0 (Timur):    tid=0 neutral, tid=1 strict, tid=2 friendly
#   sid=1 (Aiganysh): tid=0 neutral, tid=1 strict, tid=2 friendly
N_SPEAKERS = 2
N_TONES = 3
SPEC_DIR = os.path.dirname(os.path.abspath(__file__))
REF_SPEC_FILES = [
    # lid=0 (ky)
    "00001_001_news_01-05_001_1_Timur_neutral_kg.mel.pt",              # lid=0, sid=0, tid=0
    "00002_002_inter_sounds_neutral_048_1_Timur_strict_kg.mel.pt",     # lid=0, sid=0, tid=1
    "00005_005_inter_sounds_neutral_048_1_Timur_friendly_kg.mel.pt",   # lid=0, sid=0, tid=2
    "00002_002_news_01-05_018_1_Aiganysh_neutral_kg.mel.pt",           # lid=0, sid=1, tid=0
    "00000_000_inter_sounds_neutral_035_1_Aiganysh_strict_kg.mel.pt",  # lid=0, sid=1, tid=1
    "00044_003_sales_mislamic_names_addresses_001_5_Aiganysh_friendly_kg.mel.pt",  # lid=0, sid=1, tid=2
    # lid=1 (ru) — TODO: fill in Russian reference mel spec files
    "00000_000_ru_support_022_1_Timur_neutral_ru.mel.pt",              # lid=1, sid=0, tid=0
    "00000_000_ru_support_022_1_Timur_neutral_ru.mel.pt",               # lid=1, sid=0, tid=1
    "00000_000_ru_support_022_1_Timur_neutral_ru.mel.pt",             # lid=1, sid=0, tid=2
    "00000_000_ru_support_040_1_Aiganysh_strict_ru.mel.pt",           # lid=1, sid=1, tid=0
    "00000_000_ru_support_040_1_Aiganysh_strict_ru.mel.pt",            # lid=1, sid=1, tid=1
    "00000_000_ru_support_040_1_Aiganysh_strict_ru.mel.pt",          # lid=1, sid=1, tid=2
]

hps = utils.get_hparams_from_file(PATH_TO_CONFIG)

net_g = SynthesizerTrn(
    len(symbols),
    spec_channels=posterior_channels,
    segment_size=hps.train.segment_size // hps.data.hop_length,
    is_onnx=True,
    **hps.model)

_ = utils.load_checkpoint(PATH_TO_MODEL, net_g, None)

num_symbols = net_g.n_vocab

# Load precomputed mel specs, pad to same length, and bake into the model
specs = [torch.load(os.path.join(SPEC_DIR, f), weights_only=True) for f in REF_SPEC_FILES]
spec_lengths = torch.tensor([s.shape[-1] for s in specs], dtype=torch.long)
max_spec_len = max(s.shape[-1] for s in specs)
padded_specs = torch.stack([F.pad(s, (0, max_spec_len - s.shape[-1])) for s in specs])  # [6, 80, max_len]

net_g.register_buffer('ref_specs', padded_specs)
net_g.register_buffer('ref_spec_lengths', spec_lengths)


def infer_forward(text, emphasis, sid, tid, lid):
    """ONNX-compatible forward wrapper.

    lid, sid and tid select the baked-in reference spectrogram:
        idx = lid * N_SPEAKERS * N_TONES + sid * N_TONES + tid
    """
    idx = lid.long() * N_SPEAKERS * N_TONES + sid.long() * N_TONES + tid.long()  # [batch]
    spec = net_g.ref_specs[idx]              # [batch, 80, max_len]
    spec_len = net_g.ref_spec_lengths[idx]   # [batch]

    result = net_g.infer(
            text,
            spec,
            emphasis=emphasis,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0,
            spec_lengths=spec_len,
    )
    audio = result[0].squeeze(1)  # [B, 1, T] -> [B, T]
    y_lengths = result[5].to(torch.int32)

    return audio, y_lengths


with torch.no_grad():
    net_g.dec.remove_weight_norm()
    net_g.flow.remove_weight_norm()
    net_g.forward = infer_forward

net_g.eval()

# Dummy inputs for ONNX tracing (batch_size=1)
batch_size = 1
phoneme_length = 500  # Must be >= longest expected input sequence

dmy_text = torch.randint(low=0, high=num_symbols, size=(batch_size, phoneme_length), dtype=torch.int32)
dmy_emphasis = torch.zeros(batch_size, phoneme_length, dtype=torch.int32)
dmy_sid = torch.zeros(batch_size, dtype=torch.int32)
dmy_tid = torch.zeros(batch_size, dtype=torch.int32)
dmy_lid = torch.zeros(batch_size, dtype=torch.int32)

dummy_input = (dmy_text, dmy_emphasis, dmy_sid, dmy_tid, dmy_lid)

torch.onnx.export(
        model=net_g,
        args=dummy_input,
        f=OUTPUT_ONNX,
        verbose=False,
        opset_version=OPSET_VERSION,
        training=TrainingMode.EVAL,
        input_names=["input", "emphasis", "sid", "tid", "lid"],
        output_names=["raw_waveform", "y_length"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "emphasis": {0: "batch_size", 1: "phonemes"},
            "raw_waveform": {0: "batch_size", 1: "audio_length"},
            "y_length": {0: "batch_size"},
            "sid": {0: "batch_size"},
            "tid": {0: "batch_size"},
            "lid": {0: "batch_size"},
        },
)

print(f"ONNX export completed successfully! Saved to {OUTPUT_ONNX}")
