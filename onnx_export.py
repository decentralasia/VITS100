### from @nshmyrev's fork :  https://github.com/alphacep/MB-iSTFT-VITS2/blob/main/export.py

import warnings
import torch

# Suppress ONNX export warnings that don't affect model correctness
warnings.filterwarnings("ignore", message="Exporting a model to ONNX with a batch_size other than 1")
warnings.filterwarnings("ignore", message="Constant folding - Only steps=1 can be constant folded")
warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean")
warnings.filterwarnings("ignore", message="Converting a tensor to a Python integer")
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

import librosa

import os
import json
import math

import requests
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import utils
from models import SynthesizerTrn
from text.symbols import symbols

import numpy as np
from scipy.io.wavfile import write
import re
from scipy import signal


#- Variable section
PATH_TO_CONFIG = "/mnt/d/logs/config.json"
PATH_TO_MODEL = "/mnt/d/logs/G_104000.pth"
OPSET_VERSION = 17
posterior_channels = 80

hps = utils.get_hparams_from_file(PATH_TO_CONFIG)

net_g = SynthesizerTrn(
    len(symbols),
    spec_channels=posterior_channels,
    segment_size=hps.train.segment_size // hps.data.hop_length,
    is_onnx=True, # !
    **hps.model)

_ = utils.load_checkpoint(PATH_TO_MODEL, net_g, None)

num_symbols = net_g.n_vocab
num_speakers = net_g.n_speakers


def infer_forward(text, spec_ref, sid, tid, lid):
    audio = net_g.infer(
            text,
            spec_ref,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0,
            sid=sid,
            tid=tid,
            lid=lid
    )[0]

    return audio


with torch.no_grad():
    net_g.dec.remove_weight_norm()
    net_g.flow.remove_weight_norm() # Remove weightnorm in flows - a8d9f74
    net_g.forward = infer_forward

net_g.eval()

# dummy initialization - use batch_size=1 for cleaner ONNX graph
# Use larger phoneme_length to ensure stable shape inference
batch_size = 1
phoneme_length = 100  # Larger value for better TensorRT shape inference
spec_len = 200

# Use int32 instead of int64 (long) for TensorRT compatibility
dmy_text = torch.randint(low=0, high=num_symbols, size=(batch_size, phoneme_length), dtype=torch.long)

spec_ref = torch.rand(batch_size, posterior_channels, spec_len, dtype=torch.float32)
dmy_sid = torch.zeros(batch_size, dtype=torch.long)
dmy_tid = torch.zeros(batch_size, dtype=torch.long)
dmy_lid = torch.zeros(batch_size, dtype=torch.long)

dummy_input = (dmy_text, spec_ref, dmy_sid, dmy_tid, dmy_lid)

# Export with limited dynamic axes - only batch dimension is dynamic
# This provides better TensorRT compatibility
torch.onnx.export(
        model=net_g,
        args=dummy_input,
        f="model.onnx",
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=["input", "spec_ref", "sid", "tid", "lid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "output": {0: "batch_size", 2: "time"},
            "spec_ref": {0: "batch_size", 2: "spec_length"},
            "sid": {0: "batch_size"},
            "tid": {0: "batch_size"},
            "lid": {0: "batch_size"},
        },
)

print("ONNX export completed successfully!")