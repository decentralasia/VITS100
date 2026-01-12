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
import matplotlib.pyplot as plt

import os
import json
import math

import requests
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols

import numpy as np
from scipy.io.wavfile import write
import re
from scipy import signal


#- Variable section
PATH_TO_CONFIG = "/mnt/d/super_last/config.json"
PATH_TO_MODEL = "/mnt/d/super_last/G_690000.pth"
OPSET_VERSION = 15
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


def infer_forward(text, text_lengths, spec_ref, scales, sid, tid, lid):
    noise_scale = scales[0]
    length_scale = scales[1]
    noise_scale_w = scales[2]
    audio = net_g.infer(
            text,
            text_lengths,
            spec_ref,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
            tid=tid,
            lid=lid,
    )[0].unsqueeze(1)

    return audio


with torch.no_grad():
    net_g.dec.remove_weight_norm()
    net_g.flow.remove_weight_norm() # Remove weightnorm in flows - a8d9f74
    net_g.forward = infer_forward

net_g.eval()

# dummy initialization
dmy_text = torch.randint(low=0, high=num_symbols, size=(1, 50), dtype=torch.long)
dmy_text_length = torch.LongTensor([dmy_text.size(1)])
dmy_sid = torch.LongTensor([0])
dmy_tid = torch.LongTensor([0])
dmy_lid = torch.LongTensor([0])
scale_config = torch.FloatTensor([0.667, 1.0, 0.8]) # scales -> noise, noise_w, length
batch_size = 1
spec_ref= torch.rand(batch_size, posterior_channels, 77, dtype=torch.float32)
dummy_input = (dmy_text, dmy_text_length, spec_ref, scale_config, dmy_sid, dmy_tid, dmy_lid)


# Export
torch.onnx.export(
        model=net_g,
        args=dummy_input,
        f="model.onnx",
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=["input", "input_lengths", "spec_ref", "scales", "sid", "tid", "lid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        },
)