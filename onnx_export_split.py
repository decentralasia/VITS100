"""
Split ONNX Export for VITS TTS Model

This script exports the VITS model as two separate ONNX models:
1. Acoustic model: text -> latent (z)
2. Generator/Vocoder: latent (z) -> waveform

Reference embeddings are pre-computed and baked into the acoustic model.
"""

import warnings
import torch
import torch.nn as nn
import numpy as np

warnings.filterwarnings("ignore", message="Exporting a model to ONNX with a batch_size other than 1")
warnings.filterwarnings("ignore", message="Constant folding - Only steps=1 can be constant folded")
warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean")
warnings.filterwarnings("ignore", message="Converting a tensor to a Python integer")
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

import utils
from models import SynthesizerTrn
from text.symbols import symbols
import commons


#- Configuration
PATH_TO_CONFIG = "/mnt/d/logs/config.json"
PATH_TO_MODEL = "/mnt/d/logs/G_498000.pth"
OPSET_VERSION = 17
posterior_channels = 80

# Pre-computed reference mel spectrograms for each speaker
# Replace these paths with your actual reference files
REFERENCE_MELS = {
    0: "/mnt/d/timur_ref.npy",    # Speaker 0 reference
    1: "/mnt/d/aiganysh_ref.npy", # Speaker 1 reference
}

hps = utils.get_hparams_from_file(PATH_TO_CONFIG)

net_g = SynthesizerTrn(
    len(symbols),
    spec_channels=posterior_channels,
    segment_size=hps.train.segment_size // hps.data.hop_length,
    is_onnx=True,
    **hps.model)

_ = utils.load_checkpoint(PATH_TO_MODEL, net_g, None)
net_g.eval()

num_symbols = net_g.n_vocab
num_speakers = net_g.n_speakers


def precompute_reference_embeddings():
    """Pre-compute reference embeddings for all speakers."""
    embeddings = {}
    with torch.no_grad():
        for sid, ref_path in REFERENCE_MELS.items():
            ref_mel = np.load(ref_path)
            ref_mel = torch.from_numpy(ref_mel).float().unsqueeze(0)  # [1, 80, T]
            ref_emb = net_g.ref_enc(ref_mel.transpose(1, 2))  # [1, 256]
            embeddings[sid] = ref_emb
            print(f"Pre-computed reference embedding for speaker {sid}: shape {ref_emb.shape}")
    return embeddings


class AcousticModel(nn.Module):
    """
    Acoustic model that takes text and outputs latent z for the vocoder.
    Reference embeddings are pre-computed and stored as buffers.
    """
    def __init__(self, synthesizer, ref_embeddings):
        super().__init__()
        self.enc_p = synthesizer.enc_p
        self.dp = synthesizer.dp
        self.flow = synthesizer.flow
        self.emb_speaker = synthesizer.emb_speaker
        self.emb_tone = synthesizer.emb_tone
        self.emb_language = synthesizer.emb_language
        
        # Store pre-computed reference embeddings as buffers
        # Stack all embeddings: [num_speakers, 256]
        ref_emb_tensor = torch.stack([ref_embeddings[i] for i in range(len(ref_embeddings))], dim=0)
        self.register_buffer('ref_embeddings', ref_emb_tensor.squeeze(1))  # [num_speakers, 256]
        
        self.noise_scale = 0.667
        self.length_scale = 1.0
        
    def forward(self, x, sid, tid, lid):
        x_lengths = torch.ones(x.shape[0], device=x.device, dtype=torch.long) * x.shape[1]
        
        # Get pre-computed reference embedding for speaker
        reference_emb = self.ref_embeddings[sid].unsqueeze(-1)  # [B, 256, 1]
        
        # Build conditioning vector g
        g = self.emb_speaker(sid).unsqueeze(-1)  # [B, 256, 1]
        g = g + self.emb_tone(tid).unsqueeze(-1)
        g = g + self.emb_language(lid).unsqueeze(-1)
        g = g + reference_emb
        
        # Text encoder
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)
        
        # Duration predictor
        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * self.length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)
        
        # Expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        
        # Sample z_p and flow
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        
        # Apply mask to z and return
        z_masked = z * y_mask
        
        return z_masked, y_mask, y_lengths.to(torch.int32)


class VocoderModel(nn.Module):
    """
    Vocoder/Generator that takes latent z and outputs waveform.
    No g conditioning needed - z already contains all information.
    """
    def __init__(self, synthesizer):
        super().__init__()
        self.dec = synthesizer.dec
        
    def forward(self, z, y_mask):
        o, o_mb = self.dec(z * y_mask, g=None)
        return o.squeeze(1)  # [B, T] - remove channel dim


def export_acoustic_model(acoustic_model, output_path="acoustic_model.onnx"):
    """Export the acoustic model to ONNX."""
    acoustic_model.eval()
    
    batch_size = 1
    phoneme_length = 100
    
    dmy_text = torch.randint(low=0, high=num_symbols, size=(batch_size, phoneme_length), dtype=torch.long)
    dmy_sid = torch.zeros(batch_size, dtype=torch.long)
    dmy_tid = torch.zeros(batch_size, dtype=torch.long)
    dmy_lid = torch.zeros(batch_size, dtype=torch.long)
    
    dummy_input = (dmy_text, dmy_sid, dmy_tid, dmy_lid)
    
    with torch.no_grad():
        torch.onnx.export(
            model=acoustic_model,
            args=dummy_input,
            f=output_path,
            verbose=False,
            opset_version=OPSET_VERSION,
            input_names=["input", "sid", "tid", "lid"],
            output_names=["z", "y_mask", "y_length"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "phonemes"},
                "z": {0: "batch_size", 2: "time"},
                "y_mask": {0: "batch_size", 2: "time"},
                "y_length": {0: "batch_size"},
                "sid": {0: "batch_size"},
                "tid": {0: "batch_size"},
                "lid": {0: "batch_size"},
            },
        )
    print(f"Acoustic model exported to: {output_path}")


def export_vocoder_model(vocoder_model, output_path="vocoder_model.onnx"):
    """Export the vocoder model to ONNX."""
    vocoder_model.eval()
    
    batch_size = 1
    latent_channels = 80  # inter_channels
    latent_length = 200
    
    dmy_z = torch.randn(batch_size, latent_channels, latent_length)
    dmy_y_mask = torch.ones(batch_size, 1, latent_length)
    
    dummy_input = (dmy_z, dmy_y_mask)
    
    with torch.no_grad():
        torch.onnx.export(
            model=vocoder_model,
            args=dummy_input,
            f=output_path,
            verbose=False,
            opset_version=OPSET_VERSION,
            input_names=["z", "y_mask"],
            output_names=["raw_waveform"],
            dynamic_axes={
                "z": {0: "batch_size", 2: "time"},
                "y_mask": {0: "batch_size", 2: "time"},
                "raw_waveform": {0: "batch_size", 1: "audio_time"},
            },
        )
    print(f"Vocoder model exported to: {output_path}")


def export_combined_model_with_precomputed_refs(ref_embeddings, output_path="model_precomputed.onnx"):
    """
    Export a single combined model with pre-computed reference embeddings.
    No spec_ref input needed - embeddings are baked in.
    """
    
    class CombinedModelPrecomputed(nn.Module):
        def __init__(self, synthesizer, ref_embs):
            super().__init__()
            self.enc_p = synthesizer.enc_p
            self.dp = synthesizer.dp
            self.flow = synthesizer.flow
            self.dec = synthesizer.dec
            self.emb_speaker = synthesizer.emb_speaker
            self.emb_tone = synthesizer.emb_tone
            self.emb_language = synthesizer.emb_language
            
            ref_emb_tensor = torch.stack([ref_embs[i] for i in range(len(ref_embs))], dim=0)
            self.register_buffer('ref_embeddings', ref_emb_tensor.squeeze(1))
            
            self.noise_scale = 0.667
            self.length_scale = 1.0
            
        def forward(self, x, sid, tid, lid):
            x_lengths = torch.ones(x.shape[0], device=x.device, dtype=torch.long) * x.shape[1]
            
            reference_emb = self.ref_embeddings[sid].unsqueeze(-1)
            
            g = self.emb_speaker(sid).unsqueeze(-1)
            g = g + self.emb_tone(tid).unsqueeze(-1)
            g = g + self.emb_language(lid).unsqueeze(-1)
            g = g + reference_emb
            
            x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)
            
            logw = self.dp(x, x_mask, g=g)
            w = torch.exp(logw) * x_mask * self.length_scale
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = commons.generate_path(w_ceil, attn_mask)
            
            m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
            logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
            
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.noise_scale
            z = self.flow(z_p, y_mask, g=g, reverse=True)
            
            o, o_mb = self.dec(z * y_mask, g=g)
            
            return o.squeeze(1), y_lengths.to(torch.int32)
    
    combined = CombinedModelPrecomputed(net_g, ref_embeddings)
    combined.eval()
    
    # Remove weight norm
    combined.dec.remove_weight_norm()
    combined.flow.remove_weight_norm()
    
    batch_size = 1
    phoneme_length = 100
    
    dmy_text = torch.randint(low=0, high=num_symbols, size=(batch_size, phoneme_length), dtype=torch.long)
    dmy_sid = torch.zeros(batch_size, dtype=torch.long)
    dmy_tid = torch.zeros(batch_size, dtype=torch.long)
    dmy_lid = torch.zeros(batch_size, dtype=torch.long)
    
    dummy_input = (dmy_text, dmy_sid, dmy_tid, dmy_lid)
    
    with torch.no_grad():
        torch.onnx.export(
            model=combined,
            args=dummy_input,
            f=output_path,
            verbose=False,
            opset_version=OPSET_VERSION,
            input_names=["input", "sid", "tid", "lid"],
            output_names=["raw_waveform", "y_length"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "phonemes"},
                "raw_waveform": {0: "batch_size", 1: "time"},
                "y_length": {0: "batch_size"},
                "sid": {0: "batch_size"},
                "tid": {0: "batch_size"},
                "lid": {0: "batch_size"},
            },
        )
    print(f"Combined model with precomputed refs exported to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export VITS models to ONNX")
    parser.add_argument("--mode", choices=["split", "combined", "all"], default="combined",
                        help="Export mode: split (acoustic+vocoder), combined (single model), or all")
    args = parser.parse_args()
    
    print("Pre-computing reference embeddings...")
    ref_embeddings = precompute_reference_embeddings()
    
    with torch.no_grad():
        net_g.dec.remove_weight_norm()
        net_g.flow.remove_weight_norm()
    
    if args.mode in ["split", "all"]:
        print("\nExporting split models...")
        acoustic = AcousticModel(net_g, ref_embeddings)
        vocoder = VocoderModel(net_g)
        
        export_acoustic_model(acoustic, "acoustic_model.onnx")
        export_vocoder_model(vocoder, "vocoder_model.onnx")
    
    if args.mode in ["combined", "all"]:
        print("\nExporting combined model with precomputed reference embeddings...")
        export_combined_model_with_precomputed_refs(ref_embeddings, "model_precomputed.onnx")
    
    print("\nONNX export completed successfully!")
