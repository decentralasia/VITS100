import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import commons
from mel_processing import spectrogram_torch, mel_spectrogram_torch, spec_to_mel_torch
from utils import load_wav_to_torch_2, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence

from typing import Any, Optional


class TextAudioSpeakerToneLangLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, tone, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_tone_lang_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate

        self.use_mel_spec_posterior = getattr(hparams, "use_mel_posterior_encoder", False)
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)
        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_tone_lang_text)
        self._filter()

        self.speaker_dict = {
            "Timur": 0,
            "Aiganysh": 1,
        }
        self.tone_dict = {
            "neutral": 0,
            "strict": 1,
            "friendly": 2,
        }
        self.language_dict = {
            "kg": 0,
            "ru": 1,
        }
        self.hparams = hparams

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_tone_lang_text_new = []
        lengths = []
        for audiopath, sid, tone_id, lid, real_text, text in self.audiopaths_sid_tone_lang_text:
            if self.min_text_len <= len(text) <= self.max_text_len:
                audiopaths_sid_tone_lang_text_new.append([audiopath, sid, tone_id, lid, real_text, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_tone_lang_text = audiopaths_sid_tone_lang_text_new
        self.lengths = lengths

    def get_audio_text_speaker_tone_lang_pair(self, audiopath_sid_tone_lang_text):
        # separate filename, speaker_id and text
        audiopath, sid, tone, lid, real_text, pronounced_text = audiopath_sid_tone_lang_text
        text = self.get_text(pronounced_text, lid)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        tone_id = self.get_tone_id(tone)
        lid = self.get_lid(lid)
        return text, spec, wav, sid, tone_id, lid

    def get_audio(self, filename):
        # TODO : if linear spec exists convert to mel from existing linear spec
        audio, sampling_rate = load_wav_to_torch_2(filename, target_sampling_rate=22050)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename, weights_only=True)
        else:
            if self.use_mel_spec_posterior:
                ''' TODO : (need verification) 
                if linear spec exists convert to 
                mel from existing linear spec (uncomment below lines) '''
                # if os.path.exists(filename.replace(".wav", ".spec.pt")):
                #     # spec, n_fft, num_mels, sampling_rate, fmin, fmax
                #     spec = spec_to_mel_torch(
                #         torch.load(filename.replace(".wav", ".spec.pt")),
                #         self.filter_length, self.n_mel_channels, self.sampling_rate,
                #         self.hparams.mel_fmin, self.hparams.mel_fmax)
                spec = mel_spectrogram_torch(audio_norm, self.filter_length,
                    self.n_mel_channels, self.sampling_rate, self.hop_length,
                    self.win_length, self.hparams.mel_fmin, self.hparams.mel_fmax, center=False)
            else:
                spec = spectrogram_torch(audio_norm, self.filter_length,
                    self.sampling_rate, self.hop_length, self.win_length,
                    center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text, lid):
        # Convert language ID string to language code for text_to_sequence
        # lid can be 'kg' for Kyrgyz or 'ru' for Russian
        lang_code = 'ky' if lid == 'kg' else lid  # Convert 'kg' to 'ky' for consistency
        
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners, lang_code)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = self.speaker_dict[sid]
        sid = torch.LongTensor([int(sid)])
        return sid

    def get_tone_id(self, tone):
        tone_id = self.tone_dict[tone]
        tone_id = torch.LongTensor([int(tone_id)])
        return tone_id

    def get_lid(self, lid):
        l_id = self.language_dict[lid]
        l_id = torch.LongTensor([int(l_id)])
        return l_id

    def __getitem__(self, index):
        return self.get_audio_text_speaker_tone_lang_pair(self.audiopaths_sid_tone_lang_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_tone_lang_text)


class TextAudioSpeakerToneLangCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        toneid = torch.LongTensor(len(batch))
        lid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]
            toneid[i] = row[4]
            lid[i] = row[5]

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, toneid, lid, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, toneid, lid
