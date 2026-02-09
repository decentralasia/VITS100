import time
import os
import re
import json
import random
import numpy as np
import torch
import torch.utils.data

import commons
from mel_processing import spectrogram_torch, mel_spectrogram_torch, spec_to_mel_torch
from utils import load_wav_to_torch_2, load_filepaths_and_text

# Import Kyrgyz phonemizer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "AIF"))
from preprocessing_utils.ky_phonemizer import KyrgyzToIpaV2

# Load symbols encoding and create blank token
_SYMBOLS_ENCODING_PATH = os.path.join(os.path.dirname(__file__), "AIF/model_artifacts/symbols_encoding_train_v7.json")
_TRANSLITERATION_MAPPING_PATH = os.path.join(os.path.dirname(__file__), "AIF/phonimization_artifacts/transliteration_mapping.json")
_FST_PATH = os.path.join(os.path.dirname(__file__), "AIF/phonimization_artifacts/ipa.ohfst")

with open(_SYMBOLS_ENCODING_PATH, "r", encoding="utf-8") as f:
    SYMBOLS_MAPPING = json.load(f)

# Reserve blank token with biggest token id + 1
BLANK_TOKEN_ID = max(SYMBOLS_MAPPING.values()) + 2
EXHALE_TOKEN_ID = max(SYMBOLS_MAPPING.values()) + 1

with open(_TRANSLITERATION_MAPPING_PATH, "r", encoding="utf-8") as f:
    TRANSLITERATION_MAPPING = json.load(f)

# Initialize Kyrgyz phonemizer
KYRGYZ_PHONEMIZER = KyrgyzToIpaV2(
    symbols_mapping=SYMBOLS_MAPPING,
    transliteratetion_mapping=TRANSLITERATION_MAPPING,
    fst_path=_FST_PATH
)

_whitespace_re = re.compile(r"\s+")

# Mapping for paralinguistic sounds (tags to special symbols)
MAPPING_SOUND = {
    "<inhale/>": "ω",
    "<yawn/>": "ξ",
    "<cough/>": "σ",
    "<inhale>": "ω",
    "<yawn>": "ξ",
    "<cough>": "σ",
    "<exhale>": "τ",
}

def collapse_whitespace(text: str) -> str:
    return re.sub(_whitespace_re, " ", text)

def clean_spaces(text: str) -> str:
    """Remove spaces around punctuation marks."""
    return re.sub(r"\s*([@,.?!])\s*", r"\1", text)

def symbols_to_ids(text: str) -> list:
    """Convert symbols to token IDs using the symbols mapping."""
    return [SYMBOLS_MAPPING[symb] for symb in text]

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
        self.max_text_len = getattr(hparams, "max_text_len", 300)

        # random.seed(1234)
        random.shuffle(self.audiopaths_sid_tone_lang_text)
        self._filter()

        # Use regular dicts instead of defaultdict with lambda (lambdas can't be pickled for multiprocessing)
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
            "ky": 0,
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

        # Statistics tracking
        original_count = len(self.audiopaths_sid_tone_lang_text)
        filtered_too_short = 0
        filtered_too_long = 0
        filtered_missing_file = 0
        accepted_count = 0

        for audiopath, sid, tone_id, lid, real_text, text in self.audiopaths_sid_tone_lang_text:
            text_len = len(text)

            # Check text length
            if text_len < self.min_text_len:
                filtered_too_short += 1
                continue
            elif text_len > self.max_text_len:
                filtered_too_long += 1
                continue

            # Check if audio file exists
            if not os.path.exists(audiopath):
                filtered_missing_file += 1
                continue

            # Try to get file size (will fail silently if file is inaccessible)
            try:
                file_size = os.path.getsize(audiopath)
                if file_size == 0:
                    filtered_missing_file += 1
                    continue

                audiopaths_sid_tone_lang_text_new.append([audiopath, sid, tone_id, lid, real_text, text])
                lengths.append(file_size // (2 * self.hop_length))
                accepted_count += 1
            except (OSError, PermissionError):
                filtered_missing_file += 1
                continue

        self.audiopaths_sid_tone_lang_text = audiopaths_sid_tone_lang_text_new
        self.lengths = lengths

        # Print statistics
        filtered_count = original_count - accepted_count
        print(f"\n{'='*60}")
        print(f"Dataset Filtering Statistics")
        print(f"{'='*60}")
        print(f"Original entries:              {original_count}")
        print(f"Accepted entries:              {accepted_count} ({accepted_count/original_count*100:.1f}%)")
        print(f"Filtered out:                  {filtered_count} ({filtered_count/original_count*100:.1f}%)")
        print(f"\nFiltering breakdown:")
        print(f"  - Too short (< {self.min_text_len}):      {filtered_too_short}")
        print(f"  - Too long (> {self.max_text_len}):       {filtered_too_long}")
        print(f"  - Missing/invalid file:      {filtered_missing_file}")
        print(f"{'='*60}\n")

    def get_audio_text_speaker_tone_lang_pair(self, audiopath_sid_tone_lang_text):
        # separate filename, speaker_id and text
        audiopath, sid, tone, lid, real_text, pronounced_text = audiopath_sid_tone_lang_text
        text, emphasis = self.get_text(pronounced_text, lid)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        tone_id = self.get_tone_id(tone)
        lid = self.get_lid(lid)
        return text, emphasis, spec, wav, sid, tone_id, lid

    def get_audio(self, filename):
        # TODO : if linear spec exists convert to mel from existing linear spec
        audio, sampling_rate = load_wav_to_torch_2(filename, target_sampling_rate=22050)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
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
        # lid can be 'kg'/'ky' for Kyrgyz or 'ru' for Russian
        is_kyrgyz = lid in ('kg', 'ky')
        
        # Add @ at the beginning if not present
        if not text.startswith("@"):
            text = "@" + text
        
        # Replace paralinguistic tags with special symbols
        for tag, symbol in MAPPING_SOUND.items():
            text = text.replace(tag, symbol)
        
        if is_kyrgyz:
            # Phonemize Kyrgyz text (phonemizer handles uppercase with *...* markers)
            phonemized = KYRGYZ_PHONEMIZER.phonemize(text)
            phonemized = collapse_whitespace(phonemized)
            phonemized = clean_spaces(phonemized).strip()
            text_norm, is_highlighted = self._process_phonemized_with_highlights(phonemized)
        else:
            # Russian text: process with highlight detection
            text_clean = collapse_whitespace(text)
            text_clean = clean_spaces(text_clean).strip()
            text_norm, is_highlighted = self._process_russian_with_highlights(text_clean)
        
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, BLANK_TOKEN_ID)
            # Intersperse emphasis: blank tokens adjacent to emphasized tokens should also be emphasized
            is_highlighted = self._intersperse_emphasis(is_highlighted)
        
        text_norm = torch.LongTensor(text_norm)
        is_highlighted = torch.LongTensor(is_highlighted)
        return text_norm, is_highlighted
    
    def _process_phonemized_with_highlights(self, text):
        """
        Process phonemized text with *...* markers for highlighted (uppercase) words.
        The ^ sign inside or directly after highlighted region is also highlighted.
        """
        text_norm = []
        is_highlighted = []
        in_highlight = False
        i = 0
        
        while i < len(text):
            char = text[i]
            if char == '*':
                in_highlight = not in_highlight
                i += 1
                continue
            
            if char in SYMBOLS_MAPPING:
                text_norm.append(SYMBOLS_MAPPING[char])
                # ^ directly after highlighted section should also be highlighted
                if char == '^' and is_highlighted and is_highlighted[-1] == 1:
                    is_highlighted.append(1)
                else:
                    is_highlighted.append(1 if in_highlight else 0)
            i += 1
        
        return text_norm, is_highlighted
    
    def _process_russian_with_highlights(self, text):
        """
        Process Russian text: detect uppercase words (and ^ inside/after them) and mark as highlighted.
        Then lowercase and convert to tokens.
        """
        text_norm = []
        is_highlighted = []
        
        # First pass: identify highlighted positions based on uppercase words
        highlight_positions = set()
        i = 0
        while i < len(text):
            # Find word boundaries
            if text[i].isalpha() or text[i] == '^':
                word_start = i
                while i < len(text) and (text[i].isalpha() or text[i] == '^'):
                    i += 1
                word_end = i
                word = text[word_start:word_end]
                
                # Check if letters in word are all uppercase
                letters = [c for c in word if c.isalpha()]
                if letters and all(c.isupper() for c in letters):
                    for j in range(word_start, word_end):
                        highlight_positions.add(j)
            else:
                i += 1
        
        # Second pass: convert to tokens with highlight info
        for i, char in enumerate(text):
            char_lower = char.lower()
            if char_lower in SYMBOLS_MAPPING:
                text_norm.append(SYMBOLS_MAPPING[char_lower])
                is_highlighted.append(1 if i in highlight_positions else 0)
        
        return text_norm, is_highlighted
    
    def _intersperse_emphasis(self, is_highlighted):
        """
        Intersperse emphasis values with blank token emphasis.
        Blank tokens adjacent to emphasized tokens (before/after/inside word) get emphasis=1.
        
        For sequence [h0, h1, h2, ...] produces [b0, h0, b1, h1, b2, h2, ...]
        where bi = 1 if hi or h(i-1) is 1, else 0
        """
        if not is_highlighted:
            return [0]
        
        result = []
        for i, h in enumerate(is_highlighted):
            # Blank before this token: emphasized if current or previous token is emphasized
            if i == 0:
                blank_emphasis = h  # First blank: same as first token
            else:
                blank_emphasis = 1 if (is_highlighted[i-1] == 1 or h == 1) else 0
            result.append(blank_emphasis)
            result.append(h)
        
        # Final blank: same as last token
        result.append(is_highlighted[-1])
        
        return result

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
        batch: [text_normalized, emphasis, spec_normalized, wav_normalized, sid, tone_id, lid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[2].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[2].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        toneid = torch.LongTensor(len(batch))
        lid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        emphasis_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        text_padded.zero_()
        emphasis_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            emphasis = row[1]
            emphasis_padded[i, :emphasis.size(0)] = emphasis

            spec = row[2]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[4]
            toneid[i] = row[5]
            lid[i] = row[6]

        if self.return_ids:
            return text_padded, text_lengths, emphasis_padded, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, toneid, lid, ids_sorted_decreasing
        return text_padded, text_lengths, emphasis_padded, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, toneid, lid


def print_random_phonemized_samples(manifest_path: str, num_samples: int = 32):
    """
    Read a manifest file and print random samples with their phonemized text.
    
    Args:
        manifest_path: Path to the manifest file (format: audiopath|speaker|tone|lang|real_text|phonemized_text)
        num_samples: Number of random samples to print (default: 32)
    """
    from utils import load_filepaths_and_text
    
    # Load manifest
    data = load_filepaths_and_text(manifest_path)
    
    if len(data) == 0:
        print(f"No data found in manifest: {manifest_path}")
        return
    
    # Sample random entries
    sample_size = min(num_samples, len(data))
    samples = random.sample(data, sample_size)
    
    # Reverse mapping for debugging
    ID_TO_SYMBOL = {v: k for k, v in SYMBOLS_MAPPING.items()}
    
    print("=" * 80)
    print(f"Random {sample_size} Phonemized Samples from: {manifest_path}")
    print("=" * 80)
    
    for idx, entry in enumerate(samples, 1):
        audiopath, sid, tone, lid, real_text, phonemized_text = entry
        
        # Phonemize the text
        is_kyrgyz = lid in ('kg', 'ky')
        
        # Add @ at the beginning if not present
        text = phonemized_text
        if not text.startswith("@"):
            text = "@" + text
        
        # Replace paralinguistic tags with special symbols
        for tag, symbol in MAPPING_SOUND.items():
            text = text.replace(tag, symbol)

        print("kokoko     ", text)
        if is_kyrgyz:
            phonemized = KYRGYZ_PHONEMIZER.phonemize(text)
            phonemized = collapse_whitespace(phonemized)
            phonemized = clean_spaces(phonemized).strip()
        else:
            # Russian text: just clean
            phonemized = collapse_whitespace(text)
            phonemized = clean_spaces(phonemized).strip()
        
        # Convert to token IDs
        phonemized = phonemized.lower()
        token_ids = symbols_to_ids(phonemized)
        symbols_out = [ID_TO_SYMBOL.get(tid, f'[{tid}]') for tid in token_ids]
        
        print(f"\n[{idx:02d}] Speaker: {sid} | Tone: {tone} | Lang: {lid}")
        print(f"     Original:   {real_text}")
        print(f"     Phonemized: {phonemized}")
        print(f"     Symbols:    {''.join(symbols_out)}")
        print(f"     Token IDs:  {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
    
    print("\n" + "=" * 80)
    print(f"Printed {sample_size} random samples")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test phonemization or print random samples from manifest")
    parser.add_argument("--manifest", type=str, help="Path to manifest file to print random phonemized samples")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of random samples to print (default: 32)")
    args = parser.parse_args()
    
    if args.manifest:
        print_random_phonemized_samples(args.manifest, args.num_samples)
    else:
        # Test get_text method using a mock class
        print("=" * 60)
        print("Testing get_text method with phonemization and highlighting")
        print("=" * 60)
        
        # Reverse mapping for debugging
        ID_TO_SYMBOL = {v: k for k, v in SYMBOLS_MAPPING.items()}
        
        # Create a minimal mock class to test get_text
        class MockLoader:
            def __init__(self, add_blank=False):
                self.add_blank = add_blank
            
            # Copy the methods from TextAudioSpeakerToneLangLoader
            get_text = TextAudioSpeakerToneLangLoader.get_text
            _process_phonemized_with_highlights = TextAudioSpeakerToneLangLoader._process_phonemized_with_highlights
            _process_russian_with_highlights = TextAudioSpeakerToneLangLoader._process_russian_with_highlights
            _intersperse_emphasis = TextAudioSpeakerToneLangLoader._intersperse_emphasis
        
        loader = MockLoader(add_blank=False)
        loader_with_blank = MockLoader(add_blank=True)
        
        def test_and_print(text, lid):
            text_norm, is_highlighted = loader.get_text(text, lid)
            symbols_out = [ID_TO_SYMBOL.get(i.item(), f'[{i.item()}]') for i in text_norm]
            
            print(f"\nInput:        '{text}'")
            print(f"Lang:         '{lid}'")
            print(f"Symbols:      '{''.join(symbols_out)}'")
            print(f"Token IDs:    {text_norm.tolist()}")
            print(f"Is_highlight: {is_highlighted.tolist()}")
            print(f"Length:       {len(text_norm)}")
        
        # Kyrgyz examples
        kyrgyz_examples = [
            ("салам", "ky"),
            ("САЛАМ", "ky"),  # uppercase - should be highlighted
            ("мен СЕНИ^ сүйөм", "ky"),  # mixed with ^ after uppercase
            ("кыргызстан", "ky"),
            ("салам , <yawn/> , кандайсың", "ky"),
        ]
        
        # Russian examples  
        russian_examples = [
            ("привет", "ru"),
            ("ПРИВЕТ", "ru"),  # uppercase - should be highlighted
            ("я ТЕБЯ^ люблю", "ru"),  # mixed with ^ after uppercase
            ("москва", "ru"),
            ("привет , <yawn/> , как дела", "ru"),
        ]
        
        print("\n--- Kyrgyz (phonemized) ---")
        for text, lid in kyrgyz_examples:
            test_and_print(text, lid)
        
        print("\n--- Russian (no phonemization) ---")
        for text, lid in russian_examples:
            test_and_print(text, lid)
        
        # Test with add_blank=True
        print("\n" + "=" * 60)
        print("Testing with add_blank=True")
        print("=" * 60)
        
        def test_and_print_with_blank(text, lid):
            text_norm, is_highlighted = loader_with_blank.get_text(text, lid)
            symbols_out = [ID_TO_SYMBOL.get(i.item(), f'[{i.item()}]') for i in text_norm]
            
            print(f"\nInput:        '{text}'")
            print(f"Lang:         '{lid}'")
            print(f"Token IDs:    {text_norm.tolist()}")
            print(f"Is_highlight: {is_highlighted.tolist()}")
            print(f"Length:       {len(text_norm)}")
        
        test_and_print_with_blank("ПРИВЕТ", "ru")
        test_and_print_with_blank("я ТЕБЯ^ люблю", "ru")
        
        print("\n" + "=" * 60)
        print(f"BLANK_TOKEN_ID: {BLANK_TOKEN_ID}")
        print(f"Total symbols in mapping: {len(SYMBOLS_MAPPING)}")
        print("=" * 60)
