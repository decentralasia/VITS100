"""
Dataset Statistics Checker
This script checks:
1. How many files were filtered out based on text length
2. How many mel.pt files already exist
3. Detailed statistics about the dataset
"""

import os
import argparse
import wave
from utils import load_filepaths_and_text


def check_dataset_statistics(filelist_path, min_text_len=1, max_text_len=190, use_mel_posterior=True):
    """
    Check dataset statistics including filtering and mel.pt file existence
    
    Args:
        filelist_path: Path to the filelist file
        min_text_len: Minimum text length for filtering
        max_text_len: Maximum text length for filtering
        use_mel_posterior: Whether to check for .mel.pt (True) or .spec.pt (False)
    """
    print("="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"Filelist: {filelist_path}")
    print(f"Text length filter: {min_text_len} <= len(text) <= {max_text_len}")
    print(f"Looking for: {'.mel.pt' if use_mel_posterior else '.spec.pt'} files")
    print("-"*80)
    
    # Load all entries
    all_entries = load_filepaths_and_text(filelist_path)
    total_entries = len(all_entries)
    
    # Track statistics
    filtered_out = 0
    filtered_in = 0
    too_short = 0
    too_long = 0
    missing_audio = 0
    
    spec_exists = 0
    spec_missing = 0
    
    # Track audio lengths (in seconds) for files with/without mel.pt
    audio_lengths_with_mel = []
    audio_lengths_without_mel = []

    filtered_entries = []
    filtered_out_details = []
    
    print(f"\nProcessing {total_entries} entries...")
    print()
    
    # Process each entry
    for entry in all_entries:
        audiopath = entry[0]
        text = entry[-1]  # Text is the last element
        text_len = len(text)
        
        # Check if audio file exists
        if not os.path.exists(audiopath):
            missing_audio += 1
            filtered_out += 1
            filtered_out_details.append((audiopath, text_len, "missing_audio"))
            continue
        
        # Check text length filtering
        if text_len < min_text_len:
            too_short += 1
            filtered_out += 1
            filtered_out_details.append((audiopath, text_len, "too_short"))
            continue
        elif text_len > max_text_len:
            too_long += 1
            filtered_out += 1
            filtered_out_details.append((audiopath, text_len, "too_long"))
            continue
        
        # Entry passes filtering
        filtered_in += 1
        filtered_entries.append(entry)
        
        # Get audio file length
        audio_length = None
        try:
            with wave.open(audiopath, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                audio_length = frames / float(rate)  # duration in seconds
        except Exception as e:
            print(f"Warning: Could not read audio length for {audiopath}: {e}")

        # Check if spec file exists
        spec_filename = audiopath.replace(".wav", ".spec.pt")
        if use_mel_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        
        if os.path.exists(spec_filename):
            spec_exists += 1
            if audio_length is not None:
                audio_lengths_with_mel.append(audio_length)
        else:
            spec_missing += 1
            if audio_length is not None:
                audio_lengths_without_mel.append(audio_length)

    # Print results
    print("FILTERING RESULTS:")
    print("-"*80)
    print(f"Total entries in filelist:        {total_entries}")
    print(f"Entries passing filter:           {filtered_in} ({filtered_in/total_entries*100:.1f}%)")
    print(f"Entries filtered out:             {filtered_out} ({filtered_out/total_entries*100:.1f}%)")
    print()
    print("Filtering breakdown:")
    print(f"  - Too short (< {min_text_len}):        {too_short}")
    print(f"  - Too long (> {max_text_len}):         {too_long}")
    print(f"  - Missing audio file:           {missing_audio}")
    print()

    print("SPECTROGRAM FILES:")
    print("-"*80)
    print(f"Spec files exist:                 {spec_exists} ({spec_exists/filtered_in*100:.1f}% of valid)")
    print(f"Spec files missing:               {spec_missing} ({spec_missing/filtered_in*100:.1f}% of valid)")
    print()
    
    if spec_missing > 0:
        print(f"Note: {spec_missing} spec files will be generated during training")
    
    # Print audio length statistics
    if audio_lengths_with_mel or audio_lengths_without_mel:
        print()
        print("AUDIO LENGTH STATISTICS:")
        print("-"*80)

        if audio_lengths_with_mel:
            total_duration_with = sum(audio_lengths_with_mel)
            avg_duration_with = total_duration_with / len(audio_lengths_with_mel)
            print(f"Files WITH mel.pt ({len(audio_lengths_with_mel)} files):")
            print(f"  - Total duration:  {total_duration_with:.2f} seconds ({total_duration_with/60:.2f} minutes)")
            print(f"  - Average length:  {avg_duration_with:.2f} seconds")
            print(f"  - Min length:      {min(audio_lengths_with_mel):.2f} seconds")
            print(f"  - Max length:      {max(audio_lengths_with_mel):.2f} seconds")

        if audio_lengths_without_mel:
            total_duration_without = sum(audio_lengths_without_mel)
            avg_duration_without = total_duration_without / len(audio_lengths_without_mel)
            print(f"\nFiles WITHOUT mel.pt ({len(audio_lengths_without_mel)} files):")
            print(f"  - Total duration:  {total_duration_without:.2f} seconds ({total_duration_without/60:.2f} minutes)")
            print(f"  - Average length:  {avg_duration_without:.2f} seconds")
            print(f"  - Min length:      {min(audio_lengths_without_mel):.2f} seconds")
            print(f"  - Max length:      {max(audio_lengths_without_mel):.2f} seconds")

        if audio_lengths_with_mel and audio_lengths_without_mel:
            print(f"\nComparison:")
            avg_with = sum(audio_lengths_with_mel) / len(audio_lengths_with_mel)
            avg_without = sum(audio_lengths_without_mel) / len(audio_lengths_without_mel)
            diff = avg_without - avg_with
            print(f"  - Average difference: {abs(diff):.2f} seconds")
            if diff > 0:
                print(f"  - Files without mel.pt are on average LONGER")
            elif diff < 0:
                print(f"  - Files with mel.pt are on average LONGER")
            else:
                print(f"  - No significant difference in average length")

        print()

    # Show some filtered out examples
    if filtered_out > 0:
        print()
        print("FILTERED OUT EXAMPLES (first 10):")
        print("-"*80)
        for i, (path, text_len, reason) in enumerate(filtered_out_details[:10]):
            print(f"{i+1}. {os.path.basename(path)}")
            print(f"   Text length: {text_len}, Reason: {reason}")
        
        if len(filtered_out_details) > 10:
            print(f"   ... and {len(filtered_out_details) - 10} more")
    
    print()
    print("="*80)
    
    return {
        'total': total_entries,
        'filtered_in': filtered_in,
        'filtered_out': filtered_out,
        'too_short': too_short,
        'too_long': too_long,
        'missing_audio': missing_audio,
        'spec_exists': spec_exists,
        'spec_missing': spec_missing,
    }


def main():
    parser = argparse.ArgumentParser(description='Check dataset statistics')
    parser.add_argument('--filelist', type=str, required=True,
                        help='Path to the filelist file')
    parser.add_argument('--min_text_len', type=int, default=1,
                        help='Minimum text length (default: 1)')
    parser.add_argument('--max_text_len', type=int, default=190,
                        help='Maximum text length (default: 190)')
    parser.add_argument('--use_mel', action='store_true',
                        help='Check for .mel.pt files instead of .spec.pt')

    args = parser.parse_args()
    
    check_dataset_statistics(
        args.filelist,
        min_text_len=args.min_text_len,
        max_text_len=args.max_text_len,
        use_mel_posterior=args.use_mel
    )


if __name__ == '__main__':
    main()

