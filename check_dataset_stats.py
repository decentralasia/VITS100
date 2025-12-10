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
import numpy as np
from utils import load_filepaths_and_text


def check_audio_file(audiopath):
    """
    Check if audio file exists, is accessible, and has valid size.

    Returns:
        tuple: (is_valid: bool, error_reason: str or None, file_size: int or None)
    """
    if not os.path.exists(audiopath):
        return False, "file_not_found", None

    try:
        size = os.path.getsize(audiopath)
        if size == 0:
            return False, "empty_file", 0
        return True, None, size
    except PermissionError:
        return False, "permission_denied", None
    except Exception as e:
        return False, f"access_error: {str(e)}", None


def calculate_optimal_boundaries(spec_lengths, num_buckets=9, hop_length=256, sample_rate=22050):
    """
    Calculate optimal bucket boundaries based on spectrogram length distribution.

    Args:
        spec_lengths: List of spectrogram lengths in frames
        num_buckets: Number of buckets to create (default: 9)
        hop_length: Hop length used for spectrogram (default: 256)
        sample_rate: Audio sample rate (default: 22050)

    Returns:
        list: Optimal boundary values
    """
    if not spec_lengths or len(spec_lengths) == 0:
        return None

    lengths_array = np.array(spec_lengths)

    # Calculate percentile-based boundaries
    percentiles = np.linspace(0, 100, num_buckets + 1)[1:]  # Exclude 0th percentile
    boundaries = np.percentile(lengths_array, percentiles).astype(int).tolist()

    # Ensure boundaries are unique and sorted
    boundaries = sorted(list(set(boundaries)))

    return boundaries


def print_boundary_analysis(spec_lengths, current_boundaries=None, hop_length=256, sample_rate=22050):
    """
    Print detailed analysis of audio length distribution and boundary recommendations.

    Args:
        spec_lengths: List of spectrogram lengths in frames
        current_boundaries: Current bucket boundaries (optional)
        hop_length: Hop length used for spectrogram
        sample_rate: Audio sample rate
    """
    if not spec_lengths or len(spec_lengths) == 0:
        print("No valid spectrogram lengths to analyze.")
        return

    lengths_array = np.array(spec_lengths)

    def frames_to_seconds(frames):
        """Convert spectrogram frames to seconds"""
        return frames * hop_length / sample_rate

    print()
    print("="*80)
    print("AUDIO LENGTH DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"Total samples analyzed: {len(lengths_array)}")
    print()
    print("Length statistics (in spectrogram frames and seconds):")
    print(f"  Min:     {lengths_array.min():6d} frames  ({frames_to_seconds(lengths_array.min()):6.2f}s)")
    print(f"  Max:     {lengths_array.max():6d} frames  ({frames_to_seconds(lengths_array.max()):6.2f}s)")
    print(f"  Mean:    {lengths_array.mean():6.0f} frames  ({frames_to_seconds(lengths_array.mean()):6.2f}s)")
    print(f"  Median:  {np.median(lengths_array):6.0f} frames  ({frames_to_seconds(np.median(lengths_array)):6.2f}s)")
    print(f"  Std Dev: {lengths_array.std():6.0f} frames  ({frames_to_seconds(lengths_array.std()):6.2f}s)")
    print()

    # Show percentiles
    print("Percentile distribution:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(lengths_array, p)
        print(f"  {p:2d}th percentile: {val:6.0f} frames  ({frames_to_seconds(val):6.2f}s)")
    print()

    # Analyze current boundaries if provided
    if current_boundaries:
        print("CURRENT BOUNDARIES ANALYSIS:")
        print(f"Current boundaries: {current_boundaries}")
        print()
        print("Samples per bucket:")
        prev_boundary = 0
        for i, boundary in enumerate(current_boundaries):
            count = np.sum((lengths_array > prev_boundary) & (lengths_array <= boundary))
            percentage = count / len(lengths_array) * 100
            print(f"  Bucket {i+1} ({prev_boundary:4d} - {boundary:4d}]: {count:5d} samples ({percentage:5.1f}%)")
            prev_boundary = boundary

        # Count samples outside boundaries
        below = np.sum(lengths_array <= current_boundaries[0])
        above = np.sum(lengths_array > current_boundaries[-1])
        print(f"  Below min ({current_boundaries[0]:4d}):     {below:5d} samples ({below/len(lengths_array)*100:5.1f}%)")
        print(f"  Above max ({current_boundaries[-1]:4d}):     {above:5d} samples ({above/len(lengths_array)*100:5.1f}%)")
        print()

    # Calculate optimal boundaries
    print("RECOMMENDED OPTIMAL BOUNDARIES:")
    print()

    # Option 1: Quantile-based (equal number of samples per bucket)
    optimal_9 = calculate_optimal_boundaries(spec_lengths, num_buckets=9, hop_length=hop_length, sample_rate=sample_rate)
    print(f"Option 1 - Quantile-based (9 buckets, equal sample distribution):")
    print(f"  {optimal_9}")
    print()

    # Option 2: Fewer buckets for simpler distribution
    optimal_7 = calculate_optimal_boundaries(spec_lengths, num_buckets=7, hop_length=hop_length, sample_rate=sample_rate)
    print(f"Option 2 - Quantile-based (7 buckets):")
    print(f"  {optimal_7}")
    print()

    # Option 3: More granular
    optimal_11 = calculate_optimal_boundaries(spec_lengths, num_buckets=11, hop_length=hop_length, sample_rate=sample_rate)
    print(f"Option 3 - Quantile-based (11 buckets, fine-grained):")
    print(f"  {optimal_11}")
    print()

    print("Usage in train.py:")
    print("-" * 80)
    print(f"train_sampler = DistributedBucketSampler(")
    print(f"    train_dataset,")
    print(f"    hps.train.batch_size,")
    print(f"    {optimal_9},  # Replace with your chosen boundaries")
    print(f"    num_replicas=n_gpus,")
    print(f"    rank=rank,")
    print(f"    shuffle=True)")
    print()
    print("="*80)


def check_dataset_statistics(filelist_path, min_text_len=1, max_text_len=190, use_mel_posterior=True, hop_length=256):
    """
    Check dataset statistics including filtering and mel.pt file existence
    
    Args:
        filelist_path: Path to the filelist file
        min_text_len: Minimum text length for filtering
        max_text_len: Maximum text length for filtering
        use_mel_posterior: Whether to check for .mel.pt (True) or .spec.pt (False)
        hop_length: Hop length for spectrogram calculation (default: 256)
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
    
    # Track missing audio reasons
    missing_audio_reasons = {}

    spec_exists = 0
    spec_missing = 0
    
    # Track audio lengths (in seconds) for files with/without mel.pt
    audio_lengths_with_mel = []
    audio_lengths_without_mel = []

    # Track file sizes
    total_audio_size = 0
    valid_audio_count = 0

    # Track spectrogram lengths for boundary calculation
    spec_lengths = []

    filtered_entries = []
    filtered_out_details = []
    
    print(f"\nProcessing {total_entries} entries...")
    print()
    
    # Process each entry
    for entry in all_entries:
        audiopath = entry[0]
        text = entry[-1]  # Text is the last element
        text_len = len(text)
        
        # Check if audio file exists and is accessible
        audio_valid, audio_error, file_size = check_audio_file(audiopath)
        if not audio_valid:
            missing_audio += 1
            filtered_out += 1
            filtered_out_details.append((audiopath, text_len, audio_error))
            # Track reason for missing audio
            missing_audio_reasons[audio_error] = missing_audio_reasons.get(audio_error, 0) + 1
            continue
        
        # Track file size for valid audio
        if file_size is not None:
            total_audio_size += file_size
            valid_audio_count += 1

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
        
        # Calculate spectrogram length (same as in dataset._filter())
        spec_length = file_size // (2 * hop_length)
        spec_lengths.append(spec_length)

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
    print(f"  - Missing/invalid audio:        {missing_audio}")

    if missing_audio_reasons:
        print(f"\n  Missing audio breakdown:")
        for reason, count in sorted(missing_audio_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"    • {reason}: {count} ({count/missing_audio*100:.1f}%)")
    print()

    # Show audio file size statistics
    if valid_audio_count > 0:
        avg_size = total_audio_size / valid_audio_count
        total_size_mb = total_audio_size / (1024 * 1024)
        total_size_gb = total_size_mb / 1024
        print(f"Valid audio files statistics:")
        print(f"  - Total files:     {valid_audio_count}")
        print(f"  - Total size:      {total_size_mb:.2f} MB ({total_size_gb:.2f} GB)")
        print(f"  - Average size:    {avg_size / 1024:.2f} KB")
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
    
    # Print optimal boundary analysis
    if spec_lengths and len(spec_lengths) > 0:
        # Current boundaries from train.py
        current_boundaries = [32, 300, 400, 500, 600, 700, 800, 900, 1000]
        print_boundary_analysis(spec_lengths, current_boundaries=current_boundaries, hop_length=hop_length)

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
        'spec_lengths': spec_lengths,
    }


def main():
    parser = argparse.ArgumentParser(description='Check dataset statistics and calculate optimal bucket boundaries')
    parser.add_argument('--filelist', type=str, required=True,
                        help='Path to the filelist file')
    parser.add_argument('--min_text_len', type=int, default=1,
                        help='Minimum text length (default: 1)')
    parser.add_argument('--max_text_len', type=int, default=190,
                        help='Maximum text length (default: 190)')
    parser.add_argument('--use_mel', action='store_true',
                        help='Check for .mel.pt files instead of .spec.pt')
    parser.add_argument('--hop_length', type=int, default=256,
                        help='Hop length for spectrogram (default: 256)')

    args = parser.parse_args()
    
    check_dataset_statistics(
        args.filelist,
        min_text_len=args.min_text_len,
        max_text_len=args.max_text_len,
        use_mel_posterior=args.use_mel,
        hop_length=args.hop_length
    )


if __name__ == '__main__':
    main()

