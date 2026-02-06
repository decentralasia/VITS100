#!/usr/bin/env python3
"""
Manifest Checker Script
Run this before train.py to validate train and val manifests.

Checks:
- All symbols in pronounced_text (column 6) are valid
- Latin symbols are only allowed inside tags like <yawn>, <cough>, <inhale>, 
  and SSML tags like <pause time="500ms"/>, <emphasis>, </emphasis>
"""

import argparse
import os
import re
import sys
from collections import defaultdict

# Valid symbols definition
_pad = '_'
_punctuation = '!? ^,;.'
_kyrgyz_letters = 'АБВГДЕЁЖЗИЙКЛМНҢОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнңоөпрстуүфхцчшщъыьэюя'
_russian_letters = 'АБВГДЕЁЖЗИЙКЛМНҢОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнңоөпрстуүфхцчшщъыьэюя'

# Combined valid symbols set
VALID_SYMBOLS = set(_pad + _punctuation + _kyrgyz_letters + _russian_letters)

# Allowed tags (only these tags with latin characters are permitted)
ALLOWED_TAGS = {
    '<yawn>', '<cough>', '<inhale>', '<emphasis>',
    '<yawn/>', '<cough/>', '<inhale/>', '<emphasis/>',
}

# Pattern for pause tag with variable time: <pause time="500ms"/>
PAUSE_TAG_PATTERN = re.compile(r'^<pause\s+time="\d+ms"\s*/>$')

# Tag pattern: matches <tag>, </tag>, <tag attr="value"/>
TAG_PATTERN = re.compile(r'<[^>]+/?>')

def load_filepaths_and_text(filename, split="|"):
    """Load manifest file."""
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def load_symbols_mapping(symbols_path):
    """Load valid symbols from JSON file."""
    with open(symbols_path, "r", encoding="utf-8") as f:
        return json.load(f)

def remove_tags(text):
    """Remove all tags from text, returning text without tags."""
    return TAG_PATTERN.sub('', text)

def get_tags(text):
    """Extract all tags from text."""
    return TAG_PATTERN.findall(text)

def is_valid_tag(tag):
    """Check if a tag is valid (in ALLOWED_TAGS or matches pause pattern)."""
    if tag in ALLOWED_TAGS:
        return True
    if PAUSE_TAG_PATTERN.match(tag):
        return True
    return False

def check_text(text, valid_symbols, line_num, filepath):
    """
    Check if text contains only valid symbols.
    Latin characters are only allowed inside specific tags.
    
    Returns list of error messages.
    """
    errors = []
    
    # Check for invalid tags
    tags_in_text = get_tags(text)
    for tag in tags_in_text:
        if not is_valid_tag(tag):
            errors.append(
                f"Line {line_num}: Invalid tag '{tag}'. "
                f"Allowed: {ALLOWED_TAGS} or <pause time=\"Nms\"/>. "
                f"File: {filepath}, Text: '{text[:50]}...'"
            )
    
    # Get text without tags (latin is NOT allowed here)
    text_without_tags = remove_tags(text)
    
    # Check each character in text without tags
    for i, char in enumerate(text_without_tags):
        if char not in valid_symbols:
            # Check if it's a latin character (not allowed outside tags)
            if char.isascii() and char.isalpha():
                errors.append(
                    f"Line {line_num}: Latin character '{char}' found outside tags. "
                    f"File: {filepath}, Text: '{text[:50]}...'"
                )
            else:
                errors.append(
                    f"Line {line_num}: Invalid symbol '{char}' (U+{ord(char):04X}). "
                    f"File: {filepath}, Text: '{text[:50]}...'"
                )
    
    return errors

def check_manifest(manifest_path):
    """
    Check a manifest file for invalid symbols.
    
    Manifest format: audiopath|speaker|tone|lang|real_text|pronounced_text
    We check column 6 (pronounced_text, index 5).
    
    Returns tuple of (errors_list, stats_dict)
    """
    errors = []
    stats = {
        'total_lines': 0,
        'valid_lines': 0,
        'invalid_lines': 0,
        'invalid_symbols': defaultdict(int),
    }
    
    if not os.path.exists(manifest_path):
        return [f"Manifest file not found: {manifest_path}"], stats
    
    entries = load_filepaths_and_text(manifest_path)
    stats['total_lines'] = len(entries)
    
    for line_num, entry in enumerate(entries, 1):
        if len(entry) < 6:
            errors.append(f"Line {line_num}: Invalid format, expected 6 columns, got {len(entry)}")
            stats['invalid_lines'] += 1
            continue
        
        audiopath = entry[0]
        pronounced_text = entry[5]
        
        line_errors = check_text(pronounced_text, VALID_SYMBOLS, line_num, audiopath)
        
        if line_errors:
            errors.extend(line_errors)
            stats['invalid_lines'] += 1
            # Track invalid symbols
            text_without_tags = remove_tags(pronounced_text)
            for char in text_without_tags:
                if char not in VALID_SYMBOLS:
                    stats['invalid_symbols'][char] += 1
        else:
            stats['valid_lines'] += 1
    
    return errors, stats

def main():
    parser = argparse.ArgumentParser(description="Check manifest file for invalid symbols")
    parser.add_argument('manifest', type=str, help='Path to manifest file')
    parser.add_argument('--max-errors', type=int, default=50,
                        help='Maximum number of errors to display')
    args = parser.parse_args()
    
    print(f"Valid symbols: {len(VALID_SYMBOLS)} characters")
    print(f"Includes: pad, punctuation ({_punctuation}), Kyrgyz/Russian letters")
    print(f"Allowed tags: {ALLOWED_TAGS}")
    
    # Check manifest
    print(f"\n{'='*60}")
    print(f"Checking manifest: {args.manifest}")
    print('='*60)
    
    errors, stats = check_manifest(args.manifest)
    
    print(f"Total lines: {stats['total_lines']}")
    print(f"Valid lines: {stats['valid_lines']}")
    print(f"Invalid lines: {stats['invalid_lines']}")
    
    if errors:
        print(f"\nErrors found ({len(errors)} total, showing first {args.max_errors}):")
        for err in errors[:args.max_errors]:
            print(f"  {err}")
        if stats['invalid_symbols']:
            print(f"\nInvalid symbols summary:")
            for sym, count in sorted(stats['invalid_symbols'].items(), key=lambda x: -x[1]):
                print(f"  '{sym}' (U+{ord(sym):04X}): {count} occurrences")
        print(f"\n{'='*60}")
        print("✗ Validation failed!")
        sys.exit(1)
    else:
        print("✓ All lines valid")
        print(f"\n{'='*60}")
        print("✓ Manifest passed validation!")
        sys.exit(0)

if __name__ == "__main__":
    main()
