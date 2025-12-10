""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols, KYRGYZ_LANG, RUSSIAN_LANG


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

import re

def get_symbol_with_lang(char, lang_code):
    """Get the language-prefixed symbol for a character.

    Args:
        char: single character
        lang_code: language code ('kg' for Kyrgyz, 'ru' for Russian)

    Returns:
        Language-prefixed symbol string
    """
    return f'<{lang_code}>{char}'

def tokenize_pronounced_text(pronounced_text, lang_code=None):
    """Tokenize text with optional language prefix.

    Args:
        pronounced_text: text to tokenize
        lang_code: language code ('ky' or 'ru'). If None, no prefix is added.

    Returns:
        List of tokens
    """
    pattern = re.compile(r'<[^>]*>|.')
    tokens = re.findall(pattern, pronounced_text)

    # If language code is provided, prefix alphabet characters
    if lang_code:
        result_tokens = []
        for token in tokens:
            # If it's already a tag (like <inhale>) or punctuation, keep as is
            if token.startswith('<') or not token.isalpha():
                result_tokens.append(token)
            else:
                # Add language prefix to alphabet characters
                result_tokens.append(get_symbol_with_lang(token, lang_code))
        return result_tokens

    return tokens

def text_to_sequence(text, cleaner_names, lang_code=None):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      lang_code: language code ('ky' for Kyrgyz, 'ru' for Russian). 
                 If None, text must already have language prefixes or contain no alphabetic characters.
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  clean_text = _clean_text(text, cleaner_names)
  clean_text = tokenize_pronounced_text(clean_text, lang_code)
  for symbol in clean_text:
    if symbol not in _symbol_to_id.keys():
      print(text)
      raise ValueError("Not found symbol: {} (lang_code: {})".format(symbol, lang_code))
    symbol_id = _symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence


def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text if symbol in _symbol_to_id.keys()]
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
