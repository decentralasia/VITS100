'''
Defines the set of symbols used in text input to the model.
'''
'''
# english_cleaners
_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

'''

'''
# Russian - from https://github.com/FENRlR/MB-iSTFT-VITS2/issues/2
_pad = '_'
_punctuation = ' !+,-.:;?«»—'
_letters = 'абвгдежзийклмнопрстуфхцчшщъыьэюяё'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
'''

#''' Symbols for en/ko/ja cleaners - from MB-iSTFT-VITS-multilingual
_pad        = '_'
_punctuation = '!? ^,;.'

# Kyrgyz specific letters (includes Ң, Ө, Ү which are specific to Kyrgyz)
_kyrgyz_letters = 'АБВГДЕЁЖЗИЙКЛМНҢОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнңоөпрстуүфхцчшщъыьэюя'

# Russian letters (standard Russian alphabet without Kyrgyz-specific letters)
_russian_letters = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'

_paralinguistics = ["<inhale>", "<exhale>", "<yawn>", "<cough>"]

# Create language-prefixed symbols for separate token IDs
_kyrgyz_symbols = ['<kg>' + c for c in _kyrgyz_letters]
_russian_symbols = ['<ru>' + c for c in _russian_letters]

# Build complete symbol list
symbols = [_pad] + list(_punctuation) + _kyrgyz_symbols + _russian_symbols + _paralinguistics

# Special symbol ids
SPACE_ID = symbols.index(" ")

# Language codes
KYRGYZ_LANG = 'kg'
RUSSIAN_LANG = 'ru'
