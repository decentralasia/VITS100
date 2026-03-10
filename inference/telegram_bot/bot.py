#!/usr/bin/env python3
"""
Telegram Bot for VITS2 TTS — Kyrgyz + Russian, multi-voice.

Flow:  /start → choose language → choose voice → send text → get audio

Env vars:
    TELEGRAM_TOKEN  — Bot token from @BotFather
    TRITON_URL      — Triton gRPC endpoint (default: localhost:8001)

To disable a language: comment out its entry in LANGUAGES below.
"""

import io
import logging
import os

import numpy as np
import tritonclient.grpc as grpcclient
from scipy.io import wavfile
from telegram import BotCommand, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TRITON_URL = os.environ.get("TRITON_URL", "localhost:8001")
SAMPLE_RATE = 22050

# ── Languages ─────────────────────────────────────────────────────────
# Comment out a line to completely hide that language from users.
LANGUAGES = {
    "ky": {"label": "Кыргызча", "preproc": "vits-ky-preprocessing", "synth": "vits-ky-synthesis"},
    "ru": {"label": "Русский",  "preproc": "vits-ru-preprocessing", "synth": "vits-ru-synthesis"},
}

# ── Voices ────────────────────────────────────────────────────────────
# Format: key → (display_label, lang, sid, tid, lid)
# Only voices whose lang is in LANGUAGES above are shown to users.
_ALL_VOICES = {
    # Kyrgyz voices
    "timur_neutral_ky":     ("Тимур (нейтрал)",    "ky", 0, 0, 0),
    "timur_strict_ky":      ("Тимур (строгий)",    "ky", 0, 1, 0),
    "timur_friendly_ky":    ("Тимур (дружеский)",  "ky", 0, 2, 0),
    "aiganysh_neutral_ky":  ("Айганыш (нейтрал)",  "ky", 1, 0, 0),
    "aiganysh_strict_ky":   ("Айганыш (строгий)",  "ky", 1, 1, 0),
    "aiganysh_friendly_ky": ("Айганыш (дружеский)", "ky", 1, 2, 0),

    # Russian voices
    "timur_neutral_ru":     ("Тимур (нейтрал)",    "ru", 0, 0, 1),
    "timur_strict_ru":      ("Тимур (строгий)",    "ru", 0, 1, 1),
    "timur_friendly_ru":    ("Тимур (дружеский)",  "ru", 0, 2, 1),
    "aiganysh_neutral_ru":  ("Айганыш (нейтрал)",  "ru", 1, 0, 1),
    "aiganysh_strict_ru":   ("Айганыш (строгий)",  "ru", 1, 1, 1),
    "aiganysh_friendly_ru": ("Айганыш (дружеский)", "ru", 1, 2, 1),

    # Cloned voices (Russian)
    "alexander_vlasov":     ("Александр Власов",   "ru", 2, 3, 1),
    "artem_lebedev":        ("Артём Лебедев",      "ru", 3, 3, 1),
    "kari":                 ("Кари",               "ru", 4, 3, 1),
    "larisa_actrisa":       ("Лариса",             "ru", 5, 3, 1),
    "nikolay":              ("Николай",            "ru", 6, 3, 1),
    "rina":                 ("Рина",               "ru", 7, 3, 1),
    "victoria":             ("Виктория",           "ru", 8, 3, 1),
}

# Auto-filter: only keep voices whose language is enabled
VOICES = {k: v for k, v in _ALL_VOICES.items() if v[1] in LANGUAGES}

# Pick a sensible default from whatever is enabled
DEFAULT_VOICE = next(iter(VOICES)) if VOICES else None


# ══════════════════════════════════════════════════════════════════════════
# TRITON INFERENCE
# ══════════════════════════════════════════════════════════════════════════

def _triton_client():
    return grpcclient.InferenceServerClient(url=TRITON_URL)


def run_preprocessing(text: str, lang: str, sid: int, tid: int, lid: int):
    """Text → preprocessing tensors + debug strings."""
    client = _triton_client()
    model = LANGUAGES[lang]["preproc"]

    text_np = np.array([text.encode("utf-8")], dtype=object).reshape(1)
    inputs = [
        grpcclient.InferInput("INPUT_TEXT", [1], "BYTES"),
        grpcclient.InferInput("sid", [1], "INT32"),
        grpcclient.InferInput("tid", [1], "INT32"),
        grpcclient.InferInput("lid", [1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(text_np)
    inputs[1].set_data_from_numpy(np.array([sid], dtype=np.int32))
    inputs[2].set_data_from_numpy(np.array([tid], dtype=np.int32))
    inputs[3].set_data_from_numpy(np.array([lid], dtype=np.int32))

    output_names = [
        "input", "emphasis", "sid_out", "tid_out", "lid_out",
        "normalized_text", "accentuated_text", "processed_text",
    ]
    outputs = [grpcclient.InferRequestedOutput(n) for n in output_names]
    resp = client.infer(model_name=model, inputs=inputs, outputs=outputs)

    accentuated = resp.as_numpy("accentuated_text")[0].decode("utf-8")
    processed = resp.as_numpy("processed_text")[0].decode("utf-8")

    # Collect tensors needed by the synthesis model
    tensor_map = {"input": "input", "emphasis": "emphasis",
                  "sid_out": "sid", "tid_out": "tid", "lid_out": "lid"}
    tensors = {synth: resp.as_numpy(preproc) for preproc, synth in tensor_map.items()}

    return accentuated, processed, tensors


def run_synthesis(tensors: dict, lang: str) -> np.ndarray:
    """Preprocessing tensors → trimmed audio waveform (float32)."""
    client = _triton_client()
    model = LANGUAGES[lang]["synth"]

    inputs = []
    for name, arr in tensors.items():
        inp = grpcclient.InferInput(name, list(arr.shape), "INT32")
        inp.set_data_from_numpy(arr)
        inputs.append(inp)

    outputs = [
        grpcclient.InferRequestedOutput("raw_waveform"),
        grpcclient.InferRequestedOutput("y_length"),
    ]
    resp = client.infer(model_name=model, inputs=inputs, outputs=outputs)

    audio = resp.as_numpy("raw_waveform").squeeze()
    y_length = int(resp.as_numpy("y_length")[0])
    return audio[:y_length]


def audio_to_wav_bytes(audio: np.ndarray) -> io.BytesIO:
    """Normalize float32 audio and encode as WAV."""
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════════════════
# TELEGRAM BOT HANDLERS
# ══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _keyboard_rows(buttons: list, cols: int = 2) -> list[list]:
    """Split a flat list of buttons into rows of `cols`."""
    return [buttons[i : i + cols] for i in range(0, len(buttons), cols)]


def _voice_buttons_for_lang(lang: str) -> list[InlineKeyboardButton]:
    """Return voice-selection buttons for one language."""
    return [
        InlineKeyboardButton(label, callback_data=f"voice_{key}")
        for key, (label, vlang, *_) in VOICES.items()
        if vlang == lang
    ]


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show available commands."""
    voice_key = context.user_data.get("voice", DEFAULT_VOICE)
    current = VOICES[voice_key][0] if voice_key and voice_key in VOICES else "—"
    await update.message.reply_text(
        f"🔊 Текущий голос: {current}\n\n"
        "Команды:\n"
        "/start — Тил жана голос тандоо\n"
        "/voice — Голос өзгөртүү\n"
        "/help  — Жардам\n\n"
        "Текст жөнөтүңүз — аудио алуу үчүн."
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Entry point: show language picker (or jump to voices if only one lang)."""
    context.user_data.setdefault("voice", DEFAULT_VOICE)
    langs = list(LANGUAGES.items())

    # Single language → skip straight to voice selection
    if len(langs) == 1:
        lang_key, lang_cfg = langs[0]
        buttons = _voice_buttons_for_lang(lang_key)
        await update.message.reply_text(
            f"🎤 {lang_cfg['label']} — голос тандаңыз:",
            reply_markup=InlineKeyboardMarkup(_keyboard_rows(buttons)),
        )
        return

    # Multiple languages → let user choose
    lang_buttons = [
        InlineKeyboardButton(cfg["label"], callback_data=f"lang_{key}")
        for key, cfg in langs
    ]
    await update.message.reply_text(
        "Тилди тандаңыз / Выберите язык:",
        reply_markup=InlineKeyboardMarkup(_keyboard_rows(lang_buttons)),
    )


async def cmd_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show voice picker for all enabled languages."""
    context.user_data.setdefault("voice", DEFAULT_VOICE)

    all_buttons = []
    for lang_key in LANGUAGES:
        all_buttons.extend(_voice_buttons_for_lang(lang_key))

    await update.message.reply_text(
        "🎤 Голос тандаңыз / Выберите голос:",
        reply_markup=InlineKeyboardMarkup(_keyboard_rows(all_buttons)),
    )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle language and voice selection callbacks."""
    query = update.callback_query
    await query.answer()

    # ── Language selected → show voices for that language ──
    if query.data.startswith("lang_"):
        lang_key = query.data.removeprefix("lang_")
        if lang_key not in LANGUAGES:
            await query.edit_message_text("⚠️ Unknown language.")
            return

        buttons = _voice_buttons_for_lang(lang_key)
        await query.edit_message_text(
            f"🎤 {LANGUAGES[lang_key]['label']} — голос тандаңыз:",
            reply_markup=InlineKeyboardMarkup(_keyboard_rows(buttons)),
        )
        return

    # ── Voice selected → save and wait for text ──
    if query.data.startswith("voice_"):
        voice_key = query.data.removeprefix("voice_")
        if voice_key not in VOICES:
            await query.edit_message_text("⚠️ Unknown voice.")
            return

        context.user_data["voice"] = voice_key
        label = VOICES[voice_key][0]
        await query.edit_message_text(f"✅ {label}\n\nТекст жөнөтүңүз / Отправьте текст.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Synthesize audio for any plain-text message."""
    text = update.message.text.strip()
    if not text:
        return

    voice_key = context.user_data.get("voice", DEFAULT_VOICE)
    if voice_key is None or voice_key not in VOICES:
        await update.message.reply_text("Сначала выберите голос: /start")
        return

    label, lang, sid, tid, lid = VOICES[voice_key]
    await update.message.reply_text(f"⏳ {label}...")

    try:
        accentuated, processed, tensors = run_preprocessing(text, lang, sid, tid, lid)
        await update.message.reply_text(
            f"📝 Accentuated:\n{accentuated}\n\n🔤 Processed:\n{processed}"
        )

        audio = run_synthesis(tensors, lang)
        wav_buf = audio_to_wav_bytes(audio)
        await update.message.reply_voice(
            voice=wav_buf,
            caption=f"{label} | {len(audio) / SAMPLE_RATE:.1f}s",
        )
    except Exception as e:
        logger.error("TTS error: %s", e, exc_info=True)
        await update.message.reply_text(f"❌ {e}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Set TELEGRAM_TOKEN env var")
    if not VOICES:
        raise RuntimeError("No voices enabled — check LANGUAGES config")

    async def post_init(application):
        """Register bot commands so Telegram suggests them when user types '/'."""
        await application.bot.set_my_commands([
            BotCommand("start", "Баштоо / Начать"),
            BotCommand("voice", "Голос тандоо / Выбрать голос"),
            BotCommand("help", "Жардам / Помощь"),
        ])

    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("voice", cmd_voice))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot started (%d languages, %d voices)", len(LANGUAGES), len(VOICES))
    app.run_polling()


if __name__ == "__main__":
    main()
