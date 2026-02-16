#!/usr/bin/env python3
"""
Telegram Bot for VITS TTS via Triton Ensemble.

The ensemble handles everything: normalization → accentuation → phonemization → synthesis.
The bot just sends raw text + speaker/tone selection.

Usage:
    pip install python-telegram-bot tritonclient[grpc]
    python telegram_bot.py

Commands:
    /start  - Welcome message
    /model  - Choose speaker + tone, then send text
"""

import io
import logging

import numpy as np
import tritonclient.grpc as grpcclient
from scipy.io import wavfile

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ---- Config ----

TELEGRAM_TOKEN = "8375805544:AAFjWqq8-QBgVg8X77l2BH8KGJ6kLCH_Gpo"
TRITON_URL = "localhost:8001"
MODEL_NAME = "vits-ky-6-ensemble"
SAMPLE_RATE = 22050
HOP_LENGTH = 256

# ---- Triton inference ----


def run_tts(text, speaker_id, tone_id):
    """Send raw text to Triton ensemble and get audio back."""
    client = grpcclient.InferenceServerClient(url=TRITON_URL)

    text_np = np.array([text.encode("utf-8")], dtype=object).reshape(1)
    sid = np.array([speaker_id], dtype=np.int32)
    tid = np.array([tone_id], dtype=np.int32)

    inputs = [
        grpcclient.InferInput("INPUT_TEXT", [1], "BYTES"),
        grpcclient.InferInput("sid", [1], "INT32"),
        grpcclient.InferInput("tid", [1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(text_np)
    inputs[1].set_data_from_numpy(sid)
    inputs[2].set_data_from_numpy(tid)

    outputs = [
        grpcclient.InferRequestedOutput("raw_waveform"),
        grpcclient.InferRequestedOutput("y_length"),
        grpcclient.InferRequestedOutput("normalized_text"),
        grpcclient.InferRequestedOutput("accentuated_text"),
        grpcclient.InferRequestedOutput("phonemized_text"),
    ]

    response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

    audio = response.as_numpy("raw_waveform")
    y_length = response.as_numpy("y_length")

    normalized = response.as_numpy("normalized_text")[0].decode("utf-8")
    accentuated = response.as_numpy("accentuated_text")[0].decode("utf-8")
    phonemized = response.as_numpy("phonemized_text")[0].decode("utf-8")

    actual_length = int(y_length[0]) * HOP_LENGTH
    audio = audio.squeeze()[:actual_length]

    return audio, normalized, accentuated, phonemized


def audio_to_wav_bytes(audio):
    """Convert numpy audio to WAV bytes in memory."""
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    audio_int16 = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, audio_int16)
    buf.seek(0)
    return buf


# ---- Voice config ----

VOICES = {
    "timur_neutral":     {"sid": 0, "tid": 0, "label": "Timur — Neutral"},
    "timur_strict":      {"sid": 0, "tid": 1, "label": "Timur — Strict"},
    "timur_friendly":    {"sid": 0, "tid": 2, "label": "Timur — Friendly"},
    "aiganysh_neutral":  {"sid": 1, "tid": 0, "label": "Aiganysh — Neutral"},
    "aiganysh_strict":   {"sid": 1, "tid": 1, "label": "Aiganysh — Strict"},
    "aiganysh_friendly": {"sid": 1, "tid": 2, "label": "Aiganysh — Friendly"},
}

# ---- Bot handlers ----

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "MBank Кыргызча TTS Bot\n\n"
        "/model — үн тандоо, андан кийин текст жөнөтүңүз."
    )


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("Timur Neutral", callback_data="timur_neutral"),
            InlineKeyboardButton("Timur Strict", callback_data="timur_strict"),
            InlineKeyboardButton("Timur Friendly", callback_data="timur_friendly"),
        ],
        [
            InlineKeyboardButton("Aiganysh Neutral", callback_data="aiganysh_neutral"),
            InlineKeyboardButton("Aiganysh Strict", callback_data="aiganysh_strict"),
            InlineKeyboardButton("Aiganysh Friendly", callback_data="aiganysh_friendly"),
        ],
    ]
    await update.message.reply_text(
        "Choose a voice:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def voice_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    voice_key = query.data
    if voice_key not in VOICES:
        await query.edit_message_text("Unknown voice. Try /model again.")
        return

    voice = VOICES[voice_key]
    context.user_data["sid"] = voice["sid"]
    context.user_data["tid"] = voice["tid"]
    context.user_data["voice_label"] = voice["label"]

    await query.edit_message_text(
        f"Voice: {voice['label']}\n\nNow send me the text you want to synthesize."
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if "sid" not in context.user_data:
        await update.message.reply_text("Please choose a voice first with /model")
        return

    text = update.message.text.strip()
    if not text:
        await update.message.reply_text("Please send some text.")
        return

    sid = context.user_data["sid"]
    tid = context.user_data["tid"]
    voice_label = context.user_data["voice_label"]

    await update.message.reply_text(f"Generating with {voice_label}...")

    try:
        audio, normalized, accentuated, phonemized = run_tts(text, sid, tid)
        wav_buf = audio_to_wav_bytes(audio)

        caption = (
            f"{voice_label}\n"
            f"Normalized: {normalized}\n"
            f"Accentuated: {accentuated}\n"
            f"Phonemized: {phonemized}"
        )

        await update.message.reply_voice(
            voice=wav_buf,
            caption=caption,
        )
    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        await update.message.reply_text(f"Error: {e}")


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CallbackQueryHandler(voice_selected))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot started. Waiting for messages...")
    app.run_polling()


if __name__ == "__main__":
    main()
