import whisper
import re
import base64
import os
import subprocess
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from noisereduce import reduce_noise
import language_tool_python
from google.generativeai import GenerativeModel, configure

# ---- Gemini API Setup ----
configure(api_key="AIzaSyCzRlHcGEi-VQRaPkWLaK_pFP7XO5BWPfU")
gemini_model = GenerativeModel("gemini-1.5-flash")


def convert_to_wav(input_audio_path, output_audio_path):
    if not input_audio_path.lower().endswith(".wav"):
        subprocess.run(["ffmpeg", "-y", "-i", input_audio_path, "-ar", "16000", "-ac", "1", output_audio_path], check=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-i", input_audio_path, "-ar", "16000", "-ac", "1", output_audio_path], check=True)
    return output_audio_path


def denoise_audio(input_audio_path, output_audio_path):
    audio, sr = librosa.load(input_audio_path, sr=None)
    noise_profile = audio[:sr * 2]
    reduced_noise_audio = reduce_noise(y=audio, sr=sr, y_noise=noise_profile)
    sf.write(output_audio_path, reduced_noise_audio, sr)


def split_audio(input_audio_path, segment_duration=300):
    audio, sr = librosa.load(input_audio_path, sr=None)
    total_duration = librosa.get_duration(y=audio, sr=sr)
    segments = []

    for start in range(0, int(total_duration), segment_duration):
        end = min(start + segment_duration, int(total_duration))
        segment = audio[start * sr:end * sr]
        segment_path = f"temp/segment_{start // segment_duration + 1}.wav"
        sf.write(segment_path, segment, sr)
        segments.append(segment_path)

    return segments


def process_transcription(transcription):
    commands = {
        r"\bfull stop\b": ".",
        r"\bPull stop\b": ".",
        r"\bnext para\b": "\n",
        r"\bnext paragraph\b": "\n",
        r"\bcomma\b": ",",
        r"\bsemicolon\b": ";",
        r"\bcolon\b": ":"
    }
    for pattern, symbol in commands.items():
        transcription = re.sub(pattern, symbol, transcription, flags=re.IGNORECASE)
    return transcription


def correct_grammar(text):
    tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)


def encode_audio_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def transcribe_gemini(audio_path):
    audio_base64 = encode_audio_base64(audio_path)
    response = gemini_model.generate_content([
        {"mime_type": "audio/wav", "data": audio_base64},
        {"text": "Transcribe this audio segment."}
    ])
    return response.text.strip()
