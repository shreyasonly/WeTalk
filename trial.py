# trial.py
# This script is a Streamlit app for audio transcription using Gemini and Whisper models.

import streamlit as st
import tempfile
import os
import re
import base64
import librosa
import soundfile as sf
import language_tool_python
from noisereduce import reduce_noise
from google.generativeai import GenerativeModel, configure
from pydub import AudioSegment
import whisper

# ------------------ GEMINI SETUP ------------------
API_KEY = "AIzaSyDDlG2xtuTEuYci3tNTkMSPtLue9u3aURI"  # Replace with your key
configure(api_key=API_KEY)

# ------------------ HELPER FUNCTIONS ------------------

def convert_to_wav(input_audio_path, output_audio_path):
    audio = AudioSegment.from_file(input_audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_audio_path, format="wav")
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
        segment_path = f"segment_{start // segment_duration + 1}.wav"
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
    for command, symbol in commands.items():
        transcription = re.sub(command, symbol, transcription, flags=re.IGNORECASE)
    return transcription

def correct_grammar(transcription):
    tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(transcription)
    return language_tool_python.utils.correct(transcription, matches)

def encode_audio_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def transcribe_gemini(audio_path, model_name):
    audio_base64 = encode_audio_base64(audio_path)
    gemini_model = GenerativeModel(model_name)
    response = gemini_model.generate_content([
        {
            "mime_type": "audio/wav",
            "data": audio_base64
        },
        {
            "text": "Transcribe this audio segment."
        }
    ])
    return response.text.strip()

def transcribe_whisper(audio_path, model_name):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, fp16=False)
    return result["text"].strip()

# ------------------ STREAMLIT UI ------------------

st.set_page_config(page_title="Audio Transcriber", layout="centered")
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }
        .title {
            font-size: 36px;
            color: #007acc;
            text-align: center;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #007acc;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
            border: none;
        }
        .stButton>button:hover {
            background-color: #005e99;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Audio Transcription App</div>', unsafe_allow_html=True)

model_option = st.selectbox("Choose Transcription Model", ["Gemini 1.5 Flash", "Whisper (medium.en)"])
audio_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "m4a"])

if st.button("Transcribe", key="transcribe_btn"):
    if audio_file is None:
        st.warning("Please upload an audio file before clicking Transcribe.")
    else:
        with st.spinner("Processing audio..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_input:
                    temp_input.write(audio_file.read())
                    temp_input_path = temp_input.name

                temp_wav = "converted.wav"
                denoised_wav = "denoised.wav"

                convert_to_wav(temp_input_path, temp_wav)
                denoise_audio(temp_wav, denoised_wav)
                segments = split_audio(denoised_wav, segment_duration=300)

                combined_transcription = ""
                for i, segment in enumerate(segments):
                    st.info(f"Transcribing segment {i + 1}/{len(segments)}...")
                    if model_option == "Gemini 1.5 Flash":
                        raw_text = transcribe_gemini(segment, "gemini-1.5-flash")
                    else:
                        raw_text = transcribe_whisper(segment, "medium.en")

                    processed = process_transcription(raw_text)
                    final = correct_grammar(processed)
                    combined_transcription += final + "\n"
                    os.remove(segment)

                st.success("Transcription completed.")
                st.text_area("Final Transcription:", combined_transcription, height=300)

                with open("final_transcription_output.txt", "w") as f:
                    f.write(combined_transcription)
                st.download_button("Download Transcription", data=combined_transcription, file_name="final_transcription_output.txt")

            except Exception as e:
                st.error(f"Error during transcription: {e}")

            finally:
                for f in [temp_input_path, temp_wav, denoised_wav]:
                    if os.path.exists(f):
                        os.remove(f)

