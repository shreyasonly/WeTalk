# import re
# import language_tool_python
# import librosa
# from noisereduce import reduce_noise
# import soundfile as sf
# import os
# import subprocess
# import base64
# from google.generativeai import GenerativeModel, configure

# # ------------------ GEMINI SETUP ------------------
# API_KEY = "AIzaSyDDlG2xtuTEuYci3tNTkMSPtLue9u3aURI"
# configure(api_key=API_KEY)

# # Function to convert an audio file to WAV format
# def convert_to_wav(input_audio_path, output_audio_path):
#     if not input_audio_path.lower().endswith('.wav'):
#         print("Converting audio to WAV format...")
#         try:
#             subprocess.run(
#                 ["ffmpeg", "-y", "-i", input_audio_path, "-ar", "16000", "-ac", "1", output_audio_path],
#                 check=True
#             )
#             print(f"Audio converted to WAV format: {output_audio_path}")
#         except subprocess.CalledProcessError as e:
#             print(f"Error during audio conversion: {e}")
#             raise
#     else:
#         print("Audio is already in WAV format. Proceeding with format normalization.")
#         subprocess.run(
#             ["ffmpeg", "-y", "-i", input_audio_path, "-ar", "16000", "-ac", "1", output_audio_path],
#             check=True
#         )
#     return output_audio_path

# # Function to reduce noise in an audio file
# def denoise_audio(input_audio_path, output_audio_path):
#     print("Performing noise reduction...")
#     try:
#         audio, sr = librosa.load(input_audio_path, sr=None)
#         noise_profile = audio[:sr * 2]
#         reduced_noise_audio = reduce_noise(y=audio, sr=sr, y_noise=noise_profile)
#         sf.write(output_audio_path, reduced_noise_audio, sr)
#         print("Noise reduction complete.")
#     except Exception as e:
#         print(f"Error during noise reduction: {e}")
#         raise

# # Function to split audio into smaller segments
# def split_audio(input_audio_path, segment_duration=300):
#     print("Splitting audio into smaller segments...")
#     audio, sr = librosa.load(input_audio_path, sr=None)
#     total_duration = librosa.get_duration(y=audio, sr=sr)
#     segments = []

#     for start in range(0, int(total_duration), segment_duration):
#         end = min(start + segment_duration, int(total_duration))
#         segment = audio[start * sr:end * sr]
#         segment_path = f"segment_{start // segment_duration + 1}.wav"
#         sf.write(segment_path, segment, sr)
#         segments.append(segment_path)

#     print(f"Audio split into {len(segments)} segments.")
#     return segments

# # Function to process transcription for special commands
# def process_transcription(transcription):
#     commands = {
#         r"\bfull stop\b": ".",
#         r"\bPull stop\b": ".",
#         r"\bnext para\b": "\n",
#         r"\bnext paragraph\b": "\n",
#         r"\bcomma\b": ",",
#         r"\bsemicolon\b": ";",
#         r"\bcolon\b": ":"
#     }
#     for command, symbol in commands.items():
#         transcription = re.sub(command, symbol, transcription, flags=re.IGNORECASE)
#     return transcription

# # Function to correct grammar in transcription
# def correct_grammar(transcription):
#     tool = language_tool_python.LanguageTool("en-US")
#     matches = tool.check(transcription)
#     corrected_text = language_tool_python.utils.correct(transcription, matches)
#     return corrected_text

# # ----------- GEMINI TRANSCRIPTION FUNCTION ------------
# def encode_audio_base64(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# def transcribe_gemini(audio_path, model_name):
#     print(f"Transcribing with Gemini {model_name}...")
#     audio_base64 = encode_audio_base64(audio_path)
#     gemini_model = GenerativeModel(model_name)
#     response = gemini_model.generate_content([
#         {
#             "mime_type": "audio/wav",
#             "data": audio_base64
#         },
#         {
#             "text": "Transcribe this audio segment."
#         }
#     ])
#     return response.text.strip()

# # ---------------- MAIN FUNCTION ------------------------
# def main():
#     print("Choose transcription model:")
#     print("1. Gemini 1.5 Flash (cloud-based)")
#     choice = input("Enter 1: ").strip()

#     if choice != "1":
#         print("Invalid choice. Exiting.")
#         return

#     model_name = "gemini-1.5-flash"
#     print(f"Using {model_name} model. Make sure you have access enabled.")

#     audio_file = input("Enter the path to your audio file: ").strip()
#     temp_wav_file = "temp_audio.wav"
#     denoised_audio_file = "denoised_audio.wav"

#     try:
#         audio_file = convert_to_wav(audio_file, temp_wav_file)
#         denoise_audio(audio_file, denoised_audio_file)
#         segments = split_audio(denoised_audio_file, segment_duration=300)

#         combined_transcription = ""

#         for i, segment in enumerate(segments):
#             print(f"Processing segment {i + 1}/{len(segments)}: {segment}")
#             transcription = transcribe_gemini(segment, model_name)
#             print(f"Raw transcription for segment {i + 1}:")
#             print(transcription)

#             processed_transcription = process_transcription(transcription)
#             final_transcription = correct_grammar(processed_transcription)
#             combined_transcription += final_transcription + "\n"

#             os.remove(segment)

#         print("\nFinal Combined Transcription:")
#         print(combined_transcription)

#         output_file = "final_transcription_output.txt"
#         with open(output_file, "w") as file:
#             file.write(combined_transcription)

#         print(f"\nFinal transcription saved to: {output_file}")

#     except FileNotFoundError:
#         print("Error: The specified file was not found. Please check the file path and try again.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         if os.path.exists(temp_wav_file):
#             os.remove(temp_wav_file)
#         if os.path.exists(denoised_audio_file):
#             os.remove(denoised_audio_file)

# if __name__ == "__main__":
#     main()
# import re
# import language_tool_python
# import librosa
# from noisereduce import reduce_noise
# import soundfile as sf
# import os
# import base64
# from google.generativeai import GenerativeModel, configure
# from pydub import AudioSegment

# # ------------------ GEMINI SETUP ------------------
# API_KEY = "AIzaSyDDlG2xtuTEuYci3tNTkMSPtLue9u3aURI"  # Replace with your own API key securely
# configure(api_key=API_KEY)

# # Function to convert audio to WAV (16kHz, mono) using pydub
# def convert_to_wav(input_audio_path, output_audio_path):
#     print("Converting audio to WAV (16kHz, mono) using pydub...")
#     audio = AudioSegment.from_file(input_audio_path)
#     audio = audio.set_frame_rate(16000).set_channels(1)
#     audio.export(output_audio_path, format="wav")
#     print(f"Audio converted and saved as: {output_audio_path}")
#     return output_audio_path

# # Function to reduce noise
# def denoise_audio(input_audio_path, output_audio_path):
#     print("Performing noise reduction...")
#     try:
#         audio, sr = librosa.load(input_audio_path, sr=None)
#         noise_profile = audio[:sr * 2]
#         reduced_noise_audio = reduce_noise(y=audio, sr=sr, y_noise=noise_profile)
#         sf.write(output_audio_path, reduced_noise_audio, sr)
#         print("Noise reduction complete.")
#     except Exception as e:
#         print(f"Error during noise reduction: {e}")
#         raise

# # Split audio into smaller chunks
# def split_audio(input_audio_path, segment_duration=300):
#     print("Splitting audio into smaller segments...")
#     audio, sr = librosa.load(input_audio_path, sr=None)
#     total_duration = librosa.get_duration(y=audio, sr=sr)
#     segments = []

#     for start in range(0, int(total_duration), segment_duration):
#         end = min(start + segment_duration, int(total_duration))
#         segment = audio[start * sr:end * sr]
#         segment_path = f"segment_{start // segment_duration + 1}.wav"
#         sf.write(segment_path, segment, sr)
#         segments.append(segment_path)

#     print(f"Audio split into {len(segments)} segments.")
#     return segments

# # Replace spoken commands with punctuation
# def process_transcription(transcription):
#     commands = {
#         r"\bfull stop\b": ".",
#         r"\bPull stop\b": ".",
#         r"\bnext para\b": "\n",
#         r"\bnext paragraph\b": "\n",
#         r"\bcomma\b": ",",
#         r"\bsemicolon\b": ";",
#         r"\bcolon\b": ":"
#     }
#     for command, symbol in commands.items():
#         transcription = re.sub(command, symbol, transcription, flags=re.IGNORECASE)
#     return transcription

# # Fix grammar
# def correct_grammar(transcription):
#     tool = language_tool_python.LanguageTool("en-US")
#     matches = tool.check(transcription)
#     return language_tool_python.utils.correct(transcription, matches)

# # Encode audio to base64
# def encode_audio_base64(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# # Transcribe using Gemini
# def transcribe_gemini(audio_path, model_name):
#     print(f"Transcribing with Gemini {model_name}...")
#     audio_base64 = encode_audio_base64(audio_path)
#     gemini_model = GenerativeModel(model_name)
#     response = gemini_model.generate_content([
#         {
#             "mime_type": "audio/wav",
#             "data": audio_base64
#         },
#         {
#             "text": "Transcribe this audio segment."
#         }
#     ])
#     return response.text.strip()

# # ---------------- MAIN FUNCTION ------------------------
# def main():
#     print("Choose transcription model:")
#     print("1. Gemini 1.5 Flash (cloud-based)")
#     choice = input("Enter 1: ").strip()

#     if choice != "1":
#         print("Invalid choice. Exiting.")
#         return

#     model_name = "gemini-1.5-flash"
#     print(f"Using {model_name} model. Make sure you have access enabled.")

#     audio_file = input("Enter the path to your audio file: ").strip()
#     temp_wav_file = "temp_audio.wav"
#     denoised_audio_file = "denoised_audio.wav"

#     try:
#         audio_file = convert_to_wav(audio_file, temp_wav_file)
#         denoise_audio(audio_file, denoised_audio_file)
#         segments = split_audio(denoised_audio_file, segment_duration=300)

#         combined_transcription = ""

#         for i, segment in enumerate(segments):
#             print(f"Processing segment {i + 1}/{len(segments)}: {segment}")
#             transcription = transcribe_gemini(segment, model_name)
#             print(f"Raw transcription for segment {i + 1}:\n{transcription}\n")

#             processed_transcription = process_transcription(transcription)
#             final_transcription = correct_grammar(processed_transcription)
#             combined_transcription += final_transcription + "\n"

#             os.remove(segment)

#         print("\nFinal Combined Transcription:\n")
#         print(combined_transcription)

#         output_file = "final_transcription_output.txt"
#         with open(output_file, "w") as file:
#             file.write(combined_transcription)

#         print(f"\nFinal transcription saved to: {output_file}")

#     except FileNotFoundError:
#         print("Error: File not found. Check the file path.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         for f in [temp_wav_file, denoised_audio_file]:
#             if os.path.exists(f):
#                 os.remove(f)

# if __name__ == "__main__":
#     main()

# gemini_transcriber.py
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

# GEMINI SETUP
API_KEY = "AIzaSyDDlG2xtuTEuYci3tNTkMSPtLue9u3aURI"
configure(api_key=API_KEY)

# Helper Functions
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

def transcribe_gemini(audio_path):
    audio_base64 = encode_audio_base64(audio_path)
    gemini_model = GenerativeModel("gemini-1.5-flash")
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

# Streamlit App UI
st.set_page_config(page_title="Gemini Audio Transcriber", layout="centered")

st.markdown('<h1 style="text-align: center; color: #007acc;">Gemini Audio Transcription</h1>', unsafe_allow_html=True)
audio_file = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a"])

if st.button("Transcribe"):
    if not audio_file:
        st.warning("Upload an audio file first.")
    else:
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_input:
                temp_input.write(audio_file.read())
                temp_input_path = temp_input.name

            temp_wav = "converted.wav"
            denoised_wav = "denoised.wav"

            convert_to_wav(temp_input_path, temp_wav)
            denoise_audio(temp_wav, denoised_wav)
            segments = split_audio(denoised_wav)

            combined = ""
            for i, segment in enumerate(segments):
                st.info(f"Transcribing segment {i+1}/{len(segments)}...")
                raw = transcribe_gemini(segment)
                processed = process_transcription(raw)
                final = correct_grammar(processed)
                combined += final + "\n"
                os.remove(segment)

            st.success("Done.")
            st.text_area("Transcription Result", combined, height=300)

            st.download_button("Download", combined, file_name="gemini_transcription.txt")

            for file in [temp_input_path, temp_wav, denoised_wav]:
                if os.path.exists(file):
                    os.remove(file)
