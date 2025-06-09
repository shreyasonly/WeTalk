import re
import language_tool_python
import librosa
from noisereduce import reduce_noise
import soundfile as sf
import os
import base64
from google.generativeai import GenerativeModel, configure
from pydub import AudioSegment
import whisper

# ------------------ GEMINI SETUP ------------------
API_KEY = "AIzaSyDDlG2xtuTEuYci3tNTkMSPtLue9u3aURI"  # Replace with your own API key securely
configure(api_key=API_KEY)

# Function to convert audio to WAV (16kHz, mono) using pydub
def convert_to_wav(input_audio_path, output_audio_path):
    print("Converting audio to WAV (16kHz, mono) using pydub...")
    audio = AudioSegment.from_file(input_audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_audio_path, format="wav")
    print(f"Audio converted and saved as: {output_audio_path}")
    return output_audio_path

# Function to reduce noise
def denoise_audio(input_audio_path, output_audio_path):
    print("Performing noise reduction...")
    try:
        audio, sr = librosa.load(input_audio_path, sr=None)
        noise_profile = audio[:sr * 2]
        reduced_noise_audio = reduce_noise(y=audio, sr=sr, y_noise=noise_profile)
        sf.write(output_audio_path, reduced_noise_audio, sr)
        print("Noise reduction complete.")
    except Exception as e:
        print(f"Error during noise reduction: {e}")
        raise

# Split audio into smaller chunks
def split_audio(input_audio_path, segment_duration=300):
    print("Splitting audio into smaller segments...")
    audio, sr = librosa.load(input_audio_path, sr=None)
    total_duration = librosa.get_duration(y=audio, sr=sr)
    segments = []

    for start in range(0, int(total_duration), segment_duration):
        end = min(start + segment_duration, int(total_duration))
        segment = audio[start * sr:end * sr]
        segment_path = f"segment_{start // segment_duration + 1}.wav"
        sf.write(segment_path, segment, sr)
        segments.append(segment_path)

    print(f"Audio split into {len(segments)} segments.")
    return segments

# Replace spoken commands with punctuation
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

# Fix grammar
def correct_grammar(transcription):
    tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(transcription)
    return language_tool_python.utils.correct(transcription, matches)

# Encode audio to base64 for Gemini
def encode_audio_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Transcribe using Gemini
def transcribe_gemini(audio_path, model_name):
    print(f"Transcribing with Gemini {model_name}...")
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

# Transcribe using Whisper
def transcribe_whisper(audio_path, model_name):
    print(f"Transcribing with Whisper {model_name}...")
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, fp16=False)
    return result["text"].strip()

# ---------------- MAIN FUNCTION ------------------------
def main():
    print("Choose transcription model:")
    print("1. Gemini 1.5 Flash (cloud-based)")
    print("2. Whisper (local, medium model)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        model_name = "gemini-1.5-flash"
        print(f"Using {model_name} model. Make sure you have access enabled.")
        transcribe_func = transcribe_gemini
    elif choice == "2":
        model_name = "medium.en"
        print(f"Using Whisper {model_name} model.")
        transcribe_func = transcribe_whisper
    else:
        print("Invalid choice. Exiting.")
        return

    audio_file = input("Enter the path to your audio file: ").strip()
    temp_wav_file = "temp_audio.wav"
    denoised_audio_file = "denoised_audio.wav"

    try:
        audio_file = convert_to_wav(audio_file, temp_wav_file)
        denoise_audio(audio_file, denoised_audio_file)
        segments = split_audio(denoised_audio_file, segment_duration=300)

        combined_transcription = ""

        for i, segment in enumerate(segments):
            print(f"Processing segment {i + 1}/{len(segments)}: {segment}")
            transcription = transcribe_func(segment, model_name)
            print(f"Raw transcription for segment {i + 1}:\n{transcription}\n")

            processed_transcription = process_transcription(transcription)
            final_transcription = correct_grammar(processed_transcription)
            combined_transcription += final_transcription + "\n"

            os.remove(segment)

        print("\nFinal Combined Transcription:\n")
        print(combined_transcription)

        output_file = "final_transcription_output.txt"
        with open(output_file, "w") as file:
            file.write(combined_transcription)

        print(f"\nFinal transcription saved to: {output_file}")

    except FileNotFoundError:
        print("Error: File not found. Check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        for f in [temp_wav_file, denoised_audio_file]:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    main()