# import streamlit as st
# import re
# import language_tool_python
# import librosa
# from noisereduce import reduce_noise
# import soundfile as sf
# import os
# import base64
# import tempfile
# from google.generativeai import GenerativeModel, configure
# from pydub import AudioSegment
# import whisper

# # Custom CSS for sleek, minimalistic design with hover effects
# st.markdown("""
#     <style>
#     .stButton>button {
#         background-color: #1E1E1E;
#         color: white;
#         border: none;
#         padding: 10px 20px;
#         border-radius: 5px;
#         font-size: 16px;
#         transition: all 0.3s ease;
#     }
#     .stButton>button:hover {
#         background-color: #3C3C3C;
#         transform: translateY(-2px);
#         box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
#     }
#     .stRadio label {
#         font-size: 16px;
#         color: #333;
#         padding: 5px;
#         transition: color 0.3s ease;
#     }
#     .stRadio label:hover {
#         color: #007BFF;
#     }
#     .stTextArea textarea {
#         border: 1px solid #DDD;
#         border-radius: 5px;
#         font-size: 14px;
#         background-color: #F9F9F9;
#     }
#     .container {
#         max-width: 800px;
#         margin: auto;
#         padding: 20px;
#     }
#     h1, h2, h3 {
#         color: #1E1E1E;
#         font-family: 'Arial', sans-serif;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # ------------------ GEMINI SETUP ------------------
# API_KEY = "AIzaSyDDlG2xtuTEuYci3tNTkMSPtLue9u3aURI"  # Replace with your own API key securely
# configure(api_key=API_KEY)

# # Function to convert audio to WAV (16kHz, mono) using pydub
# def convert_to_wav(input_audio_path, output_audio_path):
#     st.write("Converting audio to WAV (16kHz, mono)...")
#     audio = AudioSegment.from_file(input_audio_path)
#     audio = audio.set_frame_rate(16000).set_channels(1)
#     audio.export(output_audio_path, format="wav")
#     st.write(f"Audio converted and saved as: {output_audio_path}")
#     return output_audio_path

# # Function to reduce noise
# def denoise_audio(input_audio_path, output_audio_path):
#     st.write("Performing noise reduction...")
#     try:
#         audio, sr = librosa.load(input_audio_path, sr=None)
#         noise_profile = audio[:sr * 2]
#         reduced_noise_audio = reduce_noise(y=audio, sr=sr, y_noise=noise_profile)
#         sf.write(output_audio_path, reduced_noise_audio, sr)
#         st.write("Noise reduction complete.")
#     except Exception as e:
#         st.error(f"Error during noise reduction: {e}")
#         raise

# # Split audio into smaller chunks
# def split_audio(input_audio_path, segment_duration=300):
#     st.write("Splitting audio into smaller segments...")
#     audio, sr = librosa.load(input_audio_path, sr=None)
#     total_duration = librosa.get_duration(y=audio, sr=sr)
#     segments = []

#     for start in range(0, int(total_duration), segment_duration):
#         end = min(start + segment_duration, int(total_duration))
#         segment = audio[start * sr:end * sr]
#         segment_path = os.path.join(tempfile.gettempdir(), f"segment_{start // segment_duration + 1}.wav")
#         sf.write(segment_path, segment, sr)
#         segments.append(segment_path)

#     st.write(f"Audio split into {len(segments)} segments.")
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

# # Encode audio to base64 for Gemini
# def encode_audio_base64(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# # Transcribe using Gemini
# def transcribe_gemini(audio_path, model_name):
#     st.write(f"Transcribing with Gemini {model_name}...")
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

# # Transcribe using Whisper
# def transcribe_whisper(audio_path, model_name):
#     st.write(f"Transcribing with Whisper {model_name}...")
#     model = whisper.load_model(model_name)
#     result = model.transcribe(audio_path, fp16=False)
#     return result["text"].strip()

# # ---------------- MAIN STREAMLIT APP ------------------------
# def main():
#     st.title("Audio Transcription App")
#     st.markdown('<div class="container">', unsafe_allow_html=True)
#     st.subheader("Upload an audio file and select a transcription model")

#     # File uploader
#     audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg"])

#     # Model selection
#     model_choice = st.radio(
#         "Select Transcription Model",
#         options=["Gemini 1.5 Flash (cloud-based)", "Whisper (local, medium model)"],
#         index=0
#     )

#     if model_choice == "Gemini 1.5 Flash (cloud-based)":
#         model_name = "gemini-1.5-flash"
#         transcribe_func = transcribe_gemini
#     else:
#         model_name = "medium.en"
#         transcribe_func = transcribe_whisper

#     if audio_file is not None and st.button("Transcribe Audio"):
#         with st.spinner("Processing audio..."):
#             # Create temporary files
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
#                 temp_wav_path = temp_wav_file.name
#                 temp_wav_file.write(audio_file.read())

#             denoised_audio_path = os.path.join(tempfile.gettempdir(), "denoised_audio.wav")

#             try:
#                 # Process audio
#                 convert_to_wav(temp_wav_path, temp_wav_path)
#                 denoise_audio(temp_wav_path, denoised_audio_path)
#                 segments = split_audio(denoised_audio_path)

#                 combined_transcription = ""
#                 progress_bar = st.progress(0)
#                 for i, segment in enumerate(segments):
#                     st.write(f"Processing segment {i + 1}/{len(segments)}: {segment}")
#                     transcription = transcribe_func(segment, model_name)
#                     st.write(f"Raw transcription for segment {i + 1}:\n{transcription}\n")

#                     processed_transcription = process_transcription(transcription)
#                     final_transcription = correct_grammar(processed_transcription)
#                     combined_transcription += final_transcription + "\n"

#                     os.remove(segment)
#                     progress_bar.progress((i + 1) / len(segments))

#                 # Display final transcription
#                 st.subheader("Final Transcription")
#                 st.text_area("Transcription Output", combined_transcription, height=300)

#                 # Save transcription to a temporary file for download
#                 output_file = os.path.join(tempfile.gettempdir(), "final_transcription_output.txt")
#                 with open(output_file, "w") as file:
#                     file.write(combined_transcription)

#                 # Provide download button
#                 with open(output_file, "rb") as file:
#                     st.download_button(
#                         label="Download Transcription",
#                         data=file,
#                         file_name="final_transcription_output.txt",
#                         mime="text/plain"
#                     )

#                 st.success(f"Transcription complete! Saved to: {output_file}")

#             except Exception as e:
#                 st.error(f"An error occurred: {e}")
#             finally:
#                 # Clean up temporary files
#                 for f in [temp_wav_path, denoised_audio_path]:
#                     if os.path.exists(f):
#                         os.remove(f)

#     st.markdown('</div>', unsafe_allow_html=True)

# if __name__ == "__main__":
# #     main()
# import streamlit as st
# import re
# import language_tool_python
# import librosa
# from noisereduce import reduce_noise
# import soundfile as sf
# import os
# import base64
# import tempfile
# from google.generativeai import GenerativeModel, configure
# from pydub import AudioSegment
# import whisper

# # Custom CSS for a polished, professional design
# st.markdown("""
#     <style>
#     .container {
#         max-width: 900px;
#         margin: auto;
#         padding: 30px;
#         background-color: #FFFFFF;  /* White background */
#         border-radius: 10px;
#         box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
#     }
#     .header {
#         font-size: 28px;
#         font-weight: bold;
#         color: #1E1E1E;
#         margin-bottom: 10px;
#         text-align: center;
#     }
#     .subheader {
#         font-size: 18px;
#         color: #555555;
#         margin-bottom: 20px;
#         text-align: center;
#     }
#     .stButton>button {
#         background-color: #007BFF;
#         color: white;
#         border: none;
#         padding: 10px 20px;
#         border-radius: 5px;
#         font-size: 16px;
#         margin: 5px;
#     }
#     .stButton>button:hover {
#         background-color: #0056b3;
#     }
#     .stSelectbox {
#         max-width: 300px;
#         margin: 10px 0;
#     }
#     .stTextArea textarea {
#         border: 1px solid #DDDDDD;
#         border-radius: 5px;
#         font-size: 14px;
#         background-color: #F9F9F9;
#     }
#     .file-details {
#         font-size: 14px;
#         color: #666666;
#         margin: 10px 0;
#     }
#     .status {
#         font-size: 16px;
#         font-weight: bold;
#         margin: 10px 0;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # ------------------ GEMINI SETUP ------------------
# API_KEY = "AIzaSyDDlG2xtuTEuYci3tNTkMSPtLue9u3aURI"  # Replace with your own API key securely
# configure(api_key=API_KEY)

# # Function to convert audio to WAV (16kHz, mono) using pydub
# def convert_to_wav(input_audio_path, output_audio_path):
#     st.write("Converting audio to WAV (16kHz, mono)...")
#     audio = AudioSegment.from_file(input_audio_path)
#     audio = audio.set_frame_rate(16000).set_channels(1)
#     audio.export(output_audio_path, format="wav")
#     st.write(f"Audio converted and saved as: {output_audio_path}")
#     return output_audio_path

# # Function to reduce noise
# def denoise_audio(input_audio_path, output_audio_path):
#     st.write("Performing noise reduction...")
#     try:
#         audio, sr = librosa.load(input_audio_path, sr=None)
#         noise_profile = audio[:sr * 2]
#         reduced_noise_audio = reduce_noise(y=audio, sr=sr, y_noise=noise_profile)
#         sf.write(output_audio_path, reduced_noise_audio, sr)
#         st.write("Noise reduction complete.")
#     except Exception as e:
#         st.error(f"Error during noise reduction: {e}")
#         raise

# # Split audio into smaller chunks
# def split_audio(input_audio_path, segment_duration=300):
#     st.write("Splitting audio into smaller segments...")
#     audio, sr = librosa.load(input_audio_path, sr=None)
#     total_duration = librosa.get_duration(y=audio, sr=sr)
#     segments = []

#     for start in range(0, int(total_duration), segment_duration):
#         end = min(start + segment_duration, int(total_duration))
#         segment = audio[start * sr:end * sr]
#         segment_path = os.path.join(tempfile.gettempdir(), f"segment_{start // segment_duration + 1}.wav")
#         sf.write(segment_path, segment, sr)
#         segments.append(segment_path)

#     st.write(f"Audio split into {len(segments)} segments.")
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

# # Encode audio to base64 for Gemini
# def encode_audio_base64(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# # Transcribe using Gemini
# def transcribe_gemini(audio_path, model_name):
#     st.write(f"Transcribing with Gemini {model_name}...")
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

# # Transcribe using Whisper
# def transcribe_whisper(audio_path, model_name):
#     st.write(f"Transcribing with Whisper {model_name}...")
#     model = whisper.load_model(model_name)
#     result = model.transcribe(audio_path, fp16=False)
#     return result["text"].strip()

# # ---------------- MAIN STREAMLIT APP ------------------------
# def main():
#     st.markdown('<div class="container">', unsafe_allow_html=True)
    
#     # Header
#     st.markdown('<div class="header">Audio Transcription App</div>', unsafe_allow_html=True)
#     st.markdown('<div class="subheader">Upload an audio file and select a transcription model to get started.</div>', unsafe_allow_html=True)

#     # File uploader with validation
#     audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg"])
    
#     if audio_file is None:
#         st.warning("Please upload a single audio file. Folders are not supported.")
#     else:
#         # Display file details
#         file_size = len(audio_file.getvalue()) / 1024  # Size in KB
#         st.markdown(f'<div class="file-details">File: {audio_file.name} | Size: {file_size:.2f} KB</div>', unsafe_allow_html=True)

#     # Model selection dropdown
#     model_choice = st.selectbox(
#         "Select Transcription Model",
#         options=["Gemini 1.5 Flash (cloud-based)", "Whisper (local, medium model)"],
#         index=0,
#         help="Choose the model for transcription."
#     )

#     if model_choice == "Gemini 1.5 Flash (cloud-based)":
#         model_name = "gemini-1.5-flash"
#         transcribe_func = transcribe_gemini
#     else:
#         model_name = "medium.en"
#         transcribe_func = transcribe_whisper

#     # Initialize session state for transcription output and status
#     if 'transcription_output' not in st.session_state:
#         st.session_state.transcription_output = ""
#     if 'status' not in st.session_state:
#         st.session_state.status = ""

#     # Clear output button
#     if st.session_state.transcription_output and st.button("Clear Output"):
#         st.session_state.transcription_output = ""
#         st.session_state.status = ""
#         st.experimental_rerun()

#     # Transcribe button
#     if audio_file is not None and st.button("Transcribe Audio"):
#         with st.spinner("Processing audio..."):
#             st.session_state.status = "Processing..."
#             st.markdown(f'<div class="status">{st.session_state.status}</div>', unsafe_allow_html=True)

#             # Create temporary files
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
#                 temp_wav_path = temp_wav_file.name
#                 temp_wav_file.write(audio_file.read())

#             denoised_audio_path = os.path.join(tempfile.gettempdir(), "denoised_audio.wav")

#             try:
#                 # Process audio
#                 convert_to_wav(temp_wav_path, temp_wav_path)
#                 denoise_audio(temp_wav_path, denoised_audio_path)
#                 segments = split_audio(denoised_audio_path)

#                 combined_transcription = ""
#                 progress_bar = st.progress(0)
#                 for i, segment in enumerate(segments):
#                     st.write(f"Processing segment {i + 1}/{len(segments)}: {segment}")
#                     transcription = transcribe_func(segment, model_name)
#                     st.write(f"Raw transcription for segment {i + 1}:\n{transcription}\n")

#                     processed_transcription = process_transcription(transcription)
#                     final_transcription = correct_grammar(processed_transcription)
#                     combined_transcription += final_transcription + "\n"

#                     os.remove(segment)
#                     progress_bar.progress((i + 1) / len(segments))

#                 # Update session state with transcription
#                 st.session_state.transcription_output = combined_transcription
#                 st.session_state.status = "Completed"

#                 # Display final transcription
#                 st.subheader("Transcription Result")
#                 st.text_area("Transcription Output", st.session_state.transcription_output, height=300)

#                 # Save transcription to a temporary file for download
#                 output_file = os.path.join(tempfile.gettempdir(), "final_transcription_output.txt")
#                 with open(output_file, "w") as file:
#                     file.write(st.session_state.transcription_output)

#                 # Provide download button
#                 with open(output_file, "rb") as file:
#                     st.download_button(
#                         label="Download Transcription",
#                         data=file,
#                         file_name="final_transcription_output.txt",
#                         mime="text/plain"
#                     )

#                 st.success(f"Transcription complete! Saved to: {output_file}")
#                 st.markdown(f'<div class="status">{st.session_state.status}</div>', unsafe_allow_html=True)

#             except Exception as e:
#                 st.session_state.status = "Error"
#                 st.error(f"An error occurred during transcription: {str(e)}. Please try again with a different file or model.")
#                 st.markdown(f'<div class="status">{st.session_state.status}</div>', unsafe_allow_html=True)
#             finally:
#                 # Clean up temporary files
#                 for f in [temp_wav_path, denoised_audio_path]:
#                     if os.path.exists(f):
#                         os.remove(f)

#     # Display transcription if it exists in session state
#     if st.session_state.transcription_output:
#         st.subheader("Transcription Result")
#         st.text_area("Transcription Output", st.session_state.transcription_output, height=300)
#         # Provide download button again for existing transcription
#         output_file = os.path.join(tempfile.gettempdir(), "final_transcription_output.txt")
#         if os.path.exists(output_file):
#             with open(output_file, "rb") as file:
#                 st.download_button(
#                     label="Download Transcription",
#                     data=file,
#                     file_name="final_transcription_output.txt",
#                     mime="text/plain"
#                 )

#     st.markdown('</div>', unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()
import streamlit as st
import re
import language_tool_python
import librosa
from noisereduce import reduce_noise
import soundfile as sf
import os
import base64
import tempfile
from google.generativeai import GenerativeModel, configure
from pydub import AudioSegment
import whisper

# Custom CSS for a polished, professional design with a white background
st.markdown("""
    <style>
    .container {
        max-width: 900px;
        margin: auto;
        padding: 30px;
        background-color: #FFFFFF;  /* White background */
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    .header {
        font-size: 28px;
        font-weight: bold;
        color: #1E1E1E;
        margin-bottom: 10px;
        text-align: center;
    }
    .subheader {
        font-size: 18px;
        color: #555555;
        margin-bottom: 20px;
        text-align: center;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        margin: 5px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stSelectbox {
        max-width: 300px;
        margin: 10px 0;
    }
    .stTextArea textarea {
        border: 1px solid #DDDDDD;
        border-radius: 5px;
        font-size: 14px;
        background-color: #F9F9F9;
    }
    .file-details {
        font-size: 14px;
        color: #666666;
        margin: 10px 0;
    }
    .status {
        font-size: 16px;
        font-weight: bold;
        margin: 10px 0;
    }
    .uploader-container {
        border: 2px dashed #CCCCCC;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #F5F5F5;
        margin-bottom: 10px;
    }
    .uploader-text {
        font-size: 16px;
        color: #666666;
        margin-bottom: 10px;
    }
    .uploader-subtext {
        font-size: 14px;
        color: #999999;
        margin-bottom: 10px;
    }
    .browse-button {
        background-color: #007BFF !important;
        color: white !important;
        border: none !important;
        padding: 8px 16px !important;
        border-radius: 5px !important;
        font-size: 14px !important;
    }
    .warning-message {
        background-color: #FFF3CD;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        font-size: 14px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ GEMINI SETUP ------------------
API_KEY = "AIzaSyDDlG2xtuTEuYci3tNTkMSPtLue9u3aURI"  # Replace with your own API key securely
configure(api_key=API_KEY)

# Function to convert audio to WAV (16kHz, mono) using pydub
def convert_to_wav(input_audio_path, output_audio_path):
    st.write("Converting audio to WAV (16kHz, mono)...")
    audio = AudioSegment.from_file(input_audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_audio_path, format="wav")
    st.write(f"Audio converted and saved as: {output_audio_path}")
    return output_audio_path

# Function to reduce noise
def denoise_audio(input_audio_path, output_audio_path):
    st.write("Performing noise reduction...")
    try:
        audio, sr = librosa.load(input_audio_path, sr=None)
        noise_profile = audio[:sr * 2]
        reduced_noise_audio = reduce_noise(y=audio, sr=sr, y_noise=noise_profile)
        sf.write(output_audio_path, reduced_noise_audio, sr)
        st.write("Noise reduction complete.")
    except Exception as e:
        st.error(f"Error during noise reduction: {e}")
        raise

# Split audio into smaller chunks
def split_audio(input_audio_path, segment_duration=300):
    st.write("Splitting audio into smaller segments...")
    audio, sr = librosa.load(input_audio_path, sr=None)
    total_duration = librosa.get_duration(y=audio, sr=sr)
    segments = []

    for start in range(0, int(total_duration), segment_duration):
        end = min(start + segment_duration, int(total_duration))
        segment = audio[start * sr:end * sr]
        segment_path = os.path.join(tempfile.gettempdir(), f"segment_{start // segment_duration + 1}.wav")
        sf.write(segment_path, segment, sr)
        segments.append(segment_path)

    st.write(f"Audio split into {len(segments)} segments.")
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
    st.write(f"Transcribing with Gemini {model_name}...")
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
    st.write(f"Transcribing with Whisper {model_name}...")
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, fp16=False)
    return result["text"].strip()

# ---------------- MAIN STREAMLIT APP ------------------------
def main():
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="header">Audio Transcription App</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Upload an audio file and select a transcription model to get started.</div>', unsafe_allow_html=True)

    # File uploader with styled drag-and-drop area
    st.markdown('<div class="uploader-container">', unsafe_allow_html=True)
    st.markdown('<div class="uploader-text">Choose an audio file</div>', unsafe_allow_html=True)
    st.markdown('<div class="uploader-subtext">Drag and drop file here<br>Limit 200MB per file • WAV, MP3, M4A, OGG</div>', unsafe_allow_html=True)
    audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if audio_file is None:
        st.markdown('<div class="warning-message">PLEASE upload a single audio file. Folders are not supported.</div>', unsafe_allow_html=True)
    else:
        # Display file details
        file_size = len(audio_file.getvalue()) / 1024  # Size in KB
        st.markdown(f'<div class="file-details">File: {audio_file.name} | Size: {file_size:.2f} KB</div>', unsafe_allow_html=True)

    # Model selection dropdown
    model_choice = st.selectbox(
        "Select Transcription Model",
        options=["Gemini 1.5 Flash (cloud-based)", "Whisper (local, medium model)"],
        index=0,
        help="Choose the model for transcription."
    )

    if model_choice == "Gemini 1.5 Flash (cloud-based)":
        model_name = "gemini-1.5-flash"
        transcribe_func = transcribe_gemini
    else:
        model_name = "medium.en"
        transcribe_func = transcribe_whisper

    # Initialize session state for transcription output and status
    if 'transcription_output' not in st.session_state:
        st.session_state.transcription_output = ""
    if 'status' not in st.session_state:
        st.session_state.status = ""

    # Clear output button
    if st.session_state.transcription_output and st.button("Clear Output"):
        st.session_state.transcription_output = ""
        st.session_state.status = ""
        st.experimental_rerun()

    # Transcribe button
    if audio_file is not None and st.button("Transcribe Audio"):
        with st.spinner("Processing audio..."):
            st.session_state.status = "Processing..."
            st.markdown(f'<div class="status">{st.session_state.status}</div>', unsafe_allow_html=True)

            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
                temp_wav_path = temp_wav_file.name
                temp_wav_file.write(audio_file.read())

            denoised_audio_path = os.path.join(tempfile.gettempdir(), "denoised_audio.wav")

            try:
                # Process audio
                convert_to_wav(temp_wav_path, temp_wav_path)
                denoise_audio(temp_wav_path, denoised_audio_path)
                segments = split_audio(denoised_audio_path)

                combined_transcription = ""
                progress_bar = st.progress(0)
                for i, segment in enumerate(segments):
                    st.write(f"Processing segment {i + 1}/{len(segments)}: {segment}")
                    transcription = transcribe_func(segment, model_name)
                    st.write(f"Raw transcription for segment {i + 1}:\n{transcription}\n")

                    processed_transcription = process_transcription(transcription)
                    final_transcription = correct_grammar(processed_transcription)
                    combined_transcription += final_transcription + "\n"

                    os.remove(segment)
                    progress_bar.progress((i + 1) / len(segments))

                # Update session state with transcription
                st.session_state.transcription_output = combined_transcription
                st.session_state.status = "Completed"

                # Display final transcription
                st.subheader("Transcription Result")
                st.text_area("Transcription Output", st.session_state.transcription_output, height=300)

                # Save transcription to a temporary file for download
                output_file = os.path.join(tempfile.gettempdir(), "final_transcription_output.txt")
                with open(output_file, "w") as file:
                    file.write(st.session_state.transcription_output)

                # Provide download button
                with open(output_file, "rb") as file:
                    st.download_button(
                        label="Download Transcription",
                        data=file,
                        file_name="final_transcription_output.txt",
                        mime="text/plain"
                    )

                st.success(f"Transcription complete! Saved to: {output_file}")
                st.markdown(f'<div class="status">{st.session_state.status}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.session_state.status = "Error"
                st.error(f"An error occurred during transcription: {str(e)}. Please try again with a different file or model.")
                st.markdown(f'<div class="status">{st.session_state.status}</div>', unsafe_allow_html=True)
            finally:
                # Clean up temporary files
                for f in [temp_wav_path, denoised_audio_path]:
                    if os.path.exists(f):
                        os.remove(f)

    # Display transcription if it exists in session state
    if st.session_state.transcription_output:
        st.subheader("Transcription Result")
        st.text_area("Transcription Output", st.session_state.transcription_output, height=300)
        # Provide download button again for existing transcription
        output_file = os.path.join(tempfile.gettempdir(), "final_transcription_output.txt")
        if os.path.exists(output_file):
            with open(output_file, "rb") as file:
                st.download_button(
                    label="Download Transcription",
                    data=file,
                    file_name="final_transcription_output.txt",
                    mime="text/plain"
                )

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()



# import streamlit as st
# import re
# import language_tool_python
# import librosa
# from noisereduce import reduce_noise
# import soundfile as sf
# import os
# import base64
# import tempfile
# from google.generativeai import GenerativeModel, configure
# from pydub import AudioSegment
# import whisper

# # Custom CSS for a polished, professional design with a white background

# st.markdown("""
#     <style>
#     .container {
#         max-width: 900px;
#         margin: auto;
#         padding: 30px;
#         background-color: #FFFFFF;  
#         border-radius: 10px;
#         box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
#     }
#     .header {
#         font-size: 28px;
#         font-weight: bold;
#         color: #1E1E1E;
#         margin-bottom: 10px;
#         text-align: center;
#     }
#     .subheader {
#         font-size: 18px;
#         color: #555555;
#         margin-bottom: 20px;
#         text-align: center;
#     }
#     .stButton>button {
#         background-color: #007BFF;
#         color: white;
#         border: none;
#         padding: 10px 20px;
#         border-radius: 5px;
#         font-size: 16px;
#         margin: 5px;
#     }
#     .stButton>button:hover {
#         background-color: #0056b3;
#     }
#     .stSelectbox {
#         max-width: 300px;
#         margin: 10px 0;
#     }
#     .stTextArea textarea {
#         border: 1px solid #DDDDDD;
#         border-radius: 5px;
#         font-size: 14px;
#         background-color: #F9F9F9;
#     }
#     .file-details {
#         font-size: 14px;
#         color: #666666;
#         margin: 10px 0;
#     }
#     .status {
#         font-size: 16px;
#         font-weight: bold;
#         margin: 10px 0;
#     }
#     .uploader-container {
#         border: 2px dashed #CCCCCC;
#         border-radius: 10px;
#         padding: 20px;
#         text-align: center;
#         background-color: #F5F5F5;
#         margin-bottom: 10px;
#     }
#     .uploader-text {
#         font-size: 16px;
#         color: #666666;
#         margin-bottom: 10px;
#     }
#     .uploader-subtext {
#         font-size: 14px;
#         color: #999999;
#         margin-bottom: 10px;
#     }
#     .browse-button {
#         background-color: #007BFF !important;
#         color: white !important;
#         border: none !important;
#         padding: 8px 16px !important;
#         border-radius: 5px !important;
#         font-size: 14px !important;
#     }
#     .warning-message {
#         background-color: #FFF3CD;
#         color: #856404;
#         padding: 10px;
#         border-radius: 5px;
#         font-size: 14px;
#         margin: 10px 0;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # ------------------ GEMINI SETUP ------------------
# API_KEY = "AIzaSyDDlG2xtuTEuYci3tNTkMSPtLue9u3aURI"  
# configure(api_key=API_KEY)

# # Function to convert audio to WAV (16kHz, mono) using pydub
# def convert_to_wav(input_audio_path, output_audio_path):
#     st.write("Converting audio to WAV (16kHz, mono)...")
#     audio = AudioSegment.from_file(input_audio_path)
#     audio = audio.set_frame_rate(16000).set_channels(1)
#     audio.export(output_audio_path, format="wav")
#     st.write(f"Audio converted and saved as: {output_audio_path}")
#     return output_audio_path

# # Function to reduce noise
# def denoise_audio(input_audio_path, output_audio_path):
#     st.write("Performing noise reduction...")
#     try:
#         audio, sr = librosa.load(input_audio_path, sr=None)
#         noise_profile = audio[:sr * 2]
#         reduced_noise_audio = reduce_noise(y=audio, sr=sr, y_noise=noise_profile)
#         sf.write(output_audio_path, reduced_noise_audio, sr)
#         st.write("Noise reduction complete.")
#     except Exception as e:
#         st.error(f"Error during noise reduction: {e}")
#         raise

# # Split audio into smaller chunks
# def split_audio(input_audio_path, segment_duration=300):
#     st.write("Splitting audio into smaller segments...")
#     audio, sr = librosa.load(input_audio_path, sr=None)
#     total_duration = librosa.get_duration(y=audio, sr=sr)
#     segments = []

#     for start in range(0, int(total_duration), segment_duration):
#         end = min(start + segment_duration, int(total_duration))
#         segment = audio[start * sr:end * sr]
#         segment_path = os.path.join(tempfile.gettempdir(), f"segment_{start // segment_duration + 1}.wav")
#         sf.write(segment_path, segment, sr)
#         segments.append(segment_path)

#     st.write(f"Audio split into {len(segments)} segments.")
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

# # Encode audio to base64 for Gemini
# def encode_audio_base64(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# # Transcribe using Gemini
# def transcribe_gemini(audio_path, model_name):
#     st.write(f"Transcribing with Gemini {model_name}...")
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

# # Transcribe using Whisper
# def transcribe_whisper(audio_path, model_name):
#     st.write(f"Transcribing with Whisper {model_name}...")
#     model = whisper.load_model(model_name)
#     result = model.transcribe(audio_path, fp16=False)
#     return result["text"].strip()

# # ---------------- MAIN STREAMLIT APP ------------------------
# def main():
#     st.markdown('<div class="container">', unsafe_allow_html=True)
    
#     # Header
#     st.markdown('<div class="header">Audio Transcription App</div>', unsafe_allow_html=True)
#     st.markdown('<div class="subheader">Upload an audio file and select a transcription model to get started.</div>', unsafe_allow_html=True)

#     # File uploader with styled drag-and-drop area
#     st.markdown('<div class="uploader-container">', unsafe_allow_html=True)
#     st.markdown('<div class="uploader-text">Choose an audio file</div>', unsafe_allow_html=True)
#     st.markdown('<div class="uploader-subtext">Drag and drop file here<br>Limit 200MB per file • WAV, MP3, M4A, OGG</div>', unsafe_allow_html=True)
#     audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg"], label_visibility="collapsed")
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     if audio_file is None:
#         st.markdown('<div class="warning-message">PLEASE upload a single audio file. Folders are not supported.</div>', unsafe_allow_html=True)
#     else:
#         # Display file details
#         file_size = len(audio_file.getvalue()) / 1024  # Size in KB
#         st.markdown(f'<div class="file-details">File: {audio_file.name} | Size: {file_size:.2f} KB</div>', unsafe_allow_html=True)

#     # Model selection dropdown
#     model_choice = st.selectbox(
#         "Select Transcription Model",
#         options=["Gemini 1.5 Flash (cloud-based)", "Whisper (local, medium model)"],
#         index=0,
#         help="Choose the model for transcription."
#     )

#     if model_choice == "Gemini 1.5 Flash (cloud-based)":
#         model_name = "gemini-1.5-flash"
#         transcribe_func = transcribe_gemini
#     else:
#         model_name = "medium.en"
#         transcribe_func = transcribe_whisper

#     # Initialize session state for transcription output and status
#     if 'transcription_output' not in st.session_state:
#         st.session_state.transcription_output = ""
#     if 'status' not in st.session_state:
#         st.session_state.status = ""

#     # Clear output button
#     if st.session_state.transcription_output and st.button("Clear Output"):
#         st.session_state.transcription_output = ""
#         st.session_state.status = ""
#         st.experimental_rerun()

#     # Transcribe button
#     if audio_file is not None and st.button("Transcribe Audio"):
#         with st.spinner("Processing audio..."):
#             st.session_state.status = "Processing..."
#             st.markdown(f'<div class="status">{st.session_state.status}</div>', unsafe_allow_html=True)

#             # Create temporary files
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
#                 temp_wav_path = temp_wav_file.name
#                 temp_wav_file.write(audio_file.read())

#             denoised_audio_path = os.path.join(tempfile.gettempdir(), "denoised_audio.wav")

#             try:
#                 # Process audio
#                 convert_to_wav(temp_wav_path, temp_wav_path)
#                 denoise_audio(temp_wav_path, denoised_audio_path)
#                 segments = split_audio(denoised_audio_path)

#                 combined_transcription = ""
#                 progress_bar = st.progress(0)
#                 for i, segment in enumerate(segments):
#                     st.write(f"Processing segment {i + 1}/{len(segments)}: {segment}")
#                     transcription = transcribe_func(segment, model_name)
#                     st.write(f"Raw transcription for segment {i + 1}:\n{transcription}\n")

#                     processed_transcription = process_transcription(transcription)
#                     final_transcription = correct_grammar(processed_transcription)
#                     combined_transcription += final_transcription + "\n"

#                     os.remove(segment)
#                     progress_bar.progress((i + 1) / len(segments))

#                 # Update session state with transcription
#                 st.session_state.transcription_output = combined_transcription
#                 st.session_state.status = "Completed"

#                 # Display final transcription
#                 st.subheader("Transcription Result")
#                 st.text_area("Transcription Output", st.session_state.transcription_output, height=300, key="transcription_output_1")

#                 # Save transcription to a temporary file for download
#                 output_file = os.path.join(tempfile.gettempdir(), "final_transcription_output.txt")
#                 with open(output_file, "w") as file:
#                     file.write(st.session_state.transcription_output)

#                 # Provide download button with unique key
#                 with open(output_file, "rb") as file:
#                     st.download_button(
#                         label="Download Transcription",
#                         data=file,
#                         file_name="final_transcription_output.txt",
#                         mime="text/plain",
#                         key="download_button_1"
#                     )

#                 st.success(f"Transcription complete! Saved to: {output_file}")
#                 st.markdown(f'<div class="status">{st.session_state.status}</div>', unsafe_allow_html=True)

#             except Exception as e:
#                 st.session_state.status = "Error"
#                 st.error(f"An error occurred during transcription: {str(e)}. Please try again with a different file or model.")
#                 st.markdown(f'<div class="status">{st.session_state.status}</div>', unsafe_allow_html=True)
#             finally:
#                 # Clean up temporary files
#                 for f in [temp_wav_path, denoised_audio_path]:
#                     if os.path.exists(f):
#                         os.remove(f)

#     # Display transcription if it exists in session state
#     if st.session_state.transcription_output:
#         st.subheader("Transcription Result")
#         st.text_area("Transcription Output", st.session_state.transcription_output, height=300, key="transcription_output_2")
#         # Provide download button again for existing transcription with unique key
#         output_file = os.path.join(tempfile.gettempdir(), "final_transcription_output.txt")
#         if os.path.exists(output_file):
#             with open(output_file, "rb") as file:
#                 st.download_button(
#                     label="Download Transcription",
#                     data=file,
#                     file_name="final_transcription_output.txt",
#                     mime="text/plain",
#                     key="download_button_2"
#                 )

#     st.markdown('</div>', unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import os
# import re
# import base64
# import tempfile
# import language_tool_python
# import librosa
# from noisereduce import reduce_noise
# import soundfile as sf
# from google.generativeai import GenerativeModel, configure
# from pydub import AudioSegment
# import whisper

# # ------------------ GEMINI SETUP ------------------
# API_KEY = "AIzaSyDDlG2xtuTEuYci3tNTkMSPtLue9u3aURI"  # Replace securely in prod
# configure(api_key=API_KEY)

# # ------------------ UTILITY FUNCTIONS ------------------

# def convert_to_wav(input_audio_path, output_audio_path):
#     audio = AudioSegment.from_file(input_audio_path)
#     audio = audio.set_frame_rate(16000).set_channels(1)
#     audio.export(output_audio_path, format="wav")
#     return output_audio_path

# def denoise_audio(input_audio_path, output_audio_path):
#     audio, sr = librosa.load(input_audio_path, sr=None)
#     noise_profile = audio[:sr * 2]
#     reduced_audio = reduce_noise(y=audio, sr=sr, y_noise=noise_profile)
#     sf.write(output_audio_path, reduced_audio, sr)

# def split_audio(input_audio_path, segment_duration=300):
#     audio, sr = librosa.load(input_audio_path, sr=None)
#     duration = librosa.get_duration(y=audio, sr=sr)
#     segments = []
#     for start in range(0, int(duration), segment_duration):
#         end = min(start + segment_duration, int(duration))
#         segment = audio[start * sr:end * sr]
#         path = f"segment_{start // segment_duration + 1}.wav"
#         sf.write(path, segment, sr)
#         segments.append(path)
#     return segments

# def encode_audio_base64(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# def transcribe_gemini(audio_path, model_name):
#     audio_base64 = encode_audio_base64(audio_path)
#     model = GenerativeModel(model_name)
#     response = model.generate_content([
#         {"mime_type": "audio/wav", "data": audio_base64},
#         {"text": "Transcribe this audio segment."}
#     ])
#     return response.text.strip()

# def transcribe_whisper(audio_path, model_name):
#     model = whisper.load_model(model_name)
#     result = model.transcribe(audio_path, fp16=False)
#     return result["text"].strip()

# def process_transcription(text):
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
#         text = re.sub(command, symbol, text, flags=re.IGNORECASE)
#     return text

# def correct_grammar(text):
#     tool = language_tool_python.LanguageTool("en-US")
#     matches = tool.check(text)
#     return language_tool_python.utils.correct(text, matches)

# # ------------------ STREAMLIT UI ------------------

# st.set_page_config(page_title="Audio Transcriber", layout="centered")

# st.markdown("""
#     <style>
#     body {
#         background-color: #fdfdfd;
#     }
#     .title {
#         font-size: 40px;
#         font-weight: 700;
#         color: #1f3c88;
#         text-align: center;
#         padding-bottom: 20px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown('<div class="title">🎙️ AI Audio Transcriber</div>', unsafe_allow_html=True)

# model_choice = st.selectbox("Choose Transcription Model", ["Gemini 1.5 Flash", "Whisper (medium.en)"])
# audio_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "m4a", "ogg"])

# if st.button("Transcribe", key="transcribe_btn"):
#     if audio_file is not None:
#         with st.spinner("Processing..."):
#             try:
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_input:
#                     temp_input.write(audio_file.read())
#                     temp_input.flush()

#                     temp_wav = "temp_converted.wav"
#                     denoised_wav = "temp_denoised.wav"

#                     convert_to_wav(temp_input.name, temp_wav)
#                     denoise_audio(temp_wav, denoised_wav)
#                     segments = split_audio(denoised_wav)

#                     final_transcription = ""
#                     for segment in segments:
#                         if "Gemini" in model_choice:
#                             raw_text = transcribe_gemini(segment, "gemini-1.5-flash")
#                         else:
#                             raw_text = transcribe_whisper(segment, "medium.en")

#                         processed = process_transcription(raw_text)
#                         corrected = correct_grammar(processed)
#                         final_transcription += corrected + "\n"
#                         os.remove(segment)

#                     st.subheader(" Final Transcription")
#                     st.text_area("", final_transcription, height=300)
#                     with open("final_transcription_output.txt", "w") as out_f:
#                         out_f.write(final_transcription)
#                     st.success(" Transcription Complete. Saved to final_transcription_output.txt")

#                     os.remove(temp_wav)
#                     os.remove(denoised_wav)
#                     os.remove(temp_input.name)

#             except Exception as e:
#                 st.error(f"Error during transcription: {e}")
#     else:
#         st.warning(" Please upload an audio file before clicking Transcribe.")