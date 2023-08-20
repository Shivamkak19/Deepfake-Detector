# Add tortoise_tts directory to path:
# Calculate the absolute path of the directory containing app.py, submodule directory, and append to system path
# app_directory = os.path.dirname(os.path.abspath(__file__))
# submodule_directory = os.path.join(app_directory, 'tortoise_tts')
# sys.path.append(submodule_directory)

import os
import shutil

# Save the current working directory
original_cwd = os.getcwd()

# Specify the filename and subdirectory
script_to_move = "check.py"
target_subdirectory = "tortoise_tts"

# Change the working directory to the tortoise_tts subdirectory
os.chdir(target_subdirectory)

# Execute the voiceProtect_app.py script in the subdirectory
os.system("python check.py")

# Save the current working directory
# original_cwd = os.getcwd()

# # Change the working directory to the target subdirectory
# os.chdir(target_subdirectory)

# # Move the script to the subdirectory
# shutil.move(script_to_move, target_subdirectory)

# # Execute the script in the subdirectory
# os.system(f"python {os.path.join(target_subdirectory, script_to_move)}")

# Main functionality
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import pyaudio
import torchaudio
import torch
import torch.nn.functional as F

# System imports
import wave
import sys
# import os 
import io 
from PIL import Image
from glob import glob
import subprocess


# Misc imports
import streamlit as st
import librosa
import plotly.express as px
import numpy as np 
from scipy.io.wavfile import read 
from pydub import AudioSegment
import matplotlib.pyplot as plt

# Add parent directory to system PATH to import from subdirectories
parent_directory = os.path.join(os.path.dirname(__file__), '..')
parent_directory = os.path.abspath(parent_directory)
sys.path.append(parent_directory)

# App function imports
from inputAudio import inputAudio
from configWavPlot import configWavPlot

# Removed backend matplotlib GUI - caused multithreading issues
# import matplotlib
# matplotlib.use("TkAgg")  # Change to "Qt5Agg" or "QtAgg" if you prefer

# load an audio file as tensor for model input
def load_audio(audiopath, sampling_rate =22000):
    if isinstance(audiopath, str):
        if audiopath.endswith('.mp3'):
            audio, lsr = librosa.load(audiopath, sr=sampling_rate)
            audio = torch.FloatTensor(audio)
        else:
            assert False, f"Error for unsupported audio format {audiopath[-4]}"
    elif isinstance(audiopath, io.BytesIO):
        audio, lsr = torchaudio.load(audiopath)
        audio = audio[0]
    
    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
    
    #torch.any(param) returns true if any param is true
    if torch.any(audio > 2) or not torch.any(audio < 0 ):
        print(f"Error with audio data, Max: {audio.max()} and min: {audio.min()}")
        audio.clip_(-1, 1)

    return audio.unsqueeze(0)

# function to classify audio with tortoise-tts library
def classify_audio_clip(clip):

    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim = 1, embedding_dim = 512, depth = 5, downsample_factor = 4,
                                                    resnet_blocks = 2, attn_blocks = 4, num_attn_heads = 4, base_channels = 32,
                                                    dropout = 0, kernel_size = 5, distribute_zero_label= False)
    state_dict = torch.load('../classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)
    clip = clip.cpu().unsqueeze(dim = 0)
    results = F.softmax(classifier(clip), dim= -1)
    return results[0][0]

# Create Streamlit app layout
def setStreamlitGUI():

    #App GUI
    st.set_page_config(layout = "wide")

    logo = Image.open('../resources/VoiceProtect-logo.png')
    # Resize, maintain aspect ratio
    logo.thumbnail((600, 600))

    st.image(logo, width=50, use_column_width="auto")

    st.title("VoiceProtect - Deepfake Audio Detection")
    st.info("Implemented with Tortoise Text to Speech Library. Classifier.pth model can be found below:")
    st.info("https://huggingface.co/jbetker/tortoise-tts-v2")

    col1, col2 = st.columns(2)

    # Upload local mp3 file, use as input to model
    with col1:

        st.info("Upload a .mp3 file from your local machine to analyze")
        uploaded_file = st.file_uploader("Choose a file", type = 'mp3')

        if uploaded_file is not None:

            if st.button("Analyze audio"):

                st.info("Analyzing Audio...")
                row1, row2, row3 = st.columns(3)

                with row1:
                    audio_clip = load_audio(uploaded_file)
                    results = classify_audio_clip(audio_clip)
                    results = results.item()

                    st.info("your results are below")
                    st.info(f"The probability of deepfake audio is:  {results}") 
                    st.success(f"The uploaded audio is {results * 100: .2f}% likely to be AI generated")       

                with row2:
                    st.info("audio player")
                    st.audio(uploaded_file)

                with row3:
                    st.info("tried")
                    # Convert uploaded mp3 to wav, path to WAV for matplotlib
                    output_wav_file = '../resources/upload.wav'
                    AudioSegment.from_mp3(uploaded_file).export(output_wav_file, format="wav")
                    absolute_path = os.path.abspath(output_wav_file)

                    generateWavePlot(absolute_path)

                    # st.pyplot(generateWavePlot(output_wav_file))

    # Record User audio data, use as input to model
    with col2:

        st.info("Record a live, 5 second audio clip live to analyze")

        if st.button("Record audio"):

            # Record audio and save as wav file "output.wav"
            pyaudioStream()

            row1, row2, row3 = st.columns(3)

            with row1:

                # Path to MP3 for matplotlib
                relative_path = "../resources/output.mp3"
                absolute_path = os.path.abspath(relative_path)

                audio_clip = load_audio(absolute_path)
                results = classify_audio_clip(audio_clip)
                results = results.item()

                st.info("your results are below")
                st.info(f"The probability of deepfake audio is : {results}")
                st.info(f"The uploaded audio is {results * 100: .2f}% likely to be AI generated")

            with row2:
                st.info("audio player")
                st.audio(absolute_path)

            with row3:

                # Path to WAV for matplotlib
                relative_path = "../resources/output.wav"
                absolute_path = os.path.abspath(relative_path)

                generateWavePlot(absolute_path)
                # st.pyplot(generateWavePlot(absolute_path))


# Record user audio stream with pyaudio
# Convert to mp3 using pydub (load_audio() accepts mp3)
def pyaudioStream():
    inputAudio.pyaudioStream()


# Generate and stylize waveform plot from input .WAV path
def generateWavePlot(path: str):
    configWavPlot.generateWavePlot(path)


# Run script
def main():
    setStreamlitGUI()

if __name__ == "__main__":
    main()

