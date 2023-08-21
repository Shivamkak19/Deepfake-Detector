# '##::: ##::'#######::'########:'########:
#  ###:: ##:'##.... ##:... ##..:: ##.....::
#  ####: ##: ##:::: ##:::: ##:::: ##:::::::
#  ## ## ##: ##:::: ##:::: ##:::: ######:::
#  ##. ####: ##:::: ##:::: ##:::: ##...::::
#  ##:. ###: ##:::: ##:::: ##:::: ##:::::::
#  ##::. ##:. #######::::: ##:::: ########:
# ..::::..:::.......::::::..:::::........::

# Both local deployment and the live Streamlit deployment launch the main function in tortoise_tts/voiceProtect_app.py, NOT this file. This is a duplicate file for ease of accessility.
# Deploying from the root directory causes threading issues as tortoise_tts must be launched in the main thread, which is assigned to the parent dir of __file__ on launch.
# ER: ValueError: signal only works in main thread of the main interpreter ; how to fix, or otherwise how to avoid having to download locally. See Issues.txt for full trace.

# //////////////////////////////////////////////////////////////////////////////

# PHASED OUT: Try to move script into tortoise_tts module while executing - threading issues:
# absolute_path = os.path.abspath(__file__)
# subdir_path = os.path.abspath("./tortoise_tts")
# target_path = os.path.join(subdir_path, "voiceProtect_app.py")

# st.info(absolute_path)
# st.info(target_path)

# os.rename(absolute_path, target_path)
# os.replace(absolute_path, target_path)
# shutil.move(absolute_path, target_path)

# //////////////////////////////////////////////////////////////////////////////

# System imports
import wave
import sys
import os 
import io 
from PIL import Image
from glob import glob
import subprocess

# Add tortoise_tts directory to path:
# Calculate the absolute path of the directory containing app.py, submodule directory, and append to system path
app_directory = os.path.dirname(os.path.abspath(__file__))
submodule_directory = os.path.join(app_directory, 'tortoise_tts')
sys.path.append(submodule_directory)

# Set current working directory to parent dir of main app file(/tortoise_tts)
# Resolves issue: local deploy launches CWD in parent dir, Streamlit Deploy launches CWD in root dir
os.chdir(app_directory)

# Main functionality
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import pyaudio 

import torchaudio
import torch
import torch.nn.functional as F
import streamlit as st

# Misc imports
import librosa
import plotly.express as px
import numpy as np 
from scipy.io.wavfile import read 
from pydub import AudioSegment
import matplotlib.pyplot as plt

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
    # st.info("Current working directory:", os.getcwd())

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

