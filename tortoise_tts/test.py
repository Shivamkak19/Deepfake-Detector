# Add tortoise_tts directory to path:
# Calculate the absolute path of the directory containing app.py, submodule directory, and append to system path
# app_directory = os.path.dirname(os.path.abspath(__file__))
# submodule_directory = os.path.join(app_directory, 'tortoise_tts')
# sys.path.append(submodule_directory)

# Main functionality
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import pyaudio
import torchaudio
import torch
import torch.nn.functional as F

# System imports
import wave
import sys
import os 
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

# # App function imports
# from inputAudio import inputAudio
# from configWavPlot import configWavPlot

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

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1 if sys.platform == 'darwin' else 2
    RATE = 44100
    RECORD_SECONDS = 5

    with wave.open('../resources/output.wav', 'wb') as wf:
        p = pyaudio.PyAudio()
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

        st.info('Recording Now...')
        for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
            wf.writeframes(stream.read(CHUNK))
        print('Done')

        stream.close()
        p.terminate()
        st.info("finished recording")
        
    # Convert pyaudio output from wav to mp3 with pydub
    wav_file = '../resources/output.wav'
    output_mp3_file = '../resources/output.mp3'

    AudioSegment.from_wav(wav_file).export(output_mp3_file, format="mp3")

# Generate and stylize waveform plot from input .WAV path
def generateWavePlot(path: str):

    try:
        # reading the audio file
        raw = wave.open(path)
        
        # reads all the frames
        # -1 indicates all or max frames
        signal = raw.readframes(-1)
        signal = np.frombuffer(signal, dtype="int16")
        
        # gets the frame rate
        f_rate = raw.getframerate()
    
        # to Plot the x-axis in seconds
        # you need get the frame rate
        # and divide by size of your signal
        # to create a Time Vector
        # spaced linearly with the size
        # of the audio file
        time = np.linspace(
            0,  # start
            len(signal) / f_rate,
            num=len(signal)
        )
    
        # create a new figure using plt.subplots()
        fig, ax = plt.subplots(figsize=(4, 4))

        # title of the plot
        ax.set_title("Input Audio Waveform", fontsize=14)

        # label of x-axis
        ax.set_xlabel("Time", fontsize=10)
        
        # actual plotting
        ax.plot(time, signal, color="#2e7769", linewidth=1)

        # Customize the plot appearance
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(0, color='white', linewidth=0.8)
        ax.axvline(0, color='white', linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=8)

        # Set background color
        ax.set_facecolor('#2d2d2d')

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Adjust margins
        plt.tight_layout()

        # Display the plot using st.pyplot with the Matplotlib figure
        st.pyplot(fig)

    except wave.Error as e:
        st.error(f"Error opening the WAV file: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")


# Run script
def main():
    setStreamlitGUI()

if __name__ == "__main__":
    main()

