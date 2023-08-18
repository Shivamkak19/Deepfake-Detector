import sys
import os

# Add tortoise_tts directory to path:
# Calculate the absolute path of the directory containing app.py, submodule directory, and append to system path
app_directory = os.path.dirname(os.path.abspath(__file__))
submodule_directory = os.path.join(app_directory, 'tortoise_tts')
sys.path.append(submodule_directory)

from tortoise_tts.tortoise.models.classifier import AudioMiniEncoderWithClassifierHead

import streamlit as st
import os 
from glob import glob
import io 
import librosa
import plotly.express as px
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np 
from scipy.io.wavfile import read 

#torch.any(param) returns true if any param is true

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
    
    if torch.any(audio > 2) or not torch.any(audio < 0 ):
        print(f"Error with audio data, Max: {audio.max()} and min: {audio.min()}")
        audio.clip_(-1, 1)

    return audio.unsqueeze(0)

#function for classifier
def classify_audio_clip(clip):

    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim = 1, embedding_dim = 512, depth = 5, downsample_factor = 4,
                                                    resnet_blocks = 2, attn_blocks = 4, num_attn_heads = 4, base_channels = 32,
                                                    dropout = 0, kernel_size = 5, distribute_zero_label= False)
    state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)
    clip = clip.cpu().unsqueeze(dim = 0)
    results = F.softmax(classifier(clip), dim= -1)
    return results[0][0]


#App GUI
st.set_page_config(layout = "wide")

def main():

    st.title("Deepfake Test project")
    uploaded_file = st.file_uploader("Choose a file", type = 'mp3')

    if uploaded_file is not None:
        if st.button("Analyze audio"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.info("your results are below")
                audio_clip = load_audio(uploaded_file)
                results = classify_audio_clip(audio_clip)

                results = results.item()
                st.info(f"Probability of deepfake:  {results}") 
                st.success(f"The uploaded audio is {results * 100: .2f}% likely to be AI generated")       

            # with col2:
            #     st.info("audio player")
            #     st.audio(uploaded_file)

            #     #Waveform plot - plotly express
            #     fig = px.line()
            #     fig.add_scatter(x = )

            #     ####

            #     fig.update_layout()
            # with col3:

            

if __name__ == "__main__":
    main()