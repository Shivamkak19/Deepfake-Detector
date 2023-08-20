import pyaudio
import wave
import streamlit as st
from pydub import AudioSegment
import sys

class inputAudio():

    
    # from tortoise_tts.tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
    # Record user audio stream with pyaudio
    # Convert to mp3 using pydub (load_audio() accepts mp3)

    @staticmethod
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