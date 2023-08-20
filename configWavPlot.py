import wave
import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt



class configWavPlot():

    # Generate and stylize waveform plot from input .WAV path

    @staticmethod
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