import sys
import os

# Calculate the absolute path of the directory containing app.py
app_directory = os.path.dirname(os.path.abspath(__file__))

# Calculate the absolute path of the submodule directory
submodule_directory = os.path.join(app_directory, 'tortoise_tts')
sys.path.append(submodule_directory)


from tortoise_tts.tortoise.models.classifier import AudioMiniEncoderWithClassifierHead


