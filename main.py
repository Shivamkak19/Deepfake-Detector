import os
import shutil

# Save the current working directory
original_cwd = os.getcwd()

# Specify the filename and subdirectory
target_subdirectory = "tortoise_tts"
os.chdir(target_subdirectory)

# Execute the voiceProtect_app.py script in the subdirectory
os.system("python voiceProtect_app.py")