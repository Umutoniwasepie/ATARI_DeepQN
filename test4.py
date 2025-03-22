import os
from ale_py import ALEInterface

# Set the path to your ROMs directory
os.environ["ALE_ROMS"] = "./roms"  # Replace with your ROM path

# Initialize ALE (required for Gymnasium to detect Atari)
ALEInterface()