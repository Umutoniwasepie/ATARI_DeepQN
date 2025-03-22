import os
from ale_py import ALEInterface
from stable_baselines3.common.env_util import make_atari_env

# Set ROM path (replace with your actual path)
os.environ["ALE_ROMS"] = "./roms"
ALEInterface()  # Initialize ALE

# Create the environment
env = make_atari_env("ALE/Galaxian-v5", n_envs=1)  # No "ALE/" prefix