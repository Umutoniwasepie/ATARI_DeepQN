import os
from ale_py import ALEInterface

# Set the ROM path to your downloaded directory
os.environ["ALE_ROMS"] = os.path.abspath("./roms")  # Use absolute path
ALEInterface()  # Initialize ALE



import os
from ale_py import ALEInterface
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

# Set ROM path and initialize ALE
os.environ["ALE_ROMS"] = "./roms"  # Path to your ROMs directory
ALEInterface()

# Create the environment
env = make_atari_env("ALE/Galaxian-v5", n_envs=1)
env = VecFrameStack(env, n_stack=4)  # Stack 4 frames

# Define the DQN agent
model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=0.00025,
    gamma=0.99,
    batch_size=32,
    exploration_initial_eps=1.0,  # Start with 100% random actions
    exploration_final_eps=0.01,    # End with 1% random actions
    exploration_fraction=0.1,      # Decay epsilon over 10% of training time
    tensorboard_log="./logs/"
)

# Train and save
model.learn(total_timesteps=300_000)
model.save("dqn_galaxian", exclude=["replay_buffer"])  # Exclude replay buffer