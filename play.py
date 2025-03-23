import gymnasium as gym
import numpy as np
import os
import time
from ale_py import ALEInterface
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Initialize ALE and set ROM path
os.environ["ALE_ROMS"] = os.path.abspath("./roms")
ALEInterface()

def make_atari_env(env_id, n_envs=1, seed=None, render_mode="human"):
    """
    Create a wrapped Atari environment 
    """
    def make_env(rank):
        def _init():
            # Create base environment
            env = gym.make(env_id, render_mode=render_mode)
            
            
            env = AtariWrapper(env) 
            
            if seed is not None:
                env.reset(seed=seed + rank)
            return env
        return _init
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    
    # Stack 4 frames 
    env = VecFrameStack(env, n_stack=4)
    
    return env

def play_and_evaluate(model_path, env_id, num_episodes=3):
    """
    Load the model and evaluate it on the environment
    """
    # Create environment with rendering
    env = make_atari_env(env_id, render_mode="human")
    
    # Load the trained model
    model = DQN.load(model_path, buffer_size=1, device='cpu', batch_size=32)
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = [False]
        total_reward = 0
        
        while not done[0]:
            # Get action from the model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action in environment
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            
            # Render the game
            env.render()
            time.sleep(0.02)  
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    env.close()

def explore_different_eps_values(model_path, env_id, epsilon_values=[0.0, 0.1, 0.3, 0.5]):
    """
    Test different exploration rates
    """
    env = make_atari_env(env_id)
    model = DQN.load(model_path)
    
    for eps in epsilon_values:
        model.exploration_rate = eps
        total_reward = 0
        obs = env.reset()
        done = [False]
        
        while not done[0]:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
        
        print(f"Epsilon {eps}: Total Reward = {total_reward}")
    
    env.close()

if __name__ == "__main__":
    # Configuration (MUST match training setup)
    MODEL_PATH = "dqn_galaxian.zip"  
    ENV_ID = "ALE/Galaxian-v5"  
    
    # Run evaluation
    play_and_evaluate(MODEL_PATH, ENV_ID)
    