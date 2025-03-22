# ATARI_DeepQN

# Atari Galaxian DQN Agent with Stable Baselines3

This project trains a Deep Q-Network (DQN) agent to play the Atari game **Galaxian** using Stable Baselines3 and Gymnasium. The agent learns through exploration/exploitation trade-offs and is evaluated in a real-time gameplay environment.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Key Implementation Details](#key-implementation-details)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## Features

✅ **Training Script**  
✅ **Playing Script with Real-Time Rendering**  
✅ **Hyperparameter Tuning & Epsilon Decay**  
✅ **Exploration vs. Exploitation Analysis**

---

## Installation

### Dependencies

```bash
pip install gymnasium==1.0.0 stable-baselines3==2.5.0 ale-py==0.9.0 autorom tensorboard
```

### ROM Setup

```bash
python -m AutoROM --install-dir ./roms --accept-license
```

### Usage

- Training the Agent

```bash
python train.py
```

- outputs:
  **_'dqn_galaxian.zip' (saved model)_**

**_logs/ (TensorBoard metrics)_**
