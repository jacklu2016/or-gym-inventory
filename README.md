# Gymnasium-Compatible Newsvendor Environment

This repository provides a multi-period newsvendor environment compatible with the [Gymnasium](https://gymnasium.farama.org/) reinforcement learning interface.

The environment is based on the work by Balaji et al. (2019):
- Paper: https://arxiv.org/abs/1911.10641
- Original GitHub (using OpenAI Gym): https://github.com/awslabs/or-rl-benchmarks

## Description

The environment simulates a multi-period newsvendor problem with lead times. An agent must decide how much inventory to order each period to meet stochastic demand (modeled as Poisson) while minimizing costs associated with purchasing, holding excess inventory, and stockouts (lost sales). Inventory ordered takes a fixed number of periods (`lead_time`) to arrive.

Refer to the docstrings within `newsvendor_env_gymnasium.py` for details on the observation space, action space, and reward calculation.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/gymnasium-newsvendor.git](https://www.google.com/search?q=https://github.com/YOUR_USERNAME/gymnasium-newsvendor.git)
    cd gymnasium-newsvendor
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can instantiate and use the environment like any standard Gymnasium environment:

```python
import gymnasium as gym
# Make sure newsvendor_env_gymnasium.py is in your Python path or current directory
from newsvendor_env_gymnasium import NewsvendorEnv

# Create the environment with custom parameters (optional)
env = NewsvendorEnv(lead_time=3, step_limit=100)

# Or use default parameters
# env = NewsvendorEnv()

# Reset for a new episode
observation, info = env.reset(seed=42) # Use a seed for reproducibility

terminated = False
truncated = False
total_episode_reward = 0

while not terminated and not truncated:
    # Sample a random action (replace with your agent's action)
    action = env.action_space.sample()

    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)

    total_episode_reward += reward

    # Optional: Render or print info
    # env.render()
    # print(f"Step: {info['step_count']}, Reward: {reward:.2f}")

print(f"Episode finished. Total Reward: {total_episode_reward:.2f}")

env.close()
