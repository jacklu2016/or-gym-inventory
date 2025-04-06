# Gymnasium-Compatible Inventory Management Environments & Benchmarks

This repository provides implementations of classic inventory management environments, adapted from the original [OR-Gym library](https://github.com/hubbs5/or-gym), updated for compatibility with the [Gymnasium](https://gymnasium.farama.org/) API (the successor to OpenAI Gym). It also includes comprehensive benchmarking scripts to compare various heuristic, optimization-inspired, and Reinforcement Learning (RL) policies on these environments.

The original environments are based on the work by Hubbs et al. (2020):
- Paper: https://arxiv.org/abs/2008.04001
- Original OR-Gym GitHub (using OpenAI Gym): https://github.com/hubbs5/or-gym

**Environments Adapted:**
1.  **Newsvendor (`newsvendor.py`):** Multi-period newsvendor problem with lead times and stochastic Poisson demand (based on Balaji et al. 2019, https://arxiv.org/abs/1911.10641).
2.  **Inventory Management (`inventory_management.py`):** Multi-period, multi-echelon inventory system for a single product. Includes `InvManagementBacklogEnv` and `InvManagementLostSalesEnv`.
3.  **Network Inventory Management (`network_management.py`):** Multi-period, multi-node inventory system with a network structure (factories, distributors, retailers, markets). Includes `NetInvMgmtBacklogEnv` and `NetInvMgmtLostSalesEnv`.

## Features

*   **Gymnasium Compatible:** Environments adhere to the modern Gymnasium API standard (`reset` returns `obs, info`, `step` returns `obs, reward, terminated, truncated, info`).
*   **Three Core Environments:** Covers single-item, multi-echelon, and network inventory problems.
*   **Backlog & Lost Sales Variants:** Specific environment classes (`*BacklogEnv`, `*LostSalesEnv`) implement these dynamics.
*   **Comprehensive Benchmarking:** Includes dedicated scripts (`benchmark_*.py`) for each environment variant, comparing various agents:
    *   **Baselines:** Random Agent.
    *   **Heuristics:** Relevant heuristics adapted for each environment type (e.g., Order-Up-To, Classic Newsvendor, (s,S) for Newsvendor; Base Stock, Constant Order for multi-echelon/network).
    *   **Stable Baselines3 Agents:** PPO, SAC, TD3, A2C, DDPG (plus example variations like LSTM). Scripts suffixed with `_sb3_rllib.py` explicitly include these. Simpler benchmark scripts might contain a subset.
    *   **Ray RLlib Agents:** PPO, SAC (plus framework to add more). Scripts suffixed with `_sb3_rllib.py` explicitly include these.
*   **Detailed Reporting:** Benchmarks generate:
    *   Summary tables comparing agents on average reward, consistency (std dev, min/max), operational metrics (service level, stockouts, inventory), and time (training/evaluation). Saved to CSV.
    *   Raw results per evaluation episode (CSV).
    *   Detailed step-by-step data (optional, JSON Lines).
    *   Comparison plots (Reward distribution boxplots, Reward vs. Operational Metrics scatter plots, Timing bar charts, RL learning curves). Saved to PNG.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/r2barati/or-gym-inventory.git
    cd or-gym-inventory
    ```
    *(Replace the URL if your repository path is different)*

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    *   Ensure you have Python 3.8+ installed.
    *   Upgrade pip: `python -m pip install --upgrade pip`
    *   Install core requirements and RL libraries:
        ```bash
        # Install base requirements (Gymnasium, Numpy, Scipy, Pandas, NetworkX, Matplotlib, Seaborn)
        pip install -r requirements.txt

        # --- Choose ONE framework for SB3 (torch recommended) ---
        pip install stable-baselines3[extra] torch torchvision torchaudio
        # OR
        # pip install stable-baselines3[extra] tensorflow

        # --- Install Ray RLlib (if running *_sb3_rllib.py scripts) ---
        # Choose ONE framework (can match SB3 or be different, installing both is possible)
        pip install "ray[rllib]" torch torchvision torchaudio
        # OR
        # pip install "ray[rllib]" tensorflow
        ```
        *(Note: `requirements.txt` should contain `gymnasium`, `numpy`, `scipy`, `pandas`, `networkx`, `matplotlib`, `seaborn`)*

## Usage

### 1. Using the Environments Directly

You can import and use the specific environment classes like any standard Gymnasium environment:

```python
import gymnasium as gym
# Make sure the relevant python file (e.g., inventory_management.py) is in your path
from inventory_management import InvManagementLostSalesEnv # Example

# Configuration dictionary (optional, overrides defaults)
env_config = {
    'periods': 50,
    'I0': [50, 50],
    'L': [2, 4],
    'c': [30, 30]
    # ... other parameters specific to the environment ...
}

# Create the environment
# env = InvManagementLostSalesEnv(env_config=env_config)
env = InvManagementLostSalesEnv() # Use defaults

# Standard Gymnasium loop
observation, info = env.reset(seed=42)
# ... (rest of loop as shown in previous README version) ...
env.close()
```

### 2. Running the Benchmarks

The repository includes dedicated benchmark scripts for each environment variant. The scripts ending in `_sb3_rllib.py` contain the most comprehensive set of agents, including both Stable Baselines3 and Ray RLlib implementations. Simpler scripts might focus only on SB3 or heuristics.

**Choose the script corresponding to the environment you want to benchmark:**

*   **Newsvendor (Comprehensive: SB3 + RLlib):**
    ```bash
    python benchmark_newsvendor_sb3_rllib.py
    ```
*   **Inventory Management (Backlog, Comprehensive: SB3 + RLlib):**
    ```bash
    python benchmark_InvManagementBacklogEnv_sb3_rllib.py
    ```
*   **Inventory Management (Lost Sales, Comprehensive: SB3 + RLlib):**
    ```bash
    python benchmark_InvManagementLostSalesEnv_sb3_rllib.py
    ```
*   **Network Inventory Management (Backlog, Comprehensive: SB3 + RLlib):**
    ```bash
    python benchmark_NetInvMgmtBacklogEnv_sb3_rllib.py
    ```
*   **Network Inventory Management (Lost Sales, Comprehensive: SB3 + RLlib):**
    ```bash
    python benchmark_NetInvMgmtLostSalesEnv_sb3_rllib.py
    ```

**Note:** Running the `_sb3_rllib.py` scripts, especially for the `InvManagement` and `NetInvMgmt` environments, will take a **significant amount of time** due to training multiple RL agents. Adjust `RL_TRAINING_TIMESTEPS` and `N_EVAL_EPISODES` within the chosen script for quicker tests or more thorough benchmarking. Set `FORCE_RETRAIN = True` to retrain models even if saved files exist.

Benchmark results (CSV files, PNG plots) will be saved into appropriately named subdirectories (e.g., `benchmark_InvMgmtLS_combined/results/`).

## Environments Overview

*(This section can remain largely the same as the previous version, just ensuring class names match)*

*   **`NewsvendorEnv` (`newsvendor.py`):** Single product, single location, lead time, stochastic demand. Action: order quantity.
*   **`InvManagement...Env` (`inventory_management.py`):** Multi-echelon linear chain. Action: order quantity vector per stage.
    *   `InvManagementBacklogEnv`: Unmet demand is backlogged.
    *   `InvManagementLostSalesEnv`: Unmet demand is lost.
*   **`NetInvMgmt...Env` (`network_management.py`):** Arbitrary network structure (factories, distributors, etc.). Action: order quantity vector per link.
    *   `NetInvMgmtBacklogEnv`: Unmet market demand is backlogged.
    *   `NetInvMgmtLostSalesEnv`: Unmet market demand is lost.

*(Refer to the docstrings within each environment file (`.py`) for details on observation/action spaces, reward calculation, and specific parameters.)*

## Benchmarking Details

The `benchmark_*.py` scripts provide a framework to compare agents. The `_sb3_rllib.py` versions are the most comprehensive.

**Agents Included (in `_sb3_rllib.py` versions):**

*   **Heuristics:**
    *   `RandomAgent`
    *   Environment-specific heuristics (e.g., `ConstantOrderAgent`, `BaseStockAgent`).
*   **Stable Baselines3 (SB3):** PPO, SAC, TD3, A2C, DDPG, plus variations.
*   **Ray RLlib:** PPO, SAC examples included, framework supports adding more.

**Metrics Collected (per Agent):**
*(Same list as before: Avg Reward, Median, Std Dev, Min/Max, Service Level, Stockouts, Inventory, Eval Time, Train Time, Success Rate)*

**Running & Interpreting Results:**
Execute the desired `benchmark_*.py` script. Results are saved in corresponding subdirectories. Analyze the `*_summary.csv` table and the generated `.png` plots for performance comparisons.

## Dependencies

*   Python 3.8+
*   gymnasium
*   numpy
*   scipy
*   pandas
*   networkx (for NetInvMgmt environment)
*   matplotlib
*   seaborn
*   stable-baselines3[extra] (`pip install stable-baselines3[extra] torch # or tensorflow`)
*   ray[rllib] (`pip install "ray[rllib]" torch # or tensorflow`)

See `requirements.txt` for base dependencies.

## References

*(Same as before)*

## License

MIT License
