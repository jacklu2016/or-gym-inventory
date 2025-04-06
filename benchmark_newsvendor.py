import gymnasium as gym
import numpy as np
import pandas as pd
import time
import os
import sys
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- Suppress common warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='stable_baselines3')
warnings.filterwarnings("ignore", category=FutureWarning) # Ignore numpy/pandas FutureWarnings if they appear

# --- RL specific imports ---
SB3_AVAILABLE = False
try:
    from stable_baselines3 import PPO, SAC, TD3, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.noise import NormalActionNoise
    SB3_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not found. RL agent tests will be skipped.")
    print("Install using: pip install stable-baselines3[extra] torch")
    # Define dummy classes if SB3 not found, so the script structure doesn't break
    class DummyModel: pass
    PPO, SAC, TD3, A2C = DummyModel, DummyModel, DummyModel, DummyModel


# --- Environment Import ---
# Assuming newsvendor.py is in the same directory or repo root
try:
    # Adjust path if needed, e.g., if running from parent dir of repo
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    repo_path = os.path.join(script_dir, 'or-gym-inventory') # Assumes script is outside repo dir
    if not os.path.exists(os.path.join(script_dir, 'newsvendor.py')) and os.path.exists(repo_path):
         if repo_path not in sys.path:
             print(f"Adding repo path to sys.path: {repo_path}")
             sys.path.append(repo_path)

    from newsvendor import NewsvendorEnv
    print("Successfully imported NewsvendorEnv.")
except ImportError as e:
    print(f"Error importing NewsvendorEnv: {e}")
    print("Ensure newsvendor.py is accessible (e.g., in the same directory or added to PYTHONPATH).")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

# --- Configuration ---
N_EVAL_EPISODES = 30       # Reduced for quicker testing, increase for more robust results (e.g., 50-100)
RL_TRAINING_TIMESTEPS = 30000 # Reduced for quicker testing, increase significantly for better RL performance (e.g., 100k-500k+)
ENV_ID = "NewsvendorBench"
SEED_OFFSET = 2000          # Start evaluation seeds from here
ENV_CONFIG_EVAL = {         # Consistent parameters for evaluation
    'lead_time': 5,
    'step_limit': 50,
    'p_max': 100.0,
    'h_max': 5.0,
    'k_max': 10.0,
    'mu_max': 200.0,
}

# --- Agent Definitions ---

class BaseAgent:
    """Base class for agents"""
    def __init__(self, name="BaseAgent"):
        self.name = name
        self.training_time = 0.0

    def get_action(self, observation: np.ndarray, env: NewsvendorEnv) -> np.ndarray:
        """Returns the action to take given the observation and environment instance."""
        raise NotImplementedError

    def train(self, env_config: dict, total_timesteps: int):
        """Optional training method for agents that require it."""
        print(f"Agent {self.name} does not require training.")
        pass # Default: no training needed

    def get_training_time(self) -> float:
        return self.training_time


class RandomAgent(BaseAgent):
    """Agent that takes random actions."""
    def __init__(self):
        super().__init__(name="Random")

    def get_action(self, observation: np.ndarray, env: NewsvendorEnv) -> np.ndarray:
        return env.action_space.sample()


class OrderUpToHeuristicAgent(BaseAgent):
    """ Heuristic: Order up to expected demand over lead time + 1 period. """
    def __init__(self, safety_factor=1.0):
        super().__init__(name=f"OrderUpTo_SF={safety_factor:.1f}")
        self.safety_factor = safety_factor

    def get_action(self, observation: np.ndarray, env: NewsvendorEnv) -> np.ndarray:
        mu = observation[4]
        pipeline_inventory = observation[5:]
        lead_time = env.lead_time
        target_demand = mu * (lead_time + 1) * self.safety_factor
        current_position = pipeline_inventory.sum()
        order_qty = max(0, target_demand - current_position)
        order_qty = np.clip(order_qty, env.action_space.low[0], env.action_space.high[0])
        return np.array([order_qty], dtype=env.action_space.dtype)

class ClassicNewsvendorAgent(BaseAgent):
    """ Heuristic: Uses classic Newsvendor critical ratio, adapted for lead time. """
    def __init__(self, cr_method='k_vs_h', safety_factor=1.0):
        # cr_method: 'k_vs_h' uses k / (h+k), 'profit_margin' uses (p-c)/(p-c+k) [needs check]
        super().__init__(name=f"ClassicNV_SF={safety_factor:.1f}_{cr_method}")
        self.cr_method = cr_method
        self.safety_factor = safety_factor

    def get_action(self, observation: np.ndarray, env: NewsvendorEnv) -> np.ndarray:
        price, cost, h, k, mu = observation[:5]
        pipeline_inventory = observation[5:]
        lead_time = env.lead_time

        # Fallback logic
        fallback = False
        if self.cr_method == 'profit_margin':
             underage_cost = price - cost + k # Often simplified to just p-c or k
             overage_cost = h # Often simplified to just h or c
             if underage_cost + overage_cost <= 1e-6 or underage_cost <=0 or overage_cost <=0:
                 fallback = True
             else:
                 critical_ratio = underage_cost / (underage_cost + overage_cost)
        elif self.cr_method == 'k_vs_h':
             if h + k <= 1e-6 or k < 0 or h < 0: # Avoid division by zero or nonsensical CR
                 fallback = True
             else:
                 critical_ratio = k / (h + k)
        else: # Default to k_vs_h or fallback
             if h + k <= 1e-6 or k < 0 or h < 0: fallback = True
             else: critical_ratio = k / (h + k)

        if fallback:
            # Fallback to simple OrderUpTo if CR calculation fails
            target_demand = mu * (lead_time + 1)
            current_position = pipeline_inventory.sum()
            order_qty = max(0, target_demand - current_position)
        else:
            # Demand distribution over lead_time + 1 periods
            effective_mu = mu * (lead_time + 1) * self.safety_factor
            # Find the quantile (optimal target inventory level)
            target_inv_level = poisson.ppf(critical_ratio, mu=max(1e-6, effective_mu)) # Ensure mu > 0 for ppf

            # Order-up-to logic
            current_position = pipeline_inventory.sum()
            order_qty = max(0, target_inv_level - current_position)

        # Clip by environment limits
        order_qty = np.clip(order_qty, env.action_space.low[0], env.action_space.high[0])
        return np.array([order_qty], dtype=env.action_space.dtype)


class SB3AgentWrapper(BaseAgent):
    """ Wrapper for Stable Baselines3 agents. """
    def __init__(self, model_class, policy="MlpPolicy", name="SB3_Agent", train_kwargs=None, model_kwargs=None):
        super().__init__(name=name)
        if not SB3_AVAILABLE:
             raise ImportError("Stable Baselines3 is not available.")
        self.model_class = model_class
        self.policy = policy
        self.model = None
        self.train_kwargs = train_kwargs if train_kwargs is not None else {}
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        # Add action noise for off-policy algorithms like SAC/TD3 if desired
        if model_class in [SAC, TD3]:
            action_dim = 1 # Newsvendor action space is Box(1,)
            noise_std = 0.1 * (ENV_CONFIG_EVAL.get('max_order_quantity', 2000) / 2) # Heuristic noise scale
            self.model_kwargs['action_noise'] = NormalActionNoise(mean=np.zeros(action_dim), sigma=noise_std * np.ones(action_dim))


    def train(self, env_config: dict, total_timesteps: int):
        print(f"Training {self.name}...")
        start_time = time.time()
        # Create env for training
        # Using a simple function for make_vec_env
        def _create_env():
            return NewsvendorEnv(**env_config)

        vec_env = make_vec_env(_create_env, n_envs=1) # Use n_envs > 1 for parallel training if desired

        try:
            self.model = self.model_class(self.policy, vec_env, verbose=0, **self.model_kwargs)
            self.model.learn(total_timesteps=total_timesteps, **self.train_kwargs)
            self.training_time = time.time() - start_time
            print(f"Training for {self.name} finished in {self.training_time:.2f} seconds.")
        except Exception as e:
            print(f"!!! ERROR during training {self.name}: {e}")
            import traceback
            traceback.print_exc()
            self.model = None # Ensure model is None if training failed
            self.training_time = time.time() - start_time # Record time even if failed
        finally:
             # Close the VecEnv
            vec_env.close()


    def get_action(self, observation: np.ndarray, env: NewsvendorEnv) -> np.ndarray:
        if self.model is None:
            # Return a default action (e.g., zero) if training failed or model not set
            # print(f"Warning: Model for {self.name} not available. Returning zero action.")
            return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

        action, _states = self.model.predict(observation, deterministic=True)
        return action

# --- Evaluation Function ---

def evaluate_agent(agent: BaseAgent, env_config: dict, n_episodes: int, seed_offset: int = 0) -> pd.DataFrame:
    """Evaluates an agent over n_episodes and returns results dataframe."""
    results = []
    total_eval_time = 0
    eval_env = NewsvendorEnv(**env_config) # Create dedicated env for evaluation

    print(f"\nEvaluating {agent.name} for {n_episodes} episodes...")

    for i in range(n_episodes):
        episode_seed = seed_offset + i
        try:
            obs, info = eval_env.reset(seed=episode_seed)
            terminated = False
            truncated = False
            episode_reward = 0
            episode_steps = 0
            start_time = time.perf_counter()

            while not terminated and not truncated:
                action = agent.get_action(obs, eval_env)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_steps += 1

            end_time = time.perf_counter()
            episode_time = end_time - start_time
            total_eval_time += episode_time

            results.append({
                "Agent": agent.name,
                "Episode": i + 1,
                "TotalReward": episode_reward,
                "Steps": episode_steps,
                "Time": episode_time,
                "Seed": episode_seed
            })
            # Print progress less frequently for shorter runs
            if n_episodes <= 20 or (i + 1) % max(1, n_episodes // 5) == 0:
                 print(f"  Episode {i+1}/{n_episodes} finished. Reward: {episode_reward:.2f}")

        except Exception as e:
             print(f"!!! ERROR during evaluation episode {i+1} for {agent.name}: {e}")
             # Record failure if desired, or just skip episode
             results.append({
                "Agent": agent.name, "Episode": i + 1, "TotalReward": np.nan,
                "Steps": 0, "Time": 0, "Seed": episode_seed, "Error": str(e)
             })

    eval_env.close() # Close the evaluation env

    if not results: # Handle case where all episodes failed
        print(f"Evaluation FAILED for {agent.name}. No results collected.")
        return pd.DataFrame()

    avg_eval_time = total_eval_time / len(results) if results else 0
    print(f"Evaluation finished for {agent.name}. Average successful episode time: {avg_eval_time:.4f}s")
    return pd.DataFrame(results)

# --- Plotting Function ---

def plot_results(df_summary: pd.DataFrame, df_raw: pd.DataFrame):
    """Generates and saves comparison plots."""
    print("\nGenerating plots...")
    n_agents = df_summary.shape[0]

    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style

    # 1. Box Plot of Total Rewards per Agent
    plt.figure(figsize=(10, max(6, n_agents * 0.5))) # Adjust height based on agents
    sns.boxplot(data=df_raw, x="TotalReward", y="Agent", palette="viridis", showfliers=False, # Hide outliers for better scale
                order=df_summary.sort_values("AvgReward", ascending=False).index) # Order by mean reward
    plt.title(f"Distribution of Total Rewards per Episode ({N_EVAL_EPISODES} Episodes)")
    plt.xlabel("Total Reward (Higher is Better)")
    plt.ylabel("Agent")
    plt.tight_layout()
    plt.savefig("newsvendor_benchmark_rewards_boxplot.png")
    print("Saved rewards boxplot to newsvendor_benchmark_rewards_boxplot.png")
    plt.close() # Close the figure

    # 2. Bar Chart of Average Evaluation Times (Eval Time Per Ep)
    df_summary_sorted_eval_time = df_summary.sort_values("AvgTimePerEp", ascending=True) # Sort by eval time
    plt.figure(figsize=(10, max(6, n_agents * 0.4)))
    index = np.arange(len(df_summary_sorted_eval_time))
    plt.barh(index, df_summary_sorted_eval_time["AvgTimePerEp"], color='skyblue', log=True) # Horizontal bar, log scale
    plt.yticks(index, df_summary_sorted_eval_time.index)
    plt.xlabel('Average Evaluation Time per Episode (s) - Log Scale')
    plt.ylabel('Agent')
    plt.title('Average Evaluation Time per Episode (Log Scale)')
    plt.tight_layout()
    plt.savefig("newsvendor_benchmark_eval_time_log.png")
    print("Saved evaluation time bar chart to newsvendor_benchmark_eval_time_log.png")
    plt.close() # Close the figure

    # 3. Bar chart for Training Time (only relevant agents)
    df_train = df_summary[df_summary["TrainingTime(s)"] > 1].sort_values("TrainingTime(s)", ascending=True) # Filter agents with training time > 1s
    if not df_train.empty:
        plt.figure(figsize=(8, max(4, len(df_train) * 0.5)))
        index_train = np.arange(len(df_train))
        plt.barh(index_train, df_train["TrainingTime(s)"], color='lightcoral') # Horizontal bar
        plt.yticks(index_train, df_train.index)
        plt.xlabel('Total Training Time (s)')
        plt.ylabel('Agent')
        plt.title('Training Time for RL Agents')
        plt.tight_layout()
        plt.savefig("newsvendor_benchmark_train_time.png")
        print("Saved training time bar chart to newsvendor_benchmark_train_time.png")
        plt.close() # Close the figure
    else:
        print("Skipping training time plot as no agents had significant training time.")

    plt.close('all') # Close any remaining figures


# --- Main Benchmarking ---

if __name__ == "__main__":
    all_results_list = []
    agent_objects = {} # Store agent instances for later access if needed

    # --- Define Agents to Benchmark ---
    print("Defining agents...")
    agents_to_run = [
        ("Random", RandomAgent()),
        ("OrderUpTo_SF=1.0", OrderUpToHeuristicAgent(safety_factor=1.0)),
        ("OrderUpTo_SF=1.2", OrderUpToHeuristicAgent(safety_factor=1.2)),
        ("OrderUpTo_SF=0.8", OrderUpToHeuristicAgent(safety_factor=0.8)),
        ("ClassicNV_SF=1.0_k_vs_h", ClassicNewsvendorAgent(cr_method='k_vs_h', safety_factor=1.0)),
        # ("ClassicNV_SF=1.2_k_vs_h", ClassicNewsvendorAgent(cr_method='k_vs_h', safety_factor=1.2)),
        # ("ClassicNV_SF=1.0_profit", ClassicNewsvendorAgent(cr_method='profit_margin', safety_factor=1.0)),
    ]

    # --- Add SB3 Agents if available ---
    if SB3_AVAILABLE:
        # Define SB3 agents with their classes and specific names
        # Add model_kwargs if needed (e.g., learning_rate)
        sb3_agents_def = [
            ("PPO", PPO, {}),
            ("SAC", SAC, {}),
            ("TD3", TD3, {}),
            ("A2C", A2C, {}),
            # Add more SB3 agents or variations here
            # e.g., ("PPO_tuned", PPO, {'model_kwargs': {'learning_rate': 1e-4}})
        ]
        for name, model_cls, params in sb3_agents_def:
             agents_to_run.append((name, SB3AgentWrapper(model_cls, name=name, **params)))
    else:
        print("\nSkipping RL agent definitions as Stable Baselines3 is not available.")


    # --- Train RL Agents ---
    print("\n--- Training Phase ---")
    for name, agent in agents_to_run:
        agent_objects[name] = agent # Store instance
        if isinstance(agent, SB3AgentWrapper):
            agent.train(ENV_CONFIG_EVAL, total_timesteps=RL_TRAINING_TIMESTEPS)
        else:
            agent.train(ENV_CONFIG_EVAL, 0) # Call train even for non-RL for consistency

    # --- Run Evaluation ---
    print("\n--- Evaluation Phase ---")
    for name, agent in agents_to_run:
        # Retrieve the potentially trained agent object
        current_agent = agent_objects[name]
        df_agent_results = evaluate_agent(current_agent, ENV_CONFIG_EVAL, N_EVAL_EPISODES, seed_offset=SEED_OFFSET)
        if not df_agent_results.empty:
             all_results_list.append(df_agent_results)

    # --- Process and Report Results ---
    if not all_results_list:
        print("\nNo evaluation results collected. Exiting.")
    else:
        results_df_raw = pd.concat(all_results_list, ignore_index=True)

        print("\n--- Benchmark Summary ---")

        # Calculate summary statistics per agent, handling potential NaN rewards from failed episodes
        summary = results_df_raw.dropna(subset=['TotalReward']).groupby("Agent").agg(
            AvgReward=("TotalReward", "mean"),
            MedianReward=("TotalReward", "median"), # Median can be more robust to outliers
            StdReward=("TotalReward", "std"),
            MinReward=("TotalReward", "min"),
            MaxReward=("TotalReward", "max"),
            AvgSteps=("Steps", "mean"),
            AvgTimePerEp=("Time", "mean"),
            SuccessfulEpisodes=("Episode", "count") # Count non-NaN episodes
        )
        # Add training time information using the stored agent objects
        summary["TrainingTime(s)"] = summary.index.map(lambda name: agent_objects[name].get_training_time()).fillna(0.0)

        # Add total evaluation episodes attempted
        total_episodes_attempted = results_df_raw.groupby("Agent")["Episode"].count()
        summary["EpisodesAttempted"] = total_episodes_attempted
        summary["SuccessRate(%)"] = (summary["SuccessfulEpisodes"] / summary["EpisodesAttempted"]) * 100


        # Sort and reorder columns
        summary = summary.sort_values(by="AvgReward", ascending=False)
        summary = summary[[
            "AvgReward", "MedianReward", "StdReward", "MinReward", "MaxReward",
            "AvgTimePerEp", "TrainingTime(s)",
            "AvgSteps", "SuccessfulEpisodes", "EpisodesAttempted", "SuccessRate(%)"
        ]]

        # Format for display
        pd.set_option('display.float_format', lambda x: '%.2f' % x) # Format floats
        print(summary)

        # --- Save Results ---
        try:
            results_df_raw.to_csv("newsvendor_benchmark_raw.csv", index=False)
            summary.to_csv("newsvendor_benchmark_summary.csv")
            print("\nRaw results saved to newsvendor_benchmark_raw.csv")
            print("Summary saved to newsvendor_benchmark_summary.csv")
        except Exception as e:
            print(f"\nError saving results to CSV: {e}")

        # --- Generate Plots ---
        try:
            plot_results(summary, results_df_raw)
        except Exception as e:
            print(f"\nError generating plots: {e}")
            print("Ensure matplotlib and seaborn are installed.")

    print("\nBenchmark script finished.")
