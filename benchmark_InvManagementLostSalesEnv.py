# benchmark_invmgmt_ls.py
import gymnasium as gym
from gymnasium import spaces # Import spaces explicitly
import numpy as np
import pandas as pd
import time
import os
import sys
import json
import glob
from scipy.stats import poisson # Keep for potential heuristic calculations
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- Suppress common warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- RL specific imports ---
SB3_AVAILABLE = False
try:
    from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG # Added DDPG
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise # Use Vectorized for multi-dim
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    from stable_baselines3.common.logger import configure # For setting log format if needed
    SB3_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not found. RL agent tests will be skipped.")
    print("Install using: pip install stable-baselines3[extra] torch")
    class DummyModel: pass
    PPO, SAC, TD3, A2C, DDPG = DummyModel, DummyModel, DummyModel, DummyModel, DummyModel


# --- Environment Import ---
ENV_MODULE_NAME = "inventory_management"
# <<<<<<<<<<<<<<<<< CHANGE: Target Lost Sales Env >>>>>>>>>>>>>>>>>
ENV_CLASS_NAME = "InvManagementLostSalesEnv"

try:
    # Adjust path finding if necessary (same logic as before)
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    if os.path.exists(os.path.join(script_dir, f'{ENV_MODULE_NAME}.py')):
         print(f"Found {ENV_MODULE_NAME}.py in current directory.")
         module_path = script_dir
    else:
        repo_path_guess = os.path.abspath(os.path.join(script_dir, '..'))
        module_path_in_repo = os.path.join(script_dir, 'or-gym-inventory', f'{ENV_MODULE_NAME}.py')
        module_path_in_parent = os.path.join(repo_path_guess, 'or-gym-inventory', f'{ENV_MODULE_NAME}.py')

        if os.path.exists(module_path_in_repo):
             module_path = os.path.join(script_dir, 'or-gym-inventory')
        elif os.path.exists(module_path_in_parent):
             module_path = os.path.join(repo_path_guess, 'or-gym-inventory')
        else:
             module_path = None
             print(f"Warning: Could not automatically find 'or-gym-inventory/{ENV_MODULE_NAME}.py'.")

        if module_path and module_path not in sys.path:
            print(f"Adding path to sys.path: {module_path}")
            sys.path.append(module_path)

    # Import the specific environment class
    module = __import__(ENV_MODULE_NAME)
    InvManagementEnvClass = getattr(module, ENV_CLASS_NAME)
    print(f"Successfully imported {ENV_CLASS_NAME} from {ENV_MODULE_NAME}.")

except ImportError as e:
    print(f"Error importing {ENV_CLASS_NAME} from {ENV_MODULE_NAME}: {e}")
    print("Ensure the file is accessible.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


# --- Configuration ---
# Benchmark settings
N_EVAL_EPISODES = 30
RL_TRAINING_TIMESTEPS = 50000 # Keep consistent with backlog test for comparison (or increase both)
SEED_OFFSET = 5000 # Use different offset
FORCE_RETRAIN = False
COLLECT_STEP_DETAILS = True
N_ENVS_TRAIN = 4

# <<<<<<<<<<<<<<<<< CHANGE: Paths for Lost Sales >>>>>>>>>>>>>>>>>
ENV_NAME_SHORT = "InvMgmtLS" # Changed prefix
LOG_DIR = f"./sb3_logs_{ENV_NAME_SHORT}/"
MODEL_DIR = f"./sb3_models_{ENV_NAME_SHORT}/"
RESULTS_DIR = f"./benchmark_results_{ENV_NAME_SHORT}/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Environment configuration for evaluation/training
# Using default parameters but ensuring backlog=False is implicitly handled by the class
ENV_CONFIG = {
    'periods': 50,
    # Add overrides here if needed, matching the backlog env config if desired for direct comparison
    # 'I0': [50, 50],
    # 'c': [20, 25],
    # 'L': [2, 3],
    # 'dist_param': {'mu': 15}
}


# --- Agent Definitions ---
# BaseAgent, RandomAgent, BaseStockAgent, SB3AgentWrapper can be reused directly
# They adapt based on the env's action/observation space and dynamics.
# Copying them here for completeness of the script file.

class BaseAgent:
    def __init__(self, name="BaseAgent"): self.name=name; self.training_time=0.0
    def get_action(self, o, e): raise NotImplementedError
    def train(self, ec, tt, sp): print(f"Agent {self.name} does not require training.")
    def load(self, p): print(f"Agent {self.name} does not support loading.")
    def get_training_time(self): return self.training_time

class RandomAgent(BaseAgent):
    def __init__(self): super().__init__(name="Random")
    def get_action(self, o, e): return e.action_space.sample().astype(e.action_space.dtype)

class BaseStockAgent(BaseAgent):
    def __init__(self, safety_factor=1.0):
        super().__init__(name=f"BaseStock_SF={safety_factor:.1f}")
        self.safety_factor = safety_factor
    def get_action(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        if not all(hasattr(env.unwrapped, attr) for attr in ['num_stages', 'lead_time', 'dist_param', 'lt_max', 'I', 'action_log', 'period']):
             print(f"Warning: Env missing attrs for {self.name}. Returning random."); return env.action_space.sample().astype(env.action_space.dtype)
        env_core = env.unwrapped # Access attributes of the core env if wrapped by Monitor etc.
        num_stages_inv = env_core.num_stages - 1
        lead_times = env_core.lead_time
        mu = env_core.dist_param.get('mu', 10)
        current_on_hand = observation[:num_stages_inv]
        inventory_position = current_on_hand.copy()
        t = env_core.period
        action_log = env_core.action_log
        pipeline_start_idx = num_stages_inv
        for i in range(num_stages_inv):
            L_i = lead_times[i];
            if L_i == 0: continue
            pipeline_for_stage_i = 0
            start_log_idx = max(0, t - L_i)
            if t > 0 and start_log_idx < t:
                 orders_in_pipeline = action_log[start_log_idx : t, i]
                 pipeline_for_stage_i = orders_in_pipeline.sum()
            inventory_position[i] += pipeline_for_stage_i
        target_levels = (lead_times + 1) * mu * self.safety_factor
        order_quantities = np.maximum(0, target_levels - inventory_position)
        order_quantities = np.clip(order_quantities, env.action_space.low, env.action_space.high)
        return order_quantities.astype(env.action_space.dtype)

class SB3AgentWrapper(BaseAgent):
    def __init__(self, model_class, policy="MlpPolicy", name="SB3_Agent", train_kwargs=None, model_kwargs=None):
        super().__init__(name=name)
        if not SB3_AVAILABLE: raise ImportError("Stable Baselines3 is not available.")
        self.model_class = model_class
        self.policy = policy
        self.model = None
        self.train_kwargs = train_kwargs if train_kwargs is not None else {}
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.log_dir = os.path.join(LOG_DIR, self.name) # Uses LOG_DIR specific to this script
        self.save_path = os.path.join(MODEL_DIR, f"{self.name}.zip") # Uses MODEL_DIR specific to this script

        if model_class in [SAC, TD3, DDPG]:
             try: # Need action dim for noise
                 temp_env = InvManagementEnvClass(**ENV_CONFIG); action_dim = temp_env.action_space.shape[0]; action_range = temp_env.action_space.high; temp_env.close()
                 noise_std = 0.1 * action_range / 2; noise_std[action_range <= 0] = 0.1 # Handle zero capacity cases if any
                 self.model_kwargs['action_noise'] = NormalActionNoise(mean=np.zeros(action_dim), sigma=noise_std)
                 print(f"Applied NormalActionNoise for {name} with mean std dev approx: {noise_std.mean():.2f}")
             except Exception as e_noise: print(f"Warning: Failed to setup action noise for {name}: {e_noise}")

    def train(self, env_config: dict, total_timesteps: int, save_path_prefix: str=""):
        print(f"Training {self.name}...")
        start_time = time.time()
        train_log_dir = os.path.join(self.log_dir, "train_logs")
        os.makedirs(train_log_dir, exist_ok=True)
        save_path = os.path.join(MODEL_DIR, f"{save_path_prefix}{self.name}.zip")
        best_model_save_path = os.path.join(MODEL_DIR, f"{save_path_prefix}{self.name}_best")
        os.makedirs(best_model_save_path, exist_ok=True)

        if not FORCE_RETRAIN and os.path.exists(save_path):
            print(f"Loading existing model for {self.name} from {save_path}"); self.load(save_path)
            if hasattr(self.model, '_total_timesteps') and self.model._total_timesteps >= total_timesteps:
                self.training_time = 0; print("Model already trained."); return

        def _create_env(rank: int, seed: int = 0):
            def _init():
                env = InvManagementEnvClass(**env_config); env = Monitor(env, filename=os.path.join(train_log_dir, f"monitor_{rank}")); env.reset(seed=seed + rank); return env
            return _init
        vec_env = DummyVecEnv([_create_env(0, self.model_kwargs.get('seed', 0))]) # Start with Dummy for stability
        # if N_ENVS_TRAIN > 1: try: vec_env = SubprocVecEnv([_create_env(i, self.model_kwargs.get('seed', 0)) for i in range(N_ENVS_TRAIN)], start_method='fork'); except: print("SubprocVecEnv failed, using DummyVecEnv."); pass

        eval_env_callback = Monitor(InvManagementEnvClass(**env_config))
        eval_callback = EvalCallback(eval_env_callback, best_model_save_path=best_model_save_path, log_path=os.path.join(self.log_dir, "eval_logs"), eval_freq=max(5000, total_timesteps // 10), n_eval_episodes=5, deterministic=True, render=False)
        try:
            model_seed = self.model_kwargs.get('seed', None)
            self.model = self.model_class(self.policy, vec_env, verbose=0, seed=model_seed, **self.model_kwargs)
            print(f"Starting training ({self.name}) for {total_timesteps} timesteps...")
            self.model.learn(total_timesteps=total_timesteps, callback=eval_callback, log_interval=50, **self.train_kwargs)
            self.training_time = time.time() - start_time; print(f"Training ({self.name}) finished: {self.training_time:.2f}s.")
            self.model.save(save_path); print(f"Final model saved to {save_path}")
            try:
                 best_model_path = os.path.join(best_model_save_path, "best_model.zip")
                 if os.path.exists(best_model_path): print(f"Loading best model from {best_model_path}..."); self.model = self.model_class.load(best_model_path)
                 else: print("Best model not found, using final model.")
            except Exception as e_load: print(f"Warning: Could not load best model. Error: {e_load}")
        except Exception as e: print(f"!!! ERROR training {self.name}: {e}"); import traceback; traceback.print_exc(); self.model = None; self.training_time = time.time() - start_time
        finally: vec_env.close(); eval_env_callback.close()

    def load(self, path: str):
         if not SB3_AVAILABLE: return
         try: print(f"Loading model {self.name} from {path}"); self.model = self.model_class.load(path)
         except Exception as e: print(f"!!! ERROR loading {self.name} from {path}: {e}"); self.model = None

    def get_action(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        if self.model is None: action = env.action_space.sample(); return action.astype(env.action_space.dtype)
        obs_input = observation.astype(np.float32) # Ensure float32 for model
        action, _ = self.model.predict(obs_input, deterministic=True)
        return action.astype(env.action_space.dtype) # Cast back to env dtype


# --- Evaluation Function (Adapted for InvManagement Info Dict) ---
# (evaluate_agent function remains the same as the one provided for InvManagement in the previous step)
def evaluate_agent(agent: BaseAgent,
                   env_config: dict,
                   n_episodes: int,
                   seed_offset: int = 0,
                   # fixed_params not implemented for InvManagement Env yet
                   collect_details: bool = True
                   ) -> Dict:
    # ... (Keep the function from the previous response - benchmark_invmanagement.py) ...
    # This function correctly extracts metrics like demand_realized, sales, unfulfilled, ending_inventory
    # from the info dict provided by the InvManagement env's step method.
    episode_summaries = []
    all_step_details = []
    total_eval_time = 0.0
    # <<<<<<<<<<<<<<<<< Ensure correct Env Class is used >>>>>>>>>>>>>>>>>
    eval_env = InvManagementEnvClass(**env_config)

    print(f"\nEvaluating {agent.name} for {n_episodes} episodes ({ENV_CLASS_NAME})...")
    successful_episodes = 0
    for i in range(n_episodes):
        episode_seed = seed_offset + i
        episode_step_details = []
        try:
            obs, info = eval_env.reset(seed=episode_seed) # No options needed yet
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0
            episode_demand_retailer = 0.0
            episode_sales_retailer = 0.0
            episode_stockout_qty_retailer = 0.0
            episode_inventory_sum_all_stages = 0.0

            start_time = time.perf_counter()
            while not terminated and not truncated:
                action = agent.get_action(obs, eval_env)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward; episode_steps += 1
                demand_this_step = info.get('demand_realized', 0); episode_demand_retailer += demand_this_step
                sales_vector = info.get('sales', np.zeros(eval_env.num_stages if hasattr(eval_env, 'num_stages') else 1))
                sales_at_retailer = sales_vector[0]; episode_sales_retailer += sales_at_retailer
                unfulfilled_vector = info.get('unfulfilled', np.zeros(eval_env.num_stages if hasattr(eval_env, 'num_stages') else 1))
                stockout_at_retailer = unfulfilled_vector[0]; episode_stockout_qty_retailer += stockout_at_retailer
                ending_inv_vector = info.get('ending_inventory', np.zeros(eval_env.action_space.shape[0] if hasattr(eval_env, 'action_space') else 1))
                episode_inventory_sum_all_stages += np.sum(np.maximum(0, ending_inv_vector))

                if collect_details:
                     step_data = {'step': episode_steps, 'reward': reward, 'action': action.tolist(),
                                  'demand_retailer': demand_this_step, 'sales_retailer': sales_at_retailer,
                                  'stockout_retailer': stockout_at_retailer, 'ending_inv_vector': ending_inv_vector.tolist()}
                     episode_step_details.append(step_data)
            end_time = time.perf_counter(); episode_time = end_time - start_time; total_eval_time += episode_time
            avg_ending_inv_all = episode_inventory_sum_all_stages / episode_steps if episode_steps > 0 else 0
            service_level_retailer = episode_sales_retailer / max(1e-6, episode_demand_retailer) if episode_demand_retailer > 1e-6 else 1.0
            ep_summary = {"Agent": agent.name, "Episode": i + 1, "TotalReward": episode_reward,"Steps": episode_steps, "Time": episode_time, "Seed": episode_seed,"AvgServiceLevel": service_level_retailer,"TotalStockoutQty": episode_stockout_qty_retailer,"AvgEndingInv": avg_ending_inv_all,"Error": None}
            episode_summaries.append(ep_summary); all_step_details.append(episode_step_details); successful_episodes += 1
            if n_episodes <= 20 or (i + 1) % max(1, n_episodes // 5) == 0: print(f"  Ep {i+1}/{n_episodes}: Reward={episode_reward:.2f}, ServiceLvL={service_level_retailer:.2%}")
        except Exception as e: print(f"!!! ERROR ep {i+1} for {agent.name}: {e}"); import traceback; traceback.print_exc(); episode_summaries.append({"Agent": agent.name, "Episode": i + 1, "TotalReward": np.nan, "Steps": 0,"Time": 0, "Seed": episode_seed, "AvgServiceLevel": np.nan,"TotalStockoutQty": np.nan, "AvgEndingInv": np.nan, "Error": str(e)}); all_step_details.append([])
    eval_env.close()
    if successful_episodes == 0: print(f"Eval FAILED for {agent.name}."); return {'summary': pd.DataFrame(), 'details': []}
    avg_eval_time = total_eval_time / successful_episodes; print(f"Eval finished for {agent.name}. Avg time: {avg_eval_time:.4f}s")
    return {'summary': pd.DataFrame(episode_summaries), 'details': all_step_details}


# --- Results Processing ---
# (process_and_report_results function can be reused)
def process_and_report_results(all_eval_results: List[Dict], agent_objects: Dict):
    # ... (Keep the function from benchmark_newsvendor_advanced.py) ...
    # It will use the correct RESULTS_DIR and ENV_NAME_SHORT due to global scope
    if not all_eval_results: print("No results."); return None, None
    all_summaries_list = [res['summary'] for res in all_eval_results if 'summary' in res and not res['summary'].empty]
    if not all_summaries_list: print("No summaries."); return None, None
    results_df_raw_summary = pd.concat(all_summaries_list, ignore_index=True)
    print("\n--- Benchmark Summary ---")
    summary = results_df_raw_summary.dropna(subset=['TotalReward']).groupby("Agent").agg(
        AvgReward=("TotalReward", "mean"), MedianReward=("TotalReward", "median"), StdReward=("TotalReward", "std"),
        MinReward=("TotalReward", "min"), MaxReward=("TotalReward", "max"), AvgServiceLevel=("AvgServiceLevel", "mean"),
        AvgStockoutQty=("TotalStockoutQty", "mean"), AvgEndInv=("AvgEndingInv", "mean"), AvgTimePerEp=("Time", "mean"),
        SuccessfulEpisodes=("Episode", "count"))
    summary["TrainingTime(s)"] = summary.index.map(lambda name: agent_objects.get(name, BaseAgent(name)).get_training_time()).fillna(0.0)
    total_episodes_attempted = results_df_raw_summary.groupby("Agent")["Episode"].count()
    summary["EpisodesAttempted"] = total_episodes_attempted
    summary["SuccessRate(%)"] = (summary["SuccessfulEpisodes"] / summary["EpisodesAttempted"]) * 100
    summary = summary.sort_values(by="AvgReward", ascending=False)
    summary = summary[["AvgReward", "MedianReward", "StdReward", "MinReward", "MaxReward", "AvgServiceLevel",
                       "AvgStockoutQty", "AvgEndInv", "AvgTimePerEp", "TrainingTime(s)", "SuccessfulEpisodes",
                       "EpisodesAttempted", "SuccessRate(%)"]]
    pd.set_option('display.float_format', lambda x: '%.2f' % x); print(summary)
    raw_summary_path = os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_raw_summary.csv")
    summary_path = os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_summary.csv")
    details_path = os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_step_details.jsonl")
    try:
        results_df_raw_summary.to_csv(raw_summary_path, index=False); summary.to_csv(summary_path)
        print(f"\nRaw summary saved to {raw_summary_path}"); print(f"Summary saved to {summary_path}")
        if COLLECT_STEP_DETAILS:
            print(f"Saving step details to {details_path}...")
            with open(details_path, 'w') as f:
                for i, agent_results in enumerate(all_eval_results):
                    agent_name = agent_results['summary']['Agent'].iloc[0] if 'summary' in agent_results and not agent_results['summary'].empty else f"Unknown_{i}"
                    for ep_num, steps in enumerate(agent_results.get('details', [])):
                        for step_data in steps:
                            step_data['agent'] = agent_name; step_data['episode'] = ep_num + 1
                            serializable_data = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in step_data.items()}
                            f.write(json.dumps(serializable_data) + '\n')
            print("Step details saved.")
    except Exception as e: print(f"\nError saving results: {e}")
    return summary, results_df_raw_summary


# --- Plotting Functions ---
# (plot_learning_curves and plot_benchmark_results can be reused, just update titles/filenames)
def plot_learning_curves(log_dirs: Dict[str, str], title: str = "RL Learning Curves"):
    # ... (Keep the function from benchmark_newsvendor_advanced.py) ...
    if not SB3_AVAILABLE: return
    plt.figure(figsize=(12, 7)); plt.title(title); plt.xlabel("Timesteps"); plt.ylabel("Reward")
    plotted=False
    for name, log_dir in log_dirs.items():
        monitor_files = glob.glob(os.path.join(log_dir, "*logs", "monitor.*.csv"));
        if not monitor_files: print(f"Warn: No logs for {name}"); continue
        monitor_path = os.path.dirname(monitor_files[0])
        try:
            results = load_results(monitor_path);
            if len(results['r']) > 0:
                x, y = ts2xy(results, 'timesteps');
                if len(x) > 10: y = results.rolling(window=10).mean()['r'] # Smooth
                plt.plot(x, y, label=name); plotted=True
            else: print(f"Warn: Log {name} empty.")
        except Exception as e: print(f"Error plot logs {name}: {e}")
    if plotted: plt.legend(); plt.tight_layout(); p=os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_learning_curves.png"); plt.savefig(p); print(f"Saved learning curves: {p}"); plt.close()
    else: print("Skip learning curves - no data.")

def plot_benchmark_results(df_summary: pd.DataFrame, df_raw_summary: pd.DataFrame):
    # ... (Keep the function from benchmark_newsvendor_advanced.py, use ENV_NAME_SHORT) ...
    if df_summary is None or df_raw_summary is None: print("Skip plot: no data."); return
    print("\nGenerating comparison plots..."); n=df_summary.shape[0]; plt.style.use('seaborn-v0_8-darkgrid')
    df_s = df_summary.sort_values("AvgReward", ascending=False); order = df_s.index
    # 1. Box Plot Rewards
    plt.figure(figsize=(10, max(6, n * 0.5))); sns.boxplot(data=df_raw_summary, x="TotalReward", y="Agent", palette="viridis", showfliers=False, order=order); plt.title(f"Reward Distribution ({ENV_NAME_SHORT} - {N_EVAL_EPISODES} Eps)"); plt.xlabel("Total Reward"); plt.ylabel("Agent"); plt.tight_layout(); p=os.path.join(RESULTS_DIR,f"{ENV_NAME_SHORT}_benchmark_rewards_boxplot.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    # 2. Bar Eval Time
    df_s_eval = df_s.sort_values("AvgTimePerEp", ascending=True); plt.figure(figsize=(10, max(6, n * 0.4))); idx=np.arange(len(df_s_eval)); plt.barh(idx, df_s_eval["AvgTimePerEp"], color='skyblue', log=True); plt.yticks(idx, df_s_eval.index); plt.xlabel('Avg Eval Time per Ep (s) - Log Scale'); plt.ylabel('Agent'); plt.title(f'Avg Eval Time ({ENV_NAME_SHORT})'); plt.tight_layout(); p=os.path.join(RESULTS_DIR,f"{ENV_NAME_SHORT}_benchmark_eval_time_log.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    # 3. Bar Train Time
    df_t = df_s[df_s["TrainingTime(s)"] > 1].sort_values("TrainingTime(s)", ascending=True)
    if not df_t.empty: plt.figure(figsize=(8, max(4, len(df_t)*0.5))); idx_t=np.arange(len(df_t)); plt.barh(idx_t, df_t["TrainingTime(s)"], color='lightcoral'); plt.yticks(idx_t, df_t.index); plt.xlabel('Total Training Time (s)'); plt.ylabel('Agent (RL)'); plt.title(f'Training Time ({ENV_NAME_SHORT})'); plt.tight_layout(); p=os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_train_time.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    else: print("Skip train time plot.")
    # 4. Scatter Reward vs Service
    plt.figure(figsize=(9, 7)); sns.scatterplot(data=df_s, x="AvgServiceLevel", y="AvgReward", hue="Agent", s=100, palette="viridis", legend=False);
    for _, r in df_s.iterrows(): plt.text(r["AvgServiceLevel"] + 0.005, r["AvgReward"], r.name, fontsize=9)
    plt.title(f"Reward vs. Service ({ENV_NAME_SHORT})"); plt.xlabel("Avg Service Level (Retailer)"); plt.ylabel("Avg Reward"); plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0)); plt.grid(True); plt.tight_layout(); p=os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_reward_vs_service.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    # 5. Scatter Reward vs Inventory
    plt.figure(figsize=(9, 7)); sns.scatterplot(data=df_s, x="AvgEndInv", y="AvgReward", hue="Agent", s=100, palette="viridis", legend=False);
    for _, r in df_s.iterrows(): plt.text(r["AvgEndInv"] * 1.01, r["AvgReward"], r.name, fontsize=9)
    plt.title(f"Reward vs. Inventory ({ENV_NAME_SHORT})"); plt.xlabel("Avg Ending Inv (All Stages)"); plt.ylabel("Avg Reward"); plt.grid(True); plt.tight_layout(); p=os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_reward_vs_inventory.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    plt.close('all')


# --- Main Execution ---

if __name__ == "__main__":
    # --- Define Agents ---
    print("Defining agents...")
    agent_objects = {}

    # Heuristics specific to InvManagement
    agents_to_run_defs = [
        ("Random", RandomAgent, {}),
        ("BaseStock_SF=1.0", BaseStockAgent, {'safety_factor': 1.0}),
        ("BaseStock_SF=1.2", BaseStockAgent, {'safety_factor': 1.2}),
        ("BaseStock_SF=0.8", BaseStockAgent, {'safety_factor': 0.8}),
    ]

    # RL Agents
    if SB3_AVAILABLE:
        sb3_agents_def = [
            ("PPO", PPO, {}), ("SAC", SAC, {}), ("TD3", TD3, {}), ("A2C", A2C, {}), ("DDPG", DDPG, {}),
            ("PPO-LargeBuffer", PPO, {'model_kwargs': {'n_steps': 4096}}),
            ("SAC-LowLR", SAC, {'model_kwargs': {'learning_rate': 1e-4}}),
            ("PPO-LSTM", PPO, {'policy': "MlpLstmPolicy"}),
            ("A2C-LSTM", A2C, {'policy': "MlpLstmPolicy"}),
            ("PPO-SmallNet", PPO, {'model_kwargs': {'policy_kwargs': dict(net_arch=dict(pi=[64], vf=[64]))}}),
            ("SAC-LargeNet", SAC, {'model_kwargs': {'policy_kwargs': dict(net_arch=[400, 300])}}),
        ]
        for name, model_cls, params in sb3_agents_def:
             wrapper_params = {'model_class': model_cls, 'name': name}
             if 'policy' in params: wrapper_params['policy'] = params['policy']
             if 'model_kwargs' in params: wrapper_params['model_kwargs'] = params['model_kwargs']
             if 'train_kwargs' in params: wrapper_params['train_kwargs'] = params['train_kwargs']
             agents_to_run_defs.append((name, SB3AgentWrapper, wrapper_params))
    else: print("\nSkipping RL agent definitions.")

    # Instantiate agents
    print(f"\nInstantiating {len(agents_to_run_defs)} agents...")
    for name, agent_class, params in agents_to_run_defs:
         try: print(f"  Instantiating: {name}"); agent_objects[name] = agent_class(**params)
         except Exception as e: print(f"ERROR Instantiating {name}: {e}"); import traceback; traceback.print_exc()

    # --- Train RL Agents ---
    print("\n--- Training Phase ---")
    for name, agent in agent_objects.items():
        if isinstance(agent, SB3AgentWrapper):
             agent.train(ENV_CONFIG, total_timesteps=RL_TRAINING_TIMESTEPS, save_path_prefix=f"{ENV_NAME_SHORT}_{name}_")

    # --- Run Evaluation ---
    print("\n--- Evaluation Phase ---")
    all_evaluation_results = []
    print(f"\n-- Evaluating on Standard Random Parameters ({ENV_CLASS_NAME}) --")
    for name, agent in agent_objects.items():
        if name not in agent_objects: continue
        eval_results = evaluate_agent(agent, ENV_CONFIG, N_EVAL_EPISODES,
                                       seed_offset=SEED_OFFSET, collect_details=COLLECT_STEP_DETAILS)
        if 'summary' in eval_results and not eval_results['summary'].empty: all_evaluation_results.append(eval_results)
        else: print(f"Warning: Eval for {name} produced no results.")

    # --- Process and Report Results ---
    final_summary, final_raw_summary = process_and_report_results(all_evaluation_results, agent_objects)

    # --- Generate Plots ---
    if final_summary is not None:
        rl_log_dirs = {name: agent.log_dir for name, agent in agent_objects.items() if isinstance(agent, SB3AgentWrapper)}
        if rl_log_dirs:
             try: print("\nPlotting learning curves..."); plot_learning_curves(rl_log_dirs, title=f"RL Learning Curves ({ENV_NAME_SHORT})")
             except Exception as e: print(f"Error plot learning curves: {e}"); import traceback; traceback.print_exc()
        plot_benchmark_results(final_summary, final_raw_summary)
    else: print("Skipping plotting.")

    print("\nBenchmark script finished.")
