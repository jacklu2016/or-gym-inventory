# benchmark_invmanagement.py
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
ENV_CLASS_NAME = "InvManagementBacklogEnv" # Choose Backlog or LostSales

try:
    # Adjust path finding if necessary
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
N_EVAL_EPISODES = 30          # Fewer for quicker testing
RL_TRAINING_TIMESTEPS = 50000 # Reduced for quicker testing - INCREASE for real benchmark
SEED_OFFSET = 4000
FORCE_RETRAIN = False
COLLECT_STEP_DETAILS = True
N_ENVS_TRAIN = 4 # Use parallel envs for training

# Paths (specific to this environment)
ENV_NAME_SHORT = "InvMgmt"
LOG_DIR = f"./sb3_logs_{ENV_NAME_SHORT}/"
MODEL_DIR = f"./sb3_models_{ENV_NAME_SHORT}/"
RESULTS_DIR = f"./benchmark_results_{ENV_NAME_SHORT}/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Environment configuration for evaluation/training
# Using default parameters from InvManagementMasterEnv for simplicity
# Customize this dict for specific scenarios
ENV_CONFIG = {
    'periods': 50, # Default is 30, let's use slightly longer
    # Using defaults for I0, p, r, k, h, c, L, dist_param etc.
    # Add overrides here if needed, e.g.:
    # 'I0': [50, 50],
    # 'c': [20, 25],
    # 'L': [2, 3],
    # 'dist_param': {'mu': 15}
}


# --- Agent Definitions ---
class BaseAgent:
    """Base class for agents"""
    def __init__(self, name="BaseAgent"):
        self.name = name
        self.training_time = 0.0

    def get_action(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        raise NotImplementedError

    def train(self, env_config: dict, total_timesteps: int, save_path_prefix: str):
        print(f"Agent {self.name} does not require training.")
        pass

    def load(self, path: str):
         print(f"Agent {self.name} does not support loading.")
         pass

    def get_training_time(self) -> float:
        return self.training_time

class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Random")
    def get_action(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        # Need to ensure dtype matches action space if it's int
        action = env.action_space.sample()
        return action.astype(env.action_space.dtype)

class BaseStockAgent(BaseAgent):
    """ Heuristic: Independent Base Stock level for each stage. """
    def __init__(self, safety_factor=1.0):
        # Orders up to Base Stock Level = (L+1) * mu * safety_factor
        # NOTE: This is a HUGE simplification. It uses the final customer demand 'mu'
        # for ALL stages and ignores echelon interactions / demand propagation.
        # A proper multi-echelon heuristic would be much more complex.
        super().__init__(name=f"BaseStock_SF={safety_factor:.1f}")
        self.safety_factor = safety_factor

    def get_action(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        # Check if env has the expected attributes
        if not all(hasattr(env, attr) for attr in ['num_stages', 'lead_time', 'dist_param', 'lt_max', 'I', 'action_log', 'period']):
             print(f"Warning: Env missing attributes needed for {self.name}. Returning random action.")
             return env.action_space.sample().astype(env.action_space.dtype)

        num_stages_inv = env.num_stages - 1 # Number of stages with inventory/orders
        lead_times = env.lead_time         # Vector of lead times L_0, L_1, ... L_{m-2}
        # Use the mean demand parameter mu (assuming it's relevant for all stages - simplification!)
        mu = env.dist_param.get('mu', 10) # Default mu if not found

        # Get current state info from observation
        # Obs: [I[t,0], ..., I[t,M-2], Action[t-1], ..., Action[t-lt_max]]
        current_on_hand = observation[:num_stages_inv]

        # Calculate current inventory position for EACH stage
        # Position = On Hand + Pipeline (sum of orders placed but not arrived)
        inventory_position = current_on_hand.copy()
        t = env.period
        action_log = env.action_log # Shape (periods, num_stages_inv)

        pipeline_start_idx = num_stages_inv
        for i in range(num_stages_inv): # For each stage i
            L_i = lead_times[i]
            if L_i == 0: continue # No pipeline for L=0

            pipeline_for_stage_i = 0
            # Sum orders placed in the last L_i periods for this stage
            start_log_idx = max(0, t - L_i)
            if t > 0 and start_log_idx < t :
                 # Get actions placed for stage i (column i) in the relevant period range
                 orders_in_pipeline = action_log[start_log_idx : t, i]
                 pipeline_for_stage_i = orders_in_pipeline.sum()

            inventory_position[i] += pipeline_for_stage_i

        # Calculate target base stock level for EACH stage
        # Target = (L_i + 1) * mu * safety_factor
        target_levels = (lead_times + 1) * mu * self.safety_factor

        # Calculate order quantity: Order = max(0, Target - Position)
        order_quantities = np.maximum(0, target_levels - inventory_position)

        # Clip by environment action space limits (especially capacity C which is action_space.high)
        order_quantities = np.clip(order_quantities, env.action_space.low, env.action_space.high)

        return order_quantities.astype(env.action_space.dtype)


class SB3AgentWrapper(BaseAgent):
    """ Updated Wrapper for Stable Baselines3 agents for InvManagement. """
    def __init__(self, model_class, policy="MlpPolicy", name="SB3_Agent", train_kwargs=None, model_kwargs=None):
        super().__init__(name=name)
        if not SB3_AVAILABLE: raise ImportError("Stable Baselines3 is not available.")
        self.model_class = model_class
        self.policy = policy
        self.model = None
        self.train_kwargs = train_kwargs if train_kwargs is not None else {}
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.log_dir = os.path.join(LOG_DIR, self.name)
        self.save_path = os.path.join(MODEL_DIR, f"{self.name}.zip")

        # Add action noise for off-policy algorithms
        if model_class in [SAC, TD3, DDPG]:
             # Action space is Box(m-1,)
             # Need env to get action dim, but env not available here easily.
             # Let's assume default env config to get action dim for noise setup.
             # THIS IS A HACK - better if action dim passed or noise setup deferred.
             temp_env = InvManagementEnvClass(**ENV_CONFIG) # Use default config
             action_dim = temp_env.action_space.shape[0]
             action_range = temp_env.action_space.high # Use capacity as range estimate
             temp_env.close()
             # Scale noise std dev based on typical action range (capacity)
             noise_std = 0.1 * action_range / 2 # Heuristic: 10% of half the range
             # Use VectorizedActionNoise if multiple envs used in training? SB3 handles this often.
             # Let's stick to NormalActionNoise for now, SB3 might wrap it.
             self.model_kwargs['action_noise'] = NormalActionNoise(mean=np.zeros(action_dim), sigma=noise_std)
             print(f"Applied NormalActionNoise for {name} with std dev approx: {noise_std.mean():.2f}")


    def train(self, env_config: dict, total_timesteps: int, save_path_prefix: str=""):
        print(f"Training {self.name}...")
        start_time = time.time()
        train_log_dir = os.path.join(self.log_dir, "train_logs")
        os.makedirs(train_log_dir, exist_ok=True)

        save_path = os.path.join(MODEL_DIR, f"{save_path_prefix}{self.name}.zip")
        best_model_save_path = os.path.join(MODEL_DIR, f"{save_path_prefix}{self.name}_best")
        os.makedirs(best_model_save_path, exist_ok=True)

        # Check if model exists and retraining is not forced
        if not FORCE_RETRAIN and os.path.exists(save_path):
            print(f"Loading existing model for {self.name} from {save_path}")
            self.load(save_path)
            # Manually set training time if loading existing model prevents training
            if hasattr(self.model, '_total_timesteps') and self.model._total_timesteps >= total_timesteps:
                 self.training_time = 0 # Assume no new training time needed
                 print("Model already trained sufficiently.")
                 return # Skip training

        # --- Environment Setup ---
        def _create_env(rank: int, seed: int = 0):
            def _init():
                # Use the specified Env Class
                env = InvManagementEnvClass(**env_config)
                # Check environment compatibility (optional but recommended)
                # try:
                #    check_env(env)
                # except Exception as e_check:
                #    print(f"WARNING: Env check failed for InvManagement: {e_check}")
                # Monitor wrapper for logging
                env = Monitor(env, filename=os.path.join(train_log_dir, f"monitor_{rank}"))
                env.reset(seed=seed + rank)
                return env
            return _init

        # Use VecEnv
        vec_env = DummyVecEnv([_create_env(0, self.model_kwargs.get('seed', 0))])
        # If using N_ENVS_TRAIN > 1:
        # vec_env = SubprocVecEnv([_create_env(i, self.model_kwargs.get('seed', 0)) for i in range(N_ENVS_TRAIN)], start_method='fork')


        # --- Setup Callback ---
        eval_env_callback = Monitor(InvManagementEnvClass(**env_config)) # Use Monitor here too
        eval_callback = EvalCallback(eval_env_callback,
                                     best_model_save_path=best_model_save_path,
                                     log_path=os.path.join(self.log_dir, "eval_logs"),
                                     eval_freq=max(5000, total_timesteps // 10), # Eval less frequently for longer runs
                                     n_eval_episodes=5,
                                     deterministic=True, render=False)

        # --- Training ---
        try:
            model_seed = self.model_kwargs.get('seed', None)
            # Ensure observation space dtype is float32 if necessary
            # SB3 usually handles this, but if issues arise, add observation wrapper to vec_env
            # from gymnasium.wrappers import TransformObservation
            # vec_env = TransformObservation(vec_env, lambda obs: obs.astype(np.float32))

            self.model = self.model_class(self.policy, vec_env, verbose=0, seed=model_seed, **self.model_kwargs)

            print(f"Starting training for {total_timesteps} timesteps...")
            # Set logger format to csv for easier parsing later if needed
            # new_logger = configure(self.log_dir, ["stdout", "csv"])
            # self.model.set_logger(new_logger)

            self.model.learn(total_timesteps=total_timesteps, callback=eval_callback, log_interval=50, **self.train_kwargs) # Log more often
            self.training_time = time.time() - start_time
            print(f"Training for {self.name} finished in {self.training_time:.2f} seconds.")
            self.model.save(save_path)
            print(f"Final model saved to {save_path}")
            # Load best model for evaluation
            try:
                 best_model_path = os.path.join(best_model_save_path, "best_model.zip")
                 if os.path.exists(best_model_path):
                     print(f"Loading best model from {best_model_path}...")
                     # Need to pass custom_objects if using non-standard layers/activations
                     self.model = self.model_class.load(best_model_path) # Load without env first
                 else: print("Best model not found, using final model.")
            except Exception as e_load: print(f"Warning: Could not load best model. Error: {e_load}")

        except Exception as e:
            print(f"!!! ERROR during training {self.name}: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.training_time = time.time() - start_time # Record time even if failed
        finally:
            vec_env.close()
            eval_env_callback.close()

    def load(self, path: str):
         if not SB3_AVAILABLE: return
         try:
             print(f"Loading model for {self.name} from {path}")
             self.model = self.model_class.load(path)
         except Exception as e:
             print(f"!!! ERROR loading model for {self.name} from {path}: {e}")
             self.model = None

    def get_action(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        if self.model is None:
            # Fallback to random action if model failed/not loaded
            action = env.action_space.sample()
            return action.astype(env.action_space.dtype)

        # Ensure observation is float32 if model expects it
        obs_input = observation.astype(np.float32)
        action, _states = self.model.predict(obs_input, deterministic=True)
        # Ensure action dtype matches env space (important if env expects int64)
        return action.astype(env.action_space.dtype)

# --- Evaluation Function ---

def evaluate_agent(agent: BaseAgent,
                   env_config: dict,
                   n_episodes: int,
                   seed_offset: int = 0,
                   # fixed_params not implemented for InvManagement Env yet
                   collect_details: bool = True
                   ) -> Dict:
    """
    Evaluates an agent over n_episodes for InvManagement Env.
    Returns a dict containing summary stats and optionally detailed step data.
    """
    episode_summaries = []
    all_step_details = [] # List of lists (one per episode)
    total_eval_time = 0.0
    eval_env = InvManagementEnvClass(**env_config) # Use specified class

    print(f"\nEvaluating {agent.name} for {n_episodes} episodes...")
    successful_episodes = 0
    for i in range(n_episodes):
        episode_seed = seed_offset + i
        episode_step_details = []
        try:
            obs, info = eval_env.reset(seed=episode_seed)
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0
            # Metrics specific to InvManagement
            episode_demand_retailer = 0.0
            episode_sales_retailer = 0.0
            episode_stockout_qty_retailer = 0.0
            episode_inventory_sum_all_stages = 0.0 # Sum of ending inventory each step across stages

            start_time = time.perf_counter()

            while not terminated and not truncated:
                action = agent.get_action(obs, eval_env)
                obs, reward, terminated, truncated, info = eval_env.step(action)

                # Accumulate metrics
                episode_reward += reward
                episode_steps += 1

                # --- InvManagement Specific Metrics ---
                # Assume retailer is stage 0
                demand_this_step = info.get('demand_realized', 0) # Overall customer demand
                episode_demand_retailer += demand_this_step

                sales_vector = info.get('sales', np.zeros(eval_env.num_stages)) # Sales vector S[t]
                sales_at_retailer = sales_vector[0]
                episode_sales_retailer += sales_at_retailer

                # Unfulfilled includes backlog AND current unmet demand at end of step
                unfulfilled_vector = info.get('unfulfilled', np.zeros(eval_env.num_stages))
                stockout_at_retailer = unfulfilled_vector[0] # Unmet demand at retailer
                episode_stockout_qty_retailer += stockout_at_retailer

                # Ending inventory across all stages [I(t+1,0), ..., I(t+1, M-2)]
                ending_inv_vector = info.get('ending_inventory', np.zeros(eval_env.num_stages - 1))
                episode_inventory_sum_all_stages += np.sum(np.maximum(0, ending_inv_vector)) # Sum positive inventory


                # Collect step details if requested
                if collect_details:
                     step_data = {
                         'step': episode_steps,
                         'reward': reward,
                         'action': action.tolist(), # Action is vector
                         'demand_retailer': demand_this_step,
                         'sales_retailer': sales_at_retailer,
                         'stockout_retailer': stockout_at_retailer,
                         'ending_inv_vector': ending_inv_vector.tolist(),
                         # Add env params (these don't change here usually)
                         # Add observation if needed (can be large)
                         # 'observation': obs.astype(float).tolist(), # Cast to float for JSON
                     }
                     episode_step_details.append(step_data)

            end_time = time.perf_counter()
            episode_time = end_time - start_time
            total_eval_time += episode_time

            # Calculate episode operational metrics
            avg_ending_inv_all = episode_inventory_sum_all_stages / episode_steps if episode_steps > 0 else 0
            # Service level at retailer
            service_level_retailer = episode_sales_retailer / max(1e-6, episode_demand_retailer) if episode_demand_retailer > 1e-6 else 1.0

            ep_summary = {
                "Agent": agent.name, "Episode": i + 1, "TotalReward": episode_reward,
                "Steps": episode_steps, "Time": episode_time, "Seed": episode_seed,
                "AvgServiceLevel": service_level_retailer,
                "TotalStockoutQty": episode_stockout_qty_retailer,
                "AvgEndingInv": avg_ending_inv_all,
                "Error": None
            }
            episode_summaries.append(ep_summary)
            all_step_details.append(episode_step_details)
            successful_episodes += 1

            # Print progress
            if n_episodes <= 20 or (i + 1) % max(1, n_episodes // 5) == 0:
                 print(f"  Ep {i+1}/{n_episodes}: Reward={episode_reward:.2f}, ServiceLvL={service_level_retailer:.2%}")

        except Exception as e:
             print(f"!!! ERROR during evaluation episode {i+1} for {agent.name}: {e}")
             import traceback
             traceback.print_exc()
             episode_summaries.append({ # Log failure
                "Agent": agent.name, "Episode": i + 1, "TotalReward": np.nan, "Steps": 0,
                "Time": 0, "Seed": episode_seed, "AvgServiceLevel": np.nan,
                "TotalStockoutQty": np.nan, "AvgEndingInv": np.nan, "Error": str(e)
             })
             all_step_details.append([]) # Add empty list for details

    eval_env.close()

    if successful_episodes == 0:
        print(f"Evaluation FAILED for {agent.name}. No successful episodes.")
        return {'summary': pd.DataFrame(), 'details': []}

    avg_eval_time = total_eval_time / successful_episodes
    print(f"Evaluation finished for {agent.name}. Avg success ep time: {avg_eval_time:.4f}s")

    return {'summary': pd.DataFrame(episode_summaries), 'details': all_step_details}


# --- Results Processing (Should work mostly as is) ---
# (process_and_report_results function is generally reusable)
def process_and_report_results(all_eval_results: List[Dict], agent_objects: Dict):
    """Processes evaluation results, prints summary, saves data."""
    # ... (Keep the function from benchmark_newsvendor_advanced.py) ...
    # It calculates summary stats based on the columns present in the summary dataframes
    # generated by evaluate_agent, which we updated ('AvgServiceLevel', etc.)
    if not all_eval_results:
        print("No evaluation results to process.")
        return None, None

    all_summaries_list = [res['summary'] for res in all_eval_results if 'summary' in res and not res['summary'].empty]
    if not all_summaries_list:
        print("No successful evaluation summaries found.")
        return None, None

    results_df_raw_summary = pd.concat(all_summaries_list, ignore_index=True)

    print("\n--- Benchmark Summary ---")

    # Aggregation logic - ensure column names match those from evaluate_agent
    summary = results_df_raw_summary.dropna(subset=['TotalReward']).groupby("Agent").agg(
        AvgReward=("TotalReward", "mean"),
        MedianReward=("TotalReward", "median"),
        StdReward=("TotalReward", "std"),
        MinReward=("TotalReward", "min"),
        MaxReward=("TotalReward", "max"),
        AvgServiceLevel=("AvgServiceLevel", "mean"), # Updated metric
        AvgStockoutQty=("TotalStockoutQty", "mean"), # Updated metric
        AvgEndInv=("AvgEndingInv", "mean"),         # Updated metric
        AvgTimePerEp=("Time", "mean"),
        SuccessfulEpisodes=("Episode", "count")
    )
    summary["TrainingTime(s)"] = summary.index.map(lambda name: agent_objects.get(name, BaseAgent(name)).get_training_time()).fillna(0.0)
    total_episodes_attempted = results_df_raw_summary.groupby("Agent")["Episode"].count()
    summary["EpisodesAttempted"] = total_episodes_attempted
    summary["SuccessRate(%)"] = (summary["SuccessfulEpisodes"] / summary["EpisodesAttempted"]) * 100

    summary = summary.sort_values(by="AvgReward", ascending=False)
    # Update column order to include new metrics
    summary = summary[[
        "AvgReward", "MedianReward", "StdReward", "MinReward", "MaxReward",
        "AvgServiceLevel", "AvgStockoutQty", "AvgEndInv",
        "AvgTimePerEp", "TrainingTime(s)",
        "SuccessfulEpisodes", "EpisodesAttempted", "SuccessRate(%)"
    ]]

    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(summary)

    # --- Save Results ---
    raw_summary_path = os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_raw_summary.csv")
    summary_path = os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_summary.csv")
    details_path = os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_step_details.jsonl")

    try:
        results_df_raw_summary.to_csv(raw_summary_path, index=False)
        summary.to_csv(summary_path)
        print(f"\nRaw summary results saved to {raw_summary_path}")
        print(f"Summary saved to {summary_path}")

        # Save detailed step data
        if COLLECT_STEP_DETAILS:
            print(f"Saving detailed step data to {details_path}...")
            with open(details_path, 'w') as f:
                for i, agent_results in enumerate(all_eval_results):
                    agent_name = agent_results['summary']['Agent'].iloc[0] if 'summary' in agent_results and not agent_results['summary'].empty else f"UnknownAgent_{i}"
                    episode_details = agent_results.get('details', [])
                    for episode_num, steps in enumerate(episode_details):
                        for step_data in steps:
                            step_data['agent'] = agent_name
                            step_data['episode'] = episode_num + 1
                            # Ensure all data is JSON serializable (e.g., numpy arrays to lists)
                            serializable_data = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in step_data.items()}
                            f.write(json.dumps(serializable_data) + '\n')
            print("Detailed step data saved.")

    except Exception as e:
        print(f"\nError saving results: {e}")

    return summary, results_df_raw_summary


# --- Plotting Functions (Reuse plot_learning_curves, adapt plot_benchmark_results) ---
# (plot_learning_curves function can be reused as is)
def plot_learning_curves(log_dirs: Dict[str, str], title: str = "RL Learning Curves"):
    # ... (Keep the function from benchmark_newsvendor_advanced.py) ...
    if not SB3_AVAILABLE: return
    plt.figure(figsize=(12, 7))
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    all_curves_plotted = False
    for agent_name, log_dir in log_dirs.items():
         monitor_files = glob.glob(os.path.join(log_dir, "train_logs", "monitor.*.csv")) \
                      + glob.glob(os.path.join(log_dir, "eval_logs", "monitor.*.csv")) # Check both
         if not monitor_files:
             print(f"Warning: No monitor log files found for {agent_name} in {log_dir}")
             continue

         monitor_path = os.path.dirname(monitor_files[0]) # Use directory of first found file
         try:
            results = load_results(monitor_path)
            if len(results['r']) > 0:
                 x, y = ts2xy(results, 'timesteps')
                 if len(x) > 10:
                      # Use rolling mean for smoothing
                      y = results.rolling(window=10).mean()['r']
                 # Clip x to avoid plotting beyond max timesteps if needed
                 # x = x[x <= RL_TRAINING_TIMESTEPS]
                 # y = y[:len(x)]
                 plt.plot(x, y, label=agent_name)
                 all_curves_plotted = True
            else:
                 print(f"Warning: Log file found for {agent_name} but contains no reward data.")
         except Exception as e:
              print(f"Error loading/plotting logs for {agent_name}: {e}")


    if all_curves_plotted:
         plt.legend(loc='lower right') # Adjust legend position maybe
         plt.tight_layout()
         learning_curve_path = os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_learning_curves.png")
         plt.savefig(learning_curve_path)
         print(f"Saved learning curves plot to {learning_curve_path}")
         plt.close()
    else:
         print("Skipping learning curve plot - no data found.")


def plot_benchmark_results(df_summary: pd.DataFrame, df_raw_summary: pd.DataFrame):
    """Generates and saves comparison plots based on evaluation summaries."""
    # ... (Keep the function from benchmark_newsvendor_advanced.py, BUT UPDATE METRIC NAMES) ...
    # Replace "AvgServiceLevel", "TotalStockoutQty", "AvgEndingInv" if metric names changed in summary
    if df_summary is None or df_raw_summary is None:
        print("Skipping plotting due to missing summary data.")
        return

    print("\nGenerating comparison plots...")
    n_agents = df_summary.shape[0]
    plt.style.use('seaborn-v0_8-darkgrid')
    df_summary_sorted = df_summary.sort_values("AvgReward", ascending=False)
    agent_order = df_summary_sorted.index

    # 1. Box Plot of Total Rewards
    plt.figure(figsize=(10, max(6, n_agents * 0.5)))
    sns.boxplot(data=df_raw_summary, x="TotalReward", y="Agent", palette="viridis", showfliers=False, order=agent_order)
    plt.title(f"Distribution of Total Rewards ({ENV_NAME_SHORT} - {N_EVAL_EPISODES} Eps)")
    plt.xlabel("Total Reward (Higher is Better)")
    plt.ylabel("Agent")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,f"{ENV_NAME_SHORT}_benchmark_rewards_boxplot.png"))
    print(f"Saved rewards boxplot to {RESULTS_DIR}{ENV_NAME_SHORT}_benchmark_rewards_boxplot.png")
    plt.close()

    # 2. Bar Chart - Average Evaluation Time
    df_summary_sorted_eval = df_summary_sorted.sort_values("AvgTimePerEp", ascending=True)
    plt.figure(figsize=(10, max(6, n_agents * 0.4)))
    index = np.arange(len(df_summary_sorted_eval))
    plt.barh(index, df_summary_sorted_eval["AvgTimePerEp"], color='skyblue', log=True)
    plt.yticks(index, df_summary_sorted_eval.index)
    plt.xlabel('Average Evaluation Time per Episode (s) - Log Scale')
    plt.ylabel('Agent')
    plt.title(f'Average Evaluation Time per Episode ({ENV_NAME_SHORT})')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,f"{ENV_NAME_SHORT}_benchmark_eval_time_log.png"))
    print(f"Saved evaluation time bar chart to {RESULTS_DIR}{ENV_NAME_SHORT}_benchmark_eval_time_log.png")
    plt.close()

    # 3. Bar Chart - Training Time
    df_train = df_summary_sorted[df_summary_sorted["TrainingTime(s)"] > 1].sort_values("TrainingTime(s)", ascending=True)
    if not df_train.empty:
        plt.figure(figsize=(8, max(4, len(df_train) * 0.5)))
        index_train = np.arange(len(df_train))
        plt.barh(index_train, df_train["TrainingTime(s)"], color='lightcoral')
        plt.yticks(index_train, df_train.index)
        plt.xlabel('Total Training Time (s)')
        plt.ylabel('Agent (RL)')
        plt.title(f'Training Time for RL Agents ({ENV_NAME_SHORT})')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_train_time.png"))
        print(f"Saved training time bar chart to {RESULTS_DIR}{ENV_NAME_SHORT}_benchmark_train_time.png")
        plt.close()
    else: print("Skipping training time plot.")

    # 4. Scatter Plot - Reward vs Service Level
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df_summary_sorted, x="AvgServiceLevel", y="AvgReward", hue="Agent", s=100, palette="viridis", legend=False)
    for i, row in df_summary_sorted.iterrows(): plt.text(row["AvgServiceLevel"] + 0.005, row["AvgReward"], row.name, fontsize=9)
    plt.title(f"Reward vs. Service Level ({ENV_NAME_SHORT})")
    plt.xlabel("Average Service Level (Fill Rate - Retailer)")
    plt.ylabel("Average Total Reward")
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_reward_vs_service.png"))
    print(f"Saved Reward vs Service plot to {RESULTS_DIR}{ENV_NAME_SHORT}_benchmark_reward_vs_service.png")
    plt.close()

    # 5. Scatter Plot - Reward vs Average Ending Inventory
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df_summary_sorted, x="AvgEndInv", y="AvgReward", hue="Agent", s=100, palette="viridis", legend=False)
    for i, row in df_summary_sorted.iterrows(): plt.text(row["AvgEndInv"] * 1.01, row["AvgReward"], row.name, fontsize=9)
    plt.title(f"Reward vs. Inventory ({ENV_NAME_SHORT})")
    plt.xlabel("Average Ending Inventory (All Stages)")
    plt.ylabel("Average Total Reward")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_reward_vs_inventory.png"))
    print(f"Saved Reward vs Inventory plot to {RESULTS_DIR}{ENV_NAME_SHORT}_benchmark_reward_vs_inventory.png")
    plt.close()

    plt.close('all')

# --- Main Execution ---

if __name__ == "__main__":
    # --- Define Agents ---
    print("Defining agents...")
    agent_objects = {}

    # Heuristics specific to InvManagement (or generic)
    agents_to_run_defs = [
        ("Random", RandomAgent, {}),
        ("BaseStock_SF=1.0", BaseStockAgent, {'safety_factor': 1.0}),
        ("BaseStock_SF=1.2", BaseStockAgent, {'safety_factor': 1.2}),
        ("BaseStock_SF=0.8", BaseStockAgent, {'safety_factor': 0.8}),
        # Add other InvManagement heuristics here if developed
    ]

    # RL Agents (Check compatibility with Box action space)
    if SB3_AVAILABLE:
        sb3_agents_def = [
            ("PPO", PPO, {}),
            ("SAC", SAC, {}),
            ("TD3", TD3, {}),
            ("A2C", A2C, {}),
            ("DDPG", DDPG, {}),
            # Add variations if desired, ensuring they handle Box actions
             ("PPO-LargeBuffer", PPO, {'model_kwargs': {'n_steps': 4096}}),
             ("SAC-LowLR", SAC, {'model_kwargs': {'learning_rate': 1e-4}}),
             # LSTM Policies might be relevant here due to multi-stage delays
             ("PPO-LSTM", PPO, {'policy': "MlpLstmPolicy"}),
             ("A2C-LSTM", A2C, {'policy': "MlpLstmPolicy"}),
             # Network size variations
             ("PPO-SmallNet", PPO, {'model_kwargs': {'policy_kwargs': dict(net_arch=dict(pi=[64], vf=[64]))}}), # Smaller net
             ("SAC-LargeNet", SAC, {'model_kwargs': {'policy_kwargs': dict(net_arch=[400, 300])}}), # Larger net
        ]
        for name, model_cls, params in sb3_agents_def:
             wrapper_params = {'model_class': model_cls, 'name': name}
             if 'policy' in params: wrapper_params['policy'] = params['policy']
             if 'model_kwargs' in params: wrapper_params['model_kwargs'] = params['model_kwargs']
             if 'train_kwargs' in params: wrapper_params['train_kwargs'] = params['train_kwargs']
             agents_to_run_defs.append((name, SB3AgentWrapper, wrapper_params))
    else:
        print("\nSkipping RL agent definitions.")

    # Instantiate agents
    print(f"\nInstantiating {len(agents_to_run_defs)} agents...")
    for name, agent_class, params in agents_to_run_defs:
         try:
             print(f"  Instantiating: {name}")
             agent_objects[name] = agent_class(**params)
         except Exception as e:
              print(f"ERROR Instantiating agent {name}: {e}")
              import traceback; traceback.print_exc()

    # --- Train RL Agents ---
    print("\n--- Training Phase ---")
    for name, agent in agent_objects.items():
        if isinstance(agent, SB3AgentWrapper):
             agent.train(ENV_CONFIG, total_timesteps=RL_TRAINING_TIMESTEPS, save_path_prefix=f"{ENV_NAME_SHORT}_{name}_")

    # --- Run Evaluation ---
    print("\n--- Evaluation Phase ---")
    all_evaluation_results = []
    print("\n-- Evaluating on Standard Random Parameters --")
    for name, agent in agent_objects.items():
        if name not in agent_objects: continue
        eval_results = evaluate_agent(agent, ENV_CONFIG, N_EVAL_EPISODES,
                                       seed_offset=SEED_OFFSET, collect_details=COLLECT_STEP_DETAILS)
        if 'summary' in eval_results and not eval_results['summary'].empty:
            all_evaluation_results.append(eval_results)
        else: print(f"Warning: Evaluation for agent {name} produced no results.")

    # --- Process and Report Results ---
    final_summary, final_raw_summary = process_and_report_results(all_evaluation_results, agent_objects)

    # --- Generate Plots ---
    if final_summary is not None:
        rl_log_dirs = {name: agent.log_dir for name, agent in agent_objects.items() if isinstance(agent, SB3AgentWrapper)}
        if rl_log_dirs:
             try:
                 print("\nAttempting to plot learning curves...")
                 plot_learning_curves(rl_log_dirs, title=f"RL Learning Curves ({ENV_NAME_SHORT})")
             except Exception as e:
                 print(f"Error generating learning curve plot: {e}")
                 import traceback; traceback.print_exc()

        plot_benchmark_results(final_summary, final_raw_summary)
    else:
        print("Skipping plotting as no summary results were generated.")

    print("\nBenchmark script finished.")
