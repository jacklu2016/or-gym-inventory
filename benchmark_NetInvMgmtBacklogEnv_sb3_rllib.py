# benchmark_netinvmgmt_combined.py
import gymnasium as gym
from gymnasium import spaces
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
import networkx as nx # Needed by the environment
from typing import Optional, Tuple, Dict, Any, List, Type

# --- Ray/RLlib Imports ---
RAY_AVAILABLE = False
try:
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.tune.registry import register_env
    RAY_AVAILABLE = True
except ImportError:
    print("Warning: ray[rllib] not found. RLlib agent tests will be skipped.")

# --- SB3 Imports ---
SB3_AVAILABLE = False
try:
    from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    from stable_baselines3.common.logger import configure
    SB3_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not found. SB3 agent tests will be skipped.")
    class DummyModel: pass
    PPO, SAC, TD3, A2C, DDPG = DummyModel, DummyModel, DummyModel, DummyModel, DummyModel

# --- Suppress common warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Environment Import ---
# <<<<<<<<<<<<<<<<< CHANGE: Target NetInvMgmt Env >>>>>>>>>>>>>>>>>
ENV_MODULE_NAME = "network_management"
ENV_CLASS_NAME = "NetInvMgmtBacklogEnv" # Or NetInvMgmtLostSalesEnv

try:
    # Path finding logic (same as before)
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    module_path_found = False
    potential_paths = [script_dir, os.path.join(script_dir, 'or-gym-inventory'), os.path.abspath(os.path.join(script_dir, '..', 'or-gym-inventory'))]
    for p in potential_paths:
         if os.path.exists(os.path.join(p, f'{ENV_MODULE_NAME}.py')):
             if p not in sys.path: print(f"Adding path: {p}"); sys.path.append(p)
             module_path_found = True; break
    if not module_path_found: print(f"Warning: Could not find {ENV_MODULE_NAME}.py.")

    module = __import__(ENV_MODULE_NAME)
    EnvClass = getattr(module, ENV_CLASS_NAME)
    print(f"Successfully imported {ENV_CLASS_NAME} from {ENV_MODULE_NAME}.")

except ImportError as e: print(f"Error importing {ENV_CLASS_NAME}: {e}"); sys.exit(1)
except Exception as e: print(f"Error during import: {e}"); sys.exit(1)


# --- Configuration ---
# Benchmark settings
N_EVAL_EPISODES = 10          # Reduced further for network env
RL_TRAINING_TIMESTEPS = 100000 # Reduced further for testing - INCREASE!
SEED_OFFSET = 11000
FORCE_RETRAIN = False
COLLECT_STEP_DETAILS = False
N_ENVS_TRAIN_SB3 = 1
N_WORKERS_RLLIB = 1 # 0 means use driver only

# <<<<<<<<<<<<<<<<< CHANGE: Paths for NetInvMgmt Env >>>>>>>>>>>>>>>>>
ENV_NAME_SHORT = "NetInvMgmt" # Prefix (use "NetInvMgmtLS" for lost sales)
BASE_DIR = f"./benchmark_{ENV_NAME_SHORT}_combined/"
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR_SB3 = os.path.join(LOG_DIR, "SB3"); MODEL_DIR_SB3 = os.path.join(MODEL_DIR, "SB3")
LOG_DIR_RLLIB = os.path.join(LOG_DIR, "RLlib"); MODEL_DIR_RLLIB = os.path.join(MODEL_DIR, "RLlib_checkpoints")
os.makedirs(LOG_DIR_SB3, exist_ok=True); os.makedirs(MODEL_DIR_SB3, exist_ok=True)
os.makedirs(LOG_DIR_RLLIB, exist_ok=True); os.makedirs(MODEL_DIR_RLLIB, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Environment config (using default graph from the class)
ENV_CONFIG = {
    'num_periods': 30, # Use default from env
    # Add other NetInvMgmt specific overrides here if needed
    # e.g., pass a custom graph object: 'graph': my_custom_graph
}

# --- RLlib Environment Registration ---
# <<<<<<<<<<<<<<<<< CHANGE: RLlib Env Name >>>>>>>>>>>>>>>>>
ENV_NAME_RLLIB = "netinvmgmt_rllib_env-v0"
def env_creator(env_config):
    return EnvClass(**env_config) # Use the imported EnvClass

if RAY_AVAILABLE:
    try: register_env(ENV_NAME_RLLIB, env_creator); print(f"Registered '{ENV_NAME_RLLIB}' with RLlib.")
    except Exception as e_reg: print(f"Warning: Could not register env with RLlib: {e_reg}")

# --- Agent Definitions ---
# (BaseAgent definition reused)
class BaseAgent:
    def __init__(self, name="BaseAgent"): self.name=name; self.training_time=0.0
    def get_action(self, o, e): raise NotImplementedError
    def train(self, ec, tt, sp): print(f"Agent {self.name} no train needed.")
    def load(self, p): print(f"Agent {self.name} no load.")
    def get_training_time(self): return self.training_time

# (RandomAgent definition reused)
class RandomAgent(BaseAgent):
    def __init__(self): super().__init__(name="Random")
    def get_action(self, o, e): return e.action_space.sample().astype(e.action_space.dtype)

# (ConstantOrderAgent definition reused)
class ConstantOrderAgent(BaseAgent):
    def __init__(self, order_fraction=0.1): super().__init__(name=f"ConstantOrder_{order_fraction*100:.0f}%"); self.order_fraction = order_fraction; self._action = None
    def get_action(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        if self._action is None: high = env.action_space.high.copy(); high[high == np.inf] = 1000; self._action = (high * self.order_fraction).astype(env.action_space.dtype)
        return self._action

# (SB3AgentWrapper definition reused - assumes float32 obs/action)
class SB3AgentWrapper(BaseAgent):
    def __init__(self, model_class, policy="MlpPolicy", name="SB3_Agent", train_kwargs=None, model_kwargs=None):
        super().__init__(name=name);
        if not SB3_AVAILABLE: raise ImportError("SB3 not available.")
        self.model_class=model_class; self.policy=policy; self.model=None
        self.train_kwargs=train_kwargs if train_kwargs else {}; self.model_kwargs=model_kwargs if model_kwargs else {}
        self.log_dir=os.path.join(LOG_DIR_SB3, self.name); self.save_path=os.path.join(MODEL_DIR_SB3, f"{self.name}.zip")
        os.makedirs(os.path.dirname(self.log_dir), exist_ok=True); os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        if model_class in [SAC, TD3, DDPG]:
             try: t_env = EnvClass(**ENV_CONFIG); a_dim=t_env.action_space.shape[0]; a_h=t_env.action_space.high.copy(); a_l = t_env.action_space.low.copy(); t_env.close(); a_h[a_h==np.inf]=1000; a_l[a_l==-np.inf]=0; a_r=a_h-a_l; n_std=0.1*a_r/2; n_std[n_std<=0]=0.01; self.model_kwargs['action_noise']=NormalActionNoise(mean=np.zeros(a_dim), sigma=n_std); print(f"SB3 Noise {name} std:{n_std.mean():.2f}")
             except Exception as e: print(f"Warn: SB3 noise setup fail {name}: {e}")
    def train(self, ec, tt, sp=""):
        if not SB3_AVAILABLE: print(f"Skip train {self.name}: SB3 missing"); return
        print(f"Train SB3 {self.name}..."); st=time.time(); tld=os.path.join(self.log_dir,"train_logs"); os.makedirs(tld, exist_ok=True)
        sp=os.path.join(MODEL_DIR_SB3, f"{sp}{self.name}.zip"); bmsp=os.path.join(MODEL_DIR_SB3, f"{sp}{self.name}_best"); os.makedirs(bmsp, exist_ok=True)
        if not FORCE_RETRAIN and os.path.exists(sp): print(f"Load SB3 {sp}"); self.load(sp);
        if hasattr(self.model, '_total_timesteps') and self.model._total_timesteps >= tt: self.training_time=0; print("SB3 Model already trained."); return
        def _ce(r, s=0): def _i(): env=EnvClass(**ec); env=Monitor(env,os.path.join(tld,f"m_{r}")); env.reset(seed=s+r); return env; return _i
        ve = DummyVecEnv([_ce(0, self.model_kwargs.get('seed', 0))])
        ee=Monitor(EnvClass(**ec));
        ecb=EvalCallback(ee,best_model_save_path=bmsp,log_path=os.path.join(self.log_dir,"eval_logs"),eval_freq=max(5000,tt//10),n_eval_episodes=3,deterministic=True,render=False)
        try:
            ms=self.model_kwargs.get('seed',None); self.model=self.model_class(self.policy,ve,verbose=0,seed=ms,**self.model_kwargs); print(f"Start SB3 train {self.name} ({tt} steps)...")
            self.model.learn(total_timesteps=tt,callback=ecb,log_interval=100,**self.train_kwargs); self.training_time=time.time()-st; print(f"Train SB3 {self.name} done: {self.training_time:.2f}s.")
            self.model.save(sp); print(f"SB3 final model saved: {sp}")
            try: bmp=os.path.join(bmsp,"best_model.zip");
                 if os.path.exists(bmp): print(f"Load SB3 best: {bmp}..."); self.model=self.model_class.load(bmp)
                 else: print("SB3 best not found, use final.")
            except Exception as e: print(f"Warn: SB3 load best fail. Err: {e}")
        except Exception as e: print(f"!!! ERR SB3 train {self.name}: {e}"); import traceback; traceback.print_exc(); self.model=None; self.training_time=time.time()-st
        finally: ve.close(); ee.close()
    def load(self, p):
         if not SB3_AVAILABLE: return
         try: print(f"Load SB3 {self.name} from {p}"); self.model = self.model_class.load(p)
         except Exception as e: print(f"!!! ERR SB3 load {self.name}: {e}"); self.model=None
    def get_action(self, o, e): # Assumes obs/action are float32
        if self.model is None: a=e.action_space.sample(); return a.astype(e.action_space.dtype)
        a,_=self.model.predict(o.astype(np.float32),deterministic=True);
        return a.astype(e.action_space.dtype)

# (RLlibAgentWrapper definition reused)
class RLlibAgentWrapper(BaseAgent):
    def __init__(self, algo_name: str, config_updates: dict = None, name: str = None):
        super().__init__(name=name if name else f"RLlib_{algo_name}")
        if not RAY_AVAILABLE: raise ImportError("Ray RLlib not available.")
        self.algo_name = algo_name; self.config_updates = config_updates if config_updates is not None else {}
        self.policy = None; self.algo = None; self.checkpoint_path = None
        self.log_dir=os.path.join(LOG_DIR_RLLIB, self.name); self.checkpoint_dir=os.path.join(MODEL_DIR_RLLIB, f"{self.name}_checkpoints"); self.training_log_path = os.path.join(self.log_dir, f"{self.name}_train_log.csv")
        os.makedirs(self.log_dir, exist_ok=True); os.makedirs(self.checkpoint_dir, exist_ok=True)
    def _build_config(self, env_config):
        try:
            config = AlgorithmConfig(algo_class=self.algo_name)
            config = config.environment(env=ENV_NAME_RLLIB, env_config=env_config) # Use correct registered name
            config = config.framework("torch")
            config = config.rollouts(num_rollout_workers=N_WORKERS_RLLIB)
            config = config.training(gamma=self.config_updates.get("gamma", 0.99),lr=self.config_updates.get("lr", 1e-4))
            if self.algo_name == "PPO": config = config.training(sgd_minibatch_size=128, num_sgd_iter=10)
            # Add specific model sizes if needed, e.g.
            # config = config.training(model={'fcnet_hiddens': [128, 128]})
            # config.validate(); # Optional
            return config
        except Exception as e: print(f"!!! ERROR build RLlib config {self.name}: {e}"); raise
    def train(self, ec, tt, sp=""):
        if not RAY_AVAILABLE: print(f"Skip train {self.name}: Ray missing"); return
        print(f"Train RLlib {self.name}..."); st=time.time(); log_data=[]
        last_chkpt = None; # Find latest checkpoint dir if exists
        if os.path.exists(self.checkpoint_dir) and any(os.scandir(self.checkpoint_dir)): last_chkpt = ray.rllib.train.get_checkpoint_dir(self.checkpoint_dir)
        restored=False
        try:
            config=self._build_config(ec); self.algo=config.build()
            if not FORCE_RETRAIN and last_chkpt:
                 print(f"Restore RLlib {self.name} from {last_chkpt}"); self.algo.restore(last_chkpt); restored=True
                 if self.algo.iteration*self.algo.config.train_batch_size >= tt: print("RLlib already trained."); self.training_time=0; self.policy=self.algo.get_policy(); self.algo.stop(); return
            else: print(f"Start RLlib train {self.name} from scratch ({tt} steps)...")
            t_total = self.algo.iteration * self.algo.config.train_batch_size if restored else 0; it = self.algo.iteration if restored else 0
            while t_total < tt:
                it+=1; res=self.algo.train(); t_total=res["timesteps_total"]; mean_rw=res.get("episode_reward_mean",float('nan')); log_data.append({'iter':it, 'steps':t_total, 'reward':mean_rw})
                if it%20==0: print(f"  RLlib {self.name} It:{it} St:{t_total}/{tt} Rw:{mean_rw:.1f}")
            self.training_time=time.time()-st; print(f"Train RLlib {self.name} done: {self.training_time:.2f}s.")
            cp = self.algo.save(self.checkpoint_dir); print(f"RLlib final checkpoint: {cp}"); self.checkpoint_path=cp
            self.policy = self.algo.get_policy()
        except Exception as e: print(f"!!! ERR RLlib train {self.name}: {e}"); import traceback; traceback.print_exc(); self.training_time=time.time()-st
        finally:
             if self.algo: self.algo.stop()
             if log_data: pd.DataFrame(log_data).to_csv(self.training_log_path, index=False); print(f"Saved RLlib log: {self.training_log_path}")
    def load(self, p): print(f"RLlib load {self.name}. Restore in train."); self.checkpoint_path = p
    def get_action(self, o, e): # Assumes observation o is already float32
        if self.policy is None and self.checkpoint_path:
             print(f"Policy {self.name} not loaded, try restore from {self.checkpoint_path}")
             try: cfg=self._build_config(ENV_CONFIG); temp_algo=cfg.build(); temp_algo.restore(self.checkpoint_path); self.policy=temp_algo.get_policy(); temp_algo.stop(); print(" Policy restored.")
             except Exception as e_restore: print(f" Restore failed: {e_restore}"); self.policy = None
        if self.policy is None: print(f"Warn: Policy {self.name} missing. Random."); a=e.action_space.sample(); return a.astype(e.action_space.dtype)
        try:
            action_data = self.policy.compute_single_action(o); action = action_data[0]
            return action.astype(e.action_space.dtype) # Action is already float32
        except Exception as e: print(f"!!! ERR {self.name} get_action: {e}. Random."); a=e.action_space.sample(); return a.astype(e.action_space.dtype)


# --- Evaluation Function (Adapted for NetInvMgmt Metrics) ---
def evaluate_agent(agent: BaseAgent,
                   env_config: dict,
                   n_episodes: int,
                   seed_offset: int = 0,
                   collect_details: bool = False) -> Dict:
    """Evaluates an agent over n_episodes for NetInvManagement Env."""
    episode_summaries = []; all_step_details = []; total_eval_time = 0.0
    eval_env = EnvClass(**env_config) # Use correct class

    print(f"\nEvaluating {agent.name} for {n_episodes} episodes ({ENV_CLASS_NAME})...")
    successful_episodes = 0
    for i in range(n_episodes):
        episode_seed = seed_offset + i; episode_step_details = []
        try:
            obs, info = eval_env.reset(seed=episode_seed)
            terminated, truncated = False, False; episode_reward, episode_steps = 0.0, 0
            # Use internal DataFrames for efficient metric collection
            ep_demand_df = pd.DataFrame(columns=eval_env.retail_links)
            ep_sales_df = pd.DataFrame(columns=eval_env.retail_links)
            ep_stockout_df = pd.DataFrame(columns=eval_env.retail_links)
            ep_end_inv_df = pd.DataFrame(columns=eval_env.main_nodes)
            start_time = time.perf_counter()
            while not terminated and not truncated:
                t_current = eval_env.period; action = agent.get_action(obs, eval_env)
                obs, reward, terminated, truncated, info = eval_env.step(action); episode_reward += reward; episode_steps += 1
                if t_current < eval_env.num_periods: # Check index validity
                     # Use internal env DFs directly
                     ep_demand_df.loc[t_current] = eval_env.D.loc[t_current, eval_env.retail_links]
                     ep_sales_df.loc[t_current] = eval_env.S.loc[t_current, eval_env.retail_links]
                     ep_stockout_df.loc[t_current] = eval_env.U.loc[t_current + 1, eval_env.retail_links] # U[t+1] = unmet from period t
                     ep_end_inv_df.loc[t_current] = eval_env.X.loc[t_current + 1, eval_env.main_nodes]
                if collect_details: step_data = {'step': episode_steps, 'reward': reward, 'action': action.tolist()}; episode_step_details.append(step_data)
            end_time = time.perf_counter(); episode_time = end_time - start_time; total_eval_time += episode_time
            total_demand = ep_demand_df.sum().sum(); total_sales = ep_sales_df.sum().sum(); total_stockout = ep_stockout_df.sum().sum(); avg_end_inv = ep_end_inv_df.mean().mean()
            service_lvl = total_sales / max(1e-6, total_demand) if total_demand > 1e-6 else 1.0
            ep_summary = {"Agent": agent.name, "Episode": i + 1, "TotalReward": episode_reward,"Steps": episode_steps, "Time": episode_time, "Seed": episode_seed,"AvgServiceLevel": service_lvl,"TotalStockoutQty": total_stockout,"AvgEndingInv": avg_end_inv,"Error": None}
            episode_summaries.append(ep_summary); all_step_details.append(episode_step_details); successful_episodes += 1
            if n_episodes <= 20 or (i + 1) % max(1, n_episodes // 5) == 0: print(f"  Ep {i+1}/{n_episodes}: Rw={episode_reward:.0f}, SL={service_lvl:.1%}")
        except Exception as e: print(f"!!! ERR ep {i+1} {agent.name}: {e}"); import traceback; traceback.print_exc(); episode_summaries.append({"Agent": agent.name, "Episode": i + 1, "TotalReward": np.nan, "Steps": 0,"Time": 0, "Seed": episode_seed, "AvgServiceLevel": np.nan,"TotalStockoutQty": np.nan, "AvgEndingInv": np.nan, "Error": str(e)}); all_step_details.append([])
    eval_env.close()
    if successful_episodes == 0: print(f"Eval FAIL {agent.name}."); return {'summary': pd.DataFrame(), 'details': []}
    avg_eval_time = total_eval_time / successful_episodes; print(f"Eval done {agent.name}. Avg time: {avg_eval_time:.4f}s")
    return {'summary': pd.DataFrame(episode_summaries), 'details': all_step_details}

# --- Results Processing (Reusable) ---
# (process_and_report_results function is reusable)
def process_and_report_results(all_eval_results: List[Dict], agent_objects: Dict):
    # ... (Keep the exact function from benchmark_invmgmt_combined.py) ...
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
        print(f"\nRaw summary saved: {raw_summary_path}"); print(f"Summary saved: {summary_path}")
        if COLLECT_STEP_DETAILS:
             print(f"Saving step details: {details_path}...")
             with open(details_path, 'w') as f:
                 for i, agent_results in enumerate(all_eval_results):
                     agent_name = agent_results['summary']['Agent'].iloc[0] if 'summary' in agent_results and not agent_results['summary'].empty else f"Unknown_{i}"
                     for ep_num, steps in enumerate(agent_results.get('details', [])):
                         for step_data in steps: step_data['agent'] = agent_name; step_data['episode'] = ep_num + 1; serializable_data = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in step_data.items()}; f.write(json.dumps(serializable_data) + '\n')
             print("Step details saved.")
    except Exception as e: print(f"\nError saving results: {e}")
    return summary, results_df_raw_summary


# --- Plotting Functions ---
# (plot_learning_curves and plot_benchmark_results are reusable)
def plot_learning_curves(sb3_log_dirs: Dict[str, str], rllib_log_dirs: Dict[str, str], title: str):
    # ... (Keep the exact function from benchmark_invmgmt_combined.py) ...
    # Uses ENV_NAME_SHORT for filename
    if not SB3_AVAILABLE and not RAY_AVAILABLE: print("Skip curves: No RL libs."); return
    plt.figure(figsize=(14, 8)); plt.title(title); plt.xlabel("Timesteps"); plt.ylabel("Reward (Smoothed)"); plotted=False
    if SB3_AVAILABLE:
        print("Plotting SB3 curves...");
        for name, log_dir in sb3_log_dirs.items():
            monitor_files = glob.glob(os.path.join(log_dir, "*logs", "monitor.*.csv"));
            if not monitor_files: print(f"  Warn: No SB3 logs {name}"); continue; monitor_path = os.path.dirname(monitor_files[0])
            try: results = load_results(monitor_path);
                 if len(results['r']) > 0: x, y = ts2xy(results, 'timesteps'); y = results.rolling(window=max(10, len(x)//20), min_periods=1).mean()['r'] if len(x)>10 else results['r']; plt.plot(x, y, label=f"{name} (SB3)"); plotted=True
                 else: print(f"  Warn: SB3 Log {name} empty.")
            except Exception as e: print(f"  Error plot SB3 logs {name}: {e}")
    if RAY_AVAILABLE:
        print("Plotting RLlib curves...");
        for name, log_dir_base in rllib_log_dirs.items():
             rllib_log_file = os.path.join(log_dir_base, f"{name}_train_log.csv");
             if os.path.exists(rllib_log_file):
                 try: log_df = pd.read_csv(rllib_log_file).dropna(subset=['reward']);
                      if not log_df.empty: x=log_df['steps'].values; y=log_df['reward'].rolling(window=max(10, len(x)//20), min_periods=1).mean().values if len(x)>10 else log_df['reward'].values; min_len=min(len(x),len(y)); x=x[:min_len]; y=y[:min_len]; plt.plot(x, y, label=f"{name} (RLlib)", linestyle='--'); plotted=True
                 except Exception as e: print(f"  Error plot RLlib log {name}: {e}")
             else: print(f"  Warn: No RLlib log file {rllib_log_file}")
    if plotted: plt.legend(); plt.grid(True); plt.tight_layout(); p=os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_learning_curves.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    else: print("Skip learning curves plot - no data.")

def plot_benchmark_results(df_summary: pd.DataFrame, df_raw_summary: pd.DataFrame):
    # ... (Keep the exact function from benchmark_invmgmt_combined.py) ...
    # Uses ENV_NAME_SHORT for titles/filenames and updated metric names
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
    plt.title(f"Reward vs. Service ({ENV_NAME_SHORT})"); plt.xlabel("Avg Service Level (Overall Retail)"); plt.ylabel("Avg Reward"); plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0)); plt.grid(True); plt.tight_layout(); p=os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_reward_vs_service.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    # 5. Scatter Reward vs Inventory
    plt.figure(figsize=(9, 7)); sns.scatterplot(data=df_s, x="AvgEndInv", y="AvgReward", hue="Agent", s=100, palette="viridis", legend=False);
    for _, r in df_s.iterrows(): plt.text(r["AvgEndInv"] * 1.01, r["AvgReward"], r.name, fontsize=9)
    plt.title(f"Reward vs. Inventory ({ENV_NAME_SHORT})"); plt.xlabel("Avg Ending Inv (All Nodes)"); plt.ylabel("Avg Reward"); plt.grid(True); plt.tight_layout(); p=os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_reward_vs_inventory.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    plt.close('all')


# --- Main Execution ---

if __name__ == "__main__":
    # --- Initialize Ray ---
    if RAY_AVAILABLE:
        try: ray.init(ignore_reinit_error=True, num_cpus=N_WORKERS_RLLIB + 1)
        except Exception as e_ray: print(f"Warn: Ray init fail: {e_ray}. RLlib skipped."); RAY_AVAILABLE = False

    # --- Define Agents ---
    print("Defining agents...")
    agent_objects = {}
    agents_to_run_defs = []

    # Heuristics for NetInvMgmt
    agents_to_run_defs.extend([
        ("Random", RandomAgent, {}),
        ("ConstantOrder_5%", ConstantOrderAgent, {'order_fraction': 0.05}),
        ("ConstantOrder_10%", ConstantOrderAgent, {'order_fraction': 0.10}),
        # Add other heuristics specific to network env if developed
    ])

    # SB3 RL Agents
    if SB3_AVAILABLE:
        sb3_agents_def = [
            ("SB3_PPO", PPO, {}), ("SB3_SAC", SAC, {}), ("SB3_TD3", TD3, {}),
            ("SB3_A2C", A2C, {}), ("SB3_DDPG", DDPG, {}),
            ("SB3_PPO-LSTM", PPO, {'policy': "MlpLstmPolicy"}), # LSTM might be useful
        ]
        for name, model_cls, params in sb3_agents_def:
             wrapper_params = {'model_class': model_cls, 'name': name}; wrapper_params.update(params)
             agents_to_run_defs.append((name, SB3AgentWrapper, wrapper_params))
    else: print("\nSkipping SB3 agent definitions.")

    # RLlib RL Agents
    if RAY_AVAILABLE:
         rllib_agents_def = [
             ("PPO", {'lr': 1e-4}), # Different LR maybe
             ("SAC", {'gamma': 0.99}),
             # ("TD3", {}), # Add others if desired
         ]
         for name, cfg_updates in rllib_agents_def:
              rllib_name = f"RLlib_{name}"
              agents_to_run_defs.append((rllib_name, RLlibAgentWrapper, {'algo_name': name, 'config_updates': cfg_updates, 'name': rllib_name}))
    else: print("\nSkipping RLlib agent definitions.")

    # Instantiate agents
    print(f"\nInstantiating {len(agents_to_run_defs)} agents...")
    for name, agent_class, params in agents_to_run_defs:
         try: print(f"  Instantiating: {name}"); agent_objects[name] = agent_class(**params)
         except Exception as e: print(f"ERROR Instantiating {name}: {e}"); import traceback; traceback.print_exc()

    # --- Train RL Agents ---
    print("\n--- Training Phase ---")
    for name, agent in agent_objects.items():
        if isinstance(agent, (SB3AgentWrapper, RLlibAgentWrapper)):
             save_prefix = f"{ENV_NAME_SHORT}_{name}_"
             agent.train(ENV_CONFIG, total_timesteps=RL_TRAINING_TIMESTEPS, save_path_prefix=save_prefix)
        else: agent.train(ENV_CONFIG, 0, "")

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
        sb3_log_dirs = {name: agent.log_dir for name, agent in agent_objects.items() if isinstance(agent, SB3AgentWrapper)}
        rllib_log_dirs = {name: agent.log_dir for name, agent in agent_objects.items() if isinstance(agent, RLlibAgentWrapper)}
        if sb3_log_dirs or rllib_log_dirs:
             try: print("\nPlotting learning curves..."); plot_learning_curves(sb3_log_dirs, rllib_log_dirs, title=f"RL Learning Curves ({ENV_NAME_SHORT})")
             except Exception as e: print(f"Error plot learning curves: {e}"); import traceback; traceback.print_exc()
        plot_benchmark_results(final_summary, final_raw_summary)
    else: print("Skipping plotting.")

    # --- Shutdown Ray ---
    if RAY_AVAILABLE and ray.is_initialized():
        print("Shutting down Ray...")
        ray.shutdown()

    print("\nBenchmark script finished.")
