# benchmark_newsvendor_sb3_rllib.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import time
import os
import sys
import json
import glob
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- Suppress common warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- RL specific imports ---
# SB3
SB3_AVAILABLE = False
try:
    from stable_baselines3 import PPO as PPO_SB3, SAC as SAC_SB3, TD3 as TD3_SB3, A2C as A2C_SB3, DDPG as DDPG_SB3
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor as MonitorSB3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    from stable_baselines3.common.logger import configure as configureSB3
    SB3_AVAILABLE = True
    print("Stable Baselines3 loaded.")
except ImportError:
    print("Warning: stable-baselines3 not found. SB3 agent tests will be skipped.")
    class DummyModelSB3: pass
    PPO_SB3, SAC_SB3, TD3_SB3, A2C_SB3, DDPG_SB3 = DummyModelSB3, DummyModelSB3, DummyModelSB3, DummyModelSB3, DummyModelSB3

# Ray RLlib
RLLIB_AVAILABLE = False
try:
    import ray
    from ray import air, tune # Use new AIR/Tune API
    from ray.rllib.algorithms.algorithm import Algorithm # Base class
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.algorithms.ppo import PPOConfig # Example specific config
    # Import other specific configs if needed, or use AlgorithmConfig(algo_class=...)
    from ray.tune.registry import register_env
    RLLIB_AVAILABLE = True
    print("Ray RLlib loaded.")
except ImportError:
    print("Warning: ray[rllib] not found. RLlib agent tests will be skipped.")
    print("Install using: pip install 'ray[rllib]'")
    # Define dummy classes/functions if RLlib not found
    class Algorithm: pass
    class AlgorithmU pip
    # Install libraries
    !pip install gymnasium pandas numpy scipy stable-baselines3[extra] torch matplotlib seaborn "ray[rllib]" tensorflow # Install both torch & tf for Ray's flexibility
    ```
    *(Note: Installing both PyTorch and TensorFlow for Ray might increase installation size but gives Ray flexibility).*

---

**Combined SB3 + RLlib Benchmark Script (`benchmark_newsvendor_combined.py`):**

```python
# benchmark_newsvendor_combined.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import time
import os
import sys
import json
import glob
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Optional, Tuple, Dict, Any, List, Type # Added Type

# --- Ray/RLlib Imports ---
RAY_AVAILABLE = False
try:
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    # Import specific configs IF needed for deeper customization, otherwise use string identifiers
    # from ray.rllib.algorithms.ppo import PPOConfig
    # from ray.rllib.algorithms.sac import SACConfig
    from ray.tune.registry import register_env
    RAY_AVAILABLE = True
except ImportError:
    print("Warning: ray[rllib] not found. RLlib agent tests will be skipped.")
    print("Install using: pip install 'ray[rllib]'")

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
    class DummyModel: pass # Define dummy classes if SB3 not found
    PPO, SAC, TD3, A2C, DDPG = DummyModel, DummyModel, DummyModel, DummyModel, DummyModel


# --- Suppress common warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --- Environment Import ---
# Assuming newsvendor.py is accessible
try:
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    found_path = None
    if os.path.exists(os.path.join(script_dir, 'newsvendor.py')):
         found_path = script_dir
    else:
        repo_path_guess = os.path.abspath(os.path.join(script_dir, '..'))
        if os.path.exists(os.path.join(script_dir, 'or-gym-inventory', 'newsvendor.py')):
             found_path = os.path.join(script_dir, 'or-gym-inventory')
        elif os.path.exists(os.path.join(repo_path_guess, 'or-gym-inventory', 'newsvendor.py')):
             found_path = os.path.join(repo_path_guess, 'or-gym-inventory')

    if found_path and found_path not in sys.path:
         print(f"Adding path to sys.path: {found_path}")
         sys.path.append(found_path)
    elif not found_path:
         print("Warning: Could not automatically find 'newsvendor.py'. Ensure it's accessible.")

    from newsvendor import NewsvendorEnvConfig: pass


# --- Environment Import ---
ENV_MODULE_NAME = "newsvendor"
ENV_CLASS_NAME = "NewsvendorEnv" # Target the original Newsvendor

try:
    # Path finding logic (simplified)
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    if os.path.exists(os.path.join(script_dir, f'{ENV_MODULE_NAME}.py')):
         print(f"Found {ENV_MODULE_NAME}.py in current directory.")
         if script_dir not in sys.path: sys.path.append(script_dir) # Ensure import works
    else:
        repo_path = os.path.join(script_dir, 'or-gym-inventory')
        if os.path.exists(os.path.join(repo_path, f'{ENV_MODULE_NAME}.py')):
            if repo_path not in sys.path: print(f"Adding path: {repo_path}"); sys.path.append(repo_path)
        else: print(f"Warning: Could not find {ENV_MODULE_NAME}.py.")

    # Import the specific environment class
    module = __import__(ENV_MODULE_NAME)
    EnvClass = getattr(module, ENV_CLASS_NAME)
    print(f"Successfully imported {ENV_CLASS_NAME} from {ENV_MODULE_NAME}.")

except ImportError as e: print(f"Error importing {ENV_CLASS_NAME}: {e}"); sys.exit(1)
except Exception as e: print(f"Error during import: {e}"); sys.exit(1)


# --- Configuration ---
# Benchmark settings
N_EVAL_EPISODES = 30
# Reduce timesteps for quicker testing - INCREASE SIGNIFICANTLY FOR REAL BENCHMARK!
RL_TRAINING_TIMESTEPS = 50000
SEED_OFFSET = 8000
FORCE_RETRAIN = False         # Retrain RL models?
COLLECT_STEP_DETAILS = False  # Keep False unless needed for debugging
N_ENVS_TRAIN_SB3 = 1          # Use 1 for SB3 simple VecEnv
N_WORKERS_RLLIB = 3           # Use N-1 workers + 1 driver (e.g., 3+1=4 cores)

# Paths
ENV_NAME_SHORT = "NewsVendor"
BASE_DIR = f"./benchmark_{ENV_NAME_SHORT}_sb3_rllib/"
LOG_DIR_SB3 = os.path.join(BASE_DIR, "sb3_logs/")
MODEL_DIR_SB3 = os.path.join(BASE_DIR, "sb3_models/")
LOG_DIR_RLLIB = os.path.join(BASE_DIR, "rllib_results/") # RLlib uses Ray's structure
MODEL_DIR_RLLIB = os.path.join(BASE_DIR, "rllib_checkpoints/")
RESULTS_DIR = os.path.join(BASE_DIR, "results/")
os.makedirs(LOG_DIR_SB3, exist_ok=True)
os.makedirs(MODEL_DIR_SB3, exist_ok=True)
os.makedirs(LOG_DIR_RLLIB, exist_ok=True)
os.makedirs(MODEL_DIR_RLLIB, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Environment config
ENV_CONFIG = {
    'lead_time': 5, 'step_limit': 50, 'p_max': 100.0, 'h_max': 5.0,
    'k_max': 10.0, 'mu_max': 200.0,
}

# --- RLlib Environment Registration ---
ENV_NAME_RLLIB = "newsvendor_rllib_env-v0"
if RLLIB_AVAILABLE:
    def env_creator(env_config):
        return EnvClass(**env_config) # Use the imported EnvClass
    try:
        register_env(ENV_NAME_RLLIB, env_creator)
        print(f"Registered environment '{ENV_NAME_RLLIB}' with RLlib.")
    except Exception as e_reg:
        print(f"Warning: Could not register env with RLlib: {e_reg}")


# --- Agent Definitions ---
class BaseAgent:
    def __init__(self, name="BaseAgent"): self.name=name; self.training_time=0.0
    def get_action(self, o, e): raise NotImplementedError
    def train(self, ec, tt, sp): print(f"Agent {self.name} does not require training.")
    def load(self, p): print(f"Agent {self.name} does not support loading.")
    def get_training_time(self): return self.training_time

class RandomAgent(BaseAgent):
    def __init__(self): super().__init__(name="Random")
    def get_action(self, o, e): return e.action_space.sample().astype(e.action_space.dtype)

# Include OrderUpToHeuristicAgent, ClassicNewsvendorAgent, sSPolicyAgent
# (Copy definitions from benchmark_newsvendor_advanced.py here)
class OrderUpToHeuristicAgent(BaseAgent):
    def __init__(self, safety_factor=1.0): super().__init__(name=f"OrderUpTo_SF={safety_factor:.1f}"); self.safety_factor = safety_factor
    def get_action(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        if not hasattr(env.unwrapped, 'lead_time'): return env.action_space.sample().astype(env.action_space.dtype)

    print("Successfully imported NewsvendorEnv.")
except ImportError as e:
    print(f"Error importing NewsvendorEnv: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


# --- Configuration ---
# Benchmark settings
N_EVAL_EPISODES = 30
RL_TRAINING_TIMESTEPS = 50000 # Keep relatively low for combined test run - INCREASE!
SEED_OFFSET = 8000
FORCE_RETRAIN = False
COLLECT_STEP_DETAILS = False
N_ENVS_TRAIN_SB3 = 1 # SB3 VecEnvs
N_WORKERS_RLLIB = 1 # RLlib workers (0 means use driver) - increase for parallelism

# Paths
ENV_NAME_SHORT = "NewsVendor"
BASE_DIR = f"./benchmark_{ENV_NAME_SHORT}_combined/"
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Environment configuration for evaluation/training
ENV_CONFIG = {
    'lead_time': 5,
    'step_limit': 50,
    'p_max': 100.0,
    'h_max': 5.0,
    'k_max': 10.0,
    'mu_max': 200.0,
}

# --- Modified Environment (from advanced benchmark - allows fixed params) ---
# (CustomizableNewsvendorEnv definition remains the same)
class CustomizableNewsvendorEnv(NewsvendorEnv):
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        fixed_params = options.get("fixed_params", None) if options else None
        if fixed_params:
            self.price = fixed_params.get('price', self.price); self.cost = fixed_params.get('cost', self.cost)
            self.h = fixed_params.get('h', self.h); self.k = fixed_params.get('k', self.k)
            self.mu = fixed_params.get('mu', self.mu)
        else: # Default randomize
            self.price = max(1, self.np_random.random() * self.p_max); self.cost = max(1, self.np_random.random() * self.price)
            self.h = self.np_random.random() * min(self.cost, self.h_max); self.k = self.np_random.random() * self.k_max
            self.mu = self.np_random.random() * self.mu_max
        self.state = np.zeros(self.obs_dim, dtype=np.float32); self.state[:5] = np.array([self.price, self.cost, self.h, self.k, self.mu], dtype=np.float32)
        self.step_count = 0; observation = self._get_obs(); info = self._get_info()
        if fixed_params: info['fixed_params_used'] = fixed_params
        return observation, info

# --- Environment Creator and Registration for RLlib ---
ENV_NAME_RLLIB = "newsvendor_rllib_env"
def env_creator(env_config):
    """Creator function for RLlib."""
    return CustomizableNewsvendorEnv(**env_config)

if RAY_AVAILABLE:
    register_env(ENV_NAME_RLLIB, env_creator)
    print(f"Registered environment '{ENV_NAME_RLLIB}' with Ray RLlib.")

# --- Agent Definitions ---
# (BaseAgent, RandomAgent, OrderUpToHeuristicAgent, ClassicNewsvendorAgent, sSPolicyAgent remain the same)
class BaseAgent:
    def __init__(self, name="BaseAgent"): self.name=name; self.training_time=0.0
    def get_action(self, o, e): raise NotImplementedError
    def train(self, ec, tt, sp): print(f"Agent {self.name} no train needed.")
    def load(self, p): print(f"Agent {self.name} no load.")
    def get_training_time(self): return self.training_time
class RandomAgent(BaseAgent):
    def __init__(self): super().__init__(name="Random")
    def get_action(self, o, e): return e.action_space.sample().astype(e.action_space.dtype)
class OrderUpToHeuristicAgent(BaseAgent):
    def __init__(self, safety_factor=1.0): super().__init__(name=f"OrderUpTo_SF={safety_factor:.1f}"); self.safety_factor = safety_factor
    def get_action(self, o, e):
        if not hasattr(e.unwrapped, 'lead_time'): return e.action_space.sample().astype(e.action_space.dtype)
        env = e.unwrapped; mu=o[4]; pipe=o[5:]; lt=env.lead_time; target=mu*(lt+1)*self.safety_factor; pos=pipe.sum(); order=max(0,target-pos); order=np.clip(order,e.action_space.low[0],e.action_space.high[0]); return np.array([order],dtype=e.action_space.dtype)
class ClassicNewsvendorAgent(BaseAgent):
    def __init__(self, cr_method='k_vs_h', safety_factor=1.0): super().__init__(name=f"ClassicNV_SF={safety_factor:.1f}_{cr_method}"); self.cr_method=cr_method; self.safety_factor=safety_factor
    def get_action(self, o, e):
        if not hasattr(e.unwrapped, 'lead_time'): return e.action_space.sample().astype(e.action_space.dtype)
        env = e.unwrapped; p,c,h,k,mu = o[:5]; pipe=o[5:]; lt=env.lead_time; fallback=False;        env_core = env.unwrapped; mu = observation[4]; pipeline_inventory = observation[5:]; lead_time = env_core.lead_time
        target_demand = mu * (lead_time + 1) * self.safety_factor; current_position = pipeline_inventory.sum()
        order_qty = max(0, target_demand - current_position); order_qty = np.clip(order_qty, env.action_space.low[0], env.action_space.high[0])
        return np.array([order_qty], dtype=env.action_space.dtype)

class ClassicNewsvendorAgent(BaseAgent):
    def __init__(self, cr_method='k_vs_h', safety_factor=1.0): super().__init__(name=f"ClassicNV_SF={safety_factor:.1f}_{cr_method}"); self.cr_method = cr_method; self.safety_factor = safety_factor
    def get_action(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        if not hasattr(env.unwrapped, 'lead_time'): return env.action_space.sample().astype(env.action_space.dtype)
        env_core = env.unwrapped; price, cost, h, k, mu = observation[:5]; pipeline_inventory = observation[5:]; lead_time = env_core.lead_time
        fallback = False; critical_ratio = 0.5
        try:
            if self.cr_method == 'profit_margin': under_c = max(1e-6, price - cost + k); over_c = max(1e-6, h);
            if under_c + over_c <= 1e-6: fallback = True; else: critical_ratio = under_c / (under_c + over_c)
            else: if h + k <= 1e-6: fallback = True; else: critical_ratio = k / (h + k)
        except Exception: fallback = True
        if fallback: target_demand = mu * (lead_time + 1); current_position = pipeline_inventory.sum(); order_qty = max(0, target_demand - current_position)
        else: eff_mu = mu * (lead_time + 1) * self.safety_factor; target_lvl = poisson.ppf(np.clip(critical_ratio, 0.001, 0.999), mu=max(1e-6, eff_mu)); current_position = pipeline_inventory.sum(); order_qty = max(0, target_lvl - current_position)
        order_qty = np.clip(order_qty, env.action_space.low[0], env.action_space.high[0]); return np.array([order_qty], dtype=env.action_space.dtype)

class sSPolicyAgent(BaseAgent):
    def __init__(self, s_quantile=0.5, S_buffer_factor=1.2): super().__init__(name=f"sS_Policy(s={s_quantile:.2f},S={S_buffer_factor:.1f}s)"); self.s_quantile = s_quantile; self.S_buffer_factor = S_buffer_factor
    def get_action(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        if not hasattr(env.unwrapped, 'lead_time'): return env.action_space.sample().astype(env.action_space.dtype)
        env_core = env.unwrapped; price, cost, h, k, mu = observation[:5]; pipeline_inventory = observation[5:]; lead_time = env_core.lead_time
        s_lvl = 0; if h + k > 1e-6: cr_s = k / (h + k); eff_mu_s = mu * (lead_time + 1); s_lvl = max(0, poisson.ppf(np.clip(cr_s, 0.001, 0.999), mu=max(1e-6, eff_mu_s)))
        S_lvl = s_lvl * self.S_buffer_factor; current_pos = pipeline_inventory.sum(); order_qty = 0
        if current_pos < s_lvl: order_qty = max(0, S_lvl - current_pos)
        order_qty = np.clip(order_qty, env.action_space.low[0], env.action_space.high[0]); return np.array([order_qty], dtype=env.action_space.dtype)

# SB3 Wrapper (minor path adjustments)
class SB3AgentWrapper(BaseAgent):
    def __init__(self, model_class, policy="MlpPolicy", name="SB cr=0.5
        try:
            if self.cr_method=='profit_margin': uc=max(1e-6,p-c+k); oc=max(1e-6,h); fallback=(uc+oc<=1e-6); cr=uc/(uc+oc) if not fallback else 0.5
            else: fallback=(h+k<=1e-6); cr=k/(h+k) if not fallback else 0.5
        except: fallback=True
        if fallback: target=mu*(lt+1); pos=pipe.sum(); order=max(0, target-pos)
        else: eff_mu=mu*(lt+1)*self.safety_factor; target=poisson.ppf(np.clip(cr,0.001,0.999),mu=max(1e-6,eff_mu)); pos=pipe.sum(); order=max(0,target-pos)
        order=np.clip(order,e.action_space.low[0],e.action_space.high[0]); return np.array([order],dtype=e.action_space.dtype)
class sSPolicyAgent(BaseAgent):
    def __init__(self, s_quantile=0.5, S_buffer_factor=1.2): super().__init__(name=f"sS_Policy(s={s_quantile:.2f},S={S_buffer_factor:.1f}s)"); self.s_quantile=s_quantile; self.S_buffer_factor=S_buffer_factor
    def get_action(self, o, e):
        if not hasattr(e.unwrapped, 'lead_time'): return e.action_space.sample().astype(e.action_space.dtype)
        env = e.unwrapped; p,c,h,k,mu = o[:5]; pipe=o[5:]; lt=env.lead_time; s_lvl=0
        if h+k > 1e-6: cr_s=k/(h+k); eff_mu_s=mu*(lt+1); s_lvl=poisson.ppf(np.clip(cr_s,0.001,0.999),mu=max(1e-6,eff_mu_s))
        s_level=max(0,s_lvl); S_level=s_level*self.S_buffer_factor; pos=pipe.sum(); order=0
        if pos < s_level: order=max(0, S_level-pos)
        order=np.clip(order,e.action_space.low[0],e.action_space.high[0]); return np.array([order],dtype=e.action_space.dtype)

# (SB3AgentWrapper remains the same as in advanced benchmark)
class SB3AgentWrapper(BaseAgent):
    # ... (Keep the exact class definition from benchmark_newsvendor_advanced.py) ...
    def __init__(self, model_class, policy="MlpPolicy", name="SB3_Agent", train_kwargs=None, model_kwargs=None):
        super().__init__(name=name); self.model_class=model_class; self.policy=policy; self.model=None
        self.train_kwargs=train_kwargs if train_kwargs is not None else {}; self.model_kwargs=model_kwargs if model_kwargs is not None else {}
        self.log_dir=os.path.join(LOG_DIR, self.name); self.save_path=os.path.join(MODEL_DIR, f"{self.name}.zip")
        if SB3_AVAILABLE and model_class in [SAC, TD3, DDPG]:
             try: t_env=CustomizableNewsvendorEnv(**ENV_CONFIG); a_dim=t_env.action_space.shape[0]; a_h=t_env.action_space.high; a_h[a_h==np.inf]=2000; a_r=a_h-t_env.action_space.low; t_env.close(); n_std=0.1*a_r/2; self.model_kwargs['action_noise']=NormalActionNoise(mean=np.zeros3_Agent", train_kwargs=None, model_kwargs=None):
        super().__init__(name=name)
        if not SB3_AVAILABLE: raise ImportError("SB3 not available.")
        self.model_class = model_class; self.policy = policy; self.model = None
        self.train_kwargs = train_kwargs if train_kwargs is not None else {}
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.log_dir = os.path.join(LOG_DIR_SB3, self.name) # Use SB3 log dir
        self.save_path = os.path.join(MODEL_DIR_SB3, f"{self.name}.zip") # Use SB3 model dir

        if model_class in [SAC_SB3, TD3_SB3, DDPG_SB3]:
             try:
                 temp_env = EnvClass(**ENV_CONFIG); action_dim = temp_env.action_space.shape[0]; action_range = temp_env.action_space.high; temp_env.close()
                 noise_std = 0.1 * action_range / 2; noise_std[action_range <= 0] = 0.1
                 self.model_kwargs['action_noise'] = NormalActionNoise(mean=np.zeros(action_dim), sigma=noise_std)
                 # print(f"SB3 Noise applied for {name}")
             except Exception as e_noise: print(f"Warn SB3 noise setup fail {name}: {e_noise}")

    def train(self, env_config: dict, total_timesteps: int, save_path_prefix: str=""): # prefix unused here
        print(f"Training {self.name} (SB3)...")
        start_time = time.time(); train_log_dir = os.path.join(self.log_dir, "train_logs"); os.makedirs(train_log_dir, exist_ok=True)
        save_path = self.save_path; best_model_save_path = os.path.join(MODEL_DIR_SB3, f"{self.name}_best"); os.makedirs(best_model_save_path, exist_ok=True)
        if not FORCE_RETRAIN and os.path.exists(save_path):
            print(f"Loading SB3 model: {save_path}"); self.load(save_path)
            # Simple check if loaded model is already trained enough
            if hasattr(self.model, '_total_timesteps') and self.model._total_timesteps >= total_timesteps: self.training_time = 0; print("SB3 Model already trained."); return

        def _create_env(rank: int, seed: int = 0):
            def _init(): env = EnvClass(**env_config); env = MonitorSB3(env, filename=os.path.join(train_log_dir, f"monitor_{rank}")); env.reset(seed=seed + rank); return env; return _init
        # Use DummyVecEnv for stability, SubprocVecEnv needs care with complex objects/pickling
        vec_env =(a_dim), sigma=n_std); print(f"SB3 noise {name} std:{n_std.mean():.2f}")
             except Exception as e: print(f"Warn: SB3 noise setup fail {name}: {e}")
    def train(self, ec, tt, sp):
        if not SB3_AVAILABLE: print(f"Skip train {self.name}: SB3 missing"); return
        print(f"Train SB3 {self.name}..."); st=time.time(); tld=os.path.join(self.log_dir,"train_logs"); os.makedirs(tld, exist_ok=True)
        sp=os.path.join(MODEL_DIR, f"{sp}{self.name}.zip"); bmsp=os.path.join(MODEL_DIR, f"{sp}{self.name}_best"); os.makedirs(bmsp, exist_ok=True)
        if not FORCE_RETRAIN and os.path.exists(sp): print(f"Load SB3 {sp}"); self.load(sp); self.training_time=0; return
        def _ce(r, s=0):
            def _i(): env=CustomizableNewsvendorEnv(**ec); env=Monitor(env,os.path.join(tld,f"m_{r}")); env.reset(seed=s+r); return env; return _i
        ve = DummyVecEnv([_ce(0, self.model_kwargs.get('seed', 0))])
        # if N_ENVS_TRAIN_SB3 > 1: try: ve=SubprocVecEnv([_ce(i, self.model_kwargs.get('seed',0)) for i in range(N_ENVS_TRAIN_SB3)], start_method='fork'); except: print("Warn: Subproc fail, use Dummy."); pass
        ee=Monitor(CustomizableNewsvendorEnv(**ec)); ecb=EvalCallback(ee,best_model_save_path=bmsp,log_path=os.path.join(self.log_dir,"eval_logs"),eval_freq=max(5000,tt//10),n_eval_episodes=5,deterministic=True,render=False)
        try:
            ms=self.model_kwargs.get('seed',None); self.model=self.model_class(self.policy,ve,verbose=0,seed=ms,**self.model_kwargs); print(f"Start SB3 train {self.name} ({tt} steps)...")
            self.model.learn(total_timesteps=tt,callback=ecb,log_interval=100,**self.train_kwargs); self.training_time=time.time()-st; print(f"Train SB3 {self.name} done: {self.training_time:.2f}s.")
            self.model.save(sp); print(f"SB3 final DummyVecEnv([_create_env(0, self.model_kwargs.get('seed', 0))])
        # if N_ENVS_TRAIN_SB3 > 1: try: vec_env = SubprocVecEnv([_create_env(i, self.model_kwargs.get('seed', 0)) for i in range(N_ENVS_TRAIN_SB3)], start_method='fork'); except: print("SB3 SubprocVec fail, use Dummy."); pass

        eval_env_callback = MonitorSB3(EnvClass(**env_config))
        eval_callback = EvalCallback(eval_env_callback, best_model_save_path=best_model_save_path, log_path=os.path.join(self.log_dir, "eval_logs"), eval_freq=max(5000, total_timesteps // 10), n_eval_episodes=5, deterministic=True, render=False)
        try:
            model_seed = self.model_kwargs.get('seed', None)
            self.model = self.model_class(self.policy, vec_env, verbose=0, seed=model_seed, **self.model_kwargs)
            print(f"Starting SB3 training ({self.name}, {total_timesteps} steps)...")
            self.model.learn(total_timesteps=total_timesteps, callback=eval_callback, log_interval=50, **self.train_kwargs)
            self.training_time = time.time() - start_time; print(f"SB3 Training ({self.name}) finished: {self.training_time:.2f}s.")
            self.model.save(save_path); print(f"SB3 Final model saved: {save_path}")
            try:
                 best_model_path = os.path.join(best_model_save_path, "best_model.zip")
                 if os.path.exists(best_model_path): print(f"Loading SB3 best model: {best_model_path}..."); self.model = self.model_class.load(best_model_path)
                 else: print("SB3 Best model not found, use final.")
            except Exception as e_load: print(f"Warn: SB3 No load best model. Err: {e_load}")
        except Exception as e: print(f"!!! ERROR SB3 train {self.name}: {e}"); import traceback; traceback.print_exc(); self.model = None; self.training_time = time.time() model saved: {sp}")
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
    def get_action(self, o, e):
        if self.model is None: a=e.action_space.sample(); return a.astype(e.action_space.dtype)
        a,_=self.model.predict(o.astype(np.float32),deterministic=True); return a.astype(e.action_space.dtype)

# <<<<<<<<<<<<<<<<< NEW RLlib Agent Wrapper >>>>>>>>>>>>>>>>>
class RLlibAgentWrapper(BaseAgent):
    def __init__(self, algo_name: str, config_updates: dict = None, name: str = None):
        if not RAY_AVAILABLE: raise ImportError("Ray RLlib is not available to instantiate RLlibAgentWrapper.")
        super().__init__(name=name if name else f"RLlib_{algo_name}")
        self.algo_name = algo_name
        self.config_updates = config_updates if config_updates is not None else {}
        self.policy = None
        self.algo_instance = None # Store the algorithm instance if needed
        self.checkpoint_dir = os.path.join(MODEL_DIR, f"{self.name}_checkpoints") # Specific dir for checkpoints
        self.training_log_path = os.path.join(LOG_DIR, f"{self.name}_train_log.csv") # Custom log file

    def _build_config(self, env_config):
        # --- Configure Algorithm ---
        # Use AlgorithmConfig().algo_class(...) for newer Ray versions
        # Or AlgorithmConfig(algo_name) for older versions if needed
        try:
            # - start_time
        finally: vec_env.close(); eval_env_callback.close()

    def load(self, path: str):
         if not SB3_AVAILABLE: return
         try: print(f"Loading SB3 model {self.name} from {path}"); self.model = self.model_class.load(path)
         except Exception as e: print(f"!!! ERROR loading SB3 {self.name}: {e}"); self.model = None

    def get_action(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        if self.model is None: action = env.action_space.sample(); return action.astype(env.action_space.dtype)
        action, _ = self.model.predict(observation.astype(np.float32), deterministic=True)
        return action.astype(env.action_space.dtype)


# RLlib Wrapper
class RLlibAgentWrapper(BaseAgent):
    def __init__(self, algo_name: str, config_updates: dict = None, name: str = None):
        super().__init__(name=name if name else f"RLlib_{algo_name}")
        if not RLLIB_AVAILABLE: raise ImportError("Ray RLlib not available.")
        self.algo_name = algo_name
        self.config_updates = config_updates if config_updates is not None else {}
        self.policy = None
        self.algo = None # Stores the trained Algorithm instance
        self.checkpoint_path = None # Stores path to best/last checkpoint

    def train(self, env_config: dict, total_timesteps: int, save_path_prefix: str=""):
        print(f"Training {self.name} (RLlib)...")
        start_time = time.time()
        # Define specific checkpoint dir for this agent/run
        checkpoint_dir = os.path.join(MODEL Attempt to get the specific config class if needed for deep customization
            # e.g., from ray.rllib.algorithms.ppo import PPOConfig
            # config = PPOConfig() # Start with specific config
            # If not using specific config class, start with generic AlgorithmConfig
            config = AlgorithmConfig()

            config = config.environment(env=ENV_NAME_RLLIB, env_config=env_config)
            config = config.framework("torch") # Or "tf"
            config = config.rollouts(num_rollout_workers=max(0, N_WORKERS_RLLIB)) # 0 means use driver
            # Set resources based on available hardware (optional)
            # config = config.resources(num_gpus=int(os.environ.get("GPU", 0)))

            # Apply general training parameters
            config = config.training(
                gamma=self.config_updates.get("gamma", 0.99),
                lr=self.config_updates.get("lr", 5e-5), # Example default LR
                # Add model config if needed (e.g., network size)
                # model={"fcnet_hiddens": [64, 64]}
_DIR_RLLIB, f"{save_path_prefix}{self.name}")

        # --- Configure Algorithm ---
        try:
            print(f"  Building RLlib config for {self.algo_name}...")
            config = (
                # Use AlgorithmConfig and specify algo name if needed
                AlgorithmConfig(algo_class=self.algo_name) # Use algo name string
                .environment(env=ENV_NAME_RLLIB, env_config=env_config) # Use registered env name
                .framework("torch") # Or "tf"
                .rollouts(num_rollout_workers=N_WORKERS_RLLIB) # Use configured workers
                # Add more default configs common across algorithms if desired
                .training(gamma=0.99, lr=1e-4            )
            # Apply specific algorithm config updates
            if self.algo_name == "PPO":
                 config = config.training(sgd_minibatch_size=self.config_updates.get("sgd_minibatch_size", 128))
                 config = config.training(num_sgd_iter=self.config_updates.get("num_sgd_iter", 10))
                 # config = config.training(model={"fcnet_hiddens": [64, 64]}) # Example architecture
            elif self.algo_name) # Example common parameters
                # Specify resources (optional, Ray defaults usually work)
                # .resources(num_gpus=0) # Example: force CPU
            )

            # Apply specific config updates
            # == "SAC":
                 config = config.training(optimization={"actor_learning_rate": 3e-4, "critic_learning_rate": 3e-4, "entropy_learning_rate": 3e-4})
                 # config = config.training(model={"fcnet_hiddens": [ This needs mapping from simple dict keys to RLlib's structure
            if 'lr' in self.config_updates: config = config.training(lr=self.config_updates['lr'])
            if 'gamma' in self.config_updates: config = config.training(gamma=self.config_updates['gamma'])256, 256]})
            # Add elif for TD3, DDPG, A2C specific params if needed

            # Apply direct overrides from self.config_updates (use cautiously)
            # This
            # Add more sophisticated config updates based on algo type if needed
            if self.algo_name == "PPO":
                 config = config.training(sgd_minibatch_size=128, num_sgd_iter requires knowing the exact structure of AlgorithmConfig
            # config = config.update_from_dict(self.config_updates) # Be careful with this

            config=10) # Example PPO params
            if self.algo_name == "SAC":
                 config = config.training(initial_alpha=0.2.validate() # Check if config is valid
            return config

        except Exception as e:
            print(f"!!! ERROR building RLlib config for {self.name}: {e}")
            raise # Re-raise the error

    def train(self, env_config: dict, total_timesteps, target_entropy="auto") # Example SAC params

            print(f"  : int, save_path_prefix: str=""):
        print(f"Training {self.name} with RLlib...")
        start_Config Built for {self.algo_name}.")
            # Build the Algorithm instance
            self.algo = config.build()
            print(f"  RLlib Algorithm instance built for {self.name}.")

        except Exception as e:
time = time.time()
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        last_checkpoint = ray.rllib.train.get             print(f"!!! ERROR building RLlib config/algo for {self.name}: {e}")
             import traceback; traceback.print_exc_checkpoint_dir(self.checkpoint_dir) # Find latest checkpoint

        #(); return

        # --- Restore or Train ---
        restored = False
        if not --- Build/Restore Algorithm ---
        try:
            config = self._build_config(env_config)
            # Use the correct class string identifier known to RLlib
            self.algo_ FORCE_RETRAIN:
             # Try restoring from the standard checkpoint location
             if os.path.exists(checkpoint_dir):
                 try:
                     print(f"Attempting to restore RLinstance = config.build(use_copy=False) # Build the algorithm instancelib agent {self.name} from {checkpoint_dir}")
                     self.algo.restore(

            # Restore if checkpoint exists and retraining not forced
            if not FORCE_RETRAIN and last_checkpointcheckpoint_dir) # Restore latest checkpoint in dir
                     print(f"Successfully restored RLlib agent {self.name}.")
                     # Check if already trained enough
                     if self.algo:
                print(f"Restoring {self.name} from checkpoint: {last_checkpoint}")
                self.algo_instance.restore(.iteration * self.algo.config.train_batch_size >= total_timesteps: #last_checkpoint)
                # Check if already trained sufficiently
                if self.algo_instance.iteration * self.algo_instance.config.train_batch_size >= total_timesteps:
                     print("RLlib model already trained.")
                     self.policy Approximate check
                           print("RLlib agent already trained sufficiently.")
                           self.training_time = 0
                           self.policy = self.algo.get_policy() # Get policy = self.algo_instance.get_policy() # Get policy for evaluation
                     self.training_time = 0
                     self.algo_instance.stop() # Stop instance if not training
                     self.algo_instance = None
                      from restored algo
                           self.algo.stop() # Stop algo if no training needed
                           return # Skip training loop
                     else: restored = True #return
            elif FORCE_RETRAIN and last_checkpoint:
                 print(" Continue training
                 except Exception as e_restore:
                     print(f"FORCE_RETRAIN is True, ignoring existing checkpoint.")

        except Exception as e:
             print(f"!!! ERROR building/restoring RLlib algo for {selfCould not restore RLlib agent {self.name}. Training from scratch. Error: {e_restore}")


        # --- Training Loop ---
        print(f"Starting RLlib training ({self.name},.name}: {e}"); import traceback; traceback.print_exc(); return

        # --- Training Loop {total_timesteps} steps)...")
        timesteps_current = self ---
        timesteps_total = self.algo_instance.iteration *.algo.iteration * self.algo.config.train_batch_size if restored self.algo_instance.config.train_batch_size if self. else 0
        iteration = self.algo.iteration if restored else 0
        try:
            while timesteps_current < total_timesteps:
                iteration += 1
                result = self.algo.train()
                timalgo_instance.iteration > 0 else 0
        iteration = self.algo_instance.iteration
        training_log_data = []
        try:
            print(f"Starting RLlib training ({self.name}) for approx {total_timesteps} steps (current: {timesteps_total})...")
            esteps_current = result["timesteps_total"]
                mean_reward = result["episode_reward_mean"] if "episode_reward_mean" in result else float('nan')

                if iteration % 20 == 0: # Print progress lesswhile timesteps_total < total_timesteps:
                iteration += 1
                result = self.algo_instance.train()
                timesteps_total = result["timesteps_total"]
                mean_reward = result.get("episode_reward_mean", float('nan')) # Use .get for safety
                training_log_data.append({'iteration': iteration, 'timesteps': timesteps_total, often for RLlib
                    print(f"  Iter: {iteration}, Tim 'episode_reward_mean': mean_reward})

                if iteration % 10 == 0: # Log progress and save checkpoint
                    print(f"esteps: {timesteps_current}/{total_timesteps}, Mean Reward: {mean_reward:.2f}")
                    # Save checkpoint periodically
                    checkpoint_path = self.algo.save(checkpoint_dir)
                    # print(f"  RLlib Checkpoint saved to {checkpoint_path}")

            self.training_time = time.time() - start_time
            print(f"RLlib Training ({self.name}) finished in {self.training_time:.2f} seconds.")
            #  Iter: {iteration}, Steps: {timesteps_total}/{total_timesteps}, Mean Reward: {mean_reward:.2f}")
                    checkpoint_path = self.algo_instance Save final state
            self.checkpoint_path = self.algo.save(checkpoint_dir)
            print(f"Final RLlib checkpoint saved to {self.checkpoint_path}")
            # Store the policy for evaluation
            self.policy = self.algo.get_policy()

        except Exception as e:
            print(f"!!! ERROR during RLlib training loop for {self.name}: {e.save(self.checkpoint_dir)
                    # print(f"  Checkpoint saved to {checkpoint_path}") # Can be verbose

            self.training_time = time.time() - start_time
            print(f"Training for {self.name} finished in {self.training_time:.2f} seconds.")
            final_checkpoint_path = self.algo_instance.save(self.checkpoint_dir) # Save final state
            print(f"Final RLlib checkpoint saved to {final_checkpoint_path}")
            self.policy = self.algo_instance.get_policy() # Get policy for evaluation

        except Exception as e:
            print(f"!!! ERROR during RLlib training loop for {self.name}")
            import traceback; traceback.print_exc()
            self.training_time = time.time() - start_time
        finally:
            if self.algo: self.algo.stop() # Stop algo after training/error


    def load(self, path: str):
         # This load is primarily for SB3 structure, RLlib restore happens in train
         #}: {e}")
            import traceback; traceback.print_exc()
            self.training_time = time.time() - start_time # Record time even if failed
        finally:
            if self.algo_instance: self.algo_instance.stop() # Stop Ray actors
            # Save collected training logs
            if training_log_data:
                 log_df = pd.DataFrame(training_log_data)
                 log If needed for pure evaluation later, would need to build & restore here.
         print(f"RLlibAgentWrapper.load called for {self.name} (path: {path}). Restoration_df.to_csv(self.training_log_path, index=False)
                 print(f"Saved RLlib training log to {self.training_log_path}")

    def load(self, path: str):
         # Path for RLlib is usually the checkpoint directory
         print(f"RL happens in train or via Algorithm class.")
         self.checkpoint_path = path # Storelib load called for {self.name}. Restoration happens in train() or needs path if provided externally


    def get_action(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        # If policy wasn Algorithm class.")
         # Attempt basic policy loading if needed outside train (less common for't stored after training/loading, try to restore temporarily
        # THIS benchmark)
         if self.algo_instance is None:
             print(" Cannot load - RLlib algo instance not available.")
             return
         try:
              IS INEFFICIENT - policy should ideally be ready.
        if self.policy is None and self.checkpoint_path and os.path.exists(self.checkpoint_path):
             self.algo_instance.restore(path)
             self.policy = self.algo_instance.get_policy()
             print(f"Restored {self.name} from checkpoint dir {path}")
         except Exception as e:
             print(f"!!! Error restoring {self.name} from {print(f"Policy not loaded for {self.name}, attempting restore from {self.checkpoint_path}")
             try:
                  # Need to build a temporary Algorithmpath}: {e}")
             self.policy = None

    def get_action(self, observation: np.ndarray, env: gym.Env) -> np.ndarray:
        if self.policy is None:
            print(f"Warn: Policy {self.name} not available. Random instance just to restore policy...
                  # This highlights a difference in workflow vs action."); action = env.action_space.sample(); return action.astype(env.action_ SB3 model loading.
                  # For simplicity in *this* script, ifspace.dtype)
        try:
            action_data = self.policy.compute_single_action(observation.astype(np.float32))
            action = action_data[0]
            return action.astype(env. policy is None, return random.
                  # A more robust implementation would handleaction_space.dtype)
        except Exception as e: print(f"!!! ERR {self.name} get_action: {e}. Random."); action stateful loading better.
                   print(f"Warning: Policy for {self.name} is = env.action_space.sample(); return action.astype(env.action_space.dtype)

# --- Evaluation Function (Reusable) ---
# (evaluate_agent function remains the same as in benchmark_newsvendor_advanced.py)
# It correctly uses agent None. Returning random action during evaluation.")
                   action = env.action_space.sample();.get_action() which works for both wrappers.
def evaluate_agent(agent return action.astype(env.action_space.dtype)
             except Exception as e_restore:
                  print(f"Failed to restore policy for {self.name}: BaseAgent,
                   env_config: dict,
                   n_episodes: int,
                   seed_offset: int = 0,
                   fixed_params: Optional[Dict] = None,
                   collect_details: bool = False ) -> Dict:
    # ... (Keep during get_action: {e_restore}")
                  action = env.action_space.sample(); return action.astype(env.action_space.dtype)

        elif self.policy is None:
             print(f"Warning: Policy for {self.name} not available. Returning random action.")
             action = env.action_space.sample(); return action.astype(env.action_space.dtype)

 the exact function from benchmark_newsvendor_advanced.py) ...
        # Policy exists, compute action
        try:
            action_data = self.policy.compute_single_action(observation.astype(np.float32))
            action = action_data[0]
            return action.astype(env.action_    # This function uses agent.get_action() polymorphically and extracts info
    # assuming the NewsvendorEnv info dict structure.
    episode_summaries = []; all_step_details = []; total_eval_time = 0space.dtype)
        except Exception as e:
            print(f"!!! ERROR during {self.name} get_action: {e}..0
    # Use CustomizableNewsvendorEnv for potential fixed params
    eval_env = CustomizableNewsvendorEnv(**env_config)
    fixed_str = " with fixed params" if fixed_params else " with random params"
    print(f"\nEvaluating {agent.name} for {n_episodes} episodes{ Returning random action.")
            action = env.action_space.sample(); return action.astype(env.action_space.dtype)

# --- Evaluation Function (Updated for Newsvendor Info) ---
def evaluate_agent(agent: BaseAgent,
                   env_config: dict,
                   n_episodes: int,
                   fixed_str}...")
    successful_episodes = 0
    for i in range(n_episodes):
        episode_seed = seed_offset + i; episode_step_details = []
        try:
            options = {"fixed_params": fixed_params} if fixed_params else None
            obs, info = eval_env.reset(seed=episode_seed, options=options)seed_offset: int = 0,
                   collect_details: bool = False
                   ) -> Dict:
    # ... (Keep the function adapted for NewsvendorEnv from benchmark_newsvendor_advanced.py) ...
    episode_summaries = []; all_step_details = []; total_eval_time = 0
            terminated, truncated = False, False; episode_reward, episode_steps = 0.0, 0
            episode_demand_total, episode_sales_total, episode_stockout_qty, episode_inventory_sum = 0.0, 0.0, 0.0, 0.0
            start_time = time.perf_counter()
            while not terminated and not truncated:
                action = agent.get_action(obs, eval_env)
                obs, reward.0
    eval_env = EnvClass(**env_config) # Use the correct EnvClass
    print(f"\nEvaluating {agent.name} for {n_episodes} episodes ({ENV_CLASS_NAME})...")
    successful_episodes = 0
    for i in range(n_episodes):
        episode_seed = seed_offset + i; episode_step_details = []
        try:
            obs, info = eval_env.reset(seed=episode_seed)
            terminated, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward; episode_steps += 1
                # Collect metrics from info dict
                demand = info.get('demand', 0); episode_demand_total += demand
                penalty = info.get('lost_sales, truncated = False, False; episode_reward, episode_steps = 0.0, 0
            episode_demand_total, episode_sales_total, episode_stockout_qty = 0.0, 0.0, 0.0
            episode_inventory_sum = 0.0
            start_time = time.perf_counter()
            while not terminated and not truncated:
                action = agent.get_action(obs,_penalty', 0); pen_rate = info.get('penalty_cost_rate', 1e-6)
                stockout = penalty / max(1e-6, pen_rate) if pen_rate > 1e-6 else 0; episode_stockout_qty += stockout
                sales = max(0, demand - stockout); episode_sales_total += sales
                excess = max(0, info.get('holding_cost', 0) / max( eval_env)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward; episode_steps += 1
                current_demand = info.get('demand', 0); episode_demand_total += current_demand
                lost_sales_penalty = info.get('lost_sales_penalty', 0); penalty_rate = info.get('penalty_cost_rate', 1e-6)
                stockout_this_step = lost_sales_penalty / max(1e-6, penalty_rate) if penalty_rate > 1e-6 else 0
                1e-6, info.get('holding_cost_rate', 1e-6))); episode_inventory_sum += excess
                if collect_details: episode_step_details.append({'step': episode_steps, 'reward': reward, 'action': action.item() if action.size==1 else action.tolist(), 'demand': demand, 'sales': sales, 'stocksales_this_step = max(0, current_demand - stockout_this_step); episode_sales_total += sales_this_step
                episode_stockout_qty += stockout_this_step
                excess_inv_this_step = max(0, info.get('holding_cost', 0) / max(1e-6, info.get('holding_cost_rate', 1e-6)))
                episode_inventory_sum += excess_inv_this_step
                if collect_details: step_data = {'step': episode_steps, 'reward': reward, 'action': action.item() if action.size==1out_qty': stockout, 'ending_inv': excess})
            end_time = time.perf_counter(); episode_time = end_time - start_time; total_eval_time += episode_time
            avg_ending_inv = episode_inventory_sum / episode_steps if episode_steps > 0 else 0
            service_level = episode_sales_total / max(1e-6, episode_demand_total) if episode_demand_total > 1e-6 else 1.0
            ep_summary = {"Agent": agent.name, "Episode": i + 1, "TotalReward": episode_reward, "Steps": episode_steps, "Time": episode_time, "Seed": episode_seed, "AvgServiceLevel": service_level, "Total else action.tolist(),'demand': current_demand, 'sales': sales_this_step, 'stockout_qty': stockout_this_step, 'ending_inv': excess_inv_this_step}; step_data.update({k:v for k,v in info.items() if k in ['price','cost','h','k','mu']}); episode_step_details.append(step_data)
            end_time = time.perf_counter(); episode_time = end_time - start_time; total_eval_time += episode_time
            avg_ending_inv = episode_inventory_sum / episode_steps if episode_steps > 0 else 0
            service_level = episode_sales_total / max(1StockoutQty": episode_stockout_qty, "AvgEndingInv": avg_ending_inv, "Error": None}
            episode_summaries.append(ep_summary); all_step_details.append(episode_step_details); successful_episodes += 1
            if n_episodes <= 20 or (i + 1) % max(1, n_episodes // 5) == 0: print(f"  Ep {i+1}/{n_episodes}: Rw={episode_reward:.0f}, SL={service_level:.1%}")
        except Exception as e: print(f"!!! ERR ep {i+1} {agent.name}: {e}"); import traceback; traceback.print_exc(); episode_summaries.append({"Agent": agent.name, "Episode": i + 1, "TotalReward": np.nan, "Steps": 0,"Time": 0,e-6, episode_demand_total) if episode_demand_total > 1e-6 else 1.0
            ep_summary = {"Agent": agent.name, "Episode": i + 1, "TotalReward": episode_reward,"Steps": episode_steps, "Time": episode_time, "Seed": episode_seed,"AvgServiceLevel": service_level,"TotalStockoutQty": episode_stockout_qty,"AvgEndingInv": avg_ending_inv,"Error": None}
            episode_summaries.append(ep_summary); all_step_details.append(episode_step_details); successful_episodes += 1
            if n_episodes <= 20 or (i + 1) % max(1, n_episodes // 5) == 0: print(f"  Ep {i+1}/{n_episodes}: Reward={episode_reward:.2f}, ServiceLvL={service_level:. "Seed": episode_seed, "AvgServiceLevel": np.nan,"TotalStockoutQty": np.nan, "AvgEndingInv": np.nan, "Error": str(e)}); all_step_details.append([])
    eval_env.close()
    if successful_episodes == 0: print(f"Eval FAIL {agent.name}."); return {'summary': pd.DataFrame(), 'details': []}
    avg_eval_time = total_eval_time / successful_episodes; print(f"Eval done {agent.name}. Avg time: {avg_eval_time:.4f}s")
    return {'summary': pd.DataFrame(episode_summaries), 'details': all_step_details}


# --- Results Processing (Reusable) ---
# (process_and_report_results function can be reused)
def process_and2%}")
        except Exception as e: print(f"!!! ERROR ep {i+1} {agent.name}: {e}"); import traceback; traceback.print_exc(); episode_summaries.append({"Agent": agent.name, "Episode": i + 1, "TotalReward": np.nan, "Steps": 0,"Time": 0, "Seed": episode_seed, "AvgServiceLevel": np.nan,"TotalStockoutQty": np.nan, "AvgEndingInv": np.nan, "Error": str(e)}); all_step_details.append([])
    eval_env.close()
    if successful_episodes == 0: print(f"Eval FAILED {agent.name}."); return {'summary': pd.DataFrame(), 'details': []}
    avg_eval_time = total_eval_time / successful_episodes; print(f"Eval finished {agent.name}._report_results(all_eval_results: List[Dict], agent_objects: Dict):
    # ... (Keep the exact function from benchmark_newsvendor_advanced.py) ...
    if not all_eval_results: print("No results."); return None, None
    all_summaries_list = [res['summary'] for res in all_eval_results if 'summary' in res and not res['summary'].empty]
    if not all_summaries_list: print("No summaries."); return None, None
    results_df_raw_summary = pd.concat(all_summaries_list, ignore_index=True)
    print("\n--- Benchmark Summary ---")
    summary = results_df_raw_summary.dropna(subset=['TotalReward']).groupby("Agent").agg(
        AvgReward=("TotalReward", "mean"), MedianReward=("TotalReward", "median"), StdReward=("TotalReward", Avg time: {avg_eval_time:.4f}s")
    return {'summary': pd.DataFrame(episode_summaries), 'details': all_step_details}

# --- Results Processing (Reusable) ---
def process_and_report_results(all_eval_results: List[Dict], agent_objects: Dict):
    # ... (Keep the exact function from benchmark_newsvendor_advanced.py) ...
    if not all_eval_results: print("No results."); return None, None
    all_summaries_list = [res['summary'] for res in all_eval_results if 'summary' in res and not res['summary'].empty]
    if not all_summaries_list: print("No summaries."); return None, None
    results_df_raw_summary = pd.concat(all_summaries_list, ignore_index=True)
    print("\n--- Benchmark Summary ---")
    summary = results "std"),
        MinReward=("TotalReward", "min"), MaxReward=("TotalReward", "max"), AvgServiceLevel=("AvgServiceLevel", "mean"),
        AvgStockoutQty=("TotalStockoutQty", "mean"), AvgEndInv=("AvgEndingInv", "mean"), AvgTimePerEp=("Time", "mean"),
        SuccessfulEpisodes=("Episode", "count"))
    summary["TrainingTime(s)"] = summary.index.map(lambda name: agent_objects.get(name, BaseAgent(name)).get_training_time()).fillna(0.0)
    total_episodes_attempted = results_df_raw_summary.groupby("Agent")["Episode"].count()
    summary["EpisodesAttempted"] = total_episodes_attempted
    summary["SuccessRate(%)"] = (summary["SuccessfulEpisodes"] / summary["EpisodesAttempted"]) * 100
    summary = summary.sort_values(by="AvgReward", ascending=False)
    summary = summary[["AvgReward", "MedianReward", "StdReward", "MinReward", "MaxReward", "AvgServiceLevel",
                       "AvgStock_df_raw_summary.dropna(subset=['TotalReward']).groupby("Agent").agg(
        AvgReward=("TotalReward", "mean"), MedianReward=("TotalReward", "median"), StdReward=("TotalReward", "std"),
        MinReward=("TotalReward", "min"), MaxReward=("TotalReward", "max"), AvgServiceLevel=("AvgServiceLevel", "mean"),
        AvgStockoutQty=("TotalStockoutQty", "mean"), AvgEndInv=("AvgEndingInv", "mean"), AvgTimePerEp=("Time", "mean"),
        SuccessfulEpisodes=("Episode", "count"))
    summary["TrainingTime(s)"] = summary.index.map(lambda name: agent_objects.get(name, BaseAgent(name)).get_training_time()).fillna(0.0)
    total_episodes_attemptoutQty", "AvgEndInv", "AvgTimePerEp", "TrainingTime(s)", "SuccessfulEpisodes",
                       "EpisodesAttempted", "SuccessRate(%)"]]
    pd.set_option('display.float_format', lambda x: '%.2f' % x); print(summary)
    raw_summary_path = os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_raw_summary.csv")
    summary_path = os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_summary.csv")
    details_path = os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_step_details.jsonl")
    try:
        results_df_raw_summary.to_csv(raw_summary_pathed = results_df_raw_summary.groupby("Agent")["Episode"].count()
    summary["EpisodesAttempted"] = total_episodes_attempted
    summary["SuccessRate(%)"] = (summary["SuccessfulEpisodes"] / summary["EpisodesAttempted"]) * 100
    summary = summary.sort_values(by="AvgReward", ascending=False)
    summary = summary[["AvgReward", "MedianReward", "StdReward", "MinReward", "MaxReward", "AvgServiceLevel",
                       "AvgStockoutQty", "AvgEndInv", "AvgTimePerEp", "TrainingTime(s)", "SuccessfulEpisodes",
                       "EpisodesAttempted", "SuccessRate(%)"]]
    pd.set_option('display.float_, index=False); summary.to_csv(summary_path)
        print(f"\nRaw summary saved to {raw_summary_path}"); print(f"Summary saved to {summary_path}")
        if COLLECT_STEP_DETAILS:
             print(f"Saving step details to {details_path}...")
             with open(details_path, 'w') as f:
                 for i, agent_results in enumerate(all_eval_results):
                     agent_name = agent_results['summary']['Agent'].iloc[0] if 'summary' in agent_results and not agent_results['summary'].empty else f"Unknown_{i}"
                     for ep_num, steps in enumerate(agent_results.get('details', [])):
                         for step_data in steps:
                             step_data['agent'] = agent_name; step_data['episode'] = ep_num + 1
                             serializable_data = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for kformat', lambda x: '%.2f' % x); print(summary)
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
                     agent_name = agent_results['summary']['Agent'].iloc[0] if 'summary' in, v in step_data.items()}
                             f.write(json.dumps(serializable_data) + '\n')
             print("Step details saved.")
    except Exception as e: print(f"\nError saving results: {e}")
    return summary, results_df_raw_summary


# --- Plotting Functions ---
# (Updated plot_learning_curves to handle RLlib logs)
def plot_learning_curves(log_dirs: Dict[str, str], rllib_agent_names: List[str], title: str = "RL Learning Curves"):
    """Plots training curves from SB3 Monitor logs and custom RLlib CSV logs."""
    plt.figure(figsize=(12, 7))
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward (Smoothed)")
    plotted_something = False

    for agent_name, log_dir in log_dirs.items agent_results and not agent_results['summary'].empty else f"Unknown_{i}"
                     for ep_num, steps in enumerate(agent_results.get('details', [])):
                         for step_data in steps: step_data['agent'] = agent_name; step_data['episode'] = ep_num + 1; serializable_data = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in step_data.items()}; f.write(json.dumps(serializable_data) + '\n')
             print("Step details saved.")
    except Exception as e: print(f"\nError saving results: {e}")
    return summary, results_df_raw_summary


# --- Plotting Functions ---
def plot_learning_curves(sb3_log_dirs: Dict[str, str], rllib_log_dirs: Dict[str, str], title: str):
    """Plots training curves from SB3 Monitor logs and potentially RLlib logs."""
    plt.figure(figsize=(14, 8)); plt.title(title); plt.xlabel("Timesteps"); plt.ylabel("Reward (Smoothed)"); plotted=False

():
        x, y = None, None
        is_rllib = agent_name in rllib_agent_names

        # --- Try loading SB3 Monitor logs ---
        if not is_rllib and SB3_AVAILABLE:
            monitor_files = glob.glob(os.path.join(log_dir, "*logs", "monitor.*.csv"))
            if monitor_files:
                monitor_path = os.path.dirname(monitor_files[0])
                try:
                    results = load_results(monitor_path)
                    if len(results['r']) > 0:
                        x, y_raw = ts2xy(results, 'timesteps')
                        if len(x) > 10: y = results.rolling(window=10).mean()['r'] # Smooth
                        else: y = y_raw # Use raw if too short    # Plot SB3 Curves
    if SB3_AVAILABLE:
        print("Plotting SB3 learning curves...")
        for name, log_dir in sb3_log_dirs.items():
            monitor_files = glob.glob(os.path.join(log_dir, "*logs", "monitor.*.csv"));
            if not monitor_files: print(f"  Warn: No SB3 logs for {name}"); continue
            monitor_path = os.path.dirname(monitor_files[0])
            try:
                results = load_results(monitor_path);
                if len(results['r']) > 0:
                    x, y = ts2xy(results, 'timesteps');
                    if len(x) > 50: y = results.rolling(window=50).mean()['r'] # Smooth
 to smooth
                except Exception as e: print(f"Error loading SB3 log {agent_name}: {e}")

        # --- Try loading custom RLlib CSV logs ---
        elif is_rllib and RAY_AVAILABLE:
            rllib_log_file = os.path.join(LOG_DIR, f"{agent_name}_train_log.csv") # Path defined in RLlibAgentWrapper
            if os.path.exists(rllib_log_file):
                try:
                    log_df = pd.read_csv(rllib_log_file)
                    if not log_df.empty and 'timesteps' in log_df and 'episode_reward_mean' in log_df:
                         log_df = log_df.dropna                    plt.plot(x, y, label=f"{name} (SB3)"); plotted=True
                else: print(f"  Warn: SB3 Log {name} empty.")
            except Exception as e: print(f"  Error plot SB3 logs {name}: {e}")

    # Plot RLlib Curves (Basic attempt - needs refinement based on actual log structure)
    if RLLIB_AVAILABLE:
        print("Plotting RLlib learning curves...")
        # RLlib typically logs to ~/ray_results/Experiment_Name/Algo_Name(subset=['episode_reward_mean']) # Drop rows with NaN reward
                         x = log_df['timesteps'].values
                         y_raw = log_df['episode_reward_mean'].values
                         if len(x) > 10: y = log_df['episode_reward_mean'].rolling(window=10, min_periods=1).mean().values # Smooth
                         else: y_.../progress.csv
        # This basic version assumes rllib_log_dirs points *near* the experiment dir
        for name, log_dir_hint in rllib_log_dirs.items(): # log_dir_hint might = y_raw
                except Exception as e: print(f"Error loading RLlib log {agent_name}: {e}")

        # --- Plot if data loaded ---
        if x is not None and y is not None and len(x) > 0 and len(y) > 0: not be precise enough
             try:
                 # Need a more robust way to find the specific
             # Ensure x and y have same length after potential smoothing NaNs
             min_len = min(len(x), len(y))
             x = x[:min_len]
             y = y[:min_len]
              trial log dir
                 # Example: find latest experiment dir? Or use Ray# Clip x axis if needed (e.g., if training was restored and Tune Analysis API?
                 # Simplified: Look for progress.csv within sub went longer)
             # x_plot = x[x <= RL_TRAINING_TIMESTEPS]
             # y_plot = y[xdirs
                 progress_files = glob.glob(os.path.join(log_dir_hint, "*", "progress.csv"), recursive=True) \
                                + glob.glob(os.path.join(LOG_DIR_RLLIB, f <= RL_TRAINING_TIMESTEPS]
             plt.plot(x, y, label=agent_name)
             plotted_something = True
        "{name}*", "*", "progress.csv"), recursive=True) \
                                + glob.glob(os.path.join(LOG_DIR_RLLIB, name, "progress.csv")) # More guesses
                 progress_files = sorted(list(setelif not is_rllib or (is_rllib and not os.path.exists(rllib_log_file)):
             print(f"Warn: No training log data found for {agent_name}")


    if plotted_something:
         plt.legend(loc='best')
         plt.grid(True)
         plt.tight_layout()
         learning_curve_path = os.path.join(RESULTS_DIR, f(progress_files))) # Unique, sort maybe by time?

                 if not progress_files: print(f"  Warn: No RLlib progress.csv found near {log_dir_hint} for {name}"); continue

                 #"{ENV_NAME_SHORT}_benchmark_learning_curves.png")
         plt.savefig(learning_curve_path)
         print(f"Saved learning curves plot to {learning_curve_path}")
         plt.close()
    else:
         print("Skipping learning curve plot - no data Load the latest one found (heuristic)
                 df = pd.read_csv(progress_files[-1])
                 print(f"  Loaded RLlib log found.")

# (plot_benchmark_results function can be reused as is, it just needs the summary DFs)
def plot_benchmark_results(df_summary: pd.DataFrame, df_raw_summary: pd.DataFrame):
    # ... (Keep the exact function from benchmark_netinvmgmt.py) ...
    : {progress_files[-1]}")

                 # Identify relevant columns (these names can vary!)
                 reward_col = 'episode_reward_mean'
                 # It uses the correct RESULTS_DIR and ENV_NAME_SHORT due to global scope
    # and plots based on the columns generated by process_and_report_results
    timestep_col = 'timesteps_total'
                 if reward_col not in df.columns: reward_col = 'policy_reward_mean/default_policy' # Alternative name
                 if timestep_col not in df.columns: timestep_col = 'agentif df_summary is None or df_raw_summary is None: print("Skip plot: no data."); return
    print("\nGenerating comparison plots..."); n=df_summary.shape[0]; plt.style.use('seaborn-v0_8-darkgrid')
    df_s = df_summary.sort_values("AvgReward", ascending=False); order = df_s_timesteps_total' # Alternative name

                 if reward_col in df.columns and timestep_col in df.columns:
                     df_plot = df[[timestep_col, reward_col]].dropna()
                     # Smooth
                     if len(df_plot) > 10:
.index
    # 1. Box Plot Rewards
    plt.figure(figsize=(10, max(6, n * 0.5))); sns.boxplot(data=df_raw_summary, x="TotalReward", y="Agent", palette="viridis", showfliers                          y_smooth = df_plot[reward_col].rolling(window=10, min_periods=1).mean()
                     else: y_smooth = df_plot[reward_col]
                     plt.plot(df_plot[timestep_col], y_smooth, label=f"{=False, order=order); plt.title(f"Reward Distribution ({ENV_NAME_SHORT} - {N_EVAL_EPISODES} Eps)"); plt.xlabel("Total Reward"); plt.ylabel("Agent"); plt.tight_layout(); p=os.path.join(RESULTS_DIR,f"{ENV_NAME_SHORT}_benchmark_rewards_boxplotname} (RLlib)", linestyle='--') # Dashed line for RLlib
                     plotted=True
                 else: print(f"  Warn: Could not find expected columns ({timestep_col}, {reward_col}) in RLlib log for {name}")

             except Exception as e: print(f"  Error processing RLlib logs for {name}: {.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    # 2. Bar Eval Time
    df_s_eval = df_s.sort_values("AvgTimePerEp", ascending=True); plt.figure(figsize=(10, max(6, n * 0.4))); idx=np.arange(len(df_s_eval)); plt.barh(idx, df_s_eval["AvgTimePerEp"], colore}")

    if plotted: plt.legend(); plt.tight_layout(); p=os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_learning_curves.png"); plt.savefig(p); print(f"Saved learning curves: {p}"); plt.close()
    else: print("Skip learning curves - no data.")


def plot_benchmark_results(df_summary: pd.DataFrame, df_raw_summary: pd.='skyblue', log=True); plt.yticks(idx, df_s_eval.index); plt.xlabel('Avg Eval Time per Ep (s) - Log Scale'); plt.ylabel('Agent'); plt.title(f'Avg Eval Time ({ENV_NAME_SHORT})'); plt.tight_layout(); p=os.path.join(RESULTS_DIR,f"{ENV_NAME_SHORT}_benchmark_eval_time_log.png"); plt.savefig(p); print(f"DataFrame):
    # ... (Keep the exact function from benchmark_netinvmgmt.py) ...
    # It uses ENV_NAME_SHORT correctly for filenames/titles
    if df_summary is None or df_raw_summary is None: print("Skip plot: no data."); return
    print("\nGenerating comparison plots..."); n=df_summary.shape[0]; plt.styleSaved: {p}"); plt.close()
    # 3. Bar Train Time
    df_t = df_s[df_s["TrainingTime(s)"] > 1].sort_values("TrainingTime(s)", ascending=True)
    if not df_t.empty: plt.figure(figsize=(8, max(4, len(df_t)*0.5))); idx_t=np.arange(len(df_t)); plt.barh(idx_t, df_t["TrainingTime(s)"],.use('seaborn-v0_8-darkgrid')
    df_s = df_summary.sort_values("AvgReward", ascending=False); order = df_s.index
    # 1. Box Plot Rewards
    plt.figure(figsize=(10, max(6, n * 0.5))); sns.boxplot(data=df_raw_summary, x="TotalReward", color='lightcoral'); plt.yticks(idx_t, df_t.index); plt.xlabel('Total Training Time (s)'); plt.ylabel('Agent (RL)'); plt.title(f'Training Time ({ENV_NAME_SHORT})'); plt.tight_layout(); p=os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_train_time.png"); plt.savefig(p y="Agent", palette="viridis", showfliers=False, order=order); plt.title(f"Reward Distribution ({ENV_NAME_SHORT} - {N_EVAL_EPISODES} Eps)"); plt.xlabel("Total Reward"); plt.ylabel("Agent"); plt.tight_layout(); p=os.path.join(RESULTS_DIR,f"{ENV_NAME_SHORT}_benchmark_rewards_boxplot); print(f"Saved: {p}"); plt.close()
    else: print("Skip train time plot.")
    # 4. Scatter Reward vs Service
    plt.figure(figsize=(9, 7));.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    # 2. Bar Eval Time
    df_s_eval = df_s.sort_values("AvgTimePerEp", ascending=True); plt.figure(figsize=(10, max(6, n * 0.4))); idx sns.scatterplot(data=df_s, x="AvgServiceLevel", y="AvgReward", hue="Agent", s=100, palette="viridis", legend=False);
    for _, r in df_s.iterrows(): plt.text(r["AvgServiceLevel"] + 0.005, r["AvgReward"], r.name, fontsize=9)
    plt.title(f"Reward vs. Service ({ENV_NAME_SHORT})"); plt.xlabel("Avg Service Level (Fill Rate)"); plt.ylabel("Avg Reward"); plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0)); plt.grid(True); plt.tight_layout(); p=os.path.join=np.arange(len(df_s_eval)); plt.barh(idx, df_s_eval["AvgTimePerEp"], color='skyblue', log=True); plt.yticks(idx, df_s_eval.index); plt.xlabel('Avg Eval Time per Ep (s) - Log Scale'); plt.ylabel('Agent'); plt.title(f'Avg Eval Time ({ENV_NAME_SHORT})'); plt.tight_layout(); p=os.path.join(RESULTS_DIR,f"{ENV_NAME_SHORT}_benchmark_eval_time_log.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    # 3. Bar Train Time
    df_t = df_s[df_s["TrainingTime(s)"] > 1].sort_values("TrainingTime(s)", ascending=True)(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_reward_vs_service.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    # 5. Scatter Reward vs Inventory
    plt.figure(figsize=(9, 7)); sns.scatterplot(data=df_s, x="AvgEndInv", y="AvgReward", hue="Agent", s=100, palette="viridis", legend=False);
    for _, r in df_s.iterrows(): plt.text(r["AvgEndInv"] * 1.01, r["AvgReward"], r.name, fontsize=9)
    plt.title
    if not df_t.empty: plt.figure(figsize=(8, max(4, len(df_t)*0.5))); idx_t=np.arange(len(df_t)); plt.barh(idx_t, df_t["TrainingTime(s)"], color='lightcoral'); plt.yticks(idx_t, df_t.index); plt.xlabel('Total Training Time (s)'); plt.ylabel('Agent (RL)'); plt.title(f'Training Time ({ENV_NAME_SHORT})'); plt.tight_layout(); p=os.path.(f"Reward vs. Inventory ({ENV_NAME_SHORT})"); plt.xlabel("Avg Ending Inventory"); plt.ylabel("Avg Reward"); plt.grid(True); plt.tight_layout(); p=os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_reward_vs_inventory.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    plt.close('all')


# --- Main Execution ---

if __name__ == "__main__":
    # --- Initialize Ray ---
    if RAY_AVAILABLE:
        try:
            # Start Ray. Adjust resources as needed.
            ray.init(ignore_reinit_error=True, num_cpus=Njoin(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_train_time.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    else: print("Skip train time plot.")
    # 4. Scatter Reward vs Service
    plt.figure(figsize=(9, 7)); sns.scatterplot(data=df_s, x="AvgServiceLevel", y="AvgReward", hue="Agent", s=100, palette="viridis", legend=False);
    for _, r in df_s.iterrows(): plt_WORKERS_RLLIB + 1) # +1 for driver
            print("Ray initialized successfully.")
        except Exception as e:
            print(f"Warning: Failed to initialize Ray: {e}. RLlib agents will not run.")
            RAY_AVAILABLE = False # Disable RLlib if init fails

    # --- Define Agents ---.text(r["AvgServiceLevel"] + 0.005, r["AvgReward"], r.name, fontsize=9)
    plt.title(f"Reward vs. Service ({ENV_NAME_SHORT})"); plt.xlabel("Avg Service Level"); plt.ylabel("Avg Reward"); plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0)); plt.grid(True); plt.tight_layout(); p=os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_reward_vs_service.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    # 5. Scatter Reward vs Inventory
    plt.figure(figsize=(9, 7)); sns.scatterplot(data=df_s, x="AvgEndInv", y="AvgReward", hue="Agent", s=100, palette="viridis
    print("Defining agents...")
    agent_objects = {}

    # Heuristic Agents
    agents_to_run_defs = [
        ("Random", RandomAgent, {}),
        ("OrderUpTo_SF=1.0", OrderUpToHeuristicAgent, {'safety_factor': 1.0}),
        ("OrderUpTo_SF=1.2", OrderUpToHeuristicAgent, {'safety_factor': 1.2}),
        ("OrderUpTo_SF=0.8", OrderUpToHeuristicAgent, {'safety_factor': 0.8}),
        ("ClassicNV_SF=1.0_k_vs_h", ClassicNewsvendorAgent, {'cr_method': 'k_vs_h',", legend=False);
    for _, r in df_s.iterrows(): plt.text(r["AvgEndInv"] * 1.01, r["AvgReward"], r.name, fontsize=9)
    plt.title(f"Reward vs. Inventory ({ENV_NAME_SHORT})"); plt.xlabel("Avg Ending Inv"); plt.ylabel("Avg Reward"); plt.grid(True); plt.tight_layout(); p=os.path.join(RESULTS_DIR, f"{ENV_NAME_SHORT}_benchmark_reward_vs_inventory.png"); plt.savefig(p); print(f"Saved: {p}"); plt.close()
    plt.close('all')

# --- Main Execution ---

if __name__ == "__main__":
    # --- Initialize Ray (Only if RLlib agents are used) ---
    if R 'safety_factor': 1.0}),
        ("sS_Policy_0.5_1.2s", sSPolicyAgent, {'s_quantile': 0.5, 'S_buffer_factor': 1.2}),
    ]

    # SB3 RL Agents
    if SB3_AVAILABLE:
        sb3_agents_def = [
            ("SB3_PPO", PPO, {}), ("SB3_SAC", SAC, {}), ("SB3_TD3", TD3, {}),
            ("SB3_A2C", A2C, {}), ("SB3_DDPG", DDPG, {}),
            # Add variations if desired
            ("SB3_PPO-LargeBuffer", PPO, {'model_kwargs': {'n_steps': 4096}}),
            ("SB3_SAC-LowLR", SAC, {'model_kwargs': {'learning_rate': 1e-4}}),
        ]
        for name, model_cls, params in sb3_agents_def:
             wrapper_params = {'model_class': model_cls, 'name': name}
             if 'policy' in params: wrapper_params['policy'] = params['policy']
             if 'model_kwargs' in params: wrapper_params['model_kwargs'] = params['model_kwargs']
             agents_to_run_defs.appendLLIB_AVAILABLE:
        try:
            if ray.is_initialized():
                 print("Ray already initialized.")
            else:
                 # Configure resources if needed (e.g., limit memory/CPU for Colab)
                 ray.init(ignore_reinit_error=True, num_cpus=N_WORKERS_RLLIB + 1) # +1 for driver
                 print(f"Ray initialized with {N_WORKERS_RLLIB + 1} CPUs.")
        except Exception as e_ray:
            print(f"Error initializing Ray: {e_ray}. RLlib agents might fail.")
            RLLIB_AVAILABLE = False # Disable RLlib if init fails

    # --- Define Agents ---
    print("Defining agents...")
    agent_objects = {}
    agents_to_run_defs((name, SB3AgentWrapper, wrapper_params))
    else: print("\nSkipping SB3 agent definitions.")

    # RLlib RL Agents
    if RAY_AVAILABLE:
         rllib_agents_def = [
             # Use algo name string identifiers known to RLlib
             # Pass config updates specific to RLlib's AlgorithmConfig structure
             ("RLlib_PPO", "PPO", {'lr': 5e-5}), # Example config update
             ("RLlib_SAC", "SAC", {'gamma': 0.98}), # Example config update
             # ("RLlib_TD3", "TD3", {}), # Can add more
             # ("RLlib_A2C", "A2C", {}),
             # ("RLlib_DDPG", "DDPG", {}),
         ]
         for name, algo_name_ = []

    # Heuristics for Newsvendor
    agents_to_run_defs.extend([
        ("Random", RandomAgent, {}),
        ("OrderUpTo_SF=1.0", OrderUpToHeuristicAgent, {'safety_factor': 1.0}),
        ("OrderUpTo_SF=1.2", OrderUpToHeuristicAgent, {'safety_factor': 1.2}),
        ("OrderUpTo_SF=0.8", OrderUpToHeuristicAgent, {'safety_factor': 0.8}),
        ("ClassicNV_SF=1.0_k_vs_h", ClassicNewsvendorAgent, {'cr_method': 'k_vs_h', 'safety_factor': 1.0}),
        ("sS_Policy_0.5_1.2s", sSPolicyAgentstr, cfg_updates in rllib_agents_def:
              agents_to_run_defs.append((name, RLlibAgentWrapper, {'algo_name': algo_name_str, 'config_updates': cfg_updates, 'name': name}))
    else: print("\nSkipping RLlib agent definitions.")


    # Instantiate agents
    print(f"\nInstantiating {len(agents_to_run_defs)} agents...")
    for name, agent_class, params in agents_to_run_defs:
         try:
             print(f"  Instantiating: {name}")
             agent_objects[name] =, {'s_quantile': 0.5, 'S_buffer_factor': 1.2}),
        ("sS_Policy_0.8_1.1s", sSPolicyAgent, {'s_quantile': 0.8, 'S_buffer_factor': 1.1}),
    ])

    # SB3 RL Agents
    if SB3_AVAILABLE:
        sb3_agents_def = [
            ("SB3_PPO", PPO_SB3, {}), ("SB3_SAC", SAC_SB3, {}), ("SB3_TD3", TD3_SB3, {}),
            ("SB3_A2C", A2C_SB3, {}), ("SB3_DDPG", DDPG_SB3, {}),
        ]
        for name, model_cls, params in sb3_agents_def:
             wrapper_params = {'model_class': model_cls, agent_class(**params)
         except Exception as e:
              print(f"ERROR Instantiating agent {name}: {e}")
              import traceback; traceback.print_exc()


    # --- Train RL Agents ---
    print("\n--- Training Phase ---")
    for name, agent in agent_objects.items():
        # Polymorphic call to train - works for both wrappers and heuristics
        agent.train(ENV_CONFIG, total_timesteps=RL_TRAINING_TIMESTEPS, save_path_prefix=f"{ENV_NAME_SHORT}_{name}_")


    # --- Run Evaluation ---
    print("\n--- 'name': name}; wrapper_params.update(params)
             agents_to_run_defs.append((name, SB3AgentWrapper, wrapper_params))
    else: print("\nSkipping SB3 agent definitions.")

    # RLlib RL Agents
    if RLLIB_AVAILABLE:
        rllib_agents_def = [
            ("PPO", {}), # RLlib PPO often needs specific tuning
            ("SAC", {}),
            ("TD3", {}), # Evaluation Phase ---")
    all_evaluation_results = []
    print(f"\n-- Evaluating on Standard Random Parameters ({ENV_NAME_SHORT}) --")
    for name, agent in agent_objects.items():
        if name not in agent_objects: continue # Skip if instantiation failed
        eval_results = evaluate_agent(agent, ENV_CONFIG, N_EVAL_EPISODES,
                                       seed TD3 might need exploration config adjustments
            ("A2C", {}), # A2C in_offset=SEED_OFFSET, collect_details=COLLECT_STEP_DETAILS)
        if 'summary' in eval_results and not eval_results['summary'].empty:
            all_evaluation_results.append(eval_results)
        else: print(f"Warning: Eval for {name} produced no results.")


    # --- Process RLlib might be APPO or IMPALA depending on version/config
            ("DDPG", {}), # APEX-DDPG is often preferred in RLlib for and Report Results ---
    final_summary, final_raw_summary = process_and_report_results(all_evaluation_results, agent_objects)

    # --- Generate Plots ---
    if final_summary is stability
            # Example: ("APEX_DDPG", {'num_workers': N_WORKERS_RLLIB}) # APEX benefits from workers
        ]
        for name, config_updates in rllib_agents_def: not None:
        # Get log dirs for both types of agents if they exist
        sb3_log_dirs = {name: agent.log_dir for name
             # Pass algo name string and config overrides
             agents_to_run_defs.append((f"RLlib_{name}", RLlibAgentWrapper, {'algo_name, agent in agent_objects.items() if isinstance(agent, SB3AgentWrapper)}
        rllib_agent_names = [name for name, agent in agent_objects.items() if isinstance(agent, RLlibAgentWrapper)]

        all_log_dirs = sb3_log_dirs # Start with SB3 logs
        # Add paths to custom RLlib logs
        for name in rllib_agent_names:
            all_log_dirs[name]': name, 'config_updates': config_updates}))
    else: print("\nSkipping RLlib agent definitions.")

    # Instantiate agents
    print(f"\nInstantiating {len(agents_to_run_defs)} agents...")
    for name, agent_class, params in agents_to_run_defs:
         try:
             print(f"  Instantiating: = LOG_DIR # Pass base log dir, function will find specific file

        if all_log_dirs:
             try:
                 print("\nPlotting learning curves...") {name}")
             agent_objects[name] = agent_class(**params)
         except Exception as e: print(f"ERROR Instantiating {name}: {e}"); import traceback; traceback.print_exc()

    # --- Train RL Agents
                 # Pass rllib agent names so function knows how to parse
                 plot_learning_curves(all_log_dirs, rllib_agent_names=rllib_agent_names, title=f"RL Learning Curves ({ENV_NAME_SHORT})")
             except Exception as e:
                 print(f"Error generating learning curve plot: {e}")
                  ---
    print("\n--- Training Phase ---")
    for name, agent in agent_objects.items():
        if isinstance(agent, (SB3AgentWrapper, RLlibAgentWrapper)): # Check both wrapper types
             # Pass unique prefix for saving models/logs per agent
             agent.train(ENV_CONFIG, total_timesteps=RL_TRAINING_TIMESTEPSimport traceback; traceback.print_exc()

        plot_benchmark_results(final_summary, final_raw_summary)
    else:
        print("Skipping plotting as no summary results were generated.")

    # --- Shutdown Ray ---
    if RAY_AVAILABLE, save_path_prefix=f"{ENV_NAME_SHORT}_{name}_")
        else: agent.train(ENV_CONFIG, 0, "") # Call dummy train for heuristics

    # --- Run Evaluation ---
    print("\n--- Evaluation Phase:
        print("Shutting down Ray...")
        ray.shutdown()

    print("\nBenchmark script finished.")
