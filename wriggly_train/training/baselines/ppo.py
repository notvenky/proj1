'''
In [10]: model = ARS.load("best_mod
    ...: el.zip")

In [11]: model
Out[11]: <sb3_contrib.ars.ars.ARS at 0x7fd55f6119f0>

In [12]: model.policy
Out[12]: 
CPGPolicy(
  (features_extractor): FlattenExtractor(
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (pi_features_extractor): FlattenExtractor(
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (vf_features_extractor): FlattenExtractor(
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (mlp_extractor): PassThroughObs(
    (value_net): Sequential(
      (0): Linear(in_features=13, out_features=69, bias=True)
      (1): ReLU()
    )
  )
  (action_net): MyActor()
  (value_net): Linear(in_features=69, out_features=1, bias=True)
)
'''


from wriggly_train.training.baselines import dmc2gym
import dmc2gymnasium
from stable_baselines3 import PPO
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.logger import configure, Logger, CSVOutputFormat, TensorBoardOutputFormat
from wriggly_train.envs.wriggly.robots import wriggly_from_swimmer
from wriggly_train.training import dmc
import os
from wriggly_train.policy.cpg_policy import make_cpg_policy, CPGPolicy
from wriggly_train.policy.mlp_cpgs import MLPDelta

current_date = datetime.now().strftime('%Y-%m-%d')
root_log_dir = "logs/ppo/"
daily_log_dir = os.path.join(root_log_dir, current_date)
run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
run_log_dir = os.path.join(daily_log_dir, run_id)

os.makedirs(run_log_dir, exist_ok=True)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.plot_data = {'x': [], 'y': []}
        plt.ion()

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)
                  # Plotting part
                  self.plot_data['x'].append(self.num_timesteps)
                  self.plot_data['y'].append(mean_reward)
                  plt.clf()
                  plt.plot(self.plot_data['x'], self.plot_data['y'])
                  plt.xlabel('Timesteps')
                  plt.ylabel('Mean reward')
                  plt.title('Training performance')
                  plt.draw()
                  plt.gcf().canvas.flush_events()

        return True

def make_env():
  #  env = dmc2gym.make(domain_name='wriggly', task_name='move_no_time', episode_length=1000)
  env = dmc2gymnasium.DMCGym('wriggly', 'approach_target', )
   # env = dmc2gym.make(domain_name='point_mass', task_name='easy', seed=1)
  return env


vec_env = make_vec_env(make_env, n_envs=32)
env = VecMonitor(vec_env, run_log_dir)
def make_basic_cpg(d_obs, d_act):
   return CPGPolicy(d_act)

def make_delta_cpg(d_obs, d_act):
   return MLPDelta(d_obs, d_act)

policy_class = make_cpg_policy(make_delta_cpg)
policy_kwargs = dict(log_std_init=0.5)
model = PPO(   
                policy_class,
                env,
                learning_rate=0.0001,
                batch_size=256,
                gamma=0.99,
                verbose=1,
                tensorboard_log=run_log_dir,
                device='cuda',
                policy_kwargs=policy_kwargs,
            )
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=run_log_dir)

print("------------- Start Learning -------------")

model.learn(
    total_timesteps=10000000,
    log_interval=1,
    tb_log_name='ppo',
    reset_num_timesteps=True,
    callback=callback,
    progress_bar=True
)

env.close()
plt.show