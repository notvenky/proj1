import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import dmc2gymnasium


from dm_control import mujoco, suite
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree
import typing as T
from dm_control import suite, composer
from dm_control.utils import containers


from sb3_contrib import ARS
from sb3_contrib.common.vec_env import AsyncEval
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.logger import configure, Logger, CSVOutputFormat, TensorBoardOutputFormat
from wriggly_train.envs.wriggly.robots import wriggly_from_swimmer
from wriggly_train.training import dmc
import os
from wriggly_train.policy.cpg_policy import CPGPolicy, CPGPolicy

from torch.utils.tensorboard import SummaryWriter

my_actor = CPGPolicy(num_actuators=5)

current_date = datetime.now().strftime('%Y-%m-%d')
root_log_dir = "logs/ars/"
daily_log_dir = os.path.join(root_log_dir, current_date)
run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
run_log_dir = os.path.join(daily_log_dir, run_id)
# Create a summary writer object
writer = SummaryWriter(log_dir=run_log_dir)

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
    def __init__(self, check_freq: int, log_dir: str, my_actor: CPGPolicy, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.plot_data = {'x': [], 'y': []}
        self.my_actor = my_actor
        self.writer = SummaryWriter(log_dir=log_dir)
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

                  self.my_actor.log_params(self.writer, self.num_timesteps)

        return True
    
    def _on_training_end(self) -> None:
        plt.savefig(os.path.join(self.log_dir, 'training_performance.png'))
        writer.close()

def make_env():
   env = dmc2gymnasium.DMCGym('wriggly', 'approach_target', )
   # env = dmc2gym.make(domain_nmame='point_mass', task_name='easy', seed=1)
   return env


vec_env = make_vec_env(make_env, n_envs=32)
env = VecMonitor(vec_env, run_log_dir)

policy_kwargs = dict(log_std_init=0.5)
model = ARS(   
                CPGPolicy,
                env,
                learning_rate=0.02,
                n_top = 16,
                delta_std=0.5,
                # policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=run_log_dir,
                device="cuda",
                zero_policy=False,
            )
n_envs = 32
# async_eval = AsyncEval([lambda: make_vec_env(env) for _ in range(n_envs)], model.policy)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=run_log_dir, my_actor=my_actor)

print("------------- Start Learning -------------")

model.learn(
    total_timesteps=10000000,
    log_interval=1,
    tb_log_name='ars',
    reset_num_timesteps=True,
    callback=callback,
    progress_bar=True, 
    # async_eval=async_eval,
)

env.close()
plt.show