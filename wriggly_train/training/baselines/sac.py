from wriggly_train.training.baselines import dmc2gym
from stable_baselines3 import SAC
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
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

current_date = datetime.now().strftime('%Y-%m-%d')
root_log_dir = "logs/sac/"
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


env = dmc2gym.make(domain_name='wriggly', task_name='move', episode_length=5000)
# vec_env = DummyVecEnv([lambda: env], render_mode="human")
env = Monitor(env, run_log_dir)
# done = False
# obs = env.reset()
# while not done:
#     action = env.action_space.sample()
#     obs, reward, done, trunk, info = env.step(action)

model = SAC(   
                "MlpPolicy",
                env,
                learning_rate=0.0003,
                batch_size=256,
                buffer_size=1000000,
                tau=0.005,
                gamma=0.99,
                verbose=1,
                tensorboard_log=run_log_dir,
                device='cuda'
            )
callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=run_log_dir)

print("------------- Start Learning -------------")

model.learn(
    total_timesteps=1000000,
    log_interval=1,
    tb_log_name='sac',
    reset_num_timesteps=True,
    callback=callback,
    progress_bar=True
)

env.close()
plt.show