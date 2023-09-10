import sys
sys.path.append("/home/venky/proj1")


import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)



import os
import ray
import time
import torch
import gym
import dm2gym
# import dmc2gym
import collections
import neptune.new as neptune
import numpy as np
import typing as T

import wriggly_train.tsaes as tsaes


def str_to_bool(s):
  if isinstance(s, bool):
    return s
  elif s.lower() in ('true', 'yes', 't', 'y', '1'):
    return True
  elif s.lower() in ('false', 'no', 'f', 'n', '0'):
    return False
  else:
    raise ValueError(f'Cannot interpret as bool: {s}')


def space_to_box(space, dtype=np.float32):
  if isinstance(space, gym.spaces.dict.Dict):
    bounds = [(
      np.full(b.shape if len(b.shape) > 0 else (1,), b.low),
      np.full(b.shape if len(b.shape) > 0 else (1,), b.high)
    ) for b in space.values()]
    lows, highs = zip(*bounds)
    box = gym.spaces.Box(np.concatenate(lows), np.concatenate(highs), dtype=np.float32)
    return box
  assert isinstance(space, gym.spaces.box.Box)
  return gym.spaces.box.Box(space.low.astype(dtype), space.high.astype(dtype))


def flatten_observation(obs):
  if isinstance(obs, collections.OrderedDict):
    obs = [np.array([o]) if np.isscalar(o) else o.ravel() for o in obs.values()]
    return np.concatenate(obs, axis=0)
  return obs


@ray.remote(num_cpus=1)
class GymEnvironmentWorker(tsaes.Worker):
  def __init__(self, env_builder, model_builder, **kwargs):
    super().__init__(**kwargs)
    import gym
    import dm2gym
    # import dmc2gym

    self.env = env_builder()
    # Ensure observation and action spaces are flattened `Box`.
    self.env.observation_space = space_to_box(self.env.observation_space)
    self.env.action_space = space_to_box(self.env.action_space)

    self.model = model_builder()
    self.model.initialize(self.env.observation_space, self.env.action_space)

  def objective(self, params: np.array, task_seed: int) -> float:
    torch.nn.utils.vector_to_parameters(torch.from_numpy(params), self.model.parameters())
    self.env.seed(task_seed)
    obs, done = self.env.reset(), False
    episode_reward = 0.
    steps = 0
    with torch.no_grad():
      while steps < self.env._max_episode_steps:
        obs = flatten_observation(obs)
        action = self.model(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        obs, reward, done, _ = self.env.step(action)
        episode_reward += reward
        steps += 1
        if done: break
    return episode_reward


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()

  # OpenAIES args.
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--population_size', type=int, default=8)
  parser.add_argument('--population_top_best', type=int, default=8)
  parser.add_argument('--noise_generator', type=str, default='tsaes.noise.gaussian')
  parser.add_argument('--learning_rate', type=float, default=0.02)
  parser.add_argument('--momentum', type=float, default=0.)
  parser.add_argument('--optimizer', choices=('sgd', 'adam'), default='adam')
  parser.add_argument('--sigma', type=float, default=0.02)
  parser.add_argument('--weight_decay', type=float, default=0.)
  parser.add_argument('--reward_shaping', type=str, default='tsaes.shaping.centered_rank')
  parser.add_argument('--task_sync', type=str_to_bool, nargs='?', const=True, default=True)
  parser.add_argument('--num_workers', type=int, default=2)
  parser.add_argument('--test_freq', type=int, default=1)
  parser.add_argument('--test_size', type=int, default=5)
  parser.add_argument('--save_freq', type=int, default=50)
  parser.add_argument('--seed', type=int, default=0)

  # Environment and model args.
  parser.add_argument('--environment', type=str, required=True)
  parser.add_argument('--model', type=str, required=True)
  parser.add_argument('--header', type=str)
  parser.add_argument('--job_id', type=str, default='debug')

  args = parser.parse_args()

  header = args.header
  model = args.model
  environment = args.environment

  def env_builder():
    import torch
    import tonic
    import tonic.torch
    import gym
    import dm2gym
    if header: exec(header)
    return eval(environment)

  def model_builder():
    import torch
    import tonic
    import tonic.torch
    import wriggly_train.tsaes_model
    if header: exec(header)
    return eval(model)

  env = env_builder()
  env.observation_space = space_to_box(env.observation_space)
  env.action_space = space_to_box(env.action_space)

  mod = model_builder()
  mod.initialize(env.observation_space, env.action_space)

  params = torch.nn.utils.parameters_to_vector(mod.parameters()).detach().numpy()
  if args.optimizer.lower() == 'sgd':
    optimizer = tsaes.optimizers.SGD(
      learning_rate=args.learning_rate,
      momentum=args.momentum,
    )
  else:
    optimizer = tsaes.optimizers.Adam(learning_rate=args.learning_rate,)

#   logger = neptune.init(project='neuroai/openai-es')
  algo = tsaes.OpenAIES(
    GymEnvironmentWorker,
    dict(
      env_builder=env_builder,
      model_builder=model_builder,
    ),
    epochs=args.epochs,
    population_size=args.population_size,
    population_top_best=args.population_top_best,
    noise_generator=eval(args.noise_generator),
    optimizer=optimizer,
    sigma=args.sigma,
    weight_decay=args.weight_decay,
    reward_shaping=eval(args.reward_shaping),
    task_sync=args.task_sync,
    num_workers=args.num_workers,
    test_freq=args.test_freq,
    test_size=args.test_size,
    save_freq=args.save_freq,
    save_dir=os.path.join('logs', args.job_id),
    seed=args.seed,
    # logger=logger,
  )

#   logger['model'] = str(mod)
#   logger['hyperparams'] = vars(args)
  algo.run(params)
