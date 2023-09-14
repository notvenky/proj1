import os
import ray
import collections
import time
import traceback
import numpy as np
import typing as T
from tqdm import tqdm

import wriggly_train.tsaes as tsaes


def makefile(*subpaths, file: str = None):
  path = os.path.join(*subpaths)
  os.makedirs(path, exist_ok=True)
  if file is not None:
    path = os.path.join(path, file)
  return path


class Worker:
  def __init__(
    self,
    noise: np.ndarray,
    **kwargs,
  ):
    self.noise = tsaes.noise.NoiseTable(noise)
    self.mode = None

  def train(
    self,
    params: np.ndarray,
    lookahead: np.ndarray,
    noise_seed: int,
    task_seed: T.Union[int, T.Tuple[int, int]],
  ) -> T.Tuple[int, float, float]:
    try:
      self.mode = 'train'
      idx, epsilon = self.noise.sample(params.size, noise_seed)
      if isinstance(task_seed, int): task_seed = (task_seed, task_seed)
      task_seed_pos, task_seed_neg = task_seed
      reward_pos = self.objective(params + lookahead * epsilon, task_seed_pos)
      reward_neg = self.objective(params - lookahead * epsilon, task_seed_neg)
      return idx, reward_pos, reward_neg
    except:
      traceback.print_exc()
      return None

  def test(
    self,
    params: np.ndarray,
    task_seed: int,
  ) -> float:
    try:
      self.mode = 'test'
      return self.objective(params, task_seed)
    except:
      traceback.print_exc()
      return None

  def objective(self, params: np.ndarray, task_seed: int) -> float:
    raise NotImplementedError


class TSAES:
  def __init__(
    self,
    worker_cls: T.Type,
    worker_kwargs: T.Dict[str, T.Any] = {},
    ray_kwargs: T.Dict[str, T.Any] = {},
    epochs: int = 1000,
    population_size: int = 8,
    population_top_best: int = 8,
    noise_generator: T.Callable[[], np.ndarray] = tsaes.noise.gaussian,
    learning_rate: float = 0.02,
    momentum: float = 0.,
    velocity_smoothing: float = 0.9,
    lookahead_scaling: T.Union[float, str] = 1.,
    lookahead_scaling_fn: T.Callable[[np.ndarray], np.ndarray] = tsaes.lookahead.linear,
    exploration_bias: float = 0.,
    task_sync: bool = True,
    num_workers: int = 1,
    test_freq: int = 1,
    test_size: int = 1,
    save_freq: int = 50,
    save_dir: str = None,
    seed: int = 0,
    # logger=None,  # TODO: Use default logger.
    callbacks: T.Mapping[str, T.Callable] = {},
  ):
    self.epochs = epochs
    self.population_size = population_size
    self.population_top_best = population_top_best
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.velocity_smoothing = velocity_smoothing
    self.lookahead_scaling = lookahead_scaling
    self.lookahead_scaling_fn = lookahead_scaling_fn
    self.exploration_bias = exploration_bias
    self.task_sync = task_sync
    self.num_workers = num_workers
    self.test_freq = test_freq
    self.test_size = test_size
    self.save_freq = save_freq
    self.save_dir = os.path.abspath(
      save_dir or os.path.join('logs', time.strftime("%Y-%m-%d-%H-%M-%S"))
    )
    os.makedirs(self.save_dir, exist_ok=True)
    # self.logger = logger
    self.seed = seed
    self.callbacks = collections.defaultdict(lambda: lambda **kwargs: None, **callbacks)

    ray.init(**ray_kwargs)

    # Create the shared noise table.
    np.random.seed(self.seed)
    self.noise_ref = ray.put(noise_generator(seed=self.seed))
    self.noise = tsaes.noise.NoiseTable(ray.get(self.noise_ref))

    # Initialize the workers.
    self.workers = [
      worker_cls.remote(noise=self.noise_ref, **worker_kwargs) for _ in range(self.num_workers)
    ]
    self.pool = ray.util.ActorPool(self.workers)

  def __del__(self):
    ray.shutdown

  def run(self, params: np.ndarray):
    velocity = np.zeros_like(params)
    if self.lookahead_scaling == 'adaptive':
      velocity_smooth = np.zeros_like(velocity)

    for epoch_id in tqdm(range(self.epochs), desc='epoch'):
      try:
        # ==========================================================================================
        # Train phase.
        train_time = time.time()
        self.callbacks['epoch_train_start'](algo=self, epoch=epoch_id, params=params)

        # Manager: calculate lookahead, share params.
        if self.lookahead_scaling == 'adaptive':
          velocity_smooth = (
            1 - self.velocity_smoothing
          ) * velocity + self.velocity_smoothing * velocity_smooth
          lookahead = self.learning_rate * self.lookahead_scaling_fn(
            velocity_smooth, velocity_max=self.learning_rate
          )
        else:
          lookahead = self.learning_rate * self.lookahead_scaling

        params_ref = ray.put(params)
        lookahead_ref = ray.put(lookahead)

        # Workers (train): sample noise, calculate objective, send results.
        results_train = self.pool.map(
          lambda worker,
          population_id: worker.train.remote(
            params_ref,
            lookahead_ref,
            noise_seed=epoch_id * 10_000 + population_id,
            task_seed=epoch_id * 10_000 + self.seed
            if self.task_sync else (np.random.randint(1_000_000), np.random.randint(1_000_000)),
          ),
          range(self.population_size)
        )
        results_train = list(results_train)  # [(idx, reward_pos, reward_neg)]
        results_train = [r for r in results_train if r is not None]
        results_train.sort(key=lambda x: max(x[1], x[2]), reverse=True)
        results_train = results_train[:self.population_top_best]
        idxs, reward_pos, reward_neg = zip(*results_train)

        # Manager: calculate std of all rewards, aggregate direction, algorithm quantities.
        aggregate = 0
        for idx, r_pos, r_neg in results_train:
          epsilon = self.noise.get(params.size, idx)
          aggregate = aggregate + (r_pos - r_neg + self.exploration_bias) * epsilon
        rewards_train = reward_pos + reward_neg
        rewards_std = np.std(rewards_train) + 1e-6
        aggregate = aggregate / (self.population_top_best * rewards_std)
        velocity = self.learning_rate * aggregate + self.momentum * velocity
        params = params + velocity

        self.callbacks['epoch_train_end'](algo=self, epoch=epoch_id, params=params)

        self.logger['train/epoch'].log(epoch_id + 1)
        self.logger['train/objective_evals'].log(2 * self.population_size * (epoch_id + 1))
        self.logger['train/seconds'].log(time.time() - train_time)
        self.logger['train/reward/mean'].log(np.mean(rewards_train))
        self.logger['train/reward/min'].log(np.min(rewards_train))
        self.logger['train/reward/max'].log(np.max(rewards_train))
        self.logger['train/reward/std'].log(np.std(rewards_train))

        # ==========================================================================================
        # Test phase.
        # Workers (test): calculate objective, send results.
        if self.test_size > 0 and self.test_freq > 0 and epoch_id % self.test_freq == 0 or epoch_id == self.epochs - 1:
          test_time = time.time()
          self.callbacks['epoch_test_start'](algo=self, epoch=epoch_id, params=params)

          results_test = self.pool.map(
            lambda worker,
            _: worker.test.remote(
              params_ref,
              task_seed=np.random.randint(1_000_000),
            ),
            range(self.test_size)
          )
          results_test = list(results_test)  # [reward]
          rewards_test = [r for r in results_test if r is not None]

          self.callbacks['epoch_test_end'](algo=self, epoch=epoch_id, params=params)

          self.logger['test/epoch'].log(epoch_id + 1)
          self.logger['test/objective_evals'].log(self.test_size * (epoch_id // self.test_freq + 1))
          self.logger['test/seconds'].log(time.time() - test_time)
          if len(rewards_test) > 0:
            self.logger['test/reward/mean'].log(np.mean(rewards_test))
            self.logger['test/reward/min'].log(np.min(rewards_test))
            self.logger['test/reward/max'].log(np.max(rewards_test))
            self.logger['test/reward/std'].log(np.std(rewards_test))
          np.save(makefile(self.save_dir, 'rewards_test', file=f'{epoch_id + 1}'), rewards_test)

        # ==========================================================================================
        # Save phase.
        if epoch_id % self.save_freq == 0 or epoch_id == self.epochs - 1:
          np.save(makefile(self.save_dir, 'params', file=f'{epoch_id + 1}'), params)
          np.save(makefile(self.save_dir, 'velocity', file=f'{epoch_id + 1}'), velocity)
          np.save(makefile(self.save_dir, 'aggregate', file=f'{epoch_id + 1}'), aggregate)
          np.save(makefile(self.save_dir, 'results_train', file=f'{epoch_id + 1}'), results_train)
          if self.lookahead_scaling == 'adaptive':
            np.save(
              makefile(self.save_dir, 'velocity_smooth', file=f'{epoch_id + 1}'), velocity_smooth
            )
            np.save(makefile(self.save_dir, 'lookahead', file=f'{epoch_id + 1}'), lookahead)
      except:
        traceback.print_exc()
        print(f'Error occured on epoch {epoch_id}.')
        np.save(makefile(self.save_dir, 'params', file=f'{epoch_id + 1}'), params)
        np.save(makefile(self.save_dir, 'velocity', file=f'{epoch_id + 1}'), velocity)
        np.save(makefile(self.save_dir, 'aggregate', file=f'{epoch_id + 1}'), aggregate)
        np.save(makefile(self.save_dir, 'results_train', file=f'{epoch_id + 1}'), results_train)
        if self.lookahead_scaling == 'adaptive':
          np.save(
            makefile(self.save_dir, 'velocity_smooth', file=f'{epoch_id + 1}'), velocity_smooth
          )
          np.save(makefile(self.save_dir, 'lookahead', file=f'{epoch_id + 1}'), lookahead)
        return None
    return params


class OpenAIES:
  def __init__(
    self,
    worker_cls: T.Type,
    worker_kwargs: T.Dict[str, T.Any] = {},
    ray_kwargs: T.Dict[str, T.Any] = {},
    epochs: int = 1000,
    population_size: int = 8,
    population_top_best: T.Optional[int] = None,
    noise_generator: T.Callable[[], np.ndarray] = tsaes.noise.gaussian,
    optimizer: T.Optional[tsaes.optimizers.Optimizer] = None,
    sigma: float = 0.02,
    weight_decay: float = 0.,
    reward_shaping: T.Callable[[np.ndarray], np.ndarray] = tsaes.shaping.centered_rank,
    task_sync: bool = False,
    num_workers: int = 1,
    test_freq: int = 1,
    test_size: int = 1,
    save_freq: int = 50,
    save_dir: str = None,
    seed: int = 0,
    # logger=None,  # TODO: Use default logger.
  ):
    self.epochs = epochs
    self.population_size = population_size
    self.population_top_best = population_top_best
    self.optimizer = optimizer or tsaes.optimizers.SGD()
    self.sigma = sigma
    self.weight_decay = weight_decay
    self.reward_shaping = reward_shaping
    self.task_sync = task_sync
    self.num_workers = num_workers
    self.test_freq = test_freq
    self.test_size = test_size
    self.save_freq = save_freq
    self.save_dir = os.path.abspath(
      save_dir or os.path.join('logs', time.strftime("%Y-%m-%d-%H-%M-%S"))
    )
    os.makedirs(self.save_dir, exist_ok=True)
    # self.logger = logger
    self.seed = seed

    ray.init(**ray_kwargs)

    # Create the shared noise table.
    np.random.seed(self.seed)
    self.noise_ref = ray.put(noise_generator(seed=self.seed))
    self.noise = tsaes.noise.NoiseTable(ray.get(self.noise_ref))

    # Initialize the workers.
    self.workers = ray.util.ActorPool([
      worker_cls.remote(noise=self.noise_ref, **worker_kwargs) for _ in range(self.num_workers)
    ])

  def __del__(self):
    ray.shutdown()

  def run(self, params: np.ndarray):
    self.optimizer.initialize(params)
    for epoch_id in tqdm(range(self.epochs), desc='epoch'):
      try:
        # ==========================================================================================
        # Train phase.
        train_time = time.time()

        # Manager: share params, use fixed sigma lookahead.
        params_ref = ray.put(params)

        # Workers (train): sample noise, calculate objective, send results.
        results_train = self.workers.map(
          lambda worker,
          population_id: worker.train.remote(
            params_ref,
            self.sigma,
            noise_seed=epoch_id * 10_000 + population_id,
            task_seed=epoch_id * 10_000 + self.seed
            if self.task_sync else (np.random.randint(1_000_000), np.random.randint(1_000_000)),
          ),
          range(self.population_size)
        )
        results_train = list(results_train)  # [(idx, reward_pos, reward_neg)]
        results_train = [r for r in results_train if r is not None]
        results_train.sort(key=lambda x: max(x[1], x[2]), reverse=True)
        results_train = results_train[:(self.population_top_best or self.population_size)]
        idxs, reward_pos, reward_neg = zip(*results_train)
        rewards_train = reward_pos + reward_neg
        rewards_shaped = self.reward_shaping(np.vstack((reward_pos, reward_neg)))  # (2, pop size)

        # Manager: calculate std of all rewards, aggregate direction, algorithm quantities.
        aggregate = 0
        for i, idx in enumerate(idxs):
          r_pos = rewards_shaped[0, i]
          r_neg = rewards_shaped[1, i]
          epsilon = self.noise.get(params.size, idx)
          aggregate = aggregate + (r_pos - r_neg) * epsilon
        aggregate = aggregate / (len(idxs) * self.sigma)
        params = params + self.optimizer.step(aggregate) - self.weight_decay * params

        self.logger['train/epoch'].log(epoch_id + 1)
        self.logger['train/objective_evals'].log(2 * self.population_size * (epoch_id + 1))
        self.logger['train/seconds'].log(time.time() - train_time)
        self.logger['train/reward/mean'].log(np.mean(rewards_train))
        self.logger['train/reward/min'].log(np.min(rewards_train))
        self.logger['train/reward/max'].log(np.max(rewards_train))
        self.logger['train/reward/std'].log(np.std(rewards_train))

        # ==========================================================================================
        # Test phase.
        # Workers (test): calculate objective, send results.
        if self.test_size > 0 and self.test_freq > 0 and epoch_id % self.test_freq == 0 or epoch_id == self.epochs - 1:
          test_time = time.time()
          results_test = self.workers.map(
            lambda worker,
            _: worker.test.remote(
              params_ref,
              task_seed=np.random.randint(1_000_000),
            ),
            range(self.test_size)
          )
          results_test = list(results_test)  # [reward]
          rewards_test = [r for r in results_test if r is not None]
          self.logger['test/epoch'].log(epoch_id + 1)
          self.logger['test/objective_evals'].log(self.test_size * (epoch_id // self.test_freq + 1))
          self.logger['test/seconds'].log(time.time() - test_time)
          if len(rewards_test) > 0:
            self.logger['test/reward/mean'].log(np.mean(rewards_test))
            self.logger['test/reward/min'].log(np.min(rewards_test))
            self.logger['test/reward/max'].log(np.max(rewards_test))
            self.logger['test/reward/std'].log(np.std(rewards_test))
          np.save(makefile(self.save_dir, 'rewards_test', file=f'{epoch_id + 1}'), rewards_test)

        # ==========================================================================================
        # Save phase.
        if epoch_id % self.save_freq == 0 or epoch_id == self.epochs - 1:
          np.save(makefile(self.save_dir, 'params', file=f'{epoch_id + 1}'), params)
          np.save(makefile(self.save_dir, 'aggregate', file=f'{epoch_id + 1}'), aggregate)
          np.save(makefile(self.save_dir, 'results_train', file=f'{epoch_id + 1}'), results_train)
      except:
        traceback.print_exc()
        print(f'Error occured on epoch {epoch_id}.')
        np.save(makefile(self.save_dir, 'params', file=f'{epoch_id + 1}'), params)
        np.save(makefile(self.save_dir, 'aggregate', file=f'{epoch_id + 1}'), aggregate)
        np.save(makefile(self.save_dir, 'results_train', file=f'{epoch_id + 1}'), results_train)
        return None
    return params
