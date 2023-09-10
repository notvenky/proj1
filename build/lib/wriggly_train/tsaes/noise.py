import numpy as np
import typing as T


def gaussian(size=25_000_000, seed=0, dtype=np.float32):
  return np.random.RandomState(seed).randn(size).astype(dtype)


def bernoulli(size=25_000_000, seed=0, dtype=np.float32):
  return np.random.RandomState(seed).binomial(1, 0.5, size).astype(dtype) * 2 - 1


def uniform(size=25_000_000, seed=0, dtype=np.float32):
  return np.random.RandomState(seed).uniform(-1, 1, size).astype(dtype)


class NoiseTable:
  def __init__(self, noise: np.ndarray):
    self.noise = noise

  def get(self, size: int, idx: int):
    return self.noise[idx:idx + size]

  def sample(self, size: int, seed: int):
    idx = np.random.RandomState(seed).randint(0, len(self.noise) - size + 1)
    return idx, self.noise[idx:idx + size]