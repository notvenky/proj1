import numpy as np


def centered_rank(rewards: np.ndarray, mag=0.5):
  ranks = np.empty(rewards.size, dtype=int)
  ranks[rewards.ravel().argsort()] = np.arange(rewards.size)
  centered_values = np.linspace(-mag, mag, rewards.size)
  return centered_values[ranks].reshape(rewards.shape)