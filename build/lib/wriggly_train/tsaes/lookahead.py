import numpy as np


def linear(
  velocity_smooth: np.ndarray,
  velocity_max: float = 1.,
  scaling_max: float = 10.,
):
  velocity_norm = np.clip(np.abs(velocity_smooth) / velocity_max, 0, 1)
  return (scaling_max - 1) * velocity_norm + 1
