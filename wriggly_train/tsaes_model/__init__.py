import torch
import tonic
import tonic.torch
import numpy as np


class ActorOnly(torch.nn.Module):
  def __init__(self, actor, observation_normalizer=None):
    super().__init__()
    self.actor = actor
    self.observation_normalizer = observation_normalizer

  def initialize(self, observation_space, action_space):
    if self.observation_normalizer:
      self.observation_normalizer.initialize(observation_space.shape)
    self.actor.initialize(observation_space, action_space, self.observation_normalizer)

  def forward(self, *inputs):
    return self.actor(*inputs)


class Actor(torch.nn.Module):
  def __init__(self, encoder, torso, head):
    super().__init__()
    self.encoder = encoder
    self.torso = torso
    self.head = head

  def initialize(self, observation_space, action_space, observation_normalizer=None):
    action_size = action_space.shape[0]
    size = self.encoder.initialize(observation_space, action_space, observation_normalizer)
    if self.torso is not None:
      size = self.torso.initialize(size)
    self.head.initialize(size, action_size)

  def forward(self, *inputs):
    out = self.encoder(*inputs)
    if self.torso is not None:
      out = self.torso(out)
    return self.head(out)

class ObservationEncoder(tonic.torch.models.encoders.ObservationEncoder):
  def forward(self, observations):
    if self.observation_normalizer:
      self.observation_normalizer.record([observations.numpy()])
      observations = self.observation_normalizer(observations)
    return observations


class MeanStd(tonic.torch.normalizers.MeanStd):
  def get_stats(self):
    return (self.new_sum, self.new_sum_sq, self.new_count)

  def record_stats(self, stats):
    sum, sum_sq, count = stats
    self.new_sum += sum
    self.new_sum_sq += sum_sq
    self.new_count += count

  def get_state(self):
    return (self.mean, self.mean_sq, self.count)

  def reset_state(self, state):
    self.mean, self.mean_sq, self.count = state
    self.std = self._compute_std(self.mean, self.mean_sq)
    self.new_sum = 0
    self.new_sum_sq = 0
    self.new_count = 0
    self._update(self.mean.astype(np.float32), self.std.astype(np.float32))


def init_linear_zeros_(m):
  if isinstance(m, torch.nn.Linear):
    torch.nn.init.zeros_(m.weight)
    torch.nn.init.zeros_(m.bias)
