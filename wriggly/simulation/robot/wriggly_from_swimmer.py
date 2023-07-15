"""Procedurally generated wriggly domain."""
import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree
import numpy as np

_DEFAULT_TIME_LIMIT = 30
_CONTROL_TIMESTEP = .03  # (Seconds)

SUITE = containers.TaggedTasks()


# @SUITE.add('benchmarking')
# def wriggly6(time_limit=_DEFAULT_TIME_LIMIT, random=None,
#              environment_kwargs=None):
#   """Returns a 6-link wriggly."""
#   return _make_wriggly(6, time_limit, random=random,
#                        environment_kwargs=environment_kwargs)


# @SUITE.add('benchmarking')
# def wriggly15(time_limit=_DEFAULT_TIME_LIMIT, random=None,
#               environment_kwargs=None):
#   """Returns a 15-link wriggly."""
#   return _make_wriggly(15, time_limit, random=random,
#                        environment_kwargs=environment_kwargs)


# def wriggly(n_links=3, time_limit=_DEFAULT_TIME_LIMIT,
#             random=None, environment_kwargs=None):
#   """Returns a wriggly with n links."""
#   return _make_wriggly(n_links, time_limit, random=random,
#                        environment_kwargs=environment_kwargs)


# def _make_wriggly(n_joints, time_limit=_DEFAULT_TIME_LIMIT, random=None,
#                   environment_kwargs=None):
#   """Returns a wriggly control environment."""
#   model_string, assets = get_model_and_assets(n_joints)
#   physics = Physics.from_xml_string(model_string, assets=assets)
#   task = wriggly(random=random)
#   environment_kwargs = environment_kwargs or {}
#   return control.Environment(
#       physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
#       **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the wriggly domain."""

  def nose_to_target(self):
    """Returns a vector from nose to target in local coordinate of the head."""
    nose_to_target = (self.named.data.geom_xpos['target'] -
                      self.named.data.geom_xpos['green_leg'])
    head_orientation = self.named.data.xmat['green_leg'].reshape(3, 3)
    return nose_to_target.dot(head_orientation)[:2]

  def nose_to_target_dist(self):
    """Returns the distance from the nose to the target."""
    return np.linalg.norm(self.nose_to_target())

  def body_velocities(self):
    """Returns local body velocities: x,y linear, z rotational."""
    xvel_local = self.data.sensordata[12:].reshape((-1, 6))
    vx_vy_wz = [0, 1, 5]  # Indices for linear x,y vels and rotational z vel.
    return xvel_local[:, vx_vy_wz].ravel()

  def joints(self):
    """Returns all internal joint angles (excluding root joints)."""
    return self.data.qpos[3:].copy()


class Wriggly(base.Task):
  """A wriggly `Task` to reach the target or just swim."""

  def __init__(self, random=None):
    """Initializes an instance of `wriggly`.
    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.
    Initializes the wriggly orientation to [-pi, pi) and the relative joint
    angle of each joint uniformly within its range.
    Args:
      physics: An instance of `Physics`.
    """
    # Random joint angles:
    # randomizers.randomize_limited_and_rotational_joints(physics, self.random)
    physics.named.data.qpos[7:] = 0.0

    # Random target position.
    #close_target = self.random.rand() < .2  # Probability of a close target.
    target_box = 1
    xpos, ypos = self.random.uniform(-target_box, target_box, size=2)
    physics.named.data.qpos[0] = xpos
    physics.named.data.qpos[1] = ypos
    physics.named.data.qpos[2] = 0.027
    #physics.step()


    physics.named.model.geom_pos['target', 'x'] = xpos
    physics.named.model.geom_pos['target', 'y'] = ypos
    #print(physics.named.data.qpos)
    
    #physics.named.model.light_pos['target_light', 'x'] = xpos
    #physics.named.model.light_pos['target_light', 'y'] = ypos

    super().initialize_episode(physics)
    #print(physics.named.data.qpos)

  def get_observation(self, physics):
    """Returns an observation of joint angles, body velocities and target."""
    obs = collections.OrderedDict()
    obs['joints'] = physics.joints()
    #obs['to_target'] = physics.nose_to_target()
    obs['jointvel'] = physics.body_velocities()

    obs['time'] = np.array(physics.data.time)
    return obs

  def get_reward(self, physics):
    current_xy = physics.named.data.qpos[0:2]
    start_x = physics.named.model.geom_pos['target', 'x']
    start_y = physics.named.model.geom_pos['target', 'y']
    start_xy = np.array([start_x, start_y])
    
    return np.linalg.norm(start_xy - current_xy)
  




# from datetime import datetime
# import csv
# import heapq

# # Logger class to handle logging tasks
# class Logger:
#   def __init__(self, path, top_k=5):
#     self.path = path
#     self.top_k = top_k
#     self.data = []

#   def add(self, reward, freq, amp, phase):
#     # Use a negative reward for max heap
#     heapq.heappush(self.data, (-reward, freq, amp, phase))
#     if len(self.data) > self.top_k:
#       heapq.heappop(self.data)

#   def write(self):
#     # Use datetime to distinguish each run
#     with open(f"{self.path}/rewards_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv", 'w') as f:
#       writer = csv.writer(f)
#       writer.writerow(["reward", "frequency", "amplitude", "phase"])
#       self.data.sort(reverse=True)
#       for reward, freq, amp, phase in self.data:
#         writer.writerow([-reward, freq, amp, phase])  # Convert reward back to positive

# # ...

# # Initialize logger
# logger = Logger("/home/venky/proj1/wriggly/log", top_k=5)

# for i in tqdm(range(num_params)):
#   frequencies = torch.rand(num_actuators)  # softplus/exp/
#   amplitudes = torch.rand(num_actuators)   # tanh activation
#   phases = torch.rand(num_actuators)
#   actor = MyActor(frequencies, amplitudes, phases, num_actuators)
  
#   # Store frequencies, amplitudes and phases
#   all_frequencies[i] = frequencies.numpy()
#   all_amplitudes[i] = amplitudes.numpy()
#   all_phases[i] = phases.numpy()

#   reward = evaluate(env, actor, runs_per_act, 2000)
#   all_rewards[i] = reward 

#   # Print frequencies, amplitudes, phases and rewards for each run
#   for run in range(runs_per_act):
#     print(f"Sample {i}, Run {run}: Frequencies {frequencies}, Amplitudes {amplitudes}, Phases {phases}, Reward {reward[run]}")
    
#     # Add reward to logger
#     logger.add(reward[run], frequencies, amplitudes, phases)

# # Write to log file at the end of each run
# logger.write()