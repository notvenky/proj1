import collections
import os
import time
from dm_control import mujoco, suite
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree
import numpy as np
import math
import random
import typing as T
from dm_control import suite, composer
from dm_control.utils import containers
from wriggly_train.training.baselines import dmc2gym as dmc2gym

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__dict__ = self

_DEFAULT_TIME_LIMIT = 20  # (Seconds)
# _DEFAULT_TIME_LIMIT = 0.1
_CONTROL_TIMESTEP = .02  # (Seconds)
TARGET_MOVE_SPEED = 1.0



SUITE = containers.TaggedTasks()
suite._DOMAINS["wriggly"] = AttrDict(SUITE=SUITE)

# @SUITE.add()
# def move_slow(
#   time_limit=_DEFAULT_TIME_LIMIT,
#   random=None,
#   robot_kwargs={},
#   environment_kwargs={},
#   flatten=True,
#   **task_kwargs,
# ):
#   xml_path = "/home/venky/proj1/wriggly_train/envs/wriggly_mujoco/wriggly_apr_target.xml"
#   # wriggly =  mj.MjModel.from_xml_path(xml_path)
#   physics = Physics.from_xml_path(xml_path)
#   robot = Wriggly(**robot_kwargs)
#   task_kwargs.setdefault("jointvel", TARGET_SPEED_WALK)
#   task = Physics(robot, **task_kwargs)
#   env = composer.Environment(
#     task,
#     time_limit=time_limit,
#     random_state=random,
#     **environment_kwargs,
#   )
#   env._step_limit = time_limit // task.control_timestep
#   return env

@SUITE.add('benchmarking') # MAX_DISP
def move(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  xml_relative_path = "../../wriggly_mujoco/wriggly_max_disp.xml"
  xml_path = os.path.join(current_dir, xml_relative_path)
  physics = Physics.from_xml_path(xml_path)
  task = Wriggly()
  env = control.Environment(physics, task, time_limit=time_limit,
                            legacy_step=False, n_sub_steps=2)
  return env

@SUITE.add('benchmarking') # MAX_DISP
def move_no_time(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  xml_relative_path = "../../wriggly_mujoco/wriggly_max_disp.xml"
  xml_path = os.path.join(current_dir, xml_relative_path)
  physics = Physics.from_xml_path(xml_path)
  task = Wriggly(include_time=False)
  env = control.Environment(physics, task, time_limit=time_limit,
                            legacy_step=False, n_sub_steps=2)
  return env


@SUITE.add('benchmarking')  
def approach_target(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  xml_relative_path = "../../wriggly_mujoco/wriggly_apr_target.xml"
  xml_path = os.path.join(current_dir, xml_relative_path)
  physics = Physics.from_xml_path(xml_path)
  task = WrigglyApproachTarget()
  env = control.Environment(physics, task, time_limit=time_limit,
                            legacy_step=False, n_sub_steps=2)
  
  return env
  
@SUITE.add('benchmarking')
def climb_obstacle(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  xml_relative_path = "../../wriggly_mujoco/wriggly_climb_obstacle.xml"
  xml_path = os.path.join(current_dir, xml_relative_path)
  physics = Physics.from_xml_path(xml_path)
  task = WrigglyClimbObstacle()
  env = control.Environment(physics, task, time_limit=time_limit,
                            legacy_step=False, n_sub_steps=2)
  
  return env

@SUITE.add('benchmarking')
def maxvel(time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  xml_relative_path = "../../wriggly_mujoco/wriggly_max_disp.xml"
  xml_path = os.path.join(current_dir, xml_relative_path)
  physics = Physics.from_xml_path(xml_path)
  task = WrigglyMaxVel()
  env = control.Environment(physics, task, time_limit=time_limit,
                            legacy_step=False, n_sub_steps=2)
  
  return env


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
    """Returns a vector from nose to goal spot in local coordinate of the head."""
    nose_to_target = (self.named.data.geom_xpos['goal_spot'] -
                      self.named.data.geom_xpos['green_leg'])
    head_orientation = self.named.data.xmat['green_leg'].reshape(3, 3)
    return nose_to_target.dot(head_orientation)[:2]

  def nose_to_target_dist(self):
    """Returns the distance from the nose to the target."""
    return np.linalg.norm(self.nose_to_target())

  def body_velocities(self):
    """Returns local body velocities: x,y linear, z rotational."""
    xvel_local = self.data.sensordata[13:].reshape((-1, 6))
    vx_vy_wz = [0, 1, 5]  # Indices for linear x,y vels and rotational z vel.
    return xvel_local[:, vx_vy_wz].ravel()

  def joints(self):
    """Returns all internal joint angles (excluding root joints)."""
    return self.data.qpos[3:].copy()
  
  def speed(self):
    """Returns the horizontal speed of Wriggly."""
    return self.named.data.sensordata['central_velsensor'][0]
  
  def desired_direction(self):
    """Returns the desired direction of Wriggly."""
    return np.array([1, 0, 0])
  
  def current_position(self):
    """Returns the current position of Wriggly's center part."""
    return np.abs(self.named.data.sensordata['ACT2_pos_sensor'][0])
   
  def sum_speed(self):
    """Returns the sum of speeds of all parts of Wriggly."""
    return 0.2 * (self.named.data.sensordata['ACT0_velocity_sensor'][0] + self.named.data.sensordata['ACT1_velocity_sensor'][0] + self.named.data.sensordata['ACT2_velocity_sensor'][0] + self.named.data.sensordata['ACT3_velocity_sensor'][0] + self.named.data.sensordata['ACT4_velocity_sensor'][0])

class Wriggly(base.Task):
  start_time = time.time()
  # prev_xy, prev_x_diff, prev_rotation, start_orientation, start_xy, start_x1, start_y1, start_x2, start_y2, start_x3, start_y3, start_x4, start_y4, start_x5, start_y5, start_x6, start_y6 = [None] * 18

  
  
  """Wriggly task to reach the target or just swim"""

  def __init__(self, random=None, include_time=True):

    """Initializes an instance of `wriggly`.
    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    super().__init__(random=random)
    import ipdb
    self.prev_xy = None
    start_xy = None
    self.prev_displacement_from_start = 0.0
    self.range = np.array([np.pi/2, np.pi, np.pi/2, np.pi, np.pi/2])
    self.include_time = include_time

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
    target_box = 0.2
    xpos, ypos = random.uniform(-target_box, target_box), random.uniform(-target_box, target_box)
    physics.named.data.qpos[0] = xpos
    physics.named.data.qpos[1] = ypos
    physics.named.data.qpos[2] = 0.03


    physics.named.model.geom_pos['target', 'x'] = xpos
    physics.named.model.geom_pos['target', 'y'] = ypos
    self.prev_xy = np.array([physics.named.data.qpos[0], physics.named.data.qpos[1]])

    super().initialize_episode(physics)

  def step(self, action):
    scaled_action = action * self.range # map [-1, 1] to [-range, range]
    super().step(scaled_action)

  def get_observation(self, physics, retdict=True):
    """Returns an observation of joint angles, body velocities and target."""
    if retdict:
        
      obs = collections.OrderedDict()
      obs['joints'] = physics.joints()
      obs['jointvel'] = physics.body_velocities()
      if self.include_time:
        obs['time'] = np.array(physics.data.time)
      return obs
    else:
      if self.include_time:
        obs = np.concatenate([physics.joints(), physics.body_velocities(), np.array([physics.data.time])])
      else:
        obs = np.concatenate([physics.joints(), physics.body_velocities()])
      return obs

  # def reset(self):
  #   ret = super.reset()
  #   self.prev_xy = None
  #   return ret


  # def before_step(self, action, physics):
  #   # print("action", action)
  #   # print("before_step")
  #   current_xy = physics.named.data.qpos[0:2]
  #   # print("prev prev", self.prev_xy)
  #   self.prev_xy = current_xy.copy()
    # print("cur prev", self.prev_xy)

  # # def after_step(self, physics):
  # #   print("after_step")
  # #   print("prev", self.prev_xy)
  # #   current_xy = physics.named.data.qpos[0:2]
  # #   print("cur", current_xy)

  # #   self.prev_xy = current_xy.copy()


  def get_reward(self, physics):
      """Reward for having maximum displacement"""
      a = 0.0
      b = 1
      vel_reward = rewards.tolerance(
        physics.speed(),
        bounds=(TARGET_MOVE_SPEED, float('inf')),
        margin=TARGET_MOVE_SPEED,
        value_at_margin=0.,
        sigmoid='linear'
      )
      current_xy = physics.named.data.qpos[0:2]
      start_xy = np.array([physics.named.model.geom_pos['target', 'x'], 
                            physics.named.model.geom_pos['target', 'y']])
      # dist_reward = np.linalg.norm(self.prev_xy[0] - current_xy[0])
      dist_reward = np.linalg.norm(current_xy - start_xy)
      total_rew = a * vel_reward + b * dist_reward
      return max(total_rew, 0)

class WrigglyMaxDisp(Wriggly):
  def get_reward(self, physics):
    """Reward for having maximum displacement"""
    a = 0.0
    b = 1
    vel_reward = rewards.tolerance(
      physics.speed(),
      bounds=(TARGET_MOVE_SPEED, float('inf')),
      margin=TARGET_MOVE_SPEED,
      value_at_margin=0.,
      sigmoid='linear'
    )
    current_xy = physics.named.data.qpos[0:2]
    start_xy = np.array([physics.named.model.geom_pos['target', 'x'], 
                          physics.named.model.geom_pos['target', 'y']])
    dist_reward = np.linalg.norm(current_xy - start_xy)
    total_rew = a * vel_reward + b * dist_reward
    return max(total_rew, 0)
    
class WrigglyMaxVel(Wriggly):
  def get_reward(self, physics):
    """Reward for having maximum velocity"""
    a = 1
    b = 0.0
    vel_reward = rewards.tolerance(
      physics.speed(),
      bounds=(TARGET_MOVE_SPEED, float('inf')),
      margin=TARGET_MOVE_SPEED,
      value_at_margin=0.,
      sigmoid='linear'
    )
    current_xy = physics.named.data.qpos[0:2]
    start_xy = np.array([physics.named.model.geom_pos['target', 'x'], 
                          physics.named.model.geom_pos['target', 'y']])
    dist_reward = np.linalg.norm(current_xy - start_xy)
    total_rew = a * vel_reward + b * dist_reward
    return max(total_rew, 0)
    
class WrigglyApproachTarget(Wriggly):
  def get_reward(self, physics):
    """Reward for moving towards a specified target"""
    target_size = physics.named.model.geom_size['goal_spot', 0]
    return rewards.tolerance(physics.nose_to_target_dist(),
                            bounds=(0, target_size),
                            margin=target_size,
                            sigmoid='long_tail')
    
class WrigglyClimbObstacle(Wriggly):
  def get_reward(self, physics):
    pass
    
class WrigglyAvoidObstacle(Wriggly):
  def get_reward(self, physics):
    pass
