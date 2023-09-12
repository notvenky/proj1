import collections
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
TARGET_SPEED_MOVE = 10.0



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



@SUITE.add('benchmarking')
def move(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  xml_path = "/home/venky/proj1/wriggly_train/envs/wriggly_mujoco/wriggly_apr_target.xml"
  # wriggly =  mj.MjModel.from_xml_path(xml_path)
  physics = Physics.from_xml_path(xml_path)
  # physics.legacy_step = False
  # data = mj.MjData(wriggly)
  task = Wriggly()
  # env = control.Environment(physics, task, time_limit=time_limit, legacy_step=True)
  env = control.Environment(physics, task, time_limit=time_limit,
                            legacy_step=False, n_sub_steps=2)
  return env


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
  
  def speed(self):
    """Returns the horizontal speed of Wriggly."""
    return self.named.data.sensordata['ACT0_velocity_sensor'][0]
  
  def desired_direction(self):
    """Returns the desired direction of Wriggly."""
    return np.array([1, 0, 0])
  
  def current_position(self):
    """Returns the current position of Wriggly."""
    return np.abs(self.named.data.sensordata['ACT2_pos_sensor'][0])
   
  def sum_speed(self):
    """Returns the sum of speeds of all parts of Wriggly."""
    return 0.2 * (self.named.data.sensordata['ACT0_velocity_sensor'][0] + self.named.data.sensordata['ACT1_velocity_sensor'][0] + self.named.data.sensordata['ACT2_velocity_sensor'][0] + self.named.data.sensordata['ACT3_velocity_sensor'][0] + self.named.data.sensordata['ACT4_velocity_sensor'][0])

class Wriggly(base.Task):
  start_time = time.time()
  prev_xy = None
  prev_x_diff = None 
  prev_rotation = None
  start_orientation = None
  start_xy = None
  start_x1 = None
  start_y1 = None
  start_x2 = None
  start_y2 = None
  start_x3 = None
  start_y3 = None
  start_x4 = None
  start_y4 = None
  start_x5 = None
  start_y5 = None
  start_x6 = None
  start_y6 = None
  
  
  """A wriggly `Task` to reach the target or just swim."""

  def __init__(self, random=None):

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
    target_box = 1.0
    xpos, ypos = random.uniform(-target_box, target_box), random.uniform(-target_box, target_box)
    physics.named.data.qpos[0] = xpos
    physics.named.data.qpos[1] = ypos
    physics.named.data.qpos[2] = 0.03


    physics.named.model.geom_pos['target', 'x'] = xpos
    physics.named.model.geom_pos['target', 'y'] = ypos
    self.prev_xy = np.array([physics.named.data.qpos[0], physics.named.data.qpos[1]])

    super().initialize_episode(physics)

  def get_observation(self, physics, retdict=True):
    """Returns an observation of joint angles, body velocities and target."""
    if retdict:
        
      obs = collections.OrderedDict()
      obs['joints'] = physics.joints()
      obs['jointvel'] = physics.body_velocities()
      obs['time'] = np.array(physics.data.time)
      return obs
    else:
      obs = np.concatenate([physics.joints(), physics.body_velocities(), np.array([physics.data.time])])
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
      # print("get_reward")
      # print("time", physics.time())
      # desired_direction = np.array([1, 0, 0])  # Replace with the actual desired direction unit vector
      # current_velocity = np.array([velocity_x, velocity_y, velocity_z])  # Replace with actual velocities

      # reward = np.dot(desired_direction, current_velocity)
      # if distance_covered > threshold_distance:
      # reward += large_bonus

      # vel = rewards.tolerance(physics.sum_speed(),
      #                        bounds=(TARGET_SPEED_MOVE, float('inf')),
      #                        margin=TARGET_SPEED_MOVE,
      #                        value_at_margin=0,
      #                        sigmoid='linear')
      # dist = physics.current_position()
      # return dist
      
      # velocity = np.array([physics.named.data.sensordata['ACT2_velocity_sensor'][0]])
      # start_xy = np.array([physics.named.model.geom_pos['target', 'x'], 
      #                       physics.named.model.geom_pos['target', 'y']])
      # current_xy = np.array(physics.named.data.qpos[0:2])
      # displacement_from_start = np.array([current_xy - start_xy])

      # displacement_from_prev = np.array([current_xy - self.prev_xy]) if self.prev_xy is not None else np.array([0, 0])

      # dot_product_with_start = np.dot(velocity, displacement_from_start)
      # dot_product_with_prev = np.dot(velocity, displacement_from_prev)

      # # Factor to weigh the importance of distance and velocity
      # alpha = 0.5

      # if physics.time() >= _DEFAULT_TIME_LIMIT:
      #     return np.linalg.norm(current_xy - start_xy)
      # else:
      #     return alpha * dot_product_with_start + (1 - alpha) * dot_product_with_prev
      
      '''


      if physics.time() >= _DEFAULT_TIME_LIMIT:
        current_xy = physics.named.data.qpos[0:2]
        return np.linalg.norm(self.prev_xy[0] - current_xy[0])
      else:
        return 0.0

      velocity = np.array([physics.named.sensor.data['ACT2_velocity_sensor'][0]])

      # Get the current x and y coordinates of the center of mass of the robot
      current_xy = np.array(physics.named.data.qpos[0:2])

      # Calculate the displacement from the start to the current position
      displacement_from_start = current_xy - start_xy

      # Calculate the displacement from the previous to the current position if available
      displacement_from_prev = current_xy - prev_xy if prev_xy is not None else np.array([0, 0])

      # Dot product between velocity and displacement vectors.
      # This rewards high velocity in the direction of motion.
      dot_product_with_start = np.dot(velocity, displacement_from_start)
      dot_product_with_prev = np.dot(velocity, displacement_from_prev)

      # Factor to weigh the importance of distance and velocity
      alpha = 0.5

      # Check for time termination condition
      if physics.time() >= _DEFAULT_TIME_LIMIT:
          # Sparse Reward at the end of evaluation time
          return np.linalg.norm(current_xy - start_xy)
      else:
          # Reward that takes into account both velocity and distance
          return alpha * dot_product_with_start + (1 - alpha) * dot_product_with_prev

      '''

        
        
        
        
        
        
        
        
        
      # """Computes the reward for the current timestep"""
      # current_xy = physics.named.data.qpos[0:2]
      # return np.linalg.norm(self.prev_xy[0] - current_xy[0])

    

      a = 0.1
      b = 1
      forward_velocity = physics.named.data.sensordata['ACT2_velocity_sensor'][0]
      vel_reward = rewards.tolerance(
        forward_velocity,
        bounds=(50, float('inf')),
        margin=50,
        value_at_margin=0.,
        sigmoid='linear'
      )
      current_xy = physics.named.data.qpos[0:2]
      start_xy = np.array([physics.named.model.geom_pos['target', 'x'], 
                            physics.named.model.geom_pos['target', 'y']])
      # dist_reward = np.linalg.norm(self.prev_xy[0] - current_xy[0])
      dist_reward = np.linalg.norm(current_xy - start_xy)
      total_rew = a * vel_reward + b * dist_reward
      return total_rew
    
    # reward function that rewards deltas in the x coordinate, that is the robot
    # moves in the x direction, and the distance moves is calculated by the change
    # in distance from the starting point, not in absolute distance from previous time
    # step. This is to ensure that the robot is not rewarded for moving in the y direction
      '''
        Reward for moving continuously away from starting position in the x-direction, punish for moving back
      '''








      # """Computes the reward based on the direction of movement."""
      # current_xy = physics.named.data.qpos[0:2]
      # # print("cur_Xy", current_xy)
      # # print("prev_xy", self.prev_xy, "\n")
      # if self.prev_xy is None:
      #     prev_xy = current_xy
      # else:
      #     prev_xy = self.prev_xy

      # # Calculate the displacement deltas
      # # displacement_deltas = current_xy - prev_xy

      # # Calculate the total reward as the sum of the displacement deltas
      # # reward = np.linalg.norm(displacement_deltas) / physics.timestep()  * 1000
      # reward = current_xy[0] - prev_xy[0]
      # # print(current_xy, prev_xy)

      # # print("reward a", reward)
      # # print("current_xy", current_xy)
      # # print("prev_xy", self.prev_xy)
      # # print("done")

      # # if reward > 0.001:
      # #   print("get reward")
      # #   print("prev xy", self.prev_xy)
      # #   print("current", current_xy, "\n")
      # #   print("reward", reward)
      

      # return reward