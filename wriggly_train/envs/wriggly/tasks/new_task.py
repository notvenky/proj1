import typing as T

import numpy as np
from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base as base_walker
from dm_control.utils import rewards, transformations

from wriggly_train.envs.wriggly.tasks import base as base_task

class MoveAtFixedVelocity(base_task.BaseTask):
  def __init__(
    self,
    robot: base_walker.Walker,
    observe_v: bool = False,
    observe_w: bool = False,
    termination_roll: T.Optional[float] = 30,
    termination_pitch: T.Optional[float] = 30,
    ground_friction: T.Optional[T.Tuple[float, float, float] | variation.Variation] = None,
    **kwargs,
  ):
    super().__init__(robot, **kwargs)
    assert termination_roll is None or termination_roll > 0
    assert termination_pitch is None or termination_pitch > 0
    # Orientation: roll = right down (+wx), pitch = front down (+wy).
    self.termination_roll = termination_roll and np.deg2rad(termination_roll)  
    self.termination_pitch = termination_pitch and np.deg2rad(termination_pitch)
    self._terminating = False

    # Define variations.
    if ground_friction:
      for geom in self.arena.ground_geoms:
        self.mjcf_variator.bind_attributes(geom, friction=ground_friction)

    # Define task-specific observables.
    obs = self.task_observables
    obs['v'] = observable.MJCFFeature('sensordata', robot.mjcf_model.sensor.velocimeter)
    obs['v'].enabled = observe_v
    obs['w'] = observable.MJCFFeature('sensordata', robot.mjcf_model.sensor.gyro)
    obs['w'].enabled = observe_w

  def initialize_episode(self, physics, random_state):
    super().initialize_episode(physics, random_state)
    self._terminating = False
    
  def after_step(self, physics, random_state):
    super().after_step(physics, random_state)
    if self.termination_roll or self.termination_pitch:
      framequat = physics.bind(self.robot.mjcf_model.sensor.framequat).sensordata
      roll, pitch, _ = transformations.quat_to_euler(framequat)
      if np.abs(roll) > self.termination_roll or np.abs(pitch) > self.termination_pitch:
        self._terminating = True

  def should_terminate_episode(self, physics):
    return self._terminating

  def get_discount(self, physics):
    return 0. if self._terminating else 1.


class MoveAtFixedVelocity_WalkInThePark(MoveAtFixedVelocity):
  def __init__(
    self, 
    robot: base_walker.Walker,
    target_vx: float,
    reward_vx: float = 1.,
    punish_wz: float = 0.1,
    **kwargs,
  ):
    super().__init__(robot, **kwargs)
    self.target_vx = target_vx
    self.reward_vx = reward_vx
    self.punish_wz = punish_wz
  
  def get_reward(self, physics):
    # Gather sensor data.
    vx, _, _ = physics.bind(self.robot.mjcf_model.sensor.velocimeter).sensordata
    _, _, wz = physics.bind(self.robot.mjcf_model.sensor.gyro).sensordata
    framequat = physics.bind(self.robot.mjcf_model.sensor.framequat).sensordata
    _, pitch, _ = transformations.quat_to_euler(framequat)

    # Gather target data.
    tvx = self.target_vx
    
    # Construct reward.
    return self.reward_vx * rewards.tolerance(
      np.cos(pitch) * vx,
      bounds=(tvx, 2 * tvx),
      margin=2 * tvx,
      value_at_margin=0,
      sigmoid='linear',
    ) - self.punish_wz * np.abs(wz)


class MoveAtFixedVelocity_PlanarVelocityTracking(MoveAtFixedVelocity):
  def __init__(
    self, 
    robot: base_walker.Walker,
    target_vx: float,
    target_vy: float,
    target_wz: float,
    reward_vx: float = 1.,
    reward_vy: float = 0.,
    reward_wz: float = 0.,
    punish_vz: float = 0.,
    punish_wx: float = 0.,
    punish_wy: float = 0.,
    punish_qwork: float = 0.,
    **kwargs,
  ):
    super().__init__(robot, **kwargs)
    self.target_vx = target_vx
    self.target_vy = target_vy
    self.target_wz = target_wz
    
    # All linear/angular velocities area relative to robot IMU frame.
    self.reward_vx = reward_vx  # front (+) / back (-)
    self.reward_vy = reward_vy  # left (+) / right (-)
    self.reward_wz = reward_wz  # yaw front left (+)

    self.punish_vz = punish_vz  # up (+) / down (-)
    self.punish_wx = punish_wx  # roll right down (+)
    self.punish_wy = punish_wy  # pitch front down (+)
    self.punish_qwork = punish_qwork
  
  def get_reward(self, physics):
    # TODO: Normalize rewards to be invariant to control steps?
    # Gather sensor data.
    vx, vy, vz = physics.bind(self.robot.mjcf_model.sensor.velocimeter).sensordata
    wx, wy, wz = physics.bind(self.robot.mjcf_model.sensor.gyro).sensordata

    # Gather target data.
    tvx = self.target_vx
    tvy = self.target_vy
    twz = self.target_wz
    
    # Construct reward.
    return sum((
      self.reward_vx * rewards.tolerance(vx - tvx, margin=1, value_at_margin=0.1),
      self.reward_vy * rewards.tolerance(vy - tvy, margin=1, value_at_margin=0.1),
      self.reward_wz * rewards.tolerance(wz - twz, margin=1, value_at_margin=0.1),
      -self.punish_vz * vz ** 2,
      -self.punish_wx * wx ** 2,
      -self.punish_wy * wy ** 2,
    ))
  
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
    target_size = physics.named.model.geom_size['targetball', 0]
    return rewards.tolerance(physics.nose_to_target_dist(),
                            bounds=(0, 2.5 * target_size),
                            margin=10 * target_size,
                            sigmoid='long_tail')
    
class WrigglyClimbObstacle(Wriggly):
  def get_reward(self, physics):
    pass
    
class WrigglyAvoidObstacle(Wriggly):
  def get_reward(self, physics):
    pass

class WrigglyOrient(Wriggly):
  def get_reward(self, physics):
    pass