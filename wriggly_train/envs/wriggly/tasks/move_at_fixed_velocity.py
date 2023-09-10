#Move_At_Fixed_Velocity

import typing as T

import numpy as np
from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base as base_walker
from dm_control.utils import rewards, transformations

from wriggly.simulation.tasks import base as base_task


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
