import pathlib
import os
import functools
import typing as T
import collections

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.utils import transformations
from dm_env import specs


# Path to MuJoCo XML file.
MODEL_XML_PATH = pathlib.Path('/home/venky/Desktop/wriggly/simulation/meshes/wriggly.xml')

# Control range for joint velocities (if `kd` > 0).
# Control ranges for joint positions and torques can be extracted from the XML.
JOINTS_VEL_CTRL_MIN = -21.0  # rad/s
JOINTS_VEL_CTRL_MAX = +21.0  # rad/s
JOINTS_KP_MAX = 100.0
JOINTS_KD_MAX = 10.0


class Wriggly(base.Walker):
  # ------------------------------------------------------------------------------------------------
  # Initialization

  def _build(
    self,
    include_trq: bool = True,
    include_pos: bool = True,
    include_vel: bool = True,
    kp: T.Optional[float] = None,
    kd: T.Optional[float] = None,
    qpos_initial: T.Optional[np.ndarray] = None,
    name: T.Optional[str] = None,
    model_xml_path: os.PathLike = MODEL_XML_PATH,
  ):
    assert kp is None or kp >= 0
    assert kd is None or kd >= 0
    self.include_trq = include_trq
    self.include_pos = include_pos
    self.include_vel = include_vel
    self.kp = kp
    self.kd = kd
    
    # Parse MuJoCo XML file.
    self._mjcf_model = mjcf.from_path(model_xml_path)
    if name: self._mjcf_model.model = name

    # Remove elements related to freejoints, which are only allowed as worldbody grandchildren.
    freejoint = mjcf.traversal_utils.get_freejoint(self._mjcf_model.worldbody.body['trunk'])
    if freejoint: freejoint.remove()
    keyframe = self._mjcf_model.keyframe
    if keyframe: keyframe.remove()

    # Find elements that will be exposed as properties.
    self._root_body = self._mjcf_model.find('body', 'trunk')
    self._joints = self._mjcf_model.find_all('joint')
    self._actuators = self._mjcf_model.find_all('actuator')
    assert all(a.joint == j for a, j in zip(self.actuators, self.joints))

    # Set initial joints pose for start of each episode.
    self.qpos_initial = qpos_initial if qpos_initial is not None else np.zeros(len(self.joints))

  def _build_observables(self):
    observables = WrigglyObservables(self)
    observables.enable_all()
    return observables
  
  # ------------------------------------------------------------------------------------------------
  # Properties

  @property
  def mjcf_model(self):
    # Implement `composer.Entity.mjcf_model` abstractproperty.
    return self._mjcf_model

  @property
  def root_body(self):
    # Implement `locomotion.Walker.root_body` abstractproperty.
    return self._root_body

  @property
  def observable_joints(self):
    # Implement `locomotion.Walker.observable_joints` abstractproperty.
    return self._joints

  @property
  def joints(self):
    # For completeness along with `actuators`.
    return self._joints

  @property
  def actuators(self):
    # Implement `composer.Robot.actuators` abstractproperty.
    return self._actuators

  @functools.cached_property
  def jointslimits(self):
    minimum = [j.dclass.joint.range[0] for j in self.joints]
    maximum = [j.dclass.joint.range[1] for j in self.joints]
    return minimum, maximum
  
  @functools.cached_property
  def ctrllimits(self):
    minimum = [a.dclass.motor.ctrlrange[0] for a in self.actuators]
    maximum = [a.dclass.motor.ctrlrange[1] for a in self.actuators]
    return minimum, maximum

  @functools.cached_property
  def action_spec(self):
    # Override `locomotion.Walker.action_spec` property.
    spec = collections.OrderedDict()
    names = ','.join([act.name for act in self.actuators])
    if self.include_trq:
      spec['trq'] = specs.BoundedArray(
        shape=(len(self.actuators),),
        dtype=float,
        minimum=[a.dclass.motor.ctrlrange[0] for a in self.actuators],
        maximum=[a.dclass.motor.ctrlrange[1] for a in self.actuators],
        name=f'trq:{names}',
      )
    if self.include_pos:
      spec['pos'] = specs.BoundedArray(
        shape=(len(self.actuators),),
        dtype=float,
        minimum=[a.joint.dclass.joint.range[0] for a in self.actuators],
        maximum=[a.joint.dclass.joint.range[1] for a in self.actuators],
        name=f'pos:{names}',
      )
    if self.include_vel:
      spec['vel'] = specs.BoundedArray(
        shape=(len(self.actuators),),
        dtype=float,
        minimum=[JOINTS_VEL_CTRL_MIN for a in self.actuators],
        maximum=[JOINTS_VEL_CTRL_MAX for a in self.actuators],
        name=f'vel:{names}',
      )
    if self.include_pos and self.kp is None:
      spec['kp'] = specs.BoundedArray(
        shape=(len(self.actuators),),
        dtype=float,
        minimum=[0. for a in self.actuators],
        maximum=[JOINTS_KP_MAX for a in self.actuators],
        name=f'kp:{names}',
      )
    if self.include_vel and self.kd is None:
      spec['kd'] = specs.BoundedArray(
        shape=(len(self.actuators),),
        dtype=float,
        minimum=[0. for a in self.actuators],
        maximum=[JOINTS_KD_MAX for a in self.actuators],
        name=f'kd:{names}',
      )
    return spec

  # ------------------------------------------------------------------------------------------------
  # Callbacks

  def initialize_episode(self, physics, random_state):
    # Must clip because a joint may not allow qpos = 0 given its range constraints (e.g. "calf").
    joints = physics.bind(self.joints)
    minimum, maximum = self.jointslimits
    joints.qpos = np.clip(self.qpos_initial, minimum, maximum)

  # ------------------------------------------------------------------------------------------------
  # Methods

  def apply_action(self, physics, action, random_state):
    # Override `locomotion.Walker.apply_action` method.
    joints = physics.bind(self.joints)
    ctrl = np.zeros(len(self.actuators))
    
    if self.include_trq:
      ctrl += action['trq']
    
    if self.include_pos or self.kp is not None:
      kp = action['kp'] if self.kp is None else self.kp
      pos = action['pos'] if self.include_pos else self.qpos_initial
      ctrl += kp * (pos - joints.qpos)
    
    if self.include_vel or self.kd is not None:
      kd = action['kd'] if self.kd is None else self.kd
      vel = action['vel'] if self.include_vel else 0.
      ctrl += kd * (vel - joints.qvel)

    minimum, maximum = self.ctrllimits
    actuators = physics.bind(self.actuators)
    actuators.ctrl = np.clip(ctrl, minimum, maximum)


class WrigglyObservables(base.WalkerObservables):  
  # ------------------------------------------------------------------------------------------------
  # Observables
  # -----------
  # From `base.WalkerObservables`:
  #   - joints_pos
  #   - sensors_gyro
  #   - sensors_accelerometer
  #   - sensors_framequat
  # 
  # Values corresponding the 12 joints are ordered as:
  # [ FR_hip, FR_thigh, FR_calf,
  #   FL_hip, FL_thigh, FL_calf,
  #   RR_hip, RR_thigh, RR_calf,
  #   RL_hip, RL_thigh, RL_calf, ]

  @composer.observable
  def joints_vel(self):
    return observable.MJCFFeature('qvel', self._entity.observable_joints)

  @composer.observable
  def joints_trq(self):
    return observable.MJCFFeature('force', self._entity.actuators)
  
  @composer.observable
  def sensors_euler(self):
    return observable.Generic(lambda physics: transformations.quat_to_euler(
      physics.bind(self._entity.mjcf_model.sensor.framequat).sensordata
    ))

  @composer.observable
  def sensors_foot(self):
    return observable.MJCFFeature('sensordata', self._entity.mjcf_model.sensor.touch)

  @composer.observable
  def time(self):
    return observable.Generic(lambda physics: physics.data.time)

  # ------------------------------------------------------------------------------------------------
  # Properties

  @property
  def proprioception(self):
    return [
      self.joints_pos,
      self.joints_vel,
      self.joints_trq,
    ] + self._collect_from_attachments('proprioception')

  @property
  def kinematic_sensors(self):
    return [
      self.sensors_gyro,
      self.sensors_accelerometer,
      self.sensors_framequat,
      self.sensors_euler,
      self.time,
    ] + self._collect_from_attachments('kinematic_sensors')
  
  @property
  def dynamic_sensors(self):
    return [
      self.sensors_foot,
    ] + self._collect_from_attachments('dynamic_sensors')
