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
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)
    # Random target position.
    close_target = self.random.rand() < .2  # Probability of a close target.
    target_box = .3 if close_target else 2
    xpos, ypos = self.random.uniform(-target_box, target_box, size=2)
    physics.named.model.geom_pos['target', 'x'] = xpos
    physics.named.model.geom_pos['target', 'y'] = ypos
    #physics.named.model.light_pos['target_light', 'x'] = xpos
    #physics.named.model.light_pos['target_light', 'y'] = ypos

    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of joint angles, body velocities and target."""
    obs = collections.OrderedDict()
    obs['joints'] = physics.joints()
    #obs['to_target'] = physics.nose_to_target()
    obs['jointvel'] = physics.body_velocities()
    return obs

  def get_reward(self, physics):
    """Returns a smooth reward."""
    target_size = physics.named.model.geom_size['target', 1]
    #physics.nose_to_target_dist(),
    return rewards.tolerance(physics.nose_to_target_dist(),
                             bounds=(0, target_size),
                             margin=5*target_size,
                             sigmoid='long_tail')