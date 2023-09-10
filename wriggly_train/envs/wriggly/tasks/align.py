#Align in a specific manner
import abc
import collections
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from transforms3d.euler import euler2quat

from robel.components.tracking import TrackerComponentBuilder, TrackerState
from robel.dkitty.base_env import BaseDKittyUprightEnv
from robel.simulation.randomize import SimRandomizer
from robel.utils.configurable import configurable
from robel.utils.math_utils import calculate_cosine
from robel.utils.resources import get_asset_path

DKITTY_ASSET_PATH = 'robel/dkitty/assets/dkitty_orient-v0.xml'

DEFAULT_OBSERVATION_KEYS = (
    'root_pos',
    'root_euler',
    'kitty_qpos',
    'root_vel',
    'root_angular_vel',
    'kitty_qvel',
    'last_action',
    'upright',
    'current_facing',
    'desired_facing',
)


class BaseDKittyOrient(BaseDKittyUprightEnv, metaclass=abc.ABCMeta):
    """Shared logic for DKitty orient tasks."""

    def __init__(
            self,
            asset_path: str = DKITTY_ASSET_PATH,
            observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
            target_tracker_id: Optional[Union[str, int]] = None,
            frame_skip: int = 40,
            upright_threshold: float = 0.9,  # cos(~25deg)
            upright_reward: float = 2,
            falling_reward: float = -500,
            **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            target_tracker_id: The device index or serial of the tracking device
                for the target.
            frame_skip: The number of simulation steps per environment step.
            upright_threshold: The threshold (in [0, 1]) above which the D'Kitty
                is considered to be upright. If the cosine similarity of the
                D'Kitty's z-axis with the global z-axis is below this threshold,
                the D'Kitty is considered to have fallen.
            upright_reward: The reward multiplier for uprightedness.
            falling_reward: The reward multipler for falling.
        """
        self._target_tracker_id = target_tracker_id

        super().__init__(
            sim_model=get_asset_path(asset_path),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            upright_threshold=upright_threshold,
            upright_reward=upright_reward,
            falling_reward=falling_reward,
            **kwargs)

        self._initial_angle = 0
        self._target_angle = 0

        self._markers_bid = self.model.body_name2id('markers')
        self._current_angle_bid = self.model.body_name2id('current_angle')
        self._target_angle_bid = self.model.body_name2id('target_angle')

    def _configure_tracker(self, builder):
        """Configures the tracker component."""
        super()._configure_tracker(builder)
        builder.add_tracker_group(
            'target',
            vr_tracker_id=self._target_tracker_id,
            sim_params=dict(
                element_name='target',
                element_type='site',
            ),
            mimic_xy_only=True)

    def _reset(self):
        """Resets the environment."""
        self._reset_dkitty_standing()
        # Set the initial target position.
        self.tracker.set_state({
            'torso': TrackerState(
                pos=np.zeros(3),
                rot_euler=np.array([0, 0, self._initial_angle])),
            'target': TrackerState(
                pos=np.array([
                    # The D'Kitty is offset to face the y-axis.
                    np.cos(self._target_angle + np.pi / 2),
                    np.sin(self._target_angle + np.pi / 2),
                    0,
                ])),
        })

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        self.robot.step({
            'dkitty': action,
        })

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        robot_state = self.robot.get_state('dkitty')
        torso_track_state, target_track_state = self.tracker.get_state(
            ['torso', 'target'])

        # Get the facing direction of the kitty. (the y-axis).
        current_facing = torso_track_state.rot[:2, 1]

        # Get the direction to the target.
        target_facing = target_track_state.pos[:2] - torso_track_state.pos[:2]
        target_facing = target_facing / np.linalg.norm(target_facing + 1e-8)

        # Calculate the alignment to the facing direction.
        angle_error = np.arccos(calculate_cosine(current_facing, target_facing))

        self._update_markers(torso_track_state.pos, current_facing,
                             target_facing)

        return collections.OrderedDict((
            # Add observation terms relating to being upright.
            *self._get_upright_obs(torso_track_state).items(),
            ('root_pos', torso_track_state.pos),
            ('root_euler', torso_track_state.rot_euler),
            ('root_vel', torso_track_state.vel),
            ('root_angular_vel', torso_track_state.angular_vel),
            ('kitty_qpos', robot_state.qpos),
            ('kitty_qvel', robot_state.qvel),
            ('last_action', self._get_last_action()),
            ('current_facing', current_facing),
            ('desired_facing', target_facing),
            ('angle_error', angle_error),
        ))

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        angle_error = obs_dict['angle_error']
        upright = obs_dict[self._upright_obs_key]
        center_dist = np.linalg.norm(obs_dict['root_pos'][:2], axis=1)

        reward_dict = collections.OrderedDict((
            # Add reward terms for being upright.
            *self._get_upright_rewards(obs_dict).items(),
            # Reward for closeness to desired facing direction.
            ('alignment_error_cost', -4 * angle_error),
            # Reward for closeness to center; i.e. being stationary.
            ('center_distance_cost', -4 * center_dist),
            # Bonus when mean error < 15deg or upright within 15deg.
            ('bonus_small', 5 * ((angle_error < 0.26) + (upright > 0.96))),
            # Bonus when error < 5deg and upright within 15deg.
            ('bonus_big', 10 * ((angle_error < 0.087) * (upright > 0.96))),
        ))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        return collections.OrderedDict((
            ('points', 1.0 - obs_dict['angle_error'] / np.pi),
            ('success', reward_dict['bonus_big'] > 0.0),
        ))

    def _update_markers(self, root_pos: np.ndarray, current_facing: np.ndarray,
                        target_facing: np.ndarray):
        """Updates the simulation markers denoting the facing direction."""
        self.model.body_pos[self._markers_bid][:2] = root_pos[:2]
        current_angle = np.arctan2(current_facing[1], current_facing[0])
        target_angle = np.arctan2(target_facing[1], target_facing[0])

        self.model.body_quat[self._current_angle_bid] = euler2quat(
            0, 0, current_angle, axes='rxyz')
        self.model.body_quat[self._target_angle_bid] = euler2quat(
            0, 0, target_angle, axes='rxyz')
        self.sim.forward()


@configurable(pickleable=True)
class DKittyOrientFixed(BaseDKittyOrient):
    """Stand up from a fixed position."""

    def _reset(self):
        """Resets the environment."""
        # Put target behind the D'Kitty (180deg rotation).
        self._initial_angle = 0
        self._target_angle = np.pi
        super()._reset()


@configurable(pickleable=True)
class DKittyOrientRandom(BaseDKittyOrient):
    """Walk straight towards a random location."""

    def __init__(
            self,
            *args,
            # +/-60deg
            initial_angle_range: Tuple[float, float] = (-np.pi / 3, np.pi / 3),
            # 180 +/- 60deg
            target_angle_range: Tuple[float, float] = (2 * np.pi / 3,
                                                       4 * np.pi / 3),
            **kwargs):
        """Initializes the environment.

        Args:
            initial_angle_range: The range to sample an initial orientation
                of the D'Kitty about the z-axis.
            target_angle_range: The range to sample a target orientation of
                the D'Kitty about the z-axis.
        """
        super().__init__(*args, **kwargs)
        self._initial_angle_range = initial_angle_range
        self._target_angle_range = target_angle_range

    def _reset(self):
        """Resets the environment."""
        self._initial_angle = self.np_random.uniform(*self._initial_angle_range)
        self._target_angle = self.np_random.uniform(*self._target_angle_range)
        super()._reset()


@configurable(pickleable=True)
class DKittyOrientRandomDynamics(DKittyOrientRandom):
    """Walk straight towards a random location."""

    def __init__(self,
                 *args,
                 sim_observation_noise: Optional[float] = 0.05,
                 **kwargs):
        super().__init__(
            *args, sim_observation_noise=sim_observation_noise, **kwargs)
        self._randomizer = SimRandomizer(self)
        self._dof_indices = (
            self.robot.get_config('dkitty').qvel_indices.tolist())

    def _reset(self):
        """Resets the environment."""
        # Randomize joint dynamics.
        self._randomizer.randomize_dofs(
            self._dof_indices,
            all_same=True,
            damping_range=(0.1, 0.2),
            friction_loss_range=(0.001, 0.005),
        )
        self._randomizer.randomize_actuators(
            all_same=True,
            kp_range=(2.8, 3.2),
        )
        # Randomize friction on all geoms in the scene.
        self._randomizer.randomize_geoms(
            all_same=True,
            friction_slide_range=(0.8, 1.2),
            friction_spin_range=(0.003, 0.007),
            friction_roll_range=(0.00005, 0.00015),
        )
        # Generate a random height field.
        self._randomizer.randomize_global(
            total_mass_range=(1.6, 2.0),
            height_field_range=(0, 0.05),
        )
        self.sim_scene.upload_height_field(0)
        super()._reset()

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
    physics.named.data.qpos[2] = 0.03
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