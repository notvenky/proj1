from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

# Control Suite
from dm_control import suite

# Run through corridor example
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks

# Soccer
from dm_control.locomotion import soccer

# Manipulation
from dm_control import manipulation

import copy
import os
import itertools
from IPython.display import clear_output
import numpy as np

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image


plt.rcParams['figure.fig_format'] = 'svg'

# Font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Inline video helper function
if os.environ.get('COLAB_NOTEBOOK_TEST', False):
  # We skip video generation during tests, as it is quite expensive.
  display_video = lambda *args, **kwargs: None
else:
  def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())

# Seed numpy's global RNG so that cell outputs are deterministic. We also try to
# use RandomState instances that are local to a single cell wherever possible.
np.random.seed(42)

static_model = """
<mujoco model="wriggly">
  <compiler angle="radian" autolimits="true"/>
  <default>
      <position forcelimited="true" forcerange="-3 3"/>
  </default>
  <asset>
    <!-- Meshes -->
    <mesh name="base_link" file="base_link.STL"/>
    <mesh name="green_leg_holder" file="green_leg_holder.STL"/>
    <mesh name="central_link" file="central_link.STL"/>
    <mesh name="red_central" file="red_central.STL"/>
    <mesh name="red_leg_holder" file="red_leg_holder.STL"/>
    <mesh name="red_leg" file="red_leg.STL"/>
    <!-- Textures -->
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <!-- Materials -->
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <!-- Floor -->
    <geom name="floor" condim="3" friction="0.9" size="0 0 0.05" type="plane" material="groundplane"/>
    <!-- Light -->
    <light name="ceiling" pos="0 0 2" dir="0 0 -1" directional="true"/>
    <!-- Cameras -->
    <camera name="target_from_origin" pos="0 0 1.25" xyaxes="0 -1 0 1 0 0"/>
    <camera name="fixed_right" pos="2 -3 0.5" xyaxes="1 0 0 0 0.08 0.92" fovy="15" mode="fixed"/>

    <!-- Green Leg -->
    <body name="green_leg">
        <!-- Joint -->
        <joint name="asdf" pos="0 0 0" type="free" damping="0.1" armature="0.005" frictionloss="0.01"/>
        <!-- Base Link -->
        <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="base_link"/>
        
        <!-- Green Leg Holder -->
        <body name="Green_Leg_Holder" pos="0.3235 -0.23065 0" quat="0.5 -0.5 0.5 -0.5" gravcomp="0">
          <inertial pos="0.0172732 -1.71313e-05 -0.0312874" quat="0.707094 0.00279184 0.00136492 0.707112" mass="0.101" diaginertia="1.71483e-05 1.52784e-05 1.18772e-05"/>
          <joint name="Green_Leg_Final_Joint" pos="0 0 0" axis="1 0 0" damping="0.1" armature="0.005" frictionloss="0.01"/>
          <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="green_leg_holder"/>
          
          <!-- Central Link -->
          <body name="Central_Link" pos="0.017 0 -0.1045" quat="0.707107 0 0 0.707107" gravcomp="0">
            <inertial pos="9.37688e-10 -0.0119557 -0.00143737" quat="0.707745 0.706468 -1.73858e-06 1.74172e-06" mass="0.091" diaginertia="1.40798e-05 1.31645e-05 8.83029e-06"/>
            <joint name="Green_Leg_Central_Joint" pos="0 0 0" axis="0 0 1" damping="0.1" armature="0.005" frictionloss="0.01"/>
            <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="central_link"/>
            
            <!-- Red Leg Central Link -->
            <body name="Red_Leg_Central_Link" pos="0 0 -0.04515" quat="0.5 0.5 0.5 0.5" gravcomp="0">
              <inertial pos="-0.000634684 -0.0270609 -0.000240499" quat="0.707212 0.704458 -0.0410108 -0.0436829" mass="0.113" diaginertia="2.05457e-05 1.98685e-05 1.50592e-05"/>
              <joint name="Red_Leg_Central_Joint" pos="0 0 0" axis="0 0 1" damping="0.1" armature="0.005" frictionloss="0.01"/>
              <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="red_central"/>
              
              <!-- Red Leg Holder -->
              <body name="Red_Leg_Holder" pos="0 -0.10485 0" quat="-0.5 0.5 0.5 0.5" gravcomp="0">
                <inertial pos="9.37688e-10 -0.0119557 -0.00143737" quat="0.707745 0.706468 -1.73858e-06 1.74172e-06" mass="0.091" diaginertia="1.40798e-05 1.31645e-05 8.83029e-06"/>
                <joint name="Red_Knee" pos="0 0 0" axis="0 0 1" damping="0.1" armature="0.005" frictionloss="0.01"/>
                <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="red_leg_holder"/>
                
                <!-- Red Leg -->
                <body name="Red_Leg" pos="0 0 -0.045" quat="0.5 -0.5 0.5 -0.5" gravcomp="0">
                  <inertial pos="0.00101128 0.025027 -0.000247357" quat="0.705699 0.707278 -0.0298534 -0.0292677" mass="0.112" diaginertia="2.31293e-05 2.08201e-05 1.20376e-05"/>
                  <joint name="Red_Leg_Final_Joint" pos="0 0 0" axis="0 0 1" damping="0.1" armature="0.005" frictionloss="0.01"/>
                  <geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="red_leg"/>
                </body>
              </body>
            </body>
          </body>
        </body>
    </body>
  </worldbody>
  <actuator>
      <position name="act0" kp ="2" joint="Green_Leg_Final_Joint"   ctrlrange="-1.57 1.57"/>
      <position name="act1" kp ="2" joint="Green_Leg_Central_Joint" ctrlrange="-3.14 3.14"/>
      <position name="act2" kp ="2" joint="Red_Leg_Central_Joint"   ctrlrange="-1.57 1.57"/>
      <position name="act3" kp ="2" joint="Red_Knee"                ctrlrange="-3.14 3.14"/>
      <position name="act4" kp ="2" joint="Red_Leg_Final_Joint"     ctrlrange="-1.57 1.57"/>
  </actuator>
</mujoco>
"""

physics = mujoco.Physics.from_xml_string(static_model)
# Visualize the joint axis.
scene_option = mujoco.wrapper.core.MjvOption()
scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True
pixels = physics.render(scene_option=scene_option)
PIL.Image.fromarray(pixels)

duration = 2    # (seconds)
framerate = 30  # (Hz)

# Visualize the joint axis
scene_option = mujoco.wrapper.core.MjvOption()
scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True

# Simulate and display video.
frames = []
physics.reset()  # Reset state and time
while physics.data.time < duration:
  physics.step()
  if len(frames) < physics.data.time * framerate:
    pixels = physics.render(scene_option=scene_option)
    frames.append(pixels)
display_video(frames, framerate)