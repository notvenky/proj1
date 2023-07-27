# import task class
# create instance of tasks class
# create environment and pass task into it
# import mujoco as mj
# grail@10.19.188.62
import os
os.environ['MUJOCO_GL'] = 'egl'

from dm_control import mujoco
import torch
import PIL.Image
import numpy as np
from dm_control import composer, viewer
from dm_control.rl import control
from wriggly.simulation.robot.wriggly_from_swimmer import Wriggly, Physics
from wriggly.simulation.training.drqv2 import MyActor
from tqdm import tqdm
import heapq
import datetime

xml_path = "/home/venky/proj1/wriggly/mujoco/wriggly_apr_target.xml"
# wriggly =  mj.MjModel.from_xml_path(xml_path)
physics = Physics.from_xml_path(xml_path)
# data = mj.MjData(wriggly)



task = Wriggly(
    # wriggly=wriggly,
    # wriggly_spawn_position=(0.5, 0, 0),
    # target_velocity=3.0,
    # physics_timestep=0.005,
    # control_timestep=0.03,
)


env = control.Environment(physics, task, legacy_step=True)

# env = composer.Environment(
#     task=task,
#     time_limit=10,
#     random_state=np.random.RandomState(42),
#     strip_singleton_obs_buffer_dim=True,
# )
 
env.reset()
pixels = []
for camera_id in range(2):
  pixels.append(env.physics.render(camera_id=camera_id, width=480))
PIL.Image.fromarray(np.hstack(pixels))


# import matplotlib.pyplot as plt
# plt.imshow(pixels[-1])
# plt.show()

ret = env.step([0,0,0,0,0.])
print(ret)
num_actuators = 5   
action_spec = env.action_spec()
frequencies = torch.rand(num_actuators) # softplus/exp/
amplitudes = torch.rand(num_actuators)  # tanh activation
phases = torch.rand(num_actuators)
actor = MyActor(frequencies, amplitudes, phases, num_actuators)

def evaluate(env, actor, num_episodes, T):
  rewards = np.zeros(num_episodes)
  for i in range(num_episodes):
    obs = env.reset()
    sum_rewards = 0
    for t in range(T):
      time = torch.tensor(obs.observation["time"]).unsqueeze(0)
      with torch.no_grad():
        dist = actor(None, time, 0)
      action = dist.sample(clip=None)
      obs = env.step(action.squeeze().numpy())
      sum_rewards += obs.reward
    rewards[i] = sum_rewards
  return rewards

def my_policy(obs, ):
  time = torch.tensor(obs.observation["time"]).unsqueeze(0)
  with torch.no_grad():
    dist = actor(None, time, 0)
  action = dist.mean()
  # action = dist.sample(clip=None)
  return action.squeeze().numpy()
viewer.launch(env, policy = my_policy)

  

# # Define a uniform random policy.
# def random_policy(time_step):
#   del time_step  # Unused.
#   return np.random.uniform(low=action_spec.minimum,
#                            high=action_spec.maximum,
#                            size=action_spec.shape)

# # Launch the viewer application.
# viewer.launch(env, policy=my_policy)
num_params = 3000
runs_per_act = 5
all_rewards = np.zeros((num_params, runs_per_act))

# Define data structures to store frequencies, amplitudes and phases
all_frequencies = np.zeros((num_params, num_actuators))
all_amplitudes = np.zeros((num_params, num_actuators))
all_phases = np.zeros((num_params, num_actuators))

# Variables to store maximum reward and its corresponding frequency, amplitude, and phase
max_reward = -np.inf
max_reward_freq = None
max_reward_amp = None
max_reward_phase = None


# for i in tqdm(range(num_params)):
#   frequencies = torch.rand(num_actuators) / 2 # softplus/exp/
#   amplitudes = torch.rand(num_actuators)   # tanh activation
#   amplitudes[0] = amplitudes[0] * 0.37 + 1.2 
#   amplitudes[1] = amplitudes[1] * 1.57  
#   amplitudes[2] = amplitudes[2] * 0.7 + 0.8 
#   amplitudes[3] = amplitudes[3] + 1.5
#   amplitudes[4] = amplitudes[4] * 0.7 + 0.8  

for i in tqdm(range(num_params)):
  frequencies = torch.rand(num_actuators) / 2 # softplus/exp/
  amplitudes = torch.rand(num_actuators)   # tanh activation
  amplitudes[0] = amplitudes[0] * 1.57
  amplitudes[1] = amplitudes[1] * 3.14 
  amplitudes[2] = amplitudes[2] * 1.57
  amplitudes[3] = amplitudes[3] * 3.14
  amplitudes[4] = amplitudes[4] * 1.57 

  # amplitudes[::2] = amplitudes[::2] * 1.57  # For 1st, 3rd, and 5th
  # amplitudes[1::2] = amplitudes[1::2] * 3.14  # For 2nd and 4th
  phases = torch.rand(num_actuators) * 2 * np.pi
  #np.pi for 1 and 3 np.p/2 for 0, 2 & 4
  actor = MyActor(frequencies, amplitudes, phases, num_actuators)
  
  # Store frequencies, amplitudes and phases
  # all_frequencies[i] = frequencies.numpy()
  # all_amplitudes[i] = np.pi * amplitudes.numpy() if i % 2 == 0 else np.pi/2 * amplitudes.numpy()
  # all_phases[i] = phases.numpy()

  reward = evaluate(env, actor, runs_per_act, 2500) # for 10 seconds, since 0.002s for 1 step 
  all_rewards[i] = reward 
  top_rewards = []

  mean_reward = np.mean(reward)
  print(f"Sample {i} Frequency: {frequencies}, Amplitude: {amplitudes}, Phase: {phases}, Reward {mean_reward}")
  # print(max_reward)
  print(f"Max Reward: {max_reward}, Frequency: {max_reward_freq}, Amplitude: {max_reward_amp}, Phase: {max_reward_phase}")




  # Print frequencies, amplitudes, phases and rewards for each run
  # for run in range(runs_per_act):
  #   print(f"Sample {i}, Run {run}: Frequencies {frequencies}, Amplitudes {amplitudes}, Phases {phases}, Reward {reward[run]}")
    
    # Update maximum reward and corresponding frequency, amplitude, phase if necessary
  if mean_reward > max_reward:
    max_reward = mean_reward
    max_reward_freq = frequencies
    max_reward_amp = amplitudes
    max_reward_phase = phases

  #   # If we have less than 5 rewards or the new reward is higher than the smallest of the top 5
  #   if len(top_rewards) < 5 or reward[run] > top_rewards[0][0]:
  #     # If we already have 5 rewards, remove the smallest one
  #     if len(top_rewards) == 5:
  #       heapq.heappop(top_rewards)
  #     # Add the new reward
  #     heapq.heappush(top_rewards, (-reward[run], frequencies.numpy(), amplitudes.numpy(), phases.numpy()))

# Print the final rewards for all samples and runs
with open('top_rewards.txt', 'a') as f:
    f.write(f"Time: {datetime.datetime.now().time()}, Reward: {max_reward}, Frequency: {max_reward_freq}, Amplitude: {max_reward_amp}, Phase: {max_reward_phase}\n")
print(i, all_rewards)

# Print maximum reward and corresponding frequency, amplitude, phase
print(f"Max Reward: {max_reward}, Frequency: {max_reward_freq}, Amplitude: {max_reward_amp}, Phase: {max_reward_phase}")

actor_viz = MyActor(max_reward_freq, max_reward_amp, max_reward_phase, num_actuators)

def viz_policy(obs, ):
  time = torch.tensor(obs.observation["time"]).unsqueeze(0)
  with torch.no_grad():
    dist = actor_viz(None, time, 0)
  # action = dist.sample(clip=None)
  action = dist.mean
  return action.squeeze().numpy()

viewer.launch(env, policy = viz_policy)

