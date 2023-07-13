# import task class
# create instance of tasks class
# create environment and pass task into it
# import mujoco as mj
# grail@10.19.188.62

from dm_control import mujoco
import torch
import PIL.Image
import numpy as np
from dm_control import composer, viewer
from dm_control.rl import control
from wriggly.simulation.robot.wriggly_from_swimmer import Wriggly, Physics
from wriggly.simulation.training.drqv2 import MyActor
from tqdm import tqdm

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
  action = dist.sample(clip=None)
  return action.squeeze().numpy()
#viewer.launch(env, policy = my_policy)

  

# # Define a uniform random policy.
# def random_policy(time_step):
#   del time_step  # Unused.
#   return np.random.uniform(low=action_spec.minimum,
#                            high=action_spec.maximum,
#                            size=action_spec.shape)

# # Launch the viewer application.
# viewer.launch(env, policy=random_policy)
num_params = 100
runs_per_act = 10
all_rewards = np.zeros((num_params, runs_per_act))

for i in tqdm(range(num_params)):
  frequencies = torch.rand(num_actuators) # softplus/exp/
  amplitudes = torch.rand(num_actuators)  # tanh activation
  phases = torch.rand(num_actuators)
  #log
  actor = MyActor(frequencies, amplitudes, phases, num_actuators)
  reward = evaluate(env, actor, runs_per_act, 1000)
  all_rewards[i] = reward 



