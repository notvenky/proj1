from dm_control import mujoco
import torch
import PIL.Image
import numpy as np
from dm_control import composer, viewer
from dm_control.rl import control
from wriggly.simulation.robot.wriggly_from_swimmer import Wriggly, Physics
from wriggly.simulation.training.drqv2 import MyActor
from tqdm import tqdm
import re
import imageio
import cv2
import pyautogui
import sys
from pyvirtualdisplay import Display

xml_path = "/home/venky/proj1/wriggly/mujoco/wriggly_apr_target.xml"
physics = Physics.from_xml_path(xml_path)
task = Wriggly()

env = control.Environment(physics, task, legacy_step=True)

 
env.reset()
# pixels = []
# for camera_id in range(2):
#   pixels.append(env.physics.render(camera_id=camera_id, width=480))
# PIL.Image.fromarray(np.hstack(pixels))


# import matplotlib.pyplot as plt
# plt.imshow(pixels[-1])
# plt.show()
std_dev = 0.02  # Standard deviation of the Gaussian noise
    # Provided string
# info_string = "Frequency: tensor([0.4109, 0.1589, 0.2838, 0.4712, 0.2662]), Amplitude: tensor([1.2838, 2.0613, 1.5466, 1.8574, 0.0289]), Phase: tensor([-5.1056, -1.6779, -0.5661,  5.7423, -1.4933]), Reward 64.49960242951404"
# info_string = "Frequency: tensor([0.5319, 1.0512, 0.5440, 0.4357, 0.2722]), Amplitude: tensor([0.7951, 1.9331, 1.3052, 2.8653, 0.6707]), Phase: tensor([-0.3350, -1.1112, -0.6347,  0.7875, -0.1924])"
# info_string = "Max Reward: 7778.276081399728, Frequency: tensor([-0.0914,  0.6134,  0.1397,  0.3034,  0.8469]), Amplitude: tensor([1.6468, 0.8434, 1.2160, 3.1182, 1.3151]), Phase: tensor([-4.1795, -2.2679,  3.3211, -4.6133, -3.6108])"
# info_string = "Max Reward: 8876.837187688203, Frequency: tensor([0.3289, 0.4376, 0.2811, 0.3667, 0.3732]), Amplitude: tensor([-0.2801, -2.6859, -1.4523,  3.2879,  2.0484]), Phase: tensor([-8.7795,  3.7695,  0.4016,  3.5974,  1.6966])"
# info_string = "Max Reward: 95.24811929803076, Frequency: tensor([0.1265, 0.3062, 0.2338, 0.2176, 0.4514]), Amplitude: tensor([1.4656, 1.1995, 1.0585, 0.2807, 1.4087]), Phase: tensor([ 2.0645, -0.9187,  0.2120,  5.0613, -7.1874])"
# info_string = "Max Reward: 324.91804589471843, Frequency: tensor([0.1985, 0.1542, 0.0337, 0.2188, 0.3791]), Amplitude: tensor([1.5594, 2.9358, 0.7446, 1.1392, 1.2070]), Phase: tensor([2.3062, 1.8530, 1.0443, 2.6414, 1.0222])"
# info_string = "Reward: 303.0023824954088, Frequency: tensor([0.2460, 0.3875, 0.3313, 0.4128, 0.2972]), Amplitude: tensor([1.5132, 2.1969, 1.4633, 2.5538, 0.5573]), Phase: tensor([1.2214, 2.8504, 2.0387, 0.0394, 1.9464])"
# info_string = "Reward: 1117.745925341783, Frequency: tensor([0.9421, 0.5917, 0.8685, 0.4496, 0.0852]), Amplitude: tensor([0.8103, 0.5961, 1.3018, 0.2412, 0.1356]), Phase: tensor([3.7086, 1.8786, 1.6131, 4.3720, 4.1930])"
# info_string = "Time: 20:45:21.360926, Reward: 3689.1597391055016, Frequency: tensor([0.4303, 0.4154, 0.4517, 0.3578, 0.2295]), Amplitude: tensor([1.3330, 1.2507, 0.8577, 2.2365, 0.8378]), Phase: tensor([2.1703, 1.8762, 0.6844, 6.2216, 1.6259])"
# info_string = "Max Reward: 3045.9593233112914, Frequency: tensor([0.2589, 0.2289, 0.1748, 0.2487, 0.4317]), Amplitude: tensor([1.5446, 0.6533, 0.9733, 1.7542, 1.2021]), Phase: tensor([1.2686, 2.7937, 1.6441, 6.0062, 0.2665])"
info_string = 'Max Reward: 1107.0495866678302, Frequency: tensor([0.0711, 0.2483, 0.3779, 0.2108, 0.0492]), Amplitude: tensor([1.3856, 0.6317, 1.2593, 1.5167, 0.9829]), Phase: tensor([3.4886, 2.3245, 3.4475, 2.7257, 5.6565])'

frequencies = torch.tensor(np.array(re.findall(r"Frequency: tensor\((.*?)\)", info_string)[0].replace('[','').replace(']','').split(','), dtype=float))
amplitudes = torch.tensor(np.array(re.findall(r"Amplitude: tensor\((.*?)\)", info_string)[0].replace('[','').replace(']','').split(','), dtype=float))
phases = torch.tensor(np.array(re.findall(r"Phase: tensor\((.*?)\)", info_string)[0].replace('[','').replace(']','').split(','), dtype=float))

num_actuators = 5
actor = MyActor(frequencies, amplitudes, phases, num_actuators)

def my_policy(obs, ):
  time = torch.tensor(obs.observation["time"]).unsqueeze(0)
  with torch.no_grad():
    dist = actor(None, time, 0)
  # action = dist.sample(clip=None)
  action = dist.mean
  return action.squeeze().numpy()

# # Create a virtual display to render the Mujoco window
# display = Display(visible=0, size=(640, 480))
# display.start()

# # Create an OpenCV video writer to save the frames
# video_path = 'simulation_video.mp4'
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(video_path, fourcc, 30, (640, 480))

# Run the simulation and capture frames
viewer.launch(env, policy=my_policy)
# for _ in tqdm(range(1000)):  # Change the number of frames as per your requirement
#     frame = env.physics.render(height=480, width=640, camera_id=0)
#     out.write(frame)
# out.release()

# # Close the virtual display
# display.stop()

