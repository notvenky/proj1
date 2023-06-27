# import task class
# create instance of tasks class
# create environment and pass task into it
# import mujoco as mj
# grail@10.19.188.62

from dm_control import mujoco
import PIL.Image
import numpy as np
from dm_control import composer, viewer
from dm_control.rl import control
from wriggly.simulation.robot.wriggly_from_swimmer import Wriggly, Physics
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
    
action_spec = env.action_spec()



# Define a uniform random policy.
def random_policy(time_step):
  del time_step  # Unused.
  return np.random.uniform(low=action_spec.minimum,
                           high=action_spec.maximum,
                           size=action_spec.shape)

# Launch the viewer application.
viewer.launch(env, policy=random_policy)


import torch
