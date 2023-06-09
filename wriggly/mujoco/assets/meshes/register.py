import numpy as np
from dm_control import mjcf
from dm_control import viewer

robot = mjcf.from_path('/home/venky/Desktop/wriggly/simulation/meshes/model2.xml')

robot.register()

physics = mjcf.Physics.from_mjcf_model(robot)
env = viewer.Domain(env=None, task=None, physics=physics)

def simulation_loop():
    while True:
        physics.step()
        env.render()

simulation_loop()