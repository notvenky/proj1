import dm_control.mujoco as mujoco
from dm_control import viewer

# Define the XML file path for your robot model
xml_path = '/home/venky/Desktop/wriggly/simulation/meshes/model2.xml'

# Load the custom robot model
physics = mujoco.Physics.from_xml_path(xml_path)

# Create a viewer to visualize the simulation
viewer.launch(physics)

# Set up the simulation
time_step = physics.reset()
while True:
    # Randomly sample actions for the actuators
    actions = physics.action_space.sample()

    # Step the simulation
    time_step = physics.step(actions)

    # Retrieve observations from sensors
    observations = time_step.observation

    # Do something with the observations

    # Check if the episode is done
    if time_step.last():
        break

# Close the viewer
viewer.quit()
