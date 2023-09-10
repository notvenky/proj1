import random
import numpy as np
import mujoco as mj
import math
import csv

def main():
    max_width = 200
    max_height = 200

    cam = mj.MjvCamera()
    opt = mj.MjvOption()

    mj.glfw.glfw.init()
    window = mj.glfw.glfw.create_window(1200, 900, "Demo", None, None)
    mj.glfw.glfw.make_context_current(window)
    mj.glfw.glfw.swap_interval(1)

    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)

    xml_path = "/home/venky/proj1/wriggly/mujoco/wriggly.xml"

    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    num_actuators = model.nu
    goal_positions = [random.uniform(-1.57, 1.57) for _ in range(num_actuators)]
    freqs = [random.uniform(0.5, 2.0) for _ in range(num_actuators)]
    phases = [random.uniform(-np.pi, np.pi) for _ in range(num_actuators)]
    std_dev = 0.02  # Standard deviation of the Gaussian noise

    while not mj.glfw.glfw.window_should_close(window):
        simstart = data.time

        while (data.time - simstart < 1.0 / 60.0):
            # Update goal positions
            for i in range(num_actuators):
                # Generate the sine wave for each actuator
                goal_positions[i] = np.sin(2 * np.pi * freqs[i] * data.time + phases[i])

                # Move the actuator towards the goal position
                if data.ctrl[i] < goal_positions[i]:
                    data.ctrl[i] += 0.3925
                else:
                    data.ctrl[i] -= 0.3925

                # Add Gaussian noise
                noise = np.random.normal(0, std_dev)
                data.ctrl[i] += noise

            # Log actuator positions to a csv file
            with open('actuator_positions.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data.ctrl)

            mj.mj_step(model, data)

        viewport = mj.MjrRect(0, 0, 1200, 900)

        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        mj.glfw.glfw.swap_buffers(window)
        mj.glfw.glfw.poll_events()

    mj.glfw.glfw.terminate()

if __name__ == "__main__":
    main()