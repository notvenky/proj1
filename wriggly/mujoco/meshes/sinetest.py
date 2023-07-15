import random
import numpy as np
import mujoco as mj
import math
import csv
import re

def main():
    max_width = 200
    max_height = 200
    # ctx = mj.GLContext(max_width, max_height)
    # ctx.make_current()

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
    #print(num_actuators)

    std_dev = 0.02  # Standard deviation of the Gaussian noise
    # Provided string
    info_string = "ample 78 Frequencies tensor([0.0825, 0.8131, 0.4743, 0.2370, 0.5841]), Amplitudes tensor([0.5309, 0.1336, 0.0449, 0.1713, 0.1849]), Phases tensor([0.0938, 0.7247, 0.6550, 0.9931, 0.6703]), Reward 3359.41556111543"

    frequencies = np.array(re.findall(r"Frequencies tensor\((.*?)\)", info_string)[0].replace('[','').replace(']','').split(','), dtype=float)
    amplitudes = np.array(re.findall(r"Amplitudes tensor\((.*?)\)", info_string)[0].replace('[','').replace(']','').split(','), dtype=float)
    phases = np.array(re.findall(r"Phases tensor\((.*?)\)", info_string)[0].replace('[','').replace(']','').split(','), dtype=float)

#Max Reward: 18527.40012723453, Frequency: tensor([0.6399, 0.6013, 0.2886, 0.3470, 0.7281]), Amplitude: tensor([, , , , ]), Phase: tensor([0.3629, 0.0795, 0.3683, 0.4717, 0.5971])
    while not mj.glfw.glfw.window_should_close(window):
        simstart = data.time

        while (data.time - simstart < 1.0 / 60.0):
            # Update goal positions
            for i in range(num_actuators):
                if i in [0, 1, 2, 3, 4]:
                    data.ctrl[i] = amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * data.time + phases[i])

                #     if data.ctrl[i] < goal_positions[i]:
                #         data.ctrl[i] += 0.3925
                #     else:
                #         data.ctrl[i] -= 0.3925

                # # Add Gaussian noise
                # noise = np.random.normal(0, std_dev)
                # data.ctrl[i] += noise

                #TODO: Cable Tension
            # Log actuator positions to a csv file
            # with open('actuator_positions.csv', 'a', newline='') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(data.ctrl)

            mj.mj_step(model, data)

        #viewport = mj.MjrRect(0, 0, 0, 0)
        #mj.glfw.glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, 1200, 900)

        #mj.mjv_updateScene(model, data, opt, None, cam, 0, scene)
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        mj.glfw.glfw.swap_buffers(window)
        mj.glfw.glfw.poll_events()

    mj.glfw.glfw.terminate()


if __name__ == "__main__":
    main()