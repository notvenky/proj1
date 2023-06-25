import random
import numpy as np
import mujoco as mj


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
    #xml_path = "/home/venky/robel_dev/robel/robel-scenes/dkitty/kitty-v2.1.xml"
    #xml_path = "/home/venky/Desktop/CILVR/quadruped/unitree/scene_torque.xml"


    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    # Initialize goal positions
    num_actuators = model.nu
    goal_positions = [random.uniform(-1.57, 1.57) for _ in range(num_actuators)]
    print(num_actuators)

    std_dev = 0.02  # Standard deviation of the Gaussian noise

    while not mj.glfw.glfw.window_should_close(window):
        simstart = data.time

        while (data.time - simstart < 1.0 / 60.0):
            # Update goal positions
            for i in range(num_actuators):
                # Check if the actuator has reached its goal position
                if abs(data.ctrl[i] - goal_positions[i]) < 0.01:
                    # Generate a new random goal position
                    goal_positions[i] = random.uniform(-1.57, 1.57)

                # Move the actuator towards the goal position
                if data.ctrl[i] < goal_positions[i]:
                    data.ctrl[i] += 0.3925
                else:
                    data.ctrl[i] -= 0.3925

                # Add Gaussian noise
                noise = np.random.normal(0, std_dev)
                data.ctrl[i] += noise

                #TODO: Cable Tension


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