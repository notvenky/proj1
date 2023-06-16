import mujoco as mj
import numpy as np
import time
import imgui
import random
from imgui.integrations.glfw import GlfwRenderer


def main():
    max_width = 100
    max_height = 100
    #ctx = mj.GLContext(max_width, max_height)
    #ctx.make_current()

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

    # Initialize a timer for each actuator
    actuator_timers = np.zeros(model.nu)

    # Random delay for each actuator
    actuator_delays = np.random.randint(low=1, high=10, size=model.nu)

    # create context
    imgui.create_context()
    window_impl = GlfwRenderer(window)

    simulation_speed = 0.5
    simulation_speed_ratio = 1

    # Initialize a timer for each actuator
    actuator_timers = np.zeros(model.nu)

    # Random delay for each actuator
    actuator_delays = np.random.randint(low=1, high=10, size=model.nu)

    dxl_id = [0,1,2,3,4]

    ANGLE_RANGES = {
        0: (-1.57, 1.57),
        1: (-3.14, 3.14),
        2: (-1.57, 1.57),
        3: (-3.14, 3.14),
        4: (-1.57, 1.57)
    }

    #mapped_values = {id_value: ANGLE_RANGES[id_value] for id_value in dxl_id}

    # PID control loop variables
    KP = 0.5
    KI = 0.1
    KD = 0.1
    current_positions = np.zeros(model.nu)
    errors = np.zeros(model.nu)
    last_errors = np.zeros(model.nu)
    goal_positions = np.zeros(model.nu)

    while not mj.glfw.glfw.window_should_close(window):
        imgui.new_frame()
        
        # Set window position and size
        imgui.set_next_window_position(50, 50, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(300, 200, imgui.FIRST_USE_EVER)

        imgui.begin("Simulation Speed", True)
        
        # creating a slider for simulation speed
        changed, simulation_speed = imgui.slider_float("speed", simulation_speed, 0.1, 2.0)
        imgui.end()

        imgui.render()
        window_impl.render(imgui.get_draw_data())

        simstart = data.time

        while (data.time - simstart < 1.0/60.0):
            # Count down each actuator's timer
            actuator_timers -= 1

            # Reset the timer and delay for the actuators that reached zero
            mask = actuator_timers <= 0
            actuator_timers[mask] = actuator_delays[mask]
            actuator_delays[mask] = np.random.randint(low=1, high=5, size=sum(mask))

            # Select a random subset of actuators whose timers have reached zero
            active_actuators = np.random.choice(np.where(mask)[0], size=min(5, sum(mask)), replace=False)

            # Generate random action commands for the selected actuators
            # instead of setting a random goal position, we use a PID controller to generate it
            for dxl_id in active_actuators:
                # Read present position (adapt to your simulation environment)
                current_positions[dxl_id] = data.qpos[dxl_id]
                
                # generate a random goal position
                goal_positions[dxl_id] = random.uniform(ANGLE_RANGES[dxl_id][0], ANGLE_RANGES[dxl_id][1])
                
                # PID control
                error = goal_positions[dxl_id] - current_positions[dxl_id]
                errors[dxl_id] += error
                derivative = error - last_errors[dxl_id]
                control = KP * error + KI * errors[dxl_id] + KD * derivative
                last_errors[dxl_id] = error
                
                # apply the control command to the actuator
                data.ctrl[dxl_id] = control

            mj.mj_step(model, data)
            # Sleep for a while to slow down the simulation. Adjust the sleep duration as necessary.
            #time.sleep((1.0/60.0) * simulation_speed_ratio)

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