import random
import time
import keyboard
import math
from config import *
import datetime

WRITE_FREQ = 2
write_counter = 0

# ...

# PID control loop
try:
    current_positions = {dxl_id: MEAN_POSITION for dxl_id in DXL_ID_LIST}
    errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
    last_errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
    while True:
        # Randomly select the number of Dynamixels to actuate
        num_actuators = random.choice([1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
        # Randomly select the Dynamixel IDs to actuate
        actuator_ids = random.sample(DXL_ID_LIST, num_actuators)

        # Generate random frequencies, amplitudes, and phases for each actuator
        frequencies = [random.uniform(0.1, 2.0) for _ in range(num_actuators)]
        amplitudes = [random.uniform(10, 100) for _ in range(num_actuators)]
        phases = [random.uniform(0, 2 * math.pi) for _ in range(num_actuators)]

        # Write goal position, torque, and speed for selected Dynamixels
        for dxl_id, frequency, amplitude, phase in zip(actuator_ids, frequencies, amplitudes, phases):
            goal_position = MEAN_POSITION + amplitude * math.sin(2 * math.pi * frequency * time.time() + phase)

            # Apply PID control to set goal current
            error = goal_position - current_positions[dxl_id]
            errors[dxl_id] += error
            derivative = error - last_errors[dxl_id]
            goal_current = KP * error + KI * errors[dxl_id] + KD * derivative
            last_errors[dxl_id] = error

            # Write goal current, torque, and speed
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_CURRENT, int(goal_current))
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_SPEED, speed)
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_POSITION, goal_position)
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_CURRENT, torque)

            write_counter += 1

            # Write to log file every WRITE_FREQ iterations
            if write_counter >= WRITE_FREQ:
                # Reset the write counter
                write_counter = 0

                # Write to log file
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                with open('log.txt', 'a') as f:
                    f.write('{},{},{},{},{},{}\n'.format(timestamp, [actuator_ids], [directions], speed, torque, angle))

        # Read present position and current for all Dynamixels
        for dxl_id in DXL_ID_LIST:
            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_POSITION)
            if dxl_comm_result == COMM_SUCCESS:
                current_positions[dxl_id] = dxl_present_position
                dxl_present_current, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_CURRENT)
            if dxl_comm_result == COMM_SUCCESS:
                pass

        # Wait for keyboard interrupt or random time interval
        try:
            keyboard.wait('esc', suppress=True)
            break
        except:
            pass

        time.sleep(random.uniform(0.1, 0.5))

except KeyboardInterrupt:
    pass

# ...
