import random
import time
import keyboard
from config import *
import datetime
import numpy as np

# Constants for PID control
KP = 2
KI = 0.1
KD = 0.1

WRITE_FREQ = 2
write_counter = 0

# Open log file
with open('log.txt', 'a') as f:
    f.write('Timestamp,Dynamixel ID,Direction,Speed,Torque,Angle\n')

# PID control loop
try:
    current_positions = {dxl_id: MEAN_POSITION for dxl_id in DXL_ID_LIST}
    errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
    last_errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}

    while True:
        goal_positions = {}  # Store goal positions for each Dynamixel

        # Set goal positions for all Dynamixels
        for dxl_id in DXL_ID_LIST:
            if dxl_id in [11, 20, 22]:
                phase = -1
            else:
                phase = 1

            goal_position = random.randint(*ANGLE_RANGES[dxl_id])
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_POSITION, goal_position)
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                print("Failed to set goal position for Dynamixel ID %d" % dxl_id)
            else:
                print("Dynamixel ID %d goal position set to: %d" % (dxl_id, goal_position))

            goal_positions[dxl_id] = goal_position  # Store goal position

        # Wait for Dynamixels to reach goal positions
        all_reached_goal = False
        while not all_reached_goal:
            all_reached_goal = True  # Assume all Dynamixels have reached their goal positions

            # Check current positions and update errors
            for dxl_id in DXL_ID_LIST:
                dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_POSITION)
                if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                    current_positions[dxl_id] = dxl_present_position
                    angle = current_positions[dxl_id]
                    goal_position = goal_positions[dxl_id]
                    error = goal_position - angle

                    # PID control
                    p = KP * error
                    i = KI * (errors[dxl_id] + error)
                    d = KD * (error - last_errors[dxl_id])

                    speed = int(p + i + d)
                    speed = np.clip(speed, -1023, 1023)

                    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_SPEED, speed)
                    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                        print("Failed to set goal speed for Dynamixel ID %d" % dxl_id)
                    else:
                        print("Dynamixel ID %d goal speed set to: %d" % (dxl_id, speed))

                    # Log data
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    direction = "CW" if phase == 1 else "CCW"
                    torque = 0  # You may need to read and log the actual torque if available
                    log_data = f"{timestamp},{dxl_id},{direction},{speed},{torque},{angle}\n"
                    with open('log.txt', 'a') as f:
                        f.write(log_data)

                    # Update error and last error
                    errors[dxl_id] += error
                    last_errors[dxl_id] = error

                    # Check if goal position is reached
                    if abs(error) > POSITION_THRESHOLD:
                        all_reached_goal = False

            time.sleep(0.1)

        write_counter += 1
        if write_counter >= WRITE_FREQ:
            # Save log data to disk
            with open('log.txt', 'a') as f:
                f.flush()
            write_counter = 0

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Keyboard interrupt detected. Exiting...")

# Disable Dynamixel torque
for dxl_id in DXL_ID_LIST:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_PRO_TORQUE_ENABLE, 0)
    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
        print("Failed to disable torque for Dynamixel ID %d" % dxl_id)
    else:
        print("Dynamixel ID %d torque has been successfully disabled" % dxl_id)

# Close port
portHandler.closePort()
