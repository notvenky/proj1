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

POSITION_THRESHOLD = 10

# PID control loop
try:
    current_positions = {dxl_id: MEAN_POSITION for dxl_id in DXL_ID_LIST}
    errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}
    last_errors = {dxl_id: 0 for dxl_id in DXL_ID_LIST}

    while True:
        # Flag to indicate if all Dynamixels have reached the goal positions
        all_reached_goal = True

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

            # Check if the current position is within a threshold of the goal position
            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_POSITION)
            if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                current_positions[dxl_id] = dxl_present_position
                angle = current_positions[dxl_id]
                error = goal_position - angle

                # Check if the error is within a threshold
                if abs(error) > POSITION_THRESHOLD:
                    all_reached_goal = False

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

        write_counter += 1
        if write_counter >= WRITE_FREQ:
            # Save log data to disk
            with open('log.txt', 'a') as f:
                f.flush()
            write_counter = 0

        # Wait until all Dynamixels reach the goal positions
        if all_reached_goal:
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
