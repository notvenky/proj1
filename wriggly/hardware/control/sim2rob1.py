import time
import numpy as np
import csv
import os
from datetime import datetime
from pathlib import Path
from dynamixel_sdk import *

# Control table address
ADDR_PRO_TORQUE_ENABLE = 64                  # Address for enabling the torque
ADDR_PRO_GOAL_POSITION = 116                 # Address for goal position
ADDR_PRO_PRESENT_POSITION = 132              # Address for present position
ADDR_PRO_GOAL_CURRENT = 102                  # Address for goal current
ADDR_PRO_PRESENT_CURRENT = 126               # Address for present current
ADDR_PRO_GOAL_SPEED = 104                    # Address for speed
ADDR_PRO_PRESENT_SPEED = 128
ADDR_PRO_VELOCITY_TRAJECTORY = 136
ADDR_PRO_POSITION_TRAJECTORY = 140

# Define angle ranges for each Dynamixel
ANGLE_RANGES = {
    11: (1024, 3072),
    12: (0, 4095),
    20: (1024, 3072),
    21: (1024, 3072),
    22: (0, 4095)
}


MEAN_POSITION = 2048                          # Mean, starting position for all dynamixels 

dxl_goal_position = [0,4095]
JOYSTICK_THRESHOLD = 0.8
DXL_MOVING_STATUS_THRESHOLD = 20

# Protocol version
PROTOCOL_VERSION = 2.0                        # 2.0 for XM430-W210

# Default setting
DXL_ID_LIST = [11, 12, 20, 21, 22]            # Dynamixel ID list
BAUDRATE = 1000000                            # Dynamixel communication baudrate
DEVICENAME = '/dev/ttyUSB0'                   # U2D2 USB-to-Serial converter device name

# Initialize PortHandler instance
portHandler = PortHandler(DEVICENAME)
print(DXL_ID_LIST)
# Initialize PacketHandler instance
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    quit()

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    quit()


for i in range(len(DXL_ID_LIST)):
    # Enable Dynamixel Torque
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_LIST[i], ADDR_PRO_TORQUE_ENABLE, 1)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel %d has been successfully connected" % DXL_ID_LIST[i])

    # Set initial position
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID_LIST[i], ADDR_PRO_GOAL_POSITION, MEAN_POSITION)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel %d has been successfully set to initial position" % DXL_ID_LIST[i])

# Oscillator parameters
AMPLITUDES = [0.5] * 5  # replace with your values
FREQUENCIES = [1] * 5  # replace with your values
PHASES = [1, 2, 3 ,4, 5]  # replace with your values

# Initialize time
start_time = time.time()

# Command frequency
COMMAND_FREQUENCY = 1.0
COMMAND_PERIOD = 1.0 / COMMAND_FREQUENCY

# Initialize loggers
Path("logs/command_logs").mkdir(parents=True, exist_ok=True)
Path("logs/image_logs").mkdir(parents=True, exist_ok=True)
Path("logs/dynamixel_logs").mkdir(parents=True, exist_ok=True)
command_logger = csv.writer(open(f"logs/command_logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv", "w"))
image_logger = csv.writer(open(f"logs/image_logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv", "w"))
dynamixel_logger = csv.writer(open(f"logs/dynamixel_logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv", "w"))

# Run oscillators
while True:
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time > 3600:  # Check if an hour has passed
        command_logger = csv.writer(open(f"logs/command_logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv", "w"))
        image_logger = csv.writer(open(f"logs/image_logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv", "w"))
        dynamixel_logger = csv.writer(open(f"logs/dynamixel_logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv", "w"))
        start_time = current_time

    goal_positions = []
    for i in range(len(DXL_ID_LIST)):
        A = AMPLITUDES[i]
        freq = FREQUENCIES[i]
        phi = PHASES[i]
        # get the corresponding range
        min_range, max_range = ANGLE_RANGES[DXL_ID_LIST[i]]
        mid_range = (max_range - min_range) / 2
        goal_position = int(A * np.sin(2 * np.pi * freq * elapsed_time + phi) * mid_range + mid_range + min_range)
        
        # make sure goal_position is within the defined range
        goal_position = max(min_range, min(max_range, goal_position))
        
        goal_positions.append(goal_position)

        # Write goal position
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID_LIST[i], ADDR_PRO_GOAL_POSITION, goal_position)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))

    # Log commands
    command_logger.writerow([(dxl, pos) for dxl, pos in zip(DXL_ID_LIST, goal_positions)])

    # TODO: Add image logging code here
    # image_logger.writerow(...)

    for i in range(len(DXL_ID_LIST)):
        # Read and log current position, velocity, and torque
        position = packetHandler.read4ByteTxRx(portHandler, DXL_ID_LIST[i], ADDR_PRO_PRESENT_POSITION)
        velocity = packetHandler.read4ByteTxRx(portHandler, DXL_ID_LIST[i], ADDR_PRO_PRESENT_SPEED)
        torque = packetHandler.read2ByteTxRx(portHandler, DXL_ID_LIST[i], ADDR_PRO_PRESENT_CURRENT)
        dynamixel_logger.writerow([DXL_ID_LIST[i], position[0], velocity[0], torque[0]])

    # Sleep for the remainder of the period
    time.sleep(max(0, COMMAND_PERIOD - (time.time() - current_time)))

