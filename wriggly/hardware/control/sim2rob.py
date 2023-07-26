'''
import time
import numpy as np
import csv
import os
from datetime import datetime
from pathlib import Path

# Oscillator parameters
AMPLITUDES = [0.5] * 5  # replace with your values
FREQUENCIES = [0.5] * 5  # replace with your values
PHASES = [0.0] * 5  # replace with your values

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
        # Calculate goal position
        A = AMPLITUDES[i]
        freq = FREQUENCIES[i]
        phi = PHASES[i]
        goal_position = int(A * np.sin(2 * np.pi * freq * elapsed_time + phi) * 2048 + 2048)
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


# Add the following imports
import cv2

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Initialize robot position
robot_position = None

# Start video capture loop
while True:
    # Capture frame
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Find contours in the foreground
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for contour in contours:
        # Compute the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compute the center of the bounding box
        center = (int(x + w / 2), int(y + h / 2))

        # If this is the first frame or the center is close to the previous position, update robot position
        if robot_position is None or abs(center[0] - robot_position[0]) + abs(center[1] - robot_position[1]) < 30:
            robot_position = center

    # Draw the robot position on the frame
    if robot_position is not None:
        cv2.circle(frame, robot_position, 5, (0, 0, 255), -1)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()


'''

import cv2
import os
import json
import random
import time
import keyboard
import datetime
import numpy as np
import csv
import subprocess
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


# Step 1: Define oscillators for each actuator
A_list = #[amplitude_1, amplitude_2, amplitude_3, amplitude_4, amplitude_5]  # Replace amplitude_1, ..., amplitude_5 with actual values
freq_list = #[frequency_1, frequency_2, frequency_3, frequency_4, frequency_5]  # Replace frequency_1, ..., frequency_5 with actual values
phi_list = #[phase_1, phase_2, phase_3, phase_4, phase_5]  # Replace phase_1, ..., phase_5 with actual values

def calculate_goal_position(time):
    goal_positions = []
    for i in range(len(DXL_ID_LIST)):
        goal_position = A_list[i] * np.sin(2 * np.pi * freq_list[i] * time + phi_list[i])
        goal_positions.append(int(MEAN_POSITION + goal_position))
    return goal_positions


# Step 2: Function to send sinusoidal commands to dynamixel
def send_sinusoidal_commands(portHandler, packetHandler, log_file):
    frequency = 1.0  # Set the initial frequency to 1 Hertz
    time_interval = 1.0 / frequency

    with open(log_file, 'w', newline='') as csvfile:
        command_writer = csv.writer(csvfile)
        while True:
            start_time = time.time()
            current_time = 0.0

            while current_time < time_interval:
                goal_positions = calculate_goal_position(current_time)
                for i in range(len(DXL_ID_LIST)):
                    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
                        portHandler, DXL_ID_LIST[i], ADDR_PRO_GOAL_POSITION, goal_positions[i]
                    )
                    if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                        command_writer.writerow(["dxl_no: %d" % DXL_ID_LIST[i], "goal_position: %d" % goal_positions[i]])
                    current_time = time.time() - start_time

            # Increase frequency for next iteration (optional)
            frequency += 1
            time_interval = 1.0 / frequency


# Step 3: Function to log dynamixel position, velocity, and torque data
def log_dynamixel_data(portHandler, packetHandler, log_folder):
    os.makedirs(log_folder, exist_ok=True)

    for i in range(len(DXL_ID_LIST)):
        log_file_path = os.path.join(log_folder, "dynamixel_%d_log.csv" % DXL_ID_LIST[i])
        with open(log_file_path, 'w', newline='') as csvfile:
            data_writer = csv.writer(csvfile)
            data_writer.writerow(["Time (s)", "Position", "Velocity", "Torque"])

            while True:
                dxl_present_position, _, dxl_present_speed, dxl_present_current, _ = packetHandler.read4ByteTxRx(
                    portHandler, DXL_ID_LIST[i], ADDR_PRO_PRESENT_POSITION
                )
                dxl_present_position = dxl_present_position & 0xFFF  # Convert to 12-bit value
                dxl_present_speed = dxl_present_speed & 0xFFF  # Convert to 12-bit value
                dxl_present_current = dxl_present_current & 0xFFF  # Convert to 12-bit value

                data_writer.writerow([time.time(), dxl_present_position, dxl_present_speed, dxl_present_current])
                time.sleep(1)  # Log data every 1 second


'''
Above, I have initialized and set the 5 actuators of my robot to their initial posiitons
Next Steps:-
1. I need to have an oscillator for each of the 5 actuator
defined by goal_position = A * sin (2 * np.pi * freq * time + phi)
Where A is amplitude, freq represents the frequency of the oscillator,
and phi represents the phase of the oscillator and time is continious.
They will be passed in as 3 lists with 5 elements in each, representing their 
oscillators. 

2. From the home posiiton, the oscillator should pass commands to the robot which should all be sinusoidal in nature.
The control can be purely position-based. However, ensure that the velocity is high and the robot has swift motions 

3. Each actuator's oscillator must run independantly. Everytime a goal position is given
to a dynamixel, it must be checked that the dynamixel has reached its goal position and only then 
should the next goal position should be passed to the dynamixel using that corresponding time

4. Make a separate variable to represent the frequency at which coommands must be passed to the dynamixel
We can start with 1 Hertz and gradually increase it if possible

5. I need three loggers for this (in csv files), and also a separate log folder for each of the following 3
a) The commands being Passed to the Dynamixel: Each time a goal_position is being sent to the 5 dynamixels, it must be 
documented in this text file. The format must be
(dxl_no: goal_position1), (dxl_no: goal_position2), ...

b) Image Based: Since my robot is comprised of individual segments, I have separated them visually by using gradiented tape for each segment
In order to keep track of the rotations, I need a well-documented log of RGB readings of my robot. The robot is placed on a light brown surface
and can be visually well-captured. In this log, I need to have a detailed log of the colours visible to the bird-view camera through the runtime.
Similar to last time, 

c) Dynamixel Position, Velocity and Torque Based: In this log, for each dynamixel, I need to log the current position, velocity and 
the torque of the dynamixel
At every run, the logs must be saved as new files. If the run lasts longer than an hour, a new file must be created for each hour of 
runtime.

6. 
'''