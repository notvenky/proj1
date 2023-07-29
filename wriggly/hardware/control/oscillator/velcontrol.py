import numpy as np
import time
import re
import cv2
import os
import json
from config import *
import subprocess

# Create media directory if it doesn't exist
if not os.path.exists('media_sin'):
    os.makedirs('media_sin')

# Initialize count for video name
count = 0
if os.path.exists('sin_count.json'):
    with open('sin_count.json', 'r') as f:
        data = json.load(f)
        count = data['sin_count']

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'media_sin/video_{count}.avi', fourcc, 20.0, (1280, 480))  # adjust size as per two frames

amplitude_conversion_factor = 2048 / 3.14
# paste_string = 'Frequency: tensor([0.4916, 0.2262, 0.4490, 0.4511, 0.3306]), Amplitude: tensor([1.5270, 1.4947, 0.8646, 0.8290, 1.4490]), Phase: tensor([0.7578, 2.0704, 0.4936, 5.0827, 1.1826])'
# paste_string = 'Frequency: tensor([0.4839, 0.3116, 0.1512, 0.4405, 0.0981]), Amplitude: tensor([1.2890, 2.8330, 0.9312, 2.8816, 0.0447]), Phase: tensor([1.9279, 1.2120, 0.1625, 0.4938, 5.1535])'
# bad - paste_string = 'Frequency: tensor([0.4254, 0.453, 0.1042, 0.1972, 0.4794]), Amplitude: tensor([1.0657, 1.8825, 1.4196, 0.289, 0.253]), Phase: tensor([0.6236, 0.3487, 5.1865, 4.6293, 2.9277])'

# paste_string = 'Frequency: tensor([0.4990, 0.2122, 0.4650, 0.4007, 0.3471]), Amplitude: tensor([0.9685, 2.2220, 1.4667, 0.7460, 0.9961]), Phase: tensor([5.4440, 2.9973, 2.8850, 3.7666, 5.2267])'
# # vid23
paste_string = 'Frequency: tensor([0.0389, 0.1938, 0.1731, 0.0155, 0.3725]), Amplitude: tensor([0.6815, 3.0101, 1.2761, 0.6794, 0.7488]), Phase: tensor([5.9447, 0.7377, 1.0380, 1.4478, 0.3299])'
# first result from improved sim


COMMAND_FREQUENCY = 3.0
COMMAND_PERIOD = 1.0 / COMMAND_FREQUENCY
TIME_INCREMENT = COMMAND_PERIOD/10

tensor_values = re.findall('tensor\((.*?)\)', paste_string)
frequency = eval(tensor_values[0])
amplitude = [round(a * amplitude_conversion_factor) for a in eval(tensor_values[1])]
phase = eval(tensor_values[2])

frequency[-1], frequency[-2] = frequency[-2], frequency[-1]
amplitude[-1], amplitude[-2] = amplitude[-2], amplitude[-1]
phase[-1], phase[-2] = phase[-2], phase[-1]

keys = [11, 12, 20, 21, 22]
FREQUENCIES = dict(zip(keys, frequency))
AMPLITUDES = dict(zip(keys, amplitude))
PHASES = dict(zip(keys, phase))

print('FREQUENCIES =', FREQUENCIES)
print('AMPLITUDES =', AMPLITUDES)
print('PHASES =', PHASES)


# # Define frequencies, amplitudes, and phases for each Dynamixel
# FREQUENCIES = {11: 0.7761, 12: 0.591, 20: 0.9764, 21: 0.5148, 22: 0.6341}
# AMPLITUDES = {11: 877, 12: 985, 20: 802, 21: 1309, 22: 862}
# PHASES = {11: 1.527, 12: 3.419, 20: 5.7299, 21: 3.9914, 22: 1.4527}


# Constants
PI = np.pi

# Adjust these parameters for smoother motion
COMMAND_FREQUENCY = 5.0  # Increase this value for smoother motion
COMMAND_PERIOD = 1.0 / COMMAND_FREQUENCY
TIME_INCREMENT = COMMAND_PERIOD / 10

# Function to set velocity for the Dynamixel motors
def set_motor_velocity(dxl_id, velocity):
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_SPEED, velocity)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))

def oscillate_position(dxl_id, t):
    """
    Oscillates the position of the specified Dynamixel.
    """
    omega = 2 * PI * FREQUENCIES[dxl_id]
    A = AMPLITUDES[dxl_id]
    phi = PHASES[dxl_id]

    position = MEAN_POSITION + A * np.sin(omega * t + phi)
    
    # Write the position
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_POSITION, int(position))
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel %d is oscillating at position: %d" % (dxl_id, position))

def is_moving(dxl_id):
    """
    Returns True if the specified Dynamixel is moving.
    """
    dxl_present_moving, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(portHandler, dxl_id, ADDR_MOVING)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    return dxl_present_moving != 0

def check_and_issue_next_command(dxl_id, t):
    """
    If the specified Dynamixel is not moving, this issues the next command.
    """
    if not is_moving(dxl_id):
        oscillate_position(dxl_id, t)




    # # Write goal speed
    
    # if dxl_comm_result != COMM_SUCCESS:
    #     print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    # elif dxl_error != 0:
    #     print("%s" % packetHandler.getRxPacketError(dxl_error))
    # else:
    #     print("Speed of Dynamixel %d has been changed to: %d" % (dxl_id, speed))

start_time = time.time()

try:
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1 and ret2:
            frame = np.concatenate((frame1, frame2), axis=1)
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        current_time = time.time() - start_time

        # Set velocity for all motors simultaneously
        for dxl_id in DXL_ID_LIST:
            velocity = int(1023)  # Adjust the velocity value as needed
            set_motor_velocity(dxl_id, velocity)

        # Send position commands
        for dxl_id in DXL_ID_LIST:
            check_and_issue_next_command(dxl_id, current_time)
        time.sleep(COMMAND_PERIOD)

except KeyboardInterrupt:
    pass

finally:
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()
    # Increment the count and write it back to the file
    count += 1
    with open('sin_count.json', 'w') as f:
        json.dump({'sin_count': count}, f)

    # Add the video compression functionality here
    original_video_file = f'media_sin/video_{count - 1}.avi'
    compressed_video_file = f'media_sin/video_{count - 1}_compressed.mp4'
    command = f"ffmpeg -i {original_video_file} -vcodec libx264 -crf 23 {compressed_video_file}"

    if os.path.exists(original_video_file):
        command = f"ffmpeg -i {original_video_file} -vcodec libx264 -crf 23 {compressed_video_file}"
        subprocess.call(command, shell=True)

        # Remove the original uncompressed video
        os.remove(original_video_file)
    else:
        print(f"File '{original_video_file}' does not exist.")