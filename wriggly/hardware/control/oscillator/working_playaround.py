import numpy as np
import time
import re
import cv2
import os
import json
from config import *
import subprocess

if not os.path.exists('media_sin'):
    os.makedirs('media_sin')

count = 0
if os.path.exists('sin_count.json'):
    with open('sin_count.json', 'r') as f:
        data = json.load(f)
        count = data['sin_count']

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'media_sin/video_{count}.avi', fourcc, 20.0, (1280, 480))

amplitude_conversion_factor = 2048 / 3.14

paste_string = 'MReward: 4403.559757610102, Frequency: tensor([0.8048, 0.4901, 0.4643, 0.2966, 0.3876]), Amplitude: tensor([0.1598, 2.9545, 1.4238, 0.2111, 1.4200]), Phase: tensor([5.5397, 5.5965, 3.2232, 3.2181, 2.4138])'

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


# Constants
PI = np.pi

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
            velocity = int(330)  # Adjust the velocity value as needed
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

        os.remove(original_video_file)
    else:
        print(f"File '{original_video_file}' does not exist.")