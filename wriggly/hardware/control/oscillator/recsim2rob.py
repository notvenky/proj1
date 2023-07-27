import numpy as np
import time
import re
import cv2
import os
import json
from config import *

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
out = cv2.VideoWriter(f'media_sin/video_{count}.avi', fourcc, 20.0, (640, 480))  # change resolution as needed

amplitude_conversion_factor = 2048 / 3.14
# paste_string = 'Frequency: tensor([0.4916, 0.2262, 0.4490, 0.4511, 0.3306]), Amplitude: tensor([1.5270, 1.4947, 0.8646, 0.8290, 1.4490]), Phase: tensor([0.7578, 2.0704, 0.4936, 5.0827, 1.1826])'
paste_string = 'Frequency: tensor([0.4839, 0.3116, 0.1512, 0.4405, 0.0981]), Amplitude: tensor([1.2890, 2.8330, 0.9312, 2.8816, 0.0447]), Phase: tensor([1.9279, 1.2120, 0.1625, 0.4938, 5.1535])'


COMMAND_FREQUENCY = 1.0
COMMAND_PERIOD = 1.0 / COMMAND_FREQUENCY

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

def oscillate_position(dxl_id, t):
    """
    Oscillates the position of the specified Dynamixel.
    """
    omega = 2 * PI * FREQUENCIES[dxl_id]
    A = AMPLITUDES[dxl_id]
    phi = PHASES[dxl_id]
    
    position = MEAN_POSITION + A * np.sin(omega * t + phi)
    speed = 330

    # Write the position and speed
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_POSITION, int(position))
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_SPEED, speed)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Dynamixel %d is oscillating at position: %d" % (dxl_id, position))



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
        # inside the while True: loop
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1 and ret2 is not None:
            # Concatenate both frames horizontally
            frame = np.concatenate((frame1, frame2), axis=1)
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        current_time = time.time() - start_time
        for dxl_id in DXL_ID_LIST:
            oscillate_position(dxl_id, current_time)
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
