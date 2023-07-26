import numpy as np
import time
import re
from config import *

amplitude_conversion_factor = 2048 / 3.14
# paste_string = 'Frequency: tensor([0.4303, 0.4154, 0.4517, 0.3578, 0.2295]), Amplitude: tensor([1.3330, 1.2507, 0.8577, 2.2365, 0.8378]), Phase: tensor([2.1703, 1.8762, 0.6844, 6.2216, 1.6259])'
# paste_string = 'Frequency: tensor([0.8931, 0.5379, 0.8978, 0.8667, 0.5517]), Amplitude: tensor([0.7955, 0.1553, 1.2340, 1.6642, 1.2335]), Phase: tensor([0.7258, 3.1937, 3.4631, 3.5404, 4.7361])'
paste_string = 'Frequency: tensor([0.4916, 0.2262, 0.4490, 0.4511, 0.3306]), Amplitude: tensor([1.5270, 1.4947, 0.8646, 0.8290, 1.4490]), Phase: tensor([0.7578, 2.0704, 0.4936, 5.0827, 1.1826])'



tensor_values = re.findall('tensor\((.*?)\)', paste_string)
frequency = eval(tensor_values[0])
amplitude = [round(a * amplitude_conversion_factor) for a in eval(tensor_values[1])]
phase = eval(tensor_values[2])

# Swap second last and last values
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
PI = 3.1416

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


start_time = time.time()
while True:
    current_time = time.time() - start_time
    for dxl_id in DXL_ID_LIST:
        oscillate_position(dxl_id, current_time)
    time.sleep(2)
