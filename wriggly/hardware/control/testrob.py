import numpy as np
import time
from dynamixel_sdk import *
import re

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



amplitude_conversion_factor = 2048 / 3.14
# paste_string = 'Frequency: tensor([0.4303, 0.4154, 0.4517, 0.3578, 0.2295]), Amplitude: tensor([1.3330, 1.2507, 0.8577, 2.2365, 0.8378]), Phase: tensor([2.1703, 1.8762, 0.6844, 6.2216, 1.6259])'
paste_string = 'Frequency: tensor([0.8931, 0.5379, 0.8978, 0.8667, 0.5517]), Amplitude: tensor([0.7955, 0.1553, 1.2340, 1.6642, 1.2335]), Phase: tensor([0.7258, 3.1937, 3.4631, 3.5404, 4.7361])'




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
    time.sleep(0.01)
