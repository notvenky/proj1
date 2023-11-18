import numpy as np
import time
import re
from config import *


amplitude_conversion_factor = 2048 / 3.14

paste_string = 'Max Reward: -355.4512733042265, Frequency: tensor([0.6724, 2.5047, 3.5192, 1.8001, 2.3805]), Amplitude: tensor([0.9144, 1.6127, 1.2397, 2.0497, 0.2821]), Phase: tensor([3.2062, 2.8402, 3.6439, 4.4170, 5.0696])'

COMMAND_FREQUENCY = 3
COMMAND_PERIOD = 1.0 / COMMAND_FREQUENCY
TIME_INCREMENT = COMMAND_PERIOD/10

tensor_values = re.findall('tensor\((.*?)\)', paste_string)
frequency = eval(tensor_values[0])
amplitude = [round(a * amplitude_conversion_factor) for a in eval(tensor_values[1])]
phase = eval(tensor_values[2])

frequency[0], frequency[1] = frequency[1], frequency[0]
amplitude[0], amplitude[1] = amplitude[1], amplitude[0]
phase[0], phase[1] = phase[1], phase[0]

keys = [22, 21, 20, 12, 11]


FREQUENCIES = dict(zip(keys, frequency))
AMPLITUDES = dict(zip(keys, amplitude))
PHASES = dict(zip(keys, phase))

print('FREQUENCIES =', FREQUENCIES)
print('AMPLITUDES =', AMPLITUDES)
print('PHASES =', PHASES)


# Constants
PI = np.pi

# Function to set velocity for the Dynamixel motors
def set_motor_velocity(dxl_id, velocity):
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_SPEED, velocity)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))


def oscillate_position(dxl_id, t, j):
    """
    Oscillates the position of the specified Dynamixel.
    """
    omega = 2 * PI * FREQUENCIES[dxl_id]
    A = AMPLITUDES[dxl_id]
    phi = PHASES[dxl_id]
    position = MEAN_POSITION + A * np.sin(omega * t + (j-1) * phi)
    
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

def check_and_issue_next_command(dxl_id, t, j):
    """
    If the specified Dynamixel is not moving, this issues the next command.
    """
    if not is_moving(dxl_id):
        oscillate_position(dxl_id, t, j)




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
        current_time = time.time() - start_time

        # Set velocity for all motors simultaneously
        for dxl_id in DXL_ID_LIST:
            velocity = int(330)  # Adjust the velocity value as needed
            set_motor_velocity(dxl_id, velocity)

        # Send position commands
        for idx, dxl_id in enumerate(DXL_ID_LIST, start=1):
            check_and_issue_next_command(dxl_id, current_time, idx)
        time.sleep(COMMAND_PERIOD)

except KeyboardInterrupt:
    pass

finally:
    for i in range(len(DXL_ID_LIST)):
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_LIST[i], ADDR_PRO_TORQUE_ENABLE, 0)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel %d torque has been successfully disabled" % DXL_ID_LIST[i])

    portHandler.closePort()