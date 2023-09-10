import numpy as np
import time
import re
from config import *


amplitude_conversion_factor = 2047 / 3.14

# paste_string = 'Max Reward: -2.749407358693545, Frequency: tensor([0.2970, 1.5426, 0.2534, 0.5384, 3.4290]), Amplitude: tensor([0.5172, 2.8658, 1.1718, 1.2767, 1.0078]), Phase: tensor([0.8182, 1.6443, 3.7484, 5.9257, 5.8324])'
# paste_string = 'Max Reward: -1.5619297584574523, Frequency: tensor([0.9339, 0.9652, 0.9615, 0.2451, 0.7739]), Amplitude: tensor([1.2595, 2.4288, 0.2912, 2.6775, 1.5012]), Phase: tensor([0.8567, 5.2536, 2.4380, 5.6132, 1.8356])'
paste_string = 'Max Reward: 0.30252035049343756, Frequency: tensor([0.5106, 0.2453, 0.5614, 0.1912, 0.3186]), Amplitude: tensor([0.4815, 1.5459, 1.5448, 2.9111, 1.3083]), Phase: tensor([4.5897, 0.3586, 2.3504, 2.9307, 0.2416])'

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


PI = np.pi

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



start_time = time.time()

try:
    while True:
        current_time = time.time() - start_time
        for dxl_id in DXL_ID_LIST:
            check_and_issue_next_command(dxl_id, current_time)
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