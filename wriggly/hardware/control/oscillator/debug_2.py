import time
import re
from config import *

paste_string = 'MReward: 4403.559757610102, Frequency: tensor([0.8048, 0.4901, 0.4643, 0.2966, 0.3876]), Amplitude: tensor([0.1598, 2.9545, 1.4238, 0.2111, 1.4200]), Phase: tensor([5.5397, 5.5965, 3.2232, 3.2181, 2.4138])'

COMMAND_FREQUENCY = 3
COMMAND_PERIOD = 1.0 / COMMAND_FREQUENCY
TIME_INCREMENT = COMMAND_PERIOD/10

# Function to split string to get values of frequency, amplitude and phase
def split_string(string):
    return re.findall('tensor\((.*?)\)', string)

tensor_values = re.findall('tensor\((.*?)\)', paste_string)

# Function to assign values from tensor_values to frequency, amplitude and phase
def assign_params(split_string):
    frequency = eval(split_string[0])
    amplitude = [round(a * amplitude_conversion_factor) for a in eval(split_string[1])]
    phase = eval(split_string[2])
    frequency[0], frequency[1] = frequency[1], frequency[0]
    amplitude[0], amplitude[1] = amplitude[1], amplitude[0]
    phase[0], phase[1] = phase[1], phase[0]
    return frequency, amplitude, phase

# Function to create a dictionary of frequency, amplitude and phase
def create_dict(keys, values):
    return dict(zip(keys, values))



frequency = eval(tensor_values[0])
amplitude = [round(a * amplitude_conversion_factor) for a in eval(tensor_values[1])]
phase = eval(tensor_values[2])

keys = [22, 21, 20, 12, 11]

FREQUENCIES = dict(zip(keys, frequency))
AMPLITUDES = dict(zip(keys, amplitude))
PHASES = dict(zip(keys, phase))

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

        # Set velocity for all motors simultaneously
        for dxl_id in DXL_ID_LIST:
            velocity = int(330)
            set_motor_velocity(dxl_id, velocity)

        # Send position commands
        for dxl_id in DXL_ID_LIST:
            check_and_issue_next_command(dxl_id, current_time)
        time.sleep(COMMAND_PERIOD)

except KeyboardInterrupt:
    pass

finally:
    time.sleep(0.5)
    for i in range(len(DXL_ID_LIST)):
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_LIST[i], ADDR_PRO_TORQUE_ENABLE, 0)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel %d torque has been successfully disabled" % DXL_ID_LIST[i])

    portHandler.closePort()