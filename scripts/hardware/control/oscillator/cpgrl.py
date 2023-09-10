import numpy as np
import time
import re
from config import *


amplitude_conversion_factor = 2047 / 3.14

# paste_string = 'Max Reward: -1.5619297584574523, Frequency: tensor([0.9339, 0.9652, 0.9615, 0.2451, 0.7739]), Amplitude: tensor([1.2595, 2.4288, 0.2912, 2.6775, 1.5012]), Phase: tensor([0.8567, 5.2536, 2.4380, 5.6132, 1.8356])'
paste_string = 'Max Reward: 1.1364636022338537, Frequency: tensor([0.6424, 0.7251, 0.6883, 0.6823, 0.9268]), Amplitude: tensor([1.1405, 3.0430, 0.5009, 1.8479, 1.4164]), Phase: tensor([0.4303, 4.1198, 4.8462, 3.6321, 4.2793])'
# paste_string = 'MReward: 4403.559757610102, Frequency: tensor([0.8048, 0.4901, 0.4643, 0.2966, 0.3876]), Amplitude: tensor([0.1598, 2.9545, 1.4238, 0.2111, 1.4200]), Phase: tensor([5.5397, 5.5965, 3.2232, 3.2181, 2.4138])'
# paste_string = 'Max Reward: -355.4512733042265, Frequency: tensor([0.6724, 2.5047, 3.5192, 1.8001, 2.3805]), Amplitude: tensor([0.9144, 1.6127, 1.2397, 2.0497, 0.2821]), Phase: tensor([3.2062, 2.8402, 3.6439, 4.4170, 5.0696])'
# paste_string = 'Reward: 2981.162248845091, Frequency: tensor([3.5039, 2.9618, 3.6171, 2.8623, 0.6255]), Amplitude: tensor([0.3252, 0.4024, 1.3133, 2.6606, 0.4866]), Phase: tensor([2.3109, 0.6266, 1.9524, 1.8429, 1.3122])'
# paste_string = 'Reward: 409.2699374327469, Frequency: tensor([0.6592, 0.9613, 0.3067, 0.0929, 0.6953]), Amplitude: tensor([0.4718, 2.3431, 0.3116, 0.9953, 0.3291]), Phase: tensor([3.2223, 2.5374, 0.8537, 3.7921, 2.2738])'
paste_string = 'Reward: 1.1791573039139247, Frequency: tensor([0.0965, 0.8480, 0.0579, 0.4538, 0.2606]), Amplitude: tensor([0.8267, 2.7854, 1.5259, 1.7308, 0.2948]), Phase: tensor([4.6115, 3.6546, 6.1726, 0.2648, 0.8403])'




MU_VALUES = {
    22: 2048,
    21: 1024,
    20: 1024,
    12: 2048,
    11: 1024
}


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

CURRENT_AMPLITUDES = dict(zip(keys, [0] * len(keys)))
CURRENT_AMPLITUDE_DOT = dict(zip(keys, [0] * len(keys)))
dt = 0.001


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

def rk4_step(y, dydt, dt, f):
    k1 = dt * f(y, dydt)
    k2 = dt * f(y + 0.5 * dydt * dt, dydt + 0.5 * k1)
    k3 = dt * f(y + 0.5 * dydt * dt, dydt + 0.5 * k2)
    k4 = dt * f(y + dydt * dt, dydt + k3)

    dydt_new = dydt + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    y_new = y + dydt * dt

    return y_new, dydt_new

def differential_equation(y, dydt, dxl_id):
    mu = MU_VALUES[dxl_id]  # Fetch mu for the specified actuator
    a = 50
    d2ydt2 = a * (a / 4 * (mu - y) - dydt)
    return d2ydt2


def oscillate_position(dxl_id, t):
    """
    Oscillates the position of the specified Dynamixel using RK4 integration method.
    """
    global CURRENT_AMPLITUDES, CURRENT_AMPLITUDE_DOT

    for _ in range(int(1 / dt)):
        CURRENT_AMPLITUDES[dxl_id], CURRENT_AMPLITUDE_DOT[dxl_id] = rk4_step(
            CURRENT_AMPLITUDES[dxl_id], 
            CURRENT_AMPLITUDE_DOT[dxl_id], 
            dt, 
            lambda y, dydt: differential_equation(y, dydt, dxl_id)  # lambda function to pass the additional dxl_id argument
        )

    
    position = MEAN_POSITION + CURRENT_AMPLITUDES[dxl_id] * np.sin(2 * PI * FREQUENCIES[dxl_id] * t + PHASES[dxl_id])
    
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
    # if not is_moving(dxl_id):
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