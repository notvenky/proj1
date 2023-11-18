import numpy as np
import time
import re
from dynamixel_sdk import *

# Control table address and constants
ADDR_PRO_TORQUE_ENABLE = 64
ADDR_PRO_GOAL_SPEED = 104
ADDR_PRO_PRESENT_POSITION = 132
ADDR_OPERATING_MODE = 11
VELOCITY_CONTROL = 1
PROTOCOL_VERSION = 2.0
DXL_ID_LIST = [11, 12, 20, 21, 22]
BAUDRATE = 1000000
DEVICENAME = '/dev/ttyUSB0'
TIME_INCREMENT = 0.1
PI = np.pi
MEAN_POSITION = 2048
AMPLITUDE_CONVERSION_FACTOR = 2048 / 3.14

class PID:
    def __init__(self, P=0.2, I=0.0, D=0.0):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

def desired_velocity(dxl_id, t, FREQUENCIES, AMPLITUDES, PHASES):
    """
    Calculate the desired velocity for the specified Dynamixel based on current time.
    """
    omega = 2 * PI * FREQUENCIES[dxl_id]
    A = AMPLITUDES[dxl_id]
    phi = PHASES[dxl_id]
    velocity = A * omega * np.cos(omega * t + phi)
    return velocity

def main():
    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    if not portHandler.openPort():
        print("Failed to open the port")
        return

    if not portHandler.setBaudRate(BAUDRATE):
        print("Failed to change the baudrate")
        return

    for dxl_id in DXL_ID_LIST:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_PRO_TORQUE_ENABLE, 1)
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OPERATING_MODE, VELOCITY_CONTROL)

    paste_string = 'Frequency: tensor([0.5, 0.2, 0.5, 0.2, 0.5]), Amplitude: tensor([1, 2, 1, 2, 1]), Phase: tensor([0, 3.14, 0, 1.57, 1.57])'
    tensor_values = re.findall('tensor\((.*?)\)', paste_string)
    frequency = eval(tensor_values[0])
    amplitude = [round(a * AMPLITUDE_CONVERSION_FACTOR) for a in eval(tensor_values[1])]
    phase = eval(tensor_values[2])
    keys = [22, 21, 20, 12, 11]

    FREQUENCIES = dict(zip(keys, frequency))
    AMPLITUDES = dict(zip(keys, amplitude))
    PHASES = dict(zip(keys, phase))

    pid_controllers = {dxl_id: PID() for dxl_id in DXL_ID_LIST}

    try:
        start_time = time.time()
        while True:
            elapsed_time = time.time() - start_time
            for dxl_id in DXL_ID_LIST:
                velocity_command = desired_velocity(dxl_id, elapsed_time, FREQUENCIES, AMPLITUDES, PHASES)
                current_position, _, _ = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_POSITION)
                position_error = velocity_command - current_position
                adjusted_velocity = pid_controllers[dxl_id].update(position_error, TIME_INCREMENT)

                packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_SPEED, int(adjusted_velocity))
                time.sleep(TIME_INCREMENT)

    except KeyboardInterrupt:
        pass

    finally:
        for dxl_id in DXL_ID_LIST:
            packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_PRO_TORQUE_ENABLE, 0)
        portHandler.closePort()

if __name__ == "__main__":
    main()