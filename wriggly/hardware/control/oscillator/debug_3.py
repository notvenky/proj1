import numpy as np
import time
import re
from dynamixel_sdk import *

class DynamixelControl:
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
    ADDR_MOVING = 122
    ADDR_OPERATING_MODE = 11

    # Operating Mode Addresses
    POSITION_CONTROL = 3
    VELOCITY_CONTROL = 1

    # Define angle ranges for each Dynamixel
    ANGLE_RANGES = {
        11: (1024, 3072),
        12: (0, 4095),
        20: (1024, 3072),
        21: (1024, 3072),
        22: (0, 4095)
    }


    MEAN_POSITION = 2048
    dxl_goal_position = [0,4095]
    JOYSTICK_THRESHOLD = 0.8
    DXL_MOVING_STATUS_THRESHOLD = 20

    # Protocol version
    PROTOCOL_VERSION = 2.0                        # 2.0 for XM430-W210

    # Default setting
    DXL_ID_LIST = [11, 12, 20, 21, 22]                 # Dynamixel ID list
    AMPLITUDE_CONVERSION_FACTOR = 2048 / 3.14
    
    def __init__(self, device_name='/dev/ttyUSB0', baudrate=1000000, protocol_version=2.0):
        self.portHandler = PortHandler(device_name)
        self.packetHandler = PacketHandler(protocol_version)
        self.open_port()
        self.set_baudrate(baudrate)
        self.initialize_motors()
        
    def open_port(self):
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            quit()

    def set_baudrate(self, baudrate):
        if self.portHandler.setBaudRate(baudrate):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            quit()

    def initialize_motors(self):
        #... Method code here
    
    # ... More methods for motor control, etc.
    
class MotionController:
    
    def __init__(self, dynamixel_controller, frequencies, amplitudes, phases):
        self.dxl = dynamixel_controller
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.phases = phases

    def start_motion(self):
        start_time = time.time()
        try:
            while True:
                current_time = time.time() - start_time
                #... rest of the motion control loop code here
        except KeyboardInterrupt:
            pass
        finally:
            self.dxl.disable_torque_all_motors()
            self.dxl.close_port()

# Parsing and preparing data could be in separate functions
def parse_paste_string(paste_string):
    amplitude_conversion_factor = 2048 / 3.14
    tensor_values = re.findall('tensor\((.*?)\)', paste_string)
    frequency = eval(tensor_values[0])
    amplitude = [round(a * amplitude_conversion_factor) for a in eval(tensor_values[1])]
    phase = eval(tensor_values[2])

    frequency[0], frequency[1] = frequency[1], frequency[0]
    amplitude[0], amplitude[1] = amplitude[1], amplitude[0]
    phase[0], phase[1] = phase[1], phase[0]
    return frequency, amplitude, phase

def create_id_value_dict(ids, values):
    """
    Create a dictionary with ids as keys and values as values.

    Parameters:
    - ids (list): List of ids.
    - values (list): Corresponding values.

    Returns:
    dict: Dictionary with ids as keys and values as values.
    """
    return dict(zip(ids, values))

paste_string = 'MReward: 4403.559757610102, Frequency: tensor([0.8048, 0.4901, 0.4643, 0.2966, 0.3876]), Amplitude: tensor([0.1598, 2.9545, 1.4238, 0.2111, 1.4200]), Phase: tensor([5.5397, 5.5965, 3.2232, 3.2181, 2.4138])'

frequencies, amplitudes, phases = parse_paste_string(paste_string)

dxl_controller = DynamixelControl()
motion_controller = MotionController(
    dynamixel_controller=dxl_controller,
    frequencies=create_id_value_dict(DXL_ID_LIST, frequencies),
    amplitudes=create_id_value_dict(DXL_ID_LIST, amplitudes),
    phases=create_id_value_dict(DXL_ID_LIST, phases)
)

motion_controller.start_motion()
