from dynamixel_sdk import *
from config import *
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time

# Add your PID control parameters here
Kp, Ki, Kd = 0.1, 0.01, 0.01  # Example values, need tuning for your specific application

# Other necessary configurations and addresses for current-based control
ADDR_PRO_GOAL_CURRENT = # Address for goal current
ADDR_PRO_PRESENT_CURRENT = # Address for present current
MAX_TORQUE = # Maximum torque value for your servo

# ... [Your existing initialization code] ...

def calculate_torque_error(desired_position, actual_position):
    """
    Calculate the error between the desired and actual position and convert it to torque.
    """
    position_error = desired_position - actual_position
    torque = Kp * position_error  # Add Ki and Kd terms as needed
    return np.clip(torque, -MAX_TORQUE, MAX_TORQUE)

def write_torque(torques):
    """
    Write torques to the motors using group sync write.
    """
    groupSyncWrite.clearParam()
    for dxl_id, torque in zip(DXL_ID_LIST, torques):
        # Convert torque to appropriate data format
        param_goal_torque = [DXL_LOBYTE(DXL_LOWORD(torque)), DXL_HIBYTE(DXL_LOWORD(torque)), 
                             DXL_LOBYTE(DXL_HIWORD(torque)), DXL_HIBYTE(DXL_HIWORD(torque))]
        dxl_addparam_result = groupSyncWrite.addParam(dxl_id, param_goal_torque)
        if not dxl_addparam_result:
            print("[ID:%03d] groupSyncWrite addparam failed" % dxl_id)
            quit()

    # Send packet and clear
    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    groupSyncWrite.clearParam()

# ... [Rest of your code with necessary modifications] ...

# In your control loop
for joint_angles in joint_readings:
    desired_positions = convert_to_dxl_position(joint_angles)
    actual_positions = read_actual_joint_positions()
    torques = [calculate_torque_error(desired, actual) for desired, actual in zip(desired_positions, actual_positions)]
    write_torque(torques)
    time.sleep(0.1)

# ... [Rest of your visualization code] ...



# Example addresses for XM430-W350
ADDR_PRO_GOAL_CURRENT = 102  # This is an example, please refer to the XM430 manual for the correct address
LEN_PRO_GOAL_CURRENT = 2     # Length of data for goal current is 2 bytes

# Initialize GroupSyncWrite for current control
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_PRO_GOAL_CURRENT, LEN_PRO_GOAL_CURRENT)

