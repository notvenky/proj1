from dynamixel_sdk import *
import numpy as np
import pickle
import time
from config import *

# Load goal positions from the .pkl file
with open('observations_0.pkl', 'rb') as file:
    goal_positions_list = pickle.load(file)[1:]

# Create GroupSyncWrite instance for goal position
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_PRO_GOAL_POSITION, 4)

def send_goal_positions(positions):
    """Send a tuple of positions to the servos."""
    groupSyncWrite.clearParam()
    for dxl_id, position in zip(DXL_ID_LIST, positions):
        param_goal_position = [DXL_LOBYTE(DXL_LOWORD(position)), DXL_HIBYTE(DXL_LOWORD(position)), 
                               DXL_LOBYTE(DXL_HIWORD(position)), DXL_HIBYTE(DXL_HIWORD(position))]
        dxl_addparam_result = groupSyncWrite.addParam(dxl_id, param_goal_position)
        if not dxl_addparam_result:
            print("[ID:%03d] groupSyncWrite addparam failed" % dxl_id)
            return False

    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(packetHandler.getTxRxResult(dxl_comm_result))
        return False
    groupSyncWrite.clearParam()
    return True

def wait_for_movement_completion():
    """Wait until all servos have reached their goal positions."""
    while True:
        all_at_position = True
        for dxl_id in DXL_ID_LIST:
            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_PRESENT_POSITION)
            if dxl_comm_result != COMM_SUCCESS:
                print(packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print(packetHandler.getRxPacketError(dxl_error))
            else:
                dxl_goal_position, _, _ = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRO_GOAL_POSITION)
                if abs(dxl_goal_position - dxl_present_position) > DXL_MOVING_STATUS_THRESHOLD:
                    all_at_position = False
                    break
        if all_at_position:
            break
        time.sleep(0.1)

# Main loop to send goal positions
for positions in goal_positions_list:
    if send_goal_positions(positions):
        wait_for_movement_completion()
    time.sleep(CONTROL_TIME)

# Close port
portHandler.closePort()
