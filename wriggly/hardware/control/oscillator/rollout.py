from dynamixel_sdk import *
from config import *
import pickle

groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_PRO_GOAL_POSITION, 4)

def convert_to_dxl_position(joint_angles):
    """
    Convert joint angles to Dynamixel positions.
    Assuming joint_angles are in radians and need to be mapped to Dynamixel positions.
    """
    dxl_positions = []
    for angle in joint_angles:
        position = int((angle + np.pi) / (2 * np.pi) * 4095)
        position = np.clip(position, 0, 4095)
        dxl_positions.append(position)
    return dxl_positions

def write_joint_positions(joint_positions):
    """
    Write joint positions to the motors using group sync write.
    """
    groupSyncWrite.clearParam()

    for dxl_id, position in zip(DXL_ID_LIST, joint_positions):
        param_goal_position = [DXL_LOBYTE(DXL_LOWORD(position)), DXL_HIBYTE(DXL_LOWORD(position)), 
                               DXL_LOBYTE(DXL_HIWORD(position)), DXL_HIBYTE(DXL_HIWORD(position))]

        dxl_addparam_result = groupSyncWrite.addParam(dxl_id, param_goal_position)
        if not dxl_addparam_result:
            print("[ID:%03d] groupSyncWrite addparam failed" % dxl_id)
            quit()

    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    groupSyncWrite.clearParam()

# ignore the first row
with open('observations_0.pkl', 'rb') as f:
    joint_readings = pickle.load(f)[1:]

for joint_angles in joint_readings:
    joint_positions = convert_to_dxl_position(joint_angles)
    write_joint_positions(joint_positions)
    time.sleep(1/CONTROL_HZ)